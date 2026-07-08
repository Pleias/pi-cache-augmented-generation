"""llama-server client + prompt/cache mechanics for Cache-Augmented Generation.

The whole point of CAG is to prefill the ingested document ONCE, save its KV cache
as a slot file, and reuse it on every question (and across restarts) instead of
re-prefilling a 12k-token document each time. This module owns everything that talks
to llama-server plus the pure prompt/history helpers.

Prompt format is the raw ChatML validated by bench/cag_bench.py (CHATML_SYS_CTX): the
document lives in the system message so it is a stable, cacheable prefix, and only the
user turn varies. Requests hit /completion (not /v1/chat/completions) with a manually
built prompt so the token sequence is deterministic — a reliable prefix match is what
makes slot save/restore work.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx

# --- llama-server / cache constants ---------------------------------------------------
N_CTX = 16384          # llama-server launched with -c 16384
GEN_RESERVE = 1024     # tokens kept free for the answer when sizing history budget
FICHE_CAP = 12000      # max tokens for the cached system prefix (device RAM + headroom)
SLOT_ID = 0            # single-slot server; we always use slot 0
SLOT_FILE = "slot0.bin"  # filename passed to /slots/0?action=save|restore
STOP = ["<|im_end|>", "<|im_start|>"]  # <|im_end|> is not EOS -> must stop explicitly
TEMPERATURE = 0.3

# Language-neutral grounding instruction. The === DOCUMENT === framing mirrors the
# benchmark's CHATML_SYS_CTX (which validated deterministic prefix caching); the wording
# is generalised because the app ingests arbitrary user text in any language/domain.
SYSTEM_TMPL = (
    "You are a helpful assistant. Answer the user's question using ONLY the document "
    "below. If the document does not contain the answer, say so plainly. Reply in the "
    "same language as the question.\n\n"
    "=== DOCUMENT ===\n{fiche}\n=== END OF DOCUMENT ==="
)


class LlamaError(RuntimeError):
    """Raised when llama-server is unreachable or returns an error."""


# --- pure prompt helpers (no I/O) -----------------------------------------------------

def build_system_prefix(fiche: str) -> str:
    """The stable, cacheable ChatML system turn holding the ingested document.

    This exact string is what gets prefilled and saved as the slot; every question
    reuses it as a prefix, so it must be byte-for-byte identical across calls.
    """
    return f"<|im_start|>system\n{SYSTEM_TMPL.format(fiche=fiche)}<|im_end|>\n"


def render_pair(user: str, assistant: str) -> str:
    """Render one (user, assistant) exchange as its ChatML turn block.

    Each block is bounded by special tokens (`<|im_start|>` … `<|im_end|>\n`), so when
    blocks are concatenated the seam between them is a hard special-token boundary. That
    makes a pair's token count stable and additive: summing per-pair counts equals
    tokenizing the whole concatenation. `trim_history` relies on this to budget from
    cached counts instead of re-tokenizing the entire history on every question.
    """
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    )


def build_history_block(history: list[tuple[str, str]]) -> str:
    """Render prior (user, assistant) pairs as ChatML turns."""
    return "".join(render_pair(user, assistant) for user, assistant in history)


def build_prompt(fiche: str, history: list[tuple[str, str]], question: str) -> str:
    """Full ChatML prompt: cached system prefix + history turns + open assistant turn."""
    return (
        build_system_prefix(fiche)
        + build_history_block(history)
        + f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    )


# --- persisted ingest metadata --------------------------------------------------------

@dataclass
class Meta:
    fiche_tokens: int   # token count of the cached system prefix (the budget anchor)
    model: str          # model id at ingest time; restore aborts on mismatch
    created: str        # ISO-ish timestamp

    def to_json(self) -> str:
        return json.dumps(
            {"fiche_tokens": self.fiche_tokens, "model": self.model,
             "created": self.created},
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(text: str) -> "Meta":
        d = json.loads(text)
        return Meta(fiche_tokens=int(d["fiche_tokens"]), model=str(d["model"]),
                    created=str(d.get("created", "")))


class LlamaClient:
    """Thin async wrapper over the llama-server HTTP endpoints we use."""

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self.base = base_url.rstrip("/")
        self.http = client

    async def model_id(self) -> str:
        """Best-effort model identifier from /props (basename of the GGUF path)."""
        try:
            r = await self.http.get(f"{self.base}/props", timeout=10)
            r.raise_for_status()
            path = r.json().get("model_path") or ""
            return Path(path).name or "unknown"
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            return "unknown"

    async def tokenize(self, text: str) -> int:
        """Number of tokens llama-server assigns to `text` (POST /tokenize)."""
        try:
            r = await self.http.post(f"{self.base}/tokenize",
                                     json={"content": text}, timeout=60)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LlamaError(f"tokenize failed: {e}") from e
        return len(r.json().get("tokens", []))

    async def prefill_and_save(self, system_prefix: str) -> None:
        """Populate slot 0's KV cache with the document prefix, then persist it to disk.

        A single-token generation is the cheapest way to force a prefill; the answer is
        thrown away. The subsequent slot save writes SLOT_FILE under the server's
        --slot-save-path so it survives restarts.
        """
        try:
            r = await self.http.post(
                f"{self.base}/completion",
                json={"prompt": system_prefix, "cache_prompt": True,
                      "id_slot": SLOT_ID, "n_predict": 1, "stream": False,
                      "temperature": TEMPERATURE, "stop": STOP},
                timeout=None,
            )
            r.raise_for_status()
            # No timeout: the slot file for a 12k-token context is ~300 MB and a cold,
            # slow SD card can take well over 2 min to write it. Aborting here would kill
            # the save the whole persistence feature depends on. Matches the prefill call.
            s = await self.http.post(
                f"{self.base}/slots/{SLOT_ID}?action=save",
                json={"filename": SLOT_FILE}, timeout=None,
            )
            s.raise_for_status()
        except httpx.HTTPError as e:
            raise LlamaError(f"prefill/save failed: {e}") from e

    async def restore(self) -> bool:
        """Load SLOT_FILE back into slot 0. Returns False on any failure."""
        try:
            # No timeout: reading back the ~300 MB slot from a slow SD card can exceed
            # 2 min; do not let the restore that skips the 5-min prefill time out.
            r = await self.http.post(
                f"{self.base}/slots/{SLOT_ID}?action=restore",
                json={"filename": SLOT_FILE}, timeout=None,
            )
            r.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    async def stream_completion(self, prompt: str) -> AsyncIterator[tuple[str, object]]:
        """Stream a /completion answer, yielding ("token", str) then ("done", timings).

        cache_prompt=True lets llama-server reuse the saved document prefix, so only the
        history tail + question re-prefill. The final SSE line carries llama-server's own
        timings, which we pass straight through to the client.
        """
        body = {
            "prompt": prompt, "cache_prompt": True, "id_slot": SLOT_ID,
            "stream": True, "n_predict": 512, "stop": STOP,
            "temperature": TEMPERATURE,
        }
        try:
            async with self.http.stream("POST", f"{self.base}/completion",
                                        json=body, timeout=None) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        j = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    content = j.get("content")
                    if content:
                        yield ("token", content)
                    if j.get("stop"):
                        yield ("done", j.get("timings") or {})
                        return
        except httpx.HTTPError as e:
            raise LlamaError(f"completion stream failed: {e}") from e
        # stream ended without an explicit stop marker
        yield ("done", {})


def history_budget(fiche_tokens: int) -> int:
    """Tokens available for chat history: full context minus the cached prefix and the
    generation reserve. Can go negative for a pathologically large document."""
    return N_CTX - fiche_tokens - GEN_RESERVE


def trim_history(
    fiche_tokens: int,
    history: list[tuple[str, str]],
    pair_tokens: list[int],
) -> tuple[list[tuple[str, str]], list[int], int]:
    """Drop oldest (user, assistant) pairs until the history fits the budget.

    Pure and I/O-free: `pair_tokens[i]` is the cached ChatML token count of `history[i]`,
    kept in lockstep by the caller (which tokenizes only each new pair, once). Because
    per-pair counts are additive across ChatML's special-token seams (see `render_pair`),
    the budget check is a running sum — no per-question tokenize round-trip to the Pi.
    The document prefix is already cached and accounted for via `fiche_tokens`. Returns
    the (possibly shortened) history, its matching pair-count list, and the total tokens.
    """
    budget = history_budget(fiche_tokens)
    hist = list(history)
    counts = list(pair_tokens)
    total = sum(counts)
    while hist and total > budget:
        total -= counts.pop(0)  # drop the oldest turn pair
        hist.pop(0)
    return hist, counts, total


async def restore_or_reprefill(
    client: LlamaClient,
    data_dir: Path,
    model_id: str,
) -> tuple[str, int, bool] | None:
    """Startup recovery of a prior ingest from `data_dir`.

    Returns (fiche_text, fiche_tokens, restored) or None if nothing was ingested.
    `restored` is True when the saved KV slot loaded cleanly; False when we had to fall
    back to re-prefilling from fiche.txt (still usable, just slower to warm up).
    """
    meta_path = data_dir / "meta.json"
    fiche_path = data_dir / "fiche.txt"
    if not meta_path.exists() or not fiche_path.exists():
        return None

    fiche = fiche_path.read_text(encoding="utf-8")
    try:
        meta = Meta.from_json(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError, ValueError):
        meta = None

    # Fast path: restore the saved KV slot instead of a ~5-min re-prefill.
    # Model guard: only a POSITIVE mismatch (both ids known and different) blocks restore.
    # A runtime id of "unknown" — e.g. /props polled before the model finished loading —
    # must NOT count as a mismatch, or the headline "restore in seconds" feature silently
    # never fires on restart. In that case log and attempt the restore anyway, falling
    # back to a re-prefill only if the restore itself fails.
    model_known = bool(model_id) and model_id != "unknown"
    if meta is not None and (not model_known or meta.model == model_id):
        if not model_known:
            print("[startup] runtime model id unknown (server still loading?); "
                  "attempting slot restore anyway", flush=True)
        if await client.restore():
            return fiche, meta.fiche_tokens, True

    # Fallback: corrupt meta, model mismatch, or restore failed -> rebuild the cache.
    system_prefix = build_system_prefix(fiche)
    fiche_tokens = await client.tokenize(system_prefix)
    await client.prefill_and_save(system_prefix)
    # refresh meta so the next restart can take the fast path
    save_meta(data_dir, fiche_tokens, model_id)
    return fiche, fiche_tokens, False


def save_meta(data_dir: Path, fiche_tokens: int, model_id: str) -> Meta:
    """Persist ingest metadata next to fiche.txt."""
    meta = Meta(fiche_tokens=fiche_tokens, model=model_id,
                created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (data_dir / "meta.json").write_text(meta.to_json(), encoding="utf-8")
    return meta


# --- chat-history persistence -----------------------------------------------------------
# The chat history lives in the KV cache as part of the reused prefix, but the slot file
# only ever holds the fiche prefix (saved once at ingest — re-saving a ~300 MB slot after
# every answer would grind the Pi's SD card for marginal gain). So the history is persisted
# as a small JSON file instead: on restart the fiche KV is restored from the slot and the
# first /ask re-prefills just the history tail on top of it (seconds, not the ~5-min fiche
# prefill), while the UI can replay the conversation immediately.

def save_history(data_dir: Path,
                 history: list[tuple[str, str]], pair_tokens: list[int]) -> None:
    """Persist the chat turns + their cached token counts next to the slot file.

    Written atomically (tmp + rename) so a crash mid-write can't leave a torn file that
    would make the next startup drop the whole conversation.
    """
    turns = [{"user": u, "assistant": a, "tokens": t}
             for (u, a), t in zip(history, pair_tokens)]
    path = data_dir / "history.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"turns": turns}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_history(data_dir: Path) -> tuple[list[tuple[str, str]], list[int]]:
    """Load persisted chat turns; a missing or corrupt file is an empty conversation."""
    path = data_dir / "history.json"
    if not path.exists():
        return [], []
    try:
        turns = json.loads(path.read_text(encoding="utf-8"))["turns"]
        history = [(str(t["user"]), str(t["assistant"])) for t in turns]
        pair_tokens = [int(t["tokens"]) for t in turns]
        # A tampered/garbage count would poison the budget math (negative counts make
        # trim_history think there is infinite room). Treat it as a corrupt file.
        if any(t <= 0 or t > N_CTX for t in pair_tokens):
            return [], []
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return [], []
    return history, pair_tokens


def clear_history(data_dir: Path) -> None:
    """Remove the persisted conversation (reset / re-ingest)."""
    (data_dir / "history.json").unlink(missing_ok=True)
