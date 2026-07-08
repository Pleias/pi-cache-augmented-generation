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

import io
import json
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import AsyncIterator
from uuid import uuid4

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

    async def prefill_and_save(self, system_prefix: str,
                               filename: str = SLOT_FILE) -> None:
        """Populate slot 0's KV cache with the document prefix, then persist it to disk.

        A single-token generation is the cheapest way to force a prefill; the answer is
        thrown away. The subsequent slot save writes `filename` under the server's
        --slot-save-path so it survives restarts. Multi-doc callers pass a per-document
        `<id>.bin`; the default keeps the single-slot back-compat name.
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
                json={"filename": filename}, timeout=None,
            )
            s.raise_for_status()
        except httpx.HTTPError as e:
            raise LlamaError(f"prefill/save failed: {e}") from e

    async def restore(self, filename: str = SLOT_FILE) -> bool:
        """Load `filename` back into slot 0. Returns False on any failure.

        The document library restores a per-document `<id>.bin`; the default keeps the
        single-slot back-compat name so legacy callers are unchanged.
        """
        try:
            # No timeout: reading back the ~300 MB slot from a slow SD card can exceed
            # 2 min; do not let the restore that skips the 5-min prefill time out.
            r = await self.http.post(
                f"{self.base}/slots/{SLOT_ID}?action=restore",
                json={"filename": filename}, timeout=None,
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
        # Finite READ timeout so a wedged server cannot pin _slot_lock forever. /ask holds
        # the lock for the whole stream, so an infinite hang here would freeze EVERY mutating
        # endpoint (switch/delete/ingest/reset/wipe/import). At ~12 tok/s a token arrives
        # every ~100 ms, so 300 s of silence between chunks means the server is hung: httpx
        # raises ReadTimeout -> LlamaError -> the /ask generator emits an SSE error event and
        # releases the lock. (Slot save/restore stay timeout=None on purpose — those are
        # legitimately multi-minute SINGLE operations, not a token stream; see prefill_and_save
        # / restore.)
        stream_timeout = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=None)
        try:
            async with self.http.stream("POST", f"{self.base}/completion",
                                        json=body, timeout=stream_timeout) as r:
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
    doc_dir: Path,
    model_id: str,
    slot_filename: str = SLOT_FILE,
) -> tuple[str, int, bool] | None:
    """Recover a document's KV cache from `doc_dir` (its fiche.txt + meta.json).

    Restores the slot file `slot_filename` (a per-document `<id>.bin` in the library, or
    the legacy `slot0.bin` by default). Returns (fiche_text, fiche_tokens, restored) or
    None if the document dir has no ingest. `restored` is True when the saved KV slot
    loaded cleanly; False when we had to fall back to re-prefilling from fiche.txt (still
    usable, just slower to warm up).
    """
    meta_path = doc_dir / "meta.json"
    fiche_path = doc_dir / "fiche.txt"
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
        if await client.restore(slot_filename):
            return fiche, meta.fiche_tokens, True

    # Fallback: corrupt meta, model mismatch, or restore failed -> rebuild the cache.
    system_prefix = build_system_prefix(fiche)
    fiche_tokens = await client.tokenize(system_prefix)
    await client.prefill_and_save(system_prefix, slot_filename)
    # refresh meta so the next restart can take the fast path
    save_meta(doc_dir, fiche_tokens, model_id)
    return fiche, fiche_tokens, False


def now_iso() -> str:
    """UTC timestamp in the same ISO-ish shape used across persisted metadata."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save_meta(data_dir: Path, fiche_tokens: int, model_id: str) -> Meta:
    """Persist ingest metadata next to fiche.txt."""
    meta = Meta(fiche_tokens=fiche_tokens, model=model_id, created=now_iso())
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
    data_dir.mkdir(parents=True, exist_ok=True)
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


# --- multi-document library -----------------------------------------------------------
# The single-doc layout (fiche.txt / meta.json / history.json at the DATA_DIR root, plus
# cache/slot0.bin) generalises to a library: each document gets its own directory
# docs/<id>/{fiche.txt,meta.json,history.json} and its own saved KV slot cache/<id>.bin.
# library.json is the atomic index; exactly one document is "active" — its slot is the one
# currently live in slot 0. Only ONE slot is ever in RAM, so the decode wall and ~2 GB
# ceiling are untouched; the cost is disk (~300 MB/slot), bounded by MAX_DOCS.

LIBRARY_FILE = "library.json"
MAX_DOCS = 20            # library cap (US-18 SD-size guard): applies to BOTH ingest and
                        # pack import — one number, matching CONTAINER.md's "capped at 20".

# Content-pack size ceilings. 20 docs x ~300 MB slot ≈ 6 GB, so 8 GB gives headroom while
# still catching a decompression bomb or a runaway pack before it fills the SD. The upload
# body cap is enforced in main.py (env-overridable PACK_MAX_BYTES); the extraction cap and
# free-space margin below are enforced here during import.
DEFAULT_MAX_PACK_BYTES = 8 * 1024 * 1024 * 1024   # 8 GB
FREE_SPACE_MARGIN = 500 * 1024 * 1024             # keep 500 MB free after an import


class PackError(RuntimeError):
    """Raised when an imported content pack is malformed or unsafe (→ HTTP 400)."""


class PackTooLargeError(PackError):
    """Raised when a pack exceeds a size/disk-space ceiling (→ HTTP 507)."""


@dataclass
class DocEntry:
    """One row of the library index (metadata only; the heavy slot lives in cache/)."""
    id: str
    title: str
    fiche_tokens: int
    created: str
    last_used: str

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title,
                "fiche_tokens": self.fiche_tokens,
                "created": self.created, "last_used": self.last_used}

    @staticmethod
    def from_dict(d: dict) -> "DocEntry":
        return DocEntry(
            id=str(d["id"]), title=str(d.get("title", "")),
            fiche_tokens=int(d.get("fiche_tokens", 0)),
            created=str(d.get("created", "")),
            last_used=str(d.get("last_used", d.get("created", ""))),
        )


@dataclass
class Library:
    docs: list[DocEntry]
    active: str | None

    def get(self, doc_id: str | None) -> DocEntry | None:
        if doc_id is None:
            return None
        for d in self.docs:
            if d.id == doc_id:
                return d
        return None

    def most_recent(self) -> DocEntry | None:
        """Most recently used document; later-inserted wins ties (>=), so the fallback
        after deleting the active doc lands on the freshest remaining entry."""
        if not self.docs:
            return None
        best = self.docs[0]
        for d in self.docs[1:]:
            if d.last_used >= best.last_used:
                best = d
        return best

    def to_json(self) -> str:
        return json.dumps(
            {"docs": [d.to_dict() for d in self.docs], "active": self.active},
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(text: str) -> "Library":
        d = json.loads(text)
        docs = [DocEntry.from_dict(x) for x in d.get("docs", [])]
        active = d.get("active")
        return Library(docs=docs, active=(str(active) if active else None))


# --- library paths --------------------------------------------------------------------

def library_path(data_dir: Path) -> Path:
    return data_dir / LIBRARY_FILE


def docs_root(data_dir: Path) -> Path:
    return data_dir / "docs"


def doc_dir(data_dir: Path, doc_id: str) -> Path:
    return data_dir / "docs" / doc_id


def cache_root(data_dir: Path) -> Path:
    return data_dir / "cache"


def slot_filename(doc_id: str) -> str:
    """Filename passed to llama-server's /slots API (relative to --slot-save-path)."""
    return f"{doc_id}.bin"


def slot_path(data_dir: Path, doc_id: str) -> Path:
    return cache_root(data_dir) / slot_filename(doc_id)


def new_doc_id() -> str:
    """Short random id for a library document (no timestamps — deterministic in tests)."""
    return uuid4().hex[:8]


def derive_title(text: str, provided: str | None = None) -> str:
    """A short human label for a document: the user's title if given, else the first line
    (capped ~40 chars)."""
    if provided and provided.strip():
        return provided.strip()[:80]
    stripped = text.strip()
    if not stripped:
        return "Untitled"
    first = stripped.splitlines()[0].strip()
    return first if len(first) <= 40 else first[:40].rstrip() + "…"


# --- library index persistence (atomic, mirrors save_history) -------------------------

def load_library(data_dir: Path) -> Library:
    """Load the library index. A missing file is an empty library. An UNPARSEABLE file is
    NOT silently discarded (the next ingest would overwrite it and permanently unlink every
    on-disk document): it is renamed aside to library.json.corrupt-<n> and loudly logged, so
    the user's data is recoverable, before starting from an empty index."""
    p = library_path(data_dir)
    if not p.exists():
        return Library(docs=[], active=None)
    try:
        return Library.from_json(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        n = 0
        while (data_dir / f"library.json.corrupt-{n}").exists():
            n += 1
        aside = data_dir / f"library.json.corrupt-{n}"
        try:
            p.rename(aside)
            print(f"[library] library.json is unparseable — preserved as {aside.name} and "
                  "starting from an empty index (documents on disk are NOT deleted)",
                  flush=True)
        except OSError:
            print("[library] library.json is unparseable and could not be renamed aside; "
                  "starting from an empty index", flush=True)
        return Library(docs=[], active=None)


def save_library(data_dir: Path, lib: Library) -> None:
    """Persist the index atomically (tmp + rename) so a crash can't tear the file and a
    concurrent reader always sees a whole, consistent index."""
    data_dir.mkdir(parents=True, exist_ok=True)
    p = library_path(data_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(lib.to_json(), encoding="utf-8")
    tmp.replace(p)


def remove_doc(data_dir: Path, doc_id: str) -> None:
    """Delete a document's directory and its saved slot file. Idempotent."""
    shutil.rmtree(doc_dir(data_dir, doc_id), ignore_errors=True)
    slot_path(data_dir, doc_id).unlink(missing_ok=True)


# --- legacy migration -----------------------------------------------------------------

LEGACY_ID = "legacy0"


def _write_legacy_index(data_dir: Path, dd: Path) -> None:
    """Build library.json (one entry, active) from an already-populated docs/legacy0."""
    fiche_text = (dd / "fiche.txt").read_text(encoding="utf-8")
    try:
        meta = Meta.from_json((dd / "meta.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        meta = None
    created = meta.created if (meta and meta.created) else now_iso()
    entry = DocEntry(
        id=LEGACY_ID, title=derive_title(fiche_text),
        fiche_tokens=(meta.fiche_tokens if meta else 0),
        created=created, last_used=created,
    )
    save_library(data_dir, Library(docs=[entry], active=LEGACY_ID))


def migrate_legacy(data_dir: Path) -> None:
    """Fold a pre-library single-doc layout into docs/legacy0 + library.json.

    Old layout: DATA_DIR/{fiche.txt,meta.json,history.json} + cache/slot0.bin. Each file is
    *renamed* (slot0.bin -> legacy0.bin), never re-prefilled, so the one-time ~5-min prefill
    is preserved.

    Crash-safe & idempotent: the index (library.json) is written LAST, so a power loss
    mid-migration (routine on a Pi with no UPS) leaves docs/legacy0 populated but unindexed.
    Every move below is guarded (skip if already done), so re-running completes any partial
    migration and finally writes the index — the document can never be silently orphaned.
    A no-op once library.json exists.
    """
    if library_path(data_dir).exists():
        return  # already on the library layout
    dd = doc_dir(data_dir, LEGACY_ID)
    root_fiche = data_dir / "fiche.txt"
    root_meta = data_dir / "meta.json"
    partial = dd.is_dir() and any(dd.iterdir())
    if not (root_fiche.exists() and root_meta.exists()) and not partial:
        return  # nothing ingested under the old layout, and no half-done migration

    dd.mkdir(parents=True, exist_ok=True)

    def _move_if(src: Path, dst: Path) -> None:
        # Idempotent rename: move only if the source is still present and the target is not
        # yet there (so a re-run after a crash finishes the move instead of clobbering it).
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)

    _move_if(root_fiche, dd / "fiche.txt")
    _move_if(root_meta, dd / "meta.json")
    _move_if(data_dir / "history.json", dd / "history.json")
    _move_if(cache_root(data_dir) / SLOT_FILE, slot_path(data_dir, LEGACY_ID))

    # Index it only when the document content is actually present.
    if (dd / "fiche.txt").exists() and (dd / "meta.json").exists():
        _write_legacy_index(data_dir, dd)


# --- content packs: export / import ---------------------------------------------------

def pack_manifest(data_dir: Path) -> tuple[list[tuple[str, bytes]], list[tuple[str, Path]]]:
    """Snapshot the SMALL files of the library (library.json + every docs/<id>/* — all KBs)
    into memory, and list the heavy slot files as (arcname, path) references WITHOUT reading
    them. Meant to run under _slot_lock: it is fast, so the lock is held only for the tiny
    snapshot, never for the multi-minute ~300 MB-per-slot copy (which build_pack_from_manifest
    does off-lock)."""
    small: list[tuple[str, bytes]] = []
    lib_path = library_path(data_dir)
    if lib_path.exists():
        small.append((LIBRARY_FILE, lib_path.read_bytes()))
    docs = docs_root(data_dir)
    if docs.is_dir():
        for doc_path in sorted(docs.iterdir()):
            if not doc_path.is_dir():
                continue
            for f in sorted(doc_path.iterdir()):
                if f.is_file():
                    small.append((f"docs/{doc_path.name}/{f.name}", f.read_bytes()))
    slots: list[tuple[str, Path]] = []
    cache = cache_root(data_dir)
    if cache.is_dir():
        for binf in sorted(cache.glob("*.bin")):
            slots.append((f"cache/{binf.name}", binf))
    return small, slots


def build_pack_from_manifest(
    small: list[tuple[str, bytes]],
    slots: list[tuple[str, Path]],
) -> tempfile.SpooledTemporaryFile:
    """Assemble the tar from a manifest, OFF the slot lock. Small files come from the
    in-memory snapshot; each slot .bin is streamed from its original path. A slot that
    vanished between snapshot and read (a concurrent delete/switch after the lock was
    released) is skipped with a log line — the export stays a valid tar; the importing
    device simply re-prefills that document on first switch (its saved slot is absent).
    Spooled to disk past a few MB (TMPDIR is /data/tmp on the Pi, not tmpfs — see entrypoint)."""
    spooled = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, suffix=".tar")
    with tarfile.open(fileobj=spooled, mode="w") as tar:
        now = int(time.time())
        for arcname, data in small:
            info = tarfile.TarInfo(name=arcname)
            info.size = len(data)
            info.mtime = now
            tar.addfile(info, io.BytesIO(data))
        for arcname, path in slots:
            try:
                if not path.exists():
                    print(f"[pack] slot {path.name} vanished during export; skipped",
                          flush=True)
                    continue
                tar.add(path, arcname=arcname)
            except FileNotFoundError:
                print(f"[pack] slot {path.name} vanished during export; skipped", flush=True)
    spooled.seek(0)
    return spooled


def build_pack(data_dir: Path) -> tempfile.SpooledTemporaryFile:
    """Convenience one-shot pack build (snapshot + assemble). The /api/pack/export endpoint
    calls the two phases separately so only the snapshot runs under _slot_lock."""
    small, slots = pack_manifest(data_dir)
    return build_pack_from_manifest(small, slots)


def _safe_member_name(name: str) -> bool:
    """Only library.json / docs/… / cache/… , never absolute or '..'-escaping paths."""
    if not name or name.startswith("/"):
        return False
    p = PurePosixPath(name)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        return False
    if name == LIBRARY_FILE:
        return True
    return len(p.parts) >= 1 and p.parts[0] in ("docs", "cache")


def _member_doc_id(name: str) -> str | None:
    """The document id a pack member belongs to (docs/<id>/… or cache/<id>.bin)."""
    p = PurePosixPath(name)
    if p.parts and p.parts[0] == "docs" and len(p.parts) >= 2:
        return p.parts[1]
    if p.parts and p.parts[0] == "cache" and name.endswith(".bin"):
        return PurePosixPath(name).stem
    return None


def import_pack(data_dir: Path, fileobj, *,
                max_total_bytes: int = DEFAULT_MAX_PACK_BYTES,
                free_margin_bytes: int = FREE_SPACE_MARGIN) -> str | None:
    """Extract a content pack into data_dir, merging its library index. Returns the id of
    the most-recent imported document (for activation) or None if nothing new was added.

    SECURITY: every member name is validated (no absolute paths, no '..' traversal, only
    library.json / docs/ / cache/) and non-file/dir members (symlinks, hardlinks, devices,
    fifos) are rejected BEFORE any extraction.

    SIZE (a distributed pack is a trust boundary — a poisoned or merely huge pack must not
    fill or OOM the device):
      - a member whose header size exceeds max_total_bytes is rejected (implausible);
      - a free-space pre-check requires (planned bytes + margin) ≤ free before writing;
      - extraction copies in chunks and aborts once the CUMULATIVE bytes written exceed
        max_total_bytes — this is what stops a `r:*` decompression bomb, since a few-KB xz
        member can expand to tens of GB and the header size lies.
    On any failure mid-extraction every freshly-created (non-colliding) doc is removed, so a
    partial import never leaves orphaned files behind.

    Merge: imported docs are added; an id already present is kept as-is (imported copy
    skipped). Cap: refuse if the merged library would exceed MAX_DOCS documents.
    """
    with tarfile.open(fileobj=fileobj, mode="r:*") as tar:
        members = tar.getmembers()
        for m in members:
            if not _safe_member_name(m.name):
                raise PackError(f"unsafe path in pack: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise PackError(f"unsupported entry in pack: {m.name!r}")
        try:
            lib_member = tar.getmember(LIBRARY_FILE)
        except KeyError:
            raise PackError("pack has no library.json")
        src = tar.extractfile(lib_member)
        if src is None:
            raise PackError("pack library.json is not a regular file")
        try:
            imported = Library.from_json(src.read().decode("utf-8"))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise PackError(f"pack library.json is invalid: {e}") from e

        existing = load_library(data_dir)
        existing_ids = {d.id for d in existing.docs}
        new_entries = [d for d in imported.docs if d.id not in existing_ids]
        if len(existing.docs) + len(new_entries) > MAX_DOCS:
            raise PackError(f"pack would exceed the {MAX_DOCS}-document limit")
        keep_ids = {d.id for d in new_entries}

        to_extract = [m for m in members
                      if m.name != LIBRARY_FILE and _member_doc_id(m.name) in keep_ids]

        # Free-space pre-check (guards an honest-but-large pack against a mid-extract
        # SD-full that would strand half a document). Header sizes can lie for a bomb; the
        # running-total cap below is the guard against that.
        planned = sum(m.size for m in to_extract if m.isfile())
        try:
            free = shutil.disk_usage(data_dir).free
        except OSError:
            free = None
        if free is not None and planned + free_margin_bytes > free:
            raise PackTooLargeError(
                f"not enough free space to import: need ~{planned + free_margin_bytes} "
                f"bytes, only {free} free")

        total = 0
        try:
            for m in to_extract:
                dest = data_dir / m.name
                if m.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                if m.size > max_total_bytes:
                    raise PackTooLargeError(
                        f"pack member {m.name!r} is implausibly large ({m.size} bytes)")
                dest.parent.mkdir(parents=True, exist_ok=True)
                fsrc = tar.extractfile(m)
                if fsrc is None:
                    continue
                with fsrc, open(dest, "wb") as out:
                    while True:
                        chunk = fsrc.read(256 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_total_bytes:
                            raise PackTooLargeError(
                                f"pack exceeds the {max_total_bytes}-byte extraction cap "
                                "(possible decompression bomb)")
                        out.write(chunk)
        except Exception:
            # Roll back this import: remove every freshly-created doc (all are new ids, so
            # no pre-existing document is ever touched). Leaves the volume as it was.
            for did in keep_ids:
                remove_doc(data_dir, did)
            raise

    if not new_entries:
        return None
    existing.docs.extend(new_entries)
    save_library(data_dir, existing)
    newest = max(new_entries, key=lambda d: d.last_used or d.created)
    return newest.id


def wipe_all(data_dir: Path) -> None:
    """Privacy purge (US-12): delete every document, slot, history and the library index,
    leaving a clean empty volume. The cache dir is recreated so llama-server's
    --slot-save-path still exists for the next ingest."""
    shutil.rmtree(docs_root(data_dir), ignore_errors=True)
    cache = cache_root(data_dir)
    if cache.is_dir():
        for f in cache.iterdir():
            if f.is_file() or f.is_symlink():
                f.unlink(missing_ok=True)
            else:
                shutil.rmtree(f, ignore_errors=True)
    library_path(data_dir).unlink(missing_ok=True)
    # Legacy single-doc leftovers, if any.
    for name in ("fiche.txt", "meta.json", "history.json", "library.json.tmp"):
        (data_dir / name).unlink(missing_ok=True)
    cache_root(data_dir).mkdir(parents=True, exist_ok=True)
