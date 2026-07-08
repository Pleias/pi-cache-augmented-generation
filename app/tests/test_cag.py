"""Unit tests for the CAG core — no llama-server, no network.

llama-server HTTP is replaced by a hand-rolled `FakeClient` (unittest.mock is unnecessary:
duck typing is clearer here and needs no dependency). Async coroutines are driven with
`asyncio.run`, so the suite needs only pytest itself — no pytest-asyncio, no new project deps.

Run from app/:  uv run --with pytest pytest tests/ -q
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Put app/ on the path so `import server...` resolves regardless of pytest's rootdir.
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from server import cag  # noqa: E402
from server import main as appmain  # noqa: E402


def run(coro):
    return asyncio.run(coro)


class FakeClient:
    """Stand-in for cag.LlamaClient. tokenize() counts ChatML turn-starts so per-pair /
    prefix token counts are deterministic and additive (100 tokens per `<|im_start|>`)."""

    def __init__(self, *, restore_ok: bool = True, model: str = "qwen.gguf") -> None:
        self._restore_ok = restore_ok
        self._model = model
        self.prefill_calls = 0
        self.restore_calls = 0

    async def tokenize(self, text: str) -> int:
        return text.count("<|im_start|>") * 100

    async def prefill_and_save(self, system_prefix: str) -> None:
        self.prefill_calls += 1

    async def restore(self) -> bool:
        self.restore_calls += 1
        return self._restore_ok

    async def model_id(self) -> str:
        return self._model


class FakeRequest:
    def __init__(self, payload) -> None:
        self._payload = payload

    async def json(self):
        return self._payload


# --- ChatML prompt construction -------------------------------------------------------

def test_build_system_prefix_is_chatml_with_document():
    sp = cag.build_system_prefix("DOC")
    assert sp.startswith("<|im_start|>system\n")
    assert sp.endswith("<|im_end|>\n")
    assert "=== DOCUMENT ===" in sp and "DOC" in sp
    # exactly one system turn -> one turn-start
    assert sp.count("<|im_start|>") == 1


def test_build_prompt_full_structure():
    p = cag.build_prompt("F", [("u1", "a1"), ("u2", "a2")], "q")
    assert p.startswith(cag.build_system_prefix("F"))
    # each prior pair rendered as user+assistant turns
    assert "<|im_start|>user\nu1<|im_end|>\n<|im_start|>assistant\na1<|im_end|>\n" in p
    assert "<|im_start|>user\nu2<|im_end|>\n<|im_start|>assistant\na2<|im_end|>\n" in p
    # ends with an OPEN assistant turn for the new question
    assert p.endswith("<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n")


def test_render_pair_counts_are_additive():
    # Shape check ONLY: verifies string concatenation is additive over the "<|im_start|>"
    # marker, NOT that the real tokenizer avoids merging tokens across the ChatML seam
    # (<|im_end|>\n<|im_start|>). That live property — which the cached per-pair counts
    # depend on — is exercised by smoke.sh's multi-turn exchange against the real Qwen
    # tokenizer. Kept as a fast, dependency-free guard on the rendering contract.
    block = cag.build_history_block([("u1", "a1"), ("u2", "a2")])
    per_pair = sum(cag.render_pair(u, a).count("<|im_start|>")
                   for u, a in [("u1", "a1"), ("u2", "a2")])
    assert block.count("<|im_start|>") == per_pair == 4


# --- history budget + trimming --------------------------------------------------------

def test_history_budget_formula():
    assert cag.history_budget(12000) == cag.N_CTX - 12000 - cag.GEN_RESERVE


def test_trim_history_drops_oldest_until_under_budget():
    # budget = N_CTX - fiche - GEN_RESERVE = 350; three 200-token pairs -> keep newest one
    fiche = cag.N_CTX - cag.GEN_RESERVE - 350
    hist = [("u1", "a1"), ("u2", "a2"), ("u3", "a3")]
    counts = [200, 200, 200]
    kept, kept_counts, total = cag.trim_history(fiche, hist, counts)
    assert kept == [("u3", "a3")]
    assert kept_counts == [200]
    assert total == 200


def test_trim_history_keeps_all_when_within_budget():
    hist = [("u1", "a1"), ("u2", "a2")]
    counts = [100, 100]
    kept, kept_counts, total = cag.trim_history(100, hist, counts)  # huge budget
    assert kept == hist and kept_counts == counts and total == 200


def test_trim_history_empty():
    assert cag.trim_history(100, [], []) == ([], [], 0)


def test_trim_history_negative_budget_drops_everything():
    # a document larger than the whole context leaves no room for history at all
    fiche = cag.N_CTX  # budget = -GEN_RESERVE < 0
    kept, kept_counts, total = cag.trim_history(fiche, [("u", "a")], [50])
    assert kept == [] and kept_counts == [] and total == 0


# --- startup restore / re-prefill -----------------------------------------------------

def _seed_ingest(tmp_path: Path, model: str, fiche_tokens: int = 5000, doc: str = "DOC"):
    (tmp_path / "fiche.txt").write_text(doc, encoding="utf-8")
    cag.save_meta(tmp_path, fiche_tokens, model)


def test_restore_none_when_no_prior_ingest(tmp_path):
    assert run(cag.restore_or_reprefill(FakeClient(), tmp_path, "qwen.gguf")) is None


def test_restore_fast_path_on_model_match(tmp_path):
    _seed_ingest(tmp_path, "qwen.gguf")
    c = FakeClient(restore_ok=True, model="qwen.gguf")
    res = run(cag.restore_or_reprefill(c, tmp_path, "qwen.gguf"))
    assert res == ("DOC", 5000, True)
    assert c.restore_calls == 1 and c.prefill_calls == 0


def test_unknown_runtime_model_id_still_restores_no_reprefill(tmp_path):
    # DA remark #3b: /props polled before model load -> runtime id "unknown" must NOT be
    # treated as a mismatch; attempt the restore anyway rather than a 5-min re-prefill.
    _seed_ingest(tmp_path, "qwen.gguf")
    c = FakeClient(restore_ok=True, model="qwen.gguf")
    res = run(cag.restore_or_reprefill(c, tmp_path, "unknown"))
    assert res == ("DOC", 5000, True)
    assert c.restore_calls == 1 and c.prefill_calls == 0


def test_restore_failure_falls_back_to_reprefill(tmp_path):
    _seed_ingest(tmp_path, "qwen.gguf")
    c = FakeClient(restore_ok=False, model="qwen.gguf")
    fiche, tokens, restored = run(cag.restore_or_reprefill(c, tmp_path, "qwen.gguf"))
    assert fiche == "DOC" and restored is False
    assert c.restore_calls == 1 and c.prefill_calls == 1
    # re-prefill recomputes tokens from the system prefix (one turn-start -> 100)
    assert tokens == 100


def test_positive_model_mismatch_reprefills_without_restore(tmp_path):
    # both ids known and different -> real mismatch: skip restore, rebuild the cache
    _seed_ingest(tmp_path, "other-model.gguf")
    c = FakeClient(restore_ok=True, model="qwen.gguf")
    fiche, tokens, restored = run(cag.restore_or_reprefill(c, tmp_path, "qwen.gguf"))
    assert restored is False and c.restore_calls == 0 and c.prefill_calls == 1


def test_corrupt_meta_falls_back_to_reprefill(tmp_path):
    (tmp_path / "fiche.txt").write_text("DOC", encoding="utf-8")
    (tmp_path / "meta.json").write_text("{not valid json", encoding="utf-8")
    c = FakeClient(restore_ok=True, model="qwen.gguf")
    _, _, restored = run(cag.restore_or_reprefill(c, tmp_path, "qwen.gguf"))
    assert restored is False and c.restore_calls == 0 and c.prefill_calls == 1


# --- /api/ingest route (cap enforcement + happy path) ---------------------------------

def _fresh_slot_lock():
    # asyncio.Lock binds to the loop of first use; give each asyncio.run a fresh one so it
    # never carries over a closed loop from a previous test.
    appmain._slot_lock = asyncio.Lock()


def test_ingest_over_cap_returns_400_with_token_count():
    _fresh_slot_lock()
    over = cag.FICHE_CAP + 500
    c = FakeClient()
    async def big(text): return over
    c.tokenize = big  # type: ignore[assignment]
    appmain.state.client = c
    appmain.state.model_id = "qwen.gguf"
    resp = run(appmain.ingest(FakeRequest({"text": "x" * 100})))
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"] == "over_cap"
    assert body["tokens"] == over and body["cap"] == cag.FICHE_CAP
    assert str(over) in body["message"] and c.prefill_calls == 0


def test_ingest_empty_text_returns_400():
    _fresh_slot_lock()
    appmain.state.client = FakeClient()
    resp = run(appmain.ingest(FakeRequest({"text": "   "})))
    assert resp.status_code == 400 and json.loads(resp.body)["error"] == "empty"


def test_ingest_success_persists_and_resets_history(tmp_path, monkeypatch):
    _fresh_slot_lock()
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)
    c = FakeClient(model="qwen.gguf")
    async def small(text): return 500
    c.tokenize = small  # type: ignore[assignment]
    appmain.state.client = c
    appmain.state.model_id = "qwen.gguf"
    appmain.state.history = [("old", "turn")]
    appmain.state.pair_tokens = [123]

    resp = run(appmain.ingest(FakeRequest({"text": "HELLO DOCUMENT"})))
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["ingested"] is True and body["fiche_tokens"] == 500
    assert body["budget"] == cag.history_budget(500)

    assert (tmp_path / "fiche.txt").read_text(encoding="utf-8") == "HELLO DOCUMENT"
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["fiche_tokens"] == 500 and meta["model"] == "qwen.gguf"
    assert c.prefill_calls == 1
    # a new document is a new conversation: history + cached counts reset
    assert appmain.state.ingested is True
    assert appmain.state.history == [] and appmain.state.pair_tokens == []


# --- /api/ask guard -------------------------------------------------------------------

def test_ask_before_ingest_returns_409():
    _fresh_slot_lock()
    appmain.state.client = FakeClient()
    appmain.state.ingested = False
    resp = run(appmain.ask(FakeRequest({"question": "hello?"})))
    assert resp.status_code == 409 and json.loads(resp.body)["error"] == "not_ingested"


# --- _pair_token_count helper (DA #19a) -----------------------------------------------

def test_pair_token_count_uses_tokenizer():
    # happy path: a rendered pair has two <|im_start|> turn-starts -> FakeClient 200
    c = FakeClient()
    assert run(appmain._pair_token_count(c, "q", "a")) == 200


def test_pair_token_count_fallback_on_tokenize_error():
    # If tokenize raises mid-turn, the helper must still return a count (len//3 + 1
    # estimate) so state.pair_tokens can stay in lockstep with state.history — a desync
    # would corrupt every later trim. Assert the exact fallback so the guard can't rot.
    c = FakeClient()
    async def boom(text):
        raise cag.LlamaError("tokenizer down")
    c.tokenize = boom  # type: ignore[assignment]
    rendered = cag.render_pair("question", "answer")
    assert run(appmain._pair_token_count(c, "question", "answer")) == len(rendered) // 3 + 1


# --- /api/ask streaming happy path (DA #19b) ------------------------------------------

async def _drain(aiter):
    out = []
    async for chunk in aiter:
        out.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk)
    return out


def test_ask_streaming_appends_history_and_counts():
    # The main runtime path, with no llama: a fake stream yields two tokens then done.
    # Assert the turn lands in BOTH history and pair_tokens (equal length) and that
    # _history_tokens grows by the new pair's cached count.
    _fresh_slot_lock()
    c = FakeClient(model="qwen.gguf")

    async def fake_stream(prompt):
        assert prompt.startswith(cag.build_system_prefix("FID"))  # prompt built from state
        yield ("token", "Hello ")
        yield ("token", "world")
        yield ("done", {"predicted_ms": 1})
    c.stream_completion = fake_stream  # type: ignore[assignment]

    appmain.state.client = c
    appmain.state.ingested = True
    appmain.state.fiche = "FID"
    appmain.state.fiche_tokens = 500
    appmain.state.history = []
    appmain.state.pair_tokens = []
    appmain.state._history_tokens = 0

    resp = run(appmain.ask(FakeRequest({"question": "hi?"})))
    frames = "".join(run(_drain(resp.body_iterator)))

    assert '"token": "Hello "' in frames and '"done": true' in frames
    assert appmain.state.history == [("hi?", "Hello world")]
    assert len(appmain.state.pair_tokens) == len(appmain.state.history) == 1
    # FakeClient.tokenize counts <|im_start|> *100; a rendered pair has two -> 200
    assert appmain.state.pair_tokens[0] == 200
    assert appmain.state._history_tokens == 200
