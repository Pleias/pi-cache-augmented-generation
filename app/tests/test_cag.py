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
    prefix token counts are deterministic and additive (100 tokens per `<|im_start|>`).
    restore()/prefill_and_save() record the slot filename they were asked to operate on so
    the multi-doc tests can assert the RIGHT per-document slot was touched."""

    def __init__(self, *, restore_ok: bool = True, model: str = "qwen.gguf") -> None:
        self._restore_ok = restore_ok
        self._model = model
        self.prefill_calls = 0
        self.restore_calls = 0
        self.saved_files: list[str] = []
        self.restored_files: list[str] = []

    async def tokenize(self, text: str) -> int:
        return text.count("<|im_start|>") * 100

    async def prefill_and_save(self, system_prefix: str,
                               filename: str = cag.SLOT_FILE) -> None:
        self.prefill_calls += 1
        self.saved_files.append(filename)

    async def restore(self, filename: str = cag.SLOT_FILE) -> bool:
        self.restore_calls += 1
        self.restored_files.append(filename)
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
    doc_id = body["doc_id"]

    # The document now persists into its own library directory, and its slot is saved
    # under the per-document filename.
    dd = cag.doc_dir(tmp_path, doc_id)
    assert (dd / "fiche.txt").read_text(encoding="utf-8") == "HELLO DOCUMENT"
    meta = json.loads((dd / "meta.json").read_text(encoding="utf-8"))
    assert meta["fiche_tokens"] == 500 and meta["model"] == "qwen.gguf"
    assert c.prefill_calls == 1 and c.saved_files[-1] == cag.slot_filename(doc_id)
    # ...and it is registered in the library as the active document.
    lib = cag.load_library(tmp_path)
    assert lib.active == doc_id and len(lib.docs) == 1
    # a new document is a new conversation: history + cached counts reset
    assert appmain.state.ingested is True and appmain.state.doc_id == doc_id
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


def test_ask_streaming_appends_history_and_counts(tmp_path, monkeypatch):
    # The main runtime path, with no llama: a fake stream yields two tokens then done.
    # Assert the turn lands in BOTH history and pair_tokens (equal length) and that
    # _history_tokens grows by the new pair's cached count.
    _fresh_slot_lock()
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)  # /ask persists history.json
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
    appmain.state.doc_id = "doc1"   # history persists into this document's directory
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
    # ...and the completed turn was persisted in the active document's directory
    assert cag.load_history(cag.doc_dir(tmp_path, "doc1")) == ([("hi?", "Hello world")], [200])


# --- chat-history persistence (survives restarts alongside the fiche cache) ------------

def test_history_save_load_round_trip(tmp_path):
    hist = [("q1", "a1"), ("q2", "a2 **md**")]
    toks = [110, 240]
    cag.save_history(tmp_path, hist, toks)
    assert cag.load_history(tmp_path) == (hist, toks)


def test_history_load_missing_and_corrupt_are_empty(tmp_path):
    assert cag.load_history(tmp_path) == ([], [])
    (tmp_path / "history.json").write_text("{torn", encoding="utf-8")
    assert cag.load_history(tmp_path) == ([], [])


def test_clear_history_removes_file_and_is_idempotent(tmp_path):
    cag.save_history(tmp_path, [("q", "a")], [100])
    cag.clear_history(tmp_path)
    assert not (tmp_path / "history.json").exists()
    cag.clear_history(tmp_path)  # second call must not raise


def test_reset_clears_history_on_disk_too(tmp_path, monkeypatch):
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)
    dd = cag.doc_dir(tmp_path, "doc1")
    dd.mkdir(parents=True, exist_ok=True)
    cag.save_history(dd, [("q", "a")], [100])
    appmain.state.doc_id = "doc1"
    appmain.state.history = [("q", "a")]
    appmain.state.pair_tokens = [100]
    appmain.state._history_tokens = 100
    run(appmain.reset())
    assert appmain.state.history == [] and appmain.state._history_tokens == 0
    assert not (dd / "history.json").exists()


def test_ingest_starts_new_doc_with_empty_history(tmp_path, monkeypatch):
    # A fresh ingest is a fresh document with its own (empty) conversation: no history
    # file leaks into the new document's directory.
    _fresh_slot_lock()
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)
    c = FakeClient(model="qwen.gguf")
    async def small(text): return 500
    c.tokenize = small  # type: ignore[assignment]
    appmain.state.client = c
    appmain.state.model_id = "qwen.gguf"
    resp = run(appmain.ingest(FakeRequest({"text": "NEW DOC"})))
    assert resp.status_code == 200
    doc_id = json.loads(resp.body)["doc_id"]
    assert cag.load_history(cag.doc_dir(tmp_path, doc_id)) == ([], [])


def test_history_endpoint_returns_turns():
    appmain.state.history = [("q1", "a1"), ("q2", "a2")]
    resp = run(appmain.history())
    body = json.loads(resp.body)
    assert body["turns"] == [{"user": "q1", "assistant": "a1"},
                             {"user": "q2", "assistant": "a2"}]


def test_history_load_rejects_nonpositive_or_absurd_token_counts(tmp_path):
    # DA: a tampered count must not poison trim_history's budget math — corrupt = empty.
    (tmp_path / "history.json").write_text(
        json.dumps({"turns": [{"user": "q", "assistant": "a", "tokens": -5}]}),
        encoding="utf-8")
    assert cag.load_history(tmp_path) == ([], [])
    (tmp_path / "history.json").write_text(
        json.dumps({"turns": [{"user": "q", "assistant": "a",
                               "tokens": cag.N_CTX + 1}]}), encoding="utf-8")
    assert cag.load_history(tmp_path) == ([], [])


def test_reset_waits_for_slot_lock():
    # DA MAJOR #1: reset racing a streaming /ask must serialize on the slot lock, or the
    # in-flight turn resurrects the cleared history. Hold the lock and assert reset blocks.
    _fresh_slot_lock()

    async def scenario():
        async with appmain._slot_lock:  # simulate an in-flight /ask
            task = asyncio.ensure_future(appmain.reset())
            done, pending = await asyncio.wait({task}, timeout=0.05)
            assert not done, "reset() completed while the slot lock was held"
            assert pending
        return await task  # lock released -> reset proceeds

    resp = run(scenario())
    assert json.loads(resp.body)["ok"] is True


def test_doc_endpoint_serves_container_md(tmp_path, monkeypatch):
    doc = tmp_path / "CONTAINER.md"
    doc.write_text("# CAG app\nUsage.", encoding="utf-8")
    monkeypatch.setattr(appmain, "DOC_PATH", doc)
    body = json.loads(run(appmain.doc()).body)
    assert body["markdown"].startswith("# CAG app")


def test_doc_endpoint_404_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(appmain, "DOC_PATH", tmp_path / "nope.md")
    resp = run(appmain.doc())
    assert resp.status_code == 404 and json.loads(resp.body)["error"] == "no_doc"


def test_reprefill_startup_drops_stale_history(tmp_path):
    # DA MAJOR #2: restored=False means the fiche was re-prefilled with a (possibly) new
    # tokenizer; persisted pair token counts are stale. Lifespan drops them — emulate its
    # branch exactly: load only when ok, clear the file otherwise.
    # NOTE (review #11): this deliberately mirrors the lifespan/_activate branch inline
    # rather than driving the real async lifespan, which would need a live httpx client and
    # a running server; the shipped branch is exercised end-to-end by smoke.sh's restart.
    cag.save_history(tmp_path, [("old q", "old a")], [150])
    _seed_ingest(tmp_path, "other-model.gguf")  # positive mismatch -> re-prefill path
    c = FakeClient(restore_ok=True, model="qwen.gguf")
    _, _, ok = run(cag.restore_or_reprefill(c, tmp_path, "qwen.gguf"))
    assert ok is False
    if not ok:
        cag.clear_history(tmp_path)
    assert cag.load_history(tmp_path) == ([], [])


# ======================================================================================
# Document library (multi-slot swap), content packs, and privacy purge
# ======================================================================================

def _seed_doc(data_dir: Path, doc_id: str, *, model: str = "qwen.gguf",
              fiche_tokens: int = 500, doc: str = "DOC",
              history: list[tuple[str, str, int]] | None = None) -> cag.DocEntry:
    """Materialise one library document on disk: its dir (fiche/meta[/history]) and a
    placeholder slot file, plus return a matching DocEntry the caller can index."""
    dd = cag.doc_dir(data_dir, doc_id)
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "fiche.txt").write_text(doc, encoding="utf-8")
    cag.save_meta(dd, fiche_tokens, model)
    if history:
        cag.save_history(dd, [(u, a) for u, a, _ in history],
                         [t for _, _, t in history])
    cag.cache_root(data_dir).mkdir(parents=True, exist_ok=True)
    cag.slot_path(data_dir, doc_id).write_bytes(b"SLOT-" + doc_id.encode())
    return cag.DocEntry(id=doc_id, title=doc_id.upper(), fiche_tokens=fiche_tokens,
                        created="2026-01-01T00:00:00Z", last_used="2026-01-01T00:00:00Z")


# --- library index --------------------------------------------------------------------

def test_library_index_round_trip(tmp_path):
    lib = cag.Library(docs=[cag.DocEntry("a", "Title", 100, "c", "u")], active="a")
    cag.save_library(tmp_path, lib)
    got = cag.load_library(tmp_path)
    assert got.active == "a"
    assert got.docs[0].to_dict() == lib.docs[0].to_dict()
    # atomic write leaves no temp file behind
    assert not (tmp_path / "library.json.tmp").exists()


def test_library_load_missing_or_corrupt_is_empty(tmp_path):
    assert cag.load_library(tmp_path).docs == []
    cag.library_path(tmp_path).write_text("{torn", encoding="utf-8")
    got = cag.load_library(tmp_path)
    assert got.docs == [] and got.active is None
    # a corrupt index is NEVER silently unlinked — it is preserved aside for recovery
    assert (tmp_path / "library.json.corrupt-0").exists()
    assert not cag.library_path(tmp_path).exists()


def test_library_most_recent_prefers_latest_last_used(tmp_path):
    lib = cag.Library(docs=[
        cag.DocEntry("a", "A", 1, "2026-01-01", "2026-01-01"),
        cag.DocEntry("b", "B", 1, "2026-02-01", "2026-03-01"),
        cag.DocEntry("c", "C", 1, "2026-02-15", "2026-02-15"),
    ], active="a")
    assert lib.most_recent().id == "b"


# --- ingest into the library ----------------------------------------------------------

def test_ingest_creates_library_entry_with_title(tmp_path, monkeypatch):
    _fresh_slot_lock()
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)
    c = FakeClient(model="qwen.gguf")
    async def small(text): return 500
    c.tokenize = small  # type: ignore[assignment]
    appmain.state.client = c
    appmain.state.model_id = "qwen.gguf"
    resp = run(appmain.ingest(FakeRequest({"text": "HELLO", "title": "My Doc"})))
    assert resp.status_code == 200
    body = json.loads(resp.body)
    doc_id = body["doc_id"]
    assert body["title"] == "My Doc" and body["docs_count"] == 1
    lib = cag.load_library(tmp_path)
    assert lib.active == doc_id and len(lib.docs) == 1
    assert lib.docs[0].title == "My Doc"
    assert c.saved_files[-1] == cag.slot_filename(doc_id)


def test_ingest_refuses_beyond_doc_cap(tmp_path, monkeypatch):
    _fresh_slot_lock()
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)
    docs = [cag.DocEntry(f"d{i}", f"T{i}", 100, "t", "t") for i in range(cag.MAX_DOCS)]
    cag.save_library(tmp_path, cag.Library(docs=docs, active="d0"))
    c = FakeClient(model="qwen.gguf")
    appmain.state.client = c
    appmain.state.model_id = "qwen.gguf"
    resp = run(appmain.ingest(FakeRequest({"text": "NEW"})))
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"] == "library_full" and body["cap"] == cag.MAX_DOCS
    # the cap is checked BEFORE the ~5-min prefill, so nothing was prefilled
    assert c.prefill_calls == 0


# --- switch ---------------------------------------------------------------------------

def test_switch_restores_right_slot_and_history(tmp_path, monkeypatch):
    _fresh_slot_lock()
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)
    _seed_doc(tmp_path, "aaa", history=[("q1", "a1", 120)])
    _seed_doc(tmp_path, "bbb", fiche_tokens=700, doc="DOC2",
              history=[("q2", "a2", 200)])
    cag.save_library(tmp_path, cag.Library(
        docs=[cag.DocEntry("aaa", "A", 500, "t", "t"),
              cag.DocEntry("bbb", "B", 700, "t", "t")], active="aaa"))
    c = FakeClient(model="qwen.gguf")
    appmain.state.client = c
    appmain.state.model_id = "qwen.gguf"

    resp = run(appmain.switch(FakeRequest({"id": "bbb"})))
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["doc_id"] == "bbb" and body["fiche_tokens"] == 700
    # restored the RIGHT per-document slot file, no re-prefill
    assert c.restored_files[-1] == cag.slot_filename("bbb") and c.prefill_calls == 0
    # loaded THAT document's history
    assert appmain.state.history == [("q2", "a2")]
    assert cag.load_library(tmp_path).active == "bbb"


def test_switch_unknown_doc_returns_404(tmp_path, monkeypatch):
    _fresh_slot_lock()
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)
    cag.save_library(tmp_path, cag.Library(docs=[], active=None))
    appmain.state.client = FakeClient()
    appmain.state.model_id = "qwen.gguf"
    resp = run(appmain.switch(FakeRequest({"id": "nope"})))
    assert resp.status_code == 404 and json.loads(resp.body)["error"] == "unknown_doc"


# --- delete ---------------------------------------------------------------------------

def test_delete_active_falls_back_to_most_recent(tmp_path, monkeypatch):
    _fresh_slot_lock()
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)
    _seed_doc(tmp_path, "aaa")
    _seed_doc(tmp_path, "bbb", fiche_tokens=700, doc="DOC2")
    cag.save_library(tmp_path, cag.Library(docs=[
        cag.DocEntry("aaa", "A", 500, "2026-01-01", "2026-01-01"),
        cag.DocEntry("bbb", "B", 700, "2026-02-01", "2026-02-01"),
    ], active="aaa"))
    c = FakeClient(model="qwen.gguf")
    appmain.state.client = c
    appmain.state.model_id = "qwen.gguf"

    resp = run(appmain.delete_doc("aaa"))
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["doc_id"] == "bbb"          # fell back to the most-recent remaining doc
    assert not cag.doc_dir(tmp_path, "aaa").exists()
    assert not cag.slot_path(tmp_path, "aaa").exists()
    lib = cag.load_library(tmp_path)
    assert lib.active == "bbb" and len(lib.docs) == 1
    assert c.restored_files[-1] == cag.slot_filename("bbb")


def test_delete_last_doc_resets_to_not_ingested(tmp_path, monkeypatch):
    _fresh_slot_lock()
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)
    _seed_doc(tmp_path, "solo")
    cag.save_library(tmp_path, cag.Library(
        docs=[cag.DocEntry("solo", "S", 500, "t", "t")], active="solo"))
    appmain.state.client = FakeClient(model="qwen.gguf")
    appmain.state.model_id = "qwen.gguf"
    appmain.state.ingested = True
    appmain.state.doc_id = "solo"

    resp = run(appmain.delete_doc("solo"))
    body = json.loads(resp.body)
    assert body["ingested"] is False and body["doc_id"] is None
    assert body["docs_count"] == 0
    assert cag.load_library(tmp_path).active is None
    assert appmain.state.ingested is False


def test_delete_unknown_doc_returns_404(tmp_path, monkeypatch):
    _fresh_slot_lock()
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)
    cag.save_library(tmp_path, cag.Library(docs=[], active=None))
    resp = run(appmain.delete_doc("nope"))
    assert resp.status_code == 404


# --- legacy migration -----------------------------------------------------------------

def test_migrate_legacy_folds_single_doc_and_preserves_slot(tmp_path):
    (tmp_path / "fiche.txt").write_text("Le jour et la nuit\nblah blah", encoding="utf-8")
    cag.save_meta(tmp_path, 4200, "qwen.gguf")
    cag.save_history(tmp_path, [("q", "a")], [100])
    cag.cache_root(tmp_path).mkdir(parents=True, exist_ok=True)
    (cag.cache_root(tmp_path) / cag.SLOT_FILE).write_bytes(b"LEGACYSLOT")

    cag.migrate_legacy(tmp_path)

    lib = cag.load_library(tmp_path)
    assert lib.active == "legacy0" and len(lib.docs) == 1
    assert lib.docs[0].title == "Le jour et la nuit"
    assert lib.docs[0].fiche_tokens == 4200
    dd = cag.doc_dir(tmp_path, "legacy0")
    assert (dd / "fiche.txt").exists() and (dd / "meta.json").exists()
    assert (dd / "history.json").exists()
    # the slot was RENAMED (not re-prefilled): bytes preserved under the new name
    assert cag.slot_path(tmp_path, "legacy0").read_bytes() == b"LEGACYSLOT"
    assert not (cag.cache_root(tmp_path) / cag.SLOT_FILE).exists()
    # legacy root files were moved away
    assert not (tmp_path / "fiche.txt").exists()

    # idempotent: a second migration is a no-op
    cag.migrate_legacy(tmp_path)
    lib2 = cag.load_library(tmp_path)
    assert lib2.active == "legacy0" and len(lib2.docs) == 1


def test_migrate_legacy_noop_without_legacy_files(tmp_path):
    cag.migrate_legacy(tmp_path)
    assert not cag.library_path(tmp_path).exists()


# --- content packs: export / import ---------------------------------------------------

def test_build_pack_contains_expected_members(tmp_path):
    _seed_doc(tmp_path, "aaa")
    cag.save_library(tmp_path, cag.Library(
        docs=[cag.DocEntry("aaa", "A", 500, "t", "t")], active="aaa"))
    spooled = cag.build_pack(tmp_path)
    import tarfile as _tf
    with _tf.open(fileobj=spooled, mode="r") as tar:
        names = set(tar.getnames())
    assert "library.json" in names
    assert "docs/aaa/fiche.txt" in names
    assert "cache/aaa.bin" in names


def test_import_pack_round_trip_merges_new_docs(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _seed_doc(src, "aaa")
    cag.save_library(src, cag.Library(
        docs=[cag.DocEntry("aaa", "A", 500, "t", "t")], active="aaa"))
    spooled = cag.build_pack(src)

    dst = tmp_path / "dst"; dst.mkdir()
    newest = cag.import_pack(dst, spooled)
    assert newest == "aaa"
    assert cag.slot_path(dst, "aaa").read_bytes() == b"SLOT-aaa"
    assert cag.load_library(dst).get("aaa") is not None


def test_import_pack_skips_id_collisions(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _seed_doc(src, "dup", doc="IMPORTED")
    cag.save_library(src, cag.Library(
        docs=[cag.DocEntry("dup", "Imported", 500, "t", "t")], active="dup"))
    spooled = cag.build_pack(src)

    dst = tmp_path / "dst"; dst.mkdir()
    _seed_doc(dst, "dup", doc="EXISTING")
    cag.save_library(dst, cag.Library(
        docs=[cag.DocEntry("dup", "Existing", 500, "t", "t")], active="dup"))

    newest = cag.import_pack(dst, spooled)
    assert newest is None  # id collision -> imported copy skipped
    assert (cag.doc_dir(dst, "dup") / "fiche.txt").read_text() == "EXISTING"


def test_import_pack_respects_total_cap(tmp_path):
    # The library cap is a single number (MAX_DOCS = 20) for both ingest and import.
    dst = tmp_path / "dst"; dst.mkdir()
    dst_docs = [cag.DocEntry(f"e{i}", f"E{i}", 1, "t", "t")
                for i in range(cag.MAX_DOCS)]
    cag.save_library(dst, cag.Library(docs=dst_docs, active="e0"))
    src = tmp_path / "src"; src.mkdir()
    _seed_doc(src, "extra")
    cag.save_library(src, cag.Library(
        docs=[cag.DocEntry("extra", "X", 1, "t", "t")], active="extra"))
    spooled = cag.build_pack(src)
    import pytest
    with pytest.raises(cag.PackError):
        cag.import_pack(dst, spooled)


def _tar_with_member(name: str) -> "object":
    import io, tarfile as _tf
    buf = io.BytesIO()
    with _tf.open(fileobj=buf, mode="w") as tar:
        data = b"x"
        info = _tf.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def test_import_pack_rejects_path_traversal(tmp_path):
    import pytest
    with pytest.raises(cag.PackError):
        cag.import_pack(tmp_path, _tar_with_member("../evil.txt"))
    # nothing was written outside the volume
    assert not (tmp_path.parent / "evil.txt").exists()


def test_import_pack_rejects_absolute_path(tmp_path):
    import pytest
    with pytest.raises(cag.PackError):
        cag.import_pack(tmp_path, _tar_with_member("/etc/evil"))


def test_import_pack_rejects_unexpected_toplevel(tmp_path):
    import pytest
    with pytest.raises(cag.PackError):
        cag.import_pack(tmp_path, _tar_with_member("secrets/keys.txt"))


# --- wipe -----------------------------------------------------------------------------

def test_wipe_all_empties_volume(tmp_path):
    _seed_doc(tmp_path, "aaa")
    _seed_doc(tmp_path, "bbb")
    cag.save_library(tmp_path, cag.Library(docs=[
        cag.DocEntry("aaa", "A", 500, "t", "t"),
        cag.DocEntry("bbb", "B", 500, "t", "t")], active="aaa"))
    cag.wipe_all(tmp_path)
    assert not cag.library_path(tmp_path).exists()
    assert not cag.docs_root(tmp_path).exists()
    assert list(cag.cache_root(tmp_path).glob("*.bin")) == []
    # cache dir is recreated for llama-server's --slot-save-path
    assert cag.cache_root(tmp_path).is_dir()


def test_wipe_endpoint_resets_state(tmp_path, monkeypatch):
    _fresh_slot_lock()
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)
    _seed_doc(tmp_path, "aaa")
    cag.save_library(tmp_path, cag.Library(
        docs=[cag.DocEntry("aaa", "A", 500, "t", "t")], active="aaa"))
    appmain.state.ingested = True
    appmain.state.doc_id = "aaa"
    appmain.state.docs_count = 1
    resp = run(appmain.wipe())
    assert json.loads(resp.body)["ok"] is True
    assert appmain.state.ingested is False and appmain.state.doc_id is None
    assert appmain.state.docs_count == 0
    assert not cag.library_path(tmp_path).exists()


# --- status shape ---------------------------------------------------------------------

def test_status_includes_library_fields(tmp_path, monkeypatch):
    appmain.state.reset_session()
    appmain.state.docs_count = 3
    appmain.state.doc_id = None
    body = json.loads(run(appmain.status()).body)
    assert body["docs_count"] == 3
    assert body["doc_id"] is None and body["title"] == ""


# ======================================================================================
# DA review hardening — import size/type guards, migration recovery, stream watchdog
# ======================================================================================

def _make_pack(members: list[tuple[str, bytes]], lib_ids: list[str]) -> "object":
    """Build an in-memory tar pack: a library.json listing lib_ids plus the given file
    members (name, bytes). Lets a test craft oversized / bomb-shaped payloads directly."""
    import io, json as _json, tarfile as _tf
    buf = io.BytesIO()
    with _tf.open(fileobj=buf, mode="w") as tar:
        lib = _json.dumps({
            "docs": [{"id": i, "title": i, "fiche_tokens": 1,
                      "created": "t", "last_used": "t"} for i in lib_ids],
            "active": lib_ids[0] if lib_ids else None,
        }).encode()
        info = _tf.TarInfo("library.json"); info.size = len(lib)
        tar.addfile(info, io.BytesIO(lib))
        for name, data in members:
            info = _tf.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def _tar_with_special(name: str, ttype, linkname: str) -> "object":
    import io, tarfile as _tf
    buf = io.BytesIO()
    with _tf.open(fileobj=buf, mode="w") as tar:
        info = _tf.TarInfo(name=name)
        info.type = ttype
        info.linkname = linkname
        tar.addfile(info)
    buf.seek(0)
    return buf


def test_import_pack_rejects_symlink_member(tmp_path):
    import tarfile as _tf, pytest
    buf = _tar_with_special("docs/x/evil", _tf.SYMTYPE, "/etc/passwd")
    with pytest.raises(cag.PackError):
        cag.import_pack(tmp_path, buf)
    assert not cag.doc_dir(tmp_path, "x").exists()
    assert not (tmp_path / "etc").exists()


def test_import_pack_rejects_hardlink_member(tmp_path):
    import tarfile as _tf, pytest
    buf = _tar_with_special("docs/x/evil", _tf.LNKTYPE, "docs/x/fiche.txt")
    with pytest.raises(cag.PackError):
        cag.import_pack(tmp_path, buf)
    assert not cag.doc_dir(tmp_path, "x").exists()


def test_import_pack_rejects_oversized_member(tmp_path):
    # A single member whose header size exceeds the cap is rejected before it is written.
    import pytest
    buf = _make_pack([("docs/big/fiche.txt", b"x" * 4000)], ["big"])
    with pytest.raises(cag.PackTooLargeError):
        cag.import_pack(tmp_path, buf, max_total_bytes=1000)
    assert not cag.doc_dir(tmp_path, "big").exists()  # rolled back


def test_import_pack_aborts_on_running_total(tmp_path):
    # Each member is under the per-member cap, but their cumulative bytes exceed it — the
    # running-total guard (the real decompression-bomb defence) fires mid-copy and rolls back.
    import pytest
    buf = _make_pack(
        [("docs/big/fiche.txt", b"a" * 600), ("cache/big.bin", b"b" * 600)], ["big"])
    with pytest.raises(cag.PackTooLargeError):
        cag.import_pack(tmp_path, buf, max_total_bytes=1000)
    assert not cag.doc_dir(tmp_path, "big").exists()
    assert not cag.slot_path(tmp_path, "big").exists()


def test_import_pack_free_space_guard(tmp_path, monkeypatch):
    import collections, pytest
    Usage = collections.namedtuple("Usage", "total used free")
    monkeypatch.setattr(cag.shutil, "disk_usage", lambda p: Usage(1 << 40, 1 << 40, 10))
    buf = _make_pack([("docs/big/fiche.txt", b"hello")], ["big"])
    with pytest.raises(cag.PackTooLargeError):
        cag.import_pack(tmp_path, buf)
    assert not cag.doc_dir(tmp_path, "big").exists()  # nothing extracted


def test_migrate_legacy_recovers_from_crash_before_index(tmp_path):
    # DA #2: a power loss after the files were moved into docs/legacy0 but BEFORE
    # library.json was written must not orphan the document. A fresh migrate/startup rebuilds
    # the index from docs/legacy0/meta.json.
    dd = cag.doc_dir(tmp_path, "legacy0")
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "fiche.txt").write_text("Recovered lesson\nbody", encoding="utf-8")
    cag.save_meta(dd, 1234, "qwen.gguf")
    cag.cache_root(tmp_path).mkdir(parents=True, exist_ok=True)
    cag.slot_path(tmp_path, "legacy0").write_bytes(b"SLOT")
    assert not cag.library_path(tmp_path).exists()  # crashed before the index write

    cag.migrate_legacy(tmp_path)

    lib = cag.load_library(tmp_path)
    assert lib.active == "legacy0" and len(lib.docs) == 1
    assert lib.docs[0].fiche_tokens == 1234
    assert lib.docs[0].title == "Recovered lesson"
    # the preserved slot is untouched by recovery
    assert cag.slot_path(tmp_path, "legacy0").read_bytes() == b"SLOT"


def test_migrate_legacy_completes_partial_move(tmp_path):
    # Crash between moving fiche and meta: fiche already in docs/legacy0, meta still at root.
    # Re-running must finish the move and index the doc, not lose it.
    dd = cag.doc_dir(tmp_path, "legacy0")
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "fiche.txt").write_text("Half moved", encoding="utf-8")
    cag.save_meta(tmp_path, 900, "qwen.gguf")  # meta.json still at the root
    cag.migrate_legacy(tmp_path)
    lib = cag.load_library(tmp_path)
    assert lib.active == "legacy0" and lib.docs[0].fiche_tokens == 900
    assert (dd / "meta.json").exists() and not (tmp_path / "meta.json").exists()


def test_ask_stream_error_releases_lock_and_emits_error(tmp_path, monkeypatch):
    # DA #5: a wedged stream must not pin _slot_lock forever. A LlamaError from the stream
    # (what an httpx read-timeout raises) has to surface as an SSE error frame AND leave the
    # lock free for the next mutating request.
    _fresh_slot_lock()
    monkeypatch.setattr(appmain, "DATA_DIR", tmp_path)
    c = FakeClient(model="qwen.gguf")

    async def hung_stream(prompt):
        raise cag.LlamaError("read timeout: llama-server wedged")
        yield  # unreachable — makes this an async generator

    c.stream_completion = hung_stream  # type: ignore[assignment]
    appmain.state.client = c
    appmain.state.ingested = True
    appmain.state.fiche = "F"
    appmain.state.fiche_tokens = 500
    appmain.state.doc_id = "d"
    appmain.state.history = []
    appmain.state.pair_tokens = []
    appmain.state._history_tokens = 0

    resp = run(appmain.ask(FakeRequest({"question": "hi?"})))
    frames = "".join(run(_drain(resp.body_iterator)))
    assert '"error"' in frames
    # the lock was released despite the mid-stream failure
    assert not appmain._slot_lock.locked()


def test_pack_manifest_snapshots_small_files_and_lists_slots(tmp_path):
    # DA #3: export snapshots only the small files under the lock; slot .bin are references
    # copied off-lock. Verify the split.
    _seed_doc(tmp_path, "aaa", history=[("q", "a", 100)])
    cag.save_library(tmp_path, cag.Library(
        docs=[cag.DocEntry("aaa", "A", 500, "t", "t")], active="aaa"))
    small, slots = cag.pack_manifest(tmp_path)
    small_names = {n for n, _ in small}
    assert "library.json" in small_names
    assert "docs/aaa/fiche.txt" in small_names and "docs/aaa/meta.json" in small_names
    assert "docs/aaa/history.json" in small_names
    assert [n for n, _ in slots] == ["cache/aaa.bin"]


def test_build_pack_skips_vanished_slot(tmp_path):
    # A slot deleted between snapshot and read is skipped; the tar is still valid.
    _seed_doc(tmp_path, "aaa")
    cag.save_library(tmp_path, cag.Library(
        docs=[cag.DocEntry("aaa", "A", 500, "t", "t")], active="aaa"))
    small, slots = cag.pack_manifest(tmp_path)
    cag.slot_path(tmp_path, "aaa").unlink()  # vanishes after the snapshot
    spooled = cag.build_pack_from_manifest(small, slots)
    import tarfile as _tf
    with _tf.open(fileobj=spooled, mode="r") as tar:
        names = set(tar.getnames())
    assert "library.json" in names and "docs/aaa/fiche.txt" in names
    assert "cache/aaa.bin" not in names  # skipped, no crash
