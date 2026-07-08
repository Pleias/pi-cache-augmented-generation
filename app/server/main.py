"""FastAPI app for the CAG example: static chat page + JSON/SSE API over llama-server.

One container runs llama-server (internal, 127.0.0.1:8080) and this app (published).
The app builds ChatML prompts by hand, drives the KV-cache slot save/restore, and
streams answers to the browser. All llama-server mechanics live in cag.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import tarfile
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import cag

LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
# Upper bound on an uploaded content pack (bytes). 20 docs x ~300 MB ≈ 6 GB; the 8 GB
# default leaves headroom while still rejecting a runaway/poisoned pack before it fills the
# SD. Override with PACK_MAX_BYTES for larger fleets.
PACK_MAX_BYTES = int(os.environ.get("PACK_MAX_BYTES", cag.DEFAULT_MAX_PACK_BYTES))
STATIC_DIR = Path(__file__).resolve().parent / "static"
# Usage doc baked into the image at / by the Dockerfile; when running from a checkout
# (tests, local dev) fall back to the copy next to the Dockerfile.
DOC_PATH = Path(os.environ.get("DOC_PATH",
                               "/CONTAINER.md" if Path("/CONTAINER.md").exists()
                               else str(Path(__file__).resolve().parents[1] / "CONTAINER.md")))

# llama-server runs a single KV slot (slot 0) and this process holds one shared history.
# Serialize every /ask and /ingest so two browser tabs, a reload mid-answer, or a double
# request cannot interleave completions on the same slot or race the shared state.
_slot_lock = asyncio.Lock()

# Read the single-page app once at import: serving it must not do a blocking SD read on
# the event loop for every page load.
INDEX_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")


class AppState:
    """In-memory view of the current ingest and chat session.

    The cached document + its KV slot are the durable state (on the volume); this object
    just mirrors what is loaded so requests avoid re-reading disk on every call.
    """

    def __init__(self) -> None:
        self.client: cag.LlamaClient | None = None
        self.model_id: str = "unknown"
        self.ingested: bool = False
        self.restored: bool = False
        self.fiche: str = ""
        self.fiche_tokens: int = 0
        # The active document in the library (None when nothing is loaded); its title and
        # the library size are mirrored here so /api/status never has to read disk.
        self.doc_id: str | None = None
        self.title: str = ""
        self.docs_count: int = 0
        self.history: list[tuple[str, str]] = []
        # cached ChatML token count of each history pair, kept in lockstep with `history`
        # so trim_history budgets by summation instead of re-tokenizing the whole history
        # on every /ask (only each new pair is tokenized, once).
        self.pair_tokens: list[int] = []
        # last accurate history token count (sum of pair_tokens after the latest trim)
        self._history_tokens: int = 0

    def reset_session(self) -> None:
        """Return to the "no active document" state (delete-last / wipe). docs_count is
        managed by the caller since it reflects the library, not the loaded session."""
        self.ingested = False
        self.restored = False
        self.fiche = ""
        self.fiche_tokens = 0
        self.doc_id = None
        self.title = ""
        self.history = []
        self.pair_tokens = []
        self._history_tokens = 0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the shared HTTP client, migrate any legacy layout, and restore the active
    document from the library on boot."""
    async with httpx.AsyncClient() as http:
        state.client = cag.LlamaClient(LLAMA_URL, http)
        state.model_id = await state.client.model_id()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Fold a pre-library single-doc volume into the library layout (idempotent).
        await anyio.to_thread.run_sync(cag.migrate_legacy, DATA_DIR)
        lib = await anyio.to_thread.run_sync(cag.load_library, DATA_DIR)
        state.docs_count = len(lib.docs)
        if lib.active and lib.get(lib.active) is not None:
            try:
                ok = await _activate(lib.active)
            except cag.LlamaError as e:
                # server not ready or restore/prefill failed hard: start without an active
                # doc, let the user pick one. Logged so the operator can see why.
                print(f"[startup] restore failed, no active document: {e}", flush=True)
                ok = False
            if ok:
                print(f"[startup] active doc {state.doc_id!r} loaded: {state.fiche_tokens} "
                      f"prefix tokens, restored={state.restored}, "
                      f"history turns={len(state.history)}", flush=True)
            else:
                print("[startup] active document could not be restored", flush=True)
        else:
            print(f"[startup] no active document ({state.docs_count} in library)",
                  flush=True)
        yield


app = FastAPI(title="CAG Example", lifespan=lifespan)


def _client() -> cag.LlamaClient:
    assert state.client is not None  # set in lifespan before any request is served
    return state.client


def _write_text(path: Path, text: str) -> None:
    """Blocking UTF-8 write, run off the event loop via anyio.to_thread on slow SD."""
    path.write_text(text, encoding="utf-8")


async def _pair_token_count(client: cag.LlamaClient, user: str, assistant: str) -> int:
    """Token count of one new history pair, with a conservative fallback.

    `state.pair_tokens` must stay in lockstep with `state.history`, so a tokenizer hiccup
    right after an answer cannot be allowed to desync them. On failure, estimate high
    (~3 chars/token) so the budget check errs toward trimming rather than overflowing ctx.
    """
    rendered = cag.render_pair(user, assistant)
    try:
        return await client.tokenize(rendered)
    except cag.LlamaError:
        return len(rendered) // 3 + 1


def _status_payload() -> dict:
    """The session snapshot returned by /api/status and every endpoint that changes the
    active document (switch / delete / import), so the front updates from one shape."""
    return {
        "ingested": state.ingested,
        "fiche_tokens": state.fiche_tokens,
        "history_tokens": state._history_tokens,
        "budget": cag.history_budget(state.fiche_tokens) if state.ingested else 0,
        "model": state.model_id,
        "restored": state.restored,
        "doc_id": state.doc_id,
        "title": state.title,
        "docs_count": state.docs_count,
    }


async def _activate(doc_id: str) -> bool:
    """Make `doc_id` the live document: restore its saved KV slot into slot 0 (or
    re-prefill on a miss), load its persisted history into state, and mark it active in
    the index. Caller MUST hold _slot_lock. Returns False if the id is unknown.

    Mirrors the startup restore path exactly: on a genuine restore the conversation is
    replayed; on a re-prefill (model change / restore failure) the stale-tokenizer history
    is dropped.
    """
    lib = await anyio.to_thread.run_sync(cag.load_library, DATA_DIR)
    entry = lib.get(doc_id)
    if entry is None:
        return False
    dd = cag.doc_dir(DATA_DIR, doc_id)
    restored = await cag.restore_or_reprefill(
        _client(), dd, state.model_id, cag.slot_filename(doc_id))
    if restored is None:
        return False
    fiche, tokens, ok = restored
    state.fiche = fiche
    state.fiche_tokens = tokens
    state.ingested = True
    state.restored = ok
    state.doc_id = doc_id
    state.title = entry.title
    if ok:
        state.history, state.pair_tokens = await anyio.to_thread.run_sync(
            cag.load_history, dd)
    else:
        await anyio.to_thread.run_sync(cag.clear_history, dd)
        state.history, state.pair_tokens = [], []
    state._history_tokens = sum(state.pair_tokens)
    # Record the activation (and a re-prefill's fresh token count) back to the index.
    entry.fiche_tokens = tokens
    entry.last_used = cag.now_iso()
    lib.active = doc_id
    await anyio.to_thread.run_sync(cag.save_library, DATA_DIR, lib)
    state.docs_count = len(lib.docs)
    return True


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-file chat page (read once at import)."""
    return HTMLResponse(INDEX_HTML)


@app.post("/api/ingest")
async def ingest(request: Request) -> JSONResponse:
    """Tokenize the pasted document, enforce the caps, prefill + save its KV slot, and add
    it to the library as a NEW active document (its own dir + `<id>.bin` slot).

    A fresh document is a fresh conversation. The library size cap (US-18) is checked
    before the expensive prefill so an over-full library fails fast.
    """
    payload = await request.json()
    text = (payload or {}).get("text", "")
    title = (payload or {}).get("title")
    title = title if isinstance(title, str) else None
    if not isinstance(text, str) or not text.strip():
        return JSONResponse(status_code=400,
                            content={"error": "empty", "message": "No text to ingest."})

    client = _client()
    system_prefix = cag.build_system_prefix(text)
    # Serialize against /ask and other /ingest calls: prefill + save mutate the single slot.
    async with _slot_lock:
        lib = await anyio.to_thread.run_sync(cag.load_library, DATA_DIR)
        if len(lib.docs) >= cag.MAX_DOCS:
            return JSONResponse(status_code=400, content={
                "error": "library_full", "count": len(lib.docs), "cap": cag.MAX_DOCS,
                "message": (
                    f"The library already holds {len(lib.docs)} documents (max "
                    f"{cag.MAX_DOCS}). Delete one before ingesting another — each saved "
                    "cache is ~300 MB on disk."),
            })

        try:
            tokens = await client.tokenize(system_prefix)
        except cag.LlamaError as e:
            return JSONResponse(status_code=503, content={"error": "llama_unreachable",
                                "message": f"llama-server not reachable: {e}"})

        if tokens > cag.FICHE_CAP:
            return JSONResponse(status_code=400, content={
                "error": "over_cap", "tokens": tokens, "cap": cag.FICHE_CAP,
                "message": (
                    f"Document is {tokens} tokens, over the {cag.FICHE_CAP}-token cap. "
                    "The whole document is held in the model's KV cache, which must fit the "
                    "device's RAM and still leave room for the chat. Shorten the text and "
                    "try again."),
            })

        doc_id = cag.new_doc_id()
        try:
            await client.prefill_and_save(system_prefix, cag.slot_filename(doc_id))
        except cag.LlamaError as e:
            return JSONResponse(status_code=503, content={"error": "llama_unreachable",
                                "message": f"prefill failed: {e}"})

        # Persist the document into its own library directory (blocking I/O off the loop).
        dd = cag.doc_dir(DATA_DIR, doc_id)
        await anyio.to_thread.run_sync(lambda: dd.mkdir(parents=True, exist_ok=True))
        await anyio.to_thread.run_sync(_write_text, dd / "fiche.txt", text)
        await anyio.to_thread.run_sync(cag.save_meta, dd, tokens, state.model_id)
        await anyio.to_thread.run_sync(cag.clear_history, dd)  # fresh conversation

        entry = cag.DocEntry(
            id=doc_id, title=cag.derive_title(text, title),
            fiche_tokens=tokens, created=cag.now_iso(), last_used=cag.now_iso())
        lib.docs.append(entry)
        lib.active = doc_id
        await anyio.to_thread.run_sync(cag.save_library, DATA_DIR, lib)

        state.fiche = text
        state.fiche_tokens = tokens
        state.ingested = True
        state.restored = False  # freshly prefilled this session, not restored from disk
        state.doc_id = doc_id
        state.title = entry.title
        state.history = []
        state.pair_tokens = []
        state._history_tokens = 0
        state.docs_count = len(lib.docs)
    return JSONResponse(content={
        "ingested": True, "fiche_tokens": tokens,
        "budget": cag.history_budget(tokens), "model": state.model_id,
        "doc_id": doc_id, "title": entry.title, "docs_count": state.docs_count,
    })


@app.post("/api/ask")
async def ask(request: Request):
    """Stream an answer for `question` as SSE, appending the turn to history."""
    payload = await request.json()
    question = (payload or {}).get("question", "")
    if not isinstance(question, str) or not question.strip():
        return JSONResponse(status_code=400,
                            content={"error": "empty", "message": "No question."})
    if not state.ingested:
        return JSONResponse(status_code=409, content={
            "error": "not_ingested",
            "message": "Ingest a document before asking questions."})

    client = _client()

    async def event_stream():
        # Hold the slot lock for the whole exchange: trim, prompt build, streamed
        # generation and the history append all touch the single shared slot + state.
        async with _slot_lock:
            pieces: list[str] = []
            try:
                # Trim to the budget from cached per-pair counts — no re-tokenize hop.
                state.history, state.pair_tokens, hist_tokens = cag.trim_history(
                    state.fiche_tokens, state.history, state.pair_tokens)
                state._history_tokens = hist_tokens
                prompt = cag.build_prompt(state.fiche, state.history, question)
                async for kind, chunk in client.stream_completion(prompt):
                    if kind == "token":
                        pieces.append(chunk)  # type: ignore[arg-type]
                        yield f"data: {json.dumps({'token': chunk})}\n\n"
                    else:  # done
                        answer = "".join(pieces).strip()
                        if answer:
                            # Tokenize ONLY the new pair, once, then cache its count so
                            # later /ask calls never re-tokenize the whole history.
                            pair_tok = await _pair_token_count(client, question, answer)
                            state.history.append((question, answer))
                            state.pair_tokens.append(pair_tok)
                            state._history_tokens += pair_tok
                            # Persist the conversation after each completed turn (small
                            # JSON, worker thread) so it survives restarts alongside the
                            # fiche cache. Written into the ACTIVE document's directory.
                            # Aborted answers never reach this point — they stay out of
                            # history, memory and disk alike.
                            if state.doc_id is not None:
                                await anyio.to_thread.run_sync(
                                    cag.save_history,
                                    cag.doc_dir(DATA_DIR, state.doc_id),
                                    state.history, state.pair_tokens)
                        yield f"data: {json.dumps({'done': True, 'timings': chunk})}\n\n"
            except cag.LlamaError as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/reset")
async def reset() -> JSONResponse:
    """Clear chat history (memory + disk). The cached document and its KV slot are kept.

    Takes the slot lock: a reset racing an in-flight streaming /ask would otherwise be
    silently undone (the /ask completion appends its turn and re-writes history.json
    after the clear). Under the lock, reset simply waits for the answer to finish.
    """
    async with _slot_lock:
        state.history = []
        state.pair_tokens = []
        state._history_tokens = 0
        if state.doc_id is not None:
            await anyio.to_thread.run_sync(
                cag.clear_history, cag.doc_dir(DATA_DIR, state.doc_id))
    return JSONResponse(content={"ok": True})


@app.get("/api/history")
async def history() -> JSONResponse:
    """The current conversation (mirrors the persisted file), for page replay."""
    return JSONResponse(content={
        "turns": [{"user": u, "assistant": a} for u, a in state.history],
    })


@app.get("/api/doc")
async def doc() -> JSONResponse:
    """The usage doc baked into the image (CONTAINER.md), rendered by the page's
    markdown viewer behind the header's Docs button."""
    try:
        text = await anyio.to_thread.run_sync(
            lambda: DOC_PATH.read_text(encoding="utf-8"))
    except OSError:
        return JSONResponse(status_code=404, content={
            "error": "no_doc", "message": "CONTAINER.md not found in this build."})
    return JSONResponse(content={"markdown": text})


@app.get("/api/status")
async def status() -> JSONResponse:
    """Session snapshot for the page's status bar and view routing."""
    return JSONResponse(content=_status_payload())


# --- document library ------------------------------------------------------------------

@app.get("/api/library")
async def library() -> JSONResponse:
    """The library index + active id, for the front's document picker. Lock-free: the
    index is written atomically (tmp + rename), so a reader always sees a whole file."""
    lib = await anyio.to_thread.run_sync(cag.load_library, DATA_DIR)
    return JSONResponse(content={
        "docs": [d.to_dict() for d in lib.docs], "active": lib.active,
    })


@app.post("/api/switch")
async def switch(request: Request) -> JSONResponse:
    """Make another library document active: restore its slot (seconds, no re-prefill) and
    load its history. Serialized on the slot lock — it mutates the single live slot."""
    payload = await request.json()
    doc_id = (payload or {}).get("id")
    if not isinstance(doc_id, str) or not doc_id:
        return JSONResponse(status_code=400,
                            content={"error": "empty", "message": "No document id."})
    async with _slot_lock:
        try:
            ok = await _activate(doc_id)
        except cag.LlamaError as e:
            return JSONResponse(status_code=503, content={"error": "llama_unreachable",
                                "message": f"restore failed: {e}"})
        if not ok:
            return JSONResponse(status_code=404, content={
                "error": "unknown_doc", "message": "No such document in the library."})
        payload_out = _status_payload()
    return JSONResponse(content=payload_out)


@app.delete("/api/library/{doc_id}")
async def delete_doc(doc_id: str) -> JSONResponse:
    """Remove a document (its dir + slot + history). Deleting the active one falls back to
    the most-recent remaining document, or to the no-document state if it was the last."""
    async with _slot_lock:
        lib = await anyio.to_thread.run_sync(cag.load_library, DATA_DIR)
        if lib.get(doc_id) is None:
            return JSONResponse(status_code=404, content={
                "error": "unknown_doc", "message": "No such document in the library."})
        was_active = lib.active == doc_id
        await anyio.to_thread.run_sync(cag.remove_doc, DATA_DIR, doc_id)
        lib.docs = [d for d in lib.docs if d.id != doc_id]
        if was_active:
            fallback = lib.most_recent()
            lib.active = fallback.id if fallback else None
            await anyio.to_thread.run_sync(cag.save_library, DATA_DIR, lib)
            state.docs_count = len(lib.docs)
            if fallback is not None:
                await _activate(fallback.id)  # restores its slot + reloads its history
            else:
                state.reset_session()
        else:
            await anyio.to_thread.run_sync(cag.save_library, DATA_DIR, lib)
            state.docs_count = len(lib.docs)
        payload_out = _status_payload()
    return JSONResponse(content=payload_out)


# --- content packs (export / import) + privacy purge -----------------------------------

@app.get("/api/pack/export")
async def pack_export() -> StreamingResponse:
    """Stream a tar of the whole library (index + docs + slots) for sneakernet / backup.

    Only the SMALL files (index + each doc's fiche/meta/history — KBs) are snapshotted under
    the slot lock; the lock is then released and the heavy ~300 MB slot files are streamed
    into the tar from their originals. A slot deleted between snapshot and read is skipped
    (the export stays a valid tar; that document re-prefills on first switch elsewhere), and
    /ask /switch are not blocked for the minutes the slot copy takes."""
    async with _slot_lock:
        small, slots = await anyio.to_thread.run_sync(cag.pack_manifest, DATA_DIR)
    # Lock released: assemble the tar (slow SD copy of the .bin slots) off the lock.
    spooled = await anyio.to_thread.run_sync(cag.build_pack_from_manifest, small, slots)

    def _iter():
        try:
            while True:
                chunk = spooled.read(256 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            spooled.close()

    return StreamingResponse(
        _iter(), media_type="application/x-tar",
        headers={"Content-Disposition": 'attachment; filename="cag-pack.tar"'})


@app.post("/api/pack/import")
async def pack_import(request: Request) -> JSONResponse:
    """Import a content pack (raw tar body): validate, merge into the library, and — if no
    document is currently active — activate the most-recent imported one.

    The upload is spooled to disk (TMPDIR=/data/tmp on the Pi, not tmpfs) and capped at
    PACK_MAX_BYTES so a hostile/oversized pack can't exhaust storage while being read; the
    extraction-time size guards live in cag.import_pack."""
    spooled = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, suffix=".tar")
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > PACK_MAX_BYTES:
            spooled.close()
            return JSONResponse(status_code=413, content={
                "error": "too_large",
                "message": (f"Pack exceeds the {PACK_MAX_BYTES}-byte upload limit. "
                            "Split it or raise PACK_MAX_BYTES.")})
        spooled.write(chunk)
    spooled.seek(0)
    async with _slot_lock:
        try:
            newest = await anyio.to_thread.run_sync(
                lambda: cag.import_pack(DATA_DIR, spooled, max_total_bytes=PACK_MAX_BYTES))
        except cag.PackTooLargeError as e:
            spooled.close()
            return JSONResponse(status_code=507,
                                content={"error": "pack_too_large", "message": str(e)})
        except cag.PackError as e:
            spooled.close()
            return JSONResponse(status_code=400,
                                content={"error": "bad_pack", "message": str(e)})
        except tarfile.TarError as e:
            spooled.close()
            return JSONResponse(status_code=400, content={
                "error": "bad_pack", "message": f"not a valid pack archive: {e}"})
        spooled.close()
        lib = await anyio.to_thread.run_sync(cag.load_library, DATA_DIR)
        state.docs_count = len(lib.docs)
        # Fresh volume (nothing loaded): activate the most-recent imported document.
        if not state.ingested:
            target = newest
            if target is None:
                mr = lib.most_recent()
                target = mr.id if mr else None
            if target is not None:
                await _activate(target)
        payload_out = _status_payload()
    return JSONResponse(content=payload_out)


@app.post("/api/wipe")
async def wipe() -> JSONResponse:
    """Privacy purge (US-12): delete ALL documents, slots, histories and the index, and
    reset the session to not-ingested."""
    async with _slot_lock:
        await anyio.to_thread.run_sync(cag.wipe_all, DATA_DIR)
        state.reset_session()
        state.docs_count = 0
    return JSONResponse(content={"ok": True})
