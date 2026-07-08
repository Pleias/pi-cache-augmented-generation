"""FastAPI app for the CAG example: static chat page + JSON/SSE API over llama-server.

One container runs llama-server (internal, 127.0.0.1:8080) and this app (published).
The app builds ChatML prompts by hand, drives the KV-cache slot save/restore, and
streams answers to the browser. All llama-server mechanics live in cag.py.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import cag

LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
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
        self.history: list[tuple[str, str]] = []
        # cached ChatML token count of each history pair, kept in lockstep with `history`
        # so trim_history budgets by summation instead of re-tokenizing the whole history
        # on every /ask (only each new pair is tokenized, once).
        self.pair_tokens: list[int] = []
        # last accurate history token count (sum of pair_tokens after the latest trim)
        self._history_tokens: int = 0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the shared HTTP client and attempt to restore a prior ingest on boot."""
    async with httpx.AsyncClient() as http:
        state.client = cag.LlamaClient(LLAMA_URL, http)
        state.model_id = await state.client.model_id()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            restored = await cag.restore_or_reprefill(
                state.client, DATA_DIR, state.model_id)
        except cag.LlamaError as e:
            # server not ready or restore/prefill failed hard: start empty, let the user
            # re-ingest. Logged as a warning so the operator can see why.
            print(f"[startup] restore failed, starting without ingest: {e}", flush=True)
            restored = None
        if restored is not None:
            self_fiche, self_tokens, ok = restored
            state.fiche = self_fiche
            state.fiche_tokens = self_tokens
            state.ingested = True
            state.restored = ok
            # The conversation is persisted separately from the slot file (see cag.py):
            # reload it so the UI can replay it and the next /ask keeps its context. Its
            # KV is not in the restored slot (fiche prefix only) — the first /ask
            # re-prefills the history tail, which is small.
            #
            # Only on a genuine restore, though: restored=False means the fiche was just
            # RE-prefilled (model changed / meta corrupt), so the persisted pair token
            # counts came from a different tokenizer — replaying them could overflow the
            # budget math. A changed model is a new conversation: drop it.
            if ok:
                state.history, state.pair_tokens = cag.load_history(DATA_DIR)
                state._history_tokens = sum(state.pair_tokens)
            else:
                await anyio.to_thread.run_sync(cag.clear_history, DATA_DIR)
                print("[startup] fiche re-prefilled (model changed or meta invalid); "
                      "dropping persisted chat history", flush=True)
            print(f"[startup] ingest loaded: {self_tokens} prefix tokens, "
                  f"restored={ok}, history turns={len(state.history)}", flush=True)
        else:
            print("[startup] no prior ingest found", flush=True)
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


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-file chat page (read once at import)."""
    return HTMLResponse(INDEX_HTML)


@app.post("/api/ingest")
async def ingest(request: Request) -> JSONResponse:
    """Tokenize the pasted document, enforce the cap, prefill + save its KV cache.

    On success the document text and metadata are persisted to the volume and the chat
    history is reset (a new document = a new conversation).
    """
    payload = await request.json()
    text = (payload or {}).get("text", "")
    if not isinstance(text, str) or not text.strip():
        return JSONResponse(status_code=400,
                            content={"error": "empty", "message": "No text to ingest."})

    client = _client()
    system_prefix = cag.build_system_prefix(text)
    # Serialize against /ask and other /ingest calls: prefill + save mutate the single slot.
    async with _slot_lock:
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

        try:
            await client.prefill_and_save(system_prefix)
        except cag.LlamaError as e:
            return JSONResponse(status_code=503, content={"error": "llama_unreachable",
                                "message": f"prefill failed: {e}"})

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Writes go through a worker thread: slow SD I/O must not block the event loop.
        await anyio.to_thread.run_sync(_write_text, DATA_DIR / "fiche.txt", text)
        await anyio.to_thread.run_sync(cag.save_meta, DATA_DIR, tokens, state.model_id)
        # a new document = a new conversation, in memory AND on disk
        await anyio.to_thread.run_sync(cag.clear_history, DATA_DIR)

        state.fiche = text
        state.fiche_tokens = tokens
        state.ingested = True
        state.restored = False  # freshly prefilled this session, not restored from disk
        state.history = []
        state.pair_tokens = []
        state._history_tokens = 0
    return JSONResponse(content={
        "ingested": True, "fiche_tokens": tokens,
        "budget": cag.history_budget(tokens), "model": state.model_id,
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
                            # fiche cache. Aborted answers never reach this point — they
                            # stay out of history, memory and disk alike.
                            await anyio.to_thread.run_sync(
                                cag.save_history, DATA_DIR,
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
        await anyio.to_thread.run_sync(cag.clear_history, DATA_DIR)
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
    return JSONResponse(content={
        "ingested": state.ingested,
        "fiche_tokens": state.fiche_tokens,
        "history_tokens": state._history_tokens,
        "budget": cag.history_budget(state.fiche_tokens) if state.ingested else 0,
        "model": state.model_id,
        "restored": state.restored,
    })
