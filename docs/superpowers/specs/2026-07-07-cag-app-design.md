# CAG Example App — Design

**Date:** 2026-07-07
**Status:** approved (brainstorm session)
**Goal:** shippable example app demonstrating Cache-Augmented Generation on low-power
devices. Docker image for x86_64 + arm64 (Raspberry Pi 4 = floor target), Qwen2.5-1.5B
Q4 baked in, **no data in the image** — user ingests their own text, which is persisted
as a llama.cpp KV-cache slot file and restored on restart.

## Requirements

- Ships with model, ships with **zero content data** (bench fiches were only for testing;
  cannot be redistributed).
- User ingests own text by **pasting into the web page**.
- Ingest cap: **12,000 tokens**. Over cap → **reject** with actual token count and a
  clear explanation (no silent truncation).
- Cache saved to a **user-chosen location** (volume mount). On container restart, if a
  previous ingest exists there, the app **picks it up** (slot restore) and skips
  re-prefill (~5 min for 12k tok on Pi 4 at ~38 tok/s prefill → seconds).
- Interface: **HTTP API + minimal static web page**, answers **streamed** (SSE).
- **Multi-turn chat** with a cap on context growth (see History cap).
- Two arch targets: `linux/amd64` and `linux/arm64` — arch-native llama-server in each.

### Threat model (agreed)

"Not exposing data" = the app distributes nothing; the user's data lives only in their
own volume on their own device and never leaves it. The volume stores the fiche text in
plaintext alongside the cache (required: each prompt must contain the fiche text for
llama.cpp prefix-cache token matching). KV cache is **not** encryption; a user with
volume access or chat access can recover content. That is acceptable under this model.

## Architecture (Approach 1 — single container, two processes)

```
┌────────────────────── container ──────────────────────┐
│ entrypoint.sh                                          │
│  ├─ llama-server  127.0.0.1:8080 (internal only)       │
│  │    -m qwen2.5-1.5b-instruct-q4_k_m.gguf             │
│  │    -c 16384  --slot-save-path /data/cache           │
│  └─ uvicorn app  0.0.0.0:8000 (published)              │
│       FastAPI: static page, /api/*, SSE proxy          │
└────────────────────────────────────────────────────────┘
   volume:  -v <user dir>:/data
   /data/fiche.txt    — ingested text (user's own device)
   /data/meta.json    — fiche token count, model id, prompt hash
   /data/cache/slot0.bin — saved KV cache
```

Rejected alternatives: two-container compose (heavier on Pi, no benefit);
llama-cpp-python in-process (arm64 wheel pain, diverges from benchmarked llama-server).

## Image & build

- One Dockerfile; base = official **`ghcr.io/ggml-org/llama.cpp:server-b####`** pinned
  to an exact release build number chosen at implementation time. Official image is
  multi-arch (linux/amd64 + linux/arm64, CPU-only) — verified in llama.cpp docs
  (`docs/docker.md`). `docker buildx build --platform linux/amd64,linux/arm64` produces
  a manifest with arch-native llama-server per platform.
- Model: **Qwen2.5-1.5B-Instruct Q4_K_M** GGUF (~1.0 GB) downloaded in a build stage
  from Hugging Face (`.../resolve/main/` URL) and baked into the image. Final image
  ≈ 1.3 GB.
- App layer: Python 3.12, FastAPI + uvicorn + httpx, dependencies locked with uv
  (`pyproject.toml` + `uv.lock`, same stack as `bench/`). Static page bundled.
- **Pi 4 fallback (documented, not default):** Cortex-A72 is ARMv8.0 (no dotprod/i8mm).
  If the official arm64 build crashes or underperforms on Pi 4, alternative Dockerfile
  stage builds llama.cpp from source with `-DGGML_CPU_ARM_ARCH=armv8-a` (flag verified
  in `ggml/src/ggml-cpu/CMakeLists.txt`).
- Per project rules: Claude provides docker commands; **user runs them**.

## Prompting & cache mechanics

- App builds the **raw ChatML prompt manually** — fiche inside the system message, the
  format validated by the benchmark (`CHATML_SYS_CTX` in `bench/cag_bench.py`).
- Requests go to llama-server **`/completion`** (not `/v1/chat/completions`) with
  `cache_prompt: true`, `id_slot: 0`, explicit stop tokens (`<|im_end|>`,
  `<|im_start|>`). Manual template = deterministic tokens = reliable prefix match for
  slot save/restore.
- Ingest prefill: `/completion` with the system-prefix prompt and `n_predict: 1`
  (minimal generation just to populate the KV cache), then
  `POST /slots/0?action=save {"filename": "slot0.bin"}` (endpoint verified in
  `tools/server/README.md`).
- Startup restore: if `/data/meta.json` exists and matches the baked model id →
  `POST /slots/0?action=restore`. On failure (corrupt file, model mismatch) → log
  warning, fall back to re-prefill from `/data/fiche.txt`, then re-save.

## API

| Route | Method | Behavior |
|---|---|---|
| `/` | GET | static HTML/JS chat page |
| `/api/ingest` | POST `{text}` | `/tokenize` count; **>12,000 → 400** `{tokens, cap, message}`; else prefill → slot save → write `fiche.txt` + `meta.json`; resets history |
| `/api/ask` | POST `{question}` | SSE stream of answer tokens; appends turn to history |
| `/api/reset` | POST | clear chat history (cache + fiche kept) |
| `/api/status` | GET | `{ingested, fiche_tokens, history_tokens, budget, model, restored}` |

Errors: llama-server unreachable → 503; ask before ingest → 409 with hint.

## History cap (multi-turn growth)

- `n_ctx = 16384`. Budget: `history_budget = 16384 − fiche_tokens − 1024` (generation
  reserve).
- History = list of (user, assistant) turn pairs, token-counted via `/tokenize`.
- Over budget → **drop oldest pairs** until under. Fiche prefix (the expensive part)
  stays cached; only the shifted history tail re-prefills (small).
- `/api/status` exposes history usage so the page can show a fill indicator.

## Web page

Single static HTML + vanilla JS (no framework, no external assets):
- No ingest yet → ingest view: textarea + "Ingest" button; over-cap error shown with
  token count.
- Ingested → chat view: message list, input box, streamed assistant tokens rendered
  live (fetch + ReadableStream over SSE), status bar (fiche tokens, history fill),
  "Reset chat" and "Re-ingest" buttons.

## Repo layout

```
app/
  Dockerfile
  entrypoint.sh
  pyproject.toml / uv.lock
  server/            # FastAPI app
    main.py          # routes, SSE
    cag.py           # prompt build, tokenize, prefill, slot save/restore, history cap
    static/index.html
  tests/
    test_cag.py      # cap check, history trim, prompt build — mocked llama-server
  smoke.sh           # curl sequence vs running container (user runs)
```

## Testing

1. **Unit (pytest, no llama-server):** ingest cap rejection message, history trimming
   math, ChatML prompt construction, meta/model mismatch fallback logic — httpx mocked.
2. **Live smoke (user runs, x86 first, then Pi 4):** build → run → ingest sample text →
   ask (verify streaming + grounded answer) → `docker restart` → `/api/status` shows
   `restored: true` → ask again without re-prefill delay.

## Open items for implementation plan

- Pick exact llama.cpp release tag (`server-b####`) — latest stable at impl time.
- Confirm official arm64 image runs on Pi 4 (first live smoke); else activate
  `GGML_CPU_ARM_ARCH=armv8-a` fallback stage.
- Exact Hugging Face URL for Qwen2.5-1.5B-Instruct Q4_K_M GGUF.
