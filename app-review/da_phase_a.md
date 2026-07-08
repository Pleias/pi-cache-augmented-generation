# DA Phase A Review — CAG App core (perf / deps / llama.cpp API)

Reviewer: devils-advocate (read-only). Target: Raspberry Pi 4 (Cortex-A72 ARMv8.0, 4-8GB,
slow SD I/O), Qwen2.5-1.5B Q4_K_M via llama-server, n_ctx 16384, prefill ~38 tok/s,
decode ~6 tok/s. Scope: app/server/cag.py, app/server/main.py, app/server/static/index.html,
app/pyproject.toml, app/uv.lock, spec 2026-07-07-cag-app-design.md.

API payloads/endpoints below were verified against llama.cpp `tools/server/README.md` via
context7 (`/ggml-org/llama.cpp`). Each remark is standalone for the Fixer.

Severity legend: CRITICAL = breaks core value / data loss; MAJOR = real defect, easy fix,
compounds on Pi; MINOR = polish / low-impact; CONFIRM = verified correct, no change (kept so
the Fixer does not "fix" a non-bug).

---

## PERFORMANCE (Pi 4)

1. [MAJOR] cag.py:208-226 (`trim_history`) + main.py:158 — **Every `/ask` re-tokenizes the
   ENTIRE history block over HTTP.** `trim_history` calls `client.tokenize(build_history_block(trimmed))`
   at least once per turn (more if it must drop pairs). As history grows toward the ~3,360-token
   budget this is an HTTP round-trip plus O(history) tokenizer work on the Pi on *every question*,
   all of it avoidable. Fix: cache the token count per `(user, assistant)` pair. ChatML turns are
   separated by hard special-token boundaries (`<|im_end|>\n<|im_start|>`), so per-pair counts are
   stable and additive — keep a running list of pair counts, tokenize only the newest pair after each
   answer, and sum for the budget check. This removes the per-turn tokenize hop entirely except for the
   one new pair.

2. [MAJOR] main.py:142-183 (`ask`) & main.py:89-139 (`ingest`) — **No server-side
   serialization; concurrent requests race a single shared slot + shared mutable state.**
   llama-server runs one slot (slot 0) and `state.history` is a process-global list. The only guard
   is the client-side `busy` flag (index.html:120), which does not protect against a second browser
   tab, a page reload mid-answer, or a double request. Two concurrent `/ask` calls will interleave
   reads/writes of `state.history` and issue two `/completion` calls against the same slot, producing
   a malformed prompt, a lost turn, or a slot save/restore colliding with an in-flight completion. Fix:
   add a module-level `asyncio.Lock` and acquire it around the whole `/ask` handler and around `/ingest`
   (they mutate the same slot + state). Cheap insurance for a shipped app users will open in two tabs.

3. [MAJOR] cag.py:108-116 (`model_id`) + cag.py:229-261 (`restore_or_reprefill`, line 252)
   — **`model_id()` returns `"unknown"` if `/props` is queried before the model finishes loading,
   which flips the model guard to "mismatch" and forces a full ~5-min re-prefill on every restart,
   defeating the headline CAG persistence feature.** `restore_or_reprefill` only takes the fast
   restore path when `meta.model == model_id` (line 252); a startup `model_id == "unknown"` never
   matches the real model id saved in meta.json, so the slot restore is skipped and the 12k-token
   document is re-prefilled from scratch. Two-part fix: (a) **Phase B entrypoint MUST gate uvicorn
   startup on `GET /health` returning `{"status":"ok"}`** (health contract verified: 200 `ok` when
   ready, 503 `{"error":{"message":"Loading model"...}}` while loading) so `model_id()` never runs
   against an unloaded server; (b) defensively, in `restore_or_reprefill` do not treat
   `model_id == "unknown"` as a definitive mismatch — either retry `/props` a few times at startup,
   or attempt the restore and only re-prefill if the restore itself fails. Without (a) the app's core
   promise ("restore in seconds, skip the 5-min prefill") silently never happens.

4. [MINOR] main.py:83-86 (`index`) — Synchronous `(STATIC_DIR/"index.html").read_text()` runs
   inside the async route on every page load, a blocking SD read on the event loop. Fix: read the HTML
   once at module import / lifespan startup into a string constant and return that.

5. [MINOR] main.py:126-128 (`ingest`) — Synchronous `fiche.txt` + `meta.json` writes inside the
   async handler block the event loop on slow SD. Low impact for a single-user app (no other request to
   starve), but if desired wrap in `anyio.to_thread.run_sync`. Acceptable as-is; noting for completeness.

6. [MINOR] cag.py:146-147 (save, `timeout=120`) & cag.py:154-159 (restore, `timeout=120`) —
   The slot file for a 12k-token Qwen2.5-1.5B context is on the order of ~300 MB (KV = 2·28 layers·2 KV
   heads·128 head_dim·2 bytes ≈ 28 KB/token). A cold, slow SD card writing/reading ~300 MB at
   10-20 MB/s approaches or exceeds 120 s. The `/completion` prefill call already uses `timeout=None`
   (correct); make save/restore match by raising these to `300` or `None` so a slow card cannot abort
   the save that the entire persistence feature depends on.

7. [CONFIRM] SSE streaming is correctly non-buffered — do not change. main.py:181-183 sets
   `X-Accel-Buffering: no` (defeats nginx buffering) and `Cache-Control: no-cache`; the async generator
   yields one `data:` frame per token so uvicorn flushes immediately (per-token gap at ~6 tok/s ≈ 166 ms,
   ample). `stream_completion` (cag.py:178) and `prefill_and_save` `/completion` (cag.py:142) both use
   `timeout=None`, so the ~5-min prefill and long streamed answers will not be killed by a client
   timeout. This is the right design; flagged so it is not "simplified" into buffering.

---

## DEPENDENCIES (small Pi package)

8. [MINOR] pyproject.toml:6-10 — **The app uses zero FastAPI-specific features, yet FastAPI drags
   in the only compiled dependency.** All routes use `await request.json()` and hand-built
   `JSONResponse`/`StreamingResponse`/`HTMLResponse`; there are no Pydantic request/response models, no
   dependency injection, no path-param validation, no OpenAPI requirement. FastAPI pulls `pydantic` +
   `pydantic-core`, and `pydantic-core` is the **only non-pure-python wheel in the whole lock** (a Rust
   ext). Dropping to bare **Starlette** (already an indirect dep, uv.lock:225) would remove
   pydantic / pydantic-core / typing-inspection / annotated-types, making the stack 100% pure-python
   (no arch-specific wheel at all) and smaller — a genuine win for a "small Pi package". Trade-off: it's
   a real (if small) refactor and FastAPI matches the bench stack. Recommendation: acceptable to keep
   FastAPI, but if image slimness matters, migrate `main.py` to Starlette routing. Decision for the
   orchestrator; not a blocker.

9. [CONFIRM] `python-multipart` correctly omitted — every endpoint consumes JSON via
   `request.json()`; no form or file-upload endpoint exists. FastAPI only needs python-multipart for
   form/multipart bodies. Do not add it.

10. [CONFIRM] Plain `uvicorn` (no `[standard]`) is the right call — keep. For a single-user,
    I/O-bound SSE workload, `uvloop`/`httptools` give negligible benefit and would add compiled-wheel
    surface on arm64. The pure-python asyncio loop + h11 parser is fine. The pyproject comment
    (lines 12-16) justifying this is accurate.

11. [CONFIRM] arm64 wheel availability is fine — verified in uv.lock: annotated-types, anyio, certifi,
    click, colorama, fastapi, h11, httpcore, httpx, idna, starlette, typing-extensions,
    typing-inspection, uvicorn are all `py3-none-any`. The sole compiled dep, `pydantic-core` 2.46.4,
    ships `manylinux_2_17_aarch64.manylinux2014_aarch64` **and** `musllinux_1_1_aarch64` wheels
    (uv.lock:160,167,175,182,190,197,205,212), so `uv sync` will not have to compile Rust on the Pi.

12. [MAJOR — Phase B] pyproject.toml:5 `requires-python = ">=3.12"` vs base image Python. The
    llama.cpp base image (`ghcr.io/ggml-org/llama.cpp:server`, Debian) ships Python **3.11**, not 3.12.
    The Phase B Dockerfile MUST provision Python 3.12 explicitly — cleanest is uv-managed
    (`uv python install 3.12`, python-build-standalone has aarch64 builds) rather than relying on the
    base image interpreter. Flagging so the builder does not assume `python3` in the base image
    satisfies the constraint.

13. [CONFIRM] Version pins are sane. pyproject ranges (fastapi >=0.115,<0.116; uvicorn >=0.34,<0.35;
    httpx >=0.28,<0.29) resolve to fastapi 0.115.14 / uvicorn 0.34.3 / httpx 0.28.1 and uv.lock pins
    exact versions + hashes. No action.

---

## API CORRECTNESS vs llama.cpp (context7-verified)

14. [CONFIRM] `POST /completion` payloads are correct — no change. Both the streaming body
    (cag.py:171-175: `prompt, cache_prompt, id_slot, stream:true, n_predict:512, stop, temperature`)
    and the prefill body (cag.py:138-140: same with `n_predict:1, stream:false`) use valid fields.
    Verified: in streaming mode `/completion` emits `content`, `tokens`, `stop` per SSE `data:` line and
    the final line carries `timings` — exactly what `stream_completion` reads (`content`, `stop`,
    `timings`, cag.py:190-194). Intermediate frames carry `stop:false` (falsy) and only the terminal
    frame `stop:true`, so `if j.get("stop")` is correct. `n_predict:1` with `cache_prompt:true` does
    prefill and cache the FULL prompt (the throwaway single token is fine); the subsequent save persists
    it. Correct CAG mechanic.

15. [CONFIRM] `POST /tokenize` is correct — cag.py:120-126 sends `{"content": text}` and reads
    `len(r.json().get("tokens", []))`. Verified: `/tokenize` accepts `content` and returns a `tokens`
    array (IDs by default; objects only when `with_pieces` is set, which is not). No change.

16. [CONFIRM] `POST /slots/0?action=save|restore` are correct — cag.py:144-147 & 154-159 send
    `{"filename": "slot0.bin"}`. Verified: filename is resolved relative to the server's
    `--slot-save-path`, and llama-server restricts it to a bare basename (no path separators) — "slot0.bin"
    complies. Save returns 200 `{n_saved,n_written,timings}`; restore returns 200
    `{n_restored,n_read,timings}` or an error on missing/corrupt file, which `restore()` correctly maps to
    `False` via `raise_for_status` → `except httpx.HTTPError` (cag.py:161-162). Restore reloads the token
    sequence into the slot, so the next `/completion` with the same fiche prefix + `cache_prompt:true`
    re-uses it — which is exactly what `build_prompt` sends (fiche prefix first). Correct.

17. [MINOR] cag.py:105 (cap) — token count used for the 12,000 cap comes from `/tokenize`, which by
    default does NOT add special/BOS tokens, whereas `/completion` tokenizes the prompt per the server's
    default. This can make the counted value differ from the actually-prefilled token count by the
    special-token handling. For Qwen2.5 (no BOS by default) this is ~0-1 tokens against a 12,000 cap and
    16,384 context — negligible. Informational only; no fix needed, do not over-engineer it.

18. [MINOR] cag.py:113 (`model_id`) — reads `r.json().get("model_path")` from `/props`. Verify the
    field name against the pinned build (recent llama.cpp `/props` exposes `model_path`; some versions
    nest model info differently). Note it degrades gracefully: if the field is wrong, `model_id` is
    `"unknown"` on both ingest and restart, so meta and runtime agree and restore still proceeds — only
    the model-mismatch guard and the UI model tag (index.html:143) silently no-op. Low risk; worth a
    one-line check during Phase B smoke.

19. [MAJOR — Phase B] Entrypoint llama-server flags. Save/restore only work when the server is
    launched with **`--slot-save-path /data/cache`** (verified: both actions resolve the file under that
    path; without it they error and CAG persistence is dead). The `--slots`/`--no-slots` flag toggles the
    slots *monitoring* GET endpoint (verified in `common/arg.cpp`, env `LLAMA_ARG_ENDPOINT_SLOTS`); the
    POST save/restore actions are gated on `--slot-save-path` rather than `--slots`, but recommend the
    entrypoint pass **both** `--slot-save-path /data/cache` and `--slots` to be safe, plus `-c 16384`.
    Builder must confirm the POST actions respond (not 404) on the pinned build during first smoke.

---

## BUILDER-FLAGGED DEVIATIONS (assessed)

20. [CONFIRM/keep] Language-neutral system prompt (cag.py:36-41) — APPROVE for this app. The image
    ships zero data and ingests arbitrary user text in any language/domain, so a French-specific prompt
    would be wrong. The `"Reply in the same language as the question."` clause makes Qwen2.5 answer a
    French document/question in French. Quality risk is low. Only action: include one French Q&A in the
    Phase B smoke to confirm grounded French output. Do not hard-code French.

21. [CONFIRM/keep, MINOR doc] Cap counted on the FULL system prefix, not raw fiche text
    (cag.py:103-105 vs spec line 91) — **Keep the prefix-based count; it is the correct choice**, not a
    bug. `history_budget = N_CTX − fiche_tokens − GEN_RESERVE` (cag.py:205) subtracts the same
    prefix-token count that the cap checks, so cap and budget stay consistent; switching the cap to raw
    fiche text would desync them. The ChatML wrapper + instruction overhead is ~40-50 tokens against
    12,000 — negligible. Only fix: reconcile the wording — the UI (index.html:90 "Cap: 12,000 tokens")
    and spec (line 91 "fiche text cap 12,000") should say the cap is on the document as prefilled
    (document + small instruction wrapper), or simply note the wrapper overhead. Documentation nit, not
    a code change.

22. [NOTE — Phase B] Startup restore assumes the entrypoint gates on `/health` — CONFIRMED
    REQUIRED; see remark #3. Without the gate, `model_id` is `"unknown"` and every restart re-prefills.

23. [NOTE — Phase B] `.dockerignore` must exclude `.venv` (and `__pycache__`, `*.pyc`, `.git`,
    `data/`) so the local virtualenv is not copied into the build context / image layer. Phase B item.

---

## RESOLVED ITEMS (for Phase B builder)

**Docker base image tag.** Latest llama.cpp server build at review time: **b9894** (accelerated
variants `server-cuda12-b9894`, `server-intel-b9894`, `server-rocm-b9894` confirmed published ~9 h ago
on ghcr). The plain **CPU** multi-arch image is documented (docs/docker.md) as
`ghcr.io/ggml-org/llama.cpp:server` — platforms **linux/amd64, linux/arm64, linux/s390x** (CPU-only,
exactly what the Pi needs). Caveat: docs/docker.md only documents the rolling `:server` tag; a
build-pinned plain-CPU `:server-b9894` is very likely published by CI (it tags every build) but I could
not positively confirm the plain-CPU build-number tag from the docs alone.
**Recommendation for the builder:**
`ghcr.io/ggml-org/llama.cpp:server-b9894`, and verify it resolves multi-arch with
`docker manifest inspect ghcr.io/ggml-org/llama.cpp:server-b9894`; **if that tag 404s, pin the rolling
`:server` tag by digest** (`ghcr.io/ggml-org/llama.cpp:server@sha256:<digest>`) for reproducibility
rather than leaving it floating. Pick the exact latest `bNNNN` at build time.

**Hugging Face GGUF URL.** Repo `Qwen/Qwen2.5-1.5B-Instruct-GGUF` confirmed; Q4_K_M is a **single
file** (1.12 GB, not split):
`https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf`
