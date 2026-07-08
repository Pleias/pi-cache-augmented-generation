# DA Phase B Review — CAG App packaging (Docker / entrypoint / smoke / tests)

Reviewer: devils-advocate (read-only). Scope: app/Dockerfile, app/.dockerignore,
app/entrypoint.sh, app/smoke.sh, app/tests/test_cag.py (current cag.py/main.py read to judge
whether tests assert real behavior). Target: Raspberry Pi 4 (Cortex-A72 ARMv8.0, no
dotprod/i8mm, 4-8GB, slow SD), Qwen2.5-1.5B Q4_K_M, n_ctx 16384. Base image entrypoint/binary
path verified against llama.cpp `docs/docker.md` via context7/WebFetch.

Severity: MAJOR = real defect / must-verify blocker; MINOR = polish or low-impact hardening;
CONFIRM = verified correct, do not change (kept so the Fixer doesn't "fix" a non-bug).

---

## IMAGE SIZE / LAYERS

1. [MINOR] Dockerfile:14 (`FROM debian:bookworm-slim AS model`) — the model-download stage
   runs **once per target platform** under `buildx --platform linux/amd64,linux/arm64`, so the
   identical arch-independent ~1.12 GB GGUF is downloaded **twice** (and under QEMU for the arm64
   pass). Fix: pin the download stage to the build host arch —
   `FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS model` — so it downloads once natively
   and the single file is `COPY --from=model`'d into both arch images. Saves ~1.1 GB of transfer
   and an emulated curl run per multi-arch build. (Image size is unaffected either way; this is
   build time/bandwidth.)

2. [CONFIRM] Multi-stage isolation is correct — curl + ca-certificates live only in the
   discarded `model` stage; the final image gets just `/models/model.gguf` via
   `COPY --from=model` (Dockerfile:54), appearing in exactly one layer. Good.

3. [CONFIRM] No leftover uv cache and sane venv/interpreter placement (Dockerfile:34-46) —
   `uv sync` and `rm -rf /root/.cache` are chained in one RUN so the wheel cache never persists in
   a layer; `UV_PROJECT_ENVIRONMENT=/opt/venv` and `UV_PYTHON_INSTALL_DIR=/opt/uv/python` are both
   outside `/app` (WORKDIR) and outside `/data` (the mounted VOLUME), so neither is shadowed by the
   user's volume at runtime. `UV_COMPILE_BYTECODE=1` is a fine size/startup trade. Good.

4. [CONFIRM] Layer cache order is correct — `COPY pyproject.toml uv.lock` + `uv sync` precede
   `COPY server` (Dockerfile:43-49), so app-code edits don't invalidate the expensive dependency
   layer. Good.

5. [MINOR/optional] No container-level `HEALTHCHECK`. The entrypoint gates the app on
   llama-server's internal `/health`, but Docker itself has no readiness signal, so `docker ps`
   and any orchestration can't tell when `:8000` is serving. Optional: add
   `HEALTHCHECK --start-period=600s --interval=30s --timeout=5s CMD <stdlib python hitting
   http://127.0.0.1:8000/api/status>`. (The base image's own healthcheck, if inherited, targets
   llama on :8080, not the published app.)

---

## ARM64 CORRECTNESS

6. [MAJOR — verify at build time] Dockerfile:11 & 26 (`LLAMA_TAG=server-b9894`) — **the pinned
   base tag must be a multi-arch manifest that includes `linux/arm64`.** Phase A could only confirm
   the *rolling* `:server` tag is multi-arch (amd64/arm64/s390x); the build-number-pinned
   *plain-CPU* `server-b9894` may not exist or may be published only for some arches. If it is
   amd64-only (or missing), `buildx --platform ...,linux/arm64` fails or silently pulls the wrong
   arch. **Before building, run `docker manifest inspect ghcr.io/ggml-org/llama.cpp:server-b9894`
   and confirm an `linux/arm64` entry**; if absent, switch to the digest-pinned `:server` form the
   Dockerfile already documents (lines 7-10). This is the single highest packaging risk.

7. [CONFIRM] Every other stage/tool is genuinely multi-arch — `debian:bookworm-slim`,
   `ghcr.io/astral-sh/uv:${UV_VERSION}` (the `COPY --from=image` resolves to the target platform's
   uv binary), python-build-standalone 3.12 (has aarch64 glibc builds, compatible with the Debian
   base's glibc 2.36), and pydantic-core 2.46.4 (manylinux2014_aarch64 wheel confirmed in uv.lock,
   Phase A). `uv sync --frozen` runs inside the emulated arm64 rootfs, so it resolves aarch64
   wheels from the running interpreter — no cross-compile hazard. No amd64-only assumption found.

---

## LLAMA-SERVER FLAGS / Pi 4 RAM

8. [CONFIRM] Flag set is correct for Pi 4 (entrypoint.sh:42-48) — no `-fa` (correct: flash-attn
   is a no-op/negative on ARM CPU per the bench), `--ctx-size 16384`, `--threads $(nproc)` (= 4 on
   Pi 4, matching the 4 physical A72 cores), `--slot-save-path "$CACHE_DIR"` (enables the POST
   save/restore actions the whole feature needs) and `--slots` (GET monitor). Bound to
   127.0.0.1:8080, internal-only. Good.

9. [MINOR/informational] RAM headroom on a 4 GB Pi — note the KV cache is allocated for the
   **full `--ctx-size` (16384)**, not the 12k document cap: f16 KV ≈ 2·28 layers·2 KV heads·128
   head_dim·2 B ≈ 28 KB/token × 16384 ≈ **~459 MB** (higher than the ~340 MB the brief estimated at
   12k). Peak RSS ≈ model ~1.12 GB (mmap) + ~0.46 GB KV + compute/buffers + app/OS ≈ **~2 GB** —
   fits a 4 GB Pi with headroom, comfortable on 8 GB. Recommend documenting the 4 GB floor and
   enabling swap. Optional lever if RAM-tight: `--cache-type-k q8_0 --cache-type-v q8_0` halves KV,
   but **keep f16 default** (matches the bench and avoids any slot-format surprise on save/restore).

10. [MINOR] entrypoint.sh:13 (`THREADS="${LLAMA_THREADS:-$(nproc)}"`) — under `set -e`, if
    `nproc` is somehow absent the failing command substitution can abort the entrypoint. Harden:
    `$(nproc 2>/dev/null || echo 4)`. (Debian base almost certainly ships coreutils, so low
    likelihood — but cheap.)

---

## ENTRYPOINT ROBUSTNESS

11. [CONFIRM] Health gate correctly implements DA Phase A #3 (entrypoint.sh:64-91) — it blocks
    uvicorn startup until llama-server `/health` returns `{"status":"ok"}`, polling with a stdlib
    `urllib` one-liner (smart: no dependency on curl being in the base image), re-checking the
    llama PID is still alive each iteration (`os.kill(pid,0)`), with a 600 s deadline and a
    non-zero exit + llama teardown on timeout. This is exactly what stops `model_id()` from seeing
    "unknown" on a still-loading server. Good.

12. [CONFIRM] Binary autodetect is adequate (entrypoint.sh:25-35) — docs confirm the server
    image's binary is at `/app/llama-server` (docker.md invokes `:server -m ... --host ...` directly
    against the entrypoint; light image references `/app/llama-cli`). The detection tries
    `command -v llama-server` then a hardcoded list that **includes `/app/llama-server`**, and
    FATALs loudly if none is executable. Comprehensive enough for the pinned image.

13. [MINOR] Signal handling during startup (entrypoint.sh:59-85) — the `trap shutdown TERM INT`
    is set, but the `/health` wait runs as a **foreground** child, and bash defers a trap until the
    foreground command returns. So a `docker stop` *during* the (potentially minutes-long) model
    load is deferred until health resolves or the 600 s timeout, after which Docker's grace elapses
    and PID 1 is SIGKILLed (namespace teardown still reaps llama-server — no leak, just not a
    graceful stop). Optional hardening: run the health-wait in the background and `wait` on it so the
    trap stays responsive mid-load.

14. [CONFIRM] No half-dead container (entrypoint.sh:100-102) — `wait -n "$LLAMA_PID"
    "$UVICORN_PID"` returns as soon as either process dies and then calls `shutdown` to tear down
    the peer, so a llama-server crash stops the whole container instead of leaving a zombie app.
    `shutdown` kills uvicorn then llama and exits 0. Good.

---

## SMOKE + TESTS

15. [MINOR] smoke.sh:58-60 (step 6) prints `/api/status` but does **not assert** `restored:true`
    — the headline persistence guarantee is only eyeballed, so a silent restore regression
    (falling back to a 5-min re-prefill) won't fail the script. Fix: capture the status body and
    `grep -q '"restored": *true'` (exit non-zero if missing) so a broken restore fails loudly. Same
    idea would strengthen step 3/4 (assert the answer contains "19"/"14" to prove grounding rather
    than just streaming something).

16. [CONFIRM] The DA #3b behavior is genuinely asserted — NOT vacuous.
    `test_unknown_runtime_model_id_still_restores_no_reprefill` (test_cag.py:143) seeds
    `meta.model="qwen.gguf"`, calls `restore_or_reprefill(..., model_id="unknown")`, and asserts
    `restore_calls==1 / prefill_calls==0 / restored=True`. Regressing cag.py:274-280 back to a strict
    `meta.model == model_id` guard would skip the restore and flip `prefill_calls` to 1 — the test
    would fail. And `test_positive_model_mismatch_reprefills_without_restore` (line 163) pins the
    complement (both ids known and different ⇒ still re-prefills, `restore_calls==0`), so the
    "unknown" leniency cannot silently disable the real model guard. Strong, non-tautological pair.

17. [CONFIRM] trim_history tests exercise the DA #1 cached-count fix (test_cag.py:95-121) — pure
    sync signature `(fiche_tokens, history, pair_tokens) -> (kept, kept_counts, total)`, drop-oldest
    -until-under-budget math, within-budget keep-all, empty, and negative-budget-drops-everything.
    These match the reimplemented `cag.trim_history` and genuinely guard the no-re-tokenize path.
    Good.

18. [MINOR] test_cag.py:81-86 (`test_render_pair_counts_are_additive`) is **structurally
    tautological**: it counts `"<|im_start|>"` substring occurrences, which are trivially additive
    across string concatenation, so it does NOT validate the property the #1 fix actually depends on
    — that the **real tokenizer** does not merge tokens across the ChatML seam
    (`<|im_end|>\n<|im_start|>`). The assumption is in fact sound (Qwen's `<|im_start|>`/`<|im_end|>`
    are atomic special tokens, so BPE can't merge across them), but only the **live smoke against
    the real tokenizer** proves it. Recommend: keep the unit test but note its limitation, and make
    smoke.sh drive a **multi-turn** exchange (several asks) so a real-tokenizer additivity drift
    would surface as a budget/`history_tokens` miscount.

19. [MINOR] Two runtime-path coverage gaps (unit-testable without llama):
    (a) `_pair_token_count` fallback (main.py:110-114) — when `client.tokenize` raises `LlamaError`
    it estimates `len/3+1` to keep `pair_tokens` in lockstep with `history`; this desync-prevention
    path is untested. Add a test with a raising `tokenize` asserting the pair is still appended and
    the two lists stay equal length.
    (b) The `/api/ask` streaming happy path (main.py:195-219) — `test_ask_before_ingest` returns
    early on the 409 guard, so the event_stream token loop, the history/`pair_tokens` append, and
    `_history_tokens` update are **never exercised** in unit tests (only smoke). Add a test that
    monkeypatches `state.client.stream_completion` to yield a couple of `("token", …)` then
    `("done", {...})`, drives `event_stream`, and asserts `len(state.history)` and
    `len(state.pair_tokens)` each grew by one and `_history_tokens` increased. This is the main
    runtime path and currently has zero unit coverage.

---

## BUILDER'S 4 UNVERIFIABLE-WITHOUT-A-PI ITEMS — ranked for live smoke

Do the one build-time check FIRST, then the Pi runtime checks in this order (x86 smoke first to
prove app logic, then Pi):

- **[BUILD-TIME, before any run] Does `server-b9894` include linux/arm64?** — `docker manifest
  inspect` / a `buildx --platform linux/arm64` dry pull. (= remark #6.) If it fails here, nothing
  downstream can run; fall back to the digest-pinned `:server`.
- **[Pi #1, gating] Does the official arm64 llama.cpp build actually RUN on Pi 4 (ARMv8.0, no
  dotprod/i8mm)?** — start the container, watch for illegal-instruction/crash or absurdly slow
  prefill. If it crashes or underperforms, activate the documented `-DGGML_CPU_ARM_ARCH=armv8-a`
  source-build fallback stage. Everything else is moot until this passes.
- **[Pi #2] Does KV slot save/restore round-trip on the pinned build?** — the headline feature;
  smoke.sh steps 5-7 (ingest → `docker restart` → `status.restored==true` → answer with no ~5-min
  re-prefill). Watch that POST `/slots/0?action=save|restore` return 200 on this exact build.
- **[Pi #3] Does it fit 4 GB RAM at ctx 16384 under a real ~12k ingest?** — run a near-cap ingest
  and watch for an OOM-kill during prefill (peak ~2 GB expected; see remark #9).

---

## COHERENCE WITH PHASE A (cross-check)

The Phase B artifacts correctly absorbed the Phase A fixes: health-gate entrypoint (A#3),
`--slot-save-path` + `--slots` (A#19), uv-installed Python 3.12 vs the base image's 3.11 (A#12),
the `server-b9894` tag + digest-pin fallback comment and the exact Q4_K_M HF URL (A resolved
items), save/restore timeouts raised to `None` (A#6), and the asyncio slot lock + cached per-pair
counts + read-HTML-once + threaded writes are all present in the current cag.py/main.py (A#1, #2,
#4, #5). No contradiction between the two phases found.
