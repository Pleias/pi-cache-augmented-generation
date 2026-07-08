# CLAUDE.md — pi-cache-augmented-generation

CAG (Cache-Augmented Generation) benchmark + packaged demo app for small LLMs on a
Raspberry Pi 4. Two halves:

- `bench/` — the benchmark runner (`cag_bench.py`, modes `nocontext` / `naive` / `cag`).
  `naive` **is** the RAG baseline (context re-prefilled per query); `cag` reuses the KV
  prefix. Findings live in `docs/TECH_REPORT.md`.
- `app/` — the shippable demo: one Docker image (amd64+arm64), llama-server + FastAPI +
  static chat UI, Qwen2.5-1.5B-Instruct Q4_K_M baked in, zero content data. Design spec:
  `docs/superpowers/specs/2026-07-07-cag-app-design.md`.

## Rules for working here

- **Never build/run/manage Docker containers from tools.** Provide the commands; the user
  runs them and pastes output. (Container logs flood context.)
- **No git commits/pushes/tags unless the user explicitly asks in the current turn.**
- The repo is **private**; the GHCR package `ghcr.io/pleias/cag-app` is **public**. Public
  users cannot see the repo README or GitHub Releases — all public-facing docs must live in
  the image description annotation and `/CONTAINER.md` baked into the image (also served
  in-app at `GET /api/doc`).

## Commands

```bash
# app tests (mocked llama — no server needed)
cd app && uv run --with pytest pytest tests/ -q

# app live smoke against a running container (user runs)
CONTAINER=cag ./app/smoke.sh

# benchmark (needs llama-server up; see bench/run_server.sh)
uv run bench/cag_bench.py --tokens-only

# release: push tag v* → CI builds multi-arch image → GHCR + GitHub Release
```

## Architecture invariants (violating these breaks measured behavior)

- **Context-first ChatML, fiche in the system message** — the only prompt shape where CAG
  grounding works (query-first models like Baguettotron/PleIAs-RAG cannot do CAG; see
  TECH_REPORT F1). Prompts are built **manually** in `app/server/cag.py` (not `--jinja`)
  so tokens are deterministic for slot save/restore prefix matching.
- **`--parallel 1` on llama-server is mandatory** for single-user CAG. Recent builds
  default to auto = 4 slots → 4× unified KV → CPU decode collapses ~30× (measured
  0.36 tok/s). Set in `app/entrypoint.sh`.
- **Layered persistence in `/data`:** `cache/slot0.bin` = document KV only, saved once at
  ingest (never per-turn — ~300 MB SD writes); `fiche.txt` + `meta.json` = document;
  `history.json` = conversation (atomic write per completed turn, cached per-pair token
  counts). Restart: restore slot → replay history in UI → first ask re-prefills only the
  history tail. Model change ⇒ drop history (stale tokenizer counts).
- **Single slot 0 + `asyncio.Lock`**: every MUTATING endpoint (/ask, /ingest, /reset,
  /switch, delete, pack import, wipe) takes `_slot_lock`. Read-only endpoints
  (/status, /library, /history) are deliberately lock-free — state files are written
  atomically (tmp+rename), and blocking the doc picker behind a 5-min prefill is worse.
  Pack export takes the lock only to snapshot, then streams lock-free.
- History cap: budget = `16384 − fiche_tokens − 1024`; drop oldest pairs. Fiche cap
  12,000 tokens, enforced on the full system prefix, reject with token count.
- Front (`app/server/static/index.html`) is a single file, no external assets; its
  markdown renderer is escape-first **including quotes** (XSS via crafted link — DA
  finding, don't regress).

## Perf facts to not re-derive (Pi 4, measured)

- Prefill is compute-bound (~44–53 tok/s @1.5B Q4); decode is memory-bandwidth-bound
  (~12 tok/s @Q4, 6 @Q8 — weight quant halves bytes → doubles decode).
- Flash-attention / KV-quant tuning does **not** help decode on Pi CPU (tested, issue #1).
- Decode falls with cached context size (1.2 tok/s @21k ctx @1.5B Q8) — big corpora are
  batch-only; the interactive unit is one lesson (~2–3k tok).
- Numbers live in `docs/throughput.md`; don't quote the LinkedIn drafts' numbers.

## Doc map

**README split:** root `README.md` is the PUBLIC variant — explanation + app only, no
internal links, no repo structure, no benchmark instructions. The full internal version
(structure, bench quickstart, doc links) is `docs/README_internal.md` — update BOTH when
the app or findings change.

`docs/TECH_REPORT.md` (primary, §8 = app), `docs/USE_CASES.md` (CAG-vs-RAG argument),
`docs/throughput.md` (tok/s tables + gotchas), `docs/results/results_*.md` (per-experiment),
`app/CONTAINER.md` (public container doc — single source, three delivery paths: GHCR
description, `docker run --entrypoint cat`, in-app Docs button).
