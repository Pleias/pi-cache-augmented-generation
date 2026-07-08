# Pi Cache-Augmented Generation — internal README

> **Internal (private-repo) version.** The root `README.md` is the public-facing variant:
> explanation + the CAG app only, no repo internals. This file keeps everything —
> repo structure, benchmark quickstart, and doc links. Note: links here are relative to
> the repo root (this file lives in `docs/`), prefix with `../` when navigating locally.

A benchmark and technical study of **Cache-Augmented Generation (CAG)** with small (<2B
parameter) LLMs on a Raspberry Pi, grounding answers on teacher lesson "fiches" for offline
education.

## TL;DR findings

- **CAG removes 20–76× of prefill time**, and the saving *grows* with context size: 20× at
  1.9k cached tokens, up to 65–76× at 21k tokens.
- **CAG needs context-first models.** It requires the context to sit *before* the question as
  a stable, reusable prefix. The PleIAs RAG family (Baguettotron, Pleias-RAG-350M/1B) is
  trained **query-first** (sources after the question) — a deliberate RAG-optimized design,
  not built for CAG. Forcing the context first to enable caching breaks their grounding.
- **Decode is the residual wall.** CAG doesn't cache decode, and decode throughput is
  memory-bandwidth-bound: it falls from ~15–20 tok/s at small context down to **~2.9 tok/s
  (0.5B) / ~1.2 tok/s (1.5B)** over a 21k-token cache. So total wall-clock speedups
  (1.4–6.8×) trail the prefill speedups (20–76×).
- **Qwen2.5-1.5B-Instruct is the sweet spot**: it fixes the 0.5B's reasoning failures
  (vrai-faux negation, corrigé justification) and improves 7/8 cross-fiche questions, for
  ~1.6–2× the wall-clock and 2.3× the cache size.
- **KV cache size is architecture-driven, not parameter-count-driven.** Fewer KV heads (GQA)
  and moderate depth win: the *smaller* Baguettotron (321M, 80 layers) has a 5× bigger
  per-token cache than the *larger* Qwen2.5-0.5B (24 layers); KV quantization (`q8_0`) halves
  it further, independent of weight quantization.

Full detail, methodology, and caveats: [`docs/TECH_REPORT.md`](docs/TECH_REPORT.md).

## The example app — CAG in a box

The findings are packaged as a runnable app in [`app/`](app/): a single Docker image
(x86_64 + arm64, **Raspberry Pi 4 is the floor target**) bundling `llama-server` +
**Qwen2.5-1.5B-Instruct Q4_K_M** + a FastAPI backend + a single-file chat UI. It ships
**zero content data** — you paste your own document (≤12,000 tokens; over-cap is rejected
with the actual token count), the app prefills the KV cache **once**, saves it to your
mounted volume, and **restores it on restart** (no re-prefill). Multi-turn chat streams
over the cached prefix with a capped history budget; the **conversation itself persists
too** (per-doc `history.json` — replayed in the UI after a restart, its KV re-prefilled
as a small tail on the restored prefix). Since v0.2: a **document library** (≤20 docs,
per-doc slot files under `cache/<id>.bin`, switch = restore in seconds, idempotent legacy
migration), **guided-mode preset chips**, **content packs** (validated tar export/import
+ wipe; `PACK_MAX_BYTES` guard, TMPDIR pinned to `/data/tmp`), and an **EN/FR/AR UI**
(RTL; FR default — see `docs/I18N_PLAN.md` for why mother tongues aren't viable at 1.5B).
Container usage doc ([`app/CONTAINER.md`](app/CONTAINER.md)) is baked into the image and
served in-app via the header's **Docs** button (`GET /api/doc`).

```bash
# build (or pull ghcr.io/pleias/cag-app once published — CI builds on v* tags)
docker build -t cag-app ./app

# run — /data holds your document + the saved KV cache slot
docker run -d --name cag -p 8000:8000 -v "$HOME/cag-data:/data" cag-app

# open http://localhost:8000 → paste a document → chat
# verify the full pipeline (ingest, grounded EN/FR answers, multi-turn, restart persistence):
CONTAINER=cag ./app/smoke.sh
```

Measured on a Pi 4 (Q4_K_M): **44–53 tok/s prefill, 12 tok/s decode** — 2× the benchmark's
Q8_0 decode, because decode is memory-bandwidth-bound and Q4 halves the weight bytes.
A grounded answer over a cached document lands in a few seconds. Design spec:
[`docs/superpowers/specs/2026-07-07-cag-app-design.md`](docs/superpowers/specs/2026-07-07-cag-app-design.md);
new numbers + operational notes: [`docs/TECH_REPORT.md`](docs/TECH_REPORT.md) §8.

> Gotcha we hit: recent `llama-server` builds default `--parallel` to auto → 4 slots →
> 4×16k unified KV → CPU decode collapsed to 0.36 tok/s. Single-user CAG needs
> `--parallel 1` (the app's entrypoint pins it). Details in
> [`docs/throughput.md`](docs/throughput.md).

## What is CAG (vs RAG)?

**RAG** (Retrieval-Augmented Generation) retrieves the *k* most relevant chunks for each
incoming query and re-prefills them every time. **CAG** (Cache-Augmented Generation) instead
precomputes the KV cache of a *fixed* context once and reuses that cache across many
questions — the context stays a stable prefix, so `llama.cpp`'s `cache_prompt` can skip
re-prefilling it. They're complementary: RAG picks *which* bounded material is relevant (from
a large corpus), CAG makes working deeply and repeatedly *inside* that bounded material cheap.
See [`docs/USE_CASES.md`](docs/USE_CASES.md) for when each one wins.

## Repo structure

```
.
├── extensions_geographie_deuxieme_etape_ce1.json   # dataset: 49 French CE1 geography fiches
├── app/                              # the example app (see "CAG in a box" above)
│   ├── Dockerfile / entrypoint.sh    # multi-arch image: llama-server + FastAPI, model baked in
│   ├── server/                       # FastAPI app: ingest → cache → SSE chat; static web UI
│   ├── tests/ + smoke.sh             # 21 unit tests + live curl suite (incl. restart persistence)
│   └── pyproject.toml / uv.lock      # app deps (fastapi, uvicorn, httpx)
├── .github/workflows/publish-image.yml # buildx amd64+arm64 → ghcr.io/pleias/cag-app on v* tags
├── bench/
│   ├── cag_bench.py                  # the benchmark runner (nocontext / naive / cag modes)
│   ├── questions_entry31.json        # single-fiche questions (entry 31, "water points")
│   ├── questions_corpus_water.json   # cross-fiche questions (water cluster, entries 29-38)
│   ├── questions_entry32.json        # single-fiche questions (entry 32)
│   ├── run_server.sh / stop_server.sh # launch/stop the x86 CPU llama-server prebuilt
│   ├── results/                      # raw per-question JSON logs from benchmark runs
│   └── kv_cache/                     # disk-persisted KV cache slots (--slot-save-path)
├── docs/
│   ├── TECH_REPORT.md                # primary write-up: methodology, findings, numbers
│   ├── USE_CASES.md                  # why/when CAG beats small-model RAG
│   ├── throughput.md                 # tok/s breakdown computed from the raw JSON logs
│   ├── pi_setup.md                   # Raspberry Pi launch notes
│   ├── cag_benchmark_design.md       # benchmark design rationale
│   └── results/results_*.md          # per-experiment, per-question result writeups
├── pyproject.toml / uv.lock          # uv-managed deps (cag_bench.py also carries inline
│                                      # PEP-723 metadata, so `uv run` works standalone)
└── llama-b9864-bin-ubuntu-x64/       # prebuilt x86 CPU llama.cpp (gitignored, not in repo —
                                       # download separately; the Pi instead uses a source
                                       # build of llama.cpp, `llama serve`)
```

Model weights (GGUF files) are **not** in the repo — download them separately (see
Quickstart). `models/` is where `bench/run_server.sh` expects them by default.

## Quickstart

**1. Get a GGUF model.** For example, Qwen2.5-1.5B-Instruct (or the 0.5B for a quick
smoke test):

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir models
```

**2. Launch the server** (`llama-server` on x86, `llama serve` on the Pi's source build),
with `--jinja` so the chat template is applied and a `--ctx-size` big enough for your context:

```bash
MODEL=models/qwen2.5-1.5b-instruct-q4_k_m.gguf CTX=8192 ./bench/run_server.sh
# or, manually:
llama-server --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --ctx-size 8192 --jinja --slot-save-path bench/kv_cache --host 127.0.0.1 --port 8080
```

**3. Check the context fits, then run the benchmark** with `uv` (no manual venv/install
needed — it reads `pyproject.toml`/inline PEP-723 deps):

```bash
uv run bench/cag_bench.py --tokens-only

# single-fiche, all three context variants, all three modes, quick smoke test
uv run bench/cag_bench.py --model-format chatml --context full --limit 3 --tag smoke

# single-fiche, full comparison across context variants
uv run bench/cag_bench.py --entry 31 --questions bench/questions_entry31.json \
  --model-format chatml --context answer answer_exercises full --tag qwen15b

# cross-fiche corpus mode: cache the water cluster (entries 29-38) once, ask 8 questions
uv run bench/cag_bench.py --corpus 29-38 --corpus-field answer \
  --questions bench/questions_corpus_water.json --model-format chatml --tag corpus-15b

# Qwen3 (thinking model): disable the reasoning trace for a direct answer
uv run bench/cag_bench.py --model-format chatml --no-think --tag qwen3
```

Run `python3 bench/cag_bench.py --help` for the full, current flag list.

## How the benchmark works

For each question, the runner can hit the *same* context in three **modes**:

- **`nocontext`** — question only, no fiche. Baseline: quality/leak floor and the
  "skip retrieval entirely" speed floor.
- **`naive`** — the full context is re-sent and re-prefilled on *every* question (cache-less
  — the behavior of a RAG pipeline without a cache). "Without CAG."
- **`cag`** — the context is a stable prefix whose KV is *reused* across questions
  (`llama.cpp`'s `cache_prompt`). "With CAG."

Two context regimes, selected via `--context`/`--corpus`:

- **Single-fiche** (`--context answer|answer_exercises|full`): entry 31 ("water points") fed
  at three sizes — `answer` (lesson content only, ~1.9k tok), `answer_exercises` (+exercises,
  ~2.4k tok), `full` (+corrigé/answer key, ~3.1k tok).
- **Corpus** (`--corpus 29-38 --corpus-field answer|full`): several fiches concatenated into
  one ~21k-token cached context, answering cross-fiche questions (aggregate/count/locate/
  compare/synthesis) that require reading across documents.

Each run reports **prefill** (`prompt_ms`), **decode** (`predicted_ms`), and **total**
latency, plus the resulting **KV cache size**, streaming results to a crash-safe, run-tagged
JSONL file under `bench/results/` (`--tag` names the run).

## Models tested

| Model | Params | Format | KV/tok | CAG-viable? | Notes |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | 494M | context-first ChatML | 12 KB | Yes | fast; quality floor |
| **Qwen2.5-1.5B-Instruct** | 1.54B | context-first ChatML | 28 KB | Yes | **sweet spot** |
| Qwen3-1.7B | 2.03B | ChatML + thinking | 112 KB | Yes, but heavy | 8 KV heads → 9× cache; needs `--reasoning-budget 0` / `--no-think` |
| Baguettotron | 321M | query-first `<source_N>` | 60 KB | No | trained query-first (source after question) — a RAG-optimized design, not CAG-compatible; also an `<\|im_end\|>` ≠ EOS quirk |
| Pleias-RAG-350M / 1B | 353M / 1.2B | query-first `<\|query_start\|>` | — | No | whole PleIAs family is query-first, i.e. optimized for per-query RAG, not for a stable cacheable prefix |
| LFM2-350M / 700M | 354M / 742M | context-first ChatML | — | Partial | hybrid conv+attention breaks `cache_prompt` (llama.cpp#19041); needs explicit state save/restore |

## Key results

**Total wait, without vs with CAG** (naive = re-prefill each question; cag = cache once),
representative slice — full table with all context sizes in `docs/TECH_REPORT.md` §6:

| Model | Context | In tok | Without CAG | With CAG | Speedup |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | answer (1.9k) | 1,956 | 30.0s | 20.8s | 1.4× |
| Qwen2.5-0.5B | **corpus (21k)** | 21,000 | **602.6s** | **125.1s** | **4.8×** |
| Qwen2.5-1.5B | answer (1.9k) | 1,956 | 79.4s | 40.6s | 2.0× |
| Qwen2.5-1.5B | **corpus (21k)** | 21,000 | **1,352.4s** | **198.6s** | **6.8×** |

**Prefill only — the pure CAG effect** (what CAG actually removes):

| Model | Context | Prefill without CAG | Prefill with CAG | Speedup | KV cache |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | answer (1.9k) | 10.3s | 0.5s | 20× | 22.5 MB |
| Qwen2.5-0.5B | **corpus (21k)** | **452.7s** | 6.9s | **65×** | 245 MB |
| Qwen2.5-1.5B | answer (1.9k) | 51.5s | 1.7s | 30× | 52.5 MB |
| Qwen2.5-1.5B | **corpus (21k)** | **1,153.9s** | 15.3s | **76×** | 572 MB |

Output is small (145–341 tokens), but decode is slow, so decode dominates *total* time while
prefill is where CAG delivers — and its share grows with input size. Full numbers, per-model
throughput (tok/s), and the quality scorecard are in `docs/TECH_REPORT.md`.

## Docs index

- [`docs/TECH_REPORT.md`](docs/TECH_REPORT.md) — the primary write-up: objective, method,
  models, findings, timing tables, limitations.
- [`docs/USE_CASES.md`](docs/USE_CASES.md) — why/when on-device CAG beats small-model RAG.
- [`docs/throughput.md`](docs/throughput.md) — tok/s (prefill/decode/total) computed from the
  raw per-question JSON logs in `bench/results/`.
- [`docs/pi_setup.md`](docs/pi_setup.md) — Raspberry Pi launch notes for the source-built
  `llama serve`.
- [`docs/cag_benchmark_design.md`](docs/cag_benchmark_design.md) — benchmark design rationale.
- `docs/results/results_*.md` — per-experiment, per-question result writeups (Baguettotron,
  Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen3-1.7B, and the two corpus runs).

## Status / caveats

This is a research/experimental benchmark, not a product. Quality verdicts come from **one
fiche, one cluster, and a handful of questions per type** — indicative, not statistical.
Timings were measured on a dedicated Raspberry Pi via `llama.cpp`; an x86 Xeon was tried but
discarded for timing because it was shared/contended (load ~55) and produced noise, not
signal. Decode — not prefill — is the bottleneck at large cached context (down to ~1.2–2.9
tok/s at 21k tokens), so CAG's biggest wins are on bounded, single-lesson contexts rather than
large concatenated corpora. See `docs/TECH_REPORT.md` §7 for the full limitations list
(cold-start isolation, output truncation, KV-size assumptions). The disk save/restore CAG
path — flagged there as "not yet benchmarked" — is now implemented and live-validated in
the example app (`docs/TECH_REPORT.md` §8): the cache survives container restarts.
