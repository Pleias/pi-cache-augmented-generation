# Cache-Augmented Generation on Edge Devices — Technical Report

**Scope:** Feasibility and performance of Cache-Augmented Generation (CAG) with small
(<2B) LLMs on a Raspberry Pi, using French primary-school geography teaching "fiches" as
the grounding context. **Dates:** 2026-07-03 → 07-05. **Hardware:** Raspberry Pi
(dedicated) via `llama.cpp` / `llama-server`; an x86 Xeon was available but discarded for
timing (load ~55 → contention noise). Detailed per-experiment docs are linked in §9.

---

- **Bottom line:** On a Pi, **Qwen2.5-1.5B-Instruct + CAG + a context-sized-to-the-question**
  working set gives grounded answers that would otherwise be impractical. CAG makes the
  mechanics nearly free; the residual limits are decode speed and model reasoning.

---

## 1. Objective

Test whether CAG — precompute the KV cache of a fixed context once, reuse it to answer many
questions cheaply — is viable and worthwhile for small LLMs on slow, offline hardware
(target: Raspberry Pi), using teacher lesson fiches as the grounding documents.

## 2. What CAG is (and the three modes we measured)

For each question we ran three modes against the *same* context:
- **`nocontext`** — question only, no fiche. Baseline: quality/leak floor and "skip
  retrieval" speed floor.
- **`naive`** — full context re-sent and **re-prefilled every question** (cache-less; the
  behaviour of a RAG pipeline that doesn't cache). "Without CAG."
- **`cag`** — context kept as a stable prefix, its KV **reused** across questions
  (`llama.cpp cache_prompt`). "With CAG." `cache_n` in the server timings confirms how many
  tokens were reused.

Primary metric: **prefill time** (`prompt_ms`) — the cost CAG removes. Secondary: decode
(`predicted_ms`) and total wall-clock. Runner: `bench/cag_bench.py` (streaming, incremental
crash-safe JSONL, run-tagged outputs, auto KV-cache sizing from the GGUF).

## 3. Experimental design

- **Data:** 49 CE1 geography fiches (Senegal CEB curriculum, French). Single-fiche tests use
  **entry 31** (water points) at three context sizes — `answer` (1.9k tok), `answer+exercises`
  (2.4k), `full` (3.1k). Corpus test concatenates the **water cluster (idx 29–38, ~21k tok)**
  into one cached context with **8 cross-fiche questions** (aggregate / count / locate /
  compare / synthesis) that require reading across fiches.
- **Questions:** 9 single-fiche (4 recall, 2 corrigé, 3 utility) + 8 cross-fiche.
- **Determinism:** temp 0 (greedy) for Qwen2.5; Qwen3 used its recommended non-thinking
  sampling (temp 0.7 / top_p 0.8 / top_k 20) with thinking disabled.

## 4. Models evaluated

| model | params | format | KV/tok | CAG-viable? | notes |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | 494M | context-first ChatML | 12 KB | ✅ | fast; quality floor |
| **Qwen2.5-1.5B-Instruct** | 1.54B | context-first ChatML | 28 KB | ✅ | **sweet spot** |
| Qwen3-1.7B | 2.03B | ChatML + thinking | 112 KB | ✅ but heavy | 8 KV heads → 9× cache; needs `--reasoning-budget 0` / `--no-think` |
| Baguettotron | 321M | query-first `<source_N>` | 60 KB | ❌ | grounds only query-first → not cacheable; `<|im_end|>`≠EOS bug |
| Pleias-RAG-350M / 1B | 353M / 1.2B | query-first `<|query_start|>` | — | ❌ | whole PleIAs family is query-first |
| LFM2-350M / 700M | 354M / 742M | context-first ChatML | — | ⚠️ | hybrid conv+attn breaks `cache_prompt` (llama.cpp#19041); needs explicit state save/restore |

## 5. Key findings

**F1 — Ordering decides CAG viability.** CAG requires the context *before* the query (stable
prefix). Baguettotron and the PleIAs RAG family are trained **query-first** (sources after
the question). Placing the source first to enable caching **breaks their grounding** (the
model reverts to hallucination/confabulation). Verified decisively: same model, only the
ordering changed. → No PleIAs model can combine grounding + CAG. Standard instruct models
(Qwen) put context in the system message → context-first → CAG works.

**F2 — Prompt format is make-or-break.** Getting Baguettotron to ground at all required its
exact RAG format (`<source_1>…</source_1>` special tokens, **no** system message, explicit
`stop` on `<|im_end|>` because it isn't the model's EOS — otherwise it runs on into an
invented conversation). Wrong framing → hallucination, lesson-plan mimicry, or repetition
loops. Qwen "just works" with context in the system role.

**F3 — CAG prefill saving is large and grows with context.** 20× at 1.9k tokens → 65–76× at
21k. `cache_n` = full context every cag call (faithful reuse). See §6.

**F4 — Decode is the residual wall.** CAG does not cache decode. Decode throughput falls as
the cached context grows (each token attends over the whole KV): ~15–20 tok/s on a small
context → **~2.9 tok/s (0.5B) / ~1.2 tok/s (1.5B)** over a 21k-token / 245–572 MB cache. So
total speedups (1.4–6.8×) trail prefill speedups (20–76×). Single-fiche stays interactive;
corpus scale is batch-only on a Pi.

**F5 — Model quality is the ceiling.** Qwen2.5-0.5B: grounded recall OK, but wrong on a
vrai-faux (Q6), weak corrigé justification, and **fails cross-fiche reasoning** (miscounts,
hallucinates, denies having the corpus). Qwen2.5-1.5B: **fixes Q4/Q5/Q6**, the "exercises
help" axis finally discriminates, and **improves on 7/8 cross-fiche questions vs the 0.5B**
(2/8 fully correct, 5/8 usable) — only cross-document *counting* still hard-fails, and it does
so cleanly (no loops/hallucination). Cost: ~1.6–2× wall-clock, 2.3× cache.

**F6 — KV cache size is architecture-driven.** Not param count. The *smaller* Baguettotron
(321M, 80 layers) has a 5× bigger per-token cache than the *larger* Qwen2.5-0.5B (24 layers).
Qwen3's 8 KV heads make its cache 9× the Qwen2.5-0.5B's. For CAG, few KV heads (GQA) +
moderate depth win. KV quantization (`-ctk q8_0 -ctv q8_0`) halves it independently of weight
quant.

**Quality scorecard** (strictly-correct answers per question type; from the per-question
outputs in `docs/results_*.md`, §9). Single-fiche counts the **best of the three context
variants**; corpus is the 8 cross-fiche questions.

| model | test | recall | answer-key | utility | cross-fiche |
|---|---|:---:|:---:|:---:|:---:|
| Qwen2.5-0.5B | single-fiche | 3/4 | 1/2 | 1/3 * | — |
| Qwen2.5-1.5B | single-fiche | 4/4 | 2/2 | 3/3 | — |
| Qwen2.5-0.5B | corpus (21k) | — | — | — | **1/8** · 1/8 usable |
| Qwen2.5-1.5B | corpus (21k) | — | — | — | **2/8** · 5/8 usable |

Cross-fiche is scored on two bars: **fully correct** (bold `X/8`, the doc's "Correct" label)
and **usable** (correct + "Good" — right task, minor gaps). Do **not** confuse this with the
"7/8" in the source doc and §5 F5: that is an *improved-vs-the-0.5B* count (7 of 8 questions
behave more sanely on the 1.5B), not 7 correct answers.

Reading the scorecard (the numbers alone undersell the 0.5B→1.5B gap — *how* they fail matters):

- **Single-fiche, best-variant is generous.** No single variant passes everything: the answer-key
  questions only score once the **`answer+exos`/`full`** context carries the corrigé (F1/F5). On the
  **0.5B**, answer-key still fails Q6 (vrai-faux negation) in *all* variants — a reasoning limit, not
  a context one. On the **1.5B** the answer-key axis separates cleanly (right label → right label +
  correct justification). `*` 0.5B utility: only Q7 has recorded output; Q8–Q9 are undocumented.
- **Corpus "strict correct" hides the real story.** 1/8 vs 2/8 looks close, but the **failure modes
  diverge sharply.** 0.5B: partial recall → **repetition loops that eat the token budget** (4/8
  truncated), plus hallucinated places, a wrong salt/fresh comparison, and one answer that **denies
  having the corpus**. 1.5B: **no hallucination, no denial, only 1/8 truncated**; the 4 "partial"
  answers recover real entities and just miss some — the *one* hard failure is **cross-document
  counting** (Q2: says 2, truth 6), which neither model can do. So the 1.5B is broadly usable at
  corpus scale where the 0.5B is not, despite the near-identical strict score.

## 6. Results — timing & tokens (per question, Raspberry Pi)

**Reading the `test` column.** Each name is a context size (its token count is in parentheses).
The single-fiche tests all use **fiche 31 (water points)** but feed the model progressively more
of it: **`answer`** (1.9k tok) = only the lesson's answer text; **`answer+exos`** (2.4k) = that
plus the exercises; **`full`** (3.1k) = the whole fiche (answer + exercises + corrigé/answer key).
**`corpus`** (21k) is a different, larger context: the whole water cluster (fiches 29–38) cached
together for the cross-fiche questions. More context = a bigger, slower cache but more material
the model can ground on, at the cost of ~1.5× cache and decode. The effect on answers is real but
model-dependent: on **Qwen2.5-1.5B**, adding the exercises (`answer` → `answer+exos`) is the step
that flips the "do exercises help?" corrigé questions from a right-but-guessed label to a correctly
*justified* one; going further to `full` adds little (sometimes identical, sometimes a slight
regression). On the **0.5B** the same extra context does *not* fix the corrigé questions — a model-
capability limit, not a context one (see §5 F5). Per-question outputs are in `docs/results_*.md` (§9).

**Total wait, without vs with CAG** (naive = re-prefill each query; cag = cache once):

| model | test | in tok | out tok | without CAG | with CAG | total × | no-context |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.5B | answer (1.9k) | 1,956 | 341 | 30.0s | 20.8s | 1.4× | 5.8s |
| 0.5B | answer+exos (2.4k) | 2,455 | 183 | 29.7s | 11.8s | 2.5× | 5.8s |
| 0.5B | full (3.1k) | 3,180 | 145 | 31.2s | 10.2s | 3.1× | 5.8s |
| 0.5B | **corpus (21k)** | 21,000 | 337 | **602.6s** | **125.1s** | **4.8×** | 11.7s |
| 1.5B | answer (1.9k) | 1,956 | 227 | 79.4s | 40.6s | 2.0× | 17.2s |
| 1.5B | answer+exos (2.4k) | 2,455 | 256 | 107.2s | 49.8s | 2.2× | 17.4s |
| 1.5B | full (3.1k) | 3,180 | 193 | 137.5s | 42.4s | 3.2× | 17.6s |
| 1.5B | **corpus (21k)** | 21,000 | 216 | **1,352.4s** | **198.6s** | **6.8×** | 13.4s |

**Prefill only — the pure CAG effect** (what CAG removes):

| model | test | prefill without CAG | prefill with CAG | prefill × | KV cache |
|---|---|---:|---:|---:|---:|
| 0.5B | answer (1.9k) | 10.3s | 0.5s | 20× | 22.5 MB |
| 0.5B | full (3.1k) | 20.4s | 0.8s | 25× | 36.8 MB |
| 0.5B | corpus (21k) | **452.7s** | 6.9s | **65×** | 245 MB |
| 1.5B | answer (1.9k) | 51.5s | 1.7s | 30× | 52.5 MB |
| 1.5B | full (3.1k) | 90.3s | 2.7s | 33× | 85.9 MB |
| 1.5B | corpus (21k) | **1,153.9s** | 15.3s | **76×** | 572 MB |

Reading: output is small (145–341 tok) but decode is slow, so decode dominates *total*;
prefill is where CAG delivers, and its share grows with input size. Amortized cold-start:
naive re-prefills the corpus **every** question (8× ≈ 60–150 min); true CAG pays it **once**,
then near-zero — the more questions per cached context, the bigger the win.

## 7. Limitations & caveats

- **Cold-start not isolated:** in each run `naive` executed before `cag` and warmed the slot,
  so `cag` Q1 didn't show the true one-time cold prefill (~7.5 min at corpus scale). True cold
  CAG pays it once on the first question. Run `--modes cag` alone to isolate.
- **Truncation:** `max_tokens` (512 corpus / 1024 single) cut some long aggregate/utility
  answers (0.5B corpus 4–6/8; 1.5B 1/8) — decode totals would rise with a higher limit.
- **KV size assumes f16** (2 B/elem, llama.cpp default). Using `-ctk/-ctv q8_0` halves the
  reported MB (runner currently prints the f16 figure).
- **Disk save/restore CAG not benchmarked:** we used in-memory prefix reuse. The
  "precompute the cache on a fast box, ship the KV blob, restore on the Pi" path
  (`--slot-save-path`) would remove the cold prefill entirely and is the natural next step.
- **Small quality sample:** one fiche / one cluster / a handful of questions per type;
  quality verdicts are indicative, not statistical.

