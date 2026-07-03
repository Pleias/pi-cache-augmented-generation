# CAG Benchmark — Design & Reasoning

**Status:** design / initial setup
**Date:** 2026-07-03
**Model under test:** [Baguettotron (PleIAs) — GGUF](https://huggingface.co/PleIAs/Baguettotron-GGUF), Q8_0 (343 MB, ~320M params, `llama` arch). **Reasoning model** — see §5.
**Runtime:** `llama.cpp` prebuilt (`llama-b9864-bin-ubuntu-x64`, CPU-only) via `llama-server` HTTP API
**Context window:** 4096 (server `n_ctx`, confirmed via `/props`; model trained at 4000)
**Dataset:** `extensions_geographie_deuxieme_etape_ce1.json` — 49 CE1 geography teaching fiches (Senegal CEB curriculum, French)

---

## 1. What we are doing

We are measuring **Cache-Augmented Generation (CAG)** on a small LLM running on slow
hardware. CAG = precompute the KV cache for a large, stable context (here: one teacher
fiche) **once**, then reuse that cache to answer many questions — instead of re-feeding
and re-prefilling the whole context on every request.

Test rig: run `llama-server` once (model + cache stay warm in-process), ask questions
over HTTP. Compare answering the same questions three ways:

| Mode | What happens | Cost per question |
|------|--------------|-------------------|
| **(a) CAG / cached** | Fiche prefilled once; KV cache reused (in-memory prefix reuse, or disk save/restore) | prefill **once**, then cheap decode |
| **(b) Naive / full-context** | Full fiche + question sent every time; prefill runs every time | prefill **× N** |
| **(c) No-context baseline** | Question only, no fiche | prefill of question only |

We test locally on this x86 CPU box first, then repeat on a **Raspberry Pi**. The Pi is the
real target: prefill is painfully slow there, so skipping it (CAG) is where the win shows.

## 2. What we are testing (and what we are NOT)

**Key insight:** in modes (a) and (b) the model sees *identical content*. CAG changes only
*whether the fiche is re-prefilled*, never *what the model reads*. Therefore:

- **Primary metric = speed**, specifically **prefill time / time-to-first-token (TTFT)**.
  CAG's whole value is killing repeated prefill. This is what the Pi exposes.
  - Source: `llama-server` returns `timings` per response (`prompt_ms` = prefill,
    `predicted_ms` = decode, `prompt_per_second`). No external stopwatch needed.
  - The interesting result is the **amortization curve**: naive pays prefill × N;
    CAG pays it × 1. Plot cost vs. number of questions.
- **Answer quality is a correctness check, not the headline.** (a) and (b) should give
  ~identical answers. If cached answers diverge from full-context answers, the cache is
  being corrupted → that is a bug signal, not a feature.
- **Mode (c) is the validity gate.** It proves the questions actually *require* the fiche.
  If the no-context baseline already answers a question correctly, that question is
  "leaky" (answerable from the model's world knowledge) and is dropped — it can't
  distinguish CAG from nothing.

**Determinism:** `temperature: 0`, fixed `seed`, fixed `n_predict`, so any answer
difference is a cache effect, not sampling noise.

**Verified (2026-07-03):** context window = 4096 (server `/props`); chat template ships
in the GGUF and is applied via `--jinja` (ChatML, see §5); quant = Q8_0 (use the same on
the Pi for apples-to-apples). ⚠️ The model is a **verbose reasoning model** — this breaks
the naive "fiche + Q + A fits in 4k" assumption. See §5.

## 3. Why the water-points cluster

A CAG benchmark is only meaningful when the correct answer **cannot come from the
model's parametric memory** — the model must be forced to read the provided fiche.
Otherwise CAG and no-context look the same and we measure nothing.

We ranked all 49 fiches by "requires-the-fiche" strength (topic + `situations` +
`exercises` fields). The topics cluster clearly:

- **LOW (reject):** day/night, Earth rotates, Earth is round, cardinal points, wind,
  weather. Universal astronomy/physics a small model already knows. Senegalese character
  names (Binta, Assane) are narrative flavor, not answer-bearing. → CAG indistinguishable
  from no-context. **Our original pick, entry 0 (jour/nuit), fell here and was dropped.**
- **MED:** relief vocabulary (plaine/plateau/colline/cuvette), generic seasons — standard
  primary-geography terms, partly derivable.
- **HIGH (use):** the **water cluster, idx 29–38**. Consistently anchored in real,
  specific Senegalese hydrography — **Fleuve Sénégal, Fleuve Gambie, Fleuve Casamance,
  Océan Atlantique, Lac de Guiers, marigots** — plus fine, non-obvious lexical
  distinctions (**forage / puits / mare**, **ruisseau / rivière**, **cuvette / vallée**)
  and specific exercise corrigés + barèmes. These answers are hard for a generalist French
  model to reconstruct without the fiche → CAG value is measurable.

**Why this cluster wins for the CAG story:** the answers are locally grounded and
non-derivable, so quality is genuinely context-dependent, while the fiche is large and
stable enough (~3.5k tokens) that its prefill cost is exactly what CAG amortizes away.

### Chosen entry: idx 31 (answer-only sized to fit the reasoning model)

Because Baguettotron is a verbose reasoning model in a 4k window (§5), we feed a **trimmed
fiche**, not the whole entry. We picked **idx 31 — "Les points d'eau de mon milieu
(naturels et artificiels)"**: smallest `answer` field in the water cluster *and* Sonnet's
#1 non-common-knowledge entry. Its `answer` field is self-contained and carries strong
named/local anchors — **Lac Rose, mare de Warang, forage de Mbodiène** — plus naturel/
artificiel definitions and an embedded eval. Exact tokens (model tokenizer):
`answer` = 1841, `answer`+`exercises` = 2363 → both leave ≥1.1k tokens for CoT+answer.

### Two context variants (the "does more context help?" axis)

| Variant | Content | ~tok | Purpose |
|---------|---------|------|---------|
| `answer` | lesson only | 1841 | baseline context |
| `answer_exercises` | lesson + exercises-with-corrigés | 2363 | adds the exercise bank |

Cross with the 3 cache modes → 6 cells per question. What each variant proves:
- **Corrigé questions** (Q5–6, e.g. "Le lac de Guiers — N or A?") are grounded in facts
  that exist **only in the exercises** field → variant `answer` should **fail**, variant
  `answer_exercises` should **pass**. A within-benchmark proof that adding context helps.
- **Teacher-utility questions** (Q7–9, "propose an exercise/activity for concept X") test
  real teacher value. Scored by a **rubric of expected elements** (uses local examples,
  ardoise format, N/A structure…), not exact match. Hypothesis: `answer_exercises` yields
  more grounded, curriculum-aligned proposals than `answer`, and both beat `nocontext`.
  leak_risk HIGH (generic proposals are possible without context) → the signal is
  *grounding/quality*, compared A-vs-B, not pass/fail.

Question set: `bench/questions_entry31.json` — 9 questions (4 recall/`answer`,
2 corrigé/`exercises`, 3 utility), each tagged `requires_context` + `leak_risk`.
(The earlier idx-32 set, `bench/questions_entry32.json`, is superseded — kept for reference.)

## 4. Protocol (v0)

1. Start `llama-server` once with Baguettotron + `--jinja` + `--slot-save-path`
   (`./bench/run_server.sh`; disk KV save/restore enables the Pi cold-start scenario).
2. `uv run bench/cag_bench.py --tokens-only` — confirm both variants fit 4096.
3. `uv run bench/cag_bench.py` — for each **context variant** (`answer`,
   `answer_exercises`) × **cache mode** (nocontext/naive/cag) × 9 questions, with
   `temperature 0`, fixed seed, `max_tokens 512`.
4. Runner records per request: `prompt_ms` (prefill), `predicted_ms`, `cache_n`
   (verifies cache reuse), token counts, the `<think>` reasoning, the post-`</think>`
   answer, and a truncation flag.
5. Score: recall/corrigé questions vs `expected_answer`; utility questions vs
   `expected_elements` rubric. Drop any question `nocontext` already answers (leaky).
6. Report: (a) amortization curve CAG vs naive per variant; (b) corrigé questions —
   `answer` fails / `answer_exercises` passes (context-value proof); (c) utility grounding
   A vs B vs nocontext; (d) quality parity between cag and naive (same content → same
   answer, else cache bug).
7. Repeat on the Raspberry Pi (same quant/fiche/questions) — prefill-time savings there
   is the headline result. ⚠️ Ignore speed numbers from the shared x86 host (§5).

Results → `bench/results/entry31_results.json`. Questions → `bench/questions_entry31.json`.

## 5. Model behavior — probed from the running server (2026-07-03)

Facts pulled from `/props` and a live completion. These change the runner design.

### Chat template (ChatML + forced thinking)
```
{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}
```
The template **always injects `<think>\n`** at the start of the assistant turn. Baguettotron
is a **reasoning model**: every response opens with a chain-of-thought, then closes it with
`</think>`, then emits the final answer.

### Observed output format
For `"Quelle est la capitale du Sénégal ?"` (max_tokens=200) the returned `content` was pure
reasoning, in **English**, in a fixed scaffold:
```
### 1. Query Parsing
### 2. Core Facts
### 3. Contextual Expansion
### 4. Additional Information
### 5. Response Structure
```
…and it hit `finish_reason: length` **before** emitting `</think>` → **no final answer at all**.
The real answer only appears *after* `</think>`, which needs more tokens than the reasoning
consumes.

Consequences for the benchmark:
1. **`max_tokens` must be large** (reasoning alone > 200 tokens here). `150` captures zero
   answer. Budget for full CoT + answer, or the run yields only truncated thinking.
2. **The answer must be parsed out:** split on `</think>`; score only the text *after* it.
   If `</think>` never appears (truncation), mark the question as "no-answer / truncated",
   not wrong.
3. **Language mismatch:** it *reasons* in English even for a French prompt. Check what
   language the post-`</think>` answer is in before scoring.
4. Timings object exposes `cache_n` (cached-prefix tokens) alongside `prompt_ms` — use
   `cache_n` to **verify** the CAG cache is actually being reused (should jump to ~fiche
   length on cached questions, ~0 on naive).

### ⚠️ Context-budget conflict (the real blocker)
Window is 4096. Fiche ≈ 3.3k tokens. That leaves only ~700 tokens for **thinking + answer**,
but the reasoning scaffold alone can exceed that. So the naive plan (full fiche + verbose
CoT + answer in one 4k window) **does not fit**. Options, to decide before the real run:
- **Suppress/shorten thinking** — the template hard-codes `<think>`, so this needs a custom
  `--chat-template` (or a system instruction like "réponds directement, sans raisonnement").
  Off-distribution risk; must validate answer quality.
- **Trim the fiche** to only answer-bearing fields, paired with a matching question subset.
- **RoPE-scale** the context to 6–8k — degrades a model trained at 4k; muddies results.

### ⚠️ Measurement environment
This host (Xeon 8468, 16 cores) was at **load avg ~17, 23 users** during probing → the server
was CPU-starved (observed ~1 tok/s decode, ~8 tok/s prefill). **These numbers are contention
noise, not real performance.** Trust speed numbers only from an idle host or the Pi. Correctness
(answer format, parsing, quality) is unaffected and can be developed here.

**Runner impact (TODO):** `bench/cag_bench.py` currently takes `message.content` verbatim and
defaults `max_tokens=150` — both wrong for a reasoning model. Needs: raise `max_tokens`, strip
the `<think>…</think>` block, detect truncation, and log `cache_n`.
