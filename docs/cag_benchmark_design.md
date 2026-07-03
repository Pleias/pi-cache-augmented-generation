# CAG Benchmark — Design & Reasoning

**Status:** design / initial setup
**Date:** 2026-07-03
**Model under test:** [Baguettotron (PleIAs) — GGUF](https://huggingface.co/PleIAs/Baguettotron-GGUF)
**Runtime:** `llama.cpp` prebuilt (`llama-b9864-bin-ubuntu-x64`, CPU-only) via `llama-server` HTTP API
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

**Must verify before running:** Baguettotron's trained context window (must exceed
~4k-token fiche + Q + A), chat template presence (`--jinja` / `--chat-template`), and
which quant (.gguf) — keep the same quant here and on the Pi for apples-to-apples.

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

### Chosen starting entries

- **idx 32 — "Découverte des cours d'eau"** — headline realistic HIGH case: named
  Senegal rivers + ocean, map localisation, exercise corrigés. First 7-question set is
  built here.
- **(control, optional) idx 43 — school-plan fiche** — different HIGH mechanism: an
  *invented inline layout* ("1 = Portail … 5 = Toilettes") that exists only in that fiche.
  Answer is *impossible* without the context → cleanest proof the cache is actually used
  (no-context mode must fail; CAG must pass). Kept as a control if we want a zero-leak
  anchor beyond the water cluster.

## 4. Protocol (v0)

1. Start `llama-server` once with Baguettotron + `--slot-save-path` (enables disk KV
   save/restore for the Pi cold-start scenario).
2. Load fiche idx 32 as context; save the slot (CAG cache artifact).
3. For each of the 7 questions, run modes (a)/(b)/(c) with `temperature 0`, fixed seed.
4. Record per request: `prompt_ms`, `predicted_ms`, token counts, and the answer text.
5. Score answers against expected answers; drop any question that mode (c) already
   answers (leaky).
6. Report: amortization curve (cost vs N questions) for CAG vs naive; quality parity
   between (a) and (b); leak rate from (c).
7. Repeat the whole run on the Raspberry Pi (same quant, same fiche, same questions) and
   compare the prefill-time savings — this is the headline result.

Questions for idx 32 live in `bench/questions_entry32.json`.
