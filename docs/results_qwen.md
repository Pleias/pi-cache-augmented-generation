# Results — Qwen2.5-0.5B-Instruct (the CAG win)

**Date:** 2026-07-04
**Model:** Qwen2.5-0.5B-Instruct, Q8_0 GGUF (~0.49B params), context-first ChatML instruct model
**Runtime:** `llama.cpp llama serve`, `--jinja`, `--ctx-size 4096`, on the **Raspberry Pi** (dedicated)
**Runner:** `bench/cag_bench.py --model-format chatml` (fiche in the **system** message → stable, cacheable prefix; question in the user turn)
**Data:** entry 31 (water-points fiche), 2 questions × 3 modes × 2 context variants (smoke run, `--limit 2`)
**Settings:** temp 0, seed 42, max_tokens 1024, repeat_penalty 1.0

This is the model that delivers **both** halves the PleIAs family structurally couldn't:
**grounded answers AND the CAG prefill savings.** See `results_baguettotron.md` for why
Baguettotron/PleIAs can't (query-first RAG format, CAG-hostile).

## 1. CAG works — prefill collapses via KV prefix reuse

`cag` mode keeps the fiche (in the system message) as a stable prefix; `cache_prompt`
reuses its KV so only the new question tokens are prefilled.

| variant | mode | prefill / question | `prompt_n` | `cache_n` (reused) |
|---|---|---|---|---|
| answer (1840 tok) | nocontext | ~282 ms | 68 | 0 |
| | **naive** | **~9,971 ms** | 1946 | 0 |
| | **cag** | **~320 ms** | **27** | **1919** |
| answer_exercises (2338 tok) | nocontext | ~281 ms | 68 | 0 |
| | **naive** | **~13,603 ms** | 2445 | 0 |
| | **cag** | **~372 ms** | **27** | **2418** |

**Total prefill over the 2 questions:**
- answer: naive **19,954 ms → cag 640 ms** → **~31× reduction**
- answer_exercises: naive **27,207 ms → cag 743 ms** → **~37× reduction**

`cache_n` = 1919 / 2418 confirms the *entire* fiche+system prefix is reused; `prompt_n`
drops to **27** (just the question). Bigger context → bigger CAG win (37× on the larger
variant vs 31×), exactly as expected — prefill cost is what CAG amortizes away.

Wall-clock per question: **naive ~14–16 s → cag ~3–5 s** on the Pi.

## 2. Answers are grounded (the fiche is actually used)

With the fiche in the system message, Qwen reads it. Example (Q1, "qu'est-ce que la mare
de Warang, et pourquoi naturel?"), `naive`/`cag`:

> "La mare de Warang est un point d'eau **naturel**. Elle est considérée comme tel car elle
> est **créée par la nature et n'a pas été aménagée par l'homme**. Elle est un endroit où
> **l'eau est recueillie après les pluies**…"

Matches the expected answer ("se remplit après les pluies", "naturel", "sans intervention
humaine"). Q2 (forage de Mbodiène): "point d'eau **artificiel** … **creusé pour que tout le
monde ait de l'eau potable**" — correct. No hallucination, no lesson-plan mimicry, no loops.

## 3. Cache correctness — `cag` ≈ `naive`

Same content in both modes → answers match. Q2 answers are byte-identical between naive and
cag; Q1 (answer variant) diverges only in the trailing clause (a minor greedy-decoding
effect of cached vs freshly-computed KV not being bit-identical — worth noting, not a
correctness problem). answer_exercises Q1 was identical in both modes.

## 4. Leak check — `nocontext` shows context is needed

Without the fiche the model hallucinates the supporting detail: "La mare de Warang… située
dans la région de la **Sénégalie**… point de départ important pour la **navigation et la
pêche**" (fabricated). It guesses the binary naturel/artificiel label correctly (~50/50),
but the *grounded reasoning* ("se remplit après les pluies", "creusé pour l'eau potable")
only appears with the fiche. So the questions are not fully leaky — context adds the
correct, specific justification.

## 5. Why Qwen works where PleIAs didn't

- **Context-first ChatML:** real `system` role rendered *before* the user turn → fiche is a
  stable prefix → `cache_prompt` reuses its KV. PleIAs models are query-first (context after
  the question) → no reusable prefix.
- **Pure attention:** standard KV cache, so llama.cpp prefix reuse "just works" (unlike
  LFM2's hybrid conv+attention, where `--prompt-cache`/prefix reuse is broken/crashes).
- **Proper EOS** (`<|im_end|>`): no run-on/confabulation bug (Baguettotron's `<|im_end|>`
  is not EOS).
- Multilingual (French) and instruct-tuned for extractive answers.

## 6. Conclusion

On a Raspberry Pi, **Qwen2.5-0.5B-Instruct achieves grounded QA over a cached fiche with a
~31–37× prefill reduction** (naive 10–14 s → cag ~0.3–0.4 s per question), answers matching
between cached and uncached runs. This is the positive CAG demonstration: cache the fiche's
KV once, answer many questions cheaply, without losing grounding — enabled purely by the
**context-first** prompt layout that the PleIAs RAG models don't support.

### Next
- Full run (all 9 questions, including corrigé + utility): `uv run bench/cag_bench.py --model-format chatml --tag qwen-full`.
- Optional: compare Qwen-1.5B for answer quality, and quantify the cag-vs-naive minor answer divergence across all questions.
