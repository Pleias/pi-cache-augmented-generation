# Model speed + example answer (single-fiche, ~1.9k context)

One row per model, at the moderate ~1.9k-token single-lesson context, on the Raspberry Pi.
Followed by each model's actual answer to the same question, so speed and quality can be
read together.

## Throughput

| Model | Context (tok) | Output (tok) | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1,956 | ~340 | 189 | **16.8** |
| Qwen2.5-1.5B | 1,956 | ~230 | 38 | **6.0** |
| Qwen3-1.7B * | 1,919 | n/a | 35 | **~4–5** |
| Baguettotron-321M ** | 76 | 512 | 290 | **28.5** |

\* Qwen3-1.7B: 2-question smoke run, no per-row output-token logs → decode tok/s approximate.
\** Baguettotron: single live `llama-server` probe at **76-token** context (no logged ~1.9k run); query-first, so it cannot do prefix-cache CAG anyway. Fastest raw because it is the smallest (321M).

Takeaways: decode speed tracks model size (321M→1.7B: 28.5→~4 tok/s); prefill is ~10× faster than decode on every model (parallel vs sequential). Decode tok/s is the "how fast it writes" number.

## The question (entry 31, Q1)

> **Selon la fiche, qu'est-ce que la mare de Warang, et pourquoi est-elle considérée comme un point d'eau naturel ?**

**Expected answer:** *La mare de Warang est une mare qui se remplit après les pluies ; elle est un point d'eau naturel car elle existe sans intervention humaine directe.*

## Actual answers (single fiche, `cag` mode)

**Qwen2.5-0.5B** — 72 output tokens, grounded ✅
> La mare de Warang est un point d'eau naturel. Elle est considérée comme tel car elle est créée par la nature et n'a pas été aménagée par l'homme. Elle est un endroit où l'eau est recueillie après les pluies, ce qui lui confère un caractère naturel.

**Qwen2.5-1.5B** — 46 output tokens, grounded ✅ (more concise)
> La mare de Warang est considérée comme un point d'eau naturel car elle se remplit après les pluies, ce qui indique qu'elle n'est pas créée par l'intervention humaine.

**Qwen3-1.7B** — grounded ✅ *(from the 2-question smoke; display truncated at 90 chars)*
> La mare de Warang est un point d'eau naturel car elle se remplit après les pluies et est c…

**Baguettotron-321M** — grounded only after its exact RAG format (`<source_1>` tags, question-first) ⚠️
> La Mare de Warang est une mer naturelle située dans le sud du Sénégal
>
> *(correct on "Sénégal" and "naturelle"; minor error "mer"/sea instead of "mare"/pond. With any other prompt framing it ignored the context and hallucinated a "grande rivière du Karnataka, en Inde" — see `results/results_baguettotron.md`.)*

## Reading it together

- **Qwen2.5-0.5B and 1.5B** both ground cleanly on this recall question; the 1.5B is more concise. On harder questions (cross-fiche, vrai/faux, corrigés) the 1.5B pulls ahead — see `TECH_REPORT.md` §5.
- **Qwen3-1.7B** grounds too, but is the slowest to write (~4–5 tok/s) and carries a 9× KV cache — not worth it over the 1.5B for CAG.
- **Baguettotron** is the fastest engine (321M) but query-first: it only grounds in its exact format and cannot do prefix-cache CAG, so its raw speed doesn't translate into a CAG workflow.

*Sources: throughput from `docs/throughput.md`; answers from `bench/results/entry31_qwen-full_*.json` (0.5B), `bench/results/entry31_qwen25-1.5b-full_*.json` (1.5B), the Qwen3-1.7B smoke run, and `docs/results/results_baguettotron.md`.*
