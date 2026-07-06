# Results — Qwen3-1.7B (bigger model, thinking OFF)

**Date:** 2026-07-04
**Model:** Qwen3-1.7B, Q8_0 GGUF (~2.03B total params), context-first ChatML, **thinking disabled**
**Runtime:** `llama serve --jinja --reasoning-budget 0` + runner `--no-think` (sends
`chat_template_kwargs {enable_thinking:false}`), on the Raspberry Pi.
**Run:** smoke, 2 recall questions (Q1 Warang, Q2 Mbodiène) × 3 modes × 3 context variants.
Sampling: temp 0.7, top_p 0.8, top_k 20.

Companion to `results_qwen_full.md` (Qwen2.5-0.5B). Purpose: does 3.5× the params improve
answers, and at what speed/cache cost on a Pi?

## Latency (per-question averages, ms)

| variant | mode | prefill | decode | **total** | KV cache |
|---|---|---|---|---|---|
| **answer** (1919 tok) | nocontext | 1,676 | 11,616 | 13,307 | **209.9 MB** |
| | naive | **54,256** | 9,022 | 63,296 | |
| | **cag** | **2,820** | 9,022 | **11,858** | |
| **answer_exercises** (2418 tok) | nocontext | 1,674 | 11,668 | 13,409 | **264.5 MB** |
| | naive | **71,183** | 9,897 | 81,124 | |
| | **cag** | **3,384** | 10,393 | **13,794** | |
| **full** (3143 tok) | nocontext | 1,679 | 11,809 | 13,558 | **343.8 MB** |
| | naive | **102,303** | 13,448 | 115,773 | |
| | **cag** | **4,213** | 12,863 | **17,095** | |

## Throughput on the Pi
- **Prefill ≈ 35 tok/s**, **decode ≈ 4–6 tok/s** (decode slows as the cached context grows —
  attention over more KV: 6.3 tok/s with no context → 4 tok/s with the fiche in KV).
- ~5× slower prefill and ~3–4× slower decode than Qwen2.5-0.5B (3.5× params + 8-head KV
  hitting the Pi's memory bandwidth).

## CAG impact — the strongest yet
Because this model is slow, prefill dominates and CAG's win is dramatic:

| variant | naive prefill | cag prefill | **prefill ×** | naive total | cag total | **total ×** |
|---|---|---|---|---|---|---|
| answer | 54.3s | 2.8s | 19× | 63.3s | 11.9s | 5.3× |
| answer_exercises | 71.2s | 3.4s | 21× | 81.1s | 13.8s | 5.9× |
| full | **102.3s** | 4.2s | 24× | 115.8s | 17.1s | **6.8×** |

naive is **54–102s per question** just to re-prefill the fiche — CAG drops that to ~3–4s
(cache_n confirms full-prefix reuse). On a slow big model, **CAG is not optional — it's what
makes the model usable.** And unlike the 0.5B (where decode diluted the win to 1.4–3×), here
prefill is a big enough share that the **total** speedup is 5.3–6.8×.

## The cost: KV cache is ~9× bigger
Qwen3's **8 KV heads** (vs Qwen2.5-0.5B's 2) → **112 KB/token** vs 12 KB/token:

| | Qwen2.5-0.5B | Qwen3-1.7B |
|---|---|---|
| KV per token | 12 KB | **112 KB** |
| cache: answer / full | 22.5 / 36.8 MB | **209.9 / 343.8 MB** |

A 344 MB cache for the `full` fiche is still shippable but is ~9× the 0.5B's. This is the
architectural cost the model survey flagged — bigger isn't free for CAG.

## Quality (recall smoke)
Q1/Q2 grounded and correct (Warang = naturel/se remplit après pluies; Mbodiène =
artificiel/creusé), cag == naive (cache correct). `nocontext` hallucinates ("désert du
Sahara en Algérie" for Warang) → grounding confirmed needed. **These are the easy recall
questions the 0.5B already passed** — the real quality test (does 1.7B fix the 0.5B's Q4
weak-recall and Q6 wrong-reasoning) needs the FULL 9-question run, not this smoke.

## Verdict so far
Qwen3-1.7B: much slower (5× prefill, 3–4× decode) and ~9× the KV cache, but CAG's relative
win is bigger (total 5.3–6.8× vs the 0.5B's 1.4–3×) precisely because it's slow. Whether the
extra params are worth it depends on the full-run quality vs the 0.5B — TBD. Consider a Q4_K_M
quant to roughly halve the wall-clock (weights only; does NOT shrink the KV cache — use
`-ctk q8_0 -ctv q8_0` for that).
