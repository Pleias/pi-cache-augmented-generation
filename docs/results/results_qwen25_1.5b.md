# Results — Qwen2.5-1.5B-Instruct (the CAG sweet spot)

**Date:** 2026-07-04
**Model:** Qwen2.5-1.5B-Instruct, Q8_0 GGUF (~1.54B params), context-first ChatML, no thinking mode
**Runtime:** `llama serve --jinja -c 8192`, Raspberry Pi. Runner `--model-format chatml`, temp 0.
**Run:** smoke, 2 recall questions (Q1 Warang, Q2 Mbodiène) × 3 modes × 3 context variants.

## Latency (per-question averages, ms)

| variant | mode | prefill | decode | **total** | KV cache |
|---|---|---|---|---|---|
| **answer** (1919 tok) | nocontext | 1,620 | 7,565 | 9,193 | **52.5 MB** |
| | naive | **52,577** | 7,255 | 59,845 | |
| | **cag** | **1,228** | 7,219 | **8,460** | |
| **answer_exercises** (2418 tok) | nocontext | 1,626 | 7,522 | 9,167 | **66.1 MB** |
| | naive | **68,361** | 7,962 | 76,343 | |
| | **cag** | **1,553** | 7,950 | **9,516** | |
| **full** (3143 tok) | nocontext | 1,622 | 7,582 | 9,225 | **85.9 MB** |
| | naive | **92,914** | 8,273 | 101,204 | |
| | **cag** | **1,834** | 7,989 | **9,839** | |

## CAG impact — the best total speedup so far

| variant | naive prefill | cag prefill | **prefill ×** | naive total | cag total | **total ×** |
|---|---|---|---|---|---|---|
| answer | 52.6s | 1.2s | **43×** | 59.8s | 8.5s | **7.1×** |
| answer_exercises | 68.4s | 1.6s | 44× | 76.3s | 9.5s | 8.0× |
| full | **92.9s** | 1.8s | **51×** | 101.2s | 9.8s | **10.3×** |

- naive re-prefills the fiche at **53–93s per question**; cag drops it to **1.2–1.8s** (cache_n
  confirms full-prefix reuse). Prefill saving **43–51×**, growing with context.
- **`cag` is even faster than `nocontext`** (8.5s vs 9.2s on `answer`): reusing the cached prefix
  is cheaper than prefilling the system+question from scratch — grounded answers *faster* than
  no-context ones.
- Decode ~7–8s (short recall answers), so total is prefill-dominated → **total speedup 7–10×**,
  the best of the three models tested.

## Three-model comparison (same Q1/Q2 recall smoke, `answer` variant)

| model | KV/tok | cache | naive prefill | cag prefill | prefill × | naive total | cag total | **total ×** |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 12 KB | 22.5 MB | 10.2s | 0.34s | 30× | 13.8s | 4.1s | 3.4× |
| **Qwen2.5-1.5B** | 28 KB | **52.5 MB** | 52.6s | 1.2s | **43×** | 59.8s | **8.5s** | **7.1×** |
| Qwen3-1.7B | 112 KB | 209.9 MB | 54.3s | 2.8s | 19× | 63.3s | 11.9s | 5.3× |

**Qwen2.5-1.5B dominates Qwen3-1.7B on every CAG axis here:**
- **4× smaller cache** (52 vs 210 MB) — only 2 KV heads vs Qwen3's 8.
- **Faster cag** (8.5s vs 11.9s) — and even its cag *prefill* is lower (1.2s vs 2.8s): the fat
  Qwen3 KV cache makes even cache reuse slower.
- **Better speedup** (43× / 7.1× vs 19× / 5.3×).
- No thinking-mode handling needed (direct answer, runner unchanged).

The 0.5B is fastest in absolute terms (small model) but its total speedup is diluted because
prefill is cheap on it; the 1.5B has the biggest *relative* CAG win while keeping a modest cache.

## Throughput
Prefill ~35 tok/s, decode ~5 tok/s on the Pi — ≈ Qwen3-1.7B despite fewer params (the Pi is
memory-bandwidth-bound; both big models saturate it similarly), but the 1.5B moves a 4× smaller
KV cache, which is why its cag path is faster.

## Quality
Smoke only covers easy recall (Q1/Q2) — grounded and correct, cag == naive. The quality
question (does 1.5B fix the 0.5B's Q4 weak-recall / Q6 wrong-reasoning) needs a FULL 9-question
run, not done here.

## Verdict
**Qwen2.5-1.5B is the sweet spot for CAG on a Pi:** best relative speedup (7–10×), a still-modest
52–86 MB cache, 3× the params of the 0.5B for grounding, and it beats the newer Qwen3-1.7B on
speed *and* cache because it uses 2 KV heads instead of 8. If a full-run quality check confirms
it fixes the 0.5B's misses, this is the recommended model.
