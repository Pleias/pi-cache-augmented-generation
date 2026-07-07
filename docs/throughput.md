# CAG Benchmark Throughput

Computed from the raw per-question JSON logs in `bench/results/`. All figures are
**aggregate** sums across a cell's question rows — `sum(tokens) / sum(ms) * 1000` —
not averages of per-question ratios.

Source files (PleIAs-tested models only):

| Model | Task | File |
|---|---|---|
| Qwen2.5-0.5B | single-fiche (entry 31) | `entry31_qwen-full_20260704-202428.json` |
| Qwen2.5-1.5B | single-fiche (entry 31) | `entry31_qwen25-1.5b-full_20260705-001832.json` |
| Qwen2.5-0.5B | corpus (21k, entries 29-38) | `corpus29-38_corpus-05b-full_20260704-215342.json` |
| Qwen2.5-1.5B | corpus (21k, entries 29-38) | `corpus29-38_corpus-15b-full_20260705-013454.json` |

Qwen3-1.7B and Baguettotron are **excluded** — no raw JSON logs exist for them, only
aggregate tables in `docs/results_*.md`, which don't carry the per-question
`prompt_ms`/`predicted_ms` fields needed for this computation.

Definitions:
- **prefill tok/s** = `sum(prompt_n) / sum(prompt_ms) * 1000`
- **decode tok/s** = `sum(predicted_n) / sum(predicted_ms) * 1000`
- **total tok/s** = `sum(prompt_n + predicted_n) / sum(wall_ms) * 1000`

## Qwen2.5-0.5B

| variant / mode | ctx tok | out tok | prefill tok/s | decode tok/s | total tok/s | cache_n |
|---|---:|---:|---:|---:|---:|---:|
| answer (~1.9k) / nocontext | 78 | 114 | 237.9 | 20.9 | 33.1 | 0 |
| answer (~1.9k) / naive | 1956 | 330 | 189.3 | 16.8 | 76.3 | 0 |
| answer (~1.9k) / cag | 45 | 342 | 85.2 | 16.8 | 18.6 | 1911 |
| answer_exercises (~2.4k) / nocontext | 78 | 114 | 239.8 | 20.9 | 33.3 | 0 |
| answer_exercises (~2.4k) / naive | 2455 | 255 | 174.8 | 16.3 | 91.1 | 0 |
| answer_exercises (~2.4k) / cag | 45 | 184 | 70.0 | 16.5 | 19.4 | 2410 |
| full (~3.1k) / nocontext | 78 | 114 | 238.2 | 20.9 | 33.2 | 0 |
| full (~3.1k) / naive | 3180 | 167 | 155.9 | 15.4 | 107.1 | 0 |
| full (~3.1k) / cag | 45 | 145 | 55.6 | 15.4 | 18.6 | 3135 |
| corpus (~21k) / nocontext | 97 | 229 | 244.0 | 20.4 | 28.0 | 0 |
| corpus (~21k) / naive | 21000 | 427 | 46.4 | 2.8 | 35.6 | 0 |
| corpus (~21k) / cag | 66 | 338 | 9.5 | 2.9 | 3.2 | 20934 |

## Qwen2.5-1.5B

| variant / mode | ctx tok | out tok | prefill tok/s | decode tok/s | total tok/s | cache_n |
|---|---:|---:|---:|---:|---:|---:|
| answer (~1.9k) / nocontext | 78 | 112 | 41.7 | 7.3 | 11.0 | 0 |
| answer (~1.9k) / naive | 1956 | 166 | 38.0 | 6.0 | 26.7 | 0 |
| answer (~1.9k) / cag | 45 | 227 | 25.8 | 5.9 | 6.7 | 1911 |
| answer_exercises (~2.4k) / nocontext | 78 | 112 | 41.7 | 7.2 | 10.9 | 0 |
| answer_exercises (~2.4k) / naive | 2455 | 221 | 36.8 | 5.5 | 25.0 | 0 |
| answer_exercises (~2.4k) / cag | 45 | 256 | 20.1 | 5.4 | 6.0 | 2410 |
| full (~3.1k) / nocontext | 78 | 112 | 41.6 | 7.1 | 10.8 | 0 |
| full (~3.1k) / naive | 3180 | 225 | 35.2 | 4.8 | 24.8 | 0 |
| full (~3.1k) / cag | 45 | 193 | 16.4 | 4.9 | 5.6 | 3135 |
| corpus (~21k) / nocontext | 97 | 80 | 42.0 | 7.2 | 13.2 | 0 |
| corpus (~21k) / naive | 21000 | 234 | 18.2 | 1.2 | 15.7 | 0 |
| corpus (~21k) / cag | 66 | 216 | 4.3 | 1.2 | 1.4 | 20934 |

## Interpretation

**Decode tok/s falls as context grows.** Decode cost scales with the size of the
KV cache the model attends over on every generated token, so it drops steadily
as prefilled/cached context grows from ~1.9k → ~21k tokens (comparing `naive`,
where `prompt_n` is the full context size):

- **Qwen2.5-0.5B**: ~16.8 tok/s at 1.9k ctx → ~15.4 tok/s at 3.1k ctx → **~2.8 tok/s at 21k ctx** (a ~6x drop).
- **Qwen2.5-1.5B**: ~6.0 tok/s at 1.9k ctx → ~4.8 tok/s at 3.1k ctx → **~1.2 tok/s at 21k ctx** (a ~5x drop).

**`cag` prefill tok/s is not a meaningful measure of model prefill speed.** In
`cag` mode only the new question is re-prefilled (`prompt_n` ≈ 45-66 tokens);
essentially the whole context is reused via `cache_n` (up to ~20,934 tokens at
the corpus scale) and is not counted in `prompt_n`/`prompt_ms`. Dividing a tiny
token count by the fixed per-call overhead of attending over a huge cache
produces numbers (9.5–85.2 tok/s here) that are an artifact of that overhead,
not the model's actual prefill throughput. **The real prefill throughput
figure is `naive`**, where the entire context is freshly prefilled: 155.9-244.0
tok/s (0.5B) and 18.2-42.0 tok/s (1.5B) — both falling as context grows toward
21k, consistent with attention cost scaling with context length.

**Decode tok/s is essentially identical between `naive` and `cag`** for the same
variant, as expected — both decode over a KV cache of the same total size
(`prompt_n + cache_n` context), so per-token decode cost is the same regardless
of how that context was populated:

| model | variant | naive decode | cag decode | diff |
|---|---|---:|---:|---:|
| 0.5B | answer | 16.8 | 16.8 | 0.1% |
| 0.5B | answer_exercises | 16.3 | 16.5 | 1.5% |
| 0.5B | full | 15.4 | 15.4 | 0.2% |
| 0.5B | corpus | 2.8 | 2.9 | 0.3% |
| 1.5B | answer | 6.0 | 5.9 | 2.1% |
| 1.5B | answer_exercises | 5.5 | 5.4 | 1.4% |
| 1.5B | full | 4.8 | 4.9 | 2.4% |
| 1.5B | corpus | 1.2 | 1.2 | 0.1% |

**`nocontext` decode is consistently faster** than `naive`/`cag` for the same
model, since it decodes over a near-empty context (~78-97 tokens): 20.4-20.9
tok/s for 0.5B and 7.1-7.3 tok/s for 1.5B, both well above the corresponding
`naive`/`cag` decode rates at any context size.

**Excluded models**: Qwen3-1.7B and Baguettotron have no raw per-question JSON
logs in `bench/results/` — only aggregate summary tables in `docs/results_*.md`
— so they cannot be included in this per-row throughput computation.

---

## All models — tok/s + TTFT

Consolidated table across every model tried in the CAG benchmark, adding
**TTFT** (time-to-first-token) to the Qwen2.5 figures above and folding in
**Qwen3-1.7B** and **Baguettotron**, which have no raw per-row logs. All runs
are on the Raspberry Pi (dedicated) unless marked otherwise. All arithmetic
below was done with `python3` (sums/means/divisions), not eyeballed — see the
`compute.py` derivation described in each subsection.

**Measured-vs-derived legend**
- **[measured]** — computed directly from per-row JSON in `bench/results/` (`sum(tokens)/sum(ms)*1000` for tok/s, `mean(ttft_ms)` for TTFT).
- **[derived]** — computed from the *aggregate* per-question-average ms figures in `docs/results_qwen3_1.7b.md` plus the context-token counts stated there (no per-row output-token counts exist for Qwen3, so decode tok/s could not be independently recomputed — see note below).
- **[single probe]** — one live `llama-server` `print_timing` reading, not a full benchmark run (Baguettotron only).

### Master table

| Model | Context (tok) | Output (tok) | Mode | Prefill tok/s | Decode tok/s | TTFT | Total tok/s | Basis |
|---|---:|---:|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B | 78 | 114 | nocontext | 237.9 | 20.9 | 338ms | 33.1 | measured |
| Qwen2.5-0.5B | 1956 | 330 | naive | 189.3 | 16.8 | 10.3s | 76.3 | measured |
| Qwen2.5-0.5B | 1911 (cached) | 342 | cag | 85.2* | 16.8 | 539ms | 18.6 | measured |
| Qwen2.5-0.5B | 78 | 114 | nocontext | 239.8 | 20.9 | 335ms | 33.3 | measured |
| Qwen2.5-0.5B | 2455 | 255 | naive | 174.8 | 16.3 | 14.1s | 91.1 | measured |
| Qwen2.5-0.5B | 2410 (cached) | 184 | cag | 70.0* | 16.5 | 654ms | 19.4 | measured |
| Qwen2.5-0.5B | 78 | 114 | nocontext | 238.2 | 20.9 | 337ms | 33.2 | measured |
| Qwen2.5-0.5B | 3180 | 167 | naive | 155.9 | 15.4 | 20.4s | 107.1 | measured |
| Qwen2.5-0.5B | 3135 (cached) | 145 | cag | 55.6* | 15.4 | 822ms | 18.6 | measured |
| Qwen2.5-0.5B | 97 | 229 | nocontext | 244.0 | 20.4 | 406ms | 28.0 | measured |
| Qwen2.5-0.5B | 21000 | 427 | naive | 46.4 | 2.8 | 452.8s | 35.6 | measured |
| Qwen2.5-0.5B | 20934 (cached) | 338 | cag | 9.5* | 2.9 | 7.0s | 3.2 | measured |
| Qwen2.5-1.5B | 78 | 112 | nocontext | 41.7 | 7.3 | 1.9s | 11.0 | measured |
| Qwen2.5-1.5B | 1956 | 166 | naive | 38.0 | 6.0 | 51.5s | 26.7 | measured |
| Qwen2.5-1.5B | 1911 (cached) | 227 | cag | 25.8* | 5.9 | 1.8s | 6.7 | measured |
| Qwen2.5-1.5B | 78 | 112 | nocontext | 41.7 | 7.2 | 1.9s | 10.9 | measured |
| Qwen2.5-1.5B | 2455 | 221 | naive | 36.8 | 5.5 | 66.8s | 25.0 | measured |
| Qwen2.5-1.5B | 2410 (cached) | 256 | cag | 20.1* | 5.4 | 2.2s | 6.0 | measured |
| Qwen2.5-1.5B | 78 | 112 | nocontext | 41.6 | 7.1 | 1.9s | 10.8 | measured |
| Qwen2.5-1.5B | 3180 | 225 | naive | 35.2 | 4.8 | 90.3s | 24.8 | measured |
| Qwen2.5-1.5B | 3135 (cached) | 193 | cag | 16.4* | 4.9 | 2.8s | 5.6 | measured |
| Qwen2.5-1.5B | 97 | 80 | nocontext | 42.0 | 7.2 | 2.3s | 13.2 | measured |
| Qwen2.5-1.5B | 21000 | 234 | naive | 18.2 | 1.2 | 1154.0s | 15.7 | measured |
| Qwen2.5-1.5B | 20934 (cached) | 216 | cag | 4.3* | 1.2 | 15.3s | 1.4 | measured |
| Qwen3-1.7B (Q8_0, thinking off) | 1919 | n/a† | nocontext | n/a | ~6.3† | 1.7s | n/a | derived (smoke, n=2) |
| Qwen3-1.7B (Q8_0, thinking off) | 1919 | n/a† | naive | 35.4 | ~4-5†‡ | 54.3s | n/a | derived (smoke, n=2) |
| Qwen3-1.7B (Q8_0, thinking off) | 1919 (cached) | n/a† | cag | 2820ms/tok** | ~4-5†‡ | 2.8s | n/a | derived (smoke, n=2) |
| Qwen3-1.7B (Q8_0, thinking off) | 2418 | n/a† | nocontext | n/a | ~6.3† | 1.7s | n/a | derived (smoke, n=2) |
| Qwen3-1.7B (Q8_0, thinking off) | 2418 | n/a† | naive | 34.0 | ~4-5†‡ | 71.2s | n/a | derived (smoke, n=2) |
| Qwen3-1.7B (Q8_0, thinking off) | 2418 (cached) | n/a† | cag | 3384ms/tok** | ~4-5†‡ | 3.4s | n/a | derived (smoke, n=2) |
| Qwen3-1.7B (Q8_0, thinking off) | 3143 | n/a† | nocontext | n/a | ~6.3† | 1.7s | n/a | derived (smoke, n=2) |
| Qwen3-1.7B (Q8_0, thinking off) | 3143 | n/a† | naive | 30.7 | ~4.0† | 102.3s | n/a | derived (smoke, n=2) |
| Qwen3-1.7B (Q8_0, thinking off) | 3143 (cached) | n/a† | cag | 4213ms/tok** | ~4.0† | 4.2s | n/a | derived (smoke, n=2) |
| Baguettotron 321M (Q8_0) | 76 | 512 | raw probe (no CAG variant) | 290.1 | 28.5 | 262ms | n/a | single probe |
| Baguettotron 321M (Q8_0) | ~2445 | n/a | naive (fiche) | ~2445/(15-21s)≈120-165‡‡ | n/a | 15-21s | n/a | doc estimate, no per-row log |
| Baguettotron 321M (Q8_0) | ~2445 (cached) | n/a | cag (fiche, source-first — CAG-mechanical only, **ungrounded**) | n/a (cache reused) | n/a | 0.6-0.85s | n/a | doc estimate, no per-row log |

`*` = `cag` "prefill tok/s" divides the tiny re-prefilled `prompt_n` (the new
question only, ~45-66 tok) by its prefill ms; it is **not** the model's real
prefill throughput because almost all context is reused via `cache_n` and
never re-prefilled. Compare to the `naive` row directly above it for the real
prefill rate at that context size.

`**` = Qwen3 `cag` prefill is reported as raw ms (2820-4213ms) rather than a
tok/s rate for the same reason as above — `docs/results_qwen3_1.7b.md` does
not log the small re-prefilled `prompt_n` for `cag` rows, so no denominator
token count exists to divide by.

`†` = `docs/results_qwen3_1.7b.md` reports per-question-average **ms** only
(prefill/decode/total), not output token counts (`predicted_n`). Decode tok/s
therefore could not be independently recomputed from token counts the way the
Qwen2.5 rows were. The `~6.3` (nocontext) and `~4-5` (naive/cag, context in
KV) figures are the doc's own stated aggregate decode rate ("decode ≈ 4-6
tok/s... 6.3 tok/s with no context → 4 tok/s with the fiche in KV"), carried
through here unchanged — **not** recomputed from `sum(tokens)/sum(ms)`. Back-
solving `predicted_n` from those stated rates and the logged decode-ms
(`predicted_n ≈ rate × decode_ms / 1000`) gives ~73-74 tok for `nocontext` and
~36-54 tok for `naive`/`cag`, which is internally consistent but is itself a
derived-from-derived estimate, not a logged count — included here only as a
sanity check, not as a table value.

`‡` = the `naive`/`cag` decode ms are close to each other per variant (e.g.
`answer`: both 9022ms) — consistent with the general finding (confirmed on
Qwen2.5) that decode tok/s depends on total KV-cache size, not on how that
cache was populated (fresh prefill vs cache reuse).

`‡‡` = computed here from the ~2445-tok fiche size and the doc's stated
15-21s naive-prefill range (`docs/results/results_baguettotron.md`); this is
a range estimate from prose text, not a logged per-row figure — no `predicted_n`
or decode-ms exist for Baguettotron at all, so decode tok/s is `n/a`.

Qwen3-1.7B is a 2-question smoke run (Q1/Q2 recall only), not the full
9-question suite the Qwen2.5 numbers come from — treat its numbers as
indicative, not comparable in sample size to the Qwen2.5 rows.

### Notes

**1. Decode tok/s falls as context grows — measured ranges:**
- **Qwen2.5-0.5B**: 20.9 tok/s (near-empty ctx) → 16.8 tok/s (1.9k ctx) → 15.4 tok/s (3.1k ctx) → **2.8-2.9 tok/s (21k ctx)**, a ~7x drop end to end.
- **Qwen2.5-1.5B**: 7.1-7.3 tok/s (near-empty ctx) → 6.0 tok/s (1.9k ctx) → 4.8-4.9 tok/s (3.1k ctx) → **1.2 tok/s (21k ctx)**, a ~6x drop end to end.
- **Qwen3-1.7B** [derived, doc's own figures, smoke n=2]: ~6.3 tok/s (no context) → ~4 tok/s (fiche in KV, up to 3.1k ctx) — same qualitative trend, not tested at 21k.
- Baguettotron: only one probe point exists (28.5 tok/s at effectively-empty context); no context-scaling data.

**2. `cag` TTFT is far below `naive` TTFT because the context is cached and not re-prefilled** — a couple of concrete numbers:
- Qwen2.5-0.5B, corpus (~21k ctx): naive TTFT **452.8s** vs cag TTFT **7.0s** — a ~65x reduction.
- Qwen2.5-1.5B, corpus (~21k ctx): naive TTFT **1154.0s** (~19.2 min) vs cag TTFT **15.3s** — a ~75x reduction.
- Qwen2.5-0.5B, full (~3.1k ctx): naive TTFT **20.4s** vs cag TTFT **822ms** — a ~25x reduction.
- Qwen3-1.7B, full (~3.1k ctx) [derived]: naive TTFT **102.3s** vs cag TTFT **4.2s** — a ~24x reduction.
- Baguettotron, ~2445-tok fiche [doc estimate]: naive prefill **15-21s** vs cag prefill **0.6-0.85s** — roughly an **18-35x** reduction (min/max of the stated ranges; the doc's own headline figure is "~28x") — but only holds mechanically under source-first context ordering, under which the model's answers were ungrounded (see caveat in note 3).

**3. Measured vs derived vs single-probe, by model:**
- **Qwen2.5-0.5B and Qwen2.5-1.5B**: every number above is **[measured]** — computed via Python (`sum(prompt_n)/sum(prompt_ms)*1000` etc.) directly from the per-row JSON logs in `bench/results/` (`entry31_qwen-full_20260704-202428.json`, `entry31_qwen25-1.5b-full_20260705-001832.json`, `corpus29-38_corpus-05b-full_20260704-215342.json`, `corpus29-38_corpus-15b-full_20260705-013454.json`), n=8-9 rows per cell.
- **Qwen3-1.7B**: **[derived]** from the per-question-average ms table in `docs/results_qwen3_1.7b.md` (a 2-question smoke run, not the full 9-question suite). Prefill tok/s was computed here from that doc's context-token counts ÷ its prefill-ms; decode tok/s could not be independently recomputed (no output-token counts logged) and is carried through from the doc's own stated aggregate rate.
- **Baguettotron**: **[single probe]** — prefill ≈290 tok/s / decode ≈28.5 tok/s is one live `llama-server` `print_timing` reading (262ms/76tok prefill, 17955ms/512tok decode), not a benchmark aggregate. Separately, `docs/results/results_baguettotron.md` records fiche-prefill *times only* (naive ~15-21s, cag ~0.6-0.85s, no decode data). **Baguettotron and the wider PleIAs RAG family were excluded from the formal CAG throughput analysis**: they are query-first models (context must follow the question to stay grounded), which is structurally incompatible with prefix-cache CAG (context must precede the question to form a stable, reusable KV prefix). The cag timing above is real only for the source-first ordering that CAG requires, under which Baguettotron's answers are ungrounded — so despite the ~24-28x mechanical prefill speedup, CAG is not usable with this model family.

**4. Hardware**: all rows are Raspberry Pi (dedicated), except Baguettotron's single probe and doc timings, which per `docs/results/results_baguettotron.md` were also measured on the Pi (a contended Xeon run exists in that doc but is explicitly flagged there as unusable noise and is not included here).
