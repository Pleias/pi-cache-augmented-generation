# CAG on Qwen2.5-1.5B-Instruct — Multi-Fiche Corpus Run (idx 29–38)

**Corpus:** the *water* cluster — TEN CE1 geography fiches (`fiche_indices` 29–38) concatenated into ONE context and cached as a single prefix.
**Model:** Qwen2.5-1.5B-Instruct, Q8_0 quant, context-first ChatML (corpus placed in the system message).
**Hardware:** Raspberry Pi. Temperature 0, `max_tokens` 512.
**Results file:** `bench/results/corpus29-38_corpus-15b-full_20260705-013454.json`
**Questions:** `bench/questions_corpus_water.json` (8 cross-fiche questions).
**Companion (0.5B) report:** `docs/results_corpus_05b.md` — same corpus, same 8 questions, same run harness.

This is a **model comparison**. The corpus, the questions, and the CAG mechanics are identical to the 0.5B run; the only change is 3× the parameters (0.5B → 1.5B). The single question this document exists to answer:

> **Does 3× the params fix the cross-fiche reasoning failures the 0.5B showed — and what does it cost on the clock and in the KV cache?**

The short answers: **latency — CAG works identically well (prefill down ~76×), but decode is even slower than the 0.5B (~1.2 tok/s vs ~2.9), so the total win is ~6.8×.** **Quality — yes, decisively.** The 1.5B clears nearly every failure the 0.5B hit: no repetition loops, no "river water is salty" error, no denial of the corpus, one fully correct comparison, and correct (positional) fiche indices. Cross-document **counting** (Q2) is the one operation that still fails.

---

## 1. What is being compared

Each of 8 questions is answered under three **modes**:

- **`nocontext`** — no corpus in the prompt. Quality/leak baseline.
- **`naive`** — the full 20,934-token corpus is re-prefilled from scratch for *every* question. Correct, but pays the enormous corpus prefill 8 times.
- **`cag`** — the corpus KV cache is computed **once** and reused as a prefix for all 8 questions. Only the short question tokens are prefilled per call.

`naive` and `cag` feed the model the *same* tokens and produce the *same* answer (verified below: naive Q6 and cag Q6 are byte-identical); `cag` only amortizes the corpus prefill. Every CAG row reports **`cache_n = 20934`**, confirming the entire ten-fiche prefix is reused on each call.

**KV cache footprint.** ~28 KB/token for this model (Qwen 1.5B: 28 layers, larger hidden/KV dims than the 0.5B's ~12 KB/token). The cached corpus is 20,934 tokens → **572.4 MB** of KV cache — **2.3× the 0.5B's 245.3 MB**. Big, but still shippable on a Pi.

---

## 2. Latency summary

Totals over the 8 questions, and per-question (`/Q`) averages, in milliseconds. "Prefill" = prompt/context processing, "Decode" = answer generation, "Total" = wall-clock.

| Mode | Prefill total | Decode total | Total | Prefill /Q | Decode /Q | Total /Q | Cached KV |
|---|---:|---:|---:|---:|---:|---:|---:|
| nocontext | 18,521 | 88,899 | 107,538 | 2,315 | 11,112 | 13,442 | — |
| naive | 9,231,303 | 1,587,096 | 10,818,881 | 1,153,913 | 198,387 | 1,352,360 | — |
| **cag** | **122,004** | **1,466,302** | **1,588,782** | **15,251** | **183,288** | **198,598** | **572.4 MB** |

Reading the `/Q` column in human units:

| Mode | Prefill /Q | Decode /Q | Total /Q |
|---|---|---|---|
| nocontext | 2.3 s | 11.1 s | **13.4 s** (but cannot answer — no corpus) |
| naive | **1,153.9 s (~19 min)** | 198.4 s (~3.3 min) | **1,352.4 s (~22.5 min)** |
| cag | **15.3 s** | 183.3 s (~3 min) | **198.6 s (~3.3 min)** |

The full 8-question `naive` run took **10,818,881 ms ≈ 3.0 hours**; the `cag` run took **1,588,782 ms ≈ 26.5 minutes**.

---

## 3. Prefill savings — the CAG effect at 21k tokens

CAG works exactly as intended, and the effect is even larger than on the 0.5B.

- **naive** re-prefills the whole 20,934-token corpus on *every* question: **1,153,913 ms /Q — roughly 19 MINUTES of pure prompt processing before a single output token.**
- **cag** reuses the cached prefix: **15,251 ms /Q — about 15 seconds** (only the short question tokens are processed).
- **Prefill reduction: ~76×** (1,153,913 / 15,251 ≈ 75.7). `cache_n = 20934` on every CAG row confirms full-corpus reuse.

**Why is naive so brutal here?** Long-context prefill throughput on this model/Pi is only:

> 20,934 tokens / 1,153.9 s ≈ **18 tok/s**

That is *slower* than the 0.5B's ~46 tok/s prefill — the bigger model processes each of the 21k context tokens more expensively, and the 572 MB KV working set thrashes memory harder. Paying that 18-tok/s bill from scratch on all 8 questions is what makes naive cost **~3 hours** of wall-clock (9,231,303 ms of prefill alone). CAG pays it effectively once.

---

## 4. Decode is the bottleneck — and it's slower than the 0.5B

CAG kills prefill but does **not** help decode, and at 21k tokens decode is where nearly all the remaining time goes.

- cag decode /Q ≈ **183 s (~3 min)**; it dominates the cag total completely: **183 s of the 199 s /Q total is decode; prefill is only 15 s.**
- Decode throughput over the 572 MB / 21k-token cache collapses to:

> cag: 1,730 generated tokens / 1,466.3 s ≈ **1.18 tok/s**
> naive: 1,874 generated tokens / 1,587.1 s ≈ **1.18 tok/s**

The naive/cag match (1.18 vs 1.18 tok/s) confirms the decode math is identical between them — CAG cannot touch it. For reference, the same model over the empty `nocontext` prompt runs **643 tokens / 88.9 s ≈ 7.2 tok/s**, so grounding on the 21k-token cache costs ~6× the per-token decode speed.

**This is worse than the 0.5B on two axes at once:** the 1.5B decodes at **~1.2 tok/s vs the 0.5B's ~2.9 tok/s**, *and* its cache is 2.3× bigger (572 vs 245 MB). Every generated token must attend over a larger KV set on a heavier model, so both the per-step cost and the cache it attends over grew. This is the intrinsic price of grounding a bigger model on 21k tokens; it is not a CAG failure.

---

## 5. Total speedup — and why it's 6.8×, not 76×

| Comparison | naive /Q | cag /Q | Speedup |
|---|---:|---:|---:|
| Prefill only | 1,153,913 ms | 15,251 ms | **~76×** |
| **Total wall-clock** | 1,352,360 ms | 198,598 ms | **~6.8×** |

The 76× prefill win shrinks to a **6.8×** total win because **decode dominates the total and CAG doesn't touch it.** In `cag`, prefill is 15 s and decode is 183 s — so even after eliminating ~1,139 s of prefill, the ~183 s of big-context decode remains and sets the floor. Going from **~22.5 minutes per question to ~3.3 minutes per question** is the difference between "unusable" and "usable" for a 21k-token corpus on a Pi.

(Note: the 6.8× total win is *larger* than the 0.5B's 4.8×, because the 1.5B's naive prefill is so much heavier that CAG removes proportionally more.)

---

## 6. 0.5B vs 1.5B — cost side by side

| Metric | 0.5B | 1.5B | Ratio (1.5B / 0.5B) |
|---|---:|---:|---:|
| Cached KV (21k tokens) | 245.3 MB | 572.4 MB | **2.3×** |
| naive prefill /Q | 452,721 ms (~7.5 min) | 1,153,913 ms (~19 min) | 2.5× |
| naive prefill throughput | ~46 tok/s | ~18 tok/s | 0.4× (slower) |
| cag decode throughput | ~2.9 tok/s | ~1.18 tok/s | 0.4× (slower) |
| cag total /Q | 125,105 ms (~2 min) | 198,598 ms (~3.3 min) | **1.6×** |
| cag total, 8-Q run | 1,000,839 ms (~16.7 min) | 1,588,782 ms (~26.5 min) | 1.6× |
| Prefill speedup (CAG) | ~65× | ~76× | — |
| Total speedup (CAG) | ~4.8× | ~6.8× | — |

**Bottom line on cost:** the 1.5B costs **2.3× the KV cache** and **~1.6× the cag wall-clock per question** (~3.3 min vs ~2 min). That is more expensive, but not catastrophically so — CAG keeps the 21k-token corpus usable on both models.

---

## 7. Truncation

`max_tokens` was **512**. Truncations (`finish_reason = length`):

| Mode | 1.5B truncated /8 | 0.5B truncated /8 |
|---|---|---|
| nocontext | 0 | 3 |
| naive | 1 (Q7) | 6 |
| cag | **1 (Q7)** | 4 |

The 1.5B truncates far less (cag 1/8 vs 4/8). This is a *quality* signal, not just a budget one: the 0.5B's truncations were mostly repetition loops burning the 512-token budget, whereas the 1.5B produces **concise, complete** answers that finish naturally (`finish_reason = stop`) on 7 of 8 questions. Only the Q7 synthesis genuinely wanted more than 512 tokens.

---

## 8. Quality — does 3× the params fix cross-fiche reasoning?

CAG's job is to deliver the corpus faithfully (`cache_n = 20934` proves all ten fiches are present); the answers below are a **model-capability** result. The verdict is a clear yes on 7 of 8 questions. All quotes are verbatim from the cag rows (naive rows are identical).

Note on fiche numbering: the model refers to fiches by their **position in the concatenation (1–10)**, not by their dataset index (29–38). Position 1 = idx 29, … position 9 = idx 37. Under that mapping the 1.5B's positional references are frequently **correct**.

### 8.1 Q1 — aggregate named water bodies — *Partial, but clean*

> **Expected:** Fleuve Sénégal, Fleuve Gambie, Saloum, Casamance, Océan Atlantique, Lac Rose, mare de Warang.
> **1.5B cag (verbatim, complete — 61 tokens, `stop`):**
> Voici la liste des cours d'eau et plans d'eau nommés par un nom propre que l'on trouve dans le corpus :
> 1. Fleuve Sénégal  2. Rivières  3. Lac Rose  4. Marigot  5. Océan Atlantique

Recovers **3 of 7** real proper names (Fleuve Sénégal, Lac Rose, Océan Atlantique); "Rivières" and "Marigot" are generic categories, not invented places. Misses Gambie, Saloum, Casamance, mare de Warang. Same 3/7 recall as the 0.5B — **but no hallucinated place ("Lac de Kourou"), no repetition loop, and it stops cleanly.** **Better than 0.5B.**

### 8.2 Q2 — count fiches containing « forage » — *Fail (but a cleaner fail)*

> **Expected:** 6 fiches — 29, 30, 31, 34, 36, 38.
> **1.5B cag (verbatim, complete):**
> Le terme "forage" apparaît dans la Fiche 1 (Géographie - Découverte des points d'eau du milieu) et la Fiche 2 (Géographie - Importance des cours d'eau locaux).

Wrong count: **2, not 6.** Cross-document counting is still beyond the model. **But** — unlike the 0.5B, it does **not** deny having the corpus, does **not** invent out-of-range fiche indices (39–42), and grounds its (too-short) answer in real fiche titles. Still wrong, but the failure mode is far more sane. **Marginally better than 0.5B; both fail the actual count.**

### 8.3 Q3 — locate the water-cycle fiche — *Correct*

> **Expected:** fiche 37 (position 9) introduces the water cycle; no other fiche does.
> **1.5B cag (verbatim, complete — 26 tokens, `stop`):**
> La fiche 9 introduit le cycle de l'eau. Les autres fiches ne traitent pas de ce cycle.

"Fiche 9" = position 9 = **idx 37 — correct.** It gives the index (which the 0.5B never did) *and* the uniqueness confirmation, in one clean sentence. **Better than 0.5B.**

### 8.4 Q4 — which fiches locate the Sénégal river, which is richest — *Good*

> **Expected:** fiches 30, 32, 37 locate it; fiche 32 also adds Gambie/Saloum/Casamance.
> **1.5B cag (verbatim, complete):**
> La fiche 4, "Géographie - Découverte des cours d'eau", propose de localiser le Fleuve Sénégal sur la carte du Sénégal. Elle ajoute également d'autres cours d'eau du pays, comme le Fleuve Gambie, le Saloum et la Casamance, pour une visualisation plus complète de l'hydrographie du Sénégal.

"Fiche 4" = position 4 = **idx 32 — the richest fiche, correctly identified**, along with its extra rivers (Gambie, Saloum, Casamance). It names only one fiche (misses that 30 and 37 also locate the Sénégal), but it correctly answers the harder half — *which one goes furthest*. The 0.5B named the rivers but never said which fiche. **Better than 0.5B.**

### 8.5 Q5 — compare mer/océan fiche vs ruisseau/rivière fiche — *Fail (incomplete), but no salinity error*

> **Expected:** sea/ocean = salt water, immense, fishing/transport/tourism; stream/river = fresh water, distinction by size.
> **1.5B cag (verbatim, trimmed):**
> La fiche consacrée au ruisseau et à la rivière présente l'eau de manière plus simplifiée … le ruisseau est un petit cours d'eau, souvent peu profond, et … la rivière est un grand cours d'eau, plus large et plus profond … [second paragraph repeats the same description of the ruisseau/rivière fiche]

It describes only the ruisseau/rivière fiche — correctly (size-based distinction, fresh flowing water implied) — and **repeats that half instead of describing the mer/océan fiche**, so the two-way contrast is never delivered. It does **not** commit the 0.5B's key error of calling river water "salée". So: still a failed comparison (one side only, redundant), but it avoids the outright factual error. **Slightly better than 0.5B — no salinity mistake — but still does not complete the contrast.**

### 8.6 Q6 — compare "points d'eau" vs "cours d'eau" families — *Correct and complete (standout)*

> **Expected:** points d'eau = puits, forage, mare, bassin de rétention (water you *find*, still); cours d'eau = source, ruisseau, rivière, fleuve, mer, océan (water that *flows*).
> **1.5B cag (verbatim, trimmed):**
> Points d'eau :
> - **Puits** : Un trou creusé dans le sol pour atteindre l'eau souterraine …
> - **Forage** : Un trou profond creusé avec une machine …
> - **Mare** : Une petite étendue d'eau naturelle, souvent temporaire …
> - **Bassin de rétention** : … pour recueillir l'eau de pluie …
> Cours d'eau :
> - **Source** … **Ruisseau** … **Rivière** … **Fleuve** … **Mer** … **Océan** …
> En résumé, les points d'eau sont des endroits où l'on trouve de l'eau … tandis que les cours d'eau sont des étendues d'eau qui coulent …

**Fully correct.** Both families listed completely, each item defined correctly, and the essential opposition — *water you find, still* vs *water that flows* — stated exactly right. This is the answer the 0.5B botched worst (it gave the *same* wrong description for both families and inverted the distinction). **Dramatically better than 0.5B.**

### 8.7 Q7 — synthesize all water-dependent human activities — *Good (truncated, some redundancy)*

> **Expected:** drinking, cooking, washing/laundry, irrigation, watering livestock, fishing, transport/commerce, tourism, religious festivals (Tabaski/Korité/Ndogou).
> **1.5B cag (verbatim, trimmed — truncated at 512):**
> 1. **Boire et se laver** …  2. **Cuisine** …  3. **Arrosage des cultures** … irriguer les cultures …  4. **Pêche** …  5. **Transport** …  6. **Abreuvement des animaux** …  7. **Préparation des festivités** : … des festivités comme la Korité ou la Tabaski.  8. **Hygiène** …  9.–11. [restate transport / irrigation / festivities]

Surfaces **most** of the expected activities — drinking, washing, cooking, irrigation, fishing, watering livestock, transport, and crucially the **festive/religious uses (Korité, Tabaski)** that the 0.5B never reached. It muddles some source attributions (e.g. "transport" tied to wells) and items 9–11 restate earlier ones, and it truncates at 512. But it is a genuine multi-fiche synthesis, not the 0.5B's 2–3 items followed by an infinite "abreuvoir des plantes" loop. **Much better than 0.5B.**

### 8.8 Q8 — aggregate coastal/Senegalese towns — *Good*

> **Expected:** Dakar, Mbour, Saint-Louis, Ziguinchor, Gorée, Ngor, Rufisque.
> **1.5B cag (verbatim, trimmed):**
> 1. Dakar  2. Saint-Louis  3. Gorée  4. Ngor  5. Mbour … [then attributes them to fiches 6–10]

Recovers **5 of 7** real towns (Dakar, Saint-Louis, Gorée, Ngor, Mbour) with **no hallucinated town** and **no repetition loop**. Misses Ziguinchor and Rufisque, and its fiche attribution (generic "Fiche 6–10") is wrong. Compare the 0.5B: 4/7 plus an invented "Saint-Denis", "Dakar (Côte d'Ivoire)", and a loop on "Dakar". **Better than 0.5B.**

### Quality scorecard — 1.5B vs 0.5B on the same questions

| Q | Type | 1.5B verdict | 0.5B verdict | Better? |
|---|---|---|---|---|
| 1 | aggregate | Partial — 3/7 named, clean stop, no loop/hallucination | Partial — 3/7 + hallucinated place + loop | **Yes** |
| 2 | count | Fail — count 2 (expected 6); no denial, no out-of-range | Fail — denies corpus, invents idx 39–42 | Yes (cleaner fail) |
| 3 | locate | **Correct** — "fiche 9" = idx 37, + uniqueness | Good — title only, no index | **Yes** |
| 4 | locate | Good — correctly picks richest fiche (32) + its rivers | Partial — rivers named, no fiche | **Yes** |
| 5 | compare | Fail — one side only, redundant; **no "salée" error** | Fail — calls river water "salée" | Slightly (no salinity error) |
| 6 | compare | **Correct** — both families listed + right opposition | Fail — same wrong desc both, inverted | **Yes (decisive)** |
| 7 | synthesis | Good — broad synthesis incl. festivals; minor redundancy, truncated | Partial→Fail — 2–3 items then loop | **Yes** |
| 8 | aggregate | Good — 5/7 towns, no loop, no hallucinated town | Partial — 4/7 + hallucinations + loop | **Yes** |

**Diagnosis.** The 0.5B's signature failure mode — *partial recall followed by a repetition loop that eats the token budget*, plus outright factual errors (river water "salée") and denial/invention of fiches — is **gone** at 1.5B. The 1.5B produces concise, complete, `stop`-terminated answers (7/8 finish naturally), grounds itself correctly, resolves positional fiche indices, and delivers one fully correct cross-fiche comparison (Q6). The single operation still beyond it is **cross-document counting** (Q2: 2 vs 6). Aggregation is now "partial-but-clean" rather than "partial-then-looping", and comparison/synthesis are genuinely usable.

---

## 9. Conclusion

**CAG succeeds identically on both models.** It turns a 21k-token cached corpus from economically impossible (**~19 min of re-prefill per question**, ~3 h across 8) into merely slow (**~15 s prefill per question**). Prefill drops **~76×**; total wall-clock drops **~6.8×**; the 8-question run falls from **~3 hours (naive) to ~26.5 minutes (cag)**. `cache_n = 20934` on every call confirms faithful delivery of all ten fiches.

**The decode floor is higher than the 0.5B.** Attending over a 572 MB / 21k-token KV cache on a heavier model drops decode to **~1.2 tok/s** (vs the 0.5B's ~2.9), so ~183 s of decode per question sets the floor. CAG cannot fix this — the decode math is identical for naive and cag.

**Does 3× the params fix cross-fiche reasoning? Yes, on 7 of 8 questions.** No repetition loops, no salinity error, no corpus denial, correct positional indices (Q3/Q4), and a fully correct family comparison (Q6). The lone residual failure is cross-document **counting** (Q2), where the count is wrong but the behaviour is otherwise sane.

**Is it worth the extra cost?** The 1.5B costs **2.3× the KV cache** (572 vs 245 MB) and **~1.6× the cag wall-clock** (~3.3 min vs ~2 min per question). For that, the answers go from *largely unusable* (partial-then-loop, wrong facts) to *largely usable* (concise, complete, mostly correct cross-fiche reasoning). On this corpus and these questions the trade is clearly worth it: **the 0.5B proved CAG makes a 21k-token corpus runnable on a Pi; the 1.5B is the first model in this series where the answers are good enough to actually use.**

**Implications for the next run:**
- **Counting** is the remaining capability gap — neither 0.5B nor 1.5B can count occurrences across ten concatenated documents. If exact cross-document counts matter, a larger model or a tool/regex step is needed.
- **Decode is the wall.** At ~1.2 tok/s the 1.5B is ~3.3 min/question; anything larger will be quality-up, speed-down. The 572 MB cache is near the practical ceiling for a Pi.
- **`max_tokens` 512** truncated only Q7 (vs 4/8 on the 0.5B); the 1.5B's concise complete answers mean a modest cap increase would recover the one truncated synthesis without inviting loops.
