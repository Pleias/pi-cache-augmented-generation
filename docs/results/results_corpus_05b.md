# CAG on Qwen2.5-0.5B-Instruct — Multi-Fiche Corpus Run (idx 29–38)

**Corpus:** the *water* cluster — TEN CE1 geography fiches (`fiche_indices` 29–38) concatenated into ONE context and cached as a single prefix.
**Model:** Qwen2.5-0.5B-Instruct, Q8_0 quant, context-first ChatML (corpus placed in the system message).
**Hardware:** Raspberry Pi. Temperature 0, `max_tokens` 512.
**Runner:** `bench/cag_bench.py --model-format chatml --corpus 29-38`.
**Results file:** `bench/results/corpus29-38_corpus-05b-full_20260704-215342.json`
**Questions:** `bench/questions_corpus_water.json` (8 cross-fiche questions).

This is the **multi-chunk companion** to `docs/results_qwen_full.md` (single-fiche, same model). The single-fiche run cached ~2–3k tokens of one fiche and asked questions answerable from that one fiche. Here the cached context is an order of magnitude larger — **20,934 tokens** of ten fiches — and every question is deliberately **cross-fiche**: it can only be answered by reading across several fiches at once, which is impossible from any single chunk. The point is to stress two things at once: (a) does CAG make a 21k-token cached context economically usable on a Pi, and (b) can a 0.5B model actually reason across ten concatenated documents.

The short answers: (a) **yes, dramatically** — CAG removes ~7.5 minutes of re-prefill per question; (b) **no, mostly not** — cross-fiche aggregation, counting and comparison are at or beyond this model's ceiling.

---

## 1. What is being compared

Each of 8 questions is answered under three **modes**:

- **`nocontext`** — no corpus in the prompt. Quality/leak baseline: what does the model say with no grounding?
- **`naive`** — the full 20,934-token corpus is re-prefilled from scratch for *every* question. Correct, but pays the enormous corpus prefill cost 8 times.
- **`cag`** — the corpus KV cache is computed **once** and reused as a prefix for all 8 questions. Only the (short) question tokens are prefilled per call.

CAG's promise: `naive` and `cag` feed the model the *same* tokens and should produce the *same* answer, but `cag` amortizes the corpus prefill. Decode work is identical between them.

**KV cache footprint.** ~12 KB/token for this model (Qwen 0.5B: 24 layers, 2 KV heads). The cached corpus is 20,934 tokens → **245.3 MB** of KV cache. Every CAG row reports `cache_n = 20934`, confirming the entire corpus prefix is reused on each call.

**The 8 questions** (all `leak_risk: LOW`, all require reading multiple fiches):

| Q | Type | Asks | Requires fiches |
|---|---|---|---|
| 1 | aggregate | List all named water bodies (rivers, lakes, ocean…) in the corpus | 30,31,32,33,37 |
| 2 | count | In how many fiches does the word « forage » appear, and which? (expected: **6** — 29,30,31,34,36,38) | 29,30,31,34,36,38 |
| 3 | locate | Which single fiche introduces the water cycle, and confirm the others don't (expected: fiche 37) | all 10 |
| 4 | locate | Which fiches locate the Sénégal river on a map, and which one adds the other rivers (expected: 30,32,37; richest = 32) | 30,32,37 |
| 5 | compare | Contrast the sea/ocean fiche vs the stream/river fiche (salt vs fresh, size, uses) | 33,35 |
| 6 | compare | Contrast "points d'eau" vs "cours d'eau" families and list the members of each | 29,32 |
| 7 | synthesis | Synthesize all human activities depending on water across the corpus | 30,33,36,38 |
| 8 | aggregate | List all coastal/Senegalese towns cited about water, and in which fiches | 32,33,36,37 |

---

## 2. Latency summary

Totals over the 8 questions, and per-question (`/Q`) averages, in milliseconds. "Prefill" = prompt/context processing, "Decode" = answer generation, "Total" = wall-clock.

| Mode | Prefill total | Decode total | Total | Prefill /Q | Decode /Q | Total /Q | Cached KV |
|---|---:|---:|---:|---:|---:|---:|---:|
| nocontext | 3,184 | 89,964 | 93,220 | 398 | 11,246 | 11,653 | — |
| naive | 3,621,769 | 1,198,207 | 4,820,435 | 452,721 | 149,776 | 602,554 | — |
| **cag** | **55,518** | **944,866** | **1,000,839** | **6,940** | **118,108** | **125,105** | **245.3 MB** |

Reading the `/Q` column in human units:

| Mode | Prefill /Q | Decode /Q | Total /Q |
|---|---|---|---|
| nocontext | 0.4 s | 11.2 s | **11.7 s** (but cannot answer — no corpus) |
| naive | **452.7 s (~7.5 min)** | 149.8 s (~2.5 min) | **602.6 s (~10 min)** |
| cag | **6.9 s (~7 s)** | 118.1 s (~2 min) | **125.1 s (~2 min)** |

---

## 3. Prefill savings — the CAG effect at 21k tokens

This is where CAG works exactly as intended, and the effect is far larger than in the single-fiche run.

- **naive** re-prefills the whole 20,934-token corpus on *every* question: **452,721 ms /Q — roughly 7.5 minutes of pure prompt processing before a single output token.**
- **cag** reuses the cached prefix: **6,940 ms /Q — about 7 seconds** (only the short question tokens are processed).
- **Prefill reduction: ~65×** (452,721 / 6,940 ≈ 65.2). The `cache_n = 20934` on every CAG row confirms the entire corpus prefix is being reused, not recomputed.

**Why is naive so brutal?** Long-context prefill is much slower *per token* than short-context prefill. Prefill throughput here is only:

> 20,934 tokens / 452.7 s ≈ **46 tok/s**

This is a fraction of the throughput a 0.5B model reaches on a short prompt. Attention is quadratic in sequence length and a 245 MB KV working set thrashes memory on a Pi, so each of the 21k tokens is expensive to process. Paying that 46-tok/s bill from scratch on all 8 questions is what makes naive cost ~60 minutes of prefill in aggregate (3,621,769 ms total). CAG pays it effectively once.

---

## 4. But decode is now the bottleneck — and it's slow

CAG kills prefill, but it does **not** help decode, and at 21k tokens decode is where nearly all the remaining time goes.

- cag decode /Q ≈ **118 s**; naive decode /Q ≈ **150 s**. Decode dominates the cag total completely: **118 s of the 125 s /Q total is decode; prefill is only 7 s.**
- Decode throughput over the 245 MB / 21k-token cache collapses to:

> cag: 2,701 generated tokens / 944.9 s ≈ **2.9 tok/s**
> naive: 3,414 generated tokens / 1,198.2 s ≈ **2.85 tok/s**

Compare that to **~20 tok/s** for the same 0.5B model generating over the empty `nocontext` prompt (1,831 tokens / 90.0 s ≈ 20.3 tok/s). Every generated token must attend over the entire cached context, so a big KV cache slows down *each* decode step. This is a real, unavoidable cost of grounding on 21k tokens — it is **not** a CAG failure, and CAG cannot fix it (the decode math is identical for naive and cag, as the ~2.85 vs ~2.86 tok/s match confirms).

---

## 5. Total speedup — and why it's 4.8×, not 65×

| Comparison | naive /Q | cag /Q | Speedup |
|---|---:|---:|---:|
| Prefill only | 452,721 ms | 6,940 ms | **~65×** |
| **Total wall-clock** | 602,554 ms | 125,105 ms | **~4.8×** |

The 65× prefill win shrinks to a 4.8× total win because **decode dominates the total and CAG doesn't touch decode.** In `cag`, prefill is 7 s and decode is 118 s — so even after eliminating 445 s of prefill, the ~118 s of big-context decode remains and sets the floor. The gap between 65× and 4.8× is precisely the decode-dominance effect: CAG removed the part it could remove (prefill), and what's left is the intrinsic cost of decoding over a large cached context.

Still, going from **~10 minutes per question to ~2 minutes per question** is the difference between "unusable" and "usable" for a 21k-token corpus on a Pi.

---

## 6. nocontext: fast but useless

`nocontext` looks great on the clock — **11.7 s /Q total**, ~20 tok/s decode — because there is no corpus in the KV cache to attend over. But it structurally **cannot** answer any cross-fiche question: with no fiches in context, it either invents or gives generic categories. Two illustrations:

> **Q1 (list named water bodies), nocontext answer (verbatim, complete — 26 tokens):**
> 1. Fleuve  2. Rivière  3. Lac  4. Mares  5. Océan

These are generic categories, not the *named* water bodies the question asks for (Fleuve Sénégal, Lac Rose, mare de Warang, …) — exactly the note's warning that the categories are not proper names.

> **Q2 (how many fiches mention « forage »), nocontext answer (verbatim):**
> Le terme "forage" apparaît 1 fois dans le corpus, dans le texte "Le forage est un acte de forage".

Fabricated (expected: 6 fiches). The ~118 s cag decode is the *price of grounding* on the real 21k-token corpus; nocontext avoids that price only by refusing to actually read anything.

---

## 7. Amortization — the cold-start caveat

The numbers above understate the true CAG advantage, because of how the two modes amortize the corpus prefill:

- **naive** pays the full ~452 s corpus prefill on **every** question. Over 8 questions that is **~60 minutes of prefill alone** (3,621,769 ms total prefill).
- **True CAG** should pay the ~452 s corpus prefill **once** (the first question warms the cache), then ~0 s prefill for the remaining 7 questions. Over 8 questions the corpus prefill is a single fixed cost, not a per-question tax.

**Caveat specific to this run:** in *this* JSON the cag prefill was already only ~7 s on the *first* question, not ~452 s. That is because `naive` ran before `cag` and had already warmed the model's context slot / KV state for the corpus, so cag never paid the cold 452 s. In a clean cold-start run, question 1 of cag would cost ~452 s prefill and questions 2–8 ~7 s each. Either way the accounting is the same: **naive ≈ 80 min total for 8 questions; cag ≈ 17 min total** (1,000,839 ms ≈ 16.7 min), and the corpus prefill is amortized to near-zero across questions.

---

## 8. Truncation caveat

`max_tokens` was **512**, and the long aggregate/synthesis answers wanted more than that. Truncations (`finish_reason = length`):

| Mode | Truncated /8 |
|---|---|
| nocontext | 3 (Q3, Q7, Q8) |
| naive | 6 |
| cag | 4 (Q1, Q2, Q7, Q8) |

The truncated answers are the aggregate/count/synthesis ones (Q1, Q2, Q7, Q8) — and, as Section 9 shows, they degenerate into repetition loops that burn the full 512-token budget. This inflates the decode totals: a rerun with a higher `max_tokens` would let good answers finish sooner but also let the repetition loops run longer, so the decode figures here should be read as budget-capped, not as natural answer lengths.

---

## 9. Quality — can a 0.5B reason across ten fiches?

This is the honest part. CAG's job is to deliver the full corpus faithfully into the model's context — and it does: `cache_n = 20934` proves all ten fiches are present. The failures below are **model-capability failures, not CAG failures.** A 0.5B model is being asked to aggregate, count and compare across 10 concatenated documents, and it is at or past its ceiling.

### 9.1 Aggregate (Q1) — degenerates into a repetition loop

> **Q1 — expected:** Fleuve Sénégal, Fleuve Gambie, Saloum, Casamance, Océan Atlantique, Lac Rose, mare de Warang.
> **cag answer (verbatim, trimmed):**
> En parcourant l'ENSEMBLE des fiches, je trouve les cours d'eau suivants :
> 1. Fleuve Sénégal  2. Océan Atlantique  3. Lac Rose  4. Marigot  5. Lac de Kourou
> 6. Lac de Kourou (ou Lac de Kourou)  7. Lac de Kourou (ou Lac de Kourou)  … [repeats "Lac de Kourou" to the 512-token limit]

It correctly recovers **3 of 7** real entities (Fleuve Sénégal, Océan Atlantique, Lac Rose) — a genuine improvement over nocontext's generic categories, which shows the grounding is working. But it then hallucinates "Lac de Kourou" (not in the corpus) and collapses into an infinite repetition loop, missing Gambie, Saloum, Casamance and the mare de Warang. Partial aggregation, no completeness.

### 9.2 Count (Q2) — fails outright

> **Q2 — expected:** « forage » appears in **6** fiches: 29, 30, 31, 34, 36, 38.
> **cag answer (verbatim, trimmed):**
> En tant qu'assistant pédagogique, je n'ai pas accès à un corpus complet de fiches du corpus. Cependant, je peux vous donner une liste générale des termes clés… — Géographie … (idx 30) … (idx 37) … (idx 38) … (idx 39) … (idx 40) … [invents idx 39–42, repeats to limit]

Complete failure. The model both denies having the corpus (which it *does* have — 20,934 cached tokens) and then invents fiche indices beyond the range (39–42). It never counts "forage" at all. Cross-document counting is exactly the operation a 0.5B cannot do; the answer is worse than nocontext's confident-but-wrong "1 fois". (For reference, `naive` Q3 also mis-numbered the water-cycle fiche as "fiche 9", another sign the model cannot track fiche indices across the concatenation.)

### 9.3 Locate (Q3, Q4) — the best results, roughly correct

The two `locate` questions are the model's best showing, and notably both finished cleanly (`finish_reason = stop`, not truncated):

> **Q3 — expected:** fiche 37 introduces the water cycle; no other fiche does.
> **cag answer (verbatim):**
> La fiche "Géographie - Découverte des cours d'eau et initiation au cycle de l'eau" introduit la notion de cycle de l'eau. Cependant, les autres fiches ne mentionnent pas explicitement ce cycle. Les autres fiches mentionnent généralement des cours d'eau comme puits, forages, bassins de rétention, mares, etc.

Correct on substance: it identifies the water-cycle fiche by its exact title and asserts the others don't cover it. It doesn't give the numeric index "37" and its second sentence muddles the "points d'eau" vs "cours d'eau" vocabulary, but the core claim is right.

> **Q4 — expected:** fiches 30, 32, 37 locate the Sénégal river; fiche 32 also adds Gambie/Saloum/Casamance.
> **cag answer (verbatim):**
> Pour localiser le Fleuve Sénégal sur la carte du Sénégal, les élèves ont été fournis avec une carte murale du Sénégal. Cela permet de visualiser et localiser le Fleuve Sénégal, ainsi que d'autres cours d'eau du pays comme le Fleuve Gambie, le Saloum et le Casamance.

It correctly names the additional rivers (Gambie, Saloum, Casamance) that belong to the richest fiche (32) — good cross-fiche recall of *content* — but it never answers the actual question (*which fiches?*): no indices, no comparison of completeness. Right facts, wrong task.

### 9.4 Compare / Synthesis (Q5, Q6, Q7) — confused or looping

> **Q5 (sea/ocean vs stream/river) — expected:** salt vs fresh water, immensity vs relative size, fishing/transport/tourism vs pirogues/fishing.
> **cag answer (verbatim, trimmed):**
> La fiche consacrée à la mer et à l'océan présente l'eau … en disant que l'eau est salée … La fiche consacrée au ruisseau et à la rivière … souligne que l'eau est **salée**, que les usages sont principalement agricoles …

It gets one fact (sea = salty) then wrongly says the *stream/river* water is also "salée" and blurs both fiches into the same generic description. The essential contrast (salt vs **fresh**) is lost — the comparison collapses.

> **Q6 (points d'eau vs cours d'eau):** the answer produces a clean two-part structure but fills both parts with the *same* wrong description ("cours d'eau naturels, souvent creusés par l'homme" for both), inverting the actual distinction (still water you *find* vs water that *flows*). Right format, wrong content.

> **Q7 (synthesis of water-dependent activities) — expected:** drinking, cooking, washing, irrigation, watering livestock, fishing, transport/commerce, tourism, religious festivals (Tabaski/Korité/Ndogou).
> **cag answer (verbatim, trimmed):**
> 1. … l'agriculture … irrigation …  2. … abreuvoir des animaux …  3. … abreuvoir des plantes …  4. [repeats item 2]  5. [repeats item 3]  6. [repeats item 3]  7. [repeats] … [loops to 512-token limit]

It surfaces 2–3 real activities (irrigation, watering animals) then loops on "abreuvoir des plantes" until truncation. It never reaches transport, tourism, fishing, or the festive/religious uses. No true synthesis.

### 9.5 Aggregate (Q8, towns) — same failure mode as Q1

> **Q8 — expected:** Dakar, Mbour, Saint-Louis, Ziguinchor, Gorée, Ngor, Rufisque.
> **cag answer (verbatim, trimmed):**
> 1. Dakar (Côte d'Ivoire)  2. Rufisque (Sénégal)  3. Mbour (Sénégal)  4. Saint-Louis (Sénégal)  5. Saint-Denis (Sénégal)  6. Dakar (Sénégal)  7. Dakar (Sénégal) … [repeats "Dakar" to the 512-token limit]

Recovers **4 of 7** real towns (Dakar, Rufisque, Mbour, Saint-Louis) — again showing grounding works — but mislabels Dakar as "Côte d'Ivoire", invents "Saint-Denis", misses Ziguinchor/Gorée/Ngor, and loops on "Dakar". Same pattern as Q1: partial recall, hallucinated extras, repetition collapse.

### Quality scorecard

| Q | Type | Verdict | Notes |
|---|---|---|---|
| 1 | aggregate | Partial | 3/7 entities; hallucinated "Lac de Kourou"; repetition loop |
| 2 | count | Fail | Denies having corpus; invents indices; never counts (expected 6) |
| 3 | locate | Good | Identifies the water-cycle fiche by title; no numeric index |
| 4 | locate | Partial | Correct rivers, but never says *which fiches* (the actual question) |
| 5 | compare | Fail | Calls river water "salée"; loses salt-vs-fresh contrast |
| 6 | compare | Fail | Same wrong description for both families; inverts the distinction |
| 7 | synthesis | Partial→Fail | 2–3 activities then loops; misses transport/tourism/festivals |
| 8 | aggregate | Partial | 4/7 towns; hallucinated extras; repetition loop |

**Diagnosis.** The consistent failure mode on aggregate/count/synthesis is *partial recall followed by a repetition loop that eats the token budget.* This is characteristic of a very small model overwhelmed by a long context: it can pull a few relevant entities out of the corpus (so the cache is clearly being read), but it cannot enumerate exhaustively, count, or hold a two-way comparison in mind, and it lacks the self-consistency to stop. The `locate` questions fare best because they need one fact, not aggregation.

---

## 10. Conclusion

**CAG succeeds at its job.** It turns a 21k-token cached corpus from economically impossible (~7.5 min of re-prefill *per question*, ~60 min across 8) into merely slow (~7 s prefill per question after one warm-up). Prefill drops ~65×; total wall-clock drops ~4.8×; the whole 8-question run falls from ~80 min (naive) to ~17 min (cag). The `cache_n = 20934` on every call confirms the full corpus is delivered faithfully into context.

**Two hard limits remain, and neither is CAG's fault:**

1. **Big-context decode is the new bottleneck on a Pi.** Attending over a 245 MB / 21k-token KV cache drops decode to ~2.9 tok/s (vs ~20 tok/s on an empty context). CAG cannot help here — the decode math is identical for naive and cag — so ~118 s of decode per question sets the floor. This is the intrinsic price of grounding on 21k tokens.

2. **Cross-fiche reasoning is at or beyond the 0.5B's ceiling.** The corpus is fully present, but the model can only partially aggregate, cannot count across documents, and collapses comparisons and syntheses into repetition loops. The best it manages is single-fact `locate` questions.

**Implications for the next run:**
- **Bigger model.** Cross-document aggregation/counting/synthesis needs more capacity than 0.5B; the 1.5B/1.7B companions are the natural next comparison for whether quality clears the bar (decode will be even slower, though).
- **Higher `max_tokens`.** 512 truncated 4/8 cag answers, but note the truncated ones are mostly repetition loops — raising the cap alone won't fix quality, it will just let good answers finish and bad ones loop longer.
- **The headline stands:** CAG is what makes a 21k-token corpus *usable at all* on this hardware. Whether the *answers* are usable is a separate question about model size, and on a 0.5B the answer is largely no.
