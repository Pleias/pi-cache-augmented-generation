# CAG on Qwen2.5-1.5B-Instruct — Full 9-Question Run (single fiche)

**Fiche:** `#E2-CE1-ESVS-GEO-P2-2026-02-17-003` — *"Les points d'eau de mon milieu (naturels et artificiels)"* (CE1 geography, 30-minute lesson).
**Model:** Qwen2.5-1.5B-Instruct, context-first ChatML (fiche placed in the system message).
**Hardware:** Raspberry Pi.
**Runner:** `bench/cag_bench.py --model-format chatml`, temperature 0, seed 42, `max_tokens` 1024.
**Results file:** `bench/results/entry31_qwen25-1.5b-full_20260705-001832.json`
**Questions:** `bench/questions_entry31.json`

This is the **single-fiche** full run: the cached context is one lesson sheet of ~1.9–3.1k tokens ("small chunks"), as opposed to the separate 21k-token corpus run. It is the direct 1.5B counterpart to the 0.5B run documented in `docs/results_qwen_full.md`, and the two are compared throughout (§4–§5). The question is not only "does CAG work" (it does, and identically to the 0.5B in mechanics) but **"is the larger 1.5B worth its ~2× wall-clock cost at single-fiche scale?"**

---

## 1. What is being compared

Each of 9 questions is answered under three **modes**:

- **`nocontext`** — no fiche in the prompt. Quality/leak baseline.
- **`naive`** — the full fiche is re-prefilled from scratch for *every* question. Correct, but pays the fiche prefill cost 9 times.
- **`cag`** — the fiche's KV cache is computed **once** and reused as a prefix for all 9 questions. Only the short question tokens are prefilled per call.

`naive` and `cag` feed the model identical tokens, so they produce equivalent answers; `cag` only amortizes the fiche prefill. Decode work is untouched.

Three **context variants** control how much fiche is cached:

| Variant | Cached tokens (`cache_n`) | KV cache size |
|---|---|---|
| `answer` (fiche body only) | 1919 | 52.5 MB |
| `answer_exercises` (body + exercises + corrigé) | 2418 | 66.1 MB |
| `full` (entire fiche) | 3143 | 85.9 MB |

KV footprint is **~27 KB/token** for this model — roughly 2.3× the 0.5B's ~12 KB/token, the price of the larger model's deeper attention stack. `cache_n` matches the fiche token count on every CAG call (1909–1919 for `answer`, up to 3143 for `full`), confirming the whole fiche prefix is genuinely reused.

**Questions:** 9 total — 4 recall (Q1–Q4), 2 corrigé (Q5–Q6), 3 utility/generation (Q7–Q9).

---

## 2. Latency summary

Per-question averages over the 9 questions (milliseconds). "Prefill" = prompt processing, "Decode" = answer generation, "Total" = wall-clock.

| Variant | Mode | Prefill (ms) | Decode (ms) | Total (ms) | Cached KV |
|---|---|---:|---:|---:|---:|
| **answer** (1919 tok) | nocontext | 1872 | 15359 | 17241 | — |
| | naive | 51499 | 27853 | 79366 | — |
| | **cag** | **1741** | 38850 | **40604** | 52.5 MB |
| **answer_exercises** (2418 tok) | nocontext | 1872 | 15536 | 17421 | — |
| | naive | 66763 | 40426 | 107205 | — |
| | **cag** | **2231** | 47570 | **49815** | 66.1 MB |
| **full** (3143 tok) | nocontext | 1876 | 15670 | 17559 | — |
| | naive | 90308 | 47165 | 137489 | — |
| | **cag** | **2743** | 39616 | **42376** | 85.9 MB |

---

## 3. Prefill savings — the CAG effect

This is where CAG works exactly as intended, and on the 1.5B the effect is even sharper than on the 0.5B.

| Variant | naive prefill | cag prefill | Reduction |
|---|---:|---:|---:|
| answer | 51499 ms | 1741 ms | **~30×** (29.6×) |
| answer_exercises | 66763 ms | 2231 ms | **~30×** (29.9×) |
| full | 90308 ms | 2743 ms | **~33×** (32.9×) |

1. **The saving grows with context size** — 30× for the 1919-token `answer` fiche up to 33× for the 3143-token `full` fiche. Under `naive`, prefill scales roughly linearly with fiche length (51.5 s → 66.8 s → 90.3 s). Under `cag` it stays near-flat (1.7 s → 2.2 s → 2.7 s): only the short question is processed each call, the small growth being cache-reattach overhead over a longer prefix.
2. **The whole fiche prefix is reused.** `cache_n` equals the fiche token count on every CAG call — no re-prefill of the fiche happens.

**CAG kills the prefill cost. That is the headline, and it is bigger on the 1.5B (~30–33×) than on the 0.5B (~20–25×)** — because the larger model's absolute prefill is much heavier (e.g. `full` naive prefill 90.3 s vs the 0.5B's 20.4 s), so removing it saves proportionally more.

---

## 4. The honest caveat — decode dominates, and CAG does not touch it

CAG saves prefill, **not decode**. Decode is identical whether the fiche was cached or freshly prefilled — same model, same tokens, same output. On the 1.5B, decode is not just a component of total time, it is **the overwhelming majority of CAG wall-clock**:

- `answer`/cag: decode 38850 ms of 40604 ms total → prefill is only ~4% of wall-clock.
- `full`/cag: decode 39616 ms of 42376 ms total → prefill ~6%.

The 1.5B decodes at roughly **~5 tok/s** on this hardware, and the utility questions (Q7–Q9) generate long answers (hundreds of tokens; Q8/Q9 exceed 700–840 predicted tokens), so CAG per-Q decode averages land at **~38–48 s**. Because decode is untouched and dominates, the total-time speedup is far smaller than the 30× prefill number:

| Variant | naive total | cag total | Wall-clock speedup |
|---|---:|---:|---:|
| answer | 79366 ms | 40604 ms | **2.0×** (1.95×) |
| answer_exercises | 107205 ms | 49815 ms | **2.2×** (2.15×) |
| full | 137489 ms | 42376 ms | **3.2×** (3.24×) |

The prefill reduction is ~30×, but real wall-clock lands at ~2–3.2×. This is the same shape as the 0.5B (1.4× / 2.5× / 3.1×) — the 1.5B's slightly bigger prefill savings buy a marginally better speedup on `answer`/`answer_exercises`, but the pattern is identical: **CAG's payoff is proportional to how much of your latency is prefill, and on these small models generating multi-sentence answers, decode dominates.**

**`nocontext` is the fastest mode (~17 s total) but it is ungrounded** (see §5.4) — its low total comes purely from having no fiche to prefill (~1.9 s prompt) while still paying ~15.4 s of decode. Strip out prefill entirely and you are left with pure generation cost; that is the floor CAG asymptotes toward.

### Cross-model wall-clock: 1.5B costs ~2× the 0.5B

| Variant / cag | 0.5B total | 1.5B total | 1.5B is |
|---|---:|---:|---:|
| answer | 20839 ms | 40604 ms | ~1.9× slower |
| answer_exercises | 11779 ms | 49815 ms | ~4.2× slower |
| full | 10219 ms | 42376 ms | ~4.1× slower |

The 0.5B is faster on every variant (faster decode, lower totals). Averaging across variants the 1.5B is roughly **~2× slower wall-clock** on `answer` and considerably more where the 1.5B chose to generate much longer answers. So the trade is explicit: **the 1.5B buys quality (§5) at roughly 2×+ the wall-clock of the 0.5B.**

---

## 5. Answer quality — the headline

Temperature 0. `naive` and `cag` are equivalent by construction, so grounded examples below are from `cag`. Answers are quoted verbatim, lightly trimmed.

### 5.1 Recall (Q1–Q3) — grounding works, parity with 0.5B

**Q1 (mare de Warang)** — expected: *a pond that fills after the rains; natural because it exists without direct human intervention.*

> `answer`/cag: "La mare de Warang est considérée comme un point d'eau naturel car elle se remplit après les pluies, ce qui indique qu'elle n'est pas créée par l'intervention humaine."

Correct and grounded — recovers the fiche-specific fact ("se remplit après les pluies"). Same outcome as 0.5B.

**Q2 (forage de Mbodiène)** — expected: *a borehole dug so the whole village has drinking water; artificial because dug by humans.*

> `answer`/cag: "Le forage du village de Mbodiène est un point d'eau artificiel car il a été creusé par l'homme pour fournir de l'eau potable à tout le village."

Correct — and notably tighter than the 0.5B, capturing "tout le village" explicitly. Parity/slight edge.

**Q3 (salty, coloured lake)** — expected: *Le Lac Rose.*

> `answer`/cag: "Le grand point d'eau naturel cité comme exemple est le Lac Rose, connu pour son sel et sa couleur unique."

Correct in all three variants. (0.5B also correct.)

### 5.2 Q4 (verbatim recall) — 1.5B fixes it, but only in the `answer_exercises` variant

Expected the literal instruction: *« Écris N si c'est naturel, A si c'est artificiel : forage, mare, lac, puits. »*

- `answer`/cag: *"…la consigne exacte donnée aux élèves est de dictée de mots ou de petites questions sur l'ardoise."* → **loose paraphrase**, same failure mode as the 0.5B.
- `answer_exercises`/cag:

> "La consigne donnée aux élèves lors de la phase d'évaluation sur l'ardoise […] est la suivante : \"Dictée de mots ou petites questions sur l'ardoise : 'Écris N si c'est naturel, A si c'est artificiel : forage, mare, lac, puits.' Vérifie les ardoises et note les difficultés.\""

**Verbatim correct.** The 1.5B reproduces the exact "Écris N … A …" instruction that the 0.5B never recovered in any variant.
- `full`/cag: regresses to *"…est de répondre à une question sur leur ardoise"* — a generic paraphrase again.

**Verdict:** the 1.5B *can* do the verbatim recall the 0.5B could not — but only when the exercises section is cached (which repeats the N/A format and appears to anchor the exact wording). The `answer`-only and `full` variants still paraphrase. **Genuine improvement, but variant-dependent and not robust.**

### 5.3 Corrigé (Q5–Q6) — the "do exercises help?" axis, and the Q6 win

**Q5 (Lac de Guiers: N or A?)** — expected: **N (Naturel)** — created by nature.

- `answer`/cag: *"…le lac de Guiers est un point d'eau naturel (N) […] créé par la nature, sans l'intervention directe de l'homme."* — right label, guessed (the corrigé is not in this variant).
- `answer_exercises`/cag: *"…un point d'eau naturel (N). Il explique que c'est la nature qui l'a créé…"* — right label, correct justification.
- `full`/cag:

> "Le corrigé pour « d) Le lac de Guiers » est : N (Naturel) - C'est la nature qui l'a créé."

**Verbatim match to the expected corrigé.** Contrast the 0.5B, whose `full` variant gave the right label but a *self-contradictory* justification ("created by nature" then "humans created and developed this lake"). The 1.5B's reasoning is clean. As on the 0.5B, the label is guessable from the word "lac" so `answer`-only does not cleanly *fail*, but the 1.5B additionally reproduces the corrigé correctly once it is cached — **improvement in justification quality.**

**Q6 (Vrai/Faux: "a river is artificial because humans navigate on it")** — expected: **FAUX** — a river is natural; human use does not change its origin. *This was the 0.5B's clearest miss:* it answered "l'affirmation est vraie" (wrong) and self-contradicted in **all three variants**.

- `answer`/cag: *"La réponse est fausse. […] le fleuve est un point d'eau naturel car il se forme naturellement par la dépression de l'océan et la sédimentation."* → **label FAUX is correct**, but the justification is fabricated (the corrigé is not in this variant).
- `answer_exercises`/cag:

> "L'affirmation \"a) Un fleuve est un point d'eau artificiel car l'homme navigue dessus\" est fausse selon le corrigé. La justification donnée est que le fait que l'homme navigue dessus ne change pas sa nature d'origine, c'est-à-dire qu'il est un point d'eau naturel."

- `full`/cag: **identical, fully correct** wording as above.

**The 1.5B gets Q6 RIGHT.** It answers FAUX in every variant (fixing the 0.5B's headline error), and with the exercises/corrigé cached it produces the *exact* expected justification ("ne change pas sa nature d'origine"). No self-contradiction. **This is the single biggest quality win of the 1.5B over the 0.5B**, and here the corrigé variants deliver the clean "answer-only gives right label but weak reasoning / exercises gives right label + correct justification" separation the design targeted.

### 5.4 Utility / generation (Q7–Q9) — more grounded, but classification is sloppy

**Q7 (propose a classification exercise for the ardoise)** — open-ended; expected elements: local Senegalese examples, N/A format, CE1-appropriate.

- `answer_exercises`/cag aligns tightly with the fiche's own exercise format:

> "1. a) Le puits de la mosquée 2. b) La mare après la pluie 3. c) Le forage du quartier 4. d) Le lac de Guiers 5. e) Un bassin pour les fleurs à la maison. Pour chaque point d'eau, écris N pour naturel ou A pour artificiel."

This reuses the fiche's exact items (lac de Guiers, bassin pour les fleurs) and the N/A instruction — well grounded.
- `answer`/cag is grounded in local proper nouns (Mare de Warang, Lac Rose, Forage de Mbodiène) **but misclassifies**: it lists "Puits du village de Mbodiène" under **Naturel** (a well is artificial) and puts "Puits" in both columns.
- `full`/cag degrades further: it repeats an item ("Le lac de la forêt" listed twice) and again slots "Le puits de la maison" under Naturel.

**Verdict:** the 1.5B pulls in more fiche-specific local examples than the generic 0.5B output, but its own classification is internally inconsistent in the `answer` and `full` variants. Better grounding, sloppier logic when generating long lists.

**Q8 (5-minute oral warm-up)** — the 1.5B produces a titled, structured activity with a trigger question rooted in the child's daily water-fetching experience (matching the expected elements). The `answer_exercises` variant **truncated** at 1024 tokens mid-activity (see §6).

**Q9 (memorization game)** — the `full`/cag answer explicitly names the fiche's local examples and materials:

> "**Objets locaux** : Utilisez des miniatures de puits, forages, bassins de rétention, mare, lac, marigot. **Photos/Dessins** : Utilisez des images de ces points d'eau dans le Sénégal."

This hits the expected elements (local exemplars, tri d'images / matériel from the fiche, N/A framing) well — more grounded than the 0.5B's generic card game.

### 5.5 nocontext leak check — still fabricates

Without the fiche the 1.5B invents plausible-but-false specifics, confirming the fiche is load-bearing:

| Q | nocontext answer (trimmed) | Verdict |
|---|---|---|
| Q1 Warang | "…un lac naturel situé dans le département de Warang, au nord de la République populaire de Chine." | Fabricated (invented location) |
| Q2 Mbodiène | "…point d'eau artificiel créé par l'homme… source d'eau potable pour le village." | Roughly right by luck, but ungrounded |
| Q3 salty lake | "Le lac de Salé." | Wrong (should be Lac Rose) |
| Q4 ardoise instruction | "…répondre aux questions posées par le professeur avec précision et clarté." | Fabricated |
| Q6 river V/F | "La réponse est fausse… formé par la dégradation de la roche." | Right label, invented justification |

The proper-noun questions (Warang → "Chine", Q3 → "lac de Salé") are exactly where the fiche does real work. Interestingly the 1.5B guesses the Q6 label (FAUX) even with no context, but the justification is fabricated — the *grounded* corrigé variants are what make the answer correct *and* justified.

### 5.6 Per-question quality: 0.5B vs 1.5B

Verdicts on the best-grounded variant available for each question type.

| Q | Type | 0.5B verdict | 1.5B verdict | Improved? |
|---|---|---|---|---|
| Q1 Warang | recall | Correct, grounded | Correct, grounded | = |
| Q2 Mbodiène | recall | Correct | Correct, tighter ("tout le village") | slight + |
| Q3 Lac Rose | recall | Correct | Correct | = |
| Q4 verbatim consigne | recall | **Miss** — loose paraphrase everywhere | **Verbatim correct in `answer_exercises`** (paraphrase in `answer`/`full`) | **Yes (variant-dependent)** |
| Q5 Lac de Guiers | corrigé | Right label, self-contradictory justification | Right label; **verbatim corrigé in `full`**, correct justification | **Yes** |
| Q6 river Vrai/Faux | corrigé | **Wrong** ("vraie") + self-contradicting, all variants | **Correct (FAUX) all variants; exact justification with corrigé cached** | **Yes — headline win** |
| Q7 classification exercise | utility | Coherent, mild extrapolation | More local grounding, but **misclassifies** wells in `answer`/`full` | Mixed |
| Q8 oral warm-up | utility | Reasonable | Reasonable, grounded (truncated in `answer_exercises`) | slight + |
| Q9 memory game | utility | Generic | Grounded in fiche exemplars/materials | + |

---

## 6. Truncation

`max_tokens` = 1024. Three rows hit the cap on the long utility questions:

- `answer_exercises`/naive — Q8 (1/9)
- `answer_exercises`/cag — Q8 (1/9)
- `full`/naive — Q9 (1/9)

All are utility generations (Q8/Q9) where the 1.5B writes long structured activities. Truncation is a `max_tokens` limit, not a CAG artifact (the matching naive/cag rows truncate together). It does inflate decode time for those questions and is part of why `answer_exercises`/cag decode averages the highest (47.6 s/Q).

---

## 7. Conclusion — is the 1.5B worth ~2× the wall-clock?

**On mechanics, CAG is a clear win on the 1.5B:** prefill drops ~30–33× (bigger than the 0.5B's ~20–25×, because the larger model's absolute prefill is heavier), `cache_n` confirms full fiche reuse, and the saving grows with context size. But decode is untouched and dominates the 1.5B's wall-clock even harder than the 0.5B's (prefill is only ~4–6% of CAG total), so end-to-end speedups land at **2.0× / 2.2× / 3.2×** — real, but a fraction of the 30× prefill headline.

**On quality, the 1.5B earns its keep at single-fiche scale:**

- It **fixes the 0.5B's headline error (Q6 Vrai/Faux)** — correct FAUX label in every variant, and the exact expected justification once the corrigé is cached. No self-contradiction.
- It **recovers the verbatim consigne (Q4)** and the **verbatim corrigé (Q5)** that the 0.5B only ever paraphrased or muddled — though Q4 only in the `answer_exercises` variant, so the win is real but not robust across variants.
- Utility answers (Q7–Q9) are **better grounded** in the fiche's local exemplars, with the honest caveat that the 1.5B's own classification logic goes sloppy on long generated lists (Q7 misclassifies wells).
- The corrigé axis finally separates as designed: **answer-only gives the right label but weak/guessed justification; caching the exercises adds the correct justification.**

**Bottom line:** at single-fiche scale the 1.5B costs roughly **~2× the 0.5B's wall-clock** and returns **materially better grounding and reasoning** — most importantly it turns the 0.5B's flat-wrong Vrai/Faux answer into a correct, well-justified one, and reproduces verbatim consigne/corrigé text the 0.5B could not. For a pedagogical assistant where a confidently wrong corrigé (0.5B Q6) is worse than a slow one, **the 1.5B is worth the ~2× wall-clock.** The remaining ceiling is decode speed (~5 tok/s) and long-generation discipline (misclassifications, truncation), neither of which CAG can address — CAG's job here is purely to make the fiche free to re-read, which it does, ~30× over.
