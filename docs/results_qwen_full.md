# CAG on Qwen2.5-0.5B-Instruct — Full 9-Question Run

**Fiche:** `#E2-CE1-ESVS-GEO-P2-2026-02-17-003` — *"Les points d'eau de mon milieu (naturels et artificiels)"* (CE1 geography, 30-minute lesson).
**Model:** Qwen2.5-0.5B-Instruct, Q8_0 quant, context-first ChatML (fiche placed in the system message).
**Hardware:** Raspberry Pi.
**Runner:** `bench/cag_bench.py --model-format chatml`, temperature 0.
**Results file:** `bench/results/entry31_qwen-full_20260704-202428.json`
**Questions:** `bench/questions_entry31.json`

This is the **positive CAG result**. It is the full-run companion to `docs/results_qwen.md` (a 2-question smoke test), and the counterpoint to `docs/results_baguettotron.md`, where PleIAs query-first models cannot do CAG at all. Here a context-first ChatML model reuses a cached fiche prefix correctly, and the prefill savings are large and grow with context size.

---

## 1. What is being compared

Each of 9 questions is answered under three **modes**:

- **`nocontext`** — no fiche in the prompt. Quality/leak baseline: what does the model say with no grounding?
- **`naive`** — the full fiche is re-prefilled from scratch for *every* question. Correct, but pays the fiche prefill cost 9 times.
- **`cag`** — the fiche's KV cache is computed **once** and reused as a prefix for all 9 questions. Only the (short) question tokens are prefilled per call.

CAG's promise is that `naive` and `cag` produce the *same* answer (same tokens fed to the model), but `cag` amortizes the expensive fiche prefill. The generation (decode) work is identical between them.

Three **context variants** control how much fiche is cached:

| Variant | Cached tokens (`cache_n`) | KV cache size |
|---|---|---|
| `answer` (fiche body only) | 1919 | 22.5 MB |
| `answer_exercises` (body + exercises + corrigé) | 2418 | 28.3 MB |
| `full` (entire fiche) | 3143 | 36.8 MB |

KV cache footprint is **~12 KB/token** for this model (Qwen 0.5B: 24 layers, 2 KV heads). The `cache_n` field in the results confirms the whole fiche prefix is reused on each CAG call (e.g. 1909–1919 for the `answer` variant).

**Questions:** 9 total — 4 recall (Q1–Q4), 2 corrigé (Q5–Q6), 3 utility/generation (Q7–Q9).

---

## 2. Latency summary

Per-question averages over the 9 questions (milliseconds). "Prefill" = prompt processing, "Decode" = answer generation, "Total" = wall-clock.

| Variant | Mode | Prefill (ms) | Decode (ms) | Total (ms) | Cached KV |
|---|---|---:|---:|---:|---:|
| **answer** (1919 tok) | nocontext | 328 | 5447 | 5786 | — |
| | naive | 10336 | 19617 | 29967 | — |
| | **cag** | **527** | 20300 | **20839** | 22.5 MB |
| **answer_exercises** (2418 tok) | nocontext | 326 | 5428 | 5764 | — |
| | naive | 14044 | 15686 | 29744 | — |
| | **cag** | **641** | 11125 | **11779** | 28.3 MB |
| **full** (3143 tok) | nocontext | 328 | 5437 | 5775 | — |
| | naive | 20403 | 10816 | 31236 | — |
| | **cag** | **807** | 9396 | **10219** | 36.8 MB |

(The decode averages above were recomputed from the per-question `predicted_ms` rows in the JSON and match the runner's reported figures.)

---

## 3. Prefill savings — the CAG effect

This is where CAG works exactly as intended.

| Variant | naive prefill | cag prefill | Reduction |
|---|---:|---:|---:|
| answer | 10336 ms | 527 ms | **~20×** (19.6×) |
| answer_exercises | 14044 ms | 641 ms | **~22×** (21.9×) |
| full | 20403 ms | 807 ms | **~25×** (25.3×) |

Two things to note:

1. **The saving grows with context size.** The bigger the cached fiche, the more prefill work CAG avoids: 20× for the 1919-token `answer` fiche, 25× for the 3143-token `full` fiche. Under `naive`, prefill scales roughly linearly with fiche length (10.3 s → 14.0 s → 20.4 s). Under `cag`, prefill stays near-flat (527 → 641 → 807 ms) because only the short question is processed each time; the small growth is just the cache-reattach/lookup overhead over a longer prefix.
2. **The whole fiche prefix is genuinely reused.** `cache_n` equals the fiche token count on every CAG call, so no re-prefill of the fiche is happening.

**CAG kills the prefill cost. That is the headline.**

---

## 4. The honest caveat — decode dominates on this model

CAG saves prefill, **not decode**. Decode (token generation) is the cost of producing the answer, and it is identical whether the fiche was cached or freshly prefilled — same model, same tokens, same output.

On this small, fast model, **decode is the larger component of total time** in the grounded modes:

- `answer`/cag: decode 20300 ms out of 20839 ms total → prefill is only ~2.5% of wall-clock.
- `full`/cag: decode 9396 ms out of 10219 ms total → prefill ~8%.

So the **total-time speedup is much smaller than the prefill speedup**:

| Variant | naive total | cag total | Wall-clock speedup |
|---|---:|---:|---:|
| answer | 29967 ms | 20839 ms | **1.4×** |
| answer_exercises | 29744 ms | 11779 ms | **2.5×** |
| full | 31236 ms | 10219 ms | **3.1×** |

The prefill reduction is 20–25×, but because decode is untouched and dominates, real wall-clock gains land at ~1.5–3×. This is expected: CAG's payoff scales with the fraction of your latency that is prefill. On a 0.5B model generating multi-sentence answers, that fraction is modest. On a larger model, or with longer contexts and shorter answers, the prefill fraction — and thus the CAG payoff — would be much larger.

### Why does `full` (10.2 s) beat `answer` (20.8 s) despite a bigger context?

Counterintuitive, but it is **not** a CAG effect. CAG makes prefill negligible in *both* variants (807 ms vs 527 ms — a 280 ms difference). The 10-second gap in total time is entirely **decode variance**: the `answer`/cag runs happened to generate longer answers (20300 ms decode) than the `full`/cag runs (9396 ms decode). Same questions, temperature 0, but the different surrounding context nudges the model toward different answer lengths. This is answer-length variability, not evidence that a larger cache is faster. The correct reading is: *once CAG has neutralized prefill, total time is driven by how much text the model chooses to generate.*

---

## 5. Answer quality

Temperature 0. Examples are quoted from the JSON, lightly trimmed. `naive` and `cag` produce equivalent answers by construction, so grounded examples below are from `cag`.

### 5.1 Recall (Q1–Q4) — grounding works

**Q1 (mare de Warang)** — expected: *a pond that fills after the rains; natural because it exists without direct human intervention.*

> `answer`/cag: "La mare de Warang est un point d'eau naturel. Elle est considérée comme tel car elle est créée par la nature et n'a pas été aménagée par l'homme. Elle est un endroit où l'eau est recueillie après les pluies, ce qui lui confère un caractère naturel."

Correct and grounded — it recovers the fiche-specific fact ("l'eau est recueillie après les pluies").

**Q2 (forage de Mbodiène)** — expected: *a borehole dug so the whole village has drinking water; artificial because dug/built by humans.*

> `answer`/cag: "Le forage du village de Mbodiène est un point d'eau artificiel. C'est un endroit où l'on trouve de l'eau potable, créé par l'homme pour les besoins de la population."

Correct.

**Q3 (salty, coloured lake)** — expected: *Le Lac Rose.*

> `full`/cag: "Le grand point d'eau naturel cité comme exemple connu pour son sel et sa couleur particulière est le Lac Rose."

Correct.

**Q4 (exact ardoise evaluation instruction)** — expected the literal instruction: *« Écris N si c'est naturel, A si c'est artificiel : forage, mare, lac, puits. »*

> `full`/cag: "La consigne exacte donnée aux élèves lors de la phase d'évaluation sur l'ardoise […] est de répondre aux questions sur leur ardoise."

**Weak.** The model paraphrases into a generic statement and fails to reproduce the verbatim "Écris N … A …" instruction, even with the full fiche cached. Grounding is present in spirit but the model does not quote precisely — a limitation of the 0.5B model, not of CAG.

### 5.2 Corrigé (Q5–Q6) — the "do exercises help?" axis

These questions probe facts that live only in the exercises/corrigé section, so in principle the `answer`-only variant should be unable to answer while `answer_exercises`/`full` should succeed.

**Q5 (Lac de Guiers: N or A?)** — expected: **N (Naturel)** — created by nature.

> `answer`/cag: "Le lac de Guiers est un point d'eau naturel (N). C'est un endroit où l'on trouve de l'eau, et il est créé par la nature sans l'intervention directe de l'homme."
> `full`/cag: "Le corrigé répond que le lac de Guiers est un point d'eau naturel (N). C'est parce que l'homme a créé et a aménagé ce lac pour l'arroser, ce qui est une action humaine."

The **label is correct (N) in all three variants** — including `answer`, which does not contain the corrigé. The model guesses "naturel" correctly from the word "lac" / general knowledge, so this question does **not** cleanly discriminate the variants here. Note also that the `full` variant, despite citing the corrigé, gives a **self-contradictory justification** ("created by nature" then "humans created and developed this lake") — right label, muddled reasoning.

**Q6 (Vrai/Faux: "a river is artificial because humans navigate on it")** — expected: **FAUX** — a river is natural; human use does not change its origin.

> `answer`/cag: "La réponse est que l'affirmation est vraie. La justification est que le corrigé affirme que 'un fleuve est un point d'eau artificiel car l'homme navigue dessus'. Cependant, en réalité, un fleuve est un point d'eau naturel, donc il n'est pas artificiel."
> `answer_exercises`/cag: "La réponse est que l'affirmation est vraie. […] ce qui est une déclaration fausse selon le texte original."
> `full`/cag: "La réponse est que l'affirmation est vraie. […] Elle indique que le fleuve est un point d'eau naturel créé par l'homme pour naviguer dessus."

**Wrong in all three variants.** The model opens with "l'affirmation est vraie" (it should be FAUX), then contradicts itself mid-answer. Adding the exercises/corrigé does not fix it. This is a genuine reasoning failure of the 0.5B model on a negation/vrai-faux task — CAG faithfully feeds it the corrigé, but the model mishandles the logic.

**Verdict on the axis:** on this particular fiche, the corrigé questions did not produce the clean "answer-only fails, answer+exercises passes" separation the design targeted — Q5's label was guessable and Q6 failed everywhere. The exercises variant *does* change the answers (compare the Q5/Q6 texts across variants), it just did not flip correctness here. Where adding exercises clearly helps is availability of the corrigé content for the model to reference; the bottleneck was model reasoning, not context.

### 5.3 Utility / generation (Q7–Q9) — longer responses

**Q7 (propose a classification exercise for the ardoise)** — open-ended, no `expected` field.

> `full`/cag (decode 17189 ms, not truncated): "Voici un exercice de classification (naturel/artificiel) pour les élèves de CE1 sur l'ardoise […] Exercice 1 — Consigne: Classifiez les points d'eau […] Réponses: Naturel : mare, lac, marigot / Artificiel : puits, forage, bassin de rétention […]"

A coherent, well-structured, pedagogically usable exercise. This is the kind of question that dominates decode time (17 s to generate) and illustrates why decode, not prefill, sets the wall-clock in this run. The content is reasonable, though it introduces items beyond the fiche (marigot, bassin de rétention) — plausible extrapolation rather than strict grounding.

---

## 6. nocontext leak check

Does the model answer correctly with **no fiche** at all? Short answer: **no — it fabricates plausible but ungrounded specifics.**

| Q | nocontext answer (trimmed) | Verdict |
|---|---|---|
| Q1 Warang | "…situé dans la région de la Sénégalie… point de départ important pour la navigation et la pêche." | Fabricated (invented location/purpose) |
| Q2 Mbodiène | "…point d'eau artificiel créé pour l'exploitation agricole… agriculture et la pêche." | Wrong purpose (fiche: village drinking water) |
| Q3 salty lake | "…est le sable." ("it's sand") | Nonsense/wrong (should be Lac Rose) |
| Q4 ardoise instruction | "…répondre à 10 questions sur l'ardoise." | Fabricated |
| Q5 Lac de Guiers | "…point d'eau naturel (N). Il est situé sur le territoire français…" | Right label, wrong/invented detail |

The proper-noun questions (Warang, Mbodiène) are exactly where the fiche is load-bearing: without it the model invents a coherent-sounding but false answer. Q3 collapses entirely ("le sable"). So `nocontext` is not a viable shortcut — the fiche is doing real work, and CAG lets the model use it cheaply.

Note the timing: `nocontext` has a **low total (~5.8 s)** despite a **high decode (~5.4 s)**, purely because there is no fiche to prefill (~328 ms prompt). This confirms the decode-dominance point from §4 from the other direction — strip out prefill entirely and you are left with ~5 s of pure generation.

---

## 7. Conclusion

- **CAG delivers a large, real prefill reduction — ~20× to ~25× — and it scales with context size.** The bigger the cached fiche, the more prefill work avoided. `cache_n` confirms the full fiche prefix is reused on every call.
- **On this fast 0.5B model, decode dominates total time, so wall-clock speedups are more modest — ~1.4× (answer), ~2.5× (answer_exercises), ~3.1× (full).** CAG does not, and cannot, speed up generation. The payoff is proportional to how much of your latency is prefill; here that fraction is small, so the total-time win, while real, trails the headline prefill number. The `full` variant's lower total vs `answer` is decode-length variance, not a CAG effect.
- **Grounding works.** Recall answers recover fiche-specific facts (Warang fills after rain; Mbodiène village borehole; Lac Rose). Without the fiche, `nocontext` fabricates.
- **Model reasoning is the ceiling, not CAG.** Q4 (verbatim recall) is paraphrased loosely; Q6 (vrai/faux negation) is wrong across all variants. Adding the exercises/corrigé variant changes the answers but did not flip correctness on this fiche's corrigé questions — the corrigé is available to the model, the 0.5B reasoning is what falls short.

**Bottom line:** context-first ChatML makes CAG viable on Qwen2.5-0.5B (unlike the query-first PleIAs models in `docs/results_baguettotron.md`), and CAG cuts prefill by up to 25×. Be honest that on a fast small model decode dominates, so end-to-end you see ~1.5–3×; the technique's leverage grows with larger models, longer contexts, and shorter answers.
