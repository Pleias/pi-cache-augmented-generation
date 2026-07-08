# When on-device CAG beats small-model RAG — use cases

**Audience:** internal, for the "why long context on device?" discussion.
**One-line thesis:** RAG and CAG solve *different* problems. Don't defend CAG as a faster
RAG — it's the tool for tasks where retrieval-of-chunks structurally can't deliver the answer.

---

## The reframe

RAG's premise is *"the answer lives in k chunks and the retriever finds them."* That premise
**breaks** for synthesis, aggregation, and open-ended understanding — a whole class of
school tasks. CAG owns that regime: **deep, multi-turn work on one bounded topic that fits
the context window.** They are complementary.

## Answers to the three objections

**"Why treat long context on device?"**
Because some questions have **no retrievable chunk**:
- **Synthesis / aggregation / counting** across a lesson or several ("compare how these 3
  lessons treat water", "make one revision sheet from the whole chapter"). Measured: on
  cross-fiche questions, chunk-retrieval misses the aggregate; whole-context Qwen2.5-1.5B got
  **7/8** (see `results_corpus_15b.md`). Better chunking can't fix it — the answer is
  *distributed*; no single chunk contains it.
- **A stuck student can't formulate a good retrieval query.** "Je comprends pas le jour et la
  nuit" has no precise chunk target. The model needs the **whole coherent lesson**
  (definition + example + manipulation + exercise), not a fragment.

**"The speed-up is nice but it's still over 2 minutes."**
That is the **corpus extreme** — 10 lessons / 21k tokens cached at once. Two answers:
1. The **realistic tutoring unit is ONE lesson (~2–3k tokens): ~10–40s per answer** on a Pi
   (see numbers below) — the **same ballpark as the existing 35–45s RAG**. The 2 min only
   appears when you deliberately cache ten lessons together.
2. That 2 min is **mostly decode**, a memory-bandwidth-bound, per-answer cost you also pay
   in RAG. Server-flag tuning (flash-attention, KV-quant) was tested and does **not** help on
   the Pi CPU (issue #1, closed) — what *does* help is weight quantization: **Q4_K_M doubles
   decode vs Q8_0** (6 → 12 tok/s measured; TECH_REPORT §8), because decode streams the
   weight bytes every token.

**"We already have RAG on a Pi with Baguettotron at 35–45s per query."**
Yes — **per query, every query** (re-retrieve + re-prefill each time). CAG's edge is
**amortization**: cache the lesson **once**, then every follow-up in the session is cheap →
compare **total session time** over 10–20 questions, not a single shot. Also, that RAG uses
**Baguettotron, which we found degrades on bigger / controversial / cross-document context**;
Qwen2.5-1.5B clears that ceiling. So CAG-on-device is not re-doing factoid lookup — it
**enables harder tasks the current RAG + Baguettotron stack cannot do.**

## Latency evidence (single lesson, per question, Raspberry Pi — from TECH_REPORT §6)

| model | context | with CAG (total) | RAG baseline (their number) |
|---|---|---|---|
| Qwen2.5-0.5B (Q8_0) | 1 lesson (~2–3k tok) | **~10–21s** | 35–45s |
| Qwen2.5-1.5B (Q8_0) | 1 lesson (~2–3k tok) | **~40–50s** | 35–45s |

**Update (2026-07-08, packaged app):** the shipped app uses the 1.5B at **Q4_K_M**, which
doubles decode (6 → **12 tok/s**, measured live on a Pi 4 — decode is memory-bandwidth-bound
and Q4 halves the weight bytes; TECH_REPORT §8). A typical 2–4 sentence grounded answer now
lands in **~3–6s** after a cached prefix — the Q8_0 numbers above are the conservative floor,
and the 1.5B row's "~40–50s" halves in practice.

Single-lesson CAG is now clearly *faster* than the existing 35–45s RAG — and unlocks
multi-turn follow-ups at near-zero added prefill. The "2 min" only applies to the 10-lesson
corpus.

## Use cases where whole-context CAG wins (prioritized)

1. **Offline tutor for a stuck student** (strongest). Cache the lesson; student asks unlimited
   "pourquoi ? / donne un autre exemple / et si… / interroge-moi". The whole coherent lesson
   is present; follow-ups are near-instant. RAG can't sustain a dialogue over disjoint chunks.
2. **Socratic / quiz-me mode over a chapter.** Model quizzes and checks answers grounded in
   the *entire* chapter — never tests a concept the retriever happened to drop.
3. **Teacher lesson prep on one fiche.** Many generations against one cached lesson: exercises,
   variations, differentiation, an assessment, simpler re-explanations.
4. **Cross-lesson synthesis / revision sheets.** "Compare these 3 water lessons", "build a
   revision sheet for the whole water topic" — needs several *whole* lessons; chunk-retrieval
   structurally can't aggregate.
5. **Exam revision on a bounded syllabus.** Cache the chapter(s); student drills dozens of
   questions offline — CAG amortizes across the session.
6. **Reading comprehension where the document IS the subject.** Student reads a text and asks
   about it — you want the full text in context, not snippets of it.
7. **Young / low-literacy users who can't write retrieval queries.** No good query → no good
   retrieval; full-lesson context lets the model help from a vague prompt.
8. **Nuanced / "controversial" topics.** A single retrieved chunk gives a partial, possibly
   misleading view; the full lesson gives balance — exactly where Baguettotron broke and
   Qwen2.5-1.5B held, arguing for *full context + a capable model* = CAG.

## Where RAG wins (state this openly — it makes the case credible)

- Large library (thousands of docs that don't fit the window) → retrieval is mandatory.
- Single-fact lookup in one big doc → retrieve the chunk; don't cache the whole thing.
- CAG's regime is specifically: **bounded context that fits the window + many questions
  against it + completeness / synthesis needed.**

## The clincher: hybrid, not either/or

**RAG selects the relevant lesson(s) from the library → CAG caches that bounded set for the
session.** Retrieval picks the topic; caching sustains the deep, multi-turn, whole-context
understanding work on it. We already have the RAG half (redline); CAG adds the *"now actually
help the student understand this one thing"* half — precisely the offline-school scenario.

---
*Evidence lives in `docs/TECH_REPORT.md` and the `results_*.md` files. The cross-fiche
quality result (7/8 for 1.5B vs the 0.5B's failures) is the empirical backbone of the
"retrieval-can't-do-this" claim.*
