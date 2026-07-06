# CAG Benchmark Results — Baguettotron Phase

## Overview

This document reports the Baguettotron phase of a Cache-Augmented Generation (CAG)
benchmark. The goal of the broader study is to evaluate CAG on slow devices. CAG
precomputes the KV cache of a fixed context — here, a teacher's lesson "fiche" —
once, then reuses that cache as a prefix to answer many questions cheaply, instead
of re-prefilling the fiche on every question.

This phase is the **negative-result half** of the study: it documents why
Baguettotron, and the PleIAs RAG family more broadly, are structurally
incompatible with prefix-cache CAG.

## Setup

- **Runtime:** llama.cpp `llama-server`, HTTP API. Benchmark client `bench/cag_bench.py`.
- **Hardware:** measured on a Raspberry Pi (dedicated) and an x86 Xeon (heavily
  contended, load ~55). The Xeon timings are contention noise and are not usable.
- **Model:** Baguettotron (PleIAs), 321M parameters, Q8_0 GGUF. A French "small
  reasoning + RAG" model that emits `<think>` reasoning traces. Context window 4096.
- **Dataset entry:** idx 31, a CE1 geography fiche about water points
  (naturel/artificiel), Senegal. Two context variants:
  - `answer` (~1841 tokens)
  - `answer_exercises` (~2363 tokens)
- **Cache modes:**
  - `nocontext` — no fiche, leak baseline.
  - `naive` — fiche re-prefilled on every question.
  - `cag` — fiche cached as a prefix and reused via llama.cpp `cache_prompt`.

## The Baguettotron Saga (chronological)

### 1. System-message / "FICHE:" framing → context ignored, hallucination

The first attempts fed the fiche as a system message with a "FICHE:" framing. The
model **ignored the context entirely and hallucinated** from its parametric memory.
Asked about "la mare de Warang" (a Senegal pond), it answered that this was a
"grande rivière du Karnataka, en Inde." It also mimicked the fiche's lesson-plan
genre ("Voici un plan de cours structuré…") and confabulated a fake multi-turn
conversation ("Backtracking conversation flow…").

### 2. Tighter system prompt → repetition loop

Tightening the system prompt ("réponds en 1-2 phrases, pas de plan de cours") did
not fix the grounding. At temperature 0, the model fell into a repetition loop
("de la fiche, de la citation de la fiche, de la citation de la…").

### 3. Repetition penalty → loop stopped, still confabulating

Adding a repetition penalty of 1.3 stopped the loop, but the model still
confabulated — inventing a curriculum-design task instead of answering the question.

### 4. Model card research → the actual required format

Reviewing the model card and Hugging Face discussions revealed the format the model
was actually trained for:

- **Special source tokens.** Sources must be wrapped in `<source_1>…</source_1>`
  special tokens. The model grounds on these and cites with `<ref>[quote]</ref>`.
  Any other framing causes it to ignore the context.
- **No system message.** The card's RAG example uses none.
- **`<|im_end|>` is not the EOS token (critical bug).** Only `<|end_of_text|>` is
  registered as EOS. Generation therefore runs past the answer into an invented
  next turn — the source of the "fake conversation" confabulation. The fix is to
  pass explicit stop strings `["<|im_end|>", "<|im_start|>"]`.
- **Query-first ordering.** The card's format puts the question first, then the
  `<source_1>` block **after** it.

### 5. Correct format → the model finally grounds

With the correct format (`<source_1>` tags, no system message, explicit stop
tokens) and the source positioned **after** the question, the model grounded for
the first time. For "mare de Warang" it answered:

> "La Mare de Warang est une mer naturelle située dans le sud du Sénégal"

This is correct on Sénégal and naturelle, with a minor error ("mer"/sea instead of
"mare"/pond), and it included a `<ref>` citation. finish reason = `stop`,
~488 tokens.

### 6. Moving the source before the question → grounding breaks

The benchmark's CAG mode requires the context to be a stable **prefix** (source
**before** the question) so its KV cache can be reused. When the source was moved
before the question, grounding broke: the model reverted to confabulation
("Content synthesis approach: 1. Section 1…", "Backtracking conversation flow…").
Only the ordering changed between step 5 and step 6, which makes this decisive.

## Format Experiments Summary

| # | Framing | System msg | Stop tokens | Source position | Temp / rep. penalty | Outcome |
|---|---------|-----------|-------------|-----------------|---------------------|---------|
| 1 | System msg / "FICHE:" | Yes | Default | — | — | Context ignored; parametric hallucination (Karnataka river); lesson-plan genre mimicry; fake conversation |
| 2 | Tight system prompt | Yes | Default | — | temp 0 | Repetition loop ("de la fiche, de la citation…") |
| 3 | Tight system prompt + rep. penalty | Yes | Default | — | rep. penalty 1.3 | Loop stopped; still confabulates (invents curriculum task) |
| 4 | `<source_1>` tags (card format) | No | `<|im_end|>`, `<|im_start|>` | After question (query-first) | — | Correct trained format identified |
| 5 | `<source_1>` tags | No | `<|im_end|>`, `<|im_start|>` | **After** question | — | **Grounded** — correct answer + `<ref>`; finish=stop, ~488 tok |
| 6 | `<source_1>` tags | No | `<|im_end|>`, `<|im_start|>` | **Before** question (CAG prefix) | — | Grounding **breaks**; confabulation returns |

## Core Finding

**Baguettotron grounds only when the source comes after the query** — its trained
RAG format. But prefix-cache CAG requires the context **before** the query, so that
its KV cache forms a reusable stable prefix. These two requirements are **mutually
exclusive**. On Baguettotron you can have grounded answers *or* the CAG prefill
savings — not both.

The CAG speedup itself was verified working under source-first ordering:

- naive prefill: ~15,000–21,000 ms per question
- cag prefill: ~600–850 ms per question (cache_n ≈ 2445)
- roughly a **28× prefill reduction**

But with source-first ordering, those fast answers were ungrounded garbage.

The whole PleIAs family shares this constraint. Pleias-RAG-350M and Pleias-RAG-1B
use `<|query_start|>…<|query_end|>` **first**, then `<|source_start|>…` — the same
query-first, CAG-hostile ordering. No PleIAs model is context-first: not the 57M
Monad, not the 321M Baguettotron, not the 353M RAG-350M, not the 1.2B RAG-1B.
Scaling up does not fix it.

## Conclusion

For grounded QA, Baguettotron and the PleIAs RAG family are query-first and are
therefore **structurally incompatible with prefix-cache CAG**. The CAG speedup is
real (~28× prefill reduction), but it can only be obtained with source-first
ordering, under which Baguettotron does not ground.

To get grounding *and* CAG, a **context-first instruct model** is required
(documented separately for Qwen).

A practical gotcha carries over regardless of model choice within this family: the
`<|im_end|>`-is-not-EOS bug means explicit stop strings
(`["<|im_end|>", "<|im_start|>"]`) must be passed, or the model runs past its answer
and confabulates an invented next turn.
