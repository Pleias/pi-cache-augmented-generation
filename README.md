# Cache-Augmented Generation on a Raspberry Pi

A technical study of **Cache-Augmented Generation (CAG)** with small (<2B parameter)
LLMs on low-power devices, plus a runnable demo app. Motivating scenario: offline
education — a teacher's lesson sheet grounded on a Raspberry Pi, no cloud.

## What is CAG (vs RAG)?

**RAG** (Retrieval-Augmented Generation) retrieves the most relevant chunks for each
incoming query and re-prefills them into the model every time. **CAG** (Cache-Augmented
Generation) instead precomputes the model's KV cache for a *fixed* context **once** and
reuses that cache across many questions — the context stays a stable prefix, so the
expensive prefill is paid a single time.

They are complementary: RAG picks *which* bounded material is relevant from a large
corpus; CAG makes working deeply and repeatedly *inside* that bounded material cheap —
multi-turn tutoring, synthesis, quiz generation, "explain it again simpler", all against
the same cached document.

## Findings (measured on a Raspberry Pi 4)

- **CAG removes 20–76× of prefill time**, and the saving *grows* with context size: 20×
  at ~2k cached tokens, up to 65–76× at 21k tokens. Time-to-first-token at 21k context
  drops from ~19 minutes (re-prefill every question) to ~15 seconds.
- **CAG needs context-first models.** The context must sit *before* the question as a
  stable, reusable prefix. Models trained query-first (sources after the question — a
  deliberate RAG-optimized design) lose their grounding when the ordering is flipped to
  enable caching. Standard instruct models with the document in the system message work.
- **Decode is the residual wall.** CAG doesn't cache generation, and generation speed is
  memory-bandwidth-bound: it falls as the cached context grows. Interactive use favors
  one lesson-sized context (~2–3k tokens) over big concatenated corpora.
- **Weight quantization doubles decode.** Q4 weights halve the bytes streamed per
  generated token vs Q8: measured 6 → 12 tokens/s on the same 1.5B model, same device.
- **A 1.5B model is the sweet spot.** It fixes the reasoning failures of a 0.5B (negation,
  answer-key justification, cross-document questions) at ~1.6–2× the wall-clock.
- **KV cache size is architecture-driven, not parameter-count-driven.** Few KV heads
  (GQA) and moderate depth win; some smaller models carry a 5× bigger per-token cache
  than larger ones.

## The app — CAG in a box

A single, self-contained Docker image (linux/amd64 + linux/arm64, **Raspberry Pi 4 is
the floor target**): llama.cpp's `llama-server` + a FastAPI backend + a one-file chat
UI, with **Qwen2.5-1.5B-Instruct Q4_K_M** baked in. It ships **zero content data** —
you bring your own document.

```bash
docker run -d --name cag -p 8000:8000 -v "$HOME/cag-data:/data" \
  ghcr.io/pleias/cag-app:latest
```

Open http://localhost:8000 and:

1. **Ingest** — paste a document (≤ 12,000 tokens; over the cap it is rejected with the
   actual token count and an explanation). The app prefills the KV cache once and saves
   it to your mounted volume.
2. **Ask** — chat about the document, answers stream in, grounded in the text. Multi-turn:
   follow-ups keep their context, with an automatically capped history budget. One-tap
   **guided modes** (*Quiz me · Explain simpler · Make 5 exercises · Summarize*) work in
   the document's own language.
3. **Restart-restore** — restart the container and the cache is restored from the volume
   in seconds (no re-prefill), the conversation is replayed, and follow-ups continue
   where they left off.

It keeps a **library** of up to 20 documents — each with its own saved cache and its own
conversation — and switching documents restores the cached one in seconds instead of a
minutes-long re-prefill. A prepared library can be **exported as a single pack file** and
imported on another device (USB-stick distribution to offline sites; imports are strictly
validated), and a **wipe** control erases everything for privacy. The interface speaks
**English, French, and Arabic** (right-to-left supported); answers follow the language of
the question.

Everything persistent lives in your mounted `/data` directory: the documents, the saved
KV caches, and the conversations. The image is stateless; nothing leaves the device.
Built-in documentation is available behind the **Docs** button in the app header, or:

```bash
docker run --rm --entrypoint cat ghcr.io/pleias/cag-app:latest /CONTAINER.md
```

**Measured on a Raspberry Pi 4** (Q4_K_M, single lesson-sized document): ~44–53 tokens/s
prefill, **~12 tokens/s generation** — a typical short grounded answer lands in a few
seconds, and every follow-up skips the document prefill entirely.

Resource floor: ~2 GB peak RSS → a 4 GB Pi 4 works (enable swap); 8 GB is comfortable.

## Caveats

This is a research demo, not a product. Quality verdicts come from one lesson cluster
and a handful of questions per type — indicative, not statistical. Generation speed
falls with cached-context size, so the interactive unit is one bounded document, not a
whole library: that's the regime CAG is for.
