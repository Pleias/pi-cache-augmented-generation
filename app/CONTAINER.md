# cag-app — Cache-Augmented Generation in a box

A single, self-contained demo image for **Cache-Augmented Generation (CAG)**: it prefills a
document's KV cache **once**, saves that cache to a mounted volume, and **restores it on
restart** (or when you switch documents) — so grounded answers skip re-prefilling the
context every time. Keep a library of documents and swap between them in seconds.

Inside the image: `llama-server` (llama.cpp, CPU) + a FastAPI backend + a single-file chat
UI, with **Qwen2.5-1.5B-Instruct Q4_K_M** (~1.12 GB GGUF) baked in. It ships **zero content
data** — you paste your own document at runtime.

## Quick start

```bash
docker run -d --name cag -p 8000:8000 -v "$HOME/cag-data:/data" \
  ghcr.io/pleias/cag-app:latest
```

Then open <http://localhost:8000>, paste a document (≤ 12,000 tokens; over-cap is rejected
with the actual token count), and chat. The flow is **ingest → ask → restart-restore**: the
first ingest prefills and saves the KV slot to `/data/cache`; `docker restart cag` restores
it with no re-prefill. Multi-turn chat streams over the cached prefix with a capped history
budget.

**The conversation survives restarts too**: each completed turn is persisted next to that
document's cache (`docs/<id>/history.json`), replayed in the UI on reload, and re-prefilled
as the small history tail on top of the restored document prefix. Each document keeps its
own conversation. "Clear chat" or ingesting a new document starts fresh; a model change
drops the old conversation (its token accounting would be stale). This document
(`/CONTAINER.md`) is also served in-app — the **Docs** button in the header.

## Library & packs

The app keeps a **library** of documents, not just one. Each ingest adds a new entry with
its own saved KV slot on disk; the header's **Library** button lists them, and switching
**restores that document's cache in seconds** (no 5-min re-prefill) and reloads its chat.
Only one slot is ever live in RAM, so the decode budget is unchanged — the cost is disk
(~300 MB per saved slot). The library is capped at **20 documents**; delete one to add more.

- **Guided modes**: one-tap chips above the chat box (*Quiz me*, *Explain simpler*,
  *Make 5 exercises*, *Summarize*) fire a ready-made instruction over the cached document,
  in the document's own language.
- **UI languages**: the interface is available in **English, French and Arabic** (RTL),
  picked from the header and remembered per device. Answers always follow the language of
  your question/document, independent of the UI language.
- **Content packs**: **Export pack** downloads the whole library (index + documents +
  slots + chats) as a `.tar`; **Import pack** loads one on another device — seconds per
  device, no re-prefill — so a prepared curriculum copies over a USB stick or seeds a fleet.
  A pack is a cross-device artifact, so **import is treated as untrusted** and validated
  before anything is written: archive member paths (no absolute paths, no `..` traversal,
  only `library.json` / `docs/` / `cache/`); member types (symlinks, hardlinks, devices and
  fifos are refused); and size — an upload cap (`PACK_MAX_BYTES`, default 8 GB), a
  per-member and cumulative extracted-bytes cap (stops a decompression bomb), and a
  free-space pre-check. A pack that would exceed the 20-document library cap is refused, and
  documents whose id already exists are skipped (existing copy kept). A failed import rolls
  back — no half-written documents are left behind.
- **Export is consistent but tolerant**: the small index/document files are snapshotted
  instantly; the large slots are then streamed without blocking chat. If a document is
  deleted mid-export its slot is simply omitted from the tar — the pack stays valid and that
  document re-prefills (once) on first switch on the importing device.
- **Wipe everything** (type `DELETE` to confirm) is the privacy purge: it deletes every
  document, slot, history and the index, leaving no trace on the device.

## Volume

Mount a host directory at **`/data`**. It holds the library index (`library.json`), one
directory per document under `docs/<id>/` (`fiche.txt` + `meta.json` + `history.json`), and
the saved KV slots under `cache/<id>.bin`. A pre-library single-document volume
(`fiche.txt` + `cache/slot0.bin` at the root) is migrated into the library automatically on
first start, preserving the saved slot (no re-prefill). The image itself is stateless;
everything persistent lives in this volume.

## Platforms & resources

- **Architectures:** `linux/amd64`, `linux/arm64`.
- **Floor target:** Raspberry Pi 4. Peak RSS ≈ **~2 GB** (model mmap + f16 KV cache sized for
  the full context + buffers + app). **4 GB RAM** is the practical minimum (enable swap on a
  Pi); 8 GB is comfortable.
- KV-cache-tight? Launch `llama-server` with `--cache-type-k q8_0 --cache-type-v q8_0` to
  halve the KV footprint (f16 is the tested default).

## Configuration

All overridable at `docker run` time with `-e`:

| Variable   | Default              | Purpose                                  |
|------------|----------------------|------------------------------------------|
| `MODEL`    | `/models/model.gguf` | Path to the GGUF model inside the image  |
| `CTX`      | `16384`              | llama-server context size (KV is sized to this) |
| `DATA_DIR` | `/data`              | Where the library, documents + KV cache slots live |
| `APP_PORT` | `8000`              | Port the FastAPI app listens on          |
| `PACK_MAX_BYTES` | `8589934592` (8 GB) | Max size of an imported content pack (upload + extraction cap) |

The container runs two processes from one entrypoint: `llama-server` on internal
`127.0.0.1:8080`, then `uvicorn` on `0.0.0.0:${APP_PORT}`. The app is **single-user on slot 0
by design** (`--parallel 1`). `docker stop` shuts both down cleanly (SIGTERM trapped).

## Docs

Full benchmark, technical report, and CAG-vs-RAG use cases:
<https://github.com/pleias/pi-cache-augmented-generation>
