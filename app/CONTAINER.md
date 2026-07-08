# cag-app — Cache-Augmented Generation in a box

A single, self-contained demo image for **Cache-Augmented Generation (CAG)**: it prefills
the KV cache of a *fixed* document **once**, saves that cache to a mounted volume, and
**restores it on restart** — so grounded answers skip re-prefilling the context every time.

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

## Volume

Mount a host directory at **`/data`** — it holds your document and the saved KV cache slot.
The image itself is stateless; everything persistent lives in this volume.

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
| `DATA_DIR` | `/data`              | Where the document + KV cache slot live  |
| `APP_PORT` | `8000`              | Port the FastAPI app listens on          |

The container runs two processes from one entrypoint: `llama-server` on internal
`127.0.0.1:8080`, then `uvicorn` on `0.0.0.0:${APP_PORT}`. The app is **single-user on slot 0
by design** (`--parallel 1`). `docker stop` shuts both down cleanly (SIGTERM trapped).

## Docs

Full benchmark, technical report, and CAG-vs-RAG use cases:
<https://github.com/pleias/pi-cache-augmented-generation>
