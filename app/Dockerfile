# CAG example app — single image, two processes (llama-server + FastAPI).
# Multi-arch: build with
#   docker buildx build --platform linux/amd64,linux/arm64 -t cag-app --load app/
# Ships the model baked in, ZERO user data. The user mounts a volume at /data.

# --- base llama.cpp server (CPU, multi-arch: linux/amd64, linux/arm64, linux/s390x) ---
# Pinned to a build number for reproducibility. If this tag 404s at build time, pin the
# rolling :server tag by digest instead, e.g.
#   ghcr.io/ggml-org/llama.cpp:server@sha256:<digest>
# (get it via `docker manifest inspect ghcr.io/ggml-org/llama.cpp:server`).
ARG LLAMA_TAG=server-b9894

# uv version, used via an alias stage below (BuildKit forbids variable expansion in
# COPY --from; an ARG-parameterized FROM aliased as a stage is the documented workaround).
ARG UV_VERSION=0.11.21
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --- stage 1: download the model GGUF (kept out of the final image's build history) ----
# Pin to the BUILD host's platform so the arch-independent ~1.12 GB GGUF is downloaded
# ONCE natively instead of being re-fetched under QEMU for every --platform target. The
# single file is COPY --from=model'd into each arch image below (image size is identical
# either way; this saves ~1.1 GB of transfer and an emulated curl per multi-arch build).
FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS model
# Qwen2.5-1.5B-Instruct Q4_K_M — single file, ~1.12 GB. Override at build with
#   --build-arg MODEL_URL=... to swap the model.
ARG MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /models \
 && curl -fSL --retry 3 --retry-delay 5 "$MODEL_URL" -o /models/model.gguf \
 && test -s /models/model.gguf

# --- stage 2: final image = llama.cpp server + Python 3.12 app -------------------------
FROM ghcr.io/ggml-org/llama.cpp:${LLAMA_TAG}

# uv brings its own Python 3.12: the llama.cpp base ships Debian Python 3.11, but the app
# requires >=3.12. Pinned via the global UV_VERSION ARG (uv alias stage above).
COPY --from=uv /uv /uvx /usr/local/bin/

# uv-managed interpreter + project venv at fixed, image-local paths.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install Python 3.12 + locked deps first (cached until pyproject/uv.lock change).
COPY pyproject.toml uv.lock ./
RUN uv python install 3.12 \
 && uv sync --frozen --no-dev \
 && rm -rf /root/.cache

# App code + launcher.
COPY server ./server
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# Usage doc baked into the image — the source repo is private, so the GHCR package page
# can't render the repo README for public users. Read it with:
#   docker run --rm --entrypoint cat ghcr.io/pleias/cag-app /CONTAINER.md
COPY CONTAINER.md /CONTAINER.md
RUN chmod +x /usr/local/bin/entrypoint.sh

# Baked model (from stage 1).
COPY --from=model /models/model.gguf /models/model.gguf

# RAM footprint (Pi 4 sizing): the KV cache is allocated for the FULL context, not the
# 12k document cap — f16 KV at ctx 16384 ≈ 0.46 GB. Peak RSS ≈ model ~1.12 GB (mmap) +
# ~0.46 GB KV + compute/buffers + app/OS ≈ ~2 GB. Fits a 4 GB Pi (the practical floor —
# enable swap) and is comfortable on 8 GB. To trim KV if RAM-tight, launch llama-server
# with --cache-type-k q8_0 --cache-type-v q8_0 (halves KV); f16 is the tested default.
# Runtime config (all overridable at `docker run` time).
ENV MODEL=/models/model.gguf \
    CTX=16384 \
    DATA_DIR=/data \
    APP_PORT=8000
EXPOSE 8000
VOLUME ["/data"]

# Override the base image's llama-server entrypoint with our two-process launcher.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
