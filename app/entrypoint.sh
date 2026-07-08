#!/usr/bin/env bash
# Two-process launcher for the CAG container:
#   1. start llama-server on 127.0.0.1:8080 (internal only), with the KV-cache slot dir
#   2. WAIT until it reports /health = ok (so the app never queries an unloaded model)
#   3. start uvicorn on 0.0.0.0:8000 (published)
# SIGTERM is trapped so `docker stop` shuts both processes down cleanly.
set -euo pipefail

MODEL="${MODEL:-/models/model.gguf}"
CTX="${CTX:-16384}"
DATA_DIR="${DATA_DIR:-/data}"
CACHE_DIR="$DATA_DIR/cache"
# nproc fallback: under `set -e` a failing command substitution would abort the launcher.
THREADS="${LLAMA_THREADS:-$(nproc 2>/dev/null || echo 4)}"
LLAMA_HOST="127.0.0.1"
LLAMA_PORT="8080"
APP_HOST="0.0.0.0"
APP_PORT="${APP_PORT:-8000}"
VENV_PY="/opt/venv/bin/python"

# The saved KV slot lives here; the whole persistence feature depends on this dir.
mkdir -p "$CACHE_DIR"

# Content-pack export/import stream through Python's tempfile (SpooledTemporaryFile), which
# rolls over to $TMPDIR once a pack exceeds the in-memory threshold. Pin TMPDIR to the
# SD-backed data volume, NOT the default /tmp (commonly tmpfs = RAM): a multi-GB pack spilling
# into tmpfs would exhaust the 4 GB Pi's memory. On-disk here, it just uses card space.
export TMPDIR="$DATA_DIR/tmp"
mkdir -p "$TMPDIR"

# The base image's ENTRYPOINT is the llama-server binary, but its path varies across
# builds. Resolve it defensively.
LLAMA_BIN="${LLAMA_BIN:-$(command -v llama-server || true)}"
if [ -z "$LLAMA_BIN" ]; then
  for p in /app/llama-server /llama-server /usr/bin/llama-server \
           /usr/local/bin/llama-server /llama.cpp/llama-server; do
    if [ -x "$p" ]; then LLAMA_BIN="$p"; break; fi
  done
fi
if [ -z "$LLAMA_BIN" ]; then
  echo "[entrypoint] FATAL: llama-server binary not found" >&2
  exit 1
fi

echo "[entrypoint] llama-server = $LLAMA_BIN"
echo "[entrypoint] model=$MODEL ctx=$CTX threads=$THREADS cache=$CACHE_DIR"

# RAM (Pi 4): the KV cache is sized for the FULL --ctx-size (16384), not the 12k doc cap
# (~0.46 GB f16). Peak RSS ≈ ~2 GB (model mmap + KV + buffers + app). 4 GB Pi is the
# practical floor — enable swap; 8 GB is comfortable.

# --slot-save-path enables the POST /slots/N?action=save|restore actions the app relies
# on; --slots also exposes the GET /slots monitor endpoint. Both are cheap insurance.
# --parallel 1 is CRITICAL: this build defaults to -np -1 (auto), which allocated 4 slots
# = 4x16384 unified KV; CPU decode then scans the whole 65k buffer per token (~0.36 tok/s
# observed on x86). The app is single-user on slot 0 by design — one slot, one 16k KV.
"$LLAMA_BIN" \
  --model "$MODEL" \
  --ctx-size "$CTX" \
  --parallel 1 \
  --threads "$THREADS" \
  --host "$LLAMA_HOST" --port "$LLAMA_PORT" \
  --slot-save-path "$CACHE_DIR" \
  --slots &
LLAMA_PID=$!

UVICORN_PID=""
shutdown() {
  echo "[entrypoint] received signal, shutting down…"
  [ -n "$UVICORN_PID" ] && kill -TERM "$UVICORN_PID" 2>/dev/null || true
  kill -TERM "$LLAMA_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  exit 0
}
trap shutdown TERM INT

# Block until llama-server has loaded the model (health = {"status":"ok"}). While it is
# still loading, /health returns 503, so we keep polling. A single Python process does
# the wait using only the stdlib (curl may be absent from the base image).
echo "[entrypoint] waiting for llama-server /health…"
"$VENV_PY" - "$LLAMA_PID" "$LLAMA_HOST" "$LLAMA_PORT" <<'PY'
import json, os, sys, time, urllib.request
pid, host, port = int(sys.argv[1]), sys.argv[2], sys.argv[3]
url = f"http://{host}:{port}/health"
deadline = time.time() + 600  # 12k-token models load slowly on a Pi
while time.time() < deadline:
    try:
        os.kill(pid, 0)  # llama-server still alive?
    except OSError:
        print("[entrypoint] llama-server exited during startup", file=sys.stderr)
        sys.exit(1)
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if json.loads(r.read() or b"{}").get("status") == "ok":
                sys.exit(0)
    except Exception:
        pass  # not ready yet (503 while loading, or connection refused)
    time.sleep(1)
print("[entrypoint] timed out waiting for llama-server health", file=sys.stderr)
sys.exit(2)
PY
health_rc=$?
if [ "$health_rc" -ne 0 ]; then
  echo "[entrypoint] FATAL: llama-server not healthy (rc=$health_rc)" >&2
  kill -TERM "$LLAMA_PID" 2>/dev/null || true
  exit 1
fi
echo "[entrypoint] llama-server healthy -> starting app on :$APP_PORT"

# Start the app in the background so the trap stays active; wait on it.
cd /app
/opt/venv/bin/uvicorn server.main:app --host "$APP_HOST" --port "$APP_PORT" &
UVICORN_PID=$!

# If either process exits, tear the other down so the container stops (no half-dead box).
wait -n "$LLAMA_PID" "$UVICORN_PID"
echo "[entrypoint] a process exited, stopping the container"
shutdown
