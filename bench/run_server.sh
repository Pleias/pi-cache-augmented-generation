#!/usr/bin/env bash
# Launch llama-server (CPU prebuilt) for the CAG benchmark.
# Runs ONCE and stays warm; the Python runner talks to it over HTTP.
#
# Override anything via env, e.g.:
#   MODEL=models/baguettotron-Q4_K_M.gguf CTX=4096 ./bench/run_server.sh
set -euo pipefail

# --- paths -------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$ROOT/llama-b9864-bin-ubuntu-x64/llama-b9864}"
MODEL="${MODEL:-$ROOT/models/baguettotron.gguf}"     # <-- point this at your GGUF
KV_DIR="${KV_DIR:-$ROOT/bench/kv_cache}"             # disk KV save/restore (CAG cold-start)
LOG="${LOG:-$ROOT/bench/server.log}"
PIDFILE="${PIDFILE:-$ROOT/bench/server.pid}"

# --- server params -----------------------------------------------------------
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-4096}"                                   # Baguettotron trained ctx = 4000
THREADS="${THREADS:-$(nproc)}"

# --- preflight ---------------------------------------------------------------
if [[ ! -x "$LLAMA_DIR/llama-server" ]]; then
  echo "ERROR: llama-server not found at $LLAMA_DIR" >&2; exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model not found at $MODEL" >&2
  echo "       set MODEL=/path/to/your.gguf and re-run" >&2; exit 1
fi
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Server already running (pid $(cat "$PIDFILE")). Run ./bench/stop_server.sh first." >&2; exit 1
fi

mkdir -p "$KV_DIR"

# libggml/libllama .so live next to the binary -> loader needs them on the path
export LD_LIBRARY_PATH="$LLAMA_DIR:${LD_LIBRARY_PATH:-}"

echo "Starting llama-server:"
echo "  model   = $MODEL"
echo "  ctx     = $CTX   threads = $THREADS"
echo "  listen  = http://$HOST:$PORT"
echo "  kv dir  = $KV_DIR"
echo "  log     = $LOG"

nohup "$LLAMA_DIR/llama-server" \
  --model "$MODEL" \
  --ctx-size "$CTX" \
  --threads "$THREADS" \
  --host "$HOST" --port "$PORT" \
  --jinja \
  --slot-save-path "$KV_DIR" \
  --metrics \
  > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "  pid     = $(cat "$PIDFILE")"

# --- wait for /health --------------------------------------------------------
echo -n "Waiting for server to be ready"
for _ in $(seq 1 120); do
  if curl -sf "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
    echo " -> OK"
    curl -sf "http://$HOST:$PORT/props" 2>/dev/null | \
      python3 -c "import sys,json; d=json.load(sys.stdin); print('  n_ctx =', d.get('default_generation_settings',{}).get('n_ctx'), '| model =', d.get('model_path','?').split('/')[-1])" 2>/dev/null || true
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo " -> server died. Last log lines:" >&2; tail -n 20 "$LOG" >&2; exit 1
  fi
  sleep 1; echo -n "."
done
echo " -> timeout" >&2; exit 1
