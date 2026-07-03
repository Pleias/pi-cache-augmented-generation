#!/usr/bin/env bash
# Stop the llama-server started by run_server.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="${PIDFILE:-$ROOT/bench/server.pid}"

if [[ ! -f "$PIDFILE" ]]; then
  echo "No pidfile ($PIDFILE). Nothing to stop."; exit 0
fi
PID="$(cat "$PIDFILE")"
if kill -0 "$PID" 2>/dev/null; then
  echo "Stopping llama-server (pid $PID)..."
  kill "$PID"
  for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$PID" 2>/dev/null && { echo "Force kill."; kill -9 "$PID"; }
  echo "Stopped."
else
  echo "Process $PID not running."
fi
rm -f "$PIDFILE"
