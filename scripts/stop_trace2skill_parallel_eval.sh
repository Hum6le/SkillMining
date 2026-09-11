#!/usr/bin/env bash
set -u
PID=""; PID_FILE=""
if [[ $# -eq 2 && "$1" == "--pid" ]]; then PID="$2"; fi
if [[ $# -eq 2 && "$1" == "--pid-file" ]]; then PID_FILE="$2"; fi
if [[ -n "$PID_FILE" ]]; then
  [[ -f "$PID_FILE" ]] || { echo "PID file not found: $PID_FILE" >&2; exit 2; }
  PID="$(head -n 1 "$PID_FILE")"
fi
[[ "$PID" =~ ^[0-9]+$ ]] || { echo "Usage: $0 --pid PID | --pid-file FILE" >&2; exit 2; }
is_running() { kill -0 "$1" 2>/dev/null; }
descendants() {
  local parent="$1" child
  for child in $(pgrep -P "$parent" 2>/dev/null || true); do
    descendants "$child"
    echo "$child"
  done
}
if ! is_running "$PID"; then echo "Launcher is not running: $PID"; exit 0; fi
mapfile -t CHILDREN < <(descendants "$PID")
echo "Stopping launcher $PID and ${#CHILDREN[@]} descendant process(es)..."
for child in "${CHILDREN[@]}"; do kill -TERM "$child" 2>/dev/null || true; done
kill -TERM "$PID" 2>/dev/null || true
sleep 2
for child in "${CHILDREN[@]}"; do kill -0 "$child" 2>/dev/null && kill -KILL "$child" 2>/dev/null || true; done
kill -0 "$PID" 2>/dev/null && kill -KILL "$PID" 2>/dev/null || true
echo "Stopped."
