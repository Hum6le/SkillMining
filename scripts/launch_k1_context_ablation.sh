#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../outputs/k1_context_launch_logs"
mkdir -p "$LOG_DIR"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)_$$"
LOG_PATH="$LOG_DIR/k1_context_${LAUNCH_ID}.log"
PID_PATH="$LOG_DIR/k1_context_${LAUNCH_ID}.pid"
nohup bash "$SCRIPT_DIR/run_k1_context_ablation.sh" "$@" > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"
echo "K1 context ablation launched. PID: $PID"
echo "PID file: $PID_PATH"
echo "Monitor: tail -f '$LOG_PATH'"
