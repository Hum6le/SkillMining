#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../outputs/k1_organization_launch_logs"
mkdir -p "$LOG_DIR"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)_$$"
LOG_PATH="$LOG_DIR/a3_k1_${LAUNCH_ID}.log"
PID_PATH="$LOG_DIR/a3_k1_${LAUNCH_ID}.pid"
nohup bash "$SCRIPT_DIR/run_k1_organization_ablation.sh" "$@" > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"
echo "A3 K1 organization launched. PID: $PID"
echo "PID file: $PID_PATH"
echo "Monitor: tail -f '$LOG_PATH'"
