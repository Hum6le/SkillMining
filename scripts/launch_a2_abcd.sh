#!/usr/bin/env bash
# Start an A2 ABCD run with nohup; forwards all arguments to run_a2_abcd.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="$ROOT_DIR/outputs/a2_launch_logs"
mkdir -p "$LOG_DIR"

LOG_PATH="$LOG_DIR/a2_${LAUNCH_ID}.log"
PID_PATH="$LOG_DIR/a2_${LAUNCH_ID}.pid"
nohup bash "$SCRIPT_DIR/run_a2_abcd.sh" "$@" > "$LOG_PATH" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"

echo "A2 launched in the background."
echo "PID:      $PID"
echo "PID file: $PID_PATH"
echo "Log:      $LOG_PATH"
echo "Monitor:  tail -f '$LOG_PATH'"
