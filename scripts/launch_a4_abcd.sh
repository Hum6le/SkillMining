#!/usr/bin/env bash
# Launch A4 in the background; all arguments go to run_a4_abcd.py.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/outputs/a4_launch_logs"
mkdir -p "$LOG_DIR"
LAUNCH_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_PATH="$LOG_DIR/a4_${LAUNCH_ID}.log"
PID_PATH="$LOG_DIR/a4_${LAUNCH_ID}.pid"
nohup bash "$ROOT_DIR/scripts/run_a4_abcd.sh" "$@" > "$LOG_PATH" 2>&1 < /dev/null &
echo "$!" > "$PID_PATH"
echo "A4 launched: pid=$(cat "$PID_PATH") log=$LOG_PATH"
