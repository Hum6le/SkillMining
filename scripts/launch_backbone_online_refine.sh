#!/usr/bin/env bash

# Launch one online refinement run in the background. Arguments are forwarded
# to run_backbone_online_refine.sh, which prints the final artifact paths.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$LOG_DIR"

LOG_PATH="$LOG_DIR/online_refine_nohup_${LAUNCH_ID}.log"
PID_PATH="$LOG_DIR/online_refine_nohup_${LAUNCH_ID}.pid"
nohup bash "$SCRIPT_DIR/run_backbone_online_refine.sh" "$@" > "$LOG_PATH" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"

echo "Started online refinement with nohup."
echo "PID:          $PID"
echo "PID file:     $PID_PATH"
echo "Log:          $LOG_PATH"
echo "Monitor:      tail -f $LOG_PATH"
