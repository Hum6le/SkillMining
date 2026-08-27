#!/usr/bin/env bash

# Launch the full online-refinement scheduler under nohup.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$LOG_DIR"

LOG_PATH="$LOG_DIR/full_online_refine_nohup_${LAUNCH_ID}.log"
PID_PATH="$LOG_DIR/full_online_refine_nohup_${LAUNCH_ID}.pid"
nohup bash "$SCRIPT_DIR/run_full_backbone_online_refine.sh" "$@" > "$LOG_PATH" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"

echo "Started full backbone online refinement with nohup."
echo "PID:          $PID"
echo "PID file:     $PID_PATH"
echo "Log:          $LOG_PATH"
echo "Monitor:      tail -f $LOG_PATH"
