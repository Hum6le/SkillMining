#!/usr/bin/env bash

# Launch current 10-flow HG backbone vs AWM paired analysis with nohup.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/hg_vs_awm_error_analysis_10flow_nohup_${LAUNCH_ID}.log"
PID_PATH="$OUTPUT_DIR/hg_vs_awm_error_analysis_10flow_nohup_${LAUNCH_ID}.pid"

mkdir -p "$OUTPUT_DIR"
nohup bash "$SCRIPT_DIR/run_full_hg_vs_awm_error_analysis_10flow.sh" "$@" > "$LOG_PATH" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"

echo "Started 10-flow HG backbone vs AWM paired error analysis with nohup."
echo "PID:      $PID"
echo "PID file: $PID_PATH"
echo "Log:      $LOG_PATH"
echo "Monitor:  tail -f $LOG_PATH"
