#!/usr/bin/env bash

# Launch the complete ABCD runner in the background. The underlying runner can
# distribute balanced subflow workers across --workflow-ids.
# All arguments are forwarded to run_full_abcd_experiments.sh.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/full_abcd_nohup_${LAUNCH_ID}.log"
PID_PATH="$OUTPUT_DIR/full_abcd_nohup_${LAUNCH_ID}.pid"

mkdir -p "$OUTPUT_DIR"

nohup bash "$SCRIPT_DIR/run_full_abcd_experiments.sh" "$@" > "$LOG_PATH" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"

echo "Started full ABCD experiment with nohup."
echo "PID:          $PID"
echo "PID file:     $PID_PATH"
echo "Log:          $LOG_PATH"
echo "Stop:         bash scripts/stop_full_abcd_experiments.sh --pid-file $PID_PATH"
echo "Output root:  $OUTPUT_DIR"
echo "Monitor:      tail -f $LOG_PATH"
echo "The final artifact directories and aggregate JSON paths will be printed at the end of the log."
