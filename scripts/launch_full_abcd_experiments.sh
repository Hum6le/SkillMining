#!/usr/bin/env bash

# Launch the complete sequential ABCD runner in the background.
# All arguments are forwarded to run_full_abcd_experiments.sh.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/full_abcd_nohup_${LAUNCH_ID}.log"

mkdir -p "$OUTPUT_DIR"

nohup bash "$SCRIPT_DIR/run_full_abcd_experiments.sh" "$@" > "$LOG_PATH" 2>&1 &
PID=$!

echo "Started full ABCD experiment with nohup."
echo "PID:          $PID"
echo "Log:          $LOG_PATH"
echo "Output root:  $OUTPUT_DIR"
echo "Monitor:      tail -f $LOG_PATH"
echo "The final artifact directories and aggregate JSON paths will be printed at the end of the log."
