#!/usr/bin/env bash
# Start the SGD coordinator in the background, mirroring the ABCD launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/full_sgd_nohup_${LAUNCH_ID}.log"
PID_PATH="$OUTPUT_DIR/full_sgd_nohup_${LAUNCH_ID}.pid"
mkdir -p "$OUTPUT_DIR"

nohup bash "$SCRIPT_DIR/run_full_sgd_experiments.sh" "$@" > "$LOG_PATH" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"

echo "Started SGD experiments."
echo "PID:         $PID"
echo "PID file:    $PID_PATH"
echo "Log:         $LOG_PATH"
echo "Monitor:     tail -f $LOG_PATH"
echo "Final run root and aggregate path will appear in the log."
