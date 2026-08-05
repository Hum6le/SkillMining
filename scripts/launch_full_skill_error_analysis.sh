#!/usr/bin/env bash

# Launch the full skill error analysis in the background.
# All arguments are forwarded to run_full_skill_error_analysis.sh.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/full_skill_error_analysis_nohup_${LAUNCH_ID}.log"

mkdir -p "$OUTPUT_DIR"

nohup bash "$SCRIPT_DIR/run_full_skill_error_analysis.sh" "$@" > "$LOG_PATH" 2>&1 &
PID=$!

echo "Started full skill error analysis with nohup."
echo "PID:          $PID"
echo "Log:          $LOG_PATH"
echo "Monitor:      tail -f $LOG_PATH"

