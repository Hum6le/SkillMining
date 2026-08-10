#!/usr/bin/env bash

# Launch the complete multi-subflow SKILL-DISCO ABCD runner in the background.
# All arguments are forwarded to run_skill_disco_abcd_all.sh.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/skill_disco_abcd_all_nohup_${LAUNCH_ID}.log"

mkdir -p "$OUTPUT_DIR"

nohup bash "$SCRIPT_DIR/run_skill_disco_abcd_all.sh" "$@" > "$LOG_PATH" 2>&1 &
PID=$!

echo "Started complete multi-subflow SKILL-DISCO ABCD run with nohup."
echo "PID:          $PID"
echo "Log:          $LOG_PATH"
echo "Output root:  $OUTPUT_DIR"
echo "Monitor:      tail -f $LOG_PATH"
echo "The batch manifest and per-subflow logs will be printed at the end of the log."
