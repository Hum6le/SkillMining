#!/usr/bin/env bash

# Launch the current 10-flow Skill-DisCo ABCD protocol using the same worker
# scheduler as AWM and Trace2Skill. All arguments are forwarded to the shared
# runner, including --workflow-ids.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/skill_disco_abcd_all_nohup_${LAUNCH_ID}.log"
PID_PATH="$OUTPUT_DIR/skill_disco_abcd_all_nohup_${LAUNCH_ID}.pid"

mkdir -p "$OUTPUT_DIR"

nohup bash "$SCRIPT_DIR/run_full_abcd_experiments.sh" --method skill_disco "$@" > "$LOG_PATH" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"

echo "Started complete multi-subflow SKILL-DISCO ABCD run with nohup."
echo "PID:          $PID"
echo "PID file:     $PID_PATH"
echo "Log:          $LOG_PATH"
echo "Output root:  $OUTPUT_DIR"
echo "Monitor:      tail -f $LOG_PATH"
echo "The worker logs and aggregate summary paths will be printed at the end of the log."
