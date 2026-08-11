#!/usr/bin/env bash

# Launch failed-subflow retries in the background. Arguments are forwarded to
# rerun_failed_skill_disco_abcd_subflows.sh, including the required --batch-dir.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/skill_disco_abcd_retry_nohup_${LAUNCH_ID}.log"

mkdir -p "$OUTPUT_DIR"

nohup bash "$SCRIPT_DIR/rerun_failed_skill_disco_abcd_subflows.sh" "$@" > "$LOG_PATH" 2>&1 &
PID=$!

echo "Started SKILL-DISCO failed-subflow retry with nohup."
echo "PID:          $PID"
echo "Log:          $LOG_PATH"
echo "Output root:  $OUTPUT_DIR"
echo "Monitor:      tail -f $LOG_PATH"
echo "The retry manifest and, after complete success, aggregate.json will be printed in the log."
