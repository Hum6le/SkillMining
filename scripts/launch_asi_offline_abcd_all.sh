#!/usr/bin/env bash

# Launch the complete independent-subflow ASIoffline ABCD batch in background.
# All arguments are forwarded to run_asi_offline_abcd_all.sh.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/asi_offline_abcd_all_nohup_${LAUNCH_ID}.log"

mkdir -p "$OUTPUT_DIR"

nohup env PYTHONUNBUFFERED=1 bash "$SCRIPT_DIR/run_asi_offline_abcd_all.sh" "$@" \
    > "$LOG_PATH" 2>&1 &
PID=$!

echo "Started full independent-subflow ASIoffline ABCD batch with nohup."
echo "PID:          $PID"
echo "Log:          $LOG_PATH"
echo "Output root:  $OUTPUT_DIR"
echo "Monitor:      tail -f $LOG_PATH"
echo "Each subflow also writes outputs/asi_offline_abcd_all_<timestamp>/logs/<subflow>.log."
