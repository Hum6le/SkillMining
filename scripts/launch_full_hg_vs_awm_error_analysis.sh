#!/usr/bin/env bash

# Launch full legacy HG-vs-AWM pairwise error analysis in the background.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/full_hg_vs_awm_error_analysis_nohup_${LAUNCH_ID}.log"

mkdir -p "$OUTPUT_DIR"

nohup bash "$SCRIPT_DIR/run_full_hg_vs_awm_error_analysis.sh" "$@" > "$LOG_PATH" 2>&1 &
PID=$!

echo "Started full HG vs AWM error analysis with nohup."
echo "PID:          $PID"
echo "Log:          $LOG_PATH"
echo "Monitor:      tail -f $LOG_PATH"
