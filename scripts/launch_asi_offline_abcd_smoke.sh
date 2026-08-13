#!/usr/bin/env bash

# Launch one ASIoffline ABCD smoke run in the background and report a log path.
# Arguments are passed unchanged to run_asi_offline_abcd_smoke.sh.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/asi_offline_abcd_subflow_nohup_${LAUNCH_ID}.log"

mkdir -p "$OUTPUT_DIR"

# Fail before invoking nohup if the worker has no exact target subflow.
SUBFLOW_COUNT=0
for ((i = 1; i <= $#; i++)); do
    if [[ "${!i}" == "--subflow" ]]; then
        next=$((i + 1))
        if [[ $next -gt $# || -z "${!next}" || "${!next}" == --* ]]; then
            echo "--subflow requires one subflow name." >&2
            exit 2
        fi
        SUBFLOW_COUNT=$((SUBFLOW_COUNT + 1))
    fi
done
if [[ $SUBFLOW_COUNT -ne 1 ]]; then
    echo "Provide exactly one --subflow NAME for a single-subflow run." >&2
    exit 2
fi

nohup env PYTHONUNBUFFERED=1 bash "$SCRIPT_DIR/run_asi_offline_abcd_smoke.sh" "$@" \
    > "$LOG_PATH" 2>&1 &
PID=$!

echo "Started ASIoffline ABCD single-subflow run with nohup."
echo "PID:          $PID"
echo "Log:          $LOG_PATH"
echo "Output root:  $OUTPUT_DIR"
echo "Monitor:      tail -f $LOG_PATH"
echo "The run directory, manifest, and evaluation summary are printed at the end of the log."
