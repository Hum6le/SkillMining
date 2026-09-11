#!/usr/bin/env bash
set -u

# Background launcher; follows launch_asi_offline_abcd_smoke.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/trace2skill_parallel_eval_${LAUNCH_ID}.log"
mkdir -p "$OUTPUT_DIR"

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
if [[ "$SUBFLOW_COUNT" -ne 1 ]]; then
    echo "Provide exactly one --subflow NAME for a single-subflow run." >&2
    exit 2
fi

nohup env PYTHONUNBUFFERED=1 bash "$SCRIPT_DIR/evaluate_trace2skill_subflow_parallel.sh" "$@" \
    > "$LOG_PATH" 2>&1 &
PID=$!
PID_FILE="$OUTPUT_DIR/trace2skill_parallel_eval_${LAUNCH_ID}.pid"
echo "$PID" > "$PID_FILE"

echo "Started Trace2Skill single-subflow parallel evaluation with nohup."
echo "PID:          $PID"
echo "Log:          $LOG_PATH"
echo "Monitor:      tail -f $LOG_PATH"
echo "Stop:         kill $PID"
echo "Stop all:     bash scripts/stop_trace2skill_parallel_eval.sh --pid-file $PID_FILE"
echo "The evaluation result, shard outputs, and manifest are written under the output directory."
