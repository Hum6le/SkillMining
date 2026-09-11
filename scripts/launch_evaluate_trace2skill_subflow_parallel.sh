#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_ROOT/trace2skill_parallel_eval_${LAUNCH_ID}.log"
mkdir -p "$OUTPUT_ROOT"

SUBFLOW_COUNT=0
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--subflow" ]]; then
    next=$((i+1))
    if [[ $next -gt $# || -z "${!next}" || "${!next}" == --* ]]; then
      echo "--subflow requires one subflow name." >&2; exit 2
    fi
    SUBFLOW_COUNT=$((SUBFLOW_COUNT+1))
  fi
done
[[ "$SUBFLOW_COUNT" -eq 1 ]] || { echo "Provide exactly one --subflow NAME." >&2; exit 2; }

COMMAND=(bash "$SCRIPT_DIR/evaluate_trace2skill_subflow_parallel.sh" "$@")
nohup "${COMMAND[@]}" > "$LOG_PATH" 2>&1 &
PID=$!
echo "Started Trace2Skill single-subflow parallel evaluation."
echo "PID:          $PID"
echo "Log:          $LOG_PATH"
echo "Monitor:      tail -f $LOG_PATH"
echo "Stop:         kill $PID"
printf "Reproduce:    nohup "; printf '%q ' "${COMMAND[@]}"; printf "> %q 2>&1 &\n" "$LOG_PATH"
