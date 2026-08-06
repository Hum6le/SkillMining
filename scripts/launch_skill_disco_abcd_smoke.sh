#!/usr/bin/env bash

# Launch the complete single-subflow SKILL-DISCO ABCD runner in the background.
# All arguments are forwarded to run_skill_disco_abcd_smoke.sh.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
LAUNCH_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="$OUTPUT_DIR/skill_disco_abcd_subflow_nohup_${LAUNCH_ID}.log"

mkdir -p "$OUTPUT_DIR"

# Fail synchronously when this wrapper is invoked without a single target.
# Otherwise nohup would return a PID even though the worker exits immediately.
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

nohup bash "$SCRIPT_DIR/run_skill_disco_abcd_smoke.sh" "$@" > "$LOG_PATH" 2>&1 &
PID=$!

echo "Started complete single-subflow SKILL-DISCO ABCD run with nohup."
echo "PID:          $PID"
echo "Log:          $LOG_PATH"
echo "Output root:  $OUTPUT_DIR"
echo "Monitor:      tail -f $LOG_PATH"
echo "The generated artifact, SKILL.md, evaluation result, and manifest paths will be printed at the end of the log."
