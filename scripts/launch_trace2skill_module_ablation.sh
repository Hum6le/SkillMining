#!/usr/bin/env bash
# Launch Trace2Skill module ablation in the background.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="$ROOT_DIR/outputs/trace2skill_module_ablation_$(date +%Y-%m-%d_%H-%M-%S)"
for ((index=1; index<=$#; index++)); do
  arg="${!index}"
  next_index=$((index + 1))
  if [[ "$arg" == "--output-dir" && $next_index -le $# ]]; then
    OUTPUT_ROOT="${!next_index}"
  fi
done
LOG_DIR="$OUTPUT_ROOT/launcher_logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/launch_$(date +%Y-%m-%d_%H-%M-%S).log"
PID_PATH="$LOG_DIR/module_ablation.pid"

nohup bash "$SCRIPT_DIR/run_trace2skill_module_ablation.sh" "$@" \
  > "$LOG_PATH" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"

echo "Started Trace2Skill module ablation with nohup."
echo "PID:          $PID"
echo "Output root:  $OUTPUT_ROOT"
echo "Log:          $LOG_PATH"
echo "PID file:     $PID_PATH"
echo "Stop all:     bash scripts/stop_trace2skill_module_ablation.sh --pid-file $PID_PATH"
echo "Monitor:      tail -f $LOG_PATH"
