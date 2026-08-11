#!/usr/bin/env bash

# Run the frozen-library ASIoffline protocol independently for each ABCD
# subflow. All model calls are delegated to Python scripts that use llm.chat().

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

TEMPERATURE="1.0"
PYTHON_BIN="python"
SUBFLOW_LIST=""
MAX_TRAIN=""
MAX_TEST=""
RESUME=0
DRY_RUN=0
SKIP_EVAL=0
CONTINUE_ON_ERROR=1

usage() {
    cat <<'EOF'
Usage: bash scripts/run_asi_offline_abcd_all.sh [options]

Runs ASIoffline separately for every shared ABCD subflow split. Each run uses
only <subflow>/train.json for induction and the paired frozen test.json for
evaluation. Model calls occur only through llm.chat() in the Python stages.

Options:
  --subflows "NAME ..."     Run only the named whitespace-separated subflows
  --temperature FLOAT       ASI induction sampling temperature (default: 1.0)
  --python-bin PATH         Python executable (default: python)
  --max-train N             Limit induction trajectories per subflow
  --max-test N              Limit test conversations per subflow
  --resume-induction        Continue completed per-trajectory induction artifacts
  --dry-run-induction       Write prompts only; do not call an LLM
  --skip-eval               Freeze libraries without test evaluation
  --stop-on-error           Stop after the first failed subflow
  -h, --help                Show this message
EOF
}

require_value() {
    [[ "$#" -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subflows) require_value "$@"; SUBFLOW_LIST="$2"; shift 2 ;;
        --temperature) require_value "$@"; TEMPERATURE="$2"; shift 2 ;;
        --python-bin) require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        --max-train) require_value "$@"; MAX_TRAIN="$2"; shift 2 ;;
        --max-test) require_value "$@"; MAX_TEST="$2"; shift 2 ;;
        --resume-induction) RESUME=1; shift ;;
        --dry-run-induction) DRY_RUN=1; shift ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        --stop-on-error) CONTINUE_ON_ERROR=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

SPLITS_DIR="$ROOT_DIR/data/eval/abcd/splits"
[[ -d "$SPLITS_DIR" ]] || { echo "ABCD split directory not found: $SPLITS_DIR" >&2; exit 1; }

SUBFLOWS=()
if [[ -n "$SUBFLOW_LIST" ]]; then
    read -r -a SUBFLOWS <<< "$SUBFLOW_LIST"
else
    while IFS= read -r split_dir; do
        SUBFLOWS+=("${split_dir##*/}")
    done < <(find "$SPLITS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort)
fi
[[ ${#SUBFLOWS[@]} -gt 0 ]] || { echo "No subflows selected" >&2; exit 1; }

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
BATCH_DIR="$ROOT_DIR/outputs/asi_offline_abcd_all_${RUN_ID}"
LOG_DIR="$BATCH_DIR/logs"
mkdir -p "$LOG_DIR"
SUCCESSFUL=()
FAILED=()

for subflow in "${SUBFLOWS[@]}"; do
    output_dir="$BATCH_DIR/$subflow"
    log_path="$LOG_DIR/$subflow.log"
    command=("$PYTHON_BIN" scripts/run_asi_offline_abcd.py --subflow "$subflow" --output-dir "$output_dir" --temperature "$TEMPERATURE")
    [[ -n "$MAX_TRAIN" ]] && command+=(--max-train "$MAX_TRAIN")
    [[ -n "$MAX_TEST" ]] && command+=(--max-test "$MAX_TEST")
    [[ "$RESUME" -eq 1 ]] && command+=(--resume-induction)
    [[ "$DRY_RUN" -eq 1 ]] && command+=(--dry-run-induction)
    [[ "$SKIP_EVAL" -eq 1 ]] && command+=(--skip-eval)

    echo "===== ASIoffline / $subflow ====="
    echo "Log: $log_path"
    if "${command[@]}" > "$log_path" 2>&1; then
        SUCCESSFUL+=("$subflow")
    else
        status=$?
        FAILED+=("$subflow exit=$status log=$log_path")
        echo "Failed: $subflow (see $log_path)" >&2
        [[ "$CONTINUE_ON_ERROR" -eq 1 ]] || exit "$status"
    fi
done

manifest="$BATCH_DIR/manifest.json"
write_json_array() {
    local first=1
    local value
    for value in "$@"; do
        if [[ "$first" -eq 0 ]]; then
            printf ','
        fi
        printf '\n    "%s"' "$value"
        first=0
    done
    [[ "$first" -eq 0 ]] && printf '\n  '
}
{
    printf '{\n  "method": "asioffline-abcd",\n  "successful_subflows": ['
    write_json_array "${SUCCESSFUL[@]}"
    printf '],\n  "failed_subflows": ['
    write_json_array "${FAILED[@]}"
    printf ']\n}\n'
} > "$manifest"

if [[ "$DRY_RUN" -eq 0 && "$SKIP_EVAL" -eq 0 && ${#SUCCESSFUL[@]} -gt 0 ]]; then
    "$PYTHON_BIN" scripts/aggregate_subflow_results.py \
        --runs "$BATCH_DIR" --recursive --output "$BATCH_DIR/aggregate.json" || {
        echo "Could not aggregate completed ASIoffline evaluations." >&2
        exit 1
    }
fi

echo "ASIoffline batch artifacts: $BATCH_DIR"
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
