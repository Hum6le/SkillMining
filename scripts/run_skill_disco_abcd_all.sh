#!/usr/bin/env bash

# Run the complete SKILL-DISCO ABCD protocol independently for every subflow.
# Each child run uses that subflow's full train.json and frozen test.json.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

MODEL="deepseek-chat"
GROUPING_BATCH_SIZE=20
MIN_SUPPORT=2
CONDA_ENV="skillmining310"
PYTHON_BIN="python"
SKIP_EVAL=0
CONTINUE_ON_ERROR=1
SUBFLOW_LIST=""

usage() {
    cat <<'EOF'
Usage: bash scripts/run_skill_disco_abcd_all.sh [options]

Run the complete SKILL-DISCO-Offline protocol separately on every ABCD
subflow directory. Each run uses all train conversations and all frozen test
conversations for that subflow.

Options:
  --subflows "NAME ..."      Run only this whitespace-separated subset
  --model NAME               Unified llm.chat model name (default: deepseek-chat)
  --batch-size N             Stage-3 grouping batch size (default: 20)
  --min-support N            Minimum distinct induction conversations per skill (default: 2)
  --conda-env NAME           Conda environment passed to each child run (default: skillmining310)
  --python-bin PATH          Python executable passed to each child run (default: python)
  --skip-eval                Generate SKILL.md artifacts without evaluation
  --stop-on-error            Stop after the first failed subflow
  -h, --help                 Show this help

Logs and the batch manifest are written below:
  outputs/skill_disco_abcd_all_<timestamp>/
EOF
}

require_value() {
    local option="$1"
    local remaining_args="$2"
    [[ "$remaining_args" -ge 2 ]] || { echo "Missing value for $option" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subflows)
            require_value "$1" "$#"
            SUBFLOW_LIST="$2"
            shift 2
            ;;
        --model)
            require_value "$1" "$#"
            MODEL="$2"
            shift 2
            ;;
        --batch-size)
            require_value "$1" "$#"
            GROUPING_BATCH_SIZE="$2"
            shift 2
            ;;
        --min-support)
            require_value "$1" "$#"
            MIN_SUPPORT="$2"
            shift 2
            ;;
        --conda-env)
            require_value "$1" "$#"
            CONDA_ENV="$2"
            shift 2
            ;;
        --python-bin)
            require_value "$1" "$#"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --skip-eval)
            SKIP_EVAL=1
            shift
            ;;
        --stop-on-error)
            CONTINUE_ON_ERROR=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for value_name in GROUPING_BATCH_SIZE MIN_SUPPORT; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer; got: $value" >&2
        exit 2
    fi
done

SPLITS_DIR="$ROOT_DIR/data/eval/abcd/splits"
if [[ ! -d "$SPLITS_DIR" ]]; then
    echo "ABCD split directory not found: $SPLITS_DIR" >&2
    exit 1
fi

SUBFLOWS=()
if [[ -n "$SUBFLOW_LIST" ]]; then
    read -r -a SUBFLOWS <<< "$SUBFLOW_LIST"
else
    while IFS= read -r split_dir; do
        SUBFLOWS+=("${split_dir##*/}")
    done < <(find "$SPLITS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort)
fi

if [[ ${#SUBFLOWS[@]} -eq 0 ]]; then
    echo "No subflow directories found under $SPLITS_DIR" >&2
    exit 1
fi

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
BATCH_DIR="$ROOT_DIR/outputs/skill_disco_abcd_all_${RUN_ID}"
LOG_DIR="$BATCH_DIR/logs"
MANIFEST="$BATCH_DIR/manifest.txt"
mkdir -p "$LOG_DIR"

SUCCESSFUL=()
FAILED=()
STARTED_AT="$(date -Iseconds)"

write_manifest() {
    local status="$1"
    {
        echo "run_id=$RUN_ID"
        echo "status=$status"
        echo "started_at=$STARTED_AT"
        echo "finished_at=$(date -Iseconds)"
        echo "model=$MODEL"
        echo "grouping_batch_size=$GROUPING_BATCH_SIZE"
        echo "min_support=$MIN_SUPPORT"
        echo "conda_env=$CONDA_ENV"
        echo "python_bin=$PYTHON_BIN"
        echo "skip_eval=$SKIP_EVAL"
        echo "requested_subflows=${SUBFLOWS[*]}"
        echo
        echo "[successful subflows]"
        printf '%s\n' "${SUCCESSFUL[@]}"
        echo
        echo "[failed subflows]"
        printf '%s\n' "${FAILED[@]}"
    } > "$MANIFEST"
}

echo "===== SKILL-DISCO ABCD full batch run ====="
echo "Subflows:      ${#SUBFLOWS[@]}"
echo "Model:         $MODEL"
echo "Batch output:  $BATCH_DIR"
echo "Each subflow uses complete train/test splits."

for subflow in "${SUBFLOWS[@]}"; do
    log_path="$LOG_DIR/${subflow}.log"
    child_args=(
        --subflow "$subflow"
        --model "$MODEL"
        --batch-size "$GROUPING_BATCH_SIZE"
        --min-support "$MIN_SUPPORT"
        --conda-env "$CONDA_ENV"
        --python-bin "$PYTHON_BIN"
    )
    if [[ "$SKIP_EVAL" -eq 1 ]]; then
        child_args+=(--skip-eval)
    fi

    echo
    echo "===== [$(( ${#SUCCESSFUL[@]} + ${#FAILED[@]} + 1 ))/${#SUBFLOWS[@]}] $subflow ====="
    echo "Log: $log_path"
    if bash "$SCRIPT_DIR/run_skill_disco_abcd_smoke.sh" "${child_args[@]}" > "$log_path" 2>&1; then
        SUCCESSFUL+=("$subflow log=$log_path")
        echo "Completed: $subflow"
    else
        exit_code=$?
        FAILED+=("$subflow exit=$exit_code log=$log_path")
        echo "Failed: $subflow (exit=$exit_code). See $log_path" >&2
        if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
            write_manifest "failed"
            exit "$exit_code"
        fi
    fi
    write_manifest "running"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    write_manifest "completed_with_failures"
    echo "Batch finished with ${#FAILED[@]} failed subflow(s). Manifest: $MANIFEST" >&2
    exit 1
fi

write_manifest "completed"
echo "All ${#SUCCESSFUL[@]} subflows completed. Manifest: $MANIFEST"
