#!/usr/bin/env bash

# Re-run only failed subflows from a SKILL-DISCO all-batch or prior retry, then
# aggregate every completed result retained by that source plus new successes.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

BATCH_DIR=""
SUBFLOW_LIST=""
MODEL=""
GROUPING_BATCH_SIZE=""
MIN_SUPPORT=""
CONDA_ENV=""
PYTHON_BIN=""
SKIP_EVAL=""

usage() {
    cat <<'EOF'
Usage: bash scripts/rerun_failed_skill_disco_abcd_subflows.sh --batch-dir DIR [options]

Read either an all-subflow batch manifest or a prior retry manifest. Rerun only
its failed subflows and aggregate every completed result retained by that
source plus this retry's newly successful results. This script may be chained
until every subflow succeeds.

Required:
  --batch-dir DIR           outputs/skill_disco_abcd_all_* or prior retry directory

Options:
  --subflows "NAME ..."     Override the failed-subflow list from the manifest
  --model NAME              Override the original model
  --batch-size N            Override original Stage-3 grouping batch size
  --min-support N           Override original minimum support
  --conda-env NAME          Override original conda environment
  --python-bin PATH         Override original Python executable
  --skip-eval               Rerun generation only; no total evaluation is produced
  -h, --help                Show this help

Artifacts are written to outputs/skill_disco_abcd_retry_<timestamp>/.
EOF
}

require_value() {
    [[ "$#" -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-dir) require_value "$@"; BATCH_DIR="$2"; shift 2 ;;
        --subflows) require_value "$@"; SUBFLOW_LIST="$2"; shift 2 ;;
        --model) require_value "$@"; MODEL="$2"; shift 2 ;;
        --batch-size) require_value "$@"; GROUPING_BATCH_SIZE="$2"; shift 2 ;;
        --min-support) require_value "$@"; MIN_SUPPORT="$2"; shift 2 ;;
        --conda-env) require_value "$@"; CONDA_ENV="$2"; shift 2 ;;
        --python-bin) require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$BATCH_DIR" ]] || { echo "--batch-dir is required." >&2; usage >&2; exit 2; }
BATCH_DIR="$(cd "$BATCH_DIR" && pwd)"
SOURCE_MANIFEST="$BATCH_DIR/manifest.txt"
[[ -f "$SOURCE_MANIFEST" ]] || { echo "Batch manifest not found: $SOURCE_MANIFEST" >&2; exit 1; }

manifest_value() {
    local key="$1"
    sed -n "s/^${key}=//p" "$SOURCE_MANIFEST" | head -n 1
}

MODEL="${MODEL:-$(manifest_value model)}"
GROUPING_BATCH_SIZE="${GROUPING_BATCH_SIZE:-$(manifest_value grouping_batch_size)}"
MIN_SUPPORT="${MIN_SUPPORT:-$(manifest_value min_support)}"
CONDA_ENV="${CONDA_ENV:-$(manifest_value conda_env)}"
PYTHON_BIN="${PYTHON_BIN:-$(manifest_value python_bin)}"
SKIP_EVAL="${SKIP_EVAL:-$(manifest_value skip_eval)}"

for value_name in MODEL GROUPING_BATCH_SIZE MIN_SUPPORT CONDA_ENV PYTHON_BIN SKIP_EVAL; do
    [[ -n "${!value_name}" ]] || { echo "Could not recover $value_name from $SOURCE_MANIFEST" >&2; exit 1; }
done

SOURCE_KIND="all_batch"
if grep -q '^\[failed retries\]$' "$SOURCE_MANIFEST"; then
    SOURCE_KIND="retry"
fi

SUBFLOWS=()
if [[ -n "$SUBFLOW_LIST" ]]; then
    read -r -a SUBFLOWS <<< "$SUBFLOW_LIST"
else
    if [[ "$SOURCE_KIND" == "retry" ]]; then
        # Current retry manifests use [failed retries]. Older retry manifests
        # may carry the initial runner's [failed subflows] heading instead.
        while IFS= read -r subflow; do
            [[ -n "$subflow" ]] && SUBFLOWS+=("$subflow")
        done < <(awk '
            /^\[failed retries\]$/ || /^\[failed subflows\]$/ { inside=1; next }
            /^\[/ { inside=0 }
            inside && NF { print $1 }
        ' "$SOURCE_MANIFEST")
    else
        while IFS= read -r subflow; do
            [[ -n "$subflow" ]] && SUBFLOWS+=("$subflow")
        done < <(awk '
            /^\[failed subflows\]$/ { inside=1; next }
            /^\[/ { inside=0 }
            inside && NF { print $1 }
        ' "$SOURCE_MANIFEST")
    fi
fi
if [[ ${#SUBFLOWS[@]} -eq 0 ]]; then
    echo "No failed subflows were recorded in: $SOURCE_MANIFEST" >&2
    echo "Detected source kind: $SOURCE_KIND" >&2
    echo "Available manifest sections:" >&2
    grep '^\[' "$SOURCE_MANIFEST" >&2 || true
    echo "If the run was interrupted before its manifest was updated, rerun the known failed names explicitly:" >&2
    echo "  --subflows \"subflow_a subflow_b\"" >&2
    exit 1
fi

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RETRY_DIR="$ROOT_DIR/outputs/skill_disco_abcd_retry_${RUN_ID}"
LOG_DIR="$RETRY_DIR/logs"
RETRY_MANIFEST="$RETRY_DIR/manifest.txt"
mkdir -p "$LOG_DIR"

SUCCESSFUL=()
FAILED=()
RESULTS=()
MISSING_SOURCE_RESULTS=()

collect_result_from_log() {
    local log_path="$1"
    local result_path
    result_path="$(sed -n 's/^Evaluation result:[[:space:]]*//p' "$log_path" | tail -n 1)"
    [[ -n "$result_path" && -f "$result_path" ]] && RESULTS+=("$result_path")
}

# A retry manifest explicitly persists the exact result paths used for its
# partial aggregate; reusing those paths makes retry chains lossless. An
# initial all-batch manifest only has successful child logs, so recover paths
# from their final status lines.
if [[ "$SOURCE_KIND" == "retry" ]]; then
    while IFS= read -r result_path; do
        [[ -n "$result_path" ]] || continue
        if [[ "$SKIP_EVAL" != "1" && -f "$result_path" ]]; then
            RESULTS+=("$result_path")
        elif [[ "$SKIP_EVAL" != "1" ]]; then
            MISSING_SOURCE_RESULTS+=("$result_path")
        fi
    done < <(awk '
        /^\[evaluation results used for aggregate\]$/ { inside=1; next }
        /^\[/ { inside=0 }
        inside && NF { print }
    ' "$SOURCE_MANIFEST")
else
    while IFS= read -r log_path; do
        if [[ "$SKIP_EVAL" != "1" ]] && ! collect_result_from_log "$log_path"; then
            MISSING_SOURCE_RESULTS+=("$log_path")
        fi
    done < <(awk '
        /^\[successful subflows\]$/ { inside=1; next }
        /^\[/ { inside=0 }
        inside && / log=/ { sub(/^.* log=/, ""); print }
    ' "$SOURCE_MANIFEST")
fi

if [[ ${#MISSING_SOURCE_RESULTS[@]} -gt 0 ]]; then
    echo "The source batch marks completed subflows without readable evaluation results:" >&2
    printf '  %s\n' "${MISSING_SOURCE_RESULTS[@]}" >&2
    echo "Refusing to calculate an incomplete total. Repair those source results first." >&2
    exit 1
fi

echo "===== SKILL-DISCO ABCD failed-subflow retry ====="
echo "Source batch:  $BATCH_DIR"
echo "Source kind:   $SOURCE_KIND"
echo "Retry output:  $RETRY_DIR"
echo "Subflows:      ${SUBFLOWS[*]}"

for subflow in "${SUBFLOWS[@]}"; do
    log_path="$LOG_DIR/${subflow}.log"
    child_args=(--subflow "$subflow" --model "$MODEL" --batch-size "$GROUPING_BATCH_SIZE" --min-support "$MIN_SUPPORT" --conda-env "$CONDA_ENV" --python-bin "$PYTHON_BIN")
    [[ "$SKIP_EVAL" == "1" ]] && child_args+=(--skip-eval)
    echo "Retrying $subflow (log: $log_path)"
    if bash "$SCRIPT_DIR/run_skill_disco_abcd_smoke.sh" "${child_args[@]}" > "$log_path" 2>&1; then
        SUCCESSFUL+=("$subflow log=$log_path")
        [[ "$SKIP_EVAL" != "1" ]] && collect_result_from_log "$log_path"
    else
        exit_code=$?
        FAILED+=("$subflow exit=$exit_code log=$log_path")
        echo "Failed again: $subflow (exit=$exit_code; log: $log_path)" >&2
    fi
done

{
    echo "source_batch=$BATCH_DIR"
    echo "source_kind=$SOURCE_KIND"
    echo "model=$MODEL"
    echo "grouping_batch_size=$GROUPING_BATCH_SIZE"
    echo "min_support=$MIN_SUPPORT"
    echo "conda_env=$CONDA_ENV"
    echo "python_bin=$PYTHON_BIN"
    echo "skip_eval=$SKIP_EVAL"
    echo
    echo "[successful retries]"
    printf '%s\n' "${SUCCESSFUL[@]}"
    echo
    echo "[failed retries]"
    printf '%s\n' "${FAILED[@]}"
    echo
    echo "[evaluation results used for aggregate]"
    printf '%s\n' "${RESULTS[@]}"
} > "$RETRY_MANIFEST"

if [[ "$SKIP_EVAL" != "1" && ${#FAILED[@]} -eq 0 && ${#RESULTS[@]} -gt 0 ]]; then
    AGGREGATE_PATH="$RETRY_DIR/aggregate.json"
    "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_skill_disco_abcd_results.py" --results "${RESULTS[@]}" --output "$AGGREGATE_PATH"
    echo "Aggregate: $AGGREGATE_PATH"
elif [[ "$SKIP_EVAL" != "1" && ${#FAILED[@]} -gt 0 ]]; then
    echo "A complete aggregate was not produced because ${#FAILED[@]} retry subflow(s) still failed." >&2
fi

echo "Retry manifest: $RETRY_MANIFEST"
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
