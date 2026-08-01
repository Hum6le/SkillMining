#!/usr/bin/env bash

# Run the ABCD experiments under the independent-subflow protocol.
#
# AWM and Trace2Skill are invoked once per subflow. Graph Mining uses its
# independent --all mode. The script activates conda, sets the HF mirror,
# records failed runs, and writes weighted global summaries.
#
# Examples:
#   bash scripts/run_full_abcd_experiments.sh
#   bash scripts/run_full_abcd_experiments.sh --method awm
#   bash scripts/run_full_abcd_experiments.sh --subflow recover_username
#   bash scripts/run_full_abcd_experiments.sh --method trace2skill --stop-on-error

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="skillmining310"
HF_MIRROR="https://hf-mirror.com"
METHOD="all"
ONE_SUBFLOW=""
MIN_SESSIONS=0
EVOLUTION_BATCH_SIZE=25
CONTINUE_ON_ERROR=1
PYTHON_BIN="python"
REBUILD_SPLITS=1

usage() {
    cat <<'EOF'
Usage: bash scripts/run_full_abcd_experiments.sh [options]

Options:
  --method NAME              all, awm, trace2skill, or graph (default: all)
  --subflow NAME             Run one subflow instead of the complete split list
  --min-sessions N           Graph Mining minimum train sessions (default: 0)
  --evolution-batch-size N   Trace2Skill outer batch size (default: 25)
  --stop-on-error            Stop at the first failed subflow
  --no-rebuild-splits        Reuse existing subflow session splits
  -h, --help                 Show this help

The script always activates conda environment skillmining310 and sets:
  HF_ENDPOINT=https://hf-mirror.com
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            [[ $# -ge 2 ]] || { echo "Missing value for --method" >&2; exit 2; }
            METHOD="$2"
            shift 2
            ;;
        --subflow)
            [[ $# -ge 2 ]] || { echo "Missing value for --subflow" >&2; exit 2; }
            ONE_SUBFLOW="$2"
            shift 2
            ;;
        --min-sessions)
            [[ $# -ge 2 ]] || { echo "Missing value for --min-sessions" >&2; exit 2; }
            MIN_SESSIONS="$2"
            shift 2
            ;;
        --evolution-batch-size)
            [[ $# -ge 2 ]] || { echo "Missing value for --evolution-batch-size" >&2; exit 2; }
            EVOLUTION_BATCH_SIZE="$2"
            shift 2
            ;;
        --stop-on-error)
            CONTINUE_ON_ERROR=0
            shift
            ;;
        --no-rebuild-splits)
            REBUILD_SPLITS=0
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

case "$METHOD" in
    all|awm|trace2skill|graph) ;;
    *)
        echo "Invalid --method: $METHOD" >&2
        exit 2
        ;;
esac

if ! command -v conda >/dev/null 2>&1; then
    for conda_sh in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [[ -f "$conda_sh" ]]; then
            # shellcheck disable=SC1090
            source "$conda_sh"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Initialize conda before running this script." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base 2>/dev/null)" || {
    echo "Unable to determine the conda installation path." >&2
    exit 1
}

if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

conda activate "$CONDA_ENV" || {
    echo "Unable to activate conda environment: $CONDA_ENV" >&2
    exit 1
}

export HF_ENDPOINT="$HF_MIRROR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python was not found after activating $CONDA_ENV." >&2
    exit 1
fi

echo "Environment: $CONDA_ENV"
echo "Python:      $($PYTHON_BIN --version 2>&1)"
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "Root:        $ROOT_DIR"

SPLITS_DIR="$ROOT_DIR/data/eval/abcd/splits"
OUTPUT_DIR="$ROOT_DIR/outputs"
mkdir -p "$OUTPUT_DIR"

if [[ "$REBUILD_SPLITS" -eq 1 ]]; then
    echo "Building shared per-subflow session splits..."
    "$PYTHON_BIN" scripts/split_abcd_by_intent.py --seed 42 || {
        echo "Failed to build shared ABCD splits." >&2
        exit 1
    }
fi
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
MANIFEST="$OUTPUT_DIR/full_abcd_${RUN_ID}_manifest.txt"
: > "$MANIFEST"

echo "Run ID:     $RUN_ID"
echo "Manifest:   $MANIFEST"
echo "Output root: $OUTPUT_DIR"

SUBFLOWS=()
if [[ -n "$ONE_SUBFLOW" ]]; then
    SUBFLOWS=("$ONE_SUBFLOW")
else
    for split_dir in "$SPLITS_DIR"/*; do
        [[ -d "$split_dir" ]] || continue
        SUBFLOWS+=("${split_dir##*/}")
    done
fi

if [[ ${#SUBFLOWS[@]} -eq 0 ]]; then
    echo "No subflow directories found under $SPLITS_DIR" >&2
    exit 1
fi

printf 'Subflows:   %d\n' "${#SUBFLOWS[@]}"
printf 'Methods:    %s\n' "$METHOD"

FAILED=()
AWM_RUNS=()
TRACE_RUNS=()
GRAPH_RUNS=()

latest_run_dir() {
    local pattern="$1"
    find "$OUTPUT_DIR" -maxdepth 1 -type d -name "$pattern" -printf '%T@ %p\n' 2>/dev/null |
        sort -nr |
        head -n 1 |
        cut -d' ' -f2-
}

run_subflow_command() {
    local method_name="$1"
    local subflow="$2"
    shift 2

    echo
    echo "===== $method_name / $subflow ====="
    "$PYTHON_BIN" "$@"
    local status=$?
    if [[ $status -ne 0 ]]; then
        FAILED+=("$method_name:$subflow")
        echo "FAILED: $method_name / $subflow (exit=$status)" >&2
        if [[ $CONTINUE_ON_ERROR -eq 0 ]]; then
            exit "$status"
        fi
        return "$status"
    fi
    return 0
}

if [[ "$METHOD" == "all" || "$METHOD" == "awm" ]]; then
    for subflow in "${SUBFLOWS[@]}"; do
        run_subflow_command "AWM" "$subflow" \
            scripts/run_awm_abcd.py --subflow "$subflow" || continue
        run_dir="$(latest_run_dir 'awm_abcd_*')"
        [[ -n "$run_dir" ]] && AWM_RUNS+=("$run_dir")
    done
fi

if [[ "$METHOD" == "all" || "$METHOD" == "trace2skill" ]]; then
    for subflow in "${SUBFLOWS[@]}"; do
        run_subflow_command "Trace2Skill" "$subflow" \
            scripts/run_trace2skill_abcd.py \
            --subflow "$subflow" \
            --train-file "$SPLITS_DIR/$subflow/train.json" \
            --test-file "$SPLITS_DIR/$subflow/test.json" \
            --evolution-batch-size "$EVOLUTION_BATCH_SIZE" \
            --continue-on-batch-error || continue
        run_dir="$(latest_run_dir 'abcd_trace2skill_*')"
        [[ -n "$run_dir" ]] && TRACE_RUNS+=("$run_dir")
    done
fi

if [[ "$METHOD" == "all" || "$METHOD" == "graph" ]]; then
    echo
    echo "===== Graph Mining / independent subflows ====="
    if [[ -n "$ONE_SUBFLOW" ]]; then
        "$PYTHON_BIN" scripts/run_subflow_eval.py --subflow "$ONE_SUBFLOW" --min-sessions "$MIN_SESSIONS"
    else
        "$PYTHON_BIN" scripts/run_subflow_eval.py --all --min-sessions "$MIN_SESSIONS"
    fi
    graph_status=$?
    if [[ $graph_status -ne 0 ]]; then
        FAILED+=("Graph Mining:all")
        if [[ $CONTINUE_ON_ERROR -eq 0 ]]; then
            exit "$graph_status"
        fi
    else
        run_dir="$(latest_run_dir 'subflow_eval_*')"
        [[ -n "$run_dir" ]] && GRAPH_RUNS+=("$run_dir")
    fi
fi

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"

AGGREGATE_FILES=()

if [[ ${#AWM_RUNS[@]} -gt 0 ]]; then
    aggregate_path="$OUTPUT_DIR/aggregate_awm_$STAMP.json"
    "$PYTHON_BIN" scripts/aggregate_subflow_results.py \
        --runs "${AWM_RUNS[@]}" \
        --output "$aggregate_path"
    AGGREGATE_FILES+=("$aggregate_path")
fi

if [[ ${#TRACE_RUNS[@]} -gt 0 ]]; then
    aggregate_path="$OUTPUT_DIR/aggregate_trace2skill_$STAMP.json"
    "$PYTHON_BIN" scripts/aggregate_subflow_results.py \
        --runs "${TRACE_RUNS[@]}" \
        --output "$aggregate_path"
    AGGREGATE_FILES+=("$aggregate_path")
fi

if [[ ${#GRAPH_RUNS[@]} -gt 0 ]]; then
    aggregate_path="$OUTPUT_DIR/aggregate_graph_mining_$STAMP.json"
    "$PYTHON_BIN" scripts/aggregate_subflow_results.py \
        --runs "${GRAPH_RUNS[@]}" \
        --output "$aggregate_path"
    AGGREGATE_FILES+=("$aggregate_path")
fi

{
    echo "run_id=$RUN_ID"
    echo "method=$METHOD"
    echo "hf_endpoint=$HF_ENDPOINT"
    echo "output_root=$OUTPUT_DIR"
    echo ""
    echo "[AWM run directories]"
    printf '%s\n' "${AWM_RUNS[@]}"
    echo ""
    echo "[Trace2Skill run directories]"
    printf '%s\n' "${TRACE_RUNS[@]}"
    echo ""
    echo "[Graph Mining run directories]"
    printf '%s\n' "${GRAPH_RUNS[@]}"
    echo ""
    echo "[Aggregate files]"
    printf '%s\n' "${AGGREGATE_FILES[@]}"
    echo ""
    echo "[Failed runs]"
    printf '%s\n' "${FAILED[@]}"
} > "$MANIFEST"

echo
echo "===== Full ABCD experiment finished ====="
echo "Result and artifact locations:"
echo "  Manifest: $MANIFEST"
echo "  Output root: $OUTPUT_DIR"
if [[ ${#AWM_RUNS[@]} -gt 0 ]]; then
    echo "  AWM run directories:"
    printf '    %s\n' "${AWM_RUNS[@]}"
fi
if [[ ${#TRACE_RUNS[@]} -gt 0 ]]; then
    echo "  Trace2Skill run directories:"
    printf '    %s\n' "${TRACE_RUNS[@]}"
fi
if [[ ${#GRAPH_RUNS[@]} -gt 0 ]]; then
    echo "  Graph Mining run directories:"
    printf '    %s\n' "${GRAPH_RUNS[@]}"
fi
if [[ ${#AGGREGATE_FILES[@]} -gt 0 ]]; then
    echo "  Aggregate files:"
    printf '    %s\n' "${AGGREGATE_FILES[@]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf 'Failed runs:\n'
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
echo "All requested runs completed successfully."
