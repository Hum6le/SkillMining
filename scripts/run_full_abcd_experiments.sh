#!/usr/bin/env bash

# Run ABCD under the independent-subflow protocol.
#
# When --workflow-ids is supplied, subflows are balanced by actual agent-turn
# volume and each bucket is executed by one worker bound to one workflow API.
# This avoids config.py races and keeps requests to one workflow serial.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="skillmining310"
HF_MIRROR="https://hf-mirror.com"
METHOD="all"
ONE_SUBFLOW=""
MIN_SESSIONS=0
GRAPH_MINING_METHOD="legacy"
BACKBONE_COVERAGE_LAMBDA="0.2"
SKIP_GRAPH_SEED=1
EVOLUTION_BATCH_SIZE=25
CONTINUE_ON_ERROR=1
PYTHON_BIN="python"
REBUILD_SPLITS=1
WORKFLOW_IDS_RAW=""

usage() {
    cat <<'EOF'
Usage: bash scripts/run_full_abcd_experiments.sh [options]

Options:
  --method NAME              all, awm, expel, trace2skill, or graph (default: all)
  --subflow NAME             Run one subflow instead of all complete split directories
  --workflow-ids IDS         Comma-separated workflow IDs. One balanced, serial worker is
                             started per ID; workers run in parallel. Example: id_a,id_b,id_c
  --min-sessions N           Graph Mining minimum train sessions (default: 0)
  --graph-mining-method NAME legacy, sequence, backbone, or backbone_coverage (default: legacy)
  --backbone-coverage-lambda N  Session coverage weight for backbone_coverage (default: 0.2)
  --with-graph-seed          Also run the empty-workflow HG seed baseline
  --evolution-batch-size N   Trace2Skill outer batch size (default: 25)
  --stop-on-error            Stop the affected worker at its first failed subflow
  --no-rebuild-splits        Reuse existing subflow session splits
  -h, --help                 Show this help

Worker load is balanced by train+test non-empty agent utterance turns, not
conversation count. The selected workflow ID is exported as
SKILLMINING_WORKFLOW_ID; the workflow-aware llm.py must honor this override.
Without --workflow-ids, one serial worker uses config.py unchanged.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method) METHOD="$2"; shift 2 ;;
        --subflow) ONE_SUBFLOW="$2"; shift 2 ;;
        --workflow-ids) WORKFLOW_IDS_RAW="$2"; shift 2 ;;
        --min-sessions) MIN_SESSIONS="$2"; shift 2 ;;
        --graph-mining-method) GRAPH_MINING_METHOD="$2"; shift 2 ;;
        --backbone-coverage-lambda) BACKBONE_COVERAGE_LAMBDA="$2"; shift 2 ;;
        --with-graph-seed) SKIP_GRAPH_SEED=0; shift ;;
        --evolution-batch-size) EVOLUTION_BATCH_SIZE="$2"; shift 2 ;;
        --stop-on-error) CONTINUE_ON_ERROR=0; shift ;;
        --no-rebuild-splits) REBUILD_SPLITS=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$METHOD" in all|awm|expel|trace2skill|graph) ;; *) echo "Invalid --method: $METHOD" >&2; exit 2 ;; esac
case "$GRAPH_MINING_METHOD" in legacy|sequence|backbone|backbone_coverage) ;; *) echo "Invalid --graph-mining-method: $GRAPH_MINING_METHOD" >&2; exit 2 ;; esac

if ! command -v conda >/dev/null 2>&1; then
    for conda_sh in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
        [[ -f "$conda_sh" ]] && source "$conda_sh" && break
    done
fi
command -v conda >/dev/null 2>&1 || { echo "conda was not found." >&2; exit 1; }
CONDA_BASE="$(conda info --base 2>/dev/null)" || exit 1
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "Unable to activate $CONDA_ENV" >&2; exit 1; }
export HF_ENDPOINT="$HF_MIRROR"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python was not found." >&2; exit 1; }

SPLITS_DIR="$ROOT_DIR/data/eval/abcd/splits"
OUTPUT_DIR="$ROOT_DIR/outputs"
mkdir -p "$OUTPUT_DIR"
if [[ "$REBUILD_SPLITS" -eq 1 ]]; then
    echo "Building shared per-subflow session splits..."
    "$PYTHON_BIN" scripts/split_abcd_by_intent.py --seed 42 || exit 1
fi

SUBFLOWS=()
if [[ -n "$ONE_SUBFLOW" ]]; then
    SUBFLOWS=("$ONE_SUBFLOW")
else
    for split_dir in "$SPLITS_DIR"/*; do
        [[ -d "$split_dir" && -f "$split_dir/train.json" && -f "$split_dir/test.json" ]] || continue
        SUBFLOWS+=("${split_dir##*/}")
    done
fi
[[ ${#SUBFLOWS[@]} -gt 0 ]] || { echo "No subflows found under $SPLITS_DIR" >&2; exit 1; }

WORKFLOW_IDS=()
if [[ -n "$WORKFLOW_IDS_RAW" ]]; then
    IFS=',' read -r -a raw_ids <<< "$WORKFLOW_IDS_RAW"
    for id in "${raw_ids[@]}"; do
        id="${id//[[:space:]]/}"
        [[ -n "$id" ]] && WORKFLOW_IDS+=("$id")
    done
    [[ ${#WORKFLOW_IDS[@]} -gt 0 ]] || { echo "--workflow-ids contains no IDs" >&2; exit 2; }
else
    WORKFLOW_IDS=("")
fi

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ROOT="$OUTPUT_DIR/full_abcd_$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"
PLAN_PATH="$RUN_ROOT/workflow_load_plan.json"
"$PYTHON_BIN" scripts/plan_abcd_workflow_loads.py \
    --splits-dir "$SPLITS_DIR" --workers "${#WORKFLOW_IDS[@]}" \
    --subflows "${SUBFLOWS[@]}" --output "$PLAN_PATH"

echo "Environment: $CONDA_ENV"
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "Run root:    $RUN_ROOT"
echo "Load plan:   $PLAN_PATH"
echo "Subflows:    ${#SUBFLOWS[@]}"
echo "Workers:     ${#WORKFLOW_IDS[@]}"

worker_subflows() {
    "$PYTHON_BIN" - "$PLAN_PATH" "$1" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
for item in plan["workers"][int(sys.argv[2])]["subflows"]:
    print(item["name"])
PY
}

train_session_count() {
    "$PYTHON_BIN" - "$SPLITS_DIR/$1/train.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
}

run_task() {
    local worker_index="$1" workflow_id="$2" method_name="$3" subflow="$4"
    shift 4
    local task_dir="$RUN_ROOT/$method_name/$subflow"
    mkdir -p "$task_dir"
    echo "===== worker=$worker_index workflow=${workflow_id:-config.py} method=$method_name subflow=$subflow ====="
    SKILLMINING_WORKFLOW_ID="$workflow_id" ABCD_OUTPUT_DIR="$task_dir" "$PYTHON_BIN" "$@"
}

run_worker() {
    local worker_index="$1" workflow_id="$2"
    local failed_path="$RUN_ROOT/worker_${worker_index}_failed.txt"
    : > "$failed_path"
    mapfile -t assigned < <(worker_subflows "$worker_index")
    for subflow in "${assigned[@]}"; do
        [[ -n "$subflow" ]] || continue
        if [[ "$METHOD" == "all" || "$METHOD" == "awm" ]]; then
            run_task "$worker_index" "$workflow_id" awm "$subflow" scripts/run_awm_abcd.py --subflow "$subflow" || {
                echo "awm:$subflow" >> "$failed_path"; [[ "$CONTINUE_ON_ERROR" -eq 0 ]] && return 1; }
        fi
        if [[ "$METHOD" == "all" || "$METHOD" == "expel" ]]; then
            run_task "$worker_index" "$workflow_id" expel "$subflow" scripts/run_expel_abcd.py --subflow "$subflow" || {
                echo "expel:$subflow" >> "$failed_path"; [[ "$CONTINUE_ON_ERROR" -eq 0 ]] && return 1; }
        fi
        if [[ "$METHOD" == "all" || "$METHOD" == "trace2skill" ]]; then
            # Trace2Skill nests its timestamped directory below this unique
            # subflow parent; final aggregation searches recursively.
            mkdir -p "$RUN_ROOT/trace2skill/$subflow"
            SKILLMINING_WORKFLOW_ID="$workflow_id" "$PYTHON_BIN" scripts/run_trace2skill_abcd.py \
                --subflow "$subflow" --train-file "$SPLITS_DIR/$subflow/train.json" \
                --test-file "$SPLITS_DIR/$subflow/test.json" --output-dir "$RUN_ROOT/trace2skill/$subflow" \
                --evolution-batch-size "$EVOLUTION_BATCH_SIZE" --continue-on-batch-error || {
                echo "trace2skill:$subflow" >> "$failed_path"; [[ "$CONTINUE_ON_ERROR" -eq 0 ]] && return 1; }
        fi
        if [[ "$METHOD" == "all" || "$METHOD" == "graph" ]]; then
            train_sessions="$(train_session_count "$subflow")"
            if [[ "$train_sessions" -lt "$MIN_SESSIONS" ]]; then
                echo "Skipping graph/$subflow: train_sessions=$train_sessions < min_sessions=$MIN_SESSIONS"
                continue
            fi
            graph_args=(scripts/run_subflow_eval.py --subflow "$subflow" --min-sessions "$MIN_SESSIONS"
                --mining-method "$GRAPH_MINING_METHOD" --backbone-coverage-lambda "$BACKBONE_COVERAGE_LAMBDA")
            [[ "$SKIP_GRAPH_SEED" -eq 1 ]] && graph_args+=(--skip-seed)
            run_task "$worker_index" "$workflow_id" graph "$subflow" "${graph_args[@]}" || {
                echo "graph:$subflow" >> "$failed_path"; [[ "$CONTINUE_ON_ERROR" -eq 0 ]] && return 1; }
        fi
    done
}

PIDS=()
for index in "${!WORKFLOW_IDS[@]}"; do
    run_worker "$index" "${WORKFLOW_IDS[$index]}" > "$LOG_DIR/worker_${index}.log" 2>&1 &
    pid="$!"
    PIDS+=("$pid")
    echo "Started worker $index (PID $pid), log: $LOG_DIR/worker_${index}.log"
done

WORKER_FAILURE=0
for index in "${!PIDS[@]}"; do
    wait "${PIDS[$index]}" || WORKER_FAILURE=1
done

AGGREGATES=()
aggregate_method() {
    local method_name="$1" output="$RUN_ROOT/aggregate_${method_name}.json"
    [[ -d "$RUN_ROOT/$method_name" ]] || return 0
    "$PYTHON_BIN" scripts/aggregate_subflow_results.py --runs "$RUN_ROOT/$method_name" --recursive --output "$output" && AGGREGATES+=("$output") || true
}
[[ "$METHOD" == "all" || "$METHOD" == "awm" ]] && aggregate_method awm
[[ "$METHOD" == "all" || "$METHOD" == "expel" ]] && aggregate_method expel
[[ "$METHOD" == "all" || "$METHOD" == "trace2skill" ]] && aggregate_method trace2skill
[[ "$METHOD" == "all" || "$METHOD" == "graph" ]] && aggregate_method graph

MANIFEST="$RUN_ROOT/manifest.txt"
{
    echo "run_id=$RUN_ID"
    echo "method=$METHOD"
    echo "workflow_ids=${WORKFLOW_IDS_RAW:-config.py}"
    echo "run_root=$RUN_ROOT"
    echo "load_plan=$PLAN_PATH"
    echo ""
    echo "[worker_logs]"
    find "$LOG_DIR" -type f -name 'worker_*.log' -print | sort
    echo ""
    echo "[aggregate_files]"
    printf '%s\n' "${AGGREGATES[@]}"
    echo ""
    echo "[failed_tasks]"
    cat "$RUN_ROOT"/worker_*_failed.txt 2>/dev/null || true
} > "$MANIFEST"

echo "===== Full ABCD experiment finished ====="
echo "Run root:  $RUN_ROOT"
echo "Load plan: $PLAN_PATH"
echo "Manifest:  $MANIFEST"
echo "Worker logs: $LOG_DIR"
if [[ ${#AGGREGATES[@]} -gt 0 ]]; then
    echo "Aggregate files:"
    printf '  %s\n' "${AGGREGATES[@]}"
fi
if [[ "$WORKER_FAILURE" -ne 0 ]] || grep -q . "$RUN_ROOT"/worker_*_failed.txt 2>/dev/null; then
    echo "Some subflow tasks failed; see $MANIFEST and worker logs." >&2
    exit 1
fi
echo "All requested tasks completed successfully."
