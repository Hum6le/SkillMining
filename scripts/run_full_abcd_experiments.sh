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
RESUME_RUN=""

usage() {
    cat <<'EOF'
Usage: bash scripts/run_full_abcd_experiments.sh [options]

Options:
  --method NAME              all, awm, expel, trace2skill, or graph (default: all)
  --subflow NAME             Run one subflow instead of all complete split directories
  --workflow-ids IDS         Comma-separated workflow IDs. One balanced, serial worker is
                             started per ID; workers run in parallel. Example: id_a,id_b,id_c
  --resume-run DIR           Resume an existing outputs/full_abcd_* run. Reuses its load
                             plan and skips subflows with a complete final summary.
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
Resume does not require a manifest. If the old run has no load plan, a new
plan is created from the current split list and the supplied workflow IDs.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method) METHOD="$2"; shift 2 ;;
        --subflow) ONE_SUBFLOW="$2"; shift 2 ;;
        --workflow-ids) WORKFLOW_IDS_RAW="$2"; shift 2 ;;
        --resume-run) RESUME_RUN="$2"; shift 2 ;;
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
if [[ -n "$RESUME_RUN" ]]; then
    REBUILD_SPLITS=0
fi
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

RUN_ID=""
RUN_ROOT=""
PLAN_PATH=""
if [[ -n "$RESUME_RUN" ]]; then
    RUN_ROOT="$(cd "$RESUME_RUN" 2>/dev/null && pwd)" || {
        echo "--resume-run does not exist: $RESUME_RUN" >&2; exit 2;
    }
    PLAN_PATH="$RUN_ROOT/workflow_load_plan.json"
    RUN_ID="${RUN_ROOT##*/}"
fi

WORKFLOW_IDS=()
# Resume the original worker-to-workflow mapping from worker logs when the
# caller did not provide IDs. The run log contains lines such as
# ``workflow=<id> method=...`` for each worker's tasks.
if [[ -n "$RESUME_RUN" && -z "$WORKFLOW_IDS_RAW" ]]; then
    WORKFLOW_IDS_RAW="$($PYTHON_BIN - "$RUN_ROOT/logs" <<'PY'
import re
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
ids = []
for path in sorted(log_dir.glob("worker_*.log")):
    text = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""
    match = re.search(r"\bworkflow=([^\s]+)", text)
    if match:
        value = match.group(1).strip()
        if value and value != "config.py":
            ids.append(value)
if ids:
    print(",".join(ids))
PY
    )"
    if [[ -n "$WORKFLOW_IDS_RAW" ]]; then
        echo "Recovered workflow IDs from worker logs: ${#WORKFLOW_IDS_RAW} characters across $(awk -F, '{print NF}' <<< "$WORKFLOW_IDS_RAW") worker(s)."
    fi
fi
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

if [[ -z "$RUN_ROOT" ]]; then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
    RUN_ROOT="$OUTPUT_DIR/full_abcd_$RUN_ID"
    PLAN_PATH="$RUN_ROOT/workflow_load_plan.json"
fi
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"
if [[ -n "$RESUME_RUN" ]]; then
    if [[ -f "$PLAN_PATH" ]]; then
        planned_workers="$("$PYTHON_BIN" - "$PLAN_PATH" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["workers"]))
PY
        )"
    else
        echo "Resume run has no workflow_load_plan.json; creating a new plan from current splits."
        "$PYTHON_BIN" scripts/plan_abcd_workflow_loads.py \
            --splits-dir "$SPLITS_DIR" --workers "${#WORKFLOW_IDS[@]}" \
            --subflows "${SUBFLOWS[@]}" --output "$PLAN_PATH"
        planned_workers="${#WORKFLOW_IDS[@]}"
    fi
    [[ "$planned_workers" -eq "${#WORKFLOW_IDS[@]}" ]] || {
        echo "Resume worker count mismatch: plan has $planned_workers, but ${#WORKFLOW_IDS[@]} workflow ID(s) were supplied/recovered." >&2
        exit 2
    }
    if [[ -n "$ONE_SUBFLOW" ]]; then
        "$PYTHON_BIN" - "$PLAN_PATH" "$ONE_SUBFLOW" <<'PY' || {
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
target = sys.argv[2]
raise SystemExit(0 if any(
    item["name"] == target
    for worker in plan["workers"]
    for item in worker["subflows"]
) else 1)
PY
            echo "Subflow $ONE_SUBFLOW is not part of resume plan: $PLAN_PATH" >&2
            exit 2
        }
    fi
else
    "$PYTHON_BIN" scripts/plan_abcd_workflow_loads.py \
        --splits-dir "$SPLITS_DIR" --workers "${#WORKFLOW_IDS[@]}" \
        --subflows "${SUBFLOWS[@]}" --output "$PLAN_PATH"
fi

echo "Environment: $CONDA_ENV"
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "Run root:    $RUN_ROOT"
echo "Load plan:   $PLAN_PATH"
echo "Subflows:    ${#SUBFLOWS[@]}"
echo "Workers:     ${#WORKFLOW_IDS[@]}"
[[ -n "$RESUME_RUN" ]] && echo "Resume:      enabled (completed subflows will be skipped)"

worker_subflows() {
    "$PYTHON_BIN" - "$PLAN_PATH" "$1" "$ONE_SUBFLOW" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
requested = sys.argv[3]
for item in plan["workers"][int(sys.argv[2])]["subflows"]:
    if not requested or item["name"] == requested:
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

task_is_complete() {
    local method_name="$1" subflow="$2" task_dir="$RUN_ROOT/$method_name/$subflow"
    "$PYTHON_BIN" - "$method_name" "$subflow" "$task_dir" <<'PY'
import json
import sys
from pathlib import Path

method, subflow, task_dir = sys.argv[1:]
root = Path(task_dir)
paths = [root / "summary.json"]
if method == "trace2skill" and root.exists():
    paths.extend(root.glob("**/summary.json"))

for path in paths:
    if not path.is_file():
        continue
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if method == "graph":
        row = summary.get(subflow) if isinstance(summary, dict) else None
        if isinstance(row, dict) and isinstance(row.get("mined"), dict):
            raise SystemExit(0)
    elif isinstance(summary, dict):
        config = summary.get("config", {})
        if str(config.get("subflow", "")) != subflow:
            continue
        if method == "trace2skill" and isinstance(summary.get("evolved_test"), dict):
            raise SystemExit(0)
        if method in {"awm", "expel"} and isinstance(summary.get("final_test"), dict):
            raise SystemExit(0)
raise SystemExit(1)
PY
}

run_or_resume_task() {
    local worker_index="$1" workflow_id="$2" method_name="$3" subflow="$4"
    shift 4
    if [[ -n "$RESUME_RUN" ]] && task_is_complete "$method_name" "$subflow"; then
        echo "SKIP completed: worker=$worker_index method=$method_name subflow=$subflow"
        return 0
    fi
    run_task "$worker_index" "$workflow_id" "$method_name" "$subflow" "$@"
}

run_worker() {
    local worker_index="$1" workflow_id="$2"
    local failed_path="$RUN_ROOT/worker_${worker_index}_failed.txt"
    : > "$failed_path"
    mapfile -t assigned < <(worker_subflows "$worker_index")
    for subflow in "${assigned[@]}"; do
        [[ -n "$subflow" ]] || continue
        if [[ "$METHOD" == "all" || "$METHOD" == "awm" ]]; then
            run_or_resume_task "$worker_index" "$workflow_id" awm "$subflow" scripts/run_awm_abcd.py --subflow "$subflow" || {
                echo "awm:$subflow" >> "$failed_path"; [[ "$CONTINUE_ON_ERROR" -eq 0 ]] && return 1; }
        fi
        if [[ "$METHOD" == "all" || "$METHOD" == "expel" ]]; then
            run_or_resume_task "$worker_index" "$workflow_id" expel "$subflow" scripts/run_expel_abcd.py --subflow "$subflow" || {
                echo "expel:$subflow" >> "$failed_path"; [[ "$CONTINUE_ON_ERROR" -eq 0 ]] && return 1; }
        fi
        if [[ "$METHOD" == "all" || "$METHOD" == "trace2skill" ]]; then
            # Trace2Skill nests its timestamped directory below this unique
            # subflow parent; final aggregation searches recursively.
            if [[ -n "$RESUME_RUN" ]] && task_is_complete trace2skill "$subflow"; then
                echo "SKIP completed: worker=$worker_index method=trace2skill subflow=$subflow"
            else
                mkdir -p "$RUN_ROOT/trace2skill/$subflow"
                trace_resume_args=()
                if [[ -n "$RESUME_RUN" ]]; then
                    mapfile -t trace_candidates < <(find "$RUN_ROOT/trace2skill/$subflow" -mindepth 1 -maxdepth 1 -type d -name 'abcd_trace2skill_*' | sort)
                    if [[ ${#trace_candidates[@]} -eq 1 ]]; then
                        trace_resume_args=(--resume-dir "${trace_candidates[0]}")
                        echo "Resuming Trace2Skill checkpoint: ${trace_candidates[0]}"
                    fi
                fi
                SKILLMINING_WORKFLOW_ID="$workflow_id" "$PYTHON_BIN" scripts/run_trace2skill_abcd.py \
                    --subflow "$subflow" --train-file "$SPLITS_DIR/$subflow/train.json" \
                    --test-file "$SPLITS_DIR/$subflow/test.json" --output-dir "$RUN_ROOT/trace2skill/$subflow" \
                    --evolution-batch-size "$EVOLUTION_BATCH_SIZE" --continue-on-batch-error "${trace_resume_args[@]}" || {
                echo "trace2skill:$subflow" >> "$failed_path"; [[ "$CONTINUE_ON_ERROR" -eq 0 ]] && return 1; }
            fi
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
            run_or_resume_task "$worker_index" "$workflow_id" graph "$subflow" "${graph_args[@]}" || {
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
AGGREGATE_DIRS=()
AGGREGATE_FAILURE=0
aggregate_method() {
    local method_name="$1" output="$RUN_ROOT/aggregate_${method_name}.json"
    [[ -d "$RUN_ROOT/$method_name" ]] || return 0
    AGGREGATE_DIRS+=("$RUN_ROOT/$method_name")
    if "$PYTHON_BIN" scripts/aggregate_subflow_results.py \
        --runs "$RUN_ROOT/$method_name" --recursive --output "$output"; then
        AGGREGATES+=("$output")
    else
        AGGREGATE_FAILURE=1
        echo "Aggregation failed for method=$method_name; inspect summaries under $RUN_ROOT/$method_name." >&2
    fi
}
[[ "$METHOD" == "all" || "$METHOD" == "awm" ]] && aggregate_method awm
[[ "$METHOD" == "all" || "$METHOD" == "expel" ]] && aggregate_method expel
[[ "$METHOD" == "all" || "$METHOD" == "trace2skill" ]] && aggregate_method trace2skill
[[ "$METHOD" == "all" || "$METHOD" == "graph" ]] && aggregate_method graph

FINAL_SUMMARY="$RUN_ROOT/final_summary.json"
if [[ ${#AGGREGATE_DIRS[@]} -gt 0 ]]; then
    if "$PYTHON_BIN" scripts/aggregate_subflow_results.py \
        --runs "${AGGREGATE_DIRS[@]}" --recursive --output "$FINAL_SUMMARY"; then
        echo "Final summary: $FINAL_SUMMARY"
    else
        AGGREGATE_FAILURE=1
        echo "Final cross-method aggregation failed; inspect worker outputs under $RUN_ROOT." >&2
    fi
fi

echo "===== Full ABCD experiment finished ====="
echo "Run root:  $RUN_ROOT"
echo "Load plan: $PLAN_PATH"
echo "Worker logs: $LOG_DIR"
if [[ ${#AGGREGATES[@]} -gt 0 ]]; then
    echo "Aggregate files:"
    printf '  %s\n' "${AGGREGATES[@]}"
fi
if [[ "$WORKER_FAILURE" -ne 0 ]] || [[ "$AGGREGATE_FAILURE" -ne 0 ]] || grep -q . "$RUN_ROOT"/worker_*_failed.txt 2>/dev/null; then
    echo "Some subflow tasks failed; see worker failure files and logs under $RUN_ROOT." >&2
    exit 1
fi
echo "All requested tasks completed successfully."
