#!/usr/bin/env bash

# Run online backbone refinement independently for all current ABCD coarse
# flows. Subflows are balanced by agent-turn volume and assigned to one
# workflow API per worker; each worker remains serial to respect API limits.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1

CONDA_ENV="skillmining310"
HF_ENDPOINT_VALUE="https://hf-mirror.com"
PYTHON_BIN="python"
OFFLINE_ROOT=""
OUTPUT_DIR=""
RESUME_RUN=""
WORKFLOW_IDS_RAW=""
REBUILD_SPLITS=1
CONTINUE_ON_ERROR=1
RUNNER_ARGS=()
PIDS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/run_full_backbone_online_refine.sh [options] [online-runner options]

Options:
  --offline-root DIR        Root containing DIR/<subflow>/{subgraph.json,skill.md}.
                            If omitted, every subflow re-mines its offline backbone.
  --output-dir DIR          New full-run root. Default: outputs/full_online_refine_<timestamp>
  --resume-run DIR          Resume an existing full online-refinement root.
  --workflow-ids IDS        Comma-separated workflow IDs. Subflows are balanced across
                            these workers; each worker processes its assigned flows serially.
  --conda-env NAME          Default: skillmining310
  --hf-endpoint URL         Default: https://hf-mirror.com
  --python-bin PATH         Default: python
  --no-rebuild-splits       Reuse the current 10-flow ABCD splits.
  --stop-on-error           Stop an affected worker after its first failed subflow.
  -h, --help                Show this help.

All other options are forwarded to run_backbone_online_refine.py, for example:
  --batch-size 8 --target-selection-rate 0.30 --eval-workflow-ids id_a,id_b

The final weighted result is saved as <run-root>/aggregate_online_refine.json.
EOF
}

require_value() {
    [[ "$2" -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
}

terminate_descendants() {
    local parent_pid="$1" child_pid
    while IFS= read -r child_pid; do
        [[ -n "$child_pid" ]] || continue
        terminate_descendants "$child_pid"
        kill -TERM "$child_pid" 2>/dev/null || true
    done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
}

stop_workers() {
    local signal_name="$1" pid
    echo "Received ${signal_name}; stopping ${#PIDS[@]} worker(s)..." >&2
    for pid in "${PIDS[@]}"; do
        terminate_descendants "$pid"
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}

trap 'stop_workers INT' INT
trap 'stop_workers TERM' TERM
trap 'stop_workers HUP' HUP

while [[ $# -gt 0 ]]; do
    case "$1" in
        --offline-root) require_value "$1" "$#"; OFFLINE_ROOT="$2"; shift 2 ;;
        --output-dir) require_value "$1" "$#"; OUTPUT_DIR="$2"; shift 2 ;;
        --resume-run) require_value "$1" "$#"; RESUME_RUN="$2"; shift 2 ;;
        --workflow-ids) require_value "$1" "$#"; WORKFLOW_IDS_RAW="$2"; shift 2 ;;
        --conda-env) require_value "$1" "$#"; CONDA_ENV="$2"; shift 2 ;;
        --hf-endpoint) require_value "$1" "$#"; HF_ENDPOINT_VALUE="$2"; shift 2 ;;
        --python-bin) require_value "$1" "$#"; PYTHON_BIN="$2"; shift 2 ;;
        --no-rebuild-splits) REBUILD_SPLITS=0; shift ;;
        --stop-on-error) CONTINUE_ON_ERROR=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) RUNNER_ARGS+=("$1"); shift ;;
    esac
done

[[ -z "$RESUME_RUN" || -z "$OUTPUT_DIR" ]] || {
    echo "--output-dir must be omitted when --resume-run is used." >&2; exit 2;
}
[[ -z "$OFFLINE_ROOT" || -d "$OFFLINE_ROOT" ]] || {
    echo "--offline-root does not exist: $OFFLINE_ROOT" >&2; exit 2;
}

if ! command -v conda >/dev/null 2>&1; then
    for conda_sh in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
        [[ -f "$conda_sh" ]] && source "$conda_sh" && break
    done
fi
command -v conda >/dev/null 2>&1 || { echo "conda was not found." >&2; exit 1; }
CONDA_BASE="$(conda info --base 2>/dev/null)" || exit 1
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "Unable to activate $CONDA_ENV" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python was not found: $PYTHON_BIN" >&2; exit 1; }
export HF_ENDPOINT="$HF_ENDPOINT_VALUE"

SPLITS_DIR="$ROOT_DIR/data/eval/abcd/splits"
EXPECTED_FLOW_COUNT=10
if [[ "$REBUILD_SPLITS" -eq 1 && -z "$RESUME_RUN" ]]; then
    "$PYTHON_BIN" scripts/split_abcd_by_intent.py --seed 42 || exit 1
fi

SPLIT_INDEX="$SPLITS_DIR/INDEX.json"
[[ -f "$SPLIT_INDEX" ]] || { echo "Missing split index: $SPLIT_INDEX" >&2; exit 1; }
mapfile -t SUBFLOWS < <("$PYTHON_BIN" - "$SPLIT_INDEX" "$SPLITS_DIR" <<'PY'
import json
import sys
from pathlib import Path

index = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
for name in sorted(index):
    if (root / name / "train.json").is_file() and (root / name / "test.json").is_file():
        print(name)
PY
)
[[ ${#SUBFLOWS[@]} -eq "$EXPECTED_FLOW_COUNT" ]] || {
    echo "Expected the current 10-flow split, found ${#SUBFLOWS[@]}." >&2; exit 1;
}

if [[ -n "$RESUME_RUN" ]]; then
    RUN_ROOT="$(cd "$RESUME_RUN" 2>/dev/null && pwd)" || {
        echo "--resume-run does not exist: $RESUME_RUN" >&2; exit 2;
    }
else
    RUN_ROOT="${OUTPUT_DIR:-$ROOT_DIR/outputs/full_online_refine_$(date +%Y-%m-%d_%H-%M-%S)}"
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/subflows"
PLAN_PATH="$RUN_ROOT/workflow_load_plan.json"
WORKFLOW_PATH="$RUN_ROOT/workflow_ids.txt"
MANIFEST_PATH="$RUN_ROOT/full_online_refine_manifest.txt"
if [[ -n "$RESUME_RUN" && -z "$OFFLINE_ROOT" && -f "$MANIFEST_PATH" ]]; then
    saved_offline_root="$(sed -n 's/^offline_root=//p' "$MANIFEST_PATH" | head -n 1)"
    if [[ -n "$saved_offline_root" && "$saved_offline_root" != "offline_remining" && -d "$saved_offline_root" ]]; then
        OFFLINE_ROOT="$saved_offline_root"
        echo "Recovered offline root from manifest: $OFFLINE_ROOT"
    fi
fi

WORKFLOW_IDS=()
if [[ -n "$WORKFLOW_IDS_RAW" ]]; then
    IFS=',' read -r -a raw_ids <<< "$WORKFLOW_IDS_RAW"
    for id in "${raw_ids[@]}"; do
        id="${id//[[:space:]]/}"
        [[ -n "$id" ]] && WORKFLOW_IDS+=("$id")
    done
elif [[ -f "$WORKFLOW_PATH" ]]; then
    mapfile -t WORKFLOW_IDS < <(sed '/^[[:space:]]*$/d' "$WORKFLOW_PATH")
fi
if [[ ${#WORKFLOW_IDS[@]} -eq 0 ]]; then
    WORKFLOW_IDS=("")
fi
printf '%s\n' "${WORKFLOW_IDS[@]}" > "$WORKFLOW_PATH"

if [[ -f "$PLAN_PATH" ]]; then
    planned_workers="$("$PYTHON_BIN" - "$PLAN_PATH" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["workers"]))
PY
)"
    [[ "$planned_workers" -eq "${#WORKFLOW_IDS[@]}" ]] || {
        echo "Workflow count mismatch: plan=$planned_workers, supplied=${#WORKFLOW_IDS[@]}" >&2; exit 2;
    }
else
    "$PYTHON_BIN" scripts/plan_abcd_workflow_loads.py \
        --splits-dir "$SPLITS_DIR" --workers "${#WORKFLOW_IDS[@]}" \
        --subflows "${SUBFLOWS[@]}" --output "$PLAN_PATH" || exit 1
fi

resolve_offline_dir() {
    local subflow="$1" candidate
    for candidate in "$OFFLINE_ROOT/$subflow" "$OFFLINE_ROOT/graph/$subflow"; do
        if [[ -f "$candidate/subgraph.json" && -f "$candidate/skill.md" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

worker_subflows() {
    "$PYTHON_BIN" - "$PLAN_PATH" "$1" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
for item in plan["workers"][int(sys.argv[2])]["subflows"]:
    print(item["name"])
PY
}

task_complete() {
    "$PYTHON_BIN" - "$1/online_refine_result.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if isinstance(payload.get("ast_cds"), dict) else 1)
PY
}

run_worker() {
    local worker_index="$1" workflow_id="$2" subflow task_dir offline_dir
    local failed_path="$RUN_ROOT/worker_${worker_index}_failed.txt"
    : > "$failed_path"
    mapfile -t assigned < <(worker_subflows "$worker_index")
    for subflow in "${assigned[@]}"; do
        [[ -n "$subflow" ]] || continue
        task_dir="$RUN_ROOT/subflows/$subflow"
        if task_complete "$task_dir"; then
            echo "SKIP completed: worker=$worker_index subflow=$subflow"
            continue
        fi
        cmd=("$PYTHON_BIN" scripts/run_backbone_online_refine.py --subflow "$subflow" --output-dir "$task_dir")
        if [[ -f "$task_dir/skill_dag_state.json" ]]; then
            cmd+=(--resume)
        elif [[ -n "$OFFLINE_ROOT" ]]; then
            offline_dir="$(resolve_offline_dir "$subflow")" || {
                echo "Missing offline artifacts for $subflow under $OFFLINE_ROOT" >&2
                echo "$subflow" >> "$failed_path"
                [[ "$CONTINUE_ON_ERROR" -eq 0 ]] && return 1
                continue
            }
            cmd+=(--offline-dir "$offline_dir")
        fi
        cmd+=("${RUNNER_ARGS[@]}")
        echo "===== worker=$worker_index workflow=${workflow_id:-config.py} subflow=$subflow ====="
        if ! SKILLMINING_WORKFLOW_ID="$workflow_id" "${cmd[@]}"; then
            echo "$subflow" >> "$failed_path"
            [[ "$CONTINUE_ON_ERROR" -eq 0 ]] && return 1
        fi
    done
}

cat > "$MANIFEST_PATH" <<EOF
status=running
run_root=$RUN_ROOT
offline_root=${OFFLINE_ROOT:-offline_remining}
conda_env=$CONDA_ENV
hf_endpoint=$HF_ENDPOINT
workflow_ids=$(IFS=,; echo "${WORKFLOW_IDS[*]}")
load_plan=$PLAN_PATH
EOF

echo "===== Full Backbone Online Refinement ====="
echo "Run root:     $RUN_ROOT"
echo "Subflows:     ${#SUBFLOWS[@]} (current 10-flow split)"
echo "Workers:      ${#WORKFLOW_IDS[@]}"
echo "Offline root: ${OFFLINE_ROOT:-per-subflow offline re-mining}"
echo "Load plan:    $PLAN_PATH"

for index in "${!WORKFLOW_IDS[@]}"; do
    run_worker "$index" "${WORKFLOW_IDS[$index]}" > "$RUN_ROOT/logs/worker_${index}.log" 2>&1 &
    pid="$!"
    PIDS+=("$pid")
    echo "Started worker $index (PID $pid), log: $RUN_ROOT/logs/worker_${index}.log"
done

WORKER_FAILURE=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || WORKER_FAILURE=1
done

AGGREGATE="$RUN_ROOT/aggregate_online_refine.json"
AGGREGATE_FAILURE=0
if "$PYTHON_BIN" scripts/aggregate_subflow_results.py \
    --runs "$RUN_ROOT/subflows" --recursive --output "$AGGREGATE"; then
    echo "Aggregate:    $AGGREGATE"
else
    AGGREGATE_FAILURE=1
    echo "Aggregation failed; see $RUN_ROOT/subflows" >&2
fi

if [[ "$WORKER_FAILURE" -ne 0 ]] || [[ "$AGGREGATE_FAILURE" -ne 0 ]] || grep -q . "$RUN_ROOT"/worker_*_failed.txt 2>/dev/null; then
    sed -i 's/^status=.*/status=failed/' "$MANIFEST_PATH" || true
    echo "Some subflows failed. Inspect $RUN_ROOT/logs and worker_*_failed.txt." >&2
    exit 1
fi
sed -i 's/^status=.*/status=completed/' "$MANIFEST_PATH"
echo "Completed. Run root: $RUN_ROOT"
echo "Aggregate: $AGGREGATE"
