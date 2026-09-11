#!/usr/bin/env bash

# Analyze Trace2Skill modules from existing artifacts. This wrapper never
# launches a rollout; it only reads summary/turn/rollout JSON files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="skillmining310"
HF_ENDPOINT_VALUE="https://hf-mirror.com"
PYTHON_BIN="python"
RUN_ROOT=""
ROLLOUT_ROOT=""
SUBFLOW=""
OUTPUT_PATH=""
WORKFLOW_IDS_RAW=""
SUBFLOWS_RAW=""
PARALLEL_ROOT=""
NO_CONDA=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/analyze_trace2skill_modules.sh --subflow NAME --run-root DIR [options]
  bash scripts/analyze_trace2skill_modules.sh --rollouts DIR [options]
  bash scripts/analyze_trace2skill_modules.sh --run-root DIR [options]

Inputs:
  --run-root DIR          Trace2Skill/online-refine run root or one run directory.
  --rollouts DIR          Existing rollout JSON/file or rollout directory.
  --subflow NAME          Restrict recursive lookup to one subflow.
  --subflows IDS          Comma-separated subflows for parallel workers.
  --workflow-ids IDS      Comma-separated workflow IDs; one worker per ID.

Options:
  --output PATH           JSON report path (Markdown is written beside it).
  --conda-env NAME        Default: skillmining310.
  --hf-endpoint URL       Default: https://hf-mirror.com.
  --python-bin PATH       Default: python.
  --no-conda              Do not activate conda; use --python-bin as-is.
  -h, --help              Show this help.

Examples:
  bash scripts/analyze_trace2skill_modules.sh --run-root outputs/full_run --subflow recover_password
  bash scripts/analyze_trace2skill_modules.sh --rollouts outputs/full_run/trace2skill/recover_password
EOF
}

require_value() {
    [[ "$2" -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-root) require_value "$1" "$#"; RUN_ROOT="$2"; shift 2 ;;
        --rollouts) require_value "$1" "$#"; ROLLOUT_ROOT="$2"; shift 2 ;;
        --subflow) require_value "$1" "$#"; SUBFLOW="$2"; shift 2 ;;
        --subflows) require_value "$1" "$#"; SUBFLOWS_RAW="$2"; shift 2 ;;
        --workflow-ids) require_value "$1" "$#"; WORKFLOW_IDS_RAW="$2"; shift 2 ;;
        --output) require_value "$1" "$#"; OUTPUT_PATH="$2"; shift 2 ;;
        --conda-env) require_value "$1" "$#"; CONDA_ENV="$2"; shift 2 ;;
        --hf-endpoint) require_value "$1" "$#"; HF_ENDPOINT_VALUE="$2"; shift 2 ;;
        --python-bin) require_value "$1" "$#"; PYTHON_BIN="$2"; shift 2 ;;
        --no-conda) NO_CONDA=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "$WORKFLOW_IDS_RAW" ]]; then
    [[ -n "$RUN_ROOT" ]] || { echo "--workflow-ids requires --run-root." >&2; exit 2; }
    [[ -z "$SUBFLOW" ]] || { echo "Use --subflows (plural) with --workflow-ids, not --subflow." >&2; exit 2; }
    [[ -n "$SUBFLOWS_RAW" ]] || { echo "--workflow-ids requires --subflows." >&2; exit 2; }
    IFS=',' read -r -a WORKFLOW_IDS <<< "$WORKFLOW_IDS_RAW"
    IFS=',' read -r -a SUBFLOWS <<< "$SUBFLOWS_RAW"
    [[ ${#WORKFLOW_IDS[@]} -gt 0 && ${#SUBFLOWS[@]} -gt 0 ]] || { echo "Empty workflow/subflow list." >&2; exit 2; }
fi

[[ -n "$RUN_ROOT" || -n "$ROLLOUT_ROOT" ]] || {
    echo "Provide --run-root or --rollouts." >&2; usage >&2; exit 2;
}
[[ -z "$RUN_ROOT" || -z "$ROLLOUT_ROOT" ]] || {
    echo "--run-root and --rollouts are mutually exclusive." >&2; exit 2;
}

if [[ "$NO_CONDA" -eq 0 ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        for conda_sh in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
            [[ -f "$conda_sh" ]] && source "$conda_sh" && break
        done
    fi
    command -v conda >/dev/null 2>&1 || { echo "conda was not found; use --no-conda." >&2; exit 1; }
    CONDA_BASE="$(conda info --base 2>/dev/null)" || { echo "Unable to determine conda base." >&2; exit 1; }
    [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || { echo "Unable to activate $CONDA_ENV" >&2; exit 1; }
fi

ENV_LABEL="$CONDA_ENV"
[[ "$NO_CONDA" -eq 1 ]] && ENV_LABEL="disabled"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="$HF_ENDPOINT_VALUE"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Multi-worker mode mirrors run_full_*: subflows are assigned round-robin and
# each worker runs serially with its own SKILLMINING_WORKFLOW_ID.
if [[ -n "$WORKFLOW_IDS_RAW" ]]; then
    PARALLEL_ROOT="${OUTPUT_PATH:-$ROOT_DIR/outputs/trace2skill_module_parallel_$(date +%Y-%m-%d_%H-%M-%S)}"
    [[ "$PARALLEL_ROOT" == *.json ]] && PARALLEL_ROOT="${PARALLEL_ROOT%.json}"
    mkdir -p "$PARALLEL_ROOT/logs" "$PARALLEL_ROOT/reports"
    MANIFEST="$PARALLEL_ROOT/manifest.txt"
    {
        echo "status=running"
        echo "run_root=$RUN_ROOT"
        echo "workflow_ids=$WORKFLOW_IDS_RAW"
        echo "subflows=$SUBFLOWS_RAW"
        echo "conda_env=$ENV_LABEL"
        echo "hf_endpoint=$HF_ENDPOINT"
    } > "$MANIFEST"
    PIDS=()
    for index in "${!WORKFLOW_IDS[@]}"; do
        workflow_id="${WORKFLOW_IDS[$index]}"
        (
            worker_failed=0
            for subflow_index in "${!SUBFLOWS[@]}"; do
                [[ $((subflow_index % ${#WORKFLOW_IDS[@]})) -eq "$index" ]] || continue
                subflow="${SUBFLOWS[$subflow_index]}"
                [[ -n "$subflow" ]] || continue
                report_path="$PARALLEL_ROOT/reports/${subflow}.json"
                echo "worker=$index workflow=$workflow_id subflow=$subflow"
                if ! SKILLMINING_WORKFLOW_ID="$workflow_id" "$PYTHON_BIN" scripts/analyze_trace2skill_modules.py \
                    "$RUN_ROOT" --subflow "$subflow" --output "$report_path"; then
                    echo "$subflow" >> "$PARALLEL_ROOT/worker_${index}_failed.txt"
                    worker_failed=1
                fi
            done
            exit "$worker_failed"
        ) > "$PARALLEL_ROOT/logs/worker_${index}.log" 2>&1 &
        PIDS+=("$!")
        echo "Started worker $index (PID ${PIDS[-1]}), log: $PARALLEL_ROOT/logs/worker_${index}.log"
    done
    worker_failure=0
    for pid in "${PIDS[@]}"; do wait "$pid" || worker_failure=1; done
    "$PYTHON_BIN" - "$PARALLEL_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
reports = []
for path in sorted((root / "reports").glob("*.json")):
    try: reports.append(json.loads(path.read_text(encoding="utf-8")))
    except Exception: pass
(root / "aggregate.json").write_text(json.dumps({"reports": reports, "num_reports": len(reports)}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
    if [[ "$worker_failure" -eq 0 ]]; then sed -i 's/^status=.*/status=completed/' "$MANIFEST"; else sed -i 's/^status=.*/status=failed/' "$MANIFEST"; fi
    echo "Completed parallel analysis. Root: $PARALLEL_ROOT"
    echo "Aggregate: $PARALLEL_ROOT/aggregate.json"
    [[ "$worker_failure" -eq 0 ]] || exit 1
    exit 0
fi

COMMAND=("$PYTHON_BIN" scripts/analyze_trace2skill_modules.py)
if [[ -n "$RUN_ROOT" ]]; then COMMAND+=("$RUN_ROOT"); else COMMAND+=(--rollouts "$ROLLOUT_ROOT"); fi
[[ -n "$SUBFLOW" ]] && COMMAND+=(--subflow "$SUBFLOW")

if [[ -z "$OUTPUT_PATH" ]]; then
    safe_subflow="${SUBFLOW:-all}"
    OUTPUT_PATH="$ROOT_DIR/outputs/trace2skill_module_report_${safe_subflow}.json"
fi
COMMAND+=(--output "$OUTPUT_PATH")
mkdir -p "$(dirname "$OUTPUT_PATH")"
MANIFEST="${OUTPUT_PATH%.json}.manifest.txt"
{
    echo "status=running"
    echo "run_root=${RUN_ROOT:-n/a}"
    echo "rollouts=${ROLLOUT_ROOT:-n/a}"
    echo "subflow=${SUBFLOW:-all}"
    echo "conda_env=$ENV_LABEL"
    echo "hf_endpoint=$HF_ENDPOINT"
    echo "python_bin=$PYTHON_BIN"
    printf 'command='; printf '%q ' "${COMMAND[@]}"; echo
} > "$MANIFEST"

echo "===== Trace2Skill Module Analysis ====="
echo "Run root:     ${RUN_ROOT:-n/a}"
echo "Rollouts:     ${ROLLOUT_ROOT:-n/a}"
echo "Subflow:      ${SUBFLOW:-all}"
echo "Conda env:    $ENV_LABEL"
echo "HF_ENDPOINT:  $HF_ENDPOINT"
echo "Output JSON:  $OUTPUT_PATH"
printf 'Command:      '; printf '%q ' "${COMMAND[@]}"; echo

if "${COMMAND[@]}"; then
    sed -i 's/^status=.*/status=completed/' "$MANIFEST"
    echo "Completed. Markdown report: ${OUTPUT_PATH%.json}.md"
else
    sed -i 's/^status=.*/status=failed/' "$MANIFEST" || true
    exit 1
fi
