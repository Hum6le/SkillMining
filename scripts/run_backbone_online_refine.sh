#!/usr/bin/env bash

# Online DAG refinement for a pre-mined backbone skill. This wrapper keeps the
# existing offline artifact immutable and creates a separate online run root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1

CONDA_ENV="skillmining310"
HF_ENDPOINT_VALUE="https://hf-mirror.com"
PYTHON_BIN="python"
SUBFLOW=""
OFFLINE_DIR=""
OUTPUT_DIR=""
RESUME_RUN=""
WORKFLOW_ID=""
EVAL_WORKFLOW_IDS=""
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_backbone_online_refine.sh --subflow NAME [--offline-dir DIR] [options]
  bash scripts/run_backbone_online_refine.sh --subflow NAME --resume-run DIR [options]

Required:
  --subflow NAME             ABCD coarse-flow split directory
  --offline-dir DIR          Existing offline artifact dir with subgraph.json and skill.md.
                              Omit to mine the current discriminative-MST backbone first.
  --resume-run DIR           Existing online run directory. Mutually exclusive with --offline-dir.

Options:
  --output-dir DIR           New online run directory. Default: outputs/online_refine_<subflow>_<timestamp>
  --workflow-id ID           Export SKILLMINING_WORKFLOW_ID for workflow API routing
  --eval-workflow-ids IDS    Comma-separated workflow IDs for parallel online rollouts
  --conda-env NAME           Default: skillmining310
  --hf-endpoint URL          Default: https://hf-mirror.com
  --python-bin PATH          Default: python
  --batch-size N             Online rollout batch size (default: 8)
  --per-transition-cap N     Max representative sessions per transition (default: 3)
  --target-selection-rate R  Target fraction of train sessions for online rollout (default: 0.30)
  --max-batches N            Limit online batches for a smoke experiment
  --max-train N              Limit train sessions
  --max-test N               Limit held-out test sessions
  --skip-guard-llm           Collect feedback/patches but do not induce guards
  --guard-retries N          LLM retries per local guard (default: 3)
  -h, --help                 Show this help

All unknown runner options are forwarded to run_backbone_online_refine.py.
EOF
}

require_value() {
    [[ "$2" -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subflow) require_value "$1" "$#"; SUBFLOW="$2"; shift 2 ;;
        --offline-dir) require_value "$1" "$#"; OFFLINE_DIR="$2"; shift 2 ;;
        --output-dir) require_value "$1" "$#"; OUTPUT_DIR="$2"; shift 2 ;;
        --resume-run) require_value "$1" "$#"; RESUME_RUN="$2"; shift 2 ;;
        --workflow-id) require_value "$1" "$#"; WORKFLOW_ID="$2"; shift 2 ;;
        --eval-workflow-ids) require_value "$1" "$#"; EVAL_WORKFLOW_IDS="$2"; shift 2 ;;
        --conda-env) require_value "$1" "$#"; CONDA_ENV="$2"; shift 2 ;;
        --hf-endpoint) require_value "$1" "$#"; HF_ENDPOINT_VALUE="$2"; shift 2 ;;
        --python-bin) require_value "$1" "$#"; PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

[[ -n "$SUBFLOW" ]] || { echo "--subflow is required." >&2; usage >&2; exit 2; }
if [[ -n "$OFFLINE_DIR" && -n "$RESUME_RUN" ]]; then
    echo "--offline-dir and --resume-run cannot be used together." >&2
    exit 2
fi
if [[ -n "$RESUME_RUN" ]]; then
    RUN_DIR="$(cd "$RESUME_RUN" 2>/dev/null && pwd)" || {
        echo "--resume-run does not exist: $RESUME_RUN" >&2; exit 2;
    }
    [[ -z "$OUTPUT_DIR" || "$OUTPUT_DIR" == "$RUN_DIR" ]] || {
        echo "--output-dir must be omitted or match --resume-run." >&2; exit 2;
    }
else
    if [[ -n "$OFFLINE_DIR" ]]; then
        [[ -d "$OFFLINE_DIR" ]] || { echo "--offline-dir does not exist: $OFFLINE_DIR" >&2; exit 2; }
        [[ -f "$OFFLINE_DIR/subgraph.json" && -f "$OFFLINE_DIR/skill.md" ]] || {
            echo "--offline-dir must contain subgraph.json and skill.md: $OFFLINE_DIR" >&2; exit 2;
        }
    fi
    RUN_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/online_refine_${SUBFLOW}_$(date +%Y-%m-%d_%H-%M-%S)}"
fi
mkdir -p "$RUN_DIR"

if ! command -v conda >/dev/null 2>&1; then
    for conda_sh in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
        [[ -f "$conda_sh" ]] && source "$conda_sh" && break
    done
fi
command -v conda >/dev/null 2>&1 || { echo "conda was not found." >&2; exit 1; }
CONDA_BASE="$(conda info --base 2>/dev/null)" || { echo "Unable to determine conda base." >&2; exit 1; }
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "Unable to activate $CONDA_ENV" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
export HF_ENDPOINT="$HF_ENDPOINT_VALUE"
[[ -n "$WORKFLOW_ID" ]] && export SKILLMINING_WORKFLOW_ID="$WORKFLOW_ID"

COMMAND=("$PYTHON_BIN" scripts/run_backbone_online_refine.py --subflow "$SUBFLOW" --output-dir "$RUN_DIR")
if [[ -n "$RESUME_RUN" ]]; then
    COMMAND+=(--resume)
elif [[ -n "$OFFLINE_DIR" ]]; then
    COMMAND+=(--offline-dir "$OFFLINE_DIR")
fi
[[ -n "$EVAL_WORKFLOW_IDS" ]] && COMMAND+=(--eval-workflow-ids "$EVAL_WORKFLOW_IDS")
COMMAND+=("${EXTRA_ARGS[@]}")

MANIFEST="$RUN_DIR/online_refine_manifest.txt"
{
    echo "status=running"
    echo "subflow=$SUBFLOW"
    echo "offline_dir=${OFFLINE_DIR:-n/a}"
    echo "resume_run=${RESUME_RUN:-n/a}"
    echo "run_dir=$RUN_DIR"
    echo "conda_env=$CONDA_ENV"
    echo "hf_endpoint=$HF_ENDPOINT"
    echo "workflow_id=${WORKFLOW_ID:-config.py}"
    echo "eval_workflow_ids=${EVAL_WORKFLOW_IDS:-none}"
    printf 'command='; printf '%q ' "${COMMAND[@]}"; echo
} > "$MANIFEST"

echo "===== Backbone Online Refinement ====="
echo "Subflow:       $SUBFLOW"
echo "Offline input: ${OFFLINE_DIR:-state in $RUN_DIR}"
echo "Run directory: $RUN_DIR"
echo "Workflow ID:   ${WORKFLOW_ID:-config.py}"
echo "Eval workers:  ${EVAL_WORKFLOW_IDS:-1 (workflow_id/config.py)}"
echo "HF_ENDPOINT:   $HF_ENDPOINT"
printf 'Command:       '; printf '%q ' "${COMMAND[@]}"; echo

if "${COMMAND[@]}"; then
    sed -i 's/^status=.*/status=completed/' "$MANIFEST"
    echo "Completed. Results:  $RUN_DIR/online_refine_result.json"
    echo "Skill DAG:           $RUN_DIR/skill_dag_state.json"
    echo "Refined skill:       $RUN_DIR/skill.md"
else
    sed -i 's/^status=.*/status=failed/' "$MANIFEST" || true
    exit 1
fi
