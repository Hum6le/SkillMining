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
        --output) require_value "$1" "$#"; OUTPUT_PATH="$2"; shift 2 ;;
        --conda-env) require_value "$1" "$#"; CONDA_ENV="$2"; shift 2 ;;
        --hf-endpoint) require_value "$1" "$#"; HF_ENDPOINT_VALUE="$2"; shift 2 ;;
        --python-bin) require_value "$1" "$#"; PYTHON_BIN="$2"; shift 2 ;;
        --no-conda) NO_CONDA=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

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
