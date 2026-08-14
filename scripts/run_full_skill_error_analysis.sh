#!/usr/bin/env bash

# Run full-corpus skill-content error analysis from the experiment manifest.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="python"
CONDA_ENV="skillmining310"
HF_MIRROR="https://hf-mirror.com"
MANIFEST=""
SKILLS_ROOT=""
PREDICTIONS_ROOT=""
TEST_ROOT="$ROOT_DIR/data/eval/abcd/splits"
OUTPUT_DIR=""
BATCH_SIZE=8
SAMPLE_ERRORS=10
SEED=17
EXPECTED_SUBFLOWS=96
METHODS=()
SUBFLOWS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/run_full_skill_error_analysis.sh (--manifest PATH | --skills-root PATH) [options]

Required:
  --manifest PATH           Text manifest emitted by run_full_abcd_experiments.sh
  --skills-root PATH        Root containing <subflow>/skill.md; use this when no manifest exists

Optional:
  --predictions-root PATH   Separate prediction root; normally unnecessary
  --test-root PATH          ABCD test split root
  --output-dir PATH         Analysis output directory
  --batch-size N            Subflows per batch summary (default: 8)
  --sample-errors N         Error samples per subflow (default: 10)
  --seed N                  Sampling seed (default: 17)
  --expected-subflows N     Sanity check (default: 96)
  --methods "LIST"          Method filter, e.g. "awm trace2skill"
  --subflows "LIST"         Subflow filter, e.g. "cost status"
  --python PATH             Python executable (default: python)
  --conda-env NAME         Conda environment (default: skillmining310)
  --hf-mirror URL          Hugging Face mirror (default: https://hf-mirror.com)
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest|--skills-root|--predictions-root|--test-root|--output-dir|--batch-size|--sample-errors|--seed|--expected-subflows|--methods|--subflows|--python|--conda-env|--hf-mirror)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --manifest) MANIFEST="$2" ;;
                --skills-root) SKILLS_ROOT="$2" ;;
                --predictions-root) PREDICTIONS_ROOT="$2" ;;
                --test-root) TEST_ROOT="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
                --batch-size) BATCH_SIZE="$2" ;;
                --sample-errors) SAMPLE_ERRORS="$2" ;;
                --seed) SEED="$2" ;;
                --expected-subflows) EXPECTED_SUBFLOWS="$2" ;;
                --methods) read -r -a METHODS <<< "$2" ;;
                --subflows) read -r -a SUBFLOWS <<< "$2" ;;
                --python) PYTHON_BIN="$2" ;;
                --conda-env) CONDA_ENV="$2" ;;
                --hf-mirror) HF_MIRROR="$2" ;;
            esac
            shift 2
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

if [[ -z "$MANIFEST" && -z "$SKILLS_ROOT" ]]; then
    echo "one of --manifest or --skills-root is required" >&2
    usage >&2
    exit 2
fi
if [[ -n "$MANIFEST" && ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi
if [[ ! -d "$TEST_ROOT" ]]; then
    echo "Test root not found: $TEST_ROOT" >&2
    exit 1
fi

# Match the full ABCD experiment runner: activate the shared environment and
# use the same Hugging Face mirror before importing/running project code.
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
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
    OUTPUT_DIR="$ROOT_DIR/outputs/full_skill_error_analysis_$STAMP"
fi
mkdir -p "$OUTPUT_DIR"

echo "Skill error analysis"
echo "Manifest:          ${MANIFEST:-<none>}"
echo "Skills root:       ${SKILLS_ROOT:-<none>}"
echo "Test root:         $TEST_ROOT"
echo "Output directory:  $OUTPUT_DIR"
echo "Environment:       $CONDA_ENV"
echo "Python:            $($PYTHON_BIN --version 2>&1)"
echo "HF_ENDPOINT:       $HF_ENDPOINT"
echo "Batch size:        $BATCH_SIZE"
echo "Sample errors:     $SAMPLE_ERRORS"
echo "Expected subflows: $EXPECTED_SUBFLOWS"

CMD=("$PYTHON_BIN" scripts/error_analysis_full_skills.py
    --test-root "$TEST_ROOT"
    --output-dir "$OUTPUT_DIR"
    --batch-size "$BATCH_SIZE"
    --sample-errors "$SAMPLE_ERRORS"
    --seed "$SEED"
    --expected-subflows "$EXPECTED_SUBFLOWS")

if [[ -n "$MANIFEST" ]]; then
    CMD+=(--manifest "$MANIFEST")
else
    CMD+=(--skills-root "$SKILLS_ROOT")
fi

if [[ -n "$PREDICTIONS_ROOT" ]]; then
    CMD+=(--predictions-root "$PREDICTIONS_ROOT")
fi
if [[ ${#METHODS[@]} -gt 0 ]]; then
    CMD+=(--methods "${METHODS[@]}")
fi
if [[ ${#SUBFLOWS[@]} -gt 0 ]]; then
    for subflow in "${SUBFLOWS[@]}"; do
        CMD+=(--subflow "$subflow")
    done
fi

"${CMD[@]}"
status=$?
if [[ $status -ne 0 ]]; then
    echo "Skill error analysis failed (exit=$status)" >&2
    exit "$status"
fi

echo
echo "===== Skill error analysis finished ====="
echo "  Overview:       $OUTPUT_DIR/overview.json"
echo "  Subflow audits: $OUTPUT_DIR/subflow_analyses.json"
echo "  Batch reports:  $OUTPUT_DIR/batch_summaries.json"
echo "  Final report:   $OUTPUT_DIR/final_report.md"
