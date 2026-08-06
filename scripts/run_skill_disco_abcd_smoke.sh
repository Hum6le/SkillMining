#!/usr/bin/env bash

# Run one complete-subflow SKILL-DISCO-Offline (pseudocode) experiment.
#
# The runner induces a pseudocode skill library from every training conversation
# in the requested subflow, then evaluates it on that subflow's entire frozen
# test split.
#
# Example:
#   bash scripts/run_skill_disco_abcd_smoke.sh --subflow manage_cancel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="skillmining310"
HF_MIRROR="https://hf-mirror.com"
SUBFLOW=""
MODEL="deepseek-chat"
GROUPING_BATCH_SIZE=20
MIN_SUPPORT=2
PYTHON_BIN="python"
SKIP_EVAL=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_skill_disco_abcd_smoke.sh --subflow NAME [options]

Generate SKILL-DISCO-Offline pseudocode skills from every induction conversation
and evaluate them on the complete frozen test split of one ABCD subflow.

Required:
  --subflow NAME             ABCD subflow directory under data/eval/abcd/splits

Options:
  --model NAME               Unified llm.chat model name (default: deepseek-chat)
  --batch-size N             Stage-3 grouping batch size (default: 20)
  --min-support N            Minimum distinct induction conversations per skill (default: 2)
  --conda-env NAME           Conda environment to activate (default: skillmining310)
  --python-bin PATH          Python executable after activation (default: python)
  --skip-eval                Generate artifacts only; do not call the evaluator
  -h, --help                 Show this help

The runner sets HF_ENDPOINT=https://hf-mirror.com and writes all artifacts to
outputs/skill_disco_abcd_subflow_<subflow>_<timestamp>/.
EOF
}

require_value() {
    local option="$1"
    local remaining_args="$2"
    [[ "$remaining_args" -ge 2 ]] || { echo "Missing value for $option" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subflow)
            require_value "$1" "$#"
            SUBFLOW="$2"
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

if [[ -z "$SUBFLOW" ]]; then
    echo "--subflow is required." >&2
    usage >&2
    exit 2
fi

for value_name in GROUPING_BATCH_SIZE MIN_SUPPORT; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer; got: $value" >&2
        exit 2
    fi
done

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
    echo "Python was not found after activating $CONDA_ENV: $PYTHON_BIN" >&2
    exit 1
fi

TRAIN_FILE="$ROOT_DIR/data/eval/abcd/splits/$SUBFLOW/train.json"
TEST_FILE="$ROOT_DIR/data/eval/abcd/splits/$SUBFLOW/test.json"
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Training split not found: $TRAIN_FILE" >&2
    exit 1
fi
if [[ ! -f "$TEST_FILE" ]]; then
    echo "Frozen test split not found: $TEST_FILE" >&2
    exit 1
fi

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="$ROOT_DIR/outputs/skill_disco_abcd_subflow_${SUBFLOW}_${RUN_ID}"
ARTIFACT_PATH="$RUN_DIR/generation_artifact.json"
LIBRARY_PATH="$RUN_DIR/SKILL.md"
EVALUATION_DIR="$RUN_DIR/evaluation"
RESULT_PATH="$EVALUATION_DIR/result.json"
MANIFEST="$RUN_DIR/manifest.txt"
STARTED_AT="$(date -Iseconds)"
mkdir -p "$RUN_DIR"

write_manifest() {
    local finished_at="$1"
    local status="$2"
    {
        echo "run_id=$RUN_ID"
        echo "status=$status"
        echo "started_at=$STARTED_AT"
        echo "finished_at=$finished_at"
        echo "subflow=$SUBFLOW"
        echo "train_file=$TRAIN_FILE"
        echo "test_file=$TEST_FILE"
        echo "train_sessions=all"
        echo "test_sessions=all"
        echo "model=$MODEL"
        echo "grouping_batch_size=$GROUPING_BATCH_SIZE"
        echo "min_support=$MIN_SUPPORT"
        echo "conda_env=$CONDA_ENV"
        echo "python_bin=$PYTHON_BIN"
        echo "hf_endpoint=$HF_ENDPOINT"
        echo "generation_artifact=$ARTIFACT_PATH"
        echo "skill_library=$LIBRARY_PATH"
        echo "evaluation_dir=$EVALUATION_DIR"
        echo "evaluation_result=$RESULT_PATH"
        echo "skip_eval=$SKIP_EVAL"
    } > "$MANIFEST"
}

write_manifest "" "running"
trap 'write_manifest "$(date -Iseconds)" "failed"' ERR

echo "===== SKILL-DISCO ABCD complete single-subflow run ====="
echo "Subflow:       $SUBFLOW"
echo "Induction:     $TRAIN_FILE (all conversations)"
echo "Frozen test:   $TEST_FILE (all conversations)"
echo "Model:         $MODEL"
echo "Run directory: $RUN_DIR"
echo "Environment:   $CONDA_ENV"
echo "Python:        $($PYTHON_BIN --version 2>&1)"

echo
echo "[1/2] Generating offline pseudocode skill library..."
"$PYTHON_BIN" scripts/run_skill_disco_abcd.py \
    --input "$TRAIN_FILE" \
    --output "$ARTIFACT_PATH" \
    --library-output "$LIBRARY_PATH" \
    --model "$MODEL" \
    --batch-size "$GROUPING_BATCH_SIZE" \
    --min-support "$MIN_SUPPORT" \
    --expected-subflow "$SUBFLOW"

if [[ "$SKIP_EVAL" -eq 1 ]]; then
    FINISHED_AT="$(date -Iseconds)"
    write_manifest "$FINISHED_AT" "generated_only"
    trap - ERR
    echo
    echo "Generation completed; evaluation was skipped."
    echo "Artifact: $ARTIFACT_PATH"
    echo "Library:  $LIBRARY_PATH"
    echo "Manifest: $MANIFEST"
    exit 0
fi

echo
echo "[2/2] Evaluating generated library on frozen test conversations..."
"$PYTHON_BIN" scripts/eval_skill_disco_abcd.py \
    --skill-library "$LIBRARY_PATH" \
    --test-file "$TEST_FILE" \
    --output-dir "$EVALUATION_DIR" \
    --model "$MODEL" \
    --expected-subflow "$SUBFLOW"

FINISHED_AT="$(date -Iseconds)"
write_manifest "$FINISHED_AT" "completed"
trap - ERR

echo
echo "===== SKILL-DISCO smoke run completed ====="
echo "Artifact:          $ARTIFACT_PATH"
echo "Skill library:     $LIBRARY_PATH"
echo "Evaluation result: $RESULT_PATH"
echo "Manifest:          $MANIFEST"
if [[ -f "$RESULT_PATH" ]]; then
    echo "Result summary:"
    "$PYTHON_BIN" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("summary", {}))' "$RESULT_PATH"
fi
