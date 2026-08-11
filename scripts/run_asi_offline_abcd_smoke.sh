#!/usr/bin/env bash

# Complete frozen-library ASIoffline reproduction for one ABCD subflow.
# Despite the historical "smoke" name, the default uses every train session
# for induction and every paired frozen test session for evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="skillmining310"
SUBFLOW=""
TEMPERATURE="1.0"
PYTHON_BIN="python"
MAX_TRAIN=""
MAX_TEST=""
RESUME=0
DRY_RUN=0
SKIP_EVAL=0
ALLOW_EMPTY_LIBRARY=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_asi_offline_abcd_smoke.sh --subflow NAME [options]

Run ASIoffline on one isolated ABCD subflow. The complete protocol is:
  fixed train traces -> llm.chat() induction -> static validation
  -> frozen library -> paired frozen test evaluation

Required:
  --subflow NAME             ABCD subflow directory under data/eval/abcd/splits

Options:
  --temperature FLOAT        ASI induction temperature (default: 1.0)
  --conda-env NAME           Conda environment to activate (default: skillmining310)
  --python-bin PATH          Python executable after activation (default: python)
  --max-train N              Limit train trajectories for a bounded smoke run
  --max-test N               Limit frozen test conversations for a bounded smoke run
  --resume-induction         Reuse completed raw induction artifacts
  --dry-run-induction        Build prompts only; makes no LLM call
  --skip-eval                Freeze the validated library without test evaluation
  --allow-empty-library      Evaluate an empty library (debug only)
  -h, --help                 Show this help

All model calls are performed by the repository's llm.chat() interface.
EOF
}

require_value() {
    local option="$1"
    local remaining="$2"
    [[ "$remaining" -ge 2 ]] || { echo "Missing value for $option" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subflow) require_value "$1" "$#"; SUBFLOW="$2"; shift 2 ;;
        --temperature) require_value "$1" "$#"; TEMPERATURE="$2"; shift 2 ;;
        --conda-env) require_value "$1" "$#"; CONDA_ENV="$2"; shift 2 ;;
        --python-bin) require_value "$1" "$#"; PYTHON_BIN="$2"; shift 2 ;;
        --max-train) require_value "$1" "$#"; MAX_TRAIN="$2"; shift 2 ;;
        --max-test) require_value "$1" "$#"; MAX_TEST="$2"; shift 2 ;;
        --resume-induction) RESUME=1; shift ;;
        --dry-run-induction) DRY_RUN=1; shift ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        --allow-empty-library) ALLOW_EMPTY_LIBRARY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$SUBFLOW" ]] || { echo "--subflow is required." >&2; usage >&2; exit 2; }
for value_name in MAX_TRAIN MAX_TEST; do
    value="${!value_name}"
    if [[ -n "$value" && ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer; got: $value" >&2
        exit 2
    fi
done

if ! command -v conda >/dev/null 2>&1; then
    for conda_sh in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
        [[ -f "$conda_sh" ]] && source "$conda_sh" && break
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Initialize conda before running this script." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base 2>/dev/null)" || { echo "Unable to determine conda base." >&2; exit 1; }
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "Unable to activate conda environment: $CONDA_ENV" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }

TRAIN_FILE="$ROOT_DIR/data/eval/abcd/splits/$SUBFLOW/train.json"
TEST_FILE="$ROOT_DIR/data/eval/abcd/splits/$SUBFLOW/test.json"
[[ -f "$TRAIN_FILE" ]] || { echo "Training split not found: $TRAIN_FILE" >&2; exit 1; }
[[ -f "$TEST_FILE" ]] || { echo "Frozen test split not found: $TEST_FILE" >&2; exit 1; }

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="$ROOT_DIR/outputs/asi_offline_abcd_subflow_${SUBFLOW}_${RUN_ID}"
MANIFEST="$RUN_DIR/manifest.txt"
STARTED_AT="$(date -Iseconds)"
mkdir -p "$RUN_DIR"

write_manifest() {
    local status="$1"
    {
        echo "run_id=$RUN_ID"
        echo "status=$status"
        echo "started_at=$STARTED_AT"
        echo "finished_at=$(date -Iseconds)"
        echo "subflow=$SUBFLOW"
        echo "train_file=$TRAIN_FILE"
        echo "test_file=$TEST_FILE"
        echo "temperature=$TEMPERATURE"
        echo "conda_env=$CONDA_ENV"
        echo "python_bin=$PYTHON_BIN"
        echo "max_train=${MAX_TRAIN:-all}"
        echo "max_test=${MAX_TEST:-all}"
        echo "resume_induction=$RESUME"
        echo "dry_run_induction=$DRY_RUN"
        echo "skip_eval=$SKIP_EVAL"
        echo "run_dir=$RUN_DIR"
    } > "$MANIFEST"
}

write_manifest "running"
trap 'write_manifest "failed"' ERR

command=("$PYTHON_BIN" scripts/run_asi_offline_abcd.py --subflow "$SUBFLOW" --train-file "$TRAIN_FILE" --test-file "$TEST_FILE" --output-dir "$RUN_DIR" --temperature "$TEMPERATURE")
[[ -n "$MAX_TRAIN" ]] && command+=(--max-train "$MAX_TRAIN")
[[ -n "$MAX_TEST" ]] && command+=(--max-test "$MAX_TEST")
[[ "$RESUME" -eq 1 ]] && command+=(--resume-induction)
[[ "$DRY_RUN" -eq 1 ]] && command+=(--dry-run-induction)
[[ "$SKIP_EVAL" -eq 1 ]] && command+=(--skip-eval)
[[ "$ALLOW_EMPTY_LIBRARY" -eq 1 ]] && command+=(--allow-empty-library)

echo "===== ASIoffline ABCD single-subflow run ====="
echo "Subflow:       $SUBFLOW"
echo "Induction:     $TRAIN_FILE (${MAX_TRAIN:-all} sessions)"
echo "Frozen test:   $TEST_FILE (${MAX_TEST:-all} sessions)"
echo "LLM interface:  llm.chat()"
echo "Run directory: $RUN_DIR"
echo "Environment:   $CONDA_ENV"
echo "Python:        $($PYTHON_BIN --version 2>&1)"

"${command[@]}"

write_manifest "completed"
trap - ERR
echo "===== ASIoffline single-subflow run completed ====="
echo "Run directory: $RUN_DIR"
echo "Manifest:      $MANIFEST"
if [[ -f "$RUN_DIR/evaluation/result.json" ]]; then
    echo "Result summary:"
    "$PYTHON_BIN" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("summary", ""))' "$RUN_DIR/evaluation/result.json"
fi
