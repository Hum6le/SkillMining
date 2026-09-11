#!/usr/bin/env bash
set -euo pipefail

# Parallel held-out evaluation for one Trace2Skill subflow. The underlying
# evaluator shards conversations, binds one workflow ID per process, then
# merges predictions and computes joint AST.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
CONDA_ENV="skillmining310"; HF_ENDPOINT_VALUE="https://hf-mirror.com"; PYTHON_BIN="python"
SUBFLOW=""; RUN_DIR=""; TEST_FILE=""; WORKFLOW_IDS=""; OUTPUT_DIR=""; MODEL="deepseek-chat"; NO_CONDA=0
usage() { echo "Usage: bash scripts/evaluate_trace2skill_subflow_parallel.sh --subflow NAME --run-dir DIR --test-file FILE --eval-workflow-ids ID1,ID2 [options]"; }
require_value() { [[ "$2" -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subflow) require_value "$1" "$#"; SUBFLOW="$2"; shift 2;;
    --run-dir) require_value "$1" "$#"; RUN_DIR="$2"; shift 2;;
    --test-file) require_value "$1" "$#"; TEST_FILE="$2"; shift 2;;
    --eval-workflow-ids|--workflow-ids) require_value "$1" "$#"; WORKFLOW_IDS="$2"; shift 2;;
    --output-dir) require_value "$1" "$#"; OUTPUT_DIR="$2"; shift 2;;
    --model) require_value "$1" "$#"; MODEL="$2"; shift 2;;
    --conda-env) require_value "$1" "$#"; CONDA_ENV="$2"; shift 2;;
    --hf-endpoint) require_value "$1" "$#"; HF_ENDPOINT_VALUE="$2"; shift 2;;
    --python-bin) require_value "$1" "$#"; PYTHON_BIN="$2"; shift 2;;
    --no-conda) NO_CONDA=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ -n "$SUBFLOW" && -n "$RUN_DIR" && -n "$TEST_FILE" && -n "$WORKFLOW_IDS" ]] || { echo "Required argument missing." >&2; usage >&2; exit 2; }
[[ -d "$RUN_DIR" && -f "$TEST_FILE" ]] || { echo "Missing run directory or test file." >&2; exit 2; }
if [[ "$NO_CONDA" -eq 0 ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    for conda_sh in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do [[ -f "$conda_sh" ]] && source "$conda_sh" && break; done
  fi
  command -v conda >/dev/null 2>&1 || { echo "conda not found; use --no-conda." >&2; exit 1; }
  CONDA_BASE="$(conda info --base)"; source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
export PYTHONUNBUFFERED=1 HF_ENDPOINT="$HF_ENDPOINT_VALUE" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
[[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$RUN_DIR/parallel_evaluation_$SUBFLOW"
mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/evaluation_manifest.txt"
{
  echo "status=running"; echo "subflow=$SUBFLOW"; echo "run_dir=$RUN_DIR"; echo "test_file=$TEST_FILE"
  echo "eval_workflow_ids=$WORKFLOW_IDS"; echo "model=$MODEL"; echo "conda_env=$CONDA_ENV"
} > "$MANIFEST"
"$PYTHON_BIN" scripts/evaluate_abcd_method.py \
  --method trace2skill --resource-dir "$RUN_DIR" --test-file "$TEST_FILE" \
  --subflow "$SUBFLOW" --output-dir "$OUTPUT_DIR" --model "$MODEL" \
  --eval-workflow-ids "$WORKFLOW_IDS"
sed -i 's/^status=.*/status=completed/' "$MANIFEST"
echo "Completed sharded evaluation. Result: $OUTPUT_DIR/result.json"
