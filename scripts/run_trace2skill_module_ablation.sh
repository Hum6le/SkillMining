#!/usr/bin/env bash
# Matched module ablation for Trace2Skill on one ABCD subflow.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SUBFLOW=""
TRAIN_FILE=""
TEST_FILE=""
OUTPUT_ROOT=""
WORKFLOW_IDS_RAW="${SKILLMINING_WORKFLOW_IDS:-${SKILLMINING_WORKFLOW_ID:-}}"
REUSE_ROLLOUT_DIR="${TRACE2SKILL_REUSE_ROLLOUT_DIR:-}"
REUSE_ANALYSIS_DIR="${TRACE2SKILL_REUSE_ANALYSIS_DIR:-}"
EXISTING_FULL_DIR="${TRACE2SKILL_EXISTING_FULL_DIR:-}"
VARIANTS_RAW="${TRACE2SKILL_ABLATION_VARIANTS:-full,no_failure_analysis,no_success_memory,no_structured_evolution,one_shot_update,no_adaptation}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_trace2skill_module_ablation.sh [options]

Required:
  --subflow NAME
  --train-file PATH
  --test-file PATH
  --output-dir PATH

Optional:
  --workflow-ids ID1,ID2,...
  --reuse-rollout-dir PATH
  --reuse-analysis-dir PATH
  --existing-full-dir PATH
  --force-full
  --variants NAME1,NAME2,...

Variants:
  full, no_failure_analysis, no_success_memory,
  no_structured_evolution, one_shot_update, no_adaptation
EOF
}

FORCE_FULL="${FORCE_FULL:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subflow) SUBFLOW="${2:?missing value for --subflow}"; shift 2 ;;
    --train-file) TRAIN_FILE="${2:?missing value for --train-file}"; shift 2 ;;
    --test-file) TEST_FILE="${2:?missing value for --test-file}"; shift 2 ;;
    --output-dir) OUTPUT_ROOT="${2:?missing value for --output-dir}"; shift 2 ;;
    --workflow-ids) WORKFLOW_IDS_RAW="${2:?missing value for --workflow-ids}"; shift 2 ;;
    --reuse-rollout-dir) REUSE_ROLLOUT_DIR="${2:?missing value for --reuse-rollout-dir}"; shift 2 ;;
    --reuse-analysis-dir) REUSE_ANALYSIS_DIR="${2:?missing value for --reuse-analysis-dir}"; shift 2 ;;
    --existing-full-dir) EXISTING_FULL_DIR="${2:?missing value for --existing-full-dir}"; shift 2 ;;
    --variants) VARIANTS_RAW="${2:?missing value for --variants}"; shift 2 ;;
    --force-full) FORCE_FULL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$SUBFLOW" && -n "$TRAIN_FILE" && -n "$TEST_FILE" && -n "$OUTPUT_ROOT" ]] || {
  echo "--subflow, --train-file, --test-file and --output-dir are required" >&2
  usage >&2
  exit 2
}

CONDA_ENV="${CONDA_ENV:-skillmining310}"
ANALYSIS_BATCH_SIZE="${ANALYSIS_BATCH_SIZE:-8}"
EVOLUTION_BATCH_SIZE="${EVOLUTION_BATCH_SIZE:-25}"
MAP_BATCH_SIZE="${MAP_BATCH_SIZE:-8}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[[ -n "$WORKFLOW_IDS_RAW" ]] && export SKILLMINING_WORKFLOW_ID="${WORKFLOW_IDS_RAW%%,*}"
mkdir -p "$OUTPUT_ROOT"
if [[ -n "$EXISTING_FULL_DIR" && ! -f "$OUTPUT_ROOT/full/summary.json" ]]; then
  if [[ ! -f "$EXISTING_FULL_DIR/summary.json" ]]; then
    echo "Existing full run has no summary.json: $EXISTING_FULL_DIR" >&2
    exit 1
  fi
  ln -s "$EXISTING_FULL_DIR" "$OUTPUT_ROOT/full"
  echo "Linked existing full run: $OUTPUT_ROOT/full -> $EXISTING_FULL_DIR"
fi
REPLAY_ARGS=()
if [[ -n "$REUSE_ROLLOUT_DIR" ]]; then
  REPLAY_ARGS=(--reuse-rollout-dir "$REUSE_ROLLOUT_DIR")
fi
ANALYSIS_REPLAY_ARGS=()
if [[ -n "$REUSE_ANALYSIS_DIR" ]]; then
  ANALYSIS_REPLAY_ARGS=(--reuse-analysis-dir "$REUSE_ANALYSIS_DIR")
fi

run_variant() {
  local name="$1" workflow_id="$2"; shift 2
  echo "===== Trace2Skill ablation: $name ====="
  SKILLMINING_WORKFLOW_ID="$workflow_id" python scripts/run_trace2skill_abcd.py \
    --subflow "$SUBFLOW" --train-file "$TRAIN_FILE" --test-file "$TEST_FILE" \
    --output-dir "$OUTPUT_ROOT/$name" --skip-seed-test \
    --ablation-variant "$name" \
    --evolution-batch-size "$EVOLUTION_BATCH_SIZE" \
    --analysis-batch-size "$ANALYSIS_BATCH_SIZE" --map-batch-size "$MAP_BATCH_SIZE" \
    --skip-text-eval --continue-on-batch-error \
    "${REPLAY_ARGS[@]}" "${ANALYSIS_REPLAY_ARGS[@]}" "$@"
}

IFS=',' read -r -a WORKFLOW_IDS <<< "$WORKFLOW_IDS_RAW"
if [[ ${#WORKFLOW_IDS[@]} -eq 0 || -z "${WORKFLOW_IDS[0]:-}" ]]; then
  WORKFLOW_IDS=("")
fi

workflow_for() {
  local index="$1"
  echo "${WORKFLOW_IDS[$((index % ${#WORKFLOW_IDS[@]}))]}"
}

VARIANT_INDEX=0
PIDS=()
run_parallel_variant() {
  local name="$1"; shift
  local workflow_id
  workflow_id="$(workflow_for "$VARIANT_INDEX")"
  VARIANT_INDEX=$((VARIANT_INDEX + 1))
  run_variant "$name" "$workflow_id" "$@" &
  PIDS+=("$!")
}

variant_enabled() {
  local requested="$1" item
  IFS=',' read -r -a selected_variants <<< "$VARIANTS_RAW"
  for item in "${selected_variants[@]}"; do
    [[ "$item" == "$requested" ]] && return 0
  done
  return 1
}

variant_enabled no_adaptation && run_parallel_variant no_adaptation --skip-evolution
variant_enabled no_structured_evolution && run_parallel_variant no_structured_evolution --direct-memory-update
variant_enabled no_failure_analysis && run_parallel_variant no_failure_analysis --disable-failure-analysis
variant_enabled no_success_memory && run_parallel_variant no_success_memory --disable-success-analysis
variant_enabled one_shot_update && run_parallel_variant one_shot_update --one-shot-update

# The full run is often already available from the preceding experiment. Do
# not spend another complete LLM budget on it unless it is absent or forced.
if variant_enabled full; then
  if [[ "$FORCE_FULL" -eq 1 || ! -f "$OUTPUT_ROOT/full/summary.json" ]]; then
    run_parallel_variant full
  else
    echo "===== Trace2Skill ablation: full (reuse existing summary) ====="
    echo "Reusing $OUTPUT_ROOT/full/summary.json; set FORCE_FULL=1 to rerun."
  fi
fi

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one ablation variant failed." >&2
  exit "$status"
fi

python scripts/summarize_trace2skill_module_ablation.py --root "$OUTPUT_ROOT"
python scripts/audit_trace2skill_module_ablation.py --root "$OUTPUT_ROOT" --variants "$VARIANTS_RAW"
