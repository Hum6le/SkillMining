#!/usr/bin/env bash

# Run paired HG-vs-AWM action-turn error analysis for every legacy ABCD
# subflow split. Each subflow is analyzed with the same AST mapping used by
# scripts/error_analysis.py, then summary stats are aggregated locally.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="python"
CONDA_ENV="skillmining310"
HF_MIRROR="https://hf-mirror.com"
HG_ROOT=""
AWM_ROOT=""
SPLITS_ROOT=""
OUTPUT_DIR=""
MAX_CASES=20
RESUME=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_full_hg_vs_awm_error_analysis.sh [options]

Required:
  --hg-root DIR          Legacy HG run root containing <subflow>/skill.md and
                         <subflow>/mined_predictions.json
  --awm-root DIR         Legacy AWM run root containing <subflow>/workflow.txt
                         and <subflow>/test_turn_predictions.json
  --splits-root DIR      Legacy 96-subflow split root containing
                         <subflow>/test.json

Optional:
  --output-dir DIR       Output root (default: outputs/full_hg_vs_awm_error_analysis_<time>)
  --max-cases N          Max LLM-analyzed cases per comparison category (default: 20)
  --resume               Reuse completed subflow reports under --output-dir
  --python PATH          Python executable (default: python)
  --conda-env NAME       Conda environment (default: skillmining310)
  --hf-mirror URL        Hugging Face mirror (default: https://hf-mirror.com)
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hg-root) HG_ROOT="$2"; shift 2 ;;
        --awm-root) AWM_ROOT="$2"; shift 2 ;;
        --splits-root) SPLITS_ROOT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-cases) MAX_CASES="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --hf-mirror) HF_MIRROR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for item in HG_ROOT AWM_ROOT SPLITS_ROOT; do
    [[ -n "${!item}" ]] || { echo "--${item,,} is required" >&2; usage >&2; exit 2; }
    [[ -d "${!item}" ]] || { echo "Directory not found: ${!item}" >&2; exit 1; }
done

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
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python was not found: $PYTHON_BIN" >&2; exit 1; }

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$ROOT_DIR/outputs/full_hg_vs_awm_error_analysis_$(date +%Y-%m-%d_%H-%M-%S)"
fi
if [[ "$RESUME" -eq 1 && ! -d "$OUTPUT_DIR" ]]; then
    echo "--resume requires an existing --output-dir: $OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

echo "HG vs AWM legacy full error analysis"
echo "HG root:      $HG_ROOT"
echo "AWM root:     $AWM_ROOT"
echo "Splits root:  $SPLITS_ROOT"
echo "Output root:  $OUTPUT_DIR"
echo "Environment:  $CONDA_ENV"
echo "HF_ENDPOINT:  $HF_ENDPOINT"
echo "Max cases:    $MAX_CASES"

FAILED="$OUTPUT_DIR/failed_subflows.txt"
MISSING="$OUTPUT_DIR/missing_artifacts.txt"
: > "$FAILED"
: > "$MISSING"

completed=0
attempted=0
for split_dir in "$SPLITS_ROOT"/*; do
    [[ -d "$split_dir" && -f "$split_dir/test.json" ]] || continue
    subflow="${split_dir##*/}"
    hg_dir="$HG_ROOT/$subflow"
    awm_dir="$AWM_ROOT/$subflow"
    sub_out="$OUTPUT_DIR/$subflow"

    if [[ "$RESUME" -eq 1 && -f "$sub_out/stats.json" && -f "$sub_out/error_report.md" ]]; then
        echo "SKIP completed: $subflow"
        completed=$((completed + 1))
        continue
    fi

    hg_preds="$hg_dir/mined_predictions.json"
    awm_preds="$awm_dir/test_turn_predictions.json"
    hg_skill="$hg_dir/skill.md"
    reference="$hg_dir/reference.md"
    awm_workflow="$awm_dir/workflow.txt"
    [[ -f "$awm_workflow" ]] || awm_workflow="$awm_dir/awm_workflow.txt"

    missing=()
    [[ -f "$hg_preds" ]] || missing+=("hg_predictions")
    [[ -f "$awm_preds" ]] || missing+=("awm_predictions")
    [[ -f "$hg_skill" ]] || missing+=("hg_skill")
    [[ -f "$awm_workflow" ]] || missing+=("awm_workflow")
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "$subflow: ${missing[*]}" | tee -a "$MISSING"
        continue
    fi

    echo "===== HG vs AWM: $subflow ====="
    mkdir -p "$sub_out"
    command=("$PYTHON_BIN" scripts/error_analysis.py
        --hg-preds "$hg_preds"
        --awm-preds "$awm_preds"
        --hg-skill "$hg_skill"
        --awm-workflow "$awm_workflow"
        --test-data "$split_dir/test.json"
        --subflow "$subflow"
        --max-cases "$MAX_CASES"
        --output-dir "$sub_out")
    [[ -f "$reference" ]] && command+=(--reference "$reference")
    "${command[@]}" || { echo "$subflow" | tee -a "$FAILED"; continue; }
    attempted=$((attempted + 1))
done

"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for path in sorted(root.glob("*/stats.json")):
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        total = int(row.get("total_action_turns", 0))
        if total > 0:
            records.append({**row, "run_dir": str(path.parent)})
    except (OSError, ValueError, json.JSONDecodeError):
        continue

total = sum(row["total_action_turns"] for row in records)
weighted = lambda key: round(
    sum(float(row.get(key, 0.0)) * row["total_action_turns"] for row in records) / max(total, 1), 6
)
categories = ("hg_fail_awm_pass", "awm_fail_hg_pass", "both_fail", "both_pass")
summary = {
    "protocol": "legacy_96_subflow_pairwise_hg_vs_awm",
    "num_completed_subflows": len(records),
    "total_action_turns": total,
    "weighted_ast_joint": {
        "hg": weighted("ast_joint_hg"),
        "awm": weighted("ast_joint_awm"),
    },
    "classification_counts": {
        key: sum(int(row.get("classification", {}).get(key, 0)) for row in records)
        for key in categories
    },
    "records": records,
}
(root / "comparison_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps({key: summary[key] for key in (
    "num_completed_subflows", "total_action_turns", "weighted_ast_joint", "classification_counts",
)}, indent=2, ensure_ascii=False))
PY

echo "===== Full HG vs AWM error analysis finished ====="
echo "Completed reports: $OUTPUT_DIR"
echo "Comparison summary: $OUTPUT_DIR/comparison_summary.json"
echo "Missing artifacts:  $MISSING"
echo "Failed subflows:    $FAILED"
echo "Processed this run: $attempted; skipped completed: $completed"
