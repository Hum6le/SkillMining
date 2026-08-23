#!/usr/bin/env bash

# Paired HG-backbone vs AWM error analysis for the current 10-flow ABCD split.
# This intentionally does not share assumptions with the legacy 96-subflow
# runner, whose method artifacts live under separate roots and file layouts.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="skillmining310"
HF_MIRROR="https://hf-mirror.com"
RUN_ROOT=""
HG_ROOT=""
AWM_ROOT=""
SPLITS_DIR="$ROOT_DIR/data/eval/abcd/splits"
OUTPUT_DIR=""
MAX_CASES=20
RESUME=0
PYTHON_BIN="python"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_full_hg_vs_awm_error_analysis_10flow.sh (--run-root DIR | --hg-root DIR --awm-root DIR) [options]

Required:
  --run-root DIR          Current combined full_abcd run containing awm/<flow>/ and graph/<flow>/.
  --hg-root DIR           HG method root: graph/<flow>/... or directly <flow>/... .
  --awm-root DIR          AWM method root: awm/<flow>/... or directly <flow>/... .

Options:
  --splits-dir DIR        Current 10-flow split root with INDEX.json
                           (default: data/eval/abcd/splits)
  --output-dir DIR        Output root (default: <run-root>/hg_vs_awm_error_analysis,
                           or outputs/hg_vs_awm_error_analysis_10flow_<time>)
  --max-cases N           Max LLM-analyzed cases per category per flow (default: 20)
  --resume                Skip flows with complete stats.json and error_report.md
  --conda-env NAME        Conda environment (default: skillmining310)
  --hf-mirror URL         Hugging Face endpoint (default: https://hf-mirror.com)
  --python PATH           Python executable (default: python)
  -h, --help              Show this help

Expected artifacts for each indexed flow NAME:
  <hg-root>/NAME/mined_predictions.json
  <hg-root>/NAME/skill.md
  <hg-root>/NAME/reference.md                         (optional)
  <awm-root>/NAME/test_turn_predictions.json
  <awm-root>/NAME/awm_workflow.txt
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        --hg-root) HG_ROOT="$2"; shift 2 ;;
        --awm-root) AWM_ROOT="$2"; shift 2 ;;
        --splits-dir) SPLITS_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-cases) MAX_CASES="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --hf-mirror) HF_MIRROR="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "$RUN_ROOT" && ( -n "$HG_ROOT" || -n "$AWM_ROOT" ) ]]; then
    echo "Use either --run-root or --hg-root/--awm-root, not both." >&2
    exit 2
fi
if [[ -n "$RUN_ROOT" ]]; then
    RUN_ROOT="$(cd "$RUN_ROOT" 2>/dev/null && pwd)" || { echo "Run root not found: $RUN_ROOT" >&2; exit 1; }
    HG_ROOT="$RUN_ROOT/graph"
    AWM_ROOT="$RUN_ROOT/awm"
else
    [[ -n "$HG_ROOT" && -n "$AWM_ROOT" ]] || {
        echo "Provide --run-root or both --hg-root and --awm-root." >&2; usage >&2; exit 2;
    }
    HG_ROOT="$(cd "$HG_ROOT" 2>/dev/null && pwd)" || { echo "HG root not found: $HG_ROOT" >&2; exit 1; }
    AWM_ROOT="$(cd "$AWM_ROOT" 2>/dev/null && pwd)" || { echo "AWM root not found: $AWM_ROOT" >&2; exit 1; }
    # Accept a full run root as a convenience, but normalize to the method root.
    [[ -d "$HG_ROOT/graph" ]] && HG_ROOT="$HG_ROOT/graph"
    [[ -d "$AWM_ROOT/awm" ]] && AWM_ROOT="$AWM_ROOT/awm"
fi
[[ -d "$HG_ROOT" ]] || { echo "HG method root not found: $HG_ROOT" >&2; exit 1; }
[[ -d "$AWM_ROOT" ]] || { echo "AWM method root not found: $AWM_ROOT" >&2; exit 1; }
SPLITS_DIR="$(cd "$SPLITS_DIR" 2>/dev/null && pwd)" || { echo "Splits directory not found: $SPLITS_DIR" >&2; exit 1; }
INDEX_PATH="$SPLITS_DIR/INDEX.json"
[[ -f "$INDEX_PATH" ]] || { echo "Missing 10-flow split index: $INDEX_PATH" >&2; exit 1; }

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
    if [[ -n "$RUN_ROOT" ]]; then
        OUTPUT_DIR="$RUN_ROOT/hg_vs_awm_error_analysis"
    else
        OUTPUT_DIR="$ROOT_DIR/outputs/hg_vs_awm_error_analysis_10flow_$(date +%Y-%m-%d_%H-%M-%S)"
    fi
fi
if [[ "$RESUME" -eq 1 && ! -d "$OUTPUT_DIR" ]]; then
    echo "--resume requires an existing --output-dir: $OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

mapfile -t FLOWS < <("$PYTHON_BIN" - "$INDEX_PATH" "$SPLITS_DIR" <<'PY'
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
[[ ${#FLOWS[@]} -eq 10 ]] || { echo "Expected 10 indexed flows, found ${#FLOWS[@]}. Refusing legacy split input." >&2; exit 1; }

echo "HG backbone vs AWM paired error analysis"
echo "Protocol:     current 10-flow INDEX.json"
echo "HG root:      $HG_ROOT"
echo "AWM root:     $AWM_ROOT"
echo "Splits root:  $SPLITS_DIR"
echo "Output root:  $OUTPUT_DIR"
echo "Environment:  $CONDA_ENV"
echo "HF_ENDPOINT:  $HF_ENDPOINT"
echo "Max cases:    $MAX_CASES"

FAILED="$OUTPUT_DIR/failed_flows.txt"
MISSING="$OUTPUT_DIR/missing_artifacts.txt"
: > "$FAILED"
: > "$MISSING"

completed=0
processed=0
for flow in "${FLOWS[@]}"; do
    graph_dir="$HG_ROOT/$flow"
    awm_dir="$AWM_ROOT/$flow"
    flow_out="$OUTPUT_DIR/$flow"
    if [[ "$RESUME" -eq 1 && -f "$flow_out/stats.json" && -f "$flow_out/error_report.md" ]]; then
        echo "SKIP completed: $flow"
        completed=$((completed + 1))
        continue
    fi

    hg_preds="$graph_dir/mined_predictions.json"
    hg_skill="$graph_dir/skill.md"
    hg_ref="$graph_dir/reference.md"
    awm_preds="$awm_dir/test_turn_predictions.json"
    awm_workflow="$awm_dir/awm_workflow.txt"
    [[ -f "$awm_workflow" ]] || awm_workflow="$awm_dir/workflow.txt"
    missing=()
    [[ -f "$hg_preds" ]] || missing+=("graph_predictions")
    [[ -f "$hg_skill" ]] || missing+=("graph_skill")
    [[ -f "$awm_preds" ]] || missing+=("awm_predictions")
    [[ -f "$awm_workflow" ]] || missing+=("awm_workflow")
    if [[ ${#missing[@]} -gt 0 ]]; then
        {
            echo "$flow: ${missing[*]}"
            echo "  HG directory:  $graph_dir"
            echo "  HG preds:      $hg_preds"
            echo "  HG skill:      $hg_skill"
            echo "  AWM directory: $awm_dir"
            echo "  AWM preds:     $awm_preds"
            echo "  AWM workflow:  $awm_workflow"
        } | tee -a "$MISSING"
        continue
    fi

    echo "===== HG backbone vs AWM: $flow ====="
    mkdir -p "$flow_out"
    command=("$PYTHON_BIN" scripts/error_analysis.py
        --hg-preds "$hg_preds" --awm-preds "$awm_preds"
        --hg-skill "$hg_skill" --awm-workflow "$awm_workflow"
        --test-data "$SPLITS_DIR/$flow/test.json" --subflow "$flow"
        --max-cases "$MAX_CASES" --output-dir "$flow_out")
    [[ -f "$hg_ref" ]] && command+=(--reference "$hg_ref")
    "${command[@]}" || { echo "$flow" | tee -a "$FAILED"; continue; }
    processed=$((processed + 1))
done

# Do not silently turn an artifact-path mistake into a plausible-looking
# all-zero summary.  A resumed run may legitimately process no new flow,
# but it must already contain at least one completed per-flow analysis.
existing_stats=$(find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -name stats.json -type f | wc -l)
if [[ "$existing_stats" -eq 0 ]]; then
    echo "No flow was analyzed; comparison_summary.json will not be written." >&2
    echo "Inspect expected artifact paths in: $MISSING" >&2
    exit 1
fi

"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/stats.json")):
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        total = int(row.get("total_action_turns", 0))
        if total > 0:
            rows.append({**row, "run_dir": str(path.parent)})
    except (OSError, ValueError, json.JSONDecodeError):
        continue

total = sum(row["total_action_turns"] for row in rows)
def weighted(key):
    return round(sum(float(row.get(key, 0.0)) * row["total_action_turns"] for row in rows) / max(total, 1), 6)
categories = ("hg_fail_awm_pass", "awm_fail_hg_pass", "both_fail", "both_pass")
summary = {
    "protocol": "current_10_flow_pairwise_hg_backbone_vs_awm",
    "num_completed_flows": len(rows),
    "total_action_turns": total,
    "weighted_ast_joint": {"hg_backbone": weighted("ast_joint_hg"), "awm": weighted("ast_joint_awm")},
    "classification_counts": {key: sum(int(row.get("classification", {}).get(key, 0)) for row in rows) for key in categories},
    "records": rows,
}
(root / "comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({key: summary[key] for key in ("num_completed_flows", "total_action_turns", "weighted_ast_joint", "classification_counts")}, indent=2, ensure_ascii=False))
PY

echo "===== 10-flow paired error analysis finished ====="
echo "Comparison summary: $OUTPUT_DIR/comparison_summary.json"
echo "Missing artifacts:  $MISSING"
echo "Failed flows:       $FAILED"
echo "Processed this run: $processed; skipped completed: $completed"
