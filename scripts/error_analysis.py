#!/usr/bin/env python3
r"""对比两个方法的 fail case → trace 回 skill → 分析根因 → 生成报告。

用法：
  # 从 run_subflow_eval 的产出对比 seed vs mined
  python scripts/error_analysis.py \
    --method-a outputs/subflow_eval_xxx/seed_predictions.json \
    --method-b outputs/subflow_eval_xxx/mined_predictions.json \
    --skill-a "" --skill-b outputs/subflow_eval_xxx/recover_password/skill.md \
    --reference outputs/subflow_eval_xxx/recover_password/reference.md \
    --subflow recover_password

  # 对比两个不同 skill 版本
  python scripts/error_analysis.py \
    --method-a outputs/v1/turn_predictions.json \
    --method-b outputs/v2/turn_predictions.json \
    --skill-a outputs/v1/skill.md --skill-b outputs/v2/skill.md \
    --subflow recover_password --max-cases 10
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = Path(f"outputs/error_analysis_{_TIMESTAMP}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "analysis.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Load & classify
# ═══════════════════════════════════════════════════════════════

def load_turn_results(path: str) -> list[dict]:
    """Load turn-level prediction results JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Handle both formats: flat list or {label, n, text, ...}
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "turns" in data:
        return data["turns"]
    return data


def classify_cases(
    turns_a: list[dict],
    turns_b: list[dict],
    threshold: float = 0.3,
) -> dict[str, list[dict]]:
    """Classify each turn into: a_better, b_better, both_fail, both_pass.

    Uses BERT-F1 from per_sample if available, otherwise simple text match.
    """
    from eval_tod.text_eval import evaluate_responses

    # Align by convo_id + turn_index
    key_a = {(r["convo_id"], r["turn_index"]): r for r in turns_a}
    key_b = {(r["convo_id"], r["turn_index"]): r for r in turns_b}
    common_keys = set(key_a) & set(key_b)

    results: dict[str, list[dict]] = {
        "a_better": [], "b_better": [], "both_fail": [], "both_pass": [],
    }

    for cid, tidx in sorted(common_keys):
        ra = key_a[(cid, tidx)]
        rb = key_b[(cid, tidx)]

        # Compute per-turn BERT-F1 against reference
        ref = ra.get("reference", "")
        pa = ra.get("prediction", "")
        pb = rb.get("prediction", "")

        if ref and pa and pb:
            sa = evaluate_responses([pa], [ref])
            sb = evaluate_responses([pb], [ref])
            fa, fb = sa.per_sample[0]["bert_f1"], sb.per_sample[0]["bert_f1"]
        else:
            fa = fb = 0.0

        diff = fb - fa
        entry = {
            "convo_id": cid, "turn_index": tidx,
            "subflow": ra.get("subflow", ""),
            "reference": ref[:300],
            "pred_a": pa[:300], "pred_b": pb[:300],
            "bert_a": round(fa, 4), "bert_b": round(fb, 4),
            "delta": round(diff, 4),
            "context": ra.get("context", "")[:500],
        }

        if diff > threshold:
            results["b_better"].append(entry)
        elif diff < -threshold:
            results["a_better"].append(entry)
        elif fa < 0.3 and fb < 0.3:
            results["both_fail"].append(entry)
        else:
            results["both_pass"].append(entry)

    return results


# ═══════════════════════════════════════════════════════════════
# LLM Analysis
# ═══════════════════════════════════════════════════════════════

def analyze_case(
    case: dict,
    skill_a: str,
    skill_b: str,
    reference_md: str,
    method_a_name: str,
    method_b_name: str,
) -> str:
    """LLM analyzes one divergent case."""
    from llm import chat

    prompt = f"""You are comparing two customer service agents on the same dialogue turn.

## Task
Analyze why one agent succeeded and the other failed. Trace the root cause back
to their respective skill documents.

## Dialogue Context
```
{case['context'][:800]}
```

## Reference (ground truth)
{case['reference'][:400]}

## Method A ({method_a_name}) — BERT-F1: {case['bert_a']:.4f}
{case['pred_a'][:400]}

## Method B ({method_b_name}) — BERT-F1: {case['bert_b']:.4f}
{case['pred_b'][:400]}

## Skill A ({method_a_name})
{skill_a[:3000] if skill_a else '(empty — no skill)'}

## Skill B ({method_b_name})
{skill_b[:3000] if skill_b else '(empty — no skill)'}

## Reference Snippets (training examples)
{reference_md[:2000] if reference_md else '(none)'}

## Analysis
Answer these questions:
1. **What went wrong for the worse agent?** Cite specific evidence.
2. **What did the better agent do right?** How did its skill help?
3. **Skill gap**: What is missing or wrong in the worse skill compared to the better one?
4. **Recommendation**: What concrete change would fix this? Quote the exact section to modify.

Output in Markdown:

### Case: {case['convo_id']} turn={case['turn_index']}

**Root Cause**: [1-2 sentences]

**A ({method_a_name}, BERT={case['bert_a']:.4f})**: [what A did + why it worked/failed]

**B ({method_b_name}, BERT={case['bert_b']:.4f})**: [what B did + why it worked/failed]

**Skill Gap**: [what is different between the two skills that caused this]

**Fix**: [concrete recommendation]
"""

    try:
        return chat(prompt, temperature=0.0, max_tokens=1024).strip()
    except Exception as e:
        return f"*LLM error: {e}*"


# ═══════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════

def generate_summary_report(
    classification: dict,
    analyses: list[str],
    method_a_name: str,
    method_b_name: str,
    subflow: str,
) -> str:
    """Generate the final summary report via LLM."""
    from llm import chat

    n_a_better = len(classification["a_better"])
    n_b_better = len(classification["b_better"])
    n_both_fail = len(classification["both_fail"])
    n_both_pass = len(classification["both_pass"])
    total = n_a_better + n_b_better + n_both_fail + n_both_pass
    if total == 0:
        return "# No cases to analyze"

    # Average deltas
    avg_b_better = (sum(c["delta"] for c in classification["b_better"]) /
                    max(n_b_better, 1))

    summaries = "\n\n---\n\n".join(analyses[:20])

    prompt = f"""You are writing an error analysis report comparing two customer service agents.

## Experiment
- **Subflow**: {subflow}
- **Method A**: {method_a_name}
- **Method B**: {method_b_name}
- **Total turns**: {total}

## Summary Statistics
- A better than B: {n_a_better} ({100*n_a_better/max(total,1):.0f}%)
- B better than A: {n_b_better} ({100*n_b_better/max(total,1):.0f}%)
- Both fail: {n_both_fail} ({100*n_both_fail/max(total,1):.0f}%)
- Both pass: {n_both_pass} ({100*n_both_pass/max(total,1):.0f}%)
- Average B-over-A delta: {avg_b_better:+.4f} BERT-F1

## Detailed Case Analyses
{summaries[:12000]}

## Instructions
Write a concise report:
1. **Executive Summary**: Which method is better and by how much?
2. **Common Failure Patterns**: What kinds of mistakes does each method make?
3. **Skill Comparison**: What structural differences in the skills cause the performance gap?
4. **Top 3 Recommendations**: What to change in the worse method's skill.

Output in Markdown with clear section headers."""

    try:
        return chat(prompt, temperature=0.0, max_tokens=2048).strip()
    except Exception as e:
        return f"*LLM error generating summary: {e}*\n\n{summaries}"


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare two methods' fail cases → trace to skill → report")
    parser.add_argument("--method-a", required=True,
                        help="Turn predictions JSON for method A")
    parser.add_argument("--method-b", required=True,
                        help="Turn predictions JSON for method B")
    parser.add_argument("--name-a", default="Method A")
    parser.add_argument("--name-b", default="Method B")
    parser.add_argument("--skill-a", default="",
                        help="skill.md path for method A")
    parser.add_argument("--skill-b", default="",
                        help="skill.md path for method B")
    parser.add_argument("--reference", default="",
                        help="reference.md path (shared for both)")
    parser.add_argument("--subflow", default="unknown")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Minimum BERT-F1 delta to classify as better/worse")
    parser.add_argument("--max-cases", type=int, default=15,
                        help="Max cases to analyze with LLM")
    args = parser.parse_args()

    # ── 1. Load ──────────────────────────────────────────────
    log.info(f"Loading {args.method_a}...")
    turns_a = load_turn_results(args.method_a)
    log.info(f"  {len(turns_a)} turns")

    log.info(f"Loading {args.method_b}...")
    turns_b = load_turn_results(args.method_b)
    log.info(f"  {len(turns_b)} turns")

    skill_a = Path(args.skill_a).read_text(encoding="utf-8") if args.skill_a else ""
    skill_b = Path(args.skill_b).read_text(encoding="utf-8") if args.skill_b else ""
    ref_md = Path(args.reference).read_text(encoding="utf-8") if args.reference else ""

    # ── 2. Classify ──────────────────────────────────────────
    log.info("Classifying cases...")
    classification = classify_cases(turns_a, turns_b, args.threshold)
    for cat, cases in classification.items():
        log.info(f"  {cat}: {len(cases)} turns")

    # Save classification
    (OUT_DIR / "classification.json").write_text(
        json.dumps({k: len(v) for k, v in classification.items()}, indent=2),
        encoding="utf-8")

    # ── 3. Analyze divergent cases ───────────────────────────
    # Focus on B-better (mined > seed) and both-fail
    cases_to_analyze = (
        classification["b_better"][:args.max_cases] +
        classification["a_better"][:5] +
        classification["both_fail"][:5]
    )

    log.info(f"Analyzing {len(cases_to_analyze)} cases with LLM...")
    analyses = []
    for i, case in enumerate(cases_to_analyze):
        log.info(f"  [{i+1}/{len(cases_to_analyze)}] "
                 f"{case['convo_id']} turn={case['turn_index']} "
                 f"Δ={case['delta']:+.4f}")
        result = analyze_case(
            case, skill_a, skill_b, ref_md, args.name_a, args.name_b)
        analyses.append(result)
        print(f"    {result[:120]}...")

    # ── 4. Generate summary report ───────────────────────────
    log.info("Generating summary report...")
    report = generate_summary_report(
        classification, analyses, args.name_a, args.name_b, args.subflow)

    report_path = OUT_DIR / "error_report.md"
    report_path.write_text(report, encoding="utf-8")
    log.info(f"Report saved → {report_path}")

    # Save all analyses
    (OUT_DIR / "case_analyses.md").write_text(
        "\n\n---\n\n".join(analyses), encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"DONE. Output: {OUT_DIR}")
    print(f"  error_report.md    — summary report with recommendations")
    print(f"  case_analyses.md   — per-case LLM analysis")
    print(f"  classification.json — turn classification counts")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
