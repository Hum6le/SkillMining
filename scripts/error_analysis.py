#!/usr/bin/env python3
r"""对比 HG (hypergraph skill) vs AWM (trained workflow) 的 fail case。

用 AST 判定成功/失败：predicted_action 是否匹配 ground truth。
分析每个分歧 case → trace 回各自的 skill → 生成报告。

用法：
  python scripts/error_analysis.py \
    --hg-preds outputs/subflow_eval_xxx/recover_password/mined_predictions.json \
    --awm-preds outputs/awm_abcd_xxx/test_final_preds.json \
    --hg-skill outputs/subflow_eval_xxx/recover_password/skill.md \
    --awm-workflow outputs/awm_abcd_xxx/awm_workflow.txt \
    --test-data data/eval/abcd/splits/recover_password/test.json \
    --subflow recover_password
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
    handlers=[logging.FileHandler(OUT_DIR / "analysis.log"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Ground truth extraction
# ═══════════════════════════════════════════════════════════════

def extract_gt_actions(conversations: list[dict]) -> dict[tuple[str, int], str]:
    """Build lookup: (convo_id, action_turn_index) → ground-truth action_name."""
    gt: dict[tuple[str, int], str] = {}
    for conv in conversations:
        cid = str(conv.get("convo_id", "?"))
        for turn in conv.get("delexed", []):
            targets = turn.get("targets", [])
            if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
                # Find the closest preceding agent turn
                # The turn index here is the ACTION turn's position
                gt[(cid, turn.get("turn_count", 0) - 1)] = str(targets[2])
    return gt


def find_action_turns(conversations: list[dict]) -> set[tuple[str, int]]:
    """Find all action turn positions: (convo_id, turn_index)."""
    action_turns: set[tuple[str, int]] = set()
    for conv in conversations:
        cid = str(conv.get("convo_id", "?"))
        for i, turn in enumerate(conv.get("delexed", [])):
            targets = turn.get("targets", [])
            if len(targets) >= 3 and targets[1] == "take_action":
                action_turns.add((cid, i))
    return action_turns


# ═══════════════════════════════════════════════════════════════
# Classification by AST
# ═══════════════════════════════════════════════════════════════

def classify_by_ast(
    hg_preds: list[dict],
    awm_preds: list[dict],
    test_convs: list[dict],
) -> dict[str, list[dict]]:
    """Classify each action turn: HG_fail_AWM_pass, AWM_fail_HG_pass, both_fail, both_pass.

    A turn PASSES if predicted_action matches ground truth action name.
    """
    gt_actions = extract_gt_actions(test_convs)
    action_turns = find_action_turns(test_convs)

    # Build per-method lookup: (convo_id, turn_index) → predicted_action
    def _build_lookup(preds: list[dict]) -> dict[tuple[str, int], dict]:
        lookup: dict[tuple[str, int], dict] = {}
        for r in preds:
            cid = r.get("convo_id", "")
            tidx = r.get("turn_index", -1)
            if tidx < 0:
                continue
            # Map agent turn's predicted_action to the NEXT action turns
            pa = r.get("predicted_action", "")
            lookup[(cid, tidx)] = {"agent_turn": tidx, "predicted_action": pa,
                                    "prediction": r.get("prediction", ""),
                                    "reference": r.get("reference", ""),
                                    "context": r.get("context", "")[:500]}
        return lookup

    hg_lookup = _build_lookup(hg_preds)
    awm_lookup = _build_lookup(awm_preds)

    results: dict[str, list[dict]] = {
        "hg_fail_awm_pass": [], "awm_fail_hg_pass": [],
        "both_fail": [], "both_pass": [],
    }

    # For each action turn, find the closest preceding agent prediction
    for cid, action_turn_idx in sorted(action_turns):
        gt_action = gt_actions.get((cid, action_turn_idx), "")
        if not gt_action:
            continue

        # Find closest preceding agent prediction for HG
        hg_pred = _find_nearest_agent(hg_lookup, cid, action_turn_idx)
        awm_pred = _find_nearest_agent(awm_lookup, cid, action_turn_idx)

        hg_action = hg_pred.get("predicted_action", "") if hg_pred else ""
        awm_action = awm_pred.get("predicted_action", "") if awm_pred else ""

        hg_ok = (hg_action == gt_action)
        awm_ok = (awm_action == gt_action)

        entry = {
            "convo_id": cid,
            "action_turn": action_turn_idx,
            "gt_action": gt_action,
            "hg_action": hg_action,
            "awm_action": awm_action,
            "hg_ok": hg_ok,
            "awm_ok": awm_ok,
            "hg_response": hg_pred.get("prediction", "")[:300] if hg_pred else "",
            "awm_response": awm_pred.get("prediction", "")[:300] if awm_pred else "",
            "reference": hg_pred.get("reference", "")[:300] if hg_pred else "",
            "context": hg_pred.get("context", "")[:500] if hg_pred else "",
        }

        if not hg_ok and awm_ok:
            results["hg_fail_awm_pass"].append(entry)
        elif hg_ok and not awm_ok:
            results["awm_fail_hg_pass"].append(entry)
        elif not hg_ok and not awm_ok:
            results["both_fail"].append(entry)
        else:
            results["both_pass"].append(entry)

    return results


def _find_nearest_agent(
    lookup: dict[tuple[str, int], dict],
    cid: str,
    action_turn_idx: int,
) -> dict | None:
    """Find the closest preceding agent turn prediction for an action turn."""
    best = None
    best_dist = 999
    for (lcid, tidx), pred in lookup.items():
        if lcid == cid and tidx < action_turn_idx:
            dist = action_turn_idx - tidx
            if dist < best_dist:
                best_dist = dist
                best = pred
    return best


# ═══════════════════════════════════════════════════════════════
# LLM Analysis
# ═══════════════════════════════════════════════════════════════

def analyze_case(
    entry: dict,
    hg_skill: str,
    awm_workflow: str,
    reference_md: str,
) -> str:
    """LLM analyzes one divergent case."""
    from llm import chat

    prompt = f"""Compare two agents on the same turn. One uses a hypergraph-mined skill (HG),
the other uses an AWM-trained workflow.

## Turn Details
- **Ground Truth Action**: `{entry['gt_action']}`
- **HG Predicted Action**: `{entry['hg_action']}` → {'✓ CORRECT' if entry['hg_ok'] else '✗ WRONG'}
- **AWM Predicted Action**: `{entry['awm_action']}` → {'✓ CORRECT' if entry['awm_ok'] else '✗ WRONG'}

## Conversation Context
```
{entry['context'][:600]}
```

## HG Response
{entry['hg_response'][:300]}

## AWM Response
{entry['awm_response'][:300]}

## HG Skill (mined from subgraph)
{hg_skill[:2500] if hg_skill else '(none)'}

## AWM Workflow (learned from training)
{awm_workflow[:2500] if awm_workflow else '(none)'}

## Reference Snippets
{reference_md[:1500] if reference_md else '(none)'}

## Questions
1. Why did {'HG fail' if not entry['hg_ok'] else 'AWM fail'} on this turn?
2. Why did {'AWM succeed' if entry['awm_ok'] else 'HG succeed'}?
3. What specific difference in their knowledge explains this?
4. What should be added to the failing method's skill/workflow?

Output in Markdown:

### Turn {entry['convo_id']} action@{entry['action_turn']}
**GT**: `{entry['gt_action']}` | **HG**: `{entry['hg_action']}` | **AWM**: `{entry['awm_action']}`

**Root Cause**: [why the failing method got it wrong]

**Knowledge Gap**: [what the failing method is missing that the successful one has]

**Fix**: [concrete change to the failing skill/workflow]
"""

    try:
        return chat(prompt, temperature=0.0, max_tokens=1024).strip()
    except Exception as e:
        return f"*LLM error: {e}*"


# ═══════════════════════════════════════════════════════════════
# Summary report
# ═══════════════════════════════════════════════════════════════

def generate_summary(
    classification: dict,
    analyses: list[str],
    subflow: str,
) -> str:
    """LLM synthesizes final report."""
    from llm import chat

    n = {k: len(v) for k, v in classification.items()}
    total = sum(n.values())
    if total == 0:
        return "# No action turns to analyze"

    summaries = "\n\n---\n\n".join(analyses[:20])

    prompt = f"""Synthesize an error analysis report comparing two agents: HG (hypergraph skill) and AWM (trained workflow).

## Subflow: {subflow}
## Total action turns: {total}

## Results
- HG fail, AWM pass: {n['hg_fail_awm_pass']} ({100*n['hg_fail_awm_pass']/max(total,1):.0f}%)
- AWM fail, HG pass: {n['awm_fail_hg_pass']} ({100*n['awm_fail_hg_pass']/max(total,1):.0f}%)
- Both fail: {n['both_fail']} ({100*n['both_fail']/max(total,1):.0f}%)
- Both pass: {n['both_pass']} ({100*n['both_pass']/max(total,1):.0f}%)

## Case Analyses
{summaries[:12000]}

Write a report:
1. **Which method is better** and by what margin?
2. **Common failure patterns** for each method
3. **Knowledge differences**: what does the better method know that the worse one doesn't?
4. **Top 3 recommendations** to improve the worse method

Output in Markdown."""

    try:
        return chat(prompt, temperature=0.0, max_tokens=2048).strip()
    except Exception as e:
        return f"*LLM error: {e}*\n\n{summaries}"


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare HG vs AWM fail cases by AST → trace to skill → report")
    parser.add_argument("--hg-preds", required=True,
                        help="HG mined_predictions.json")
    parser.add_argument("--awm-preds", required=True,
                        help="AWM turn predictions JSON")
    parser.add_argument("--hg-skill", default="",
                        help="HG skill.md path")
    parser.add_argument("--awm-workflow", default="",
                        help="AWM workflow.txt path")
    parser.add_argument("--reference", default="",
                        help="reference.md path (from HG mining)")
    parser.add_argument("--test-data", required=True,
                        help="Subflow test.json (for ground truth actions)")
    parser.add_argument("--subflow", default="unknown")
    parser.add_argument("--max-cases", type=int, default=15)
    args = parser.parse_args()

    # ── 1. Load ──────────────────────────────────────────────
    log.info(f"Loading HG predictions: {args.hg_preds}")
    hg_preds = json.loads(Path(args.hg_preds).read_text(encoding="utf-8"))
    log.info(f"  {len(hg_preds)} turns")

    log.info(f"Loading AWM predictions: {args.awm_preds}")
    awm_preds_raw = json.loads(Path(args.awm_preds).read_text(encoding="utf-8"))
    # AWM preds might be flat list or nested
    if isinstance(awm_preds_raw, list) and len(awm_preds_raw) > 0:
        if isinstance(awm_preds_raw[0], dict) and "turns" in awm_preds_raw[0]:
            awm_preds = []
            for d in awm_preds_raw:
                awm_preds.extend(d.get("turns", []))
        else:
            awm_preds = awm_preds_raw
    else:
        awm_preds = awm_preds_raw
    log.info(f"  {len(awm_preds)} turns")

    log.info(f"Loading test data: {args.test_data}")
    test_convs = json.loads(Path(args.test_data).read_text(encoding="utf-8"))
    log.info(f"  {len(test_convs)} conversations")

    hg_skill = Path(args.hg_skill).read_text(encoding="utf-8") if args.hg_skill else ""
    awm_workflow = Path(args.awm_workflow).read_text(encoding="utf-8") if args.awm_workflow else ""
    ref_md = Path(args.reference).read_text(encoding="utf-8") if args.reference else ""

    # ── 2. Classify by AST ───────────────────────────────────
    log.info("Classifying by AST (action match)...")
    classification = classify_by_ast(hg_preds, awm_preds, test_convs)
    for cat, cases in classification.items():
        log.info(f"  {cat}: {len(cases)} turns")

    (OUT_DIR / "classification.json").write_text(
        json.dumps({k: len(v) for k, v in classification.items()}, indent=2),
        encoding="utf-8")

    # Save per-category details
    for cat, cases in classification.items():
        (OUT_DIR / f"{cat}.json").write_text(
            json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 3. Analyze ───────────────────────────────────────────
    cases_to_analyze = (
        classification["hg_fail_awm_pass"][:args.max_cases] +
        classification["awm_fail_hg_pass"][:5] +
        classification["both_fail"][:5]
    )

    log.info(f"Analyzing {len(cases_to_analyze)} cases with LLM...")
    analyses = []
    for i, entry in enumerate(cases_to_analyze):
        log.info(f"  [{i+1}/{len(cases_to_analyze)}] "
                 f"{entry['convo_id']} action@{entry['action_turn']} "
                 f"GT={entry['gt_action']} HG={entry['hg_action']} AWM={entry['awm_action']}")
        result = analyze_case(entry, hg_skill, awm_workflow, ref_md)
        analyses.append(result)

    # ── 4. Summary report ────────────────────────────────────
    log.info("Generating summary report...")
    report = generate_summary(classification, analyses, args.subflow)

    (OUT_DIR / "error_report.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "case_analyses.md").write_text(
        "\n\n---\n\n".join(analyses), encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"DONE. Output: {OUT_DIR}")
    print(f"  error_report.md       — summary report")
    print(f"  case_analyses.md      — per-case LLM analysis")
    print(f"  hg_fail_awm_pass.json — cases AWWM wins")
    print(f"  awm_fail_hg_pass.json — cases HG wins")
    print(f"  both_fail.json        — both fail")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
