#!/usr/bin/env python3
r"""HG vs AWM 错误分析 — 统一用项目标准 AST 公式，统计全部 fail case。

用 `turn_results_to_abcd_predictions` + `compute_ast` 保证两种方法算 AST 完全一致。

用法：
  # 1. 先跑 HG mining（per-subflow）
  python scripts/run_subflow_eval.py --subflow recover_password --skip-seed

  # 2. 跑 AWM 训练（限定同一个 subflow）
  python scripts/run_full_experiment.py \
    --train-file data/eval/abcd/splits/recover_password/train.json \
    --test-file data/eval/abcd/splits/recover_password/test.json

  # 3. 对比分析
  python scripts/error_analysis.py \
    --hg-preds outputs/subflow_eval_xxx/recover_password/mined_predictions.json \
    --awm-preds outputs/full_experiment_xxx/trained_predictions.json \
    --hg-skill outputs/subflow_eval_xxx/recover_password/skill.md \
    --awm-workflow outputs/full_experiment_xxx/awm_workflow.txt \
    --test-data data/eval/abcd/splits/recover_password/test.json \
    --subflow recover_password

注意：AWM preds 用 trained_predictions.json（含 predicted_action），
不能用 test_final_preds.json（只有 NL text）。
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

from eval_tod.abcd.agent import turn_results_to_abcd_predictions
from eval_tod.abcd.data import extract_ground_truth
from eval_tod.abcd.metrics import compute_ast

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
# Classification — unified AST formula
# ═══════════════════════════════════════════════════════════════

def classify_by_ast_unified(
    hg_turn_results: list[dict],
    awm_turn_results: list[dict],
    test_convs: list[dict],
) -> tuple[dict, dict, dict]:
    """Use the SAME turn_results_to_abcd_predictions + compute_ast for both methods."""
    # ── Normalise convo_id to str everywhere ──
    for r in hg_turn_results:
        r["convo_id"] = str(r.get("convo_id", ""))
    for r in awm_turn_results:
        r["convo_id"] = str(r.get("convo_id", ""))
        # Fallback: if no predicted_action, try to parse from prediction text
        if "predicted_action" not in r and "prediction" in r:
            from eval_tod.abcd.agent import _parse_action_response
            action, slots, _ = _parse_action_response(r.get("prediction", ""))
            r["predicted_action"] = action
            r["predicted_slots"] = slots
    for conv in test_convs:
        conv["convo_id"] = str(conv.get("convo_id", ""))

    hg_abcd = turn_results_to_abcd_predictions(hg_turn_results, test_convs)
    awm_abcd = turn_results_to_abcd_predictions(awm_turn_results, test_convs)

    log.info(f"  HG: {len(hg_abcd)} convs mapped, AWM: {len(awm_abcd)} convs mapped "
             f"(test has {len(test_convs)} convs)")

    # Build lookup by convo_id for both
    hg_by_cid: dict[str, dict[int, str]] = {}
    for p in hg_abcd:
        hg_by_cid[str(p.conversation_id)] = {t.turn_index: t.predicted_action or "" for t in p.turns}
    awm_by_cid: dict[str, dict[int, str]] = {}
    for p in awm_abcd:
        awm_by_cid[str(p.conversation_id)] = {t.turn_index: t.predicted_action or "" for t in p.turns}

    # Debug: check a sample mapping
    if hg_by_cid and awm_by_cid:
        sample_cid = list(hg_by_cid.keys())[0]
        hg_sample = hg_by_cid.get(sample_cid, {})
        awm_sample = awm_by_cid.get(sample_cid, {})
        log.info(f"  Sample convo={sample_cid}: HG actions={len(hg_sample)}, AWM actions={len(awm_sample)}")
        common_turns = set(hg_sample) & set(awm_sample)
        diffs = sum(1 for t in common_turns if hg_sample[t] != awm_sample[t])
        log.info(f"  Common turns: {len(common_turns)}, different predictions: {diffs}")

    # Per-action-turn classification
    results: dict[str, list[dict]] = {
        "hg_fail_awm_pass": [], "awm_fail_hg_pass": [],
        "both_fail": [], "both_pass": [],
    }

    for conv in test_convs:
        cid = str(conv.get("convo_id", ""))
        truths = extract_ground_truth(conv)
        hg_preds = hg_by_cid.get(cid, {})
        awm_preds = awm_by_cid.get(cid, {})

        for gt in truths:
            if gt.turn_type != "action" or not gt.action_name:
                continue

            hg_action = hg_preds.get(gt.turn_index, "")
            awm_action = awm_preds.get(gt.turn_index, "")
            hg_ok = (hg_action == gt.action_name)
            awm_ok = (awm_action == gt.action_name)

            entry = {
                "convo_id": cid,
                "action_turn": gt.turn_index,
                "gt_action": gt.action_name,
                "hg_action": hg_action,
                "awm_action": awm_action,
                "hg_ok": hg_ok,
                "awm_ok": awm_ok,
                "hg_response": _find_turn_response(hg_turn_results, cid, gt.turn_index)[:300],
                "awm_response": _find_turn_response(awm_turn_results, cid, gt.turn_index)[:300],
                "reference": gt.text[:300],
                "context": _extract_context(conv, gt.turn_index),
            }

            if not hg_ok and awm_ok:
                results["hg_fail_awm_pass"].append(entry)
            elif hg_ok and not awm_ok:
                results["awm_fail_hg_pass"].append(entry)
            elif not hg_ok and not awm_ok:
                results["both_fail"].append(entry)
            else:
                results["both_pass"].append(entry)

    return results, {}, {}  # AST aggregates computed separately if needed


def _find_turn_response(turn_results: list[dict], cid: str, action_turn: int) -> str:
    """Find the agent response preceding this action turn."""
    best = ""
    for r in turn_results:
        if str(r.get("convo_id", "")) == cid and r.get("turn_index", 999) < action_turn:
            best = r.get("prediction", "")
    return best


def _extract_context(conv: dict, turn_idx: int, n_before: int = 4) -> str:
    """Extract conversation context before a given turn."""
    delexed = conv.get("delexed", [])
    lines = []
    for i in range(max(0, turn_idx - n_before), turn_idx):
        t = delexed[i]
        spk = t.get("speaker", "unknown")
        txt = t.get("text", "").strip()
        if txt:
            label = {"agent": "Agent", "customer": "Customer", "action": "System"}.get(spk, spk)
            lines.append(f"[{label}] {txt}")
    return "\n".join(lines)


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

    prompt = f"""Compare HG (hypergraph skill) vs AWM (trained workflow) on one action turn.

## Turn
- **GT Action**: `{entry['gt_action']}`
- **HG**: `{entry['hg_action']}` → {'✓' if entry['hg_ok'] else '✗'}
- **AWM**: `{entry['awm_action']}` → {'✓' if entry['awm_ok'] else '✗'}

## Context
```
{entry['context'][:500]}
```

## HG Response
{entry['hg_response'][:250]}

## AWM Response
{entry['awm_response'][:250]}

## HG Skill
{hg_skill[:2000] if hg_skill else '(none)'}

## AWM Workflow
{awm_workflow[:2000] if awm_workflow else '(none)'}

## Reference Snippets
{reference_md[:1000] if reference_md else '(none)'}

用中文回复。输出格式：

### {entry['convo_id']} action@{entry['action_turn']}
**GT**: `{entry['gt_action']}` | **HG**: `{entry['hg_action']}` | **AWM**: `{entry['awm_action']}`

**根因**: [1-2句，为什么失败的方法在这个 turn 上错了]

**知识差距**: [成功的方法具备什么知识/模式是失败的方法缺少的]

**改进建议**: [具体怎么修改失败方法的 skill/workflow]
"""

    try:
        return chat(prompt, temperature=0.0, max_tokens=1024).strip()
    except Exception as e:
        return f"*LLM error: {e}*"


# ═══════════════════════════════════════════════════════════════
# Summary
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
        return "# No action turns"

    hg_correct = n.get("awm_fail_hg_pass", 0) + n.get("both_pass", 0)
    awm_correct = n.get("hg_fail_awm_pass", 0) + n.get("both_pass", 0)
    hg_joint = hg_correct / max(total, 1)
    awm_joint = awm_correct / max(total, 1)

    summaries = "\n\n---\n\n".join(analyses[:25])

    prompt = f"""Synthesize an error analysis report: HG (hypergraph skill) vs AWM (trained workflow).

## Subflow: {subflow}
## Total action turns: {total}
## AST Joint: HG={hg_joint:.4f}, AWM={awm_joint:.4f}

## Results (by action match)
- HG fail, AWM pass: {n['hg_fail_awm_pass']} ({100*n['hg_fail_awm_pass']/max(total,1):.0f}%)
- AWM fail, HG pass: {n['awm_fail_hg_pass']} ({100*n['awm_fail_hg_pass']/max(total,1):.0f}%)
- Both fail: {n['both_fail']} ({100*n['both_fail']/max(total,1):.0f}%)
- Both pass: {n['both_pass']} ({100*n['both_pass']/max(total,1):.0f}%)

## Case Analyses
{summaries[:12000]}

用中文写报告：

1. **总体结论**: 哪个方法更好，差距多大
2. **HG 常见失败模式**: 哪些类型的 action 容易错
3. **AWM 常见失败模式**: 哪些类型的 action 容易错
4. **知识差异**: 两个方法的 skill/workflow 结构差异是什么
5. **Top 3 改进建议**: 针对较弱方法的改进方案

Markdown 格式。"""

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
        description="HG vs AWM error analysis — unified AST, all errors")
    parser.add_argument("--hg-preds", required=True)
    parser.add_argument("--awm-preds", required=True)
    parser.add_argument("--hg-skill", default="")
    parser.add_argument("--awm-workflow", default="")
    parser.add_argument("--reference", default="")
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--subflow", default="unknown")
    parser.add_argument("--max-cases", type=int, default=20,
                        help="Max cases to LLM-analyze (all are counted in stats)")
    args = parser.parse_args()

    # ── 1. Load ──────────────────────────────────────────────
    log.info("Loading predictions...")
    hg_preds = json.loads(Path(args.hg_preds).read_text(encoding="utf-8"))
    awm_preds = json.loads(Path(args.awm_preds).read_text(encoding="utf-8"))
    # Detect format: do predictions have predicted_action?
    hg_has_actions = any("predicted_action" in r for r in hg_preds[:10])
    awm_has_actions = any("predicted_action" in r for r in awm_preds[:10])
    log.info(f"  HG: {len(hg_preds)} turns (has predicted_action: {hg_has_actions})")
    log.info(f"  AWM: {len(awm_preds)} turns (has predicted_action: {awm_has_actions})")

    if not hg_has_actions:
        log.warning("  HG preds lack predicted_action — use "
                    "generate_all_turn_predictions(predict_actions=True) output!")
    if not awm_has_actions:
        log.warning("  AWM preds lack predicted_action — use "
                    "generate_all_turn_predictions(predict_actions=True) output! "
                    "Try: test_turn_predictions.json instead of test_final_preds.json")

    log.info(f"Loading test data: {args.test_data}")
    test_convs = json.loads(Path(args.test_data).read_text(encoding="utf-8"))
    log.info(f"  {len(test_convs)} conversations")

    hg_skill = Path(args.hg_skill).read_text(encoding="utf-8") if args.hg_skill else ""
    awm_wf = Path(args.awm_workflow).read_text(encoding="utf-8") if args.awm_workflow else ""
    ref_md = Path(args.reference).read_text(encoding="utf-8") if args.reference else ""

    # ── 2. Classify — unified AST ────────────────────────────
    log.info("Classifying by AST (unified formula)...")
    classification, _, _ = classify_by_ast_unified(hg_preds, awm_preds, test_convs)

    n = {k: len(v) for k, v in classification.items()}
    total = sum(n.values())

    hg_correct = n["awm_fail_hg_pass"] + n["both_pass"]
    awm_correct = n["hg_fail_awm_pass"] + n["both_pass"]
    hg_joint = hg_correct / max(total, 1)
    awm_joint = awm_correct / max(total, 1)

    log.info(f"  AST Joint — HG: {hg_joint:.4f}, AWM: {awm_joint:.4f}")
    log.info(f"  HG fail / AWM pass: {n['hg_fail_awm_pass']} ({100*n['hg_fail_awm_pass']/max(total,1):.0f}%)")
    log.info(f"  AWM fail / HG pass: {n['awm_fail_hg_pass']} ({100*n['awm_fail_hg_pass']/max(total,1):.0f}%)")
    log.info(f"  Both fail:          {n['both_fail']} ({100*n['both_fail']/max(total,1):.0f}%)")
    log.info(f"  Both pass:          {n['both_pass']} ({100*n['both_pass']/max(total,1):.0f}%)")

    # Save ALL cases (not sampled)
    for cat, cases in classification.items():
        (OUT_DIR / f"{cat}.json").write_text(
            json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"  Saved {len(cases)} cases → {cat}.json")

    # ── 3. LLM analyze errors ────────────────────────────────
    # Prioritize divergent + both_fail; sample up to max_cases
    cases_to_analyze = (
        classification["hg_fail_awm_pass"][:args.max_cases] +
        classification["awm_fail_hg_pass"][:args.max_cases] +
        classification["both_fail"][:args.max_cases]
    )

    log.info(f"LLM analyzing {len(cases_to_analyze)} cases...")
    analyses = []
    for i, entry in enumerate(cases_to_analyze):
        log.info(f"  [{i+1}/{len(cases_to_analyze)}] {entry['convo_id']} "
                 f"action@{entry['action_turn']} "
                 f"GT={entry['gt_action']} HG={entry['hg_action']} AWM={entry['awm_action']}")
        result = analyze_case(entry, hg_skill, awm_wf, ref_md)
        analyses.append(result)

    # ── 4. Summary ───────────────────────────────────────────
    log.info("Generating summary report...")
    report = generate_summary(classification, analyses, args.subflow)

    (OUT_DIR / "error_report.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "case_analyses.md").write_text(
        "\n\n---\n\n".join(analyses), encoding="utf-8")

    stats = {
        "subflow": args.subflow,
        "total_action_turns": total,
        "ast_joint_hg": round(hg_joint, 4),
        "ast_joint_awm": round(awm_joint, 4),
        "classification": n,
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"DONE. Output: {OUT_DIR}")
    print(f"  error_report.md   — summary report")
    print(f"  case_analyses.md  — per-case LLM analysis")
    print(f"  stats.json        — aggregate AST + classification")
    for cat in classification:
        print(f"  {cat}.json  — {len(classification[cat])} cases")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
