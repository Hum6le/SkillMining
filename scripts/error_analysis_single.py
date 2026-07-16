#!/usr/bin/env python3
r"""Single-method ABCD error analysis with the same AST mapping as evaluation.

Example:
  python scripts/error_analysis_single.py \
    --preds outputs/subflow_eval_xxx/recover_username/mined_predictions.json \
    --react-traces outputs/subflow_eval_xxx/recover_username/mined_react_traces.json \
    --test-data data/eval/abcd/splits/recover_username/test.json \
    --skill outputs/subflow_eval_xxx/recover_username/skill.md \
    --reference outputs/subflow_eval_xxx/recover_username/reference.md \
    --method-name HG \
    --subflow recover_username

Use prediction files produced with ``predict_actions=True``. If a row only has a
raw ``prediction`` string, this script will try the local ABCD action parser as
a fallback.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from eval_tod.abcd.agent import _parse_action_response, turn_results_to_abcd_predictions
from eval_tod.abcd.data import extract_ground_truth
from eval_tod.abcd.metrics import compute_ast


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_text(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _infer_react_trace_path(preds_path: str) -> str:
    p = Path(preds_path)
    name = p.name
    if name.endswith("_predictions.json"):
        candidate = p.with_name(name.replace("_predictions.json", "_react_traces.json"))
        if candidate.exists():
            return str(candidate)
    return ""


def _build_react_lookup(react_traces: list[dict[str, Any]] | None) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in react_traces or []:
        try:
            key = (str(row.get("convo_id", "")), int(row.get("turn_index", 0)))
        except (TypeError, ValueError):
            continue
        lookup[key] = row
    return lookup


def _normalise_turn_results(turn_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(r) for r in turn_results]
    for row in rows:
        row["convo_id"] = str(row.get("convo_id", ""))
        if "predicted_action" not in row and "prediction" in row:
            action, slots, response = _parse_action_response(row.get("prediction", ""))
            row["predicted_action"] = action
            row["predicted_slots"] = slots
            if response and not row.get("response_text"):
                row["response_text"] = response
        row.setdefault("predicted_slots", [])
    rows.sort(key=lambda x: (str(x.get("convo_id", "")), int(x.get("turn_index", 0))))
    return rows


def _find_agent_turn_row(turn_results: list[dict[str, Any]], cid: str, action_turn: int) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for row in sorted(turn_results, key=lambda x: int(x.get("turn_index", 0))):
        if str(row.get("convo_id", "")) != cid:
            continue
        if int(row.get("turn_index", 999999)) < action_turn:
            best = row
        else:
            break
    return best


def _find_turn_response(turn_results: list[dict[str, Any]], cid: str, action_turn: int) -> str:
    row = _find_agent_turn_row(turn_results, cid, action_turn)
    return row.get("response_text") or row.get("prediction") or ""


def _compact_react_trace(trace: Any, max_chars: int = 3500) -> str:
    if not trace:
        return "(no react trace)"
    if isinstance(trace, str):
        text = trace
    else:
        text = json.dumps(trace, indent=2, ensure_ascii=False)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [truncated]"
    return text


def _extract_context(conv: dict[str, Any], turn_idx: int, n_before: int = 6) -> str:
    delexed = conv.get("delexed", [])
    lines: list[str] = []
    for i in range(max(0, turn_idx - n_before), min(turn_idx, len(delexed))):
        turn = delexed[i]
        speaker = turn.get("speaker", "unknown")
        label = {
            "agent": "Agent",
            "customer": "Customer",
            "action": "SystemAction",
        }.get(speaker, speaker)
        text = str(turn.get("text", "")).strip()
        if text:
            lines.append(f"[{label}] {text}")
    return "\n".join(lines)


def classify_single_method(
    turn_results: list[dict[str, Any]],
    test_convs: list[dict[str, Any]],
    react_traces: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Classify every ground-truth action turn for one prediction method."""
    rows = _normalise_turn_results(turn_results)
    react_lookup = _build_react_lookup(react_traces)
    for conv in test_convs:
        conv["convo_id"] = str(conv.get("convo_id", ""))

    predictions = turn_results_to_abcd_predictions(rows, test_convs)
    pred_by_cid: dict[str, dict[int, tuple[str, list[str]]]] = {}
    for pred in predictions:
        pred_by_cid[str(pred.conversation_id)] = {
            t.turn_index: (t.predicted_action or "", t.predicted_slots or [])
            for t in pred.turns
        }

    categories: dict[str, list[dict[str, Any]]] = {
        "joint_pass": [],
        "joint_fail": [],
        "action_ok_slot_wrong": [],
        "action_wrong_slot_ok": [],
        "both_wrong": [],
        "missing_prediction": [],
    }
    by_action: dict[str, Counter[str]] = defaultdict(Counter)

    total_action_turns = 0
    action_correct = 0
    slot_correct = 0
    joint_correct = 0
    per_dialogue = []

    for conv, pred in zip(test_convs, predictions):
        cid = str(conv.get("convo_id", ""))
        truths = extract_ground_truth(conv)
        ast = compute_ast(truths, pred, conversation_id=cid)
        per_dialogue.append({
            "convo_id": cid,
            "num_action_turns": ast.num_action_turns,
            "action_name_accuracy": round(ast.action_name_accuracy, 4),
            "slot_value_accuracy": round(ast.slot_value_accuracy, 4),
            "joint_accuracy": round(ast.joint_accuracy, 4),
        })

        pred_turns = pred_by_cid.get(cid, {})
        for gt in truths:
            if gt.turn_type != "action" or not gt.action_name:
                continue
            total_action_turns += 1
            pred_action, pred_slots = pred_turns.get(gt.turn_index, ("", []))
            gt_slots = gt.slot_values or []
            action_ok = pred_action == gt.action_name
            slot_ok = pred_slots == gt_slots
            joint_ok = action_ok and slot_ok
            action_correct += int(action_ok)
            slot_correct += int(slot_ok)
            joint_correct += int(joint_ok)
            agent_row = _find_agent_turn_row(rows, cid, gt.turn_index)
            agent_turn = agent_row.get("turn_index")
            trace_row = {}
            if agent_turn is not None:
                try:
                    trace_row = react_lookup.get((cid, int(agent_turn)), {})
                except (TypeError, ValueError):
                    trace_row = {}
            react_trace = agent_row.get("react_trace") or trace_row.get("react_trace") or []

            entry = {
                "convo_id": cid,
                "action_turn": gt.turn_index,
                "agent_turn": agent_turn,
                "gt_action": gt.action_name,
                "pred_action": pred_action,
                "gt_slots": gt_slots,
                "pred_slots": pred_slots,
                "action_ok": action_ok,
                "slot_ok": slot_ok,
                "joint_ok": joint_ok,
                "reference": str(gt.text or "")[:500],
                "prediction": _find_turn_response(rows, cid, gt.turn_index)[:500],
                "react_trace": react_trace,
                "reference_lookup": agent_row.get("reference_lookup", {}),
                "context": _extract_context(conv, gt.turn_index),
            }

            by_action[gt.action_name]["total"] += 1
            by_action[gt.action_name]["action_correct"] += int(action_ok)
            by_action[gt.action_name]["slot_correct"] += int(slot_ok)
            by_action[gt.action_name]["joint_correct"] += int(joint_ok)

            if joint_ok:
                categories["joint_pass"].append(entry)
            else:
                categories["joint_fail"].append(entry)
                if not pred_action and not pred_slots:
                    categories["missing_prediction"].append(entry)
                if action_ok and not slot_ok:
                    categories["action_ok_slot_wrong"].append(entry)
                elif not action_ok and slot_ok:
                    categories["action_wrong_slot_ok"].append(entry)
                else:
                    categories["both_wrong"].append(entry)

    action_breakdown = []
    for action, counts in sorted(by_action.items(), key=lambda x: (-x[1]["total"], x[0])):
        total = counts["total"]
        action_breakdown.append({
            "action": action,
            "total": total,
            "action_accuracy": round(counts["action_correct"] / max(total, 1), 4),
            "slot_accuracy": round(counts["slot_correct"] / max(total, 1), 4),
            "joint_accuracy": round(counts["joint_correct"] / max(total, 1), 4),
            "joint_errors": total - counts["joint_correct"],
        })

    stats = {
        "num_conversations": len(test_convs),
        "total_action_turns": total_action_turns,
        "ast_joint": round(joint_correct / max(total_action_turns, 1), 4),
        "ast_action_name": round(action_correct / max(total_action_turns, 1), 4),
        "ast_slot_value": round(slot_correct / max(total_action_turns, 1), 4),
        "counts": {k: len(v) for k, v in categories.items()},
        "react_trace_available_failures": sum(
            1 for row in categories["joint_fail"] if row.get("react_trace")
        ),
        "action_breakdown": action_breakdown,
        "per_dialogue": per_dialogue,
    }
    return categories, stats


def analyze_case(
    entry: dict[str, Any],
    method_name: str,
    skill_text: str,
    reference_text: str,
    model: str,
) -> str:
    from llm import chat

    prompt = f"""Analyze one ABCD action-turn error for a single method.

Method: {method_name}

Ground truth action: `{entry['gt_action']}`
Predicted action: `{entry['pred_action']}`
Action correct: {entry['action_ok']}

Ground truth slots: {entry['gt_slots']}
Predicted slots: {entry['pred_slots']}
Slots correct: {entry['slot_ok']}

Context:
```
{entry['context'][:1200]}
```

Reference response:
{entry['reference'][:500]}

Method response:
{entry['prediction'][:500]}

Skill / workflow excerpt:
{skill_text[:2500] if skill_text else '(none)'}

Reference notes excerpt:
{reference_text[:1500] if reference_text else '(none)'}

ReAct trace for the agent turn that produced this action prediction:
```json
{_compact_react_trace(entry.get('react_trace'))}
```

Reference lookup metadata on that agent turn:
```json
{_compact_react_trace(entry.get('reference_lookup'), max_chars=1800)}
```

Respond in Chinese. Use this markdown format:

### {entry['convo_id']} action@{entry['action_turn']}
**GT**: `...` / slots `...`
**Prediction**: `...` / slots `...`

**错误类型**: action wrong / slot wrong / both wrong / missing prediction

**根因**: one or two sentences.

**证据**: cite the specific dialogue clue or skill gap.

**ReAct诊断**: explain whether the issue came from reference retrieval/query, selected snippets, reasoning, action parsing, or missing skill guidance.

**改进建议**: concrete skill/workflow/reference edit.
"""
    try:
        return chat(prompt, model=model, temperature=0.0, max_tokens=1200).strip()
    except Exception as exc:
        return f"### {entry['convo_id']} action@{entry['action_turn']}\n\n*LLM error: {exc}*"


def generate_report(
    method_name: str,
    subflow: str,
    stats: dict[str, Any],
    analyses: list[str],
    model: str,
) -> str:
    summaries = "\n\n---\n\n".join(analyses[:40])
    worst_actions = stats["action_breakdown"][:15]

    prompt = f"""Synthesize a single-method ABCD error analysis report.

Method: {method_name}
Subflow: {subflow}

Metrics:
- Total action turns: {stats['total_action_turns']}
- AST joint: {stats['ast_joint']}
- Action-name accuracy: {stats['ast_action_name']}
- Slot-value accuracy: {stats['ast_slot_value']}
- Failed turns with ReAct trace: {stats.get('react_trace_available_failures', 0)}
- Counts: {json.dumps(stats['counts'], ensure_ascii=False)}

Frequent actions:
{json.dumps(worst_actions, indent=2, ensure_ascii=False)}

Case analyses:
{summaries[:14000]}

Respond in Chinese with markdown sections:
1. Overall conclusion
2. Main error types
3. Action-level patterns
4. Slot/value failure patterns
5. ReAct/retrieval failure patterns, if traces are available
6. Top 5 concrete improvements to the skill/workflow/reference
"""
    if not analyses:
        return _fallback_report(method_name, subflow, stats)
    try:
        from llm import chat
        return chat(prompt, model=model, temperature=0.0, max_tokens=2400).strip()
    except Exception as exc:
        return f"*LLM error while generating report: {exc}*\n\n" + _fallback_report(method_name, subflow, stats)


def _fallback_report(method_name: str, subflow: str, stats: dict[str, Any]) -> str:
    lines = [
        f"# Single-Method Error Analysis: {method_name}",
        "",
        f"Subflow: `{subflow}`",
        "",
        "## Metrics",
        f"- Total action turns: {stats['total_action_turns']}",
        f"- AST joint: {stats['ast_joint']:.4f}",
        f"- Action-name accuracy: {stats['ast_action_name']:.4f}",
        f"- Slot-value accuracy: {stats['ast_slot_value']:.4f}",
        f"- Failed turns with ReAct trace: {stats.get('react_trace_available_failures', 0)}",
        "",
        "## Counts",
    ]
    for key, value in stats["counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Worst Frequent Actions"])
    for row in stats["action_breakdown"][:15]:
        lines.append(
            f"- {row['action']}: total={row['total']}, "
            f"joint={row['joint_accuracy']:.4f}, errors={row['joint_errors']}"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Single-method ABCD error analysis using unified AST mapping",
    )
    parser.add_argument("--preds", required=True, help="Turn-level prediction JSON")
    parser.add_argument(
        "--react-traces",
        default="",
        help=(
            "Optional ReAct trace JSON. If omitted, the script tries to infer "
            "*_react_traces.json next to *_predictions.json."
        ),
    )
    parser.add_argument("--test-data", required=True, help="ABCD test conversations JSON")
    parser.add_argument("--skill", default="", help="Optional skill/workflow text file")
    parser.add_argument("--reference", default="", help="Optional reference markdown/text file")
    parser.add_argument("--method-name", default="method")
    parser.add_argument("--subflow", default="unknown")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--no-llm", action="store_true", help="Only compute stats and JSON cases")
    parser.add_argument("--output-dir", default="", help="Optional explicit output directory")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/error_analysis_single_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(out_dir / "analysis.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("error_analysis_single")

    log.info("Loading predictions: %s", args.preds)
    preds = _load_json(args.preds)
    react_path = args.react_traces or _infer_react_trace_path(args.preds)
    react_traces = []
    if react_path:
        log.info("Loading ReAct traces: %s", react_path)
        react_traces = _load_json(react_path)
    else:
        log.info("No ReAct trace file provided or inferred")
    log.info("Loading test data: %s", args.test_data)
    test_convs = _load_json(args.test_data)
    skill_text = _load_text(args.skill)
    reference_text = _load_text(args.reference)

    has_actions = any("predicted_action" in row for row in preds[:20])
    log.info("Pred rows: %d (has predicted_action: %s)", len(preds), has_actions)
    if not has_actions:
        log.warning("Prediction rows lack predicted_action; falling back to parsing raw prediction strings.")

    categories, stats = classify_single_method(preds, test_convs, react_traces)
    log.info(
        "AST joint=%.4f action=%.4f slot=%.4f total_action_turns=%d",
        stats["ast_joint"],
        stats["ast_action_name"],
        stats["ast_slot_value"],
        stats["total_action_turns"],
    )

    for name, cases in categories.items():
        (out_dir / f"{name}.json").write_text(
            json.dumps(cases, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Saved %s: %d cases", name, len(cases))
    (out_dir / "stats.json").write_text(
        json.dumps({
            "method_name": args.method_name,
            "subflow": args.subflow,
            **stats,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    analyses: list[str] = []
    if not args.no_llm:
        cases_to_analyze = (
            categories["action_ok_slot_wrong"][:args.max_cases]
            + categories["action_wrong_slot_ok"][:args.max_cases]
            + categories["both_wrong"][:args.max_cases]
            + categories["missing_prediction"][:args.max_cases]
        )
        cases_to_analyze = cases_to_analyze[:args.max_cases]
        log.info("LLM analyzing %d cases", len(cases_to_analyze))
        for idx, entry in enumerate(cases_to_analyze, start=1):
            log.info(
                "[%d/%d] %s action@%s GT=%s PRED=%s",
                idx,
                len(cases_to_analyze),
                entry["convo_id"],
                entry["action_turn"],
                entry["gt_action"],
                entry["pred_action"],
            )
            analyses.append(
                analyze_case(entry, args.method_name, skill_text, reference_text, args.model)
            )

    (out_dir / "case_analyses.md").write_text(
        "\n\n---\n\n".join(analyses),
        encoding="utf-8",
    )
    report = generate_report(args.method_name, args.subflow, stats, analyses, args.model)
    (out_dir / "error_report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"DONE. Output: {out_dir}")
    print(f"AST joint:  {stats['ast_joint']:.4f}")
    print(f"Action acc: {stats['ast_action_name']:.4f}")
    print(f"Slot acc:   {stats['ast_slot_value']:.4f}")
    print("Files: stats.json, error_report.md, case_analyses.md, joint_fail.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
