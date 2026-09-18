#!/usr/bin/env python3
"""Diagnose action gains accompanied by slot-value regression.

Read-only: aligns saved action-turn predictions with the current ABCD test
split. Optionally compares a baseline run/prediction file turn by turn.

Examples:
  python scripts/analyze_online_slot_regression.py outputs/RUN --subflow account_access
  python scripts/analyze_online_slot_regression.py outputs/NEW --baseline outputs/OLD
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_path(value: Path) -> Path:
    if value.is_file():
        return value
    for name in ("online_refined_predictions.json", "mined_predictions.json"):
        candidate = value / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No prediction JSON found under {value}")


def infer_subflow(run: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    state_path = run / "skill_dag_state.json" if run.is_dir() else run.parent / "skill_dag_state.json"
    if state_path.is_file():
        state = read_json(state_path)
        for key in ("subflow", "skill_id", "name"):
            if state.get(key):
                return str(state[key])
    manifest = run / "online_refine_manifest.txt" if run.is_dir() else run.parent / "online_refine_manifest.txt"
    if manifest.is_file():
        match = re.search(r"(?m)^subflow=(.+)$", manifest.read_text(encoding="utf-8"))
        if match:
            return match.group(1).strip()
    raise ValueError("Cannot infer subflow; pass --subflow")


def normal(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def slot_error_type(gold: list[str], predicted: list[str]) -> str:
    if gold == predicted:
        return "correct"
    if not predicted and gold:
        return "missing_all"
    if predicted and not gold:
        return "spurious_slots"
    if len(predicted) < len(gold):
        return "missing_values"
    if len(predicted) > len(gold):
        return "extra_values"
    if sorted(predicted) == sorted(gold):
        return "wrong_order"
    if [normal(v) for v in predicted] == [normal(v) for v in gold]:
        return "normalization_only"
    placeholders = {"n/a", "na", "none", "unknown", "null", "<value>", "value"}
    if any(str(value).strip().lower() in placeholders for value in predicted):
        return "placeholder_value"
    return "wrong_value"


def lookup_diagnostics(row: dict[str, Any], gold_action: str) -> dict[str, bool]:
    card = row.get("action_card_lookup") or {}
    selected = [str(value) for value in card.get("selected_actions", []) or []]
    selection = row.get("action_selection") or {}
    reference = row.get("reference_lookup") or {}
    return {
        "two_stage": bool(selection.get("two_stage_applied")),
        "card_executed": bool(card.get("executed")),
        "any_card_selected": bool(selected),
        "gold_card_selected": gold_action in selected,
        "reference_retrieved": bool(reference.get("observation") or reference.get("selected_sections")),
    }


def align(predictions: list[dict[str, Any]], conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_turn = {
        (str(row.get("convo_id", "")), int(row.get("turn_index", -1))): row
        for row in predictions if row.get("target_type", "action") == "action"
    }
    aligned = []
    for conversation in conversations:
        cid = str(conversation.get("convo_id", ""))
        for turn_index, turn in enumerate(conversation.get("delexed") or []):
            targets = turn.get("targets") or []
            if len(targets) < 3 or targets[1] != "take_action" or not targets[2]:
                continue
            row = by_turn.get((cid, turn_index), {})
            gold_action = str(targets[2])
            gold_slots = [str(value) for value in (targets[3] if len(targets) > 3 and isinstance(targets[3], list) else [])]
            pred_action = str(row.get("predicted_action", ""))
            pred_slots = [str(value) for value in (row.get("predicted_slots") or [])]
            aligned.append({
                "id": f"{cid}:{turn_index}", "convo_id": cid, "turn_index": turn_index,
                "gold_action": gold_action, "gold_slots": gold_slots,
                "predicted_action": pred_action, "predicted_slots": pred_slots,
                "action_ok": pred_action == gold_action, "slot_ok": pred_slots == gold_slots,
                "joint_ok": pred_action == gold_action and pred_slots == gold_slots,
                "slot_error_type": slot_error_type(gold_slots, pred_slots),
                "context": str(row.get("context", ""))[-1200:],
                "lookup": lookup_diagnostics(row, gold_action),
            })
    return aligned


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    action_ok = sum(row["action_ok"] for row in rows)
    slot_ok = sum(row["slot_ok"] for row in rows)
    joint_ok = sum(row["joint_ok"] for row in rows)
    action_correct_rows = [row for row in rows if row["action_ok"]]
    by_action: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        bucket = by_action[row["gold_action"]]
        bucket["total"] += 1; bucket["action_ok"] += int(row["action_ok"])
        bucket["slot_ok"] += int(row["slot_ok"]); bucket["joint_ok"] += int(row["joint_ok"])
        bucket[f"slot_error:{row['slot_error_type']}"] += 1
    lookup = Counter()
    for row in action_correct_rows:
        for key, value in row["lookup"].items():
            lookup[key] += int(value)
    return {
        "counts": {"turns": total, "action_correct": action_ok, "slot_correct": slot_ok,
                   "joint_correct": joint_ok, "action_correct_slot_wrong": sum(
                       row["action_ok"] and not row["slot_ok"] for row in rows)},
        "metrics": {"action_accuracy": action_ok / total if total else 0.0,
                    "slot_accuracy": slot_ok / total if total else 0.0,
                    "joint_accuracy": joint_ok / total if total else 0.0,
                    "slot_accuracy_given_action": sum(row["slot_ok"] for row in action_correct_rows) /
                                                  len(action_correct_rows) if action_correct_rows else 0.0},
        "slot_errors": dict(Counter(row["slot_error_type"] for row in rows if not row["slot_ok"])),
        "lookup_among_action_correct": dict(lookup),
        "per_action": {action: dict(counts) for action, counts in sorted(by_action.items())},
    }


def compare(current: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    old = {row["id"]: row for row in baseline}
    transitions = Counter(); examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in current:
        previous = old.get(row["id"])
        if previous is None:
            continue
        action_change = f"action_{'ok' if previous['action_ok'] else 'wrong'}_to_{'ok' if row['action_ok'] else 'wrong'}"
        slot_change = f"slot_{'ok' if previous['slot_ok'] else 'wrong'}_to_{'ok' if row['slot_ok'] else 'wrong'}"
        key = f"{action_change}__{slot_change}"
        transitions[key] += 1
        if len(examples[key]) < 20:
            examples[key].append({
                "id": row["id"], "gold_action": row["gold_action"], "gold_slots": row["gold_slots"],
                "baseline_action": previous["predicted_action"], "baseline_slots": previous["predicted_slots"],
                "current_action": row["predicted_action"], "current_slots": row["predicted_slots"],
                "current_slot_error_type": row["slot_error_type"], "lookup": row["lookup"],
            })
    return {"paired_turns": sum(transitions.values()), "transition_counts": dict(transitions),
            "examples": dict(examples)}


def artifact_signals(run: Path) -> dict[str, Any]:
    root = run if run.is_dir() else run.parent
    resources = Counter(); text_ops = Counter(); root_causes = Counter()
    reflection_files = sorted((root / "group_reflections").glob("reflection_*.json"))
    for path in reflection_files:
        payload = read_json(path)
        for update in payload.get("accepted_updates", payload.get("updates", [])) or []:
            if isinstance(update, dict):
                resources[str(update.get("resource", "unknown"))] += 1
        for operation in payload.get("applied_skill_operations", payload.get("skill_operations", [])) or []:
            if isinstance(operation, dict):
                status = "error" if operation.get("error") else (
                    "skipped" if operation.get("skipped") else "applied")
                text_ops[status] += 1
        for cause in payload.get("merged_root_causes", []) or []:
            if isinstance(cause, dict):
                root_causes[str(cause.get("category", "unspecified"))] += 1
    files = {}
    for name in ("base_skill.md", "skill.md", "action_rules.md", "slot_policies.md",
                 "online_action_rules.md", "online_slot_policies.md"):
        path = root / name
        files[name] = {"exists": path.is_file(), "chars": len(path.read_text(encoding="utf-8")) if path.is_file() else 0}
    return {
        "reflection_files": len(reflection_files),
        "semantic_updates_by_resource": dict(resources),
        "textual_skill_operations": dict(text_ops),
        "reported_root_causes": dict(root_causes),
        "files": files,
        "text_edits_without_action_card_updates": bool(
            text_ops.get("applied", 0) > 0
            and resources.get("action_rule", 0) + resources.get("slot_policy", 0) == 0
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    cur = report["current"]
    lines = ["# Online Slot Regression Analysis", "", "## Raw metric table", "",
             "| Run | Action acc | Slot acc | Joint acc | Slot acc given action | Turns |",
             "|---|---:|---:|---:|---:|---:|",
             f"| current | {cur['metrics']['action_accuracy']:.4f} | {cur['metrics']['slot_accuracy']:.4f} | {cur['metrics']['joint_accuracy']:.4f} | {cur['metrics']['slot_accuracy_given_action']:.4f} | {cur['counts']['turns']} |"]
    if report.get("baseline"):
        base = report["baseline"]
        lines.append(f"| baseline | {base['metrics']['action_accuracy']:.4f} | {base['metrics']['slot_accuracy']:.4f} | {base['metrics']['joint_accuracy']:.4f} | {base['metrics']['slot_accuracy_given_action']:.4f} | {base['counts']['turns']} |")
    lines.extend(["", "## Slot error decomposition", "", "```json",
                  json.dumps(cur["slot_errors"], ensure_ascii=False, indent=2), "```", "",
                  "## Retrieval/action-card signals among action-correct turns", "", "```json",
                  json.dumps(cur["lookup_among_action_correct"], ensure_ascii=False, indent=2), "```"])
    if report.get("comparison"):
        lines.extend(["", "## Paired transition table", "", "```json",
                      json.dumps(report["comparison"]["transition_counts"], ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "## Artifact/update signals", "", "```json",
                  json.dumps(report["artifact_signals"], ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "## Interpretation guide", "",
        "1. A large `action_wrong_to_ok__slot_wrong_to_wrong` count means action routing improved but the newly selected correct action still lacks reliable slot grounding.",
        "2. A large `action_ok_to_ok__slot_ok_to_wrong` count is a true slot regression under unchanged correct routing; inspect action cards and textual revisions first.",
        "3. Many `missing_all`/`missing_values` errors indicate unavailable or ignored value-source guidance; `extra_values` points to contract leakage; `normalization_only` points to formatting/canonicalization.",
        "4. Low `gold_card_selected` among action-correct turns indicates action-card retrieval failure. High card retrieval with low slot accuracy points to card content or grounding compliance."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--subflow")
    parser.add_argument("--splits-root", type=Path, default=Path("data/eval/abcd/splits"))
    args = parser.parse_args()
    run = args.run.resolve(); subflow = infer_subflow(run, args.subflow)
    conversations = read_json(args.splits_root / subflow / "test.json")
    current_rows = align(read_json(prediction_path(run)), conversations)
    report: dict[str, Any] = {"subflow": subflow, "current": summarize(current_rows),
                              "artifact_signals": artifact_signals(run)}
    if args.baseline:
        baseline_rows = align(read_json(prediction_path(args.baseline.resolve())), conversations)
        report["baseline"] = summarize(baseline_rows)
        report["comparison"] = compare(current_rows, baseline_rows)
    out_dir = run if run.is_dir() else run.parent
    json_path = out_dir / "slot_regression_analysis.json"
    md_path = out_dir / "slot_regression_analysis.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path),
                      "current": report["current"]["metrics"],
                      "baseline": (report.get("baseline") or {}).get("metrics"),
                      "slot_errors": report["current"]["slot_errors"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
