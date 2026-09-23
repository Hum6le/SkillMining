#!/usr/bin/env python3
"""Audit how offline ABCD mining artifacts flow into online refinement.

The audit is read-only. It can compare paired online runs and optionally ask an
LLM to propose explanations from the compact artifact report.

Example:
  python scripts/analyze_offline_online_contribution.py \
    --run-dir outputs/online_with_offline \
    --test-data data/eval/abcd/splits/recover_password/test.json \
    --comparison-dir outputs/online_control \
    --llm-analysis --model qwen3.6-flash
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


RESOURCE_FILES = (
    "base_skill.md", "base_reference.md", "action_rules.md", "slot_policies.md",
    "skill.md", "working_skill.md", "online_reference.md",
    "online_action_rules.md", "online_slot_policies.md",
)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _metric(payload: Any, *keys: str) -> float | None:
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metrics(run_dir: Path) -> dict[str, float | None]:
    result = read_json(run_dir / "online_refine_result.json") or {}
    ast = result.get("ast_cds", {}) if isinstance(result, dict) else {}
    return {
        "ast_joint": _metric(ast, "ast_joint"),
        "ast_action_name": _metric(ast, "ast_action_name"),
        "ast_slot_value": _metric(ast, "ast_slot_value"),
        "ast_slot_value_given_action": _metric(ast, "ast_slot_value_given_action"),
    }


def _resource_inventory(run_dir: Path) -> dict[str, Any]:
    output = {}
    for name in RESOURCE_FILES:
        path = run_dir / name
        text = read_text(path)
        output[name] = {
            "exists": path.is_file(), "chars": len(text),
            "lines": len(text.splitlines()),
            "nonempty": bool(text.strip()),
            "sha256_prefix": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] if text else None,
        }
    skill_dir = run_dir / "trace2skill_hybrid_skill"
    refs_dir = skill_dir / "references"
    output["trace2skill_hybrid_references"] = {
        "exists": refs_dir.is_dir(),
        "files": sorted(p.name for p in refs_dir.iterdir() if p.is_file()) if refs_dir.is_dir() else [],
    }
    return output


def _online_update_stats(run_dir: Path) -> dict[str, Any]:
    files = sorted((run_dir / "autonomous_reflection").glob("batch_*.json"))
    decisions = Counter()
    accepted = Counter()
    rejected = Counter()
    op_errors = Counter()
    accepted_examples = []
    for path in files:
        item = read_json(path)
        if not isinstance(item, dict):
            continue
        decisions[str(item.get("model_decision", "missing"))] += 1
        for row in item.get("accepted", []) or []:
            if isinstance(row, dict):
                resource = str(row.get("resource", "unknown"))
                accepted[resource] += 1
                if len(accepted_examples) < 30:
                    accepted_examples.append({"batch": path.name, **row})
        for row in item.get("rejected", []) or []:
            if isinstance(row, dict):
                rejected[str(row.get("reason", "unknown"))] += 1
        for row in item.get("skill_operations", []) or []:
            if isinstance(row, dict) and row.get("error"):
                op_errors[str(row["error"])] += 1
    return {
        "reflection_batches": len(files), "decisions": dict(decisions),
        "accepted_by_resource": dict(accepted), "rejected_by_reason": dict(rejected),
        "skill_operation_errors": dict(op_errors), "accepted_examples": accepted_examples,
    }


def _evidence_stats(run_dir: Path) -> dict[str, Any]:
    files = sorted((run_dir / "online_evidence").glob("batch_*.json"))
    groups = Counter()
    actions: set[str] = set()
    transitions: set[str] = set()
    records: set[tuple[str, int, str]] = set()
    error_types = Counter()
    for path in files:
        payload = read_json(path)
        packets = payload.get("packets", {}) if isinstance(payload, dict) else {}
        for group in ("action_card", "transition"):
            entries = packets.get(group, {}) if isinstance(packets, dict) else {}
            if not isinstance(entries, dict):
                continue
            for key, buckets in entries.items():
                (actions if group == "action_card" else transitions).add(str(key))
                if not isinstance(buckets, dict):
                    continue
                for bucket, values in buckets.items():
                    if not isinstance(values, list):
                        continue
                    groups[f"{group}.{bucket}"] += len(values)
                    for value in values:
                        if not isinstance(value, dict):
                            continue
                        turn_value = value.get("target_turn", -1)
                        try:
                            turn_value = int(turn_value) if turn_value is not None else -1
                        except (TypeError, ValueError):
                            turn_value = -1
                        records.add((
                            str(value.get("conversation_id", "")),
                            turn_value,
                            str(value.get("gold_action", "")),
                        ))
                        for error in value.get("error_types", []) or []:
                            error_types[str(error)] += 1
    return {
        "batch_files": len(files), "packet_bucket_counts": dict(groups),
        "actions_with_packets": sorted(actions), "transition_edges_with_packets": len(transitions),
        "unique_evidence_turns": len(records), "error_types": dict(error_types),
    }


def _load_prediction_rows(run_dir: Path) -> list[dict[str, Any]]:
    for name in ("online_refined_predictions.json", "evolved_test_turns.json", "online_refine_predictions.json"):
        payload = read_json(run_dir / name)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
    return []


def _card_stats(run_dir: Path, test_data: Path | None) -> dict[str, Any]:
    rows = _load_prediction_rows(run_dir)
    action_rows = [r for r in rows if r.get("target_type") == "action"]
    executed = 0
    selected_any = 0
    by_selected = Counter()
    pred_stats = Counter()
    per_turn: dict[tuple[str, int], dict[str, Any]] = {}
    for row in action_rows:
        lookup = row.get("action_card_lookup") or {}
        selected = [str(x) for x in lookup.get("selected_actions", []) or []]
        executed += int(bool(lookup.get("executed")))
        selected_any += int(bool(selected))
        by_selected.update(selected)
        pred_stats["rows"] += 1
        pred_stats["predicted_action_present"] += int(bool(row.get("predicted_action")))
        try:
            key = (str(row.get("convo_id", "")), int(row.get("turn_index", -1)))
            per_turn[key] = {
                "action": str(row.get("predicted_action") or ""),
                "slots": row.get("predicted_slots") or [],
                "selected_cards": selected,
                "card_executed": bool(lookup.get("executed")),
                "workflow_chars": row.get("workflow_injected_chars"),
                "reference_selected": (row.get("reference_lookup") or {}).get("selected_sections", []),
            }
        except (TypeError, ValueError):
            continue
    gold_turns = {}
    if test_data and test_data.is_file():
        from eval_tod.abcd.data import extract_ground_truth
        conversations = read_json(test_data)
        for conv in conversations if isinstance(conversations, list) else []:
            if not isinstance(conv, dict):
                continue
            for truth in extract_ground_truth(conv):
                if truth.turn_type == "action":
                    gold_turns[(str(conv.get("convo_id", "")), int(truth.turn_index))] = {
                        "action": str(truth.action_name or ""), "slots": list(truth.slot_values or []),
                    }
    card_gold = Counter()
    correctness = Counter()
    for key, prediction in per_turn.items():
        gold = gold_turns.get(key)
        if not gold:
            continue
        card_gold["scored_turns"] += 1
        card_gold["gold_action_card_selected"] += int(gold["action"] in prediction["selected_cards"])
        action_ok = prediction["action"] == gold["action"]
        slot_ok = prediction["slots"] == gold["slots"]
        correctness["action_correct"] += int(action_ok)
        correctness["slot_correct"] += int(slot_ok)
        correctness["joint_correct"] += int(action_ok and slot_ok)
        correctness["joint_wrong_with_gold_card"] += int(not (action_ok and slot_ok) and gold["action"] in prediction["selected_cards"])
        correctness["joint_wrong_without_gold_card"] += int(not (action_ok and slot_ok) and gold["action"] not in prediction["selected_cards"])
    return {
        "prediction_file_rows": len(rows), "action_target_rows": len(action_rows),
        "card_lookup_executed": executed, "any_card_selected": selected_any,
        "selected_actions_top20": by_selected.most_common(20),
        "gold_aligned": dict(card_gold), "correctness_on_aligned_turns": dict(correctness),
        "per_turn": per_turn, "gold_turns": gold_turns,
        "prediction_file": next((name for name in ("online_refined_predictions.json", "evolved_test_turns.json", "online_refine_predictions.json") if (run_dir / name).is_file()), None),
    }


def _paired_comparison(primary: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    p, o = primary["card_runtime"], other["card_runtime"]
    shared = sorted(set(p["per_turn"]) & set(o["per_turn"]))
    p_gold, o_gold = p["gold_turns"], o["gold_turns"]
    bins = Counter()
    prediction_changes = Counter()
    examples = []
    for key in shared:
        a, b, gold = p["per_turn"][key], o["per_turn"][key], p_gold.get(key) or o_gold.get(key)
        prediction_changes["turns_with_any_prediction_change"] += int(
            a["action"] != b["action"] or a["slots"] != b["slots"]
        )
        prediction_changes["action_changed"] += int(a["action"] != b["action"])
        prediction_changes["slots_changed"] += int(a["slots"] != b["slots"])
        prediction_changes["selected_cards_changed"] += int(a["selected_cards"] != b["selected_cards"])
        if not gold:
            continue
        a_ok = a["action"] == gold["action"] and a["slots"] == gold["slots"]
        b_ok = b["action"] == gold["action"] and b["slots"] == gold["slots"]
        label = "both_correct" if a_ok and b_ok else "comparison_only_correct" if a_ok else "primary_only_correct" if b_ok else "both_wrong"
        bins[label] += 1
        if label in {"comparison_only_correct", "primary_only_correct"} and len(examples) < 30:
            examples.append({
                "conversation_id": key[0], "turn_index": key[1], "bucket": label,
                "gold": gold, "primary": a, "comparison": b,
            })
    return {
        "shared_turns": len(shared), "prediction_changes": dict(prediction_changes),
        "paired_correctness": dict(bins), "discordant_examples": examples,
    }


def analyze(run_dir: Path, *, comparison_dir: Path | None = None, test_data: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    metrics = _metrics(run_dir)
    result: dict[str, Any] = {
        "run_dir": str(run_dir), "metrics": metrics,
        "resources": _resource_inventory(run_dir),
        "online_updates": _online_update_stats(run_dir),
        "online_evidence": _evidence_stats(run_dir),
        "card_runtime": _card_stats(run_dir, test_data),
        "comparison": None,
        "causal_warning": "One run cannot establish the offline contribution. A valid paired comparison needs the same held-out turns, model/settings, and online procedure, differing only in the offline artifact condition.",
    }
    if comparison_dir:
        comparison_dir = comparison_dir.resolve()
        comparison_metrics = _metrics(comparison_dir)
        result["comparison"] = {
            "comparison_dir": str(comparison_dir), "metrics": comparison_metrics,
            "resources": _resource_inventory(comparison_dir),
            "metric_delta_primary_minus_comparison": {
                key: round(value - comparison_metrics[key], 6)
                if value is not None and comparison_metrics[key] is not None else None
                for key, value in metrics.items()
            },
            "paired_turns": _paired_comparison(result, {
                "card_runtime": _card_stats(comparison_dir, test_data),
            }),
        }
    return result


def _llm_synthesis(report: dict[str, Any], model: str) -> str:
    from llm import chat
    compact = {
        key: report.get(key) for key in (
            "run_dir", "metrics", "resources", "online_updates", "online_evidence",
            "comparison", "causal_warning",
        )
    }
    card = report["card_runtime"]
    compact["card_runtime"] = {key: value for key, value in card.items() if key not in {"per_turn", "gold_turns"}}
    prompt = """Analyze why offline mining artifacts appear to add little benefit to the online ToD method. Treat the JSON as artifact evidence, not causal proof. Explicitly distinguish: artifact generation, loading, runtime exposure/card retrieval, learning/update, retention, and held-out metric effect. Identify missing evidence; do not claim causality without a matched control. Propose up to three competing explanations, each with a falsifiable experiment, then recommend the smallest next experiment. Respond in Chinese with evidence paths/values.

ARTIFACT AUDIT JSON:
""" + json.dumps(compact, ensure_ascii=False, indent=2)
    answer = chat(prompt, model=model, temperature=0.1, call_tag="offline_online_contribution_analysis")
    if not answer.strip():
        raise RuntimeError("LLM returned an empty analysis")
    return answer.strip() + "\n"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Offline-to-Online Contribution Audit: {report['run_dir']}", "",
        "| Metric | Primary run | Comparison run | Delta |", "|---|---:|---:|---:|",
    ]
    compare = report.get("comparison") or {}
    other = compare.get("metrics") or {}
    delta = compare.get("metric_delta_primary_minus_comparison") or {}
    for key, value in report["metrics"].items():
        lines.append(f"| {key} | {value} | {other.get(key)} | {delta.get(key)} |")
    resources = report["resources"]
    lines.extend(["", "## Offline Resource Supply", ""])
    for name in RESOURCE_FILES:
        item = resources[name]
        lines.append(f"- `{name}`: exists={item['exists']}, nonempty={item['nonempty']}, chars={item['chars']}")
    lines.append(f"- Trace2Skill evolved references: {resources['trace2skill_hybrid_references']}")
    update = report["online_updates"]
    lines.extend(["", "## Online Learning", "",
        f"- Reflection batches: {update['reflection_batches']}; decisions: `{update['decisions']}`",
        f"- Accepted updates: `{update['accepted_by_resource']}`; rejected: `{update['rejected_by_reason']}`",
        f"- Evidence: `{report['online_evidence']}`",
        "", "## Runtime Action Cards", "",
    ])
    card = report["card_runtime"]
    lines.extend([
        f"- Prediction rows: {card['prediction_file_rows']}; action targets: {card['action_target_rows']}",
        f"- Card lookup executed: {card['card_lookup_executed']}; selected a card: {card['any_card_selected']}",
        f"- Gold-aligned card/correctness: `{card['gold_aligned']}` / `{card['correctness_on_aligned_turns']}`",
        f"- Most selected cards: `{card['selected_actions_top20']}`",
    ])
    if compare:
        lines.extend(["", "## Paired Comparison", "", f"- {compare['paired_turns']}"])
    lines.extend(["", "## Interpretation Boundary", "", report["causal_warning"]])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Online run that used offline artifacts")
    parser.add_argument("--comparison-dir", type=Path, help="Matched online ablation/control run")
    parser.add_argument("--test-data", type=Path, help="Held-out conversations for per-turn Action Card alignment")
    parser.add_argument("--llm-analysis", action="store_true", help="Call configured LLM for competing explanations and next experiment")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"run directory does not exist: {run_dir}")
    report = analyze(
        run_dir, comparison_dir=args.comparison_dir,
        test_data=args.test_data.resolve() if args.test_data else None,
    )
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "offline_online_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report)
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "audit.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    if args.llm_analysis:
        answer = _llm_synthesis(report, args.model)
        (output_dir / "llm_analysis.md").write_text(answer, encoding="utf-8")
        print("\n## LLM Analysis\n\n" + answer)
    print(f"\nSaved artifacts: {output_dir}")


if __name__ == "__main__":
    main()
