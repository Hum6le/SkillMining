#!/usr/bin/env python3
"""Audit likely causes of an online-refinement slot regression.

This is a read-only artifact audit. It does not call an LLM or modify the run.
It measures prompt inflation, packet duplication, resource retrieval, accepted
slot updates, and changes between the base and final skill/resource files.

Examples:
  python scripts/audit_online_refine_slot_regression.py \
      outputs/online_refine_account_access_x
  python scripts/audit_online_refine_slot_regression.py RUN \
      --baseline-dir outputs/online_refine_account_access_previous
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def sha(text: str) -> str | None:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] if text else None


def metric(payload: Any, *keys: str) -> float | None:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _mean(values: list[int | float]) -> float | None:
    return round(statistics.mean(values), 2) if values else None


def _file_summary(run: Path, name: str) -> dict[str, Any]:
    path = run / name
    text = read_text(path)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "lines": len(text.splitlines()) if text else 0,
        "chars": len(text) if text else 0,
        "sha256_prefix": sha(text),
    }


def _reflection_stats(run: Path) -> dict[str, Any]:
    files = sorted((run / "autonomous_reflection").glob("batch_*.json"))
    prompt_chars: list[int] = []
    planner_chars: list[int] = []
    accepted = Counter()
    rejected = Counter()
    lookup_resources = Counter()
    retrieved_resources = Counter()
    decisions = Counter()
    operation_errors = Counter()
    rows = []
    for path in files:
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        prompt_chars.append(int(payload.get("prompt_chars", 0) or 0))
        planner_chars.append(int(payload.get("planner_prompt_chars", 0) or 0))
        decisions[str(payload.get("model_decision", "missing"))] += 1
        for item in payload.get("accepted", []) or []:
            if isinstance(item, dict):
                accepted[str(item.get("resource", "unknown"))] += 1
        for item in payload.get("rejected", []) or []:
            if isinstance(item, dict):
                rejected[str(item.get("reason", "unknown"))] += 1
        for item in payload.get("lookups", []) or []:
            if isinstance(item, dict):
                lookup_resources[str(item.get("resource", "unknown"))] += 1
        for item in payload.get("retrieved_resources", []) or []:
            if isinstance(item, dict):
                retrieved_resources[str(item.get("resource", "unknown"))] += 1
        for item in payload.get("skill_operations", []) or []:
            if isinstance(item, dict) and item.get("error"):
                operation_errors[str(item.get("error"))] += 1
        rows.append({
            "batch": path.stem,
            "prompt_chars": int(payload.get("prompt_chars", 0) or 0),
            "planner_prompt_chars": int(payload.get("planner_prompt_chars", 0) or 0),
            "accepted": len(payload.get("accepted", []) or []),
            "accepted_slot_policy": sum(
                1 for item in payload.get("accepted", []) or []
                if isinstance(item, dict) and item.get("resource") == "slot_policy"
            ),
            "skill_operations": len(payload.get("skill_operations", []) or []),
            "skill_operation_errors": sum(
                1 for item in payload.get("skill_operations", []) or []
                if isinstance(item, dict) and item.get("error")
            ),
            "lookups": len(payload.get("lookups", []) or []),
            "retrieved_sections": len(payload.get("retrieved_resources", []) or []),
            "packet_examples": (payload.get("evidence_packet_counts") or {}).get("examples", 0),
        })
    return {
        "reflection_files": len(files),
        "prompt_chars": {
            "min": min(prompt_chars) if prompt_chars else None,
            "mean": _mean(prompt_chars),
            "max": max(prompt_chars) if prompt_chars else None,
            "over_30000": sum(value > 30000 for value in prompt_chars),
            "over_50000": sum(value > 50000 for value in prompt_chars),
        },
        "planner_prompt_chars": {"mean": _mean(planner_chars), "max": max(planner_chars) if planner_chars else None},
        "accepted_by_resource": dict(accepted),
        "rejected_by_reason": dict(rejected),
        "lookup_resources": dict(lookup_resources),
        "retrieved_resources": dict(retrieved_resources),
        "model_decisions": dict(decisions),
        "skill_operation_errors": dict(operation_errors),
        "batches": rows,
    }


def _packet_stats(run: Path) -> dict[str, Any]:
    files = sorted((run / "online_evidence").glob("batch_*.json"))
    bucket_counts = Counter()
    error_types = Counter()
    total_records = 0
    unique_records: set[tuple[str, int, str]] = set()
    transition_edges = set()
    actions = set()
    for path in files:
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        packets = payload.get("packets", {})
        for group_name in ("transition", "action_card"):
            group = packets.get(group_name, {})
            if not isinstance(group, dict):
                continue
            for key, buckets in group.items():
                if group_name == "transition":
                    transition_edges.add(str(key))
                else:
                    actions.add(str(key))
                if not isinstance(buckets, dict):
                    continue
                for bucket, values in buckets.items():
                    if not isinstance(values, list):
                        continue
                    bucket_counts[f"{group_name}.{bucket}"] += len(values)
                    for item in values:
                        if not isinstance(item, dict):
                            continue
                        total_records += 1
                        identity = (
                            str(item.get("conversation_id", "?")),
                            int(item.get("target_turn", -1) or -1),
                            str(item.get("gold_action", "")),
                        )
                        unique_records.add(identity)
                        for error in item.get("error_types", []) or []:
                            error_types[str(error)] += 1
    duplicated = total_records - len(unique_records)
    return {
        "packet_files": len(files),
        "transition_edges": len(transition_edges),
        "actions": len(actions),
        "bucket_counts": dict(bucket_counts),
        "error_types": dict(error_types),
        "packet_records": total_records,
        "unique_turn_records": len(unique_records),
        "duplicate_records": duplicated,
        "duplicate_rate": round(duplicated / total_records, 4) if total_records else None,
    }


def _pool_stats(run: Path) -> dict[str, Any]:
    payload = read_json(run / "skill_dag_state.json") or {}
    pool = payload.get("evidence_pool", {}) if isinstance(payload, dict) else {}
    result = {}
    for group in ("transition", "action_card"):
        entries = pool.get(group, {}) if isinstance(pool, dict) else {}
        result[group] = {
            "keys": len(entries) if isinstance(entries, dict) else 0,
            "records": sum(
                len(values)
                for record in (entries.values() if isinstance(entries, dict) else [])
                if isinstance(record, dict)
                for values in record.values()
                if isinstance(values, list)
            ),
        }
    result["unresolved_records"] = len(pool.get("unresolved", []) or []) if isinstance(pool, dict) else 0
    return result


def _metrics(run: Path) -> dict[str, Any]:
    payload = read_json(run / "online_refine_result.json") or {}
    ast = payload.get("ast_cds", {}) if isinstance(payload, dict) else {}
    return {
        "ast_joint": metric(ast, "ast_joint"),
        "ast_action_name": metric(ast, "ast_action_name"),
        "ast_slot_value": metric(ast, "ast_slot_value"),
        "ast_slot_value_given_action": metric(ast, "ast_slot_value_given_action"),
        "result_file": str(run / "online_refine_result.json"),
        "exists": (run / "online_refine_result.json").is_file(),
    }


def audit(run: Path, baseline: Path | None = None) -> dict[str, Any]:
    result = {
        "run_dir": str(run.resolve()),
        "metrics": _metrics(run),
        "baseline_metrics": _metrics(baseline) if baseline else None,
        "reflection": _reflection_stats(run),
        "packets": _packet_stats(run),
        "evidence_pool": _pool_stats(run),
        "files": {
            name: _file_summary(run, name)
            for name in (
                "base_skill.md", "skill.md", "working_skill.md",
                "base_reference.md", "reference.md", "slot_policies.md",
                "online_slot_policies.md", "action_rules.md",
            )
        },
    }
    files = result["files"]
    result["signals"] = {
        "prompt_inflation": bool(
            result["reflection"]["prompt_chars"]["max"]
            and result["reflection"]["prompt_chars"]["max"] > 50000
        ),
        "slot_policy_updates": result["reflection"]["accepted_by_resource"].get("slot_policy", 0),
        "reference_was_retrieved": bool(result["reflection"]["retrieved_resources"].get("reference", 0)),
        "action_rules_were_retrieved": bool(result["reflection"]["retrieved_resources"].get("action_rules", 0)),
        "slot_policies_were_retrieved": bool(result["reflection"]["retrieved_resources"].get("slot_policies", 0)),
        "packet_duplication_over_20pct": bool(
            result["packets"]["duplicate_rate"] is not None
            and result["packets"]["duplicate_rate"] > 0.20
        ),
        "final_skill_changed": files["base_skill.md"]["sha256_prefix"] != files["skill.md"]["sha256_prefix"],
        "online_slot_resource_nonempty": files["online_slot_policies.md"]["lines"] > 2,
    }
    return result


def markdown(report: dict[str, Any]) -> str:
    reflection = report["reflection"]
    packets = report["packets"]
    signals = report["signals"]
    lines = [
        "# Online Refine Slot Regression Audit",
        "",
        f"- Run: `{report['run_dir']}`",
        f"- Metrics: `{json.dumps(report['metrics'], ensure_ascii=False)}`",
        "",
        "## Main signals",
        "",
        f"- Prompt inflation: `{signals['prompt_inflation']}`; max chars={reflection['prompt_chars']['max']}",
        f"- Accepted slot-policy updates: `{signals['slot_policy_updates']}`",
        f"- Reference retrieved by optimizer: `{signals['reference_was_retrieved']}`",
        f"- Action rules retrieved: `{signals['action_rules_were_retrieved']}`",
        f"- Slot policies retrieved: `{signals['slot_policies_were_retrieved']}`",
        f"- Packet duplicate rate: `{packets['duplicate_rate']}`",
        f"- Final skill changed from base: `{signals['final_skill_changed']}`",
        f"- Online slot resource non-empty: `{signals['online_slot_resource_nonempty']}`",
        "",
        "## Reflection",
        "",
        f"- Reflection files: `{reflection['reflection_files']}`",
        f"- Accepted by resource: `{json.dumps(reflection['accepted_by_resource'], ensure_ascii=False)}`",
        f"- Retrieved resources: `{json.dumps(reflection['retrieved_resources'], ensure_ascii=False)}`",
        f"- Model decisions: `{json.dumps(reflection['model_decisions'], ensure_ascii=False)}`",
        f"- Skill operation errors: `{json.dumps(reflection['skill_operation_errors'], ensure_ascii=False)}`",
        "",
        "## Evidence packets",
        "",
        f"- Packet files: `{packets['packet_files']}`",
        f"- Bucket counts: `{json.dumps(packets['bucket_counts'], ensure_ascii=False)}`",
        f"- Error types: `{json.dumps(packets['error_types'], ensure_ascii=False)}`",
        f"- Records={packets['packet_records']}, unique turns={packets['unique_turn_records']}, duplicates={packets['duplicate_records']}",
        "",
        "## Interpretation",
        "",
        "Use this report together with autonomous_reflection/batch_*.json. A large prompt, accepted slot-policy updates, and a changed online slot resource point to optimizer-induced slot regression. A zero reference retrieval count points to a progressive-disclosure recall issue, but does not by itself explain a slot-only drop.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="JSON output; defaults to RUN/slot_regression_audit.json")
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not run.is_dir():
        parser.error(f"run directory does not exist: {run}")
    report = audit(run, args.baseline_dir.resolve() if args.baseline_dir else None)
    output = args.output.resolve() if args.output else run / "slot_regression_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = args.markdown_output.resolve() if args.markdown_output else run / "slot_regression_audit.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({
        "run_dir": str(run),
        "json": str(output),
        "markdown": str(md),
        "metrics": report["metrics"],
        "signals": report["signals"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
