#!/usr/bin/env python3
"""Summarize a matched Trace2Skill module-ablation run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _get(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        usage = summary.get("llm_usage", {}).get("generation", {}).get("total", {})
        evolved = _get(summary, "evolved_test", "ast_cds") or {}
        seed = _get(summary, "seed_test", "ast_cds") or {}
        history = summary.get("batch_history") or []
        rows.append({
            "variant": path.parent.name,
            "evolved_ast_joint": evolved.get("ast_joint"),
            "evolved_action": evolved.get("ast_action_name"),
            "evolved_slot": evolved.get("ast_slot_value"),
            "seed_ast_joint": seed.get("ast_joint"),
            "delta_from_seed": (
                round(float(evolved["ast_joint"]) - float(seed["ast_joint"]), 6)
                if evolved.get("ast_joint") is not None and seed.get("ast_joint") is not None else None
            ),
            "generation_calls": usage.get("calls", 0),
            "generation_tokens": usage.get("total_tokens", 0),
            "failed_cases": sum(int(item.get("failed_cases", 0)) for item in history),
            "success_cases": sum(int(item.get("successful_cases", 0)) for item in history),
            "skill_changes": len(summary.get("changelog") or []),
            "batches": len(history),
        })
    if not rows:
        raise SystemExit(f"No <variant>/summary.json found below {args.root}")
    output = args.output or args.root / "module_ablation_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output.with_suffix(".json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} variants to {output}")
    for row in rows:
        print(
            f"{row['variant']:16s} AST={row['evolved_ast_joint']} "
            f"action={row['evolved_action']} slot={row['evolved_slot']} "
            f"calls={row['generation_calls']} changes={row['skill_changes']}"
        )


if __name__ == "__main__":
    main()
