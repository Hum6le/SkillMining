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
    # Trace2Skill creates a timestamped run directory below the requested
    # variant directory. Pick the newest summary per variant, rather than
    # assuming <variant>/summary.json exists.
    summary_paths: dict[str, Path] = {}
    for path in args.root.rglob("summary.json"):
        if not path.is_file():
            continue
        relative = path.relative_to(args.root)
        if len(relative.parts) < 2:
            continue
        variant = relative.parts[0]
        current = summary_paths.get(variant)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            summary_paths[variant] = path

    for variant, path in sorted(summary_paths.items()):
        summary = json.loads(path.read_text(encoding="utf-8"))
        usage = summary.get("llm_usage", {}).get("generation", {}).get("total", {})
        evolved = _get(summary, "evolved_test", "ast_cds") or {}
        seed = _get(summary, "seed_test", "ast_cds") or {}
        history = summary.get("batch_history") or []
        rows.append({
            "variant": variant,
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
            "status": "error" if any(item.get("status") == "error" for item in history) else "completed",
            "summary_path": str(path),
            "config_variant": _get(summary, "config", "ablation_variant"),
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
            f"{row['variant']:16s} status={row['status']:9s} AST={row['evolved_ast_joint']} "
            f"action={row['evolved_action']} slot={row['evolved_slot']} "
            f"calls={row['generation_calls']} changes={row['skill_changes']} "
            f"path={row['summary_path']}"
        )


if __name__ == "__main__":
    main()
