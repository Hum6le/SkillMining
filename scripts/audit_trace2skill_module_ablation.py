#!/usr/bin/env python3
"""Audit whether Trace2Skill ablation variants used distinct skills and predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_CONFIG = {
    "full": (True, True, False, False),
    "no_failure_analysis": (True, False, False, False),
    "no_success_memory": (False, True, False, False),
    "no_evolution": (True, True, True, False),
    "one_shot_update": (True, True, False, True),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _latest_run(variant_dir: Path) -> Path | None:
    candidates = [
        path.parent
        for path in variant_dir.glob("abcd_trace2skill_*/summary.json")
        if path.is_file()
    ]
    if (variant_dir / "summary.json").is_file():
        candidates.append(variant_dir)
    return max(candidates, key=lambda path: (path / "summary.json").stat().st_mtime) if candidates else None


def _prediction_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("convo_id", "")),
        int(row.get("turn_index", -1)),
        str(row.get("predicted_action", "")),
        tuple(str(value) for value in row.get("predicted_slots", []) or []),
    )


def _output_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return _prediction_key(row) + (str(row.get("prediction", "")),)


def _prompt_hashes(rows: list[dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for row in rows:
        for step in row.get("react_trace", []) or []:
            action_input = step.get("action_input", {}) if isinstance(step, dict) else {}
            messages = action_input.get("messages", []) if isinstance(action_input, dict) else []
            for message in messages if isinstance(messages, list) else []:
                if isinstance(message, dict) and message.get("role") == "system":
                    hashes.add(_sha(str(message.get("content", ""))))
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    records: dict[str, dict[str, Any]] = {}
    for variant in EXPECTED_CONFIG:
        run_dir = _latest_run(args.root / variant)
        if run_dir is None:
            print(f"{variant:22s} MISSING")
            continue
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        prediction_path = run_dir / "evolved_test_turns.json"
        skill_path = run_dir / "evolved_skill" / "SKILL.md"
        rows = json.loads(prediction_path.read_text(encoding="utf-8"))
        skill_text = skill_path.read_text(encoding="utf-8")
        config = summary.get("config", {})
        observed_config = (
            bool(config.get("success_analysis_enabled", True)),
            bool(config.get("failure_analysis_enabled", True)),
            bool(config.get("skip_evolution", False)),
            bool(config.get("one_shot_update", False)),
        )
        records[variant] = {
            "run_dir": str(run_dir),
            "skill_hash": _sha(skill_text),
            "prediction_hash": _sha(json.dumps([_output_key(row) for row in rows], ensure_ascii=False)),
            "ast_hash": _sha(json.dumps([_prediction_key(row) for row in rows], ensure_ascii=False)),
            "keys": [_prediction_key(row) for row in rows],
            "prompt_hashes": _prompt_hashes(rows),
            "changes": len(summary.get("changelog") or []),
            "expected_config": EXPECTED_CONFIG[variant],
            "observed_config": observed_config,
            "ast": (summary.get("evolved_test") or {}).get("ast_cds", {}),
        }

    print("variant                skill hash       AST-pred hash    raw-pred hash    changes config")
    for variant, record in records.items():
        config_status = "OK" if record["observed_config"] == record["expected_config"] else "MISMATCH"
        print(
            f"{variant:22s} {record['skill_hash']} {record['ast_hash']} "
            f"{record['prediction_hash']} {record['changes']:7d} {config_status}"
        )
        if config_status != "OK":
            print(f"  expected={record['expected_config']} observed={record['observed_config']}")
        print(f"  run={record['run_dir']}")
        print(f"  evolved_ast={record['ast']}")

    variants = list(records)
    pairwise: list[dict[str, Any]] = []
    print("\nPairwise AST prediction differences:")
    for left_index, left in enumerate(variants):
        for right in variants[left_index + 1:]:
            left_keys = records[left]["keys"]
            right_keys = records[right]["keys"]
            compared = min(len(left_keys), len(right_keys))
            differences = sum(left_keys[i] != right_keys[i] for i in range(compared))
            differences += abs(len(left_keys) - len(right_keys))
            same_prompts = records[left]["prompt_hashes"] == records[right]["prompt_hashes"]
            pairwise.append({
                "left": left,
                "right": right,
                "different_ast_turns": differences,
                "total_turns": max(len(left_keys), len(right_keys)),
                "same_skill": records[left]["skill_hash"] == records[right]["skill_hash"],
                "same_ast_predictions": records[left]["ast_hash"] == records[right]["ast_hash"],
                "same_outputs": records[left]["prediction_hash"] == records[right]["prediction_hash"],
                "same_system_prompt_set": same_prompts,
            })
            print(
                f"{left:22s} vs {right:22s}: "
                f"different_turns={differences}/{max(len(left_keys), len(right_keys))}, "
                f"same_system_prompt_set={same_prompts}"
            )

    serializable_records = {
        variant: {key: value for key, value in record.items() if key != "keys"}
        for variant, record in records.items()
    }
    for record in serializable_records.values():
        record["prompt_hashes"] = sorted(record["prompt_hashes"])
    output = args.root / "module_ablation_audit.json"
    output.write_text(
        json.dumps({"variants": serializable_records, "pairwise": pairwise}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote audit report to {output}")


if __name__ == "__main__":
    main()
