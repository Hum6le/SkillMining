#!/usr/bin/env python3
"""Aggregate completed independent SKILL-DISCO ABCD evaluation results.

The action-state metrics are weighted by action turns, text metrics by text
samples, and CDS by test conversations. This avoids treating a small subflow
as equally large as a much larger one.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TEXT_METRICS = (
    "bert_f1", "bert_precision", "bert_recall", "bleu_1", "bleu_4",
    "rouge_1", "rouge_2", "rouge_l", "meteor",
)
AST_METRICS = ("ast_joint", "ast_action_name", "ast_slot_value")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result must be a JSON object: {path}")
    return payload


def _subflow_from_result_path(path: Path) -> str:
    run_dir = path.parent.parent
    manifest = run_dir / "manifest.txt"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.startswith("subflow=") and line.partition("=")[2].strip():
                return line.partition("=")[2].strip()
    match = re.fullmatch(r"skill_disco_abcd_subflow_(.+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", run_dir.name)
    if match:
        return match.group(1)
    raise ValueError(f"could not determine subflow from {path}; expected a run manifest")


def collect_records(result_paths: list[Path]) -> list[dict[str, Any]]:
    """Load one completed ``evaluation/result.json`` per subflow."""
    records: list[dict[str, Any]] = []
    seen_subflows: set[str] = set()
    for path in result_paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"evaluation result not found: {resolved}")
        payload = _read_json(resolved)
        ast = payload.get("ast_cds")
        if not isinstance(ast, dict) or "num_action_turns" not in ast:
            raise ValueError(f"missing ast_cds metrics in {resolved}")
        subflow = _subflow_from_result_path(resolved)
        if subflow in seen_subflows:
            raise ValueError(f"duplicate result for subflow {subflow!r}: {resolved}")
        seen_subflows.add(subflow)
        text = payload.get("text") if isinstance(payload.get("text"), dict) else {}
        records.append({
            "subflow": subflow,
            "result_path": str(resolved),
            "test_sessions": int(payload.get("num_conversations", 0) or 0),
            "text_samples": int(text.get("num_samples", 0) or 0),
            "action_turns": int(ast.get("num_action_turns", 0) or 0),
            "metrics": {
                **{name: float(text[name]) for name in TEXT_METRICS if name in text},
                **{name: float(ast[name]) for name in AST_METRICS if name in ast},
                **({"cds_overall": float(ast["cds_overall"])} if "cds_overall" in ast else {}),
            },
        })
    return sorted(records, key=lambda record: record["subflow"])


def _weighted_metric(records: list[dict[str, Any]], metric: str, weight_key: str) -> float | None:
    values = [
        (record["metrics"][metric], max(int(record[weight_key]), 1))
        for record in records
        if metric in record["metrics"]
    ]
    if not values:
        return None
    return sum(value * weight for value, weight in values) / sum(weight for _, weight in values)


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no completed result files were supplied")
    metrics: dict[str, float] = {}
    for metric in TEXT_METRICS:
        value = _weighted_metric(records, metric, "text_samples")
        if value is not None:
            metrics[metric] = round(value, 6)
    for metric in AST_METRICS:
        value = _weighted_metric(records, metric, "action_turns")
        if value is not None:
            metrics[metric] = round(value, 6)
    cds = _weighted_metric(records, "cds_overall", "test_sessions")
    if cds is not None:
        metrics["cds_overall"] = round(cds, 6)
    return {
        "protocol": "independent_skill_disco_subflow_runs",
        "num_subflows": len(records),
        "subflows": [record["subflow"] for record in records],
        "weights": {
            "test_sessions": sum(record["test_sessions"] for record in records),
            "text_samples": sum(record["text_samples"] for record in records),
            "action_turns": sum(record["action_turns"] for record in records),
        },
        "metrics": metrics,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SKILL-DISCO ABCD evaluation result.json files")
    parser.add_argument("--results", nargs="+", required=True, help="Completed evaluation/result.json files; one per subflow")
    parser.add_argument("--output", required=True, help="Path for the aggregate JSON artifact")
    args = parser.parse_args()

    aggregate = aggregate_records(collect_records([Path(value) for value in args.results]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "num_subflows": aggregate["num_subflows"],
        "weights": aggregate["weights"],
        "metrics": aggregate["metrics"],
        "output": str(output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
