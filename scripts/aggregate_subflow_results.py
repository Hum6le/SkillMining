#!/usr/bin/env python3
"""Aggregate independent ABCD subflow evaluation runs.

Each method is trained/evaluated separately for one subflow. This script only
aggregates the saved summaries; it never mixes conversations during inference.

Examples:
    python scripts/aggregate_subflow_results.py --runs outputs/awm_abcd_a outputs/awm_abcd_b
    python scripts/aggregate_subflow_results.py --runs outputs/abcd_runs --recursive
    python scripts/aggregate_subflow_results.py --runs outputs/subflow_eval_x
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TEXT_METRICS = (
    "bert_f1", "bert_precision", "bert_recall", "bleu_1", "bleu_4",
    "rouge_1", "rouge_2", "rouge_l", "meteor",
)
AST_METRICS = ("ast_joint", "ast_action_name", "ast_slot_value", "cds_overall")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_paths(inputs: list[str], recursive: bool) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw).resolve()
        if path.is_file() and path.name == "summary.json":
            paths.append(path)
        elif path.is_dir():
            pattern = "**/summary.json" if recursive else "summary.json"
            paths.extend(path.glob(pattern))
    return sorted(set(p for p in paths if p.is_file()))


def _record_from_eval(
    *, method: str, phase: str, subflow: str, run_dir: Path,
    payload: dict[str, Any], data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or "text" not in payload or "ast_cds" not in payload:
        return None
    text = payload["text"]
    ast = payload["ast_cds"]
    data = data or {}
    return {
        "method": method,
        "phase": phase,
        "subflow": subflow,
        "run_dir": str(run_dir),
        "test_sessions": int(data.get("test_sessions", payload.get("num_conversations", 0)) or 0),
        "text_samples": int(text.get("num_samples", payload.get("num_turns", 0)) or 0),
        "action_turns": int(ast.get("num_action_turns", 0) or 0),
        "metrics": {
            **{name: float(text[name]) for name in TEXT_METRICS if name in text},
            **{name: float(ast[name]) for name in AST_METRICS if name in ast},
        },
    }


def _records_from_summary(path: Path) -> list[dict[str, Any]]:
    summary = _read_json(path)
    run_dir = path.parent

    # Graph Mining currently stores {subflow: {seed, mined, ...}} directly.
    graph_rows = {
        key: value for key, value in summary.items()
        if not str(key).startswith("__")
    } if isinstance(summary, dict) else {}
    if graph_rows and all(
        isinstance(value, dict) and ("mined" in value or "seed" in value)
        for value in graph_rows.values()
    ):
        records = []
        for subflow, row in graph_rows.items():
            data = {
                "test_sessions": row.get("test_sessions", 0),
            }
            # ``unordered`` is the flat node/edge compiler control used by
            # the backbone skill-organization ablation. Older graph summaries
            # simply omit it.
            for phase in ("seed", "mined", "unordered"):
                record = _record_from_eval(
                    method="graph_mining", phase=phase, subflow=str(subflow),
                    run_dir=run_dir, payload=row.get(phase), data=data,
                )
                if record:
                    records.append(record)
        return records

    if not isinstance(summary, dict):
        return []
    config = summary.get("config", {})
    subflow = str(config.get("subflow", "unknown"))
    data = summary.get("data", {})
    records = []
    method = str(config.get("method", "awm"))
    if summary.get("final_test"):
        record = _record_from_eval(
            method=method, phase="final", subflow=subflow,
            run_dir=run_dir, payload=summary["final_test"], data=data,
        )
        if record:
            records.append(record)
    for phase, key in (("seed", "seed_test"), ("evolved", "evolved_test")):
        if summary.get(key):
            record = _record_from_eval(
                method="trace2skill", phase=phase, subflow=subflow,
                run_dir=run_dir, payload=summary[key], data=data,
            )
            if record:
                records.append(record)
    return records


def _weighted_average(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_key in sorted({(r["method"], r["phase"]) for r in records}):
        method, phase = group_key
        group = [r for r in records if (r["method"], r["phase"]) == group_key]
        metric_values: dict[str, float] = {}
        weights = {
            "text": sum(max(r["text_samples"], 1) for r in group),
            "ast": sum(max(r["action_turns"], 1) for r in group),
            "cds": sum(max(r["test_sessions"], 1) for r in group),
        }
        for metric in TEXT_METRICS:
            values = [(r["metrics"][metric], max(r["text_samples"], 1)) for r in group if metric in r["metrics"]]
            if values:
                metric_values[metric] = sum(value * weight for value, weight in values) / sum(weight for _, weight in values)
        for metric in ("ast_joint", "ast_action_name", "ast_slot_value"):
            values = [(r["metrics"][metric], max(r["action_turns"], 1)) for r in group if metric in r["metrics"]]
            if values:
                metric_values[metric] = sum(value * weight for value, weight in values) / sum(weight for _, weight in values)
        values = [(r["metrics"]["cds_overall"], max(r["test_sessions"], 1)) for r in group if "cds_overall" in r["metrics"]]
        if values:
            metric_values["cds_overall"] = sum(value * weight for value, weight in values) / sum(weight for _, weight in values)
        result[f"{method}:{phase}"] = {
            "num_subflows": len(group),
            "subflows": sorted(r["subflow"] for r in group),
            "weights": weights,
            "metrics": {key: round(value, 6) for key, value in metric_values.items()},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate independent ABCD subflow results")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories, summary.json files, or a parent directory")
    parser.add_argument("--recursive", action="store_true", help="Search summary.json recursively below --runs directories")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    paths = _summary_paths(args.runs, args.recursive)
    records = [record for path in paths for record in _records_from_summary(path)]
    if not records:
        raise SystemExit("No recognizable ABCD subflow summaries found")
    result = {
        "protocol": "independent_subflow_runs",
        "summary_files": [str(path) for path in paths],
        "records": records,
        "aggregate": _weighted_average(records),
    }
    output = Path(args.output) if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
