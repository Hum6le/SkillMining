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
AST_METRICS = (
    "ast_joint", "ast_action_name", "ast_slot_value",
    "ast_slot_value_given_action", "cds_overall",
)


def _empty_usage() -> dict[str, Any]:
    return {
        "calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "calls_with_usage": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "exact_calls": 0,
        "estimated_calls": 0,
        "usage_available": False,
        "usage_source": "unavailable",
    }


def _add_usage(total: dict[str, Any], usage: dict[str, Any] | None) -> None:
    if not isinstance(usage, dict):
        return
    # The workflow-aware server llm.py stores the aggregate under `total`,
    # while older runners emitted the bucket directly. Accept both formats.
    if isinstance(usage.get("total"), dict):
        usage = usage["total"]
    for key in (
        "calls", "successful_calls", "failed_calls", "calls_with_usage",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "exact_calls", "estimated_calls",
    ):
        total[key] += int(usage.get(key, 0) or 0)
    total["usage_available"] = bool(
        total["usage_available"] or usage.get("usage_available", False)
    )
    sources = {str(total.get("usage_source", "unavailable")), str(usage.get("usage_source", "unavailable"))}
    sources.discard("unavailable")
    total["usage_source"] = (
        "mixed" if len(sources) > 1 else next(iter(sources), "unavailable")
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_usage(run_dir: Path) -> dict[str, Any] | None:
    """Load usage beside a summary for compatibility with older runners."""
    path = run_dir / "llm_usage.json"
    if not path.is_file():
        return None
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _has_usage(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    bucket = usage.get("total") if isinstance(usage.get("total"), dict) else usage
    return any(int(bucket.get(key, 0) or 0) for key in (
        "calls", "prompt_tokens", "completion_tokens", "total_tokens",
    ))


def _estimate_raw_log_usage(run_dir: Path) -> dict[str, Any] | None:
    """Recover usage from saved prompts when an old runner wrote no tracker.

    Workflow APIs often omit token metadata, but ResponseLogger still keeps
    every prompt and response. This is intentionally a last-resort estimate
    for existing runs; newly produced ``llm_usage.json`` remains authoritative.
    """
    prompt_paths = sorted(run_dir.glob("**/*_prompt.json"))
    if not prompt_paths:
        return None

    def estimate(text: Any) -> int:
        text = str(text or "")
        return max(1, (len(text) + 3) // 4) if text else 0

    def bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out = {
            "calls": len(rows), "successful_calls": len(rows), "failed_calls": 0,
            "calls_with_usage": len(rows), "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0, "exact_calls": 0,
            "estimated_calls": len(rows), "usage_available": True,
            "usage_source": "estimated",
        }
        for row in rows:
            out["prompt_tokens"] += row["prompt_tokens"]
            out["completion_tokens"] += row["completion_tokens"]
            out["total_tokens"] += row["total_tokens"]
        return out

    rows: list[dict[str, Any]] = []
    for prompt_path in prompt_paths:
        try:
            prompt = _read_json(prompt_path)
        except (OSError, json.JSONDecodeError):
            continue
        messages = prompt.get("messages", []) if isinstance(prompt, dict) else []
        prompt_text = "\n".join(
            str(message.get("content", ""))
            for message in messages if isinstance(message, dict)
        )
        response_path = prompt_path.with_name(prompt_path.name.replace("_prompt.json", "_response.json"))
        response_text = ""
        if response_path.is_file():
            try:
                response = _read_json(response_path)
                choices = response.get("choices", []) if isinstance(response, dict) else []
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get("message", {})
                    response_text = message.get("content", "") if isinstance(message, dict) else ""
                if not response_text and isinstance(response, dict):
                    response_text = response.get("content", "")
            except (OSError, json.JSONDecodeError):
                pass
        prompt_tokens = estimate(prompt_text)
        completion_tokens = estimate(response_text)
        rows.append({
            "call_tag": str(prompt.get("call_tag", "chat")) if isinstance(prompt, dict) else "chat",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })
    if not rows:
        return None
    by_tag = {}
    for tag in sorted({row["call_tag"] for row in rows}):
        by_tag[tag] = bucket([row for row in rows if row["call_tag"] == tag])
    total = bucket(rows)
    return {
        "schema_version": 1,
        "generated_at": "recovered-from-response-logs",
        "total": total,
        "by_call_tag": by_tag,
        "by_provider": {"workflow_or_openai": total},
    }


def _summary_paths(inputs: list[str], recursive: bool) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw).resolve()
        if path.is_file() and path.name in {"summary.json", "online_refine_result.json"}:
            paths.append(path)
        elif path.is_dir():
            names = ("summary.json", "online_refine_result.json")
            for name in names:
                pattern = f"**/{name}" if recursive else name
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
        "action_correct_turns": int(
            ast.get("num_action_correct_turns", ast.get("num_action_turns", 0)) or 0
        ),
        "metrics": {
            **{name: float(text[name]) for name in TEXT_METRICS if name in text},
            **{name: float(ast[name]) for name in AST_METRICS if name in ast},
        },
        "llm_usage": payload.get("llm_usage"),
    }


def _records_from_summary(path: Path) -> list[dict[str, Any]]:
    summary = _read_json(path)
    run_dir = path.parent

    if path.name == "online_refine_result.json":
        subflow = run_dir.name
        record = _record_from_eval(
            method="backbone_online_refine", phase="online_refined",
            subflow=subflow, run_dir=run_dir, payload=summary,
        )
        return [record] if record else []

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
                    record["llm_usage"] = row.get("llm_usage")
                    records.append(record)
        return records

    if not isinstance(summary, dict):
        return []
    config = summary.get("config", {})
    subflow = str(config.get("subflow", "unknown"))
    data = summary.get("data", {})
    summary_usage = summary.get("llm_usage")
    if not _has_usage(summary_usage):
        file_usage = _load_run_usage(run_dir)
        if _has_usage(file_usage):
            summary_usage = file_usage
    if not _has_usage(summary_usage):
        raw_usage = _estimate_raw_log_usage(run_dir)
        if _has_usage(raw_usage):
            summary_usage = raw_usage
    records = []
    method = str(config.get("method", "awm"))
    if summary.get("final_test"):
        record = _record_from_eval(
            method=method, phase="final", subflow=subflow,
            run_dir=run_dir, payload=summary["final_test"], data=data,
        )
        if record:
            # Older unified-evaluation summaries kept usage at the summary
            # level instead of embedding it in final_test. Accept both forms
            # so existing runs can be aggregated without re-evaluation.
            if record.get("llm_usage") is None and isinstance(summary_usage, dict):
                record["llm_usage"] = summary_usage
            records.append(record)
    for phase, key in (("seed", "seed_test"), ("evolved", "evolved_test")):
        if summary.get(key):
            record = _record_from_eval(
                method="trace2skill", phase=phase, subflow=subflow,
                run_dir=run_dir, payload=summary[key], data=data,
            )
            if record:
                if record.get("llm_usage") is None and isinstance(summary_usage, dict):
                    record["llm_usage"] = summary_usage
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
        values = [
            (r["metrics"]["ast_slot_value_given_action"], max(r.get("action_correct_turns", 0), 1))
            for r in group if "ast_slot_value_given_action" in r["metrics"]
        ]
        if values:
            metric_values["ast_slot_value_given_action"] = sum(
                value * weight for value, weight in values
            ) / sum(weight for _, weight in values)
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
    usage_total = _empty_usage()
    for record in records:
        _add_usage(usage_total, record.get("llm_usage"))
    result["llm_usage"] = usage_total
    output = Path(args.output) if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
