#!/usr/bin/env python3
"""Aggregate LLM usage for all online-refinement runs below one directory.

Unlike the low-level raw usage script, this command discovers run directories
automatically. It supports both the current ResponseLogger layout and older
online-refine runs whose planner/reflection calls are embedded in JSON files.

Example:
    python scripts/summarize_online_refine_usage.py \
        /data1/liuxiangfeng/SkillMining/outputs/online_refine \
        --output /data1/liuxiangfeng/SkillMining/outputs/online_refine/usage_all.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.summarize_raw_llm_usage import summarize


def _is_run_dir(path: Path) -> bool:
    """Identify an online-refine run without depending on its subflow name."""
    return (
        (path / "online_refine_result.json").is_file()
        or (path / "skill_dag_state.json").is_file()
        or (path / "autonomous_reflection").is_dir()
        or (path / "llm_responses").is_dir()
    )


def _discover_runs(root: Path) -> list[Path]:
    root = root.resolve()
    if _is_run_dir(root):
        return [root]
    runs = [path for path in root.rglob("*") if path.is_dir() and _is_run_dir(path)]
    # A parent run may contain child artifacts, but should still be counted
    # once. Keep the deepest candidate when candidates overlap.
    selected: list[Path] = []
    for path in sorted(set(runs), key=lambda item: (len(item.parts), str(item))):
        if any(path.is_relative_to(existing) for existing in selected):
            continue
        selected.append(path)
    return sorted(selected)


def _bucket() -> dict[str, int]:
    return {
        "calls": 0,
        "calls_with_usage": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "exact_calls": 0,
        "estimated_calls": 0,
    }


def _add_bucket(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in _bucket():
        target[key] = int(target.get(key, 0) or 0) + int(source.get(key, 0) or 0)


def _finalize(bucket: dict[str, Any]) -> dict[str, Any]:
    exact = int(bucket.get("exact_calls", 0))
    estimated = int(bucket.get("estimated_calls", 0))
    if exact and estimated:
        source = "mixed"
    elif exact:
        source = "exact"
    elif estimated:
        source = "estimated"
    else:
        source = "unavailable"
    return {**bucket, "usage_source": source}


def _subflow_name(run_dir: Path, payload: dict[str, Any]) -> str:
    summary_path = run_dir / "online_refine_result.json"
    for candidate in (run_dir / "summary.json", summary_path):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            for key in ("subflow", "scenario"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            config = data.get("config")
            if isinstance(config, dict) and str(config.get("subflow", "")).strip():
                return str(config["subflow"]).strip()
    # The runner always receives --subflow, but older runs may lack a summary.
    return run_dir.name


def aggregate(root: Path) -> dict[str, Any]:
    runs = _discover_runs(root)
    total = _bucket()
    by_tag: dict[str, dict[str, Any]] = {}
    by_subflow: dict[str, dict[str, Any]] = {}
    details: list[dict[str, Any]] = []

    for run_dir in runs:
        report = summarize(run_dir)
        run_total = report.get("total", {})
        _add_bucket(total, run_total)
        for tag, bucket in report.get("by_call_tag", {}).items():
            target = by_tag.setdefault(tag, _bucket())
            _add_bucket(target, bucket)
        subflow = _subflow_name(run_dir, report)
        target = by_subflow.setdefault(subflow, _bucket())
        _add_bucket(target, run_total)
        details.append({
            "run_dir": str(run_dir),
            "subflow": subflow,
            "response_files": report.get("response_files", 0),
            "embedded_online_calls": report.get("embedded_online_calls", 0),
            "total": _finalize(dict(run_total)),
            "by_call_tag": report.get("by_call_tag", {}),
        })

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "run_count": len(details),
        "total": _finalize(total),
        "by_subflow": {
            name: _finalize(bucket) for name, bucket in sorted(by_subflow.items())
        },
        "by_call_tag": {
            tag: _finalize(bucket) for tag, bucket in sorted(by_tag.items())
        },
        "runs": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate raw LLM usage across all online-refine runs"
    )
    parser.add_argument("root", type=Path, help="Parent directory containing online-refine runs")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = aggregate(args.root)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
