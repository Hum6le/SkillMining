#!/usr/bin/env python3
"""Summarize LLM calls from persisted experiment artifacts.

The script is intended for old ASI/AWM/online-refine runs whose ``llm_usage.json`` is
missing or incomplete. Response files are counted as calls; API ``usage`` is
used when present, otherwise tokens are estimated from the matching prompt
and response text. Older online-refine runs also persisted optimizer calls as
``prompt``/``raw_response`` fields in ``autonomous_reflection/batch_*.json``;
those records are imported as well.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _text_tokens(value: object) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return max(1, (len(text) + 3) // 4) if text else 0


def _response_text(record: dict) -> str:
    choices = record.get("choices")
    if isinstance(choices, list):
        parts = []
        for choice in choices:
            if isinstance(choice, dict):
                message = choice.get("message", {})
                if isinstance(message, dict):
                    parts.append(str(message.get("content", "") or ""))
        return "\n".join(parts)
    return str(record.get("content", record.get("raw", "")) or "")


def _prompt_text(record: dict) -> str:
    messages = record.get("messages", [])
    if isinstance(messages, list):
        return "\n".join(
            str(item.get("content", "") or "") for item in messages
            if isinstance(item, dict)
        )
    return ""


def _bucket() -> dict:
    return {"calls": 0, "calls_with_usage": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0,
            "exact_calls": 0, "estimated_calls": 0,
            "usage_source": "unavailable"}


def _add(bucket: dict, response: dict, prompt: dict | None) -> None:
    bucket["calls"] += 1
    usage = response.get("usage")
    exact = isinstance(usage, dict) and any(
        usage.get(key) is not None
        for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens")
    )
    if exact:
        def number(*keys):
            return next((int(usage[key]) for key in keys if usage.get(key) is not None), 0)
        prompt_tokens = number("prompt_tokens", "input_tokens")
        completion_tokens = number("completion_tokens", "output_tokens")
        total_tokens = number("total_tokens") or prompt_tokens + completion_tokens
        bucket["exact_calls"] += 1
    else:
        prompt_tokens = _text_tokens(_prompt_text(prompt or {}))
        completion_tokens = _text_tokens(_response_text(response))
        total_tokens = prompt_tokens + completion_tokens
        bucket["estimated_calls"] += 1
    bucket["calls_with_usage"] += 1
    bucket["prompt_tokens"] += prompt_tokens
    bucket["completion_tokens"] += completion_tokens
    bucket["total_tokens"] += total_tokens


def _add_embedded(bucket: dict, prompt_text: object, response_text: object,
                  usage: object = None) -> None:
    """Count an optimizer call embedded in an old online-refine JSON record."""
    bucket["calls"] += 1
    if isinstance(usage, dict) and any(
        usage.get(key) is not None
        for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens")
    ):
        def number(*keys):
            return next((int(usage[key]) for key in keys if usage.get(key) is not None), 0)
        prompt_tokens = number("prompt_tokens", "input_tokens")
        completion_tokens = number("completion_tokens", "output_tokens")
        total_tokens = number("total_tokens") or prompt_tokens + completion_tokens
        bucket["exact_calls"] += 1
    else:
        prompt_tokens = _text_tokens(prompt_text)
        completion_tokens = _text_tokens(response_text)
        total_tokens = prompt_tokens + completion_tokens
        bucket["estimated_calls"] += 1
    bucket["calls_with_usage"] += 1
    bucket["prompt_tokens"] += prompt_tokens
    bucket["completion_tokens"] += completion_tokens
    bucket["total_tokens"] += total_tokens


def _finalize(bucket: dict) -> dict:
    if bucket["exact_calls"] and bucket["estimated_calls"]:
        bucket["usage_source"] = "mixed"
    elif bucket["exact_calls"]:
        bucket["usage_source"] = "exact"
    elif bucket["estimated_calls"]:
        bucket["usage_source"] = "estimated"
    return bucket


def summarize(root: Path) -> dict:
    root = root.resolve()
    # Users often pass the logger child directory. The old online optimizer
    # artifacts live beside it at the run root, so retain both locations.
    scan_roots = [root]
    if root.is_dir() and root.name in {"llm_responses", "llm_logs", "llm_calls"}:
        scan_roots.append(root.parent)
    response_paths = []
    prompt_paths = []
    for scan_root in scan_roots:
        response_paths.extend(scan_root.rglob("*_response.json"))
        prompt_paths.extend(scan_root.rglob("*_prompt.json"))
    response_paths = sorted(set(response_paths))
    prompt_paths = sorted(set(prompt_paths))
    prompts = {}
    for path in prompt_paths:
        try:
            prompts[path.stem.rsplit("_prompt", 1)[0]] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    total = _bucket()
    by_tag = defaultdict(_bucket)
    errors = 0
    for path in response_paths:
        try:
            response = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors += 1
            continue
        tag = str(response.get("call_tag", "unknown"))
        key = path.stem.rsplit("_response", 1)[0]
        _add(total, response, prompts.get(key))
        _add(by_tag[tag], response, prompts.get(key))
    # Old online refinement did not pass a ResponseLogger to the resource
    # planner/reflection calls. They are persisted in one JSON object per
    # batch. Scan only autonomous_reflection, because batch_diagnostics embeds
    # the same object and would double count these calls.
    embedded_calls = 0
    reflection_paths = []
    for scan_root in scan_roots:
        if scan_root.is_dir():
            reflection_paths.extend(scan_root.rglob("autonomous_reflection/batch_*.json"))
    for path in sorted(set(reflection_paths)):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        embedded = [
            ("online_resource_planner", record.get("planner_prompt"), record.get("planner_raw_response")),
            ("online_resource_reflection", record.get("prompt"), record.get("raw_response")),
        ]
        for tag, prompt_text, response_text in embedded:
            if not isinstance(prompt_text, str) or not prompt_text.strip():
                continue
            if not isinstance(response_text, str) or not response_text.strip():
                continue
            _add_embedded(total, prompt_text, response_text)
            _add_embedded(by_tag[tag], prompt_text, response_text)
            embedded_calls += 1

    # Some interrupted/older runs saved the same reflection object under a
    # different JSON path. Scan fallback JSON files only for batches that were
    # not already found canonically; this also handles diagnostics-only runs
    # without double counting their nested copy.
    canonical_batches = {path.stem for path in reflection_paths}
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            continue
        for path in sorted(scan_root.rglob("*.json")):
            if "autonomous_reflection" in path.parts:
                continue
            if path.name == "llm_usage.json" or path.resolve() in set(response_paths):
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict) or not (
                "planner_raw_response" in record or "raw_response" in record
            ):
                continue
            batch_name = path.stem if path.stem.startswith("batch_") else ""
            if batch_name and batch_name in canonical_batches:
                continue
            for tag, prompt_key, response_key in (
                ("online_resource_planner", "planner_prompt", "planner_raw_response"),
                ("online_resource_reflection", "prompt", "raw_response"),
            ):
                prompt_text, response_text = record.get(prompt_key), record.get(response_key)
                if not isinstance(prompt_text, str) or not prompt_text.strip():
                    continue
                if not isinstance(response_text, str) or not response_text.strip():
                    continue
                _add_embedded(total, prompt_text, response_text)
                _add_embedded(by_tag[tag], prompt_text, response_text)
                embedded_calls += 1
    return {"root": str(root), "response_files": len(response_paths),
            "embedded_online_calls": embedded_calls,
            "unreadable_response_files": errors, "total": _finalize(total),
            "by_call_tag": {tag: _finalize(value) for tag, value in sorted(by_tag.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize old raw LLM response logs")
    parser.add_argument("roots", nargs="+", type=Path,
                        help="Run directories or llm_responses directories")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    reports = []
    for path in args.roots:
        # Make the common mistake of passing llm_usage.json useful: preserve
        # its already-computed values instead of reporting an empty scan.
        if path.is_file() and path.name == "llm_usage.json":
            try:
                reports.append({"root": str(path), "source_usage_file": True,
                                "usage": json.loads(path.read_text(encoding="utf-8"))})
                continue
            except (OSError, json.JSONDecodeError):
                pass
        reports.append(summarize(path))
    payload = {"schema_version": 1,
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "runs": reports}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
