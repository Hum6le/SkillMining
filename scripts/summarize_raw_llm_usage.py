#!/usr/bin/env python3
"""Summarize LLM calls from persisted ResponseLogger artifacts.

The script is intended for old ASI/AWM runs whose ``llm_usage.json`` is
missing or incomplete. Response files are counted as calls; API ``usage`` is
used when present, otherwise tokens are estimated from the matching prompt
and response text.
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


def _finalize(bucket: dict) -> dict:
    if bucket["exact_calls"] and bucket["estimated_calls"]:
        bucket["usage_source"] = "mixed"
    elif bucket["exact_calls"]:
        bucket["usage_source"] = "exact"
    elif bucket["estimated_calls"]:
        bucket["usage_source"] = "estimated"
    return bucket


def summarize(root: Path) -> dict:
    response_paths = sorted(root.rglob("*_response.json"))
    prompts = {}
    for path in root.rglob("*_prompt.json"):
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
    return {"root": str(root), "response_files": len(response_paths),
            "unreadable_response_files": errors, "total": _finalize(total),
            "by_call_tag": {tag: _finalize(value) for tag, value in sorted(by_tag.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize old raw LLM response logs")
    parser.add_argument("roots", nargs="+", type=Path,
                        help="Run directories or llm_responses directories")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    reports = [summarize(path) for path in args.roots]
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
