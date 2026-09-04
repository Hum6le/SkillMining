"""Runner-side adapter for the server-provided shared ``llm.py`` tracker.

The repository keeps the production replacement as ``llm_new.py``; on the
server it is deployed as ``llm.py``.  This adapter keeps local legacy runners
importable while making every runner persist the same usage schema when the
new module is present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone


def _empty_usage_summary() -> dict[str, Any]:
    bucket = {
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
    return {
        "schema_version": 1,
        "started_at": "",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": bucket,
        "by_call_tag": {},
        "by_provider": {},
    }


def merge_usage_summaries(*summaries: dict[str, Any]) -> dict[str, Any]:
    """Merge process-local usage snapshots, including parallel workers."""
    valid = [item for item in summaries if isinstance(item, dict)]
    if not valid:
        return get_usage()

    def merge_bucket(items):
        keys = ("calls", "successful_calls", "failed_calls", "calls_with_usage",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "exact_calls", "estimated_calls")
        out = {key: sum(int((item or {}).get(key, 0) or 0) for item in items) for key in keys}
        out["usage_available"] = any(bool((item or {}).get("usage_available")) for item in items)
        sources = {str((item or {}).get("usage_source", "unavailable")) for item in items
                   if (item or {}).get("usage_source") not in {None, "unavailable"}}
        out["usage_source"] = "mixed" if len(sources) > 1 else (next(iter(sources)) if sources else "unavailable")
        return out

    total = merge_bucket([item.get("total", {}) for item in valid])
    tags = set().union(*(item.get("by_call_tag", {}).keys() for item in valid))
    providers = set().union(*(item.get("by_provider", {}).keys() for item in valid))
    return {
        "schema_version": 1,
        "started_at": min((item.get("started_at", "") for item in valid), default=""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "by_call_tag": {
            tag: merge_bucket([item.get("by_call_tag", {}).get(tag, {}) for item in valid])
            for tag in sorted(tags)
        },
        "by_provider": {
            provider: merge_bucket([item.get("by_provider", {}).get(provider, {}) for item in valid])
            for provider in sorted(providers)
        },
    }


def split_usage_summary(
    generation: dict[str, Any] | None,
    testing: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the common generation/testing/total usage schema.

    ``generation`` includes mining, induction, compilation and refinement;
    ``testing`` includes seed/dev/final test rollouts.  Keeping the snapshots
    separate prevents a later evaluation pass from obscuring where calls were
    spent.
    """
    generation = generation if isinstance(generation, dict) else _empty_usage_summary()
    testing = testing if isinstance(testing, dict) else _empty_usage_summary()
    return {
        "schema_version": 2,
        "generation": generation,
        "testing": testing,
        "total": merge_usage_summaries(generation, testing),
    }


def usage_total(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Return the aggregate bucket from either old or phase-split usage."""
    if not isinstance(usage, dict):
        return {}
    value = usage.get("total")
    return value if isinstance(value, dict) else usage


def reset_usage() -> None:
    import llm
    fn = getattr(llm, "reset_usage_summary", None)
    if fn:
        fn()


def get_usage() -> dict[str, Any]:
    import llm
    fn = getattr(llm, "get_usage_summary", None)
    if fn:
        return fn()
    return {
        "schema_version": 0,
        "usage_available": False,
        "note": "The active llm.py does not expose shared usage tracking.",
    }


def write_usage(path: str | Path) -> dict[str, Any]:
    import llm
    fn = getattr(llm, "write_usage_summary", None)
    if fn:
        return fn(path)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = get_usage()
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
