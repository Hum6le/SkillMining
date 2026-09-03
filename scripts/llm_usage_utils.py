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
