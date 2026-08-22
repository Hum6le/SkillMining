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

