"""SGD request accounting, independent of which server llm.py is installed.

Native provider counters are preferred. A missing tracker is NOT zero usage:
each wrapper attempt then gets an explicitly estimated token count. Estimates,
especially failed requests, are not provider billing figures.
"""
from __future__ import annotations

import json
import os
from contextvars import ContextVar
from pathlib import Path

from scripts.llm_usage_utils import merge_usage_summaries

_NUMERIC = ("calls", "successful_calls", "failed_calls", "calls_with_usage",
            "prompt_tokens", "completion_tokens", "total_tokens", "exact_calls",
            "estimated_calls")
_PHASE = ContextVar("sgd_usage_phase", default="testing")
_PHASE_SUMMARIES: dict[str, dict] = {}
_DIRECTORY: Path | None = None


def configure_usage(directory: Path) -> None:
    global _DIRECTORY
    _PHASE_SUMMARIES.clear()
    _DIRECTORY = directory
    directory.mkdir(parents=True, exist_ok=True)
    if os.getenv("SKILLMINING_LLM_USAGE_FILE"):
        # Do not let inherited provider auto-flush overwrite our phase-split
        # report or a sibling process's file.
        os.environ["SKILLMINING_LLM_USAGE_FILE"] = str((directory / "native_llm_usage.json").resolve())
    existing = directory / "llm_usage.json"
    if existing.is_file():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        _PHASE_SUMMARIES.update(previous.get("by_phase", {}))
        if not _PHASE_SUMMARIES and previous.get("total"):
            value = previous["total"]
            _PHASE_SUMMARIES["previous_run_unclassified"] = value if "total" in value else {"total": value}
    set_phase("testing")
    persist_usage()


def set_phase(phase: str) -> None:
    _PHASE.set(phase)


def native_snapshot() -> dict:
    import llm
    fn = getattr(llm, "get_usage_summary", None)
    if not callable(fn):
        return {}
    value = fn()
    return dict(value.get("total", {})) if isinstance(value, dict) else {}


def record_attempt(before: dict, messages, reply: str, *, tag: str, success: bool) -> None:
    after = native_snapshot()
    delta = {key: max(0, int(after.get(key, 0)) - int(before.get(key, 0))) for key in _NUMERIC}
    content = messages if isinstance(messages, str) else json.dumps(messages, ensure_ascii=False)
    prompt_tokens = (len(content) + 3) // 4
    completion_tokens = (len(reply) + 3) // 4
    if not delta["calls"]:
        delta.update(calls=1, successful_calls=int(success), failed_calls=int(not success),
                     calls_with_usage=1, prompt_tokens=prompt_tokens,
                     completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens,
                     exact_calls=0, estimated_calls=1)
        source = "estimated"
        accounting = "sgd_wrapper_fallback"
    else:
        missing = max(0, delta["calls"] - delta["calls_with_usage"])
        if missing:
            # A workflow tracker may count calls but omit all token metadata.
            # Preserve real request counts, estimate ONLY missing usage, and
            # count the final reply at most once when no native reply exists.
            output_estimate = completion_tokens if not delta["calls_with_usage"] else 0
            delta["prompt_tokens"] += prompt_tokens * missing
            delta["completion_tokens"] += output_estimate
            delta["total_tokens"] += prompt_tokens * missing + output_estimate
            delta["calls_with_usage"] += missing
            delta["estimated_calls"] += missing
        source = ("mixed" if delta["exact_calls"] and delta["estimated_calls"] else
                  "exact" if delta["exact_calls"] else
                  "estimated" if delta["estimated_calls"] else "unavailable")
        accounting = "native_calls_estimated_missing_tokens" if missing else "native_tracker_delta"
    delta.update(usage_source=source, usage_available=bool(delta["calls_with_usage"]))
    row = {"phase": _PHASE.get(), "call_tag": tag, "accounting": accounting, "total": delta}
    phase = row["phase"]
    summary = _PHASE_SUMMARIES.get(phase, {"total": {}})
    summary = merge_usage_summaries(summary, {"total": delta, "by_call_tag": {tag: delta}})
    _PHASE_SUMMARIES[phase] = summary
    if _DIRECTORY is not None:
        with (_DIRECTORY / "llm_requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        persist_usage()


def usage_payload() -> dict:
    generation = merge_usage_summaries({"total": {}}, *(
        summary for phase, summary in _PHASE_SUMMARIES.items() if phase != "testing"))
    testing = merge_usage_summaries({"total": {}}, _PHASE_SUMMARIES.get("testing", {}))
    return {
        "schema_version": 3,
        "note": "Native counts where available; fallback tokens are estimates, not billed usage. "
                "Failed-request tokens may be unknown. Loaded resources exclude prior-run mining cost.",
        "generation": generation, "testing": testing,
        "total": merge_usage_summaries({"total": {}}, *_PHASE_SUMMARIES.values()),
        "by_phase": dict(sorted(_PHASE_SUMMARIES.items())),
    }


def persist_usage() -> dict:
    payload = usage_payload()
    if _DIRECTORY is not None:
        path = _DIRECTORY / "llm_usage.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    return payload
