"""SGD request accounting, independent of which server llm.py is installed.

Native provider counters are preferred. A missing tracker is NOT zero usage:
each wrapper attempt then gets an explicitly estimated token count. Estimates,
especially failed requests, are not provider billing figures.
"""
from __future__ import annotations

import json
import os
import logging
import time
from uuid import uuid4
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from scripts.llm_usage_utils import merge_usage_summaries

_NUMERIC = ("calls", "successful_calls", "failed_calls", "calls_with_usage",
            "prompt_tokens", "completion_tokens", "total_tokens", "exact_calls",
            "estimated_calls")
_PHASE = ContextVar("sgd_usage_phase", default="testing")
@dataclass
class _UsageState:
    summaries: dict = field(default_factory=dict)
    directory: Path | None = None
    incomplete: set = field(default_factory=set)
    warnings: list = field(default_factory=list)


_DEFAULT_STATE = _UsageState()
_CASE_STATE = ContextVar('sgd_case_usage', default=None)


def _state():
    return _CASE_STATE.get() or _DEFAULT_STATE


@contextmanager
def case_usage_scope():
    token = _CASE_STATE.set(_UsageState())
    phase = _PHASE.set('testing')
    try:
        yield
    finally:
        _PHASE.reset(phase)
        _CASE_STATE.reset(token)


def configure_usage(directory: Path) -> None:
    state = _state()
    state.summaries.clear()
    state.incomplete.clear()
    state.warnings.clear()
    state.directory = directory
    directory.mkdir(parents=True, exist_ok=True)
    if _CASE_STATE.get() is None and os.getenv("SKILLMINING_LLM_USAGE_FILE"):
        # Do not let inherited provider auto-flush overwrite our phase-split
        # report or a sibling process's file.
        os.environ["SKILLMINING_LLM_USAGE_FILE"] = str((directory / "native_llm_usage.json").resolve())
    existing = directory / "llm_usage.json"
    if existing.is_file():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        state.incomplete.update(previous.get("incomplete_phases", []))
        state.summaries.update(previous.get("by_phase", {}))
        if not state.summaries and previous.get("total"):
            value = previous["total"]
            state.summaries["previous_run_unclassified"] = value if "total" in value else {"total": value}
    set_phase("testing")
    persist_usage()


def set_phase(phase: str) -> None:
    _PHASE.set(phase)


def reset_case_usage() -> None:
    """Child-only reset: native deltas still measure each individual call."""
    _state().summaries.clear()
    _state().incomplete.clear()


def merge_worker_usage(payload: dict) -> None:
    for phase, snapshot in payload.get("by_phase", {}).items():
        _state().summaries[phase] = merge_usage_summaries(
            _state().summaries.get(phase, {"total": {}}), snapshot)
    _state().incomplete.update(payload.get("incomplete_phases", []))
    persist_usage()


def mark_incomplete_usage(phase: str) -> None:
    _state().incomplete.add(phase)
    persist_usage()


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
    summary = _state().summaries.get(phase, {"total": {}})
    summary = merge_usage_summaries(summary, {"total": delta, "by_call_tag": {tag: delta}})
    _state().summaries[phase] = summary
    if _state().directory is not None:
        with (_state().directory / "llm_requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        persist_usage()


def usage_payload() -> dict:
    generation = merge_usage_summaries({"total": {}}, *(
        summary for phase, summary in _state().summaries.items() if phase != "testing"))
    testing = merge_usage_summaries({"total": {}}, _state().summaries.get("testing", {}))
    return {
        "schema_version": 3,
        "note": "Native counts where available; fallback tokens are estimates, not billed usage. "
                "Failed-request tokens may be unknown. Loaded resources exclude prior-run mining cost.",
        "generation": generation, "testing": testing,
        "total": merge_usage_summaries({"total": {}}, *_state().summaries.values()),
        "by_phase": dict(sorted(_state().summaries.items())),
        "incomplete_phases": sorted(_state().incomplete),
        "accounting_complete": not _state().incomplete,
        "persistence_warnings": list(_state().warnings),
    }


def persist_usage() -> dict:
    payload = usage_payload()
    if _state().directory is not None:
        path = _state().directory / "llm_usage.json"
        temporary = path.with_name(f'.llm_usage.{os.getpid()}.{uuid4().hex}.tmp')
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for attempt in range(6):
                try:
                    temporary.replace(path)
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(.025 * (attempt+1))
        except OSError as exc:
            # Accounting remains in memory and is flushed on the next call.
            # A reader/antivirus briefly locking this file must not discard an
            # otherwise valid model prediction or trigger paid model retries.
            message = f'{type(exc).__name__}: usage file persistence failed'
            _state().warnings.append(message)
            logging.getLogger(__name__).warning(message)
            payload['persistence_warnings'] = list(_state().warnings)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return payload
