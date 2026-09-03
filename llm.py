"""LLM client — single import, single call.

Usage:
    from llm import chat

    reply = chat("What is the capital of France?")
    reply = chat([{"role": "user", "content": "Hello"}])
    reply = chat([...], model="deepseek-chat", temperature=0.0)

Every LLM call in the project goes through ``chat()``.  There is no
other public API — just ``prompt in, response out``.
"""

from __future__ import annotations

import logging
import json
import os
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_TRACE2SKILL = Path(__file__).resolve().parent / "Trace2Skill"
_CLIENT_CACHE: dict[str, object] = {}


# The server-side replacement has a richer tracker. Keep the local client
# compatible as well, so standalone graph/online runs never silently report
# zero calls. Workflow providers often omit usage, hence the deterministic
# character-based estimate below.
_USAGE_LOCK = threading.Lock()
_USAGE_STARTED_AT = datetime.now(timezone.utc).isoformat()
_USAGE_CALLS: list[dict] = []


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def _record_usage(messages: list[dict], response: str, *, model: str,
                  call_tag: str, success: bool, usage: dict | None = None,
                  error_type: str | None = None) -> None:
    prompt = "\n".join(str(m.get("content", "")) for m in messages)
    token_usage = usage or {
        "prompt_tokens": _estimate_tokens(prompt),
        "completion_tokens": _estimate_tokens(response),
        "total_tokens": _estimate_tokens(prompt) + _estimate_tokens(response),
    }
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "call_tag": call_tag,
        "provider": "openai_compatible",
        "model": model,
        "success": bool(success),
        "usage_available": True,
        "usage_source": "exact" if usage else "estimated",
        "usage": token_usage,
    }
    if error_type:
        row["error_type"] = error_type
    with _USAGE_LOCK:
        _USAGE_CALLS.append(row)


def get_usage_summary() -> dict:
    with _USAGE_LOCK:
        calls = list(_USAGE_CALLS)
        started_at = _USAGE_STARTED_AT
    def bucket(rows):
        out = {"calls": 0, "successful_calls": 0, "failed_calls": 0,
               "calls_with_usage": 0, "prompt_tokens": 0,
               "completion_tokens": 0, "total_tokens": 0,
               "usage_available": False, "exact_calls": 0,
               "estimated_calls": 0, "usage_source": "unavailable"}
        for row in rows:
            out["calls"] += 1
            out["successful_calls"] += int(row["success"])
            out["failed_calls"] += int(not row["success"])
            u = row.get("usage")
            if u:
                out["calls_with_usage"] += 1
                out["usage_available"] = True
                source = row.get("usage_source")
                out["estimated_calls"] += int(source == "estimated")
                out["exact_calls"] += int(source == "exact")
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    out[key] += int(u.get(key, 0) or 0)
        if out["exact_calls"] and out["estimated_calls"]:
            out["usage_source"] = "mixed"
        elif out["exact_calls"]:
            out["usage_source"] = "exact"
        elif out["estimated_calls"]:
            out["usage_source"] = "estimated"
        return out
    by_tag = defaultdict(list)
    by_provider = defaultdict(list)
    for row in calls:
        by_tag[row["call_tag"]].append(row)
        by_provider[row["provider"]].append(row)
    return {"schema_version": 1, "started_at": started_at,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": bucket(calls),
            "by_call_tag": {k: bucket(v) for k, v in sorted(by_tag.items())},
            "by_provider": {k: bucket(v) for k, v in sorted(by_provider.items())}}


def reset_usage_summary() -> None:
    global _USAGE_STARTED_AT
    with _USAGE_LOCK:
        _USAGE_STARTED_AT = datetime.now(timezone.utc).isoformat()
        _USAGE_CALLS.clear()


def write_usage_summary(path: str | Path) -> dict:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = get_usage_summary()
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# ══════════════════════════════════════════════════════════════════
# Config resolution
# ══════════════════════════════════════════════════════════════════

def resolve_config(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "deepseek-chat",
) -> dict[str, str]:
    """Resolve effective API configuration.

    Priority: explicit args > environment variables.
    Also tries AWM/config.py and .env for backward compat.

    Returns dict with keys: model, api_key, base_url.
    """
    if not api_key:
        api_key = _try_local_config("DEEPSEEK_API_KEY")
    if not base_url:
        base_url = _try_local_config("DEEPSEEK_BASE_URL")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "")

    if not api_key or not base_url:
        raise RuntimeError(
            "LLM API not configured. Set OPENAI_API_KEY / OPENAI_BASE_URL "
            "environment variables, or pass api_key/base_url explicitly."
        )

    return {"model": model, "api_key": api_key, "base_url": base_url}


# ══════════════════════════════════════════════════════════════════
# Core API: prompt in, response out
# ══════════════════════════════════════════════════════════════════

def chat(
    messages: str | list[dict],
    *,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    response_logger=None,
    call_tag: str = "chat",
    **kwargs,
) -> str:
    """Send messages to the LLM and return the response text.

    The single entry point for all LLM calls in the project.
    Handles config resolution, client creation, and error recovery.

    Args:
        messages: Either a string (auto-wrapped as a user message) or a list
                  of ``{"role": "...", "content": "..."}`` dicts.
        model: Model name (default ``"deepseek-chat"``).
        temperature: Sampling temperature.
        max_tokens: Deprecated compatibility argument. Generation length is
                    intentionally left unrestricted by this wrapper.
        api_key: API key (resolved from env if None).
        base_url: API base URL (resolved from env if None).
        response_logger: Optional ``ResponseLogger`` to record raw I/O.
        **kwargs: Extra args passed to the API (e.g. ``stop=["Task:"]``).

    Returns:
        The response text string.  Empty string on failure.
    """
    from openai import OpenAI

    # Normalize messages
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    # Handle Message objects from Trace2Skill
    clean = []
    for m in messages:
        if hasattr(m, "role") and hasattr(m, "content"):
            clean.append({"role": m.role, "content": m.content})
        elif isinstance(m, dict):
            clean.append(m)
        else:
            clean.append({"role": "user", "content": str(m)})

    # Resolve config
    cfg = resolve_config(api_key=api_key, base_url=base_url, model=model)

    # Create client
    client_kwargs = {"api_key": cfg["api_key"]}
    if cfg["base_url"]:
        client_kwargs["base_url"] = cfg["base_url"]
    client = OpenAI(**client_kwargs)

    # Build request
    request_kwargs: dict = {
        "model": cfg["model"],
        "messages": clean,
        "temperature": temperature,
    }
    request_kwargs.update(kwargs)
    request_kwargs.pop("max_tokens", None)
    request_kwargs.pop("max_completion_tokens", None)

    # Call
    try:
        resp = client.chat.completions.create(**request_kwargs)

        # Log raw response if logger configured
        if response_logger is not None:
            try:
                response_logger.log(messages=clean, response=resp, call_tag=call_tag)
            except Exception as e:
                log.warning(f"Response logger failed: {e}")

        response_text = resp.choices[0].message.content or ""
        provider_usage = getattr(resp, "usage", None)
        if hasattr(provider_usage, "model_dump"):
            provider_usage = provider_usage.model_dump()
        _record_usage(clean, response_text, model=cfg["model"], call_tag=call_tag,
                      success=True, usage=provider_usage if isinstance(provider_usage, dict) else None)
        return response_text
    except Exception as exc:
        _record_usage(clean, "", model=cfg["model"], call_tag=call_tag,
                      success=False, error_type=type(exc).__name__)
        log.warning(f"LLM call failed: {exc}")
        return ""


# ══════════════════════════════════════════════════════════════════
# Internal: OpenAIClient factory (needed by Trace2Skill evolver)
# ══════════════════════════════════════════════════════════════════

def _get_client(
    model: str = "deepseek-chat",
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    cache: bool = True,
    cache_tag: str = "",
    **kwargs,
):
    """Get (or reuse) an OpenAIClient with caching and retry logic.

    Used by the skill evolver (Trace2Skill) which needs disk caching
    and token-aware retry.  For simple LLM calls, use ``chat()`` instead.

    Args:
        model: Model name.
        api_key: API key. If None, resolved from config/env.
        base_url: API base URL. If None, resolved from config/env.
        cache: If True, reuse cached client for same model+url+key.
        cache_tag: Optional tag to separate caches (e.g. "map", "merge").
        **kwargs: Passed through to OpenAIClient.

    Returns:
        OpenAIClient instance.
    """
    config = resolve_config(api_key=api_key, base_url=base_url, model=model)

    cache_key = f"{config['model']}:{config['api_key']}:{config['base_url']}:{cache_tag}"
    if cache and cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]

    sys.path.insert(0, str(_TRACE2SKILL))
    from src.react_agent.models import OpenAIClient

    client = OpenAIClient(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        **kwargs,
    )

    if cache:
        _CLIENT_CACHE[cache_key] = client
    return client


def _try_local_config(key: str) -> str | None:
    """Try to read config from local config.py files."""
    from pathlib import Path
    for config_dir in ["awm", "AWM"]:
        config_path = Path(__file__).resolve().parent / config_dir / "config.py"
        if config_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(f"_{config_dir}_config", str(config_path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return getattr(mod, key, None)
            except Exception:
                pass
    return None


def clear_cache() -> None:
    """Clear the client cache (useful for testing)."""
    _CLIENT_CACHE.clear()
