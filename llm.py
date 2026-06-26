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
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_TRACE2SKILL = Path(__file__).resolve().parent / "Trace2Skill"
_CLIENT_CACHE: dict[str, object] = {}


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
        max_tokens: Max tokens in response (None = model default).
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
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens
    request_kwargs.update(kwargs)

    # Call
    try:
        resp = client.chat.completions.create(**request_kwargs)

        # Log raw response if logger configured
        if response_logger is not None:
            try:
                response_logger.log(messages=clean, response=resp, call_tag="chat")
            except Exception:
                pass

        return resp.choices[0].message.content or ""
    except Exception as exc:
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
