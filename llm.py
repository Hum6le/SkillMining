"""LLM client factory — single import for all pipeline components.

Usage:
    from llm import get_client, resolve_config

    client = get_client("deepseek-chat")
    # or with explicit config:
    client = get_client("deepseek-chat", api_key="sk-...", base_url="https://...")
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_TRACE2SKILL = Path(__file__).resolve().parent / "Trace2Skill"
_CLIENT_CACHE: dict[str, object] = {}


# ── Config resolution ─────────────────────────────────────────


def resolve_config(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "deepseek-chat",
) -> dict[str, str]:
    """Resolve effective API configuration.

    Priority: explicit args > AWM config > environment variables.

    Returns dict with keys: model, api_key, base_url.
    """
    if not api_key:
        api_key = _try_awm_config("DEEPSEEK_API_KEY")
    if not base_url:
        base_url = _try_awm_config("DEEPSEEK_BASE_URL")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "")

    if not api_key or not base_url:
        raise RuntimeError(
            "LLM API not configured. Provide api_key/base_url, set "
            "OPENAI_API_KEY/OPENAI_BASE_URL env vars, or clone "
            "https://github.com/zorazrw/agent-workflow-memory.git as AWM/"
        )

    return {"model": model, "api_key": api_key, "base_url": base_url}


def _try_awm_config(key: str) -> str | None:
    """Try to read a config value from AWM/config.py."""
    try:
        awm_path = str(_TRACE2SKILL.parent / "AWM")
        if awm_path not in sys.path:
            sys.path.insert(0, awm_path)
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
        mapping = {"DEEPSEEK_API_KEY": DEEPSEEK_API_KEY, "DEEPSEEK_BASE_URL": DEEPSEEK_BASE_URL}
        return mapping.get(key)
    except ImportError:
        return None


# ── Client factory ────────────────────────────────────────────


def get_client(
    model: str = "deepseek-chat",
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    cache: bool = True,
    cache_tag: str = "",
    **kwargs,
):
    """Get (or reuse) an OpenAI-compatible LLM client.

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


def clear_cache() -> None:
    """Clear the client cache (useful for testing)."""
    _CLIENT_CACHE.clear()
