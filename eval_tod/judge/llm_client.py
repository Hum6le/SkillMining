"""LLM 客户端 —— OpenAI 兼容接口，JSON 结构化输出与重试。"""

from __future__ import annotations

import json
import os
import time
from typing import Any


_HAS_OPENAI = False
try:
    from openai import OpenAI  # type: ignore[import-untyped]

    _HAS_OPENAI = True
except ImportError:
    pass


def _get_client(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    """Create an OpenAI-compatible client.

    API credentials are resolved in order:
    1. Explicit ``api_key`` / ``base_url`` arguments
    2. Environment variables ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``
    """
    if not _HAS_OPENAI:
        raise RuntimeError("openai SDK not installed. Run: pip install openai")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "API key not found. Set OPENAI_API_KEY or pass api_key."
        )

    url = base_url or os.environ.get("OPENAI_BASE_URL", None)
    kwargs: dict[str, Any] = {"api_key": key}
    if url:
        kwargs["base_url"] = url
    return OpenAI(**kwargs)


def call_llm(
    system_prompt: str,
    user_message: str,
    model: str = "deepseek-chat",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """Single-turn LLM call, returns raw text response.

    Args:
        system_prompt: System-level instructions for the LLM.
        user_message: The user message / prompt body.
        model: Model name (OpenAI-compatible).
        max_tokens: Max generation tokens.
        temperature: Sampling temperature.
        api_key: Override API key.
        base_url: Override API base URL.
    """
    client = _get_client(model, api_key, base_url)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    return response.choices[0].message.content


def call_llm_structured(
    system_prompt: str,
    user_message: str,
    model: str = "deepseek-chat",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    """LLM call expecting JSON output. Retries once on parse failure.

    Returns:
        Parsed JSON dict.
    """
    for attempt in range(2):
        raw = call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )
        try:
            clean = raw.strip()
            # Strip markdown code fences if present
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()
            return json.loads(clean)
        except (json.JSONDecodeError, IndexError) as exc:
            if attempt == 0:
                user_message = (
                    f"{user_message}\n\n"
                    f"[IMPORTANT] Output ONLY valid JSON, no extra text. "
                    f"Previous response was not valid JSON: {exc}"
                )
            else:
                raise RuntimeError(
                    f"LLM failed to produce valid JSON twice. "
                    f"Raw response: {raw[:500]}"
                ) from exc

    # Unreachable — the loop always returns or raises
    return {}
