"""LLM client for judge — delegates to llm.chat()."""

from __future__ import annotations

import json
from typing import Any


def call_llm(
    system_prompt: str,
    user_message: str,
    model: str = "deepseek-chat",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """Single-turn LLM call, returns raw text response."""
    from llm import chat

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    return chat(
        messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def call_llm_structured(
    system_prompt: str,
    user_message: str,
    model: str = "deepseek-chat",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    """LLM call expecting JSON output. Retries once on parse failure."""
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
    return {}
