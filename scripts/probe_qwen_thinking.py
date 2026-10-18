#!/usr/bin/env python3
"""Compare gateway usage for one tiny request with and without thinking."""

from pathlib import Path
from time import monotonic

from dotenv import dotenv_values
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
values = dotenv_values(ROOT / "third_party/agent-workflow-memory/mind2web/.env")
client = OpenAI(api_key=values["SHUBIAOBIAO_API_KEY"],
                base_url=values["MIND2WEB_API_BASE_URL"], timeout=90, max_retries=0)
for label, extra in (("default", None), ("thinking_false", {"enable_thinking": False})):
    kwargs = {"model": values["MIND2WEB_MODEL"], "temperature": 0,
              "messages": [{"role": "user", "content": 'Return only this JSON: {"ok":true}'}]}
    if extra is not None:
        kwargs["extra_body"] = extra
    started = monotonic()
    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    usage = response.usage
    print({"mode": label, "seconds": round(monotonic() - started, 2),
           "prompt_tokens": usage.prompt_tokens if usage else None,
           "completion_tokens": usage.completion_tokens if usage else None,
           "content_chars": len(message.content or ""),
           "reasoning_chars": len(getattr(message, "reasoning_content", None) or "")},
          flush=True)
