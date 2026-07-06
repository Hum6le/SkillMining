"""
LLM API utilities for Skill Mining project.
Supports DeepSeek (V4 Flash) and Qwen workflow APIs.

Environment variables:
  DEEPSEEK_API_KEY   DeepSeek API key (required)
  DEEPSEEK_MODEL     Model name (default: deepseek-chat)
  QWEN_WORKFLOW_ID   Qwen workflow ID (for run_flow_retry)
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict

QWEN_WORKFLOW_ID = os.environ.get("QWEN_WORKFLOW_ID", "")

# DeepSeek configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


def _call_deepseek_api(prompt: str) -> Dict[str, Any]:
    """Single call to DeepSeek chat completion API (OpenAI-compatible)."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY environment variable not set. "
            "Set it via: $env:DEEPSEEK_API_KEY='sk-...'"
        )

    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 8192,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"DeepSeek API HTTP {e.code}: {body_text[:500]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"DeepSeek API connection error: {e}") from e

    content = result["choices"][0]["message"]["content"]
    # Return format compatible with extract_workflow_text in caller modules
    return {"data": {"text": content.strip()}}


def ds_api_retry(prompt: str) -> Dict[str, Any]:
    """Call DeepSeek API with exponential-backoff retry.

    Returns dict compatible with extract_workflow_text: {"data": {"text": "..."}}
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return _call_deepseek_api(prompt)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (attempt + 1)
                print(f"  DeepSeek API 第 {attempt + 1} 次失败，{delay}s 后重试: {e}")
                time.sleep(delay)
    raise RuntimeError(
        f"DeepSeek API failed after {MAX_RETRIES} retries: {last_error}"
    )


def _call_qwen_workflow(workflow_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Call Qwen workflow API."""
    raise NotImplementedError(
        "Qwen workflow API not configured. Use deepseek API instead "
        "(set environment variable LLM_API=deepseek)."
    )


def run_flow_retry(workflow_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Call Qwen workflow with retry logic.

    Returns dict compatible with extract_workflow_text: {"data": {"text": "..."}}
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return _call_qwen_workflow(workflow_id, params)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (attempt + 1)
                print(f"  Qwen workflow 第 {attempt + 1} 次失败，{delay}s 后重试: {e}")
                time.sleep(delay)
    raise RuntimeError(
        f"Qwen workflow failed after {MAX_RETRIES} retries: {last_error}"
    )
