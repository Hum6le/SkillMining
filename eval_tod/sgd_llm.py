"""SGD-only request retries; leave the shared LLM client unchanged."""

from __future__ import annotations

import logging
import math
import os
import random
import re
import time


log = logging.getLogger(__name__)
_RETRYABLE_STATUS = {429, 502, 503, 504}
_WORKFLOW_HTTP_ERROR = re.compile(r"^Workflow HTTP error (\d{3}):")


def _http_status(error: Exception) -> int | None:
    """Recover a status from an HTTP exception or the Workflow wrapper.

    The server wraps urllib.HTTPError with ``raise RuntimeError(...) from
    exc``. Some deployments preserve only its message, so recognize that
    specific format too; an arbitrary occurrence of '502' is not sufficient.
    """
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        response = getattr(current, "response", None)
        for value in (
            getattr(current, "code", None),
            getattr(current, "status_code", None),
            getattr(response, "status_code", None),
        ):
            if isinstance(value, int) and 100 <= value <= 599:
                return value
        match = _WORKFLOW_HTTP_ERROR.match(str(current))
        if match:
            return int(match.group(1))
        current = current.__cause__ or current.__context__
    return None


def _retry_settings() -> tuple[int, float, float]:
    attempts = int(os.getenv("SKILLMINING_SGD_LLM_MAX_ATTEMPTS", "5"))
    base = float(os.getenv("SKILLMINING_SGD_LLM_RETRY_BASE_SECONDS", "2"))
    cap = float(os.getenv("SKILLMINING_SGD_LLM_RETRY_MAX_SECONDS", "30"))
    if attempts < 1 or not math.isfinite(base) or not math.isfinite(cap) or base < 0 or cap < 0:
        raise ValueError("SGD retry attempts must be positive; delays must be finite and nonnegative")
    return attempts, base, cap


def sgd_chat_with_retry(messages, **kwargs) -> str:
    """Retry only one LLM request, never a rollout or resource update.

    Defaults: five total attempts, capped exponential backoff with jitter.
    Only explicit HTTP 429/502/503/504 failures are retried. Exhaustion
    re-raises the provider error. A swallowed failure (empty reply) also
    aborts instead of becoming an apparently valid empty policy.
    """
    from llm import chat

    attempts, base, cap = _retry_settings()
    delay = min(base, cap)
    for attempt in range(1, attempts + 1):
        try:
            reply = chat(messages, **kwargs)
        except Exception as exc:
            status = _http_status(exc)
            if status not in _RETRYABLE_STATUS or attempt == attempts:
                raise
            wait = min(cap, delay * random.uniform(1.0, 1.25))
            log.warning(
                "SGD LLM retry: tag=%s workflow=%s HTTP=%s next_attempt=%s/%s wait=%.2fs",
                kwargs.get("call_tag", "chat"),
                os.getenv("SKILLMINING_WORKFLOW_ID", "<configured>"),
                status, attempt + 1, attempts, wait,
            )
            time.sleep(wait)
            delay = min(cap, delay * 2)
            continue
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError(
                "SGD LLM returned an empty or non-text response "
                f"(tag={kwargs.get('call_tag', 'chat')}); aborting instead of scoring a failed request"
            )
        return reply
    raise AssertionError("Unreachable: retry loop must return or raise")
