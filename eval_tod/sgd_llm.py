"""SGD-only request retries; leave the shared LLM client unchanged."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
import math
import os
import random
import re
import time


log = logging.getLogger(__name__)
_SELECTED_PAYLOAD_HEADS = ContextVar('sgd_selected_payload_heads', default=None)
_RETRYABLE_STATUS = {429, 502, 503, 504}
_WORKFLOW_HTTP_ERROR = re.compile(r"^Workflow HTTP error (\d{3}):")


@contextmanager
def selected_payload_heads(heads):
    """Scope an already selected action set to this synchronous payload call.

    None means the predictor must still make its own action decision. An empty
    action set is a real fixed decision. The scope never consults gold labels
    or parses dialogue/prompt text, and is reset even if grounding raises.
    """
    token = _SELECTED_PAYLOAD_HEADS.set(None if heads is None else tuple(tuple(head) for head in heads))
    try:
        yield
    finally:
        _SELECTED_PAYLOAD_HEADS.reset(token)


def skipped_payload_requests(heads):
    """Return requests whose payload is empty under the fixed action set."""
    if heads is None:
        return []
    skipped = []
    if not any(head[0] == 'call' for head in heads):
        skipped.append('sgd_call_selection')
    if not any(head[0] != 'call' for head in heads):
        skipped.append('sgd_dialogue_acts')
    return skipped


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
    Explicit HTTP 429/502/503/504 and the workflow-not-ready response are retried. Exhaustion
    re-raises the provider error. A swallowed failure (empty reply) also
    aborts instead of becoming an apparently valid empty policy.
    """
    # With a fixed action set, absent action kinds have no payload to bind.
    # Return the deterministic empty payload before touching the provider or
    # usage tracker. The caller still applies its original parsing and API
    # observation gate. Baselines do not enter this explicit scope.
    skipped = skipped_payload_requests(_SELECTED_PAYLOAD_HEADS.get())
    tag = kwargs.get('call_tag', 'chat')
    if tag in skipped:
        return '{"call":null}' if tag == 'sgd_call_selection' else '{"acts":[]}'
    from llm import chat
    from eval_tod.sgd_usage import native_snapshot, record_attempt

    attempts, base, cap = _retry_settings()
    delay = min(base, cap)
    for attempt in range(1, attempts + 1):
        before = native_snapshot()
        try:
            reply = chat(messages, **kwargs)
        except Exception as exc:
            record_attempt(before, messages, "", tag=kwargs.get("call_tag", "chat"), success=False)
            status = _http_status(exc)
            not_ready = str(exc).startswith("Workflow returned ret=1:") and "workflow not ready to run" in str(exc).lower()
            if (status not in _RETRYABLE_STATUS and not not_ready) or attempt == attempts:
                raise
            wait = min(cap, delay * random.uniform(1.0, 1.25))
            log.warning(
                "SGD LLM retry: tag=%s workflow=%s HTTP=%s next_attempt=%s/%s wait=%.2fs",
                kwargs.get("call_tag", "chat"),
                os.getenv("SKILLMINING_WORKFLOW_ID", "<configured>"),
                status if status is not None else "workflow_not_ready", attempt + 1, attempts, wait,
            )
            time.sleep(wait)
            delay = min(cap, delay * 2)
            continue
        record_attempt(before, messages, reply if isinstance(reply, str) else "",
                       tag=kwargs.get("call_tag", "chat"), success=isinstance(reply, str) and bool(reply.strip()))
        if not isinstance(reply, str) or not reply.strip():
            if attempt < attempts and os.getenv("SKILLMINING_RETRY_EMPTY_LLM") == "1":
                wait = min(cap, delay * random.uniform(1.0, 1.25))
                log.warning("SGD LLM returned empty/non-text response: tag=%s retry=%s/%s wait=%.2fs",
                            kwargs.get("call_tag", "chat"), attempt + 1, attempts, wait)
                time.sleep(wait)
                delay = min(cap, delay * 2)
                continue
            raise RuntimeError("SGD LLM returned an empty or non-text response "
                               f"(tag={kwargs.get('call_tag', 'chat')}); aborting instead of scoring a failed request")
        return reply
    raise AssertionError("Unreachable: retry loop must return or raise")
