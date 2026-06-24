"""Raw LLM response logger.

Saves every LLM prompt and its raw response to timestamped JSON files
so that model outputs can be inspected, replayed, or debugged later.

Usage::

    from eval_tod.response_logger import ResponseLogger

    logger = ResponseLogger("outputs/logs/llm_calls")
    logger.log(messages=[...], response=raw_response, call_tag="agent_turn_3")
    # -> writes llm_calls/0001_agent_turn_3_prompt.json
    #             llm_calls/0001_agent_turn_3_response.json
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class ResponseLogger:
    """Thread-safe logger for raw LLM prompts and responses.

    Each call to :meth:`log` writes two files: ``{counter:04d}_{tag}_prompt.json``
    and ``{counter:04d}_{tag}_response.json``. The counter is global across all
    threads using this logger instance.
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self._counter = 0
        self._lock = threading.Lock()

    def log(
        self,
        messages: list[dict[str, str]],
        response: Any,
        *,
        call_tag: str = "call",
        extra: dict | None = None,
    ) -> int:
        """Save one LLM interaction to disk.

        Args:
            messages: The prompt messages sent to the LLM (list of
                      ``{"role": ..., "content": ...}`` dicts).
            response: The raw response object (or dict) from the API.
            call_tag: Short label to identify this call in filenames
                      (e.g. ``"agent_turn_3"``, ``"error_analysis"``).
            extra: Optional extra metadata to include in the prompt file.

        Returns:
            The call index (counter value).
        """
        with self._lock:
            self._counter += 1
            idx = self._counter

        safe_tag = _sanitize_tag(call_tag)
        prefix = f"{idx:04d}_{safe_tag}"

        # Serialize messages (handle Message objects that have .role/.content)
        clean_messages = []
        for m in messages:
            if hasattr(m, "role") and hasattr(m, "content"):
                clean_messages.append({"role": m.role, "content": m.content})
            elif isinstance(m, dict):
                clean_messages.append({"role": m.get("role", ""), "content": m.get("content", "")})
            else:
                clean_messages.append(str(m))

        prompt_record = {
            "call_index": idx,
            "call_tag": call_tag,
            "timestamp": datetime.now().isoformat(),
            "num_messages": len(clean_messages),
            "messages": clean_messages,
        }
        if extra:
            prompt_record["extra"] = extra

        prompt_path = self.log_dir / f"{prefix}_prompt.json"
        prompt_path.write_text(
            json.dumps(prompt_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Serialize response — handle openai response objects
        response_record = _serialize_response(response)
        response_record["call_index"] = idx
        response_record["call_tag"] = call_tag
        response_record["timestamp"] = datetime.now().isoformat()

        resp_path = self.log_dir / f"{prefix}_response.json"
        resp_path.write_text(
            json.dumps(response_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return idx

    @property
    def count(self) -> int:
        return self._counter


def _sanitize_tag(tag: str) -> str:
    """Replace characters that are unsafe in filenames."""
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in tag)[:80]


def _serialize_response(response: Any) -> dict:
    """Convert an openai response object (or dict) to a JSON-safe dict."""
    if isinstance(response, dict):
        return dict(response)

    record: dict = {}

    # Try openai SDK response object
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:
            pass
    if hasattr(response, "dict"):
        try:
            return response.dict()
        except Exception:
            pass

    # Manual extraction for common openai response shape
    try:
        record["id"] = getattr(response, "id", "")
        record["model"] = getattr(response, "model", "")
        record["created"] = getattr(response, "created", None)
        record["object"] = getattr(response, "object", "")

        choices = getattr(response, "choices", [])
        serialized_choices = []
        for choice in choices:
            c = {"index": getattr(choice, "index", None),
                 "finish_reason": getattr(choice, "finish_reason", None)}
            message = getattr(choice, "message", None)
            if message:
                c["message"] = {
                    "role": getattr(message, "role", ""),
                    "content": getattr(message, "content", ""),
                }
                reasoning = getattr(message, "reasoning_content", None)
                if reasoning:
                    c["message"]["reasoning_content"] = reasoning
            serialized_choices.append(c)
        record["choices"] = serialized_choices

        usage = getattr(response, "usage", None)
        if usage:
            record["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
    except Exception:
        record["raw"] = str(response)[:1000]

    return record
