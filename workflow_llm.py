"""LLM client — single import, single call.

Usage:
    from llm import chat, resolve_config

    reply = chat([{"role": "user", "content": "Hello"}])
    reply = chat("What is the capital of France?")
    reply = chat([...], model="qwenA3B", temperature=0.0)

All LLM calls in the project should go through ``chat()``.

Workflow mode (``config.py`` ``provider: workflow``):
    messages are split into ``system_prompt`` + ``user_prompt`` and sent via
    ``utils.llm_utils.run_flow_retry`` (POST workflow_url with parameters).

Skill evolver (ParallelSkillEvolver) needs a client object — use
``workflow_llm_client.make_evolver_client()`` instead.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from time import sleep as _workflow_retry_sleep

log = logging.getLogger(__name__)

_SKILLMINING = Path(__file__).resolve().parent
_CLIENT_CACHE: dict[str, Any] = {}
_WORKFLOW_RATE_LIMIT_RETRIES = (0, 2, 5, 10, 20, 30)


class LLMUsageTracker:
    """Process-local, provider-agnostic accounting for every ``chat`` call.

    The tracker records all attempted calls. Token figures are recorded only
    when the provider returns an OpenAI-compatible ``usage`` object; workflow
    providers commonly do not expose this information and are explicitly
    represented as calls without usage rather than being heuristically
    estimated.  This keeps cost reports comparable and auditable.

    A runner may persist the snapshot with :func:`write_usage_summary`, or set
    ``SKILLMINING_LLM_USAGE_FILE`` to make the module write a refreshed JSON
    summary after every call. The latter is useful without changing callers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._calls: list[dict[str, Any]] = []

    def record(
        self,
        *,
        call_tag: str,
        provider: str,
        model: str | None,
        success: bool,
        usage: dict[str, int] | None = None,
        usage_source: str = "exact",
        error_type: str | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_tag": call_tag,
            "provider": provider,
            "model": model or "",
            "success": bool(success),
            "usage_available": usage is not None,
            "usage_source": usage_source if usage is not None else "unavailable",
        }
        if usage is not None:
            row["usage"] = usage
        if error_type:
            row["error_type"] = error_type
        with self._lock:
            self._calls.append(row)

    def reset(self) -> None:
        with self._lock:
            self._started_at = datetime.now(timezone.utc).isoformat()
            self._calls.clear()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            calls = list(self._calls)
            started_at = self._started_at

        total = _empty_usage_bucket()
        by_tag: dict[str, dict[str, Any]] = defaultdict(_empty_usage_bucket)
        by_provider: dict[str, dict[str, Any]] = defaultdict(_empty_usage_bucket)
        for row in calls:
            _accumulate_usage_bucket(total, row)
            _accumulate_usage_bucket(by_tag[row["call_tag"]], row)
            _accumulate_usage_bucket(by_provider[row["provider"]], row)
        return {
            "schema_version": 1,
            "started_at": started_at,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": total,
            "by_call_tag": dict(sorted(by_tag.items())),
            "by_provider": dict(sorted(by_provider.items())),
        }


def _empty_usage_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "calls_with_usage": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "usage_available": False,
        "exact_calls": 0,
        "estimated_calls": 0,
        "usage_source": "unavailable",
    }


def _accumulate_usage_bucket(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    bucket["calls"] += 1
    bucket["successful_calls"] += int(row["success"])
    bucket["failed_calls"] += int(not row["success"])
    usage = row.get("usage")
    if isinstance(usage, dict):
        bucket["calls_with_usage"] += 1
        bucket["usage_available"] = True
        if row.get("usage_source") == "estimated":
            bucket["estimated_calls"] += 1
        else:
            bucket["exact_calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            bucket[key] += int(usage.get(key, 0) or 0)
        bucket["usage_source"] = (
            "estimated" if bucket["estimated_calls"] and not bucket["exact_calls"]
            else "mixed" if bucket["estimated_calls"] else "exact"
        )


_USAGE_TRACKER = LLMUsageTracker()
_SCOPED_USAGE = ContextVar('workflow_native_usage', default=None)


def _usage_tracker():
    return _SCOPED_USAGE.get() or _USAGE_TRACKER


@contextmanager
def usage_scope():
    """Isolate one threaded case's native counters from concurrent requests."""
    token = _SCOPED_USAGE.set(LLMUsageTracker())
    try:
        yield
    finally:
        _SCOPED_USAGE.reset(token)


def get_usage_summary() -> dict[str, Any]:
    """Return usage across every ``chat`` call in the current process."""
    return _usage_tracker().summary()


def reset_usage_summary() -> None:
    """Clear process-local usage, normally once at the beginning of a run."""
    _usage_tracker().reset()


def write_usage_summary(path: str | Path) -> dict[str, Any]:
    """Persist the current usage snapshot and return the same payload."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = get_usage_summary()
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _flush_usage_if_configured() -> None:
    if _SCOPED_USAGE.get() is not None:
        # Threaded SGD writes a separate per-case report, never a shared file.
        return
    output = os.getenv("SKILLMINING_LLM_USAGE_FILE", "").strip()
    if output:
        try:
            write_usage_summary(output)
        except Exception as exc:
            log.warning("Could not persist LLM usage summary to %s: %s", output, exc)


def _instrument_openai_client(client: Any, *, model: str, call_tag: str) -> Any:
    """Instrument legacy Trace2Skill clients that bypass :func:`chat`.

    ``OpenAIClient`` owns an OpenAI SDK client internally. Wrapping its
    completion endpoint captures real provider requests (including retries)
    without changing the evolver's public client interface. Cache hits make no
    provider request and are intentionally not charged as LLM calls.
    """
    completion_api = client._client.chat.completions
    original_create = completion_api.create

    def tracked_create(*args, **kwargs):
        try:
            response = original_create(*args, **kwargs)
        except Exception as exc:
            _usage_tracker().record(
                call_tag=call_tag,
                provider="openai_compatible",
                model=model,
                success=False,
                error_type=type(exc).__name__,
            )
            _flush_usage_if_configured()
            raise
        exact_usage = _normalize_usage(getattr(response, "usage", None))
        request_messages = _normalize_messages(kwargs.get("messages", []))
        response_text = ""
        try:
            response_text = response.choices[0].message.content or ""
        except Exception:
            pass
        _usage_tracker().record(
            call_tag=call_tag,
            provider="openai_compatible",
            model=model,
            success=True,
            usage=exact_usage or _estimate_usage(request_messages, response_text, model),
            usage_source="exact" if exact_usage is not None else "estimated",
        )
        _flush_usage_if_configured()
        return response

    completion_api.create = tracked_create
    return client


class _WorkflowEvolverClient:
    """Minimal ``LLMClient``-compatible adapter for workflow deployments."""

    def __init__(self, model: str, config: dict[str, Any], call_tag: str) -> None:
        self.model = model
        self._config = config
        self._call_tag = call_tag

    def chat(self, messages, settings=None, return_reasoning: bool = False):
        temperature = getattr(settings, "temperature", None) if settings is not None else None
        reply = chat(
            messages,
            model=self.model,
            temperature=temperature,
            config=self._config,
            call_tag=self._call_tag,
        )
        return (reply, "") if return_reasoning else reply


def _get_client(
    model: str = "deepseek-chat",
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    cache: bool = True,
    cache_tag: str = "",
    **kwargs,
):
    """Compatibility factory for Trace2Skill's evolver client.

    It participates in the same global usage tracker as :func:`chat`.
    """
    project_cfg = _load_project_config()
    if _should_use_workflow(project_cfg, api_key=api_key, base_url=base_url, config=None):
        assert project_cfg is not None
        return _WorkflowEvolverClient(
            model=str(project_cfg.get("model", model)),
            config=project_cfg,
            call_tag=f"evolver_{cache_tag or 'chat'}",
        )

    config = resolve_config(api_key=api_key, base_url=base_url, model=model)
    cache_key = f"{config['model']}:{config['api_key']}:{config['base_url']}:{cache_tag}"
    if cache and cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]

    trace2skill_root = _SKILLMINING / "Trace2Skill"
    if str(trace2skill_root) not in sys.path:
        sys.path.insert(0, str(trace2skill_root))
    from src.react_agent.models import OpenAIClient

    client = OpenAIClient(
        model=str(config["model"]),
        api_key=config["api_key"],
        base_url=config["base_url"],
        **kwargs,
    )
    client = _instrument_openai_client(
        client,
        model=str(config["model"]),
        call_tag=f"evolver_{cache_tag or 'chat'}",
    )
    if cache:
        _CLIENT_CACHE[cache_key] = client
    return client


def clear_cache() -> None:
    """Clear compatibility clients, primarily for tests."""
    _CLIENT_CACHE.clear()

if str(_SKILLMINING) not in sys.path:
    sys.path.insert(0, str(_SKILLMINING))

# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════

def _load_project_config() -> dict[str, Any] | None:
    """Load SkillMining/config.py LLM_CONFIG when present."""
    try:
        from config import LLM_CONFIG

        config = deepcopy(LLM_CONFIG)
        # A full ABCD launch may run several independent subflows in parallel.
        # Let each worker select its workflow endpoint without rewriting the
        # shared config.py (which would be racy across processes).
        workflow_id = os.getenv("SKILLMINING_WORKFLOW_ID", "").strip()
        if workflow_id:
            config["workflow_id"] = workflow_id
        return config
    except ImportError:
        return None

def get_llm_config() -> dict[str, Any]:
    """Return a copy of LLM_CONFIG from config.py."""
    cfg = _load_project_config()
    if cfg is None:
        raise RuntimeError(
            "Missing config.py. Copy config.example.py to config.py and edit LLM settings."
        )
    return cfg

def _is_workflow_config(cfg: dict[str, Any] | None) -> bool:
    return bool(cfg and cfg.get("provider") == "workflow" and cfg.get("workflow_id"))

def _should_use_workflow(
    project_cfg: dict[str, Any] | None,
    *,
    api_key: str | None,
    base_url: str | None,
    config: dict[str, Any] | None,
) -> bool:
    """Use workflow when config says so and caller did not force OpenAI creds."""
    cfg = config or project_cfg
    if not _is_workflow_config(cfg):
        return False
    return not api_key and not base_url

def resolve_config(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "deepseek-chat",
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, str | None]:
    """Resolve effective API configuration.

    Priority: explicit args > config.py > AWM config > environment variables.

    For workflow mode returns ``api_key`` and ``base_url`` as None.
  """
    project_cfg = config or _load_project_config()
    if _should_use_workflow(project_cfg, api_key=api_key, base_url=base_url, config=config):
        assert project_cfg is not None
        resolved_model = project_cfg.get("model", model)
        if model and model != "deepseek-chat":
            resolved_model = model
        return {
            "model": str(resolved_model),
            "api_key": None,
            "base_url": None,
        }

    if not api_key:
        api_key = _try_local_config("DEEPSEEK_API_KEY")
    if not base_url:
        base_url = _try_local_config("DEEPSEEK_BASE_URL")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "")

    if project_cfg and project_cfg.get("provider") == "openai":
        if not api_key:
            api_key = project_cfg.get("api_key", "")
        if not base_url:
            base_url = project_cfg.get("base_url", "")
        if model == "deepseek-chat" and project_cfg.get("model"):
            model = str(project_cfg["model"])

    if not api_key or not base_url:
        raise RuntimeError(
            "LLM API not configured. Provide api_key/base_url, set "
            "OPENAI_API_KEY/OPENAI_BASE_URL env vars, or configure SkillMining/config.py."
        )

    return {"model": model, "api_key": api_key, "base_url": base_url}

def _try_local_config(key: str) -> str | None:
    """Try to read a config value from AWM/config.py."""
    for config_dir in ("awm", "AWM"):
        config_path = _SKILLMINING.parent / config_dir / "config.py"
        if not config_path.exists():
            config_path = _SKILLMINING / config_dir / "config.py"
        if not config_path.exists():
            continue
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                f"_{config_dir}_config", str(config_path)
            )
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, key, None)
        except Exception:
            pass
    return None

# ══════════════════════════════════════════════════════════════════
# Workflow helpers
# ══════════════════════════════════════════════════════════════════

def _normalize_messages(messages: str | list) -> list[dict[str, str]]:
    """Normalize input to a list of role/content dicts."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    clean: list[dict[str, str]] = []
    for msg in messages:
        if hasattr(msg, "role") and hasattr(msg, "content"):
            clean.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg, dict):
            clean.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )
        else:
            clean.append({"role": "user", "content": str(msg)})
    return clean

def _messages_to_workflow_prompts(messages: list[dict[str, str]]) -> tuple[str, str]:
    system_parts: list[str] = []
    dialog_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            dialog_parts.append(f"助手：\n{content}")
        else:
            dialog_parts.append(f"用户：\n{content}")
    return "\n\n".join(system_parts), "\n\n".join(dialog_parts)

def _build_workflow_parameters(
    system_prompt: str,
    user_message: str,
    config: dict[str, Any],
) -> dict[str, str]:
    merge_field = config.get("workflow_merge_field")
    if merge_field:
        merged = user_message
        if system_prompt:
            merged = f"系统提示：\n{system_prompt}\n\n用户输入：\n{user_message}"
        return {merge_field: merged}

    user_field = config.get("workflow_user_field", "user_prompt")
    system_field = config.get("workflow_system_field", "system_prompt")
    params: dict[str, str] = {user_field: user_message}
    if system_prompt:
        params[system_field] = system_prompt
    return params

def _extract_workflow_text(response_data: Any) -> str:
    if isinstance(response_data, str):
        return response_data
    if isinstance(response_data, dict):
        for key in ("data", "res", "output", "text", "content", "answer", "result"):
            if key in response_data:
                return _extract_workflow_text(response_data[key])
    if isinstance(response_data, list):
        return "".join(_extract_workflow_text(item) for item in response_data)
    return json.dumps(response_data, ensure_ascii=False)


def _normalize_usage(usage: Any) -> dict[str, int] | None:
    """Extract exact token usage from OpenAI-compatible provider responses."""
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        try:
            usage = usage.model_dump()
        except Exception:
            pass
    elif hasattr(usage, "dict"):
        try:
            usage = usage.dict()
        except Exception:
            pass
    if not isinstance(usage, dict):
        return None
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    normalized: dict[str, int] = {}
    for target, names in aliases.items():
        value = next((usage.get(name) for name in names if usage.get(name) is not None), 0)
        try:
            normalized[target] = int(value)
        except (TypeError, ValueError):
            normalized[target] = 0
    if not normalized["total_tokens"]:
        normalized["total_tokens"] = normalized["prompt_tokens"] + normalized["completion_tokens"]
    return normalized if any(normalized.values()) else None


def _estimate_text_tokens(text: str, model: str | None = None) -> int:
    """Estimate tokens locally when the workflow API omits usage metadata."""
    text = str(text or "")
    if not text:
        return 0
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model or "deepseek-chat")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        # Deterministic fallback for servers without tiktoken. CJK characters
        # are close to one token; whitespace-delimited Latin text is roughly
        # four characters per token.
        import re
        cjk = len(re.findall(r"[\u3400-\u9fff]", text))
        rest = re.sub(r"[\u3400-\u9fff]", " ", text)
        latin = len(re.findall(r"\S+", rest))
        punctuation = len(re.findall(r"[^\w\s]", rest, flags=re.UNICODE))
        return max(1, cjk + latin + punctuation + len(rest) // 12)


def _estimate_usage(messages: list[dict[str, str]], response_text: str, model: str | None) -> dict[str, int]:
    """Estimate chat input/output/total tokens using the same local counter."""
    prompt_tokens = 2
    for message in messages:
        prompt_tokens += 4
        prompt_tokens += _estimate_text_tokens(message.get("role", ""), model)
        prompt_tokens += _estimate_text_tokens(message.get("content", ""), model)
    completion_tokens = _estimate_text_tokens(response_text, model)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

def _call_workflow(
    messages: list[dict[str, str]],
    config: dict[str, Any],
    *, call_tag: str = 'chat',
) -> tuple[str, dict[str, int] | None]:
    system_prompt, user_message = _messages_to_workflow_prompts(messages)
    parameters = _build_workflow_parameters(system_prompt, user_message, config)
    #print(parameters)
    def transport(workflow_id, send):
        return send(
            flow_id=workflow_id, tweaks=parameters, timeout=config.get("timeout", 180),
            workflow_url=config.get("workflow_url"),
            server_app_id=config.get("server_app_id", "ios-lma-svc"),
            user_id=config.get("user_id", "{{user_id}}"))
    from workflow_request_pool import from_environment
    pool = from_environment()
    if pool is None:
        from utils.llm_utils import run_flow_retry
        response_json = transport(config['workflow_id'], run_flow_retry)
    else:
        from workflow_http_transport import run_flow_once
        with pool.request(call_tag) as workflow_id:
            response_json = transport(workflow_id, run_flow_once)
    text = _extract_workflow_text(response_json.get("data", response_json))
    #print(text)
    if not text.strip():
        raise RuntimeError(f"Workflow returned empty data: {response_json}")
    usage_source = response_json.get("usage") if isinstance(response_json, dict) else None
    usage = _normalize_usage(usage_source)
    return text, usage

def _call_openai(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str,
    base_url: str | None,
    temperature: float,
    max_tokens: int | None,
    response_logger=None,
    call_tag: str = "chat",
    **kwargs,
) -> str:
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens
    request_kwargs.update(kwargs)

    resp = client.chat.completions.create(**request_kwargs)

    if response_logger is not None:
        try:
            response_logger.log(messages=messages, response=resp, call_tag=call_tag)
        except Exception:
            pass
    _usage_tracker().record(
        call_tag=call_tag,
        provider="openai_compatible",
        model=model,
        success=True,
        usage=(
            _normalize_usage(getattr(resp, "usage", None))
            or _estimate_usage(messages, resp.choices[0].message.content or "", model)
        ),
        usage_source=(
            "exact" if _normalize_usage(getattr(resp, "usage", None)) is not None
            else "estimated"
        ),
    )
    _flush_usage_if_configured()
    return resp.choices[0].message.content or ""

# ══════════════════════════════════════════════════════════════════
# Core API: prompt in, response out
# ══════════════════════════════════════════════════════════════════

def chat(
    messages: str | list,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    config: dict[str, Any] | None = None,
    response_logger=None,
    call_tag: str = "chat",
    **kwargs,
) -> str:
    """Send messages to the LLM and return the response text.

    The single entry point for all LLM calls in the project.

    Args:
        messages: A string (auto-wrapped as user message) or a list of
                  ``{"role": "...", "content": "..."}`` dicts / Message objects.
        model: Model name (default from config.py).
        temperature: Sampling temperature (default from config.py).
        max_tokens: Max tokens in response (None = model default).
        api_key: API key for OpenAI-compatible APIs (ignored in workflow mode).
        base_url: API base URL (ignored in workflow mode).
        config: Optional LLM config override (defaults to config.py LLM_CONFIG).
        response_logger: Optional ResponseLogger to record raw I/O.
        call_tag: Logical call category used in the shared usage summary.
        **kwargs: Extra args passed to the OpenAI API.

    Returns:
        The response text string. Empty string on failure.
    """
    clean = _normalize_messages(messages)
    project_cfg = config or _load_project_config()

    if project_cfg is None:
        raise RuntimeError(
            "Missing config.py. Copy config.example.py to config.py and edit LLM settings."
        )

    resolved_model = model or str(project_cfg.get("model", "deepseek-chat"))
    resolved_temp = (
        temperature if temperature is not None else float(project_cfg.get("temperature", 0.3))
    )
    resolved_max_tokens = max_tokens if max_tokens is not None else project_cfg.get("max_tokens")

    workflow_failure_recorded = False
    try:
        if _should_use_workflow(project_cfg, api_key=api_key, base_url=base_url, config=config):
            pooled = bool(os.getenv('SKILLMINING_WORKFLOW_POOL_IDS', '').strip())
            waits = _WORKFLOW_RATE_LIMIT_RETRIES if pooled else (0,)
            for attempt, wait_s in enumerate(waits):
                if wait_s:
                    _workflow_retry_sleep(wait_s)
                workflow_failure_recorded = False
                try:
                    resp, workflow_usage = _call_workflow(clean, project_cfg, call_tag=call_tag)
                except Exception as exc:
                    # Every physical attempt has its own native counter, even
                    # when a logical chat eventually succeeds after throttling.
                    _usage_tracker().record(call_tag=call_tag, provider='workflow',
                        model=resolved_model, success=False, error_type=type(exc).__name__)
                    workflow_failure_recorded = True
                    _flush_usage_if_configured()
                    if pooled and getattr(exc, 'status_code', None) == 429 and attempt+1 < len(waits):
                        log.warning('Workflow rate-limit retry: tag=%s next_attempt=%s/%s wait=%ss',
                            call_tag, attempt+2, len(waits), waits[attempt+1])
                        continue
                    raise
                break
            if response_logger is not None:
                try:
                    response_logger.log(messages=clean, response={"content": resp}, call_tag=call_tag)
                except Exception:
                    pass
            provider_usage = workflow_usage or _estimate_usage(
                clean, resp, resolved_model,
            )
            _usage_tracker().record(
                call_tag=call_tag,
                provider="workflow",
                model=resolved_model,
                success=True,
                usage=provider_usage,
                usage_source="exact" if workflow_usage is not None else "estimated",
            )
            _flush_usage_if_configured()
            return resp

        cfg = resolve_config(
            api_key=api_key,
            base_url=base_url,
            model=resolved_model,
            config=project_cfg,
        )
        return _call_openai(
            clean,
            model=cfg["model"],
            api_key=cfg["api_key"] or "",
            base_url=cfg["base_url"],
            temperature=resolved_temp,
            max_tokens=resolved_max_tokens,
            response_logger=response_logger,
            call_tag=call_tag,
            **kwargs,
        )
    except Exception as exc:
        provider = (
            "workflow"
            if _should_use_workflow(project_cfg, api_key=api_key, base_url=base_url, config=config)
            else "openai_compatible"
        )
        if not workflow_failure_recorded:
            _usage_tracker().record(
                call_tag=call_tag,
                provider=provider,
                model=resolved_model,
                success=False,
                error_type=type(exc).__name__,
            )
        _flush_usage_if_configured()
        log.warning("LLM call failed: %s", exc)
        raise
