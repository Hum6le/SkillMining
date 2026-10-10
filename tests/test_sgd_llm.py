from urllib.error import HTTPError

import pytest

from eval_tod import sgd_llm


@pytest.fixture(autouse=True)
def retry_config(monkeypatch):
    monkeypatch.setenv("SKILLMINING_SGD_LLM_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("SKILLMINING_SGD_LLM_RETRY_BASE_SECONDS", "2")
    monkeypatch.setenv("SKILLMINING_SGD_LLM_RETRY_MAX_SECONDS", "30")
    monkeypatch.setattr(sgd_llm.random, "uniform", lambda *_args: 1.0)


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_wrapped_http_error_retries_same_request(monkeypatch, status):
    calls, sleeps = [], []

    def chat(messages, **kwargs):
        calls.append((messages, kwargs))
        if len(calls) < 3:
            cause = HTTPError("http://example.invalid", status, "gateway", {}, None)
            raise RuntimeError("Provider request failed") from cause
        return "OK"

    monkeypatch.setattr("llm.chat", chat)
    monkeypatch.setattr(sgd_llm.time, "sleep", sleeps.append)
    messages = [{"role": "user", "content": "request"}]
    assert sgd_llm.sgd_chat_with_retry(messages, call_tag="sgd_call_selection") == "OK"
    assert len(calls) == 3
    assert all(call == calls[0] for call in calls)
    assert sleeps == [2, 4]


def test_message_only_workflow_error_is_supported(monkeypatch):
    replies = iter([RuntimeError("Workflow HTTP error 502: gateway"), "OK"])

    def chat(*_args, **_kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr("llm.chat", chat)
    monkeypatch.setattr(sgd_llm.time, "sleep", lambda _seconds: None)
    assert sgd_llm.sgd_chat_with_retry("request") == "OK"


def test_exhaustion_preserves_error(monkeypatch):
    calls, sleeps = [], []
    error = RuntimeError("Workflow HTTP error 502: gateway")

    def chat(*_args, **_kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr("llm.chat", chat)
    monkeypatch.setattr(sgd_llm.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError) as raised:
        sgd_llm.sgd_chat_with_retry("request")
    assert raised.value is error
    assert len(calls) == 3
    assert sleeps == [2, 4]


@pytest.mark.parametrize("error", [
    RuntimeError("Workflow HTTP error 400: bad request"),
    RuntimeError("Workflow HTTP error 401: authentication"),
    RuntimeError("Workflow returned ret=1004: rate limit"),
    ValueError("Invalid schema with 502 entries"),
])
def test_nontransient_errors_are_not_retried(monkeypatch, error):
    calls = []

    def chat(*_args, **_kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr("llm.chat", chat)
    monkeypatch.setattr(sgd_llm.time, "sleep", lambda _seconds: pytest.fail("must not retry"))
    with pytest.raises(type(error)) as raised:
        sgd_llm.sgd_chat_with_retry("request")
    assert raised.value is error
    assert len(calls) == 1


@pytest.mark.parametrize("reply", ["", "   ", None])
def test_empty_reply_aborts_instead_of_scoring(monkeypatch, reply):
    monkeypatch.setattr("llm.chat", lambda *_args, **_kwargs: reply)
    monkeypatch.setattr(sgd_llm.time, "sleep", lambda _seconds: pytest.fail("status unknown"))
    with pytest.raises(RuntimeError, match="aborting instead of scoring"):
        sgd_llm.sgd_chat_with_retry("request")


def test_backoff_cap(monkeypatch):
    monkeypatch.setenv("SKILLMINING_SGD_LLM_RETRY_MAX_SECONDS", "3")
    sleeps = []
    monkeypatch.setattr(sgd_llm.time, "sleep", sleeps.append)

    def chat(*_args, **_kwargs):
        raise RuntimeError("Workflow HTTP error 503: unavailable")

    monkeypatch.setattr("llm.chat", chat)
    with pytest.raises(RuntimeError):
        sgd_llm.sgd_chat_with_retry("request")
    assert sleeps == [2, 3]
