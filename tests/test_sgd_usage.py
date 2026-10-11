import json

import pytest

from eval_tod import sgd_usage, sgd_llm


def test_fallback_counts_retries_phases_and_persists_exhaustion(monkeypatch, tmp_path):
    sgd_usage.configure_usage(tmp_path)
    monkeypatch.setattr("llm.get_usage_summary", lambda: {})
    monkeypatch.setenv("SKILLMINING_SGD_LLM_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(sgd_llm.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr("llm.chat", lambda *_args, **_kwargs: "OK")
    sgd_usage.set_phase("offline_compile")
    sgd_llm.sgd_chat_with_retry("abc", call_tag="compile")
    sgd_usage.set_phase("testing")
    def failed(*_args, **_kwargs): raise RuntimeError("Workflow HTTP error 502: failed")
    monkeypatch.setattr("llm.chat", failed)
    with pytest.raises(RuntimeError):
        sgd_llm.sgd_chat_with_retry("abc", call_tag="eval")
    payload = json.loads((tmp_path / "llm_usage.json").read_text())
    assert payload["total"]["total"]["calls"] == 3
    assert payload["total"]["total"]["failed_calls"] == 2
    assert payload["generation"]["total"]["calls"] == 1
    assert payload["testing"]["total"]["calls"] == 2
    assert payload["total"]["total"]["usage_source"] == "estimated"
    sgd_usage.configure_usage(tmp_path)
    assert sgd_usage.usage_payload()["total"]["total"]["calls"] == 3


def test_native_exact_usage_is_not_double_counted(monkeypatch, tmp_path):
    sgd_usage.configure_usage(tmp_path)
    native = {"calls": 0, "successful_calls": 0, "failed_calls": 0, "calls_with_usage": 0,
              "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
              "exact_calls": 0, "estimated_calls": 0}
    monkeypatch.setattr("llm.get_usage_summary", lambda: {"total": dict(native)})
    def chat(*_args, **_kwargs):
        native.update(calls=1, successful_calls=1, calls_with_usage=1,
                      prompt_tokens=100, completion_tokens=20, total_tokens=120, exact_calls=1)
        return "OK"
    monkeypatch.setattr("llm.chat", chat)
    sgd_llm.sgd_chat_with_retry("request")
    total = sgd_usage.usage_payload()["total"]["total"]
    assert total["calls"] == 1
    assert total["total_tokens"] == 120
    assert total["usage_source"] == "exact"


def test_native_calls_without_token_metadata_use_explicit_estimates(monkeypatch, tmp_path):
    sgd_usage.configure_usage(tmp_path)
    native = {"calls": 0}
    monkeypatch.setattr("llm.get_usage_summary", lambda: {"total": dict(native)})
    def chat(*_args, **_kwargs):
        native.update(calls=1, successful_calls=1)
        return "answer"
    monkeypatch.setattr("llm.chat", chat)
    sgd_llm.sgd_chat_with_retry("request")
    total = sgd_usage.usage_payload()["total"]["total"]
    assert total["calls"] == 1
    assert total["total_tokens"] > 0
    assert total["usage_source"] == "estimated"


def test_full_aggregate_sums_usage_and_partial_cost_without_partial_metrics(tmp_path):
    from scripts.run_full_sgd_experiments import _aggregate
    tasks = [{"method": "awm", "domain": "A"}, {"method": "awm", "domain": "B"},
             {"method": "awm", "domain": "Failed"}]
    for domain, turns, correct, generation_calls, test_calls in [("A", 10, 8, 3, 2), ("B", 30, 12, 4, 5)]:
        path = tmp_path / "awm" / domain
        path.mkdir(parents=True)
        generation = {"total": {"calls": generation_calls, "total_tokens": generation_calls * 100}}
        testing = {"total": {"calls": test_calls, "total_tokens": test_calls * 100}}
        summary = {"counts": {"turns": turns, "action_exact": correct, "joint_exact": correct},
                   "llm_usage": {"generation": generation, "testing": testing,
                                  "total": {"total": {"calls": generation_calls + test_calls,
                                                       "total_tokens": (generation_calls + test_calls) * 100}}}}
        (path / "summary.json").write_text(json.dumps(summary))
    failed = tmp_path / "awm" / "Failed"
    failed.mkdir()
    (failed / "llm_usage.json").write_text(json.dumps({"total": {"total": {"calls": 6}}}))
    aggregate = _aggregate(tmp_path, tasks)
    result = aggregate["methods"]["awm"]
    assert result["action_set_exact"] == .5
    assert result["mining_usage"]["calls"] == 7
    assert result["usage"]["calls"] == 7
    assert result["total_usage"]["calls"] == 14
    assert aggregate["run_total_usage"]["total"]["calls"] == 20
    assert len(aggregate["missing"]) == 1


def test_awm_domain_usage_keeps_training_not_just_testing(monkeypatch, tmp_path):
    from scripts import run_sgd_domain_eval as runner
    from tests.test_sgd_skill_agent import dialogue, ONTOLOGY
    from io import BytesIO
    from types import SimpleNamespace
    class Archive:
        def __init__(self, _path): pass
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def open(self, _name): return BytesIO(json.dumps(ONTOLOGY).encode())
    monkeypatch.setattr(runner, "iter_family_dialogues", lambda *_args: iter([dialogue()]))
    monkeypatch.setattr(runner, "ZipFile", Archive)
    monkeypatch.setattr("llm.resolve_config", lambda **_kwargs: {})
    def chat(_prompt, **kwargs):
        if kwargs["call_tag"] == "sgd_awm_induction": return "### Ask for time"
        if kwargs["call_tag"] == "sgd_call_selection": return '{"call":null}'
        return '{"acts":[]}'
    monkeypatch.setattr("llm.chat", chat)
    args = SimpleNamespace(method="awm", domain="Restaurants", split="test", splits_dir=tmp_path,
        archive=tmp_path / "fake.zip", output_dir=tmp_path / "run", model="mock", max_dialogues=1,
        awm_resource_dir=None, awm_batch_size=1, awm_max_train=1, awm_max_batches=None,
        awm_workflow_max_chars=8000, awm_exemplar_max_chars=3000)
    summary = runner.run_domain(args)
    assert summary["mining_usage"]["calls"] == 3
    assert summary["usage"]["calls"] == 2
    assert summary["total_usage"]["calls"] == 5
