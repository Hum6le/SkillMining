import json
from collections import Counter
from io import BytesIO
from types import SimpleNamespace

import pytest

from eval_tod.sgd_agent import SGDAWMAgent


def _dialogue():
    return {
        "dialogue_id": "sample-1",
        "services": ["Restaurants_1"],
        "all_services": ["Restaurants_1"],
        "target_indices": {"policy_turn_indices": [1]},
        "turns": [
            {"speaker": "user", "utterance": "Book a table for two"},
            {"speaker": "system", "utterance": "What time?", "service_call": {},
             "dialogue_acts": {"non-categorical": [
                 {"intent": "request", "domain": "Restaurants_1", "slot": "time"},
                 {"intent": "confirm", "domain": "Restaurants_1", "slot": "party_size", "value": "2"},
             ]}},
        ],
    }


def test_sgd_awm_online_train_save_and_frozen_reuse(monkeypatch, tmp_path):
    from eval_tod.sgd_adapter import gold_policy

    calls = []
    def predictor(dialogue, turn_index, **kwargs):
        calls.append(kwargs)
        return gold_policy(dialogue, turn_index), {}

    def fake_chat(prompt, **kwargs):
        assert kwargs["call_tag"] == "sgd_awm_induction"
        assert '"intent": "request"' in prompt
        assert '"intent": "confirm"' in prompt
        return "### Ask for missing time\n**When**: booking time is missing\n**Do**: REQUEST(time)"

    monkeypatch.setattr("llm.chat", fake_chat)
    agent = SGDAWMAgent(family="Restaurants", model="mock", ontology={}, predictor=predictor)
    summary = agent.train_batch([_dialogue()], batch_index=1)
    assert summary["rollout"]["turn_joint_ast"] == 1.0
    assert summary["successful_turns_added"] == 1
    assert len(agent.memory) == 1
    assert "Ask for missing time" in agent.workflow.text
    assert calls[0]["workflow_text"] == ""
    agent.predict(_dialogue(), 1)
    assert "Ask for missing time" in calls[-1]["workflow_text"]
    assert "verified_policy" in calls[-1]["exemplar_text"]
    directory = tmp_path / "resources"
    agent.save(directory)
    frozen = SGDAWMAgent(family="Restaurants", model="mock", ontology={}, predictor=predictor)
    frozen.load(directory)
    assert frozen.workflow.text == agent.workflow.text
    assert json.loads((directory / "awm_exemplars.json").read_text(encoding="utf-8"))


def test_sgd_awm_wrong_turn_not_admitted_to_memory(monkeypatch):
    monkeypatch.setattr("llm.chat", lambda *_args, **_kwargs: "### Learned rule")
    def predictor(_dialogue, _turn_index, **_kwargs):
        return {"call": None, "acts": []}, {}
    agent = SGDAWMAgent(family="Restaurants", model="mock", ontology={}, predictor=predictor)
    summary = agent.train_batch([_dialogue()], batch_index=1)
    assert summary["successful_turns_added"] == 0
    assert len(agent.memory) == 0


def test_sgd_awm_domain_runner_uses_train_only_for_induction(monkeypatch, tmp_path):
    from scripts import run_sgd_domain_eval as runner

    seen_splits = []
    def dialogues(_root, _family, split):
        seen_splits.append(split)
        yield _dialogue()

    class Archive:
        def __init__(self, _path):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def open(self, _name):
            return BytesIO(json.dumps({"domains": {"Restaurants_1": {
                "description": "restaurant booking", "active_intents": []
            }}}).encode())

    def fake_chat(_prompt, **kwargs):
        tag = kwargs["call_tag"]
        if tag == "sgd_awm_induction":
            return "### Ask for booking time"
        if tag == "sgd_call_selection":
            return '{"call":null}'
        return '{"acts":[]}'

    monkeypatch.setattr(runner, "iter_family_dialogues", dialogues)
    monkeypatch.setattr(runner, "ZipFile", Archive)
    monkeypatch.setattr("llm.resolve_config", lambda **_kwargs: {})
    monkeypatch.setattr("llm.chat", fake_chat)
    args = SimpleNamespace(
        method="awm", domain="Restaurants", split="validation", splits_dir=tmp_path,
        archive=tmp_path / "fake.zip", output_dir=tmp_path / "run", model="mock",
        max_dialogues=1, awm_resource_dir=None, awm_batch_size=1, awm_max_train=1,
        awm_max_batches=None, awm_workflow_max_chars=8000, awm_exemplar_max_chars=3000,
    )
    summary = runner.run_domain(args)
    assert seen_splits == ["train", "validation"]
    assert summary["counts"]["turns"] == 1
    assert (args.output_dir / "awm_workflow.txt").is_file()
    assert (args.output_dir / "awm_training.json").is_file()
    args.output_dir = tmp_path / "frozen"
    args.awm_resource_dir = tmp_path / "run"
    seen_splits.clear()
    runner.run_domain(args)
    assert seen_splits == ["validation"]


def test_sgd_policy_requests_retry_individually(monkeypatch):
    from eval_tod import sgd_llm
    from scripts.run_sgd_domain_eval import predict_policy

    calls = Counter()
    monkeypatch.setattr(sgd_llm.time, "sleep", lambda _seconds: None)

    def chat(_prompt, **kwargs):
        tag = kwargs["call_tag"]
        calls[tag] += 1
        if calls[tag] == 1:
            raise RuntimeError("Workflow HTTP error 502: gateway")
        if tag == "sgd_call_selection":
            return '{"call":null}'
        return '{"acts":[]}'

    monkeypatch.setattr("llm.chat", chat)
    prediction, _ = predict_policy(
        _dialogue(), 1, model="mock", frequency_graph=None,
        ontology={"domains": {"Restaurants_1": {
            "description": "restaurant booking", "active_intents": [],
        }}},
    )
    assert prediction == {"call": None, "acts": []}
    assert calls == {"sgd_call_selection": 2, "sgd_dialogue_acts": 2}


def test_sgd_induction_retry_does_not_repeat_rollout_or_memory(monkeypatch):
    from eval_tod import sgd_llm
    from eval_tod.sgd_adapter import gold_policy

    calls = Counter()
    monkeypatch.setattr(sgd_llm.time, "sleep", lambda _seconds: None)

    def predictor(dialogue, turn_index, **_kwargs):
        calls["rollout"] += 1
        return gold_policy(dialogue, turn_index), {}

    def chat(_prompt, **kwargs):
        assert kwargs["call_tag"] == "sgd_awm_induction"
        calls["induction"] += 1
        if calls["induction"] == 1:
            raise RuntimeError("Workflow HTTP error 502: gateway")
        return "### Ask for missing time"

    monkeypatch.setattr("llm.chat", chat)
    agent = SGDAWMAgent(family="Restaurants", model="mock", ontology={}, predictor=predictor)
    summary = agent.train_batch([_dialogue()], batch_index=1)
    assert calls == {"rollout": 1, "induction": 2}
    assert summary["successful_turns_added"] == 1
    assert len(agent.memory) == 1


def test_sgd_exhausted_induction_leaves_resources_unchanged(monkeypatch):
    from eval_tod import sgd_llm
    from eval_tod.sgd_adapter import gold_policy

    monkeypatch.setenv("SKILLMINING_SGD_LLM_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(sgd_llm.time, "sleep", lambda _seconds: None)

    def predictor(dialogue, turn_index, **_kwargs):
        return gold_policy(dialogue, turn_index), {}

    def chat(*_args, **_kwargs):
        raise RuntimeError("Workflow HTTP error 502: gateway")

    monkeypatch.setattr("llm.chat", chat)
    agent = SGDAWMAgent(family="Restaurants", model="mock", ontology={}, predictor=predictor)
    agent.workflow.replace("### Existing workflow")
    old_workflow = agent.workflow.text
    with pytest.raises(RuntimeError, match="Workflow HTTP error 502"):
        agent.train_batch([_dialogue()], batch_index=1)
    assert agent.workflow.text == old_workflow
    assert len(agent.memory) == 0
