import copy
import json

import pytest

from eval_tod.sgd_adapter import gold_policy
from eval_tod.sgd_standard_skill_agent import SGDStandardSkillAgent
from tests.test_sgd_skill_agent import dialogue, ONTOLOGY


def make_agent(monkeypatch, tmp_path):
    monkeypatch.setattr("llm.chat", lambda *_args, **_kwargs: '{"updates":[]}')
    agent = SGDStandardSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY,
        predictor=lambda *_args, **_kwargs: ({"call": None, "acts": []}, {}))
    agent.mine([dialogue(), dialogue("train-2")], tmp_path / "offline")
    return agent


def test_standard_all_policy_turns_and_limit_unit(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    seen = []
    initial = agent.resource_hash()

    def predict(d, index):
        seen.append((d["dialogue_id"], index, agent.resource_hash()))
        return gold_policy(d, index), {"source_node": "<START>"}

    agent.predict = predict
    result = agent.refine([dialogue(), dialogue("train-2")], tmp_path,
                          selection_rate=0, batch_size=1, max_batches=1)
    assert result["rollout_turns"] == 1  # limit is turns, NOT dialogues
    assert seen == [("train-1", 1, initial)]
    result = agent.refine([dialogue(), dialogue("train-2")], tmp_path, selection_rate=0)
    assert result["rollout_turns"] == 2  # standard ignores sampling
    assert json.loads((tmp_path / "online_schedule.json").read_text())["schedule_unit"] == "policy_turn"


def test_shared_clustering_and_sequential_reflection_current_text(monkeypatch, tmp_path):
    from eval_tod import sgd_standard_skill_agent as module
    from skill_mining import online_refinement as shared
    assert module.build_post_rollout_batches is shared.build_post_rollout_batches
    assert module.group_batch_reports is shared.group_batch_reports
    assert module.apply_reflection_updates_to_state is shared.apply_reflection_updates_to_state
    agent = make_agent(monkeypatch, tmp_path)
    agent.predict = lambda d, i: (gold_policy(d, i), {"source_node": "<START>"})
    label = agent.graph["nodes"][0]["label"]
    rounds = []

    def chat(messages, **kwargs):
        prompt = messages[0]["content"]
        assert "CURRENT_SYSTEM_SECRET" not in prompt
        if kwargs["call_tag"] == "online_batch_root_cause":
            return json.dumps({"summary": "named payload", "graph_footprint": {"nodes": [label]},
                               "candidate_updates": [], "root_causes": []})
        assert kwargs["call_tag"] == "online_group_reflection"
        rounds.append(prompt)
        text = "# SGD backbone skill" if len(rounds) == 1 else "# UPDATED SGD skill"
        new = "# UPDATED SGD skill" if len(rounds) == 1 else "# FINAL SGD skill"
        return json.dumps({"updates": [{"resource": "slot_policy", "action": label,
                            "content": "Bind time from user context", "status": "resolved"}],
                           "skill_operations": [{"operation_id": f"edit-{len(rounds)}", "op": "replace",
                               "match_text": text, "new_text": new}]})

    monkeypatch.setattr("llm.chat", chat)
    # Force two neighborhoods to test the inter-group write contract.
    monkeypatch.setattr(module, "group_batch_reports", lambda reports, max_reports: [[r] for r in reports])
    old_graph = copy.deepcopy(agent.graph)
    result = agent.refine([dialogue(), dialogue("train-2")], tmp_path, analysis_batch_size=1)
    assert len(rounds) == 2
    assert "# UPDATED SGD skill" in rounds[1]
    assert agent.working_skill.startswith("# FINAL SGD skill")
    assert agent.state["slot_policies"][label]["policy"] == "Bind time from user context"
    assert result["reflection_groups"] == 2
    assert agent.graph == old_graph
    loaded = SGDStandardSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY, predictor=agent.predictor)
    loaded.load(tmp_path)
    assert loaded.resource_hash() == agent.resource_hash()
    assert all(item["skill_operations"][0]["applied"] for item in agent.ledger if item["phase"] == "online_reflection")


def test_runtime_reference_plan_then_action_selection_then_payload(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    tags = []
    heads = [["request", "Restaurants_1"]]

    def chat(prompt, **kwargs):
        assert "CURRENT_SYSTEM_SECRET" not in prompt
        tags.append(kwargs["call_tag"])
        if tags[-1] == "sgd_reference_plan":
            return '{"query_text":"missing booking time","candidate_actions":[]}'
        return json.dumps({"action_heads": heads})

    def predictor(d, i, **kwargs):
        tags.append("payload")
        assert kwargs["selected_action_heads"] == heads
        assert agent.working_skill in kwargs["skill_text"]
        return {"call": None, "acts": []}, {}

    monkeypatch.setattr("llm.chat", chat)
    agent.predictor = predictor
    before = agent.resource_hash()
    _, diagnostic = agent.predict(dialogue(), 1)
    assert tags == ["sgd_reference_plan", "sgd_action_selection", "payload"]
    assert diagnostic["selected_action_heads"] == heads
    assert agent.resource_hash() == before


def test_standard_rejects_prototype_checkpoint(monkeypatch, tmp_path):
    from eval_tod.sgd_skill_agent import SGDSkillAgent
    monkeypatch.setattr("llm.chat", lambda *_args, **_kwargs: '{"updates":[]}')
    prototype = SGDSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY, predictor=None)
    prototype.mine([dialogue()], tmp_path)
    standard = SGDStandardSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY, predictor=None)
    with pytest.raises(ValueError, match="prototype"):
        standard.load(tmp_path)


def test_shared_abcd_prompt_defaults_unchanged():
    from skill_mining.online_refinement import diagnose_rollout_batch, reflect_batch_report_group
    import inspect
    assert inspect.signature(diagnose_rollout_batch).parameters["record_projector"].default is None
    assert inspect.signature(reflect_batch_report_group).parameters["chat_fn"].default is None


def test_grounding_failure_does_not_reselect_actions_or_unlock_observation(monkeypatch):
    from scripts.run_sgd_domain_eval import predict_policy
    from eval_tod.sgd_adapter import action_counter
    monkeypatch.setattr("llm.chat", lambda *_args, **_kwargs: '{"call":null,"acts":[]}')
    selected = [["call", "Restaurants_1"], ["request", "Restaurants_1"]]
    prediction, diagnostic = predict_policy(dialogue(), 1, model="mock", ontology=ONTOLOGY,
        frequency_graph=None, selected_action_heads=selected)
    assert set(action_counter(prediction)) == {tuple(head) for head in selected}
    assert prediction["call"]["method"] == ""
    assert diagnostic["observation_visible"] is False
    assert diagnostic["call_parse_error"] == "missing_selected_call_payload"


def test_refining_checkpoint_cannot_be_reported_as_offline(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path)
    agent.stage = "refining"
    agent.save(tmp_path)
    loaded = SGDStandardSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY, predictor=None)
    loaded.load(tmp_path)
    assert loaded.stage == "refining"


def test_standard_runner_usage_and_frozen_reload(monkeypatch, tmp_path):
    import argparse
    from io import BytesIO
    from scripts import run_sgd_domain_eval as runner

    class Archive:
        def __init__(self, _path): pass
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def open(self, _name): return BytesIO(json.dumps(ONTOLOGY).encode())

    def chat(prompt, **kwargs):
        tag = kwargs["call_tag"]
        if tag == "sgd_reference_plan": return '{"query_text":"time","candidate_actions":[]}'
        if tag == "sgd_action_selection":
            return '{"action_heads":[["request","Restaurants_1"],["confirm","Restaurants_1"]]}'
        if tag == "sgd_call_selection": return '{"call":null}'
        if tag == "sgd_dialogue_acts": return json.dumps({"acts": gold_policy(dialogue(), 1)["acts"]})
        if tag == "online_batch_root_cause": return '{"summary":"correct","candidate_updates":[]}'
        return '{"updates":[]}'

    monkeypatch.setattr(runner, "ZipFile", Archive)
    monkeypatch.setattr(runner, "iter_family_dialogues", lambda _r, _f, split: iter([dialogue(split)]))
    monkeypatch.setattr("llm.resolve_config", lambda **_kwargs: {})
    monkeypatch.setattr("llm.chat", chat)
    parser = argparse.ArgumentParser()
    runner.add_ours_arguments(parser)
    args = parser.parse_args([])
    assert args.ours_refinement_mode == "standard"
    vars(args).update(method="ours", domain="Restaurants", split="test", splits_dir=tmp_path,
        archive=tmp_path / "fake.zip", output_dir=tmp_path / "run", model="mock", max_dialogues=1)
    summary = runner.run_domain(args)
    assert summary["turn_joint_ast"] == 1
    assert summary["total_usage"]["calls"] == 11
    assert summary["mining_usage"]["calls"] == 7
    assert summary["usage"]["calls"] == 4
    assert set(summary["llm_usage"]["by_phase"]) == {
        "offline_compile", "train_rollout", "online_diagnosis", "online_reflection", "testing"}
    checkpoint = (args.output_dir / "sgd_standard_state.json").read_bytes()
    args.ours_resource_dir = args.output_dir
    args.output_dir = tmp_path / "frozen"
    frozen = runner.run_domain(args)
    assert frozen["mining_usage"]["calls"] == 0
    assert frozen["usage"]["calls"] == 4
    assert (args.ours_resource_dir / "sgd_standard_state.json").read_bytes() == checkpoint
