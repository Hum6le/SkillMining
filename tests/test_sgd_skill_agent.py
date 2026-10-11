import argparse
import copy
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval_tod.sgd_adapter import canonical_acts, gold_policy
from eval_tod.sgd_skill_agent import SGDSkillAgent, policy_token


def dialogue(identifier="train-1", service="Restaurants_1"):
    return {"dialogue_id": identifier, "services": [service], "all_services": [service],
            "target_indices": {"policy_turn_indices": [1]}, "turns": [
                {"speaker": "user", "utterance": "Please book a table for two"},
                {"speaker": "system", "utterance": "CURRENT_SYSTEM_SECRET", "service_call": {},
                 "dialogue_acts": {"non-categorical": [
                     {"intent": "request", "domain": service, "slot": "time"},
                     {"intent": "confirm", "domain": service, "slot": "party_size", "value": "2"}]}}]}


ONTOLOGY = {"domains": {"Restaurants_1": {"description": "restaurant booking", "active_intents": []}}}


def test_policy_bundle_is_order_value_and_service_version_invariant():
    first = gold_policy(dialogue(), 1)
    other = copy.deepcopy(first)
    other["acts"].reverse()
    other["acts"][0]["value"] = "different"
    assert policy_token(first) == policy_token(other)
    assert policy_token(first) == policy_token(gold_policy(dialogue(service="Restaurants_2"), 1))
    assert policy_token(first) != policy_token({"call": None, "acts": first["acts"][:1]})


def test_shared_miner_atomic_bundle_no_future_text_or_abcd_contract(monkeypatch, tmp_path):
    prompts = []
    monkeypatch.setattr("llm.chat", lambda prompt, **kwargs: prompts.append(prompt) or '{"updates":[]}')
    agent = SGDSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY,
                         predictor=lambda *_args, **_kwargs: ({"call": None, "acts": []}, {}))
    agent.mine([dialogue()], tmp_path)
    assert len(agent.graph["nodes"]) == 1
    assert agent.graph["nodes"][0]["action_heads"] == [["confirm", "Restaurants"], ["request", "Restaurants"]]
    assert agent.graph["edges"] == []
    assert "slot_contract" not in agent.graph["nodes"][0]
    assert "CURRENT_SYSTEM_SECRET" not in "\n".join(prompts)
    saved = agent.resource_hash()
    agent.predict(dialogue(), 1)
    assert agent.resource_hash() == saved
    agent.apply_updates([{"resource": "action_rules", "key": "missing", "text": "bad"}], phase="test")
    assert agent.ledger[-1]["applied"] is False
    agent.save(tmp_path)
    loaded = SGDSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY, predictor=agent.predictor)
    loaded.load(tmp_path)
    assert loaded.resource_hash() == saved
    assert loaded.graph == agent.graph


def test_posthoc_reflection_uses_frozen_snapshot_and_stable_ids(monkeypatch, tmp_path):
    snapshots, calls = [], []
    agent = SGDSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY,
                         predictor=lambda *_args, **_kwargs: ({"call": None, "acts": []}, {}))
    monkeypatch.setattr("llm.chat", lambda *_args, **_kwargs: '{"updates":[]}')
    agent.mine([dialogue(), dialogue("train-2")], tmp_path)
    old_graph = copy.deepcopy(agent.graph)
    initial = agent.resource_hash()

    def predictor(*_args, **_kwargs):
        snapshots.append(agent.resource_hash())
        return {"call": None, "acts": []}, {}

    def chat(prompt, **kwargs):
        calls.append(kwargs["call_tag"])
        if kwargs["call_tag"] == "sgd_ours_diagnosis":
            payload = json.loads(prompt[prompt.index('{"resources"'):])
            assert all(row["error_type"] == "action" for row in payload["evidence"])
            key = next(iter(payload["resources"]["action_rules"]))
            return json.dumps({"updates": [{"resource": "action_rules", "key": key, "text": "proposal"}]})
        payload = json.loads(prompt[prompt.index('{"targets"'):])
        return json.dumps({"updates": [{"resource": item["resource"], "key": item["key"],
                                         "text": "Ask for missing time; confirm party size from the user."}
                                        for item in payload["targets"]]})

    agent.predictor = predictor
    monkeypatch.setattr("llm.chat", chat)
    summary = agent.refine([dialogue(), dialogue("train-2")], tmp_path,
                           selection_rate=1, batch_size=1, analysis_batch_size=1)
    assert snapshots == [initial, initial]
    assert calls == ["sgd_ours_diagnosis", "sgd_ours_diagnosis", "sgd_ours_reflection"]
    assert summary["final_sha256"] != initial
    assert agent.graph == old_graph
    assert all(row["applied"] for row in agent.ledger if row["phase"] == "online_reflection")


def test_full_runner_forwards_every_ours_option_and_same_defaults(monkeypatch, tmp_path):
    from scripts import run_full_sgd_experiments as full
    from scripts.run_sgd_domain_eval import add_ours_arguments

    monkeypatch.setattr("sys.argv", ["runner", "--method", "ours", "--seed", "17",
        "--ours-selection-rate", ".6", "--ours-offline-only", "--ours-candidate-limit", "5",
        "--ours-resource-root", str(tmp_path / "resources")])
    args = full._arguments()
    command = full._domain_command(args, {"method": "ours", "domain": "Restaurants"}, tmp_path / "output")
    parser = argparse.ArgumentParser()
    add_ours_arguments(parser)
    selected = command[command.index("--ours-offline-only"):]
    # Main args are before ours flags; the remaining segment is exactly the
    # standalone method argument contract, including the resource-dir mapping.
    parsed = parser.parse_args(selected)
    for key, value in vars(args).items():
        if key.startswith("ours_") and key != "ours_resource_root" or key == "seed":
            assert getattr(parsed, key) == value
    assert parsed.ours_resource_dir == args.ours_resource_root.resolve() / "Restaurants"


def test_ours_runner_train_test_separation_usage_and_freeze(monkeypatch, tmp_path):
    from scripts import run_sgd_domain_eval as runner
    seen_splits = []
    def dialogues(_root, _family, split):
        seen_splits.append(split)
        yield dialogue(split + "-1")
    class Archive:
        def __init__(self, _path): pass
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def open(self, _name): return BytesIO(json.dumps(ONTOLOGY).encode())
    def chat(prompt, **kwargs):
        tag = kwargs["call_tag"]
        if tag == "sgd_call_selection": return '{"call":null}'
        if tag == "sgd_dialogue_acts": return '{"acts":[]}'
        if tag == "sgd_ours_diagnosis":
            payload = json.loads(prompt[prompt.index('{"resources"'):])
            key = next(iter(payload["resources"]["action_rules"]))
            return json.dumps({"updates": [{"resource": "action_rules", "key": key, "text": "proposal"}]})
        if tag == "sgd_ours_reflection":
            payload = json.loads(prompt[prompt.index('{"targets"'):])
            item = payload["targets"][0]
            return json.dumps({"updates": [{"resource": item["resource"], "key": item["key"], "text": "final"}]})
        return '{"updates":[]}'
    parser = argparse.ArgumentParser()
    runner.add_ours_arguments(parser)
    args = parser.parse_args(["--ours-refinement-mode", "prototype", "--ours-selection-rate", "1"])
    vars(args).update(method="ours", domain="Restaurants", split="test", splits_dir=tmp_path,
                      archive=tmp_path / "fake.zip", output_dir=tmp_path / "run", model="mock", max_dialogues=1)
    monkeypatch.setattr(runner, "iter_family_dialogues", dialogues)
    monkeypatch.setattr(runner, "ZipFile", Archive)
    monkeypatch.setattr("llm.resolve_config", lambda **_kwargs: {})
    monkeypatch.setattr("llm.chat", chat)
    summary = runner.run_domain(args)
    assert seen_splits == ["train", "test"]
    assert summary["total_usage"]["calls"] == 7
    assert summary["mining_usage"]["calls"] == 5
    assert summary["usage"]["calls"] == 2
    assert summary["counts"]["turns"] == 1
    assert summary["total_usage"]["usage_source"] == "estimated"
    state_bytes = (args.output_dir / "sgd_skill_state.json").read_bytes()
    args.ours_resource_dir = args.output_dir
    args.output_dir = tmp_path / "frozen"
    seen_splits.clear()
    frozen = runner.run_domain(args)
    assert seen_splits == ["test"]
    assert frozen["mining_usage"]["calls"] == 0
    assert frozen["total_usage"]["calls"] == 2
    assert (args.ours_resource_dir / "sgd_skill_state.json").read_bytes() == state_bytes
