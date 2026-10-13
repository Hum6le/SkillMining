import argparse
import json
from io import BytesIO

from eval_tod.sgd_adapter import gold_policy, score_policies
from eval_tod.sgd_errors import error_summary
from tests.test_sgd_standard_skill_agent import make_agent
from tests.test_sgd_skill_agent import dialogue, ONTOLOGY


def test_failed_empty_policy_cannot_count_as_correct():
    empty = {"call": None, "acts": []}
    result = score_policies([empty], [{**empty, "_execution_error": True}])
    assert result["counts"]["turns"] == 1
    assert result["counts"]["execution_failed_turns"] == 1
    assert result["action_set_exact"] == result["turn_joint_ast"] == 0


def test_standard_keeps_running_after_rollout_and_diagnosis_errors(monkeypatch, tmp_path):
    from eval_tod import sgd_standard_skill_agent as module
    agent = make_agent(monkeypatch, tmp_path)
    visited = []

    def predict(d, index):
        visited.append(d["dialogue_id"])
        if d["dialogue_id"] == "bad":
            raise RuntimeError("workflow not ready to run")
        return gold_policy(d, index), {}

    def diagnose(batch_id, *args, **kwargs):
        if batch_id == "1":
            return {"status": "error", "error": "empty_or_invalid_json_response"}
        return {"status": "ok", "batch_id": batch_id, "summary": "ok"}

    monkeypatch.setattr(module, "diagnose_rollout_batch", diagnose)
    monkeypatch.setattr(module, "reflect_batch_report_group", lambda *args, **kwargs: {
        "status": "ok", "updates": [], "skill_operations": []})
    agent.predict = predict
    result = agent.refine([dialogue("bad"), dialogue("good1"), dialogue("good2")], tmp_path,
                          analysis_batch_size=1)
    assert visited == ["bad", "good1", "good2"]
    assert result["rollout_turns"] == 2 and result["failed_rollout_turns"] == 1
    assert result["reports"] == 1
    assert error_summary(tmp_path)["by_phase"] == {"train_rollout": 1, "online_diagnosis": 1}


def test_malformed_write_rolls_back_group_and_next_group_runs(monkeypatch, tmp_path):
    from eval_tod import sgd_standard_skill_agent as module
    agent = make_agent(monkeypatch, tmp_path)
    agent.predict = lambda d, i: (gold_policy(d, i), {})
    monkeypatch.setattr(module, "diagnose_rollout_batch", lambda bid, *args, **kwargs: {
        "batch_id": bid, "status": "ok", "summary": "ok"})
    monkeypatch.setattr(module, "group_batch_reports", lambda reports, max_reports: [[r] for r in reports])
    seen = []

    def reflect(group, state, skill, *args, **kwargs):
        seen.append(skill)
        if len(seen) == 1:
            edge_id = next(iter(state["edges"]))
            return {"status": "ok", "updates": [{"resource": "transition_guard", "edge_id": edge_id,
                    "op": "upsert", "content": "supported branch", "status": "resolved"}],
                    "skill_operations": [{"op": "delete", "match_text": "<!-- ACTION_RULES_START -->"}]}
        return {"status": "ok", "updates": [], "skill_operations": []}

    monkeypatch.setattr(module, "reflect_batch_report_group", reflect)
    agent.refine([dialogue("one"), dialogue("two")], tmp_path, analysis_batch_size=1)
    assert len(seen) == 2 and seen[0] == seen[1]
    assert "<!-- ACTION_RULES_START -->" in agent.working_skill
    assert error_summary(tmp_path)["by_phase"] == {"online_reflection": 1}
    assert not agent.state.get("applied_skill_operation_ids")
    assert (tmp_path / "reflection_0001_error.json").is_file()


def test_missing_action_rules_legacy_skill_can_promote(monkeypatch, tmp_path):
    from eval_tod import sgd_standard_skill_agent as module
    agent = make_agent(monkeypatch, tmp_path)
    agent.working_skill = "# Legacy SGD skill\n"
    agent.predict = lambda d, i: (gold_policy(d, i), {})
    monkeypatch.setattr(module, "diagnose_rollout_batch", lambda bid, *args, **kwargs: {
        "batch_id": bid, "status": "ok", "summary": "ok"})

    def reflect(group, state, *args, **kwargs):
        return {"status": "ok", "updates": [{"resource": "transition_guard",
            "edge_id": next(iter(state["edges"])), "op": "upsert", "content": "supported", "status": "resolved"}]}

    monkeypatch.setattr(module, "reflect_batch_report_group", reflect)
    agent.refine([dialogue()], tmp_path)
    assert "<!-- ACTION_RULES_START -->" in agent.working_skill
    assert not error_summary(tmp_path)["total"]


def test_testing_error_is_recorded_and_keeps_denominator(monkeypatch, tmp_path):
    from scripts import run_sgd_domain_eval as runner

    class Archive:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def open(self, name): return BytesIO(json.dumps(ONTOLOGY).encode())

    monkeypatch.setattr(runner, "ZipFile", Archive)
    monkeypatch.setattr("llm.resolve_config", lambda **kwargs: {})
    monkeypatch.setattr(runner, "iter_family_dialogues", lambda *args: iter([dialogue("bad"), dialogue("good")]))
    visited = []

    def predict(d, index, **kwargs):
        visited.append(d["dialogue_id"])
        if d["dialogue_id"] == "bad":
            raise RuntimeError("Workflow returned ret=1: workflow not ready to run")
        return gold_policy(d, index), {}

    monkeypatch.setattr(runner, "predict_policy", predict)
    args = argparse.Namespace(output_dir=tmp_path, method="react", domain="Restaurants", model="mock",
        splits_dir=tmp_path, archive=tmp_path / "mock.zip", split="test", max_dialogues=None)
    result = runner.run_domain(args)
    assert visited == ["bad", "good"]
    assert result["counts"]["turns"] == 2 and result["counts"]["execution_failed_turns"] == 1
    assert result["turn_joint_ast"] == .5
    assert result["execution_status"] == "completed_with_errors"
    assert (tmp_path / "summary.json").is_file()
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["diagnostic"]["status"] == "error" and len(rows) == 2
