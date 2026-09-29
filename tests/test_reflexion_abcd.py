"""ABCD Reflexion memory, feedback boundary, and crash resume."""

from __future__ import annotations

import json
import sys

from reflexion_adapter.abcd import (
    Reflection, ReflectionStore, ReflexionABCDAgent, build_reflection_feedback,
    _redact_training_values, generate_reflection,
)
from eval_tod.abcd.agent import ABCDAgent
from scripts import run_reflexion_abcd
from scripts import evaluate_abcd_method
from scripts.aggregate_subflow_results import _records_from_summary


def _conversation(cid: str) -> dict:
    return {
        "convo_id": cid,
        "scenario": {"subflow": "example"},
        "delexed": [
            {"speaker": "customer", "text": "Please find my account"},
            {"speaker": "action", "text": "Account found",
             "targets": ["x", "take_action", "pull-up-account", ["private-id"], -1]},
        ],
    }


def test_reflection_selection_is_scoped_and_uses_visible_query():
    store = ReflectionStore([
        Reflection("train_a", 1, "find account", "Check account", 0.0),
        Reflection("train_a", 2, "find account", "Check slot order", 0.5),
        Reflection("train_b", 1, "cancel order", "Check cancellation", 0.0),
    ])
    own = store.select("", conversation_id="train_a", same_conversation=True)
    assert [row.text for row in own] == ["Check account", "Check slot order"]
    test = store.select("find account", conversation_id="test_x", same_conversation=False, limit=1)
    assert test[0].text == "Check slot order"


def test_training_feedback_includes_gold_action_and_ordered_slots():
    feedback = build_reflection_feedback(
        _conversation("train_a"),
        [{"convo_id": "train_a", "turn_index": 1, "target_type": "action",
          "predicted_action": "verify-identity", "predicted_slots": []}],
        {"ast_score": 0.0, "action_correct": 0, "action_total": 1},
    )
    assert feedback["errors"][0]["expected_action"] == "pull-up-account"
    assert feedback["errors"][0]["expected_slots"] == ["private-id"]
    assert feedback["errors"][0]["predicted_slots"] == []
    assert _redact_training_values("Use private-id for the account", _conversation("train_a")) == (
        "Use [customer_value] for the account"
    )


def test_reflector_sees_train_gold_slots_but_persisted_note_is_redacted(monkeypatch):
    captured = []

    def fake_chat(prompt, **_kwargs):
        captured.append(prompt)
        return "Next time use private-id in the account slot."

    monkeypatch.setattr("llm.chat", fake_chat)
    feedback = {"errors": [{"expected_action": "pull-up-account",
                            "expected_slots": ["private-id"]}]}
    note = generate_reflection(_conversation("train_a"), 1, feedback, [])
    assert '"expected_slots": ["private-id"]' in captured[0]
    assert "private-id" not in note


def test_actor_injects_only_selected_reflections(monkeypatch):
    store = ReflectionStore([
        Reflection("train_a", 1, "find account", "Check action", 0.0),
        Reflection("train_b", 1, "cancel order", "Check cancellation", 0.0),
    ])
    monkeypatch.setattr(ABCDAgent, "_build_system_prompt", lambda *_args, **_kwargs: "base prompt")
    actor = object.__new__(ReflexionABCDAgent)
    actor.reflection_store = store
    actor.same_conversation = True
    actor.reflection_limit = 2
    actor._active_conversation_id = "train_a"
    prompt = actor._build_system_prompt({}, context="find account")
    assert "Check action" in prompt and "Check cancellation" not in prompt
    actor.same_conversation = False
    actor._active_conversation_id = "test_c"
    actor.reflection_limit = 1
    prompt = actor._build_system_prompt({}, context="find account")
    assert "Check action" in prompt and "Check cancellation" not in prompt


def test_each_reflection_persists_and_resume_skips_completed_trials(tmp_path, monkeypatch):
    train = tmp_path / "train.json"
    test = tmp_path / "test.json"
    train.write_text(json.dumps([_conversation("train_a")]), encoding="utf-8")
    test.write_text(json.dumps([_conversation("test_b")]), encoding="utf-8")
    out = tmp_path / "output"
    calls = []

    class FakeActor:
        def __init__(self, **kwargs):
            self.store = kwargs["reflection_store"]

        def generate_all_turn_predictions(self, conversations, **kwargs):
            calls.append(len(self.store.reflections))
            return [{"convo_id": conversations[0]["convo_id"], "turn_index": 1,
                     "target_type": "action", "predicted_action": "wrong",
                     "predicted_slots": []}]

    def metrics(_conversations, _turns):
        correct = int(len(calls) == 2)
        return [{"ast_score": float(correct), "action_correct": correct, "action_total": 1}]

    monkeypatch.setattr(run_reflexion_abcd, "ReflexionABCDAgent", FakeActor)
    monkeypatch.setattr(run_reflexion_abcd, "compute_ast_from_turn_results", metrics)
    monkeypatch.setattr(run_reflexion_abcd, "build_reflection_feedback", lambda *_args: {"ast_score": 0})
    monkeypatch.setattr(run_reflexion_abcd, "generate_reflection", lambda *_args, **_kwargs: "Check action")
    argv = ["run_reflexion_abcd.py", "--subflow", "example", "--train-file", str(train),
            "--test-file", str(test), "--output-dir", str(out), "--skip-final-test"]
    monkeypatch.setattr(sys, "argv", argv)
    run_reflexion_abcd.main()

    rows = [json.loads(line) for line in (out / "training_trials.jsonl").read_text().splitlines()]
    assert [row["trial_index"] for row in rows] == [1, 2]
    assert [row["reflection"] for row in rows] == ["Check action", ""]
    assert calls == [0, 1]
    assert len(ReflectionStore.load(out / "reflections.json").reflections) == 1
    assert (out / "llm_usage.json").is_file()

    monkeypatch.setattr(sys, "argv", argv + ["--resume-from", str(out)])
    run_reflexion_abcd.main()
    assert calls == [0, 1]
    assert len((out / "training_trials.jsonl").read_text().splitlines()) == 2

    def fake_evaluate(method, resource, conversations, model, output):
        assert method == "reflexion" and resource == out
        assert len(ReflectionStore.load(resource / "reflections.json").reflections) == 1
        output.mkdir()
        (output / "result.json").write_text(json.dumps({
            "summary": {"ast_joint": 0.5},
            "text": {}, "ast_cds": {},
            "llm_usage": {"testing": {"total": {"calls": 2, "successful_calls": 2}}},
        }), encoding="utf-8")

    monkeypatch.setattr(evaluate_abcd_method, "_evaluate_rows", fake_evaluate)
    monkeypatch.setattr(sys, "argv", [item for item in argv if item != "--skip-final-test"]
                        + ["--resume-from", str(out)])
    run_reflexion_abcd.main()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_test"]["summary"]["ast_joint"] == 0.5
    assert summary["llm_usage"]["testing"]["total"]["calls"] == 2
    assert summary["final_test"]["llm_usage"] == summary["llm_usage"]
    records = _records_from_summary(out / "summary.json")
    assert len(records) == 1 and records[0]["method"] == "reflexion"
    assert records[0]["llm_usage"]["testing"]["total"]["calls"] == 2
    assert calls == [0, 1]


def test_shared_evaluator_loads_frozen_training_memory(tmp_path, monkeypatch):
    store = ReflectionStore([Reflection("train_a", 1, "find account", "Check action", 0.0)])
    store.save(tmp_path / "reflections.json")
    (tmp_path / "summary.json").write_text(
        json.dumps({"config": {"reflection_limit": 2}}), encoding="utf-8",
    )
    captured = {}

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("reflexion_adapter.ReflexionABCDAgent", fake_agent)
    evaluate_abcd_method._build_agent("reflexion", tmp_path, "deepseek-chat", None)
    assert captured["same_conversation"] is False
    assert captured["reflection_limit"] == 2
    assert captured["reflection_store"].reflections[0].conversation_id == "train_a"
