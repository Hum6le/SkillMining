import json

import pytest

from eval_tod.sgd_standard_skill_agent import normalize_action_heads, SGDStandardSkillAgent
from tests.test_sgd_skill_agent import dialogue, ONTOLOGY


def test_format_normalization_and_duplicate_call_are_unambiguous():
    result = normalize_action_heads({"action_heads": [
        [" CALL ", "restaurants"], ["call", "Restaurants_1"],
        {"intent": "REQUEST", "domain": "Restaurants"}, ["GOODBYE", None]]}, ["Restaurants_1"])
    assert result == [["call", "Restaurants_1"], ["goodbye", ""], ["request", "Restaurants_1"]]


@pytest.mark.parametrize("heads,services", [
    ([["call", "Restaurants"]], ["Restaurants_1", "Restaurants_2"]),
    ([["call", "Restaurants_2"]], ["Restaurants_1"]),
    ([["invented", "Restaurants_1"]], ["Restaurants_1"]),
    ([["call", ""]], ["Restaurants_1"]),
    ([["call", "Restaurants_1"], ["call", "Restaurants_2"]], ["Restaurants_1", "Restaurants_2"]),
    ([[[], "Restaurants_1"]], ["Restaurants_1"]),
])
def test_normalization_does_not_guess_invalid_semantics(heads, services):
    with pytest.raises(ValueError):
        normalize_action_heads({"action_heads": heads}, services)


def test_invalid_output_repair_is_bounded_logged_and_counted(monkeypatch, tmp_path):
    from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase
    configure_usage(tmp_path)
    set_phase("train_rollout")
    replies = iter(['{"action_heads":[["call","WRONG"]]}',
                    '{"action_heads":[["REQUEST","Restaurants"]]}'])
    tags = []
    def chat(prompt, **kwargs):
        assert "CURRENT_SYSTEM_SECRET" not in prompt
        tags.append(kwargs["call_tag"])
        return next(replies)
    monkeypatch.setattr("llm.chat", chat)
    agent = SGDStandardSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY, predictor=None)
    agent.runtime_diagnostics_dir = tmp_path
    heads, diagnostic = agent._select_heads("visible prior context", dialogue(), 1)
    assert heads == [["request", "Restaurants_1"]]
    assert diagnostic["attempts"] == 2 and diagnostic["normalized"]
    assert tags == ["sgd_action_selection", "sgd_action_selection_repair"]
    logged = [json.loads(line) for line in (tmp_path / "action_selection_failures.jsonl").read_text().splitlines()]
    assert len(logged) == 1 and "WRONG" in logged[0]["raw_response"]
    assert persist_usage()["generation"]["total"]["calls"] == 2


def test_exhausted_repairs_remain_explicit_failure(monkeypatch, tmp_path):
    calls = []
    def chat(*_args, **kwargs):
        calls.append(kwargs["call_tag"])
        return '{"action_heads":[["bad","Restaurants_1"]]}'
    monkeypatch.setattr("llm.chat", chat)
    agent = SGDStandardSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY, predictor=None)
    agent.runtime_diagnostics_dir = tmp_path
    with pytest.raises(ValueError, match="failed after 3 attempts.*train-1"):
        agent._select_heads("visible context", dialogue(), 1)
    assert len(calls) == 3
    assert len((tmp_path / "action_selection_failures.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("raw", [
    '{"query_text":"booking\ntime\tplease","candidate_actions":[]}',
    'Here is the JSON:\n```json\n{"query_text":"booking","candidate_actions":[]}\n```',
    '<think>Reasoning with {not JSON}</think>\n{"query_text":"booking","candidate_actions":[]}',
])
def test_json_parser_tolerates_control_chars_and_response_wrappers(raw):
    from eval_tod.sgd_skill_agent import _json_object
    result = _json_object(raw)
    assert result["query_text"].startswith("booking")


@pytest.mark.parametrize("raw", [
    '{"query_text":"truncated',
    '{"broken": [ {"query_text":"nested valid object must not replace broken parent"}',
    '[{"query_text":"array element is not the requested root object"}]',
])
def test_json_parser_does_not_invent_or_salvage_broken_structure(raw):
    from eval_tod.sgd_skill_agent import _json_object
    with pytest.raises(ValueError):
        _json_object(raw)


def test_reference_planner_retries_and_preserves_raw_output_and_usage(monkeypatch, tmp_path):
    from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase
    configure_usage(tmp_path)
    set_phase("train_rollout")
    replies = iter(['not json', '{"query_text":"book\na table","candidate_actions":[]}'])
    tags = []
    def chat(prompt, **kwargs):
        assert "CURRENT_SYSTEM_SECRET" not in prompt
        tags.append(kwargs["call_tag"])
        return next(replies)
    monkeypatch.setattr("llm.chat", chat)
    agent = SGDStandardSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY, predictor=None)
    agent.runtime_diagnostics_dir = tmp_path
    plan, diagnostic = agent._request_validated_json("visible context", dialogue(), 1,
        stage="reference_plan", validator=agent._validate_reference_plan)
    assert plan["query_text"] == "book\na table"
    assert diagnostic["attempts"] == 2
    assert tags == ["sgd_reference_plan", "sgd_reference_plan_repair"]
    logged = json.loads((tmp_path / "reference_plan_failures.jsonl").read_text())
    assert logged["raw_response"] == "not json" and logged["stage"] == "reference_plan"
    assert persist_usage()["generation"]["total"]["calls"] == 2


def test_reference_planner_exhaustion_has_stage_and_trace_identity(monkeypatch, tmp_path):
    monkeypatch.setattr("llm.chat", lambda *_args, **_kwargs: '{"query_text":[] }')
    agent = SGDStandardSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY, predictor=None)
    agent.runtime_diagnostics_dir = tmp_path
    with pytest.raises(ValueError, match="reference plan failed after 3 attempts.*train-1"):
        agent._request_validated_json("visible context", dialogue(), 1,
            stage="reference_plan", validator=agent._validate_reference_plan)
    assert len((tmp_path / "reference_plan_failures.jsonl").read_text().splitlines()) == 3
