from __future__ import annotations

import json
from unittest.mock import patch

from scripts.evaluate_abcd_method import _build_agent
from skill_disco.abcd_runtime import CompiledSkillDiscoABCDAgent, summarize_skill_invocations
from skill_disco.callable_runtime import CompiledSkillLibrary


_SOURCE = (
    "def recover_account(username):\n"
    "    process_trace = []\n"
    "    observation, available_actions = env.step('pull-up-account', [username])\n"
    "    process_trace.append(('pull-up-account', observation))\n"
    "    observation, available_actions = env.step('make-password', [])\n"
    "    process_trace.append(('make-password', observation))\n"
    "    return {'success': True, 'observation': observation, "
    "'available_actions': available_actions, 'process_trace': process_trace}"
)


def _library() -> CompiledSkillLibrary:
    return CompiledSkillLibrary({"compiled_skills": [{
        "status": "verified", "implementation": _SOURCE,
        "contract": {
            "skill_name": "recover_account", "description": "Recover an account",
            "parameters": [{"name": "username", "type": "str", "required": True}],
            "preconditions": ["username supplied"], "postconditions": ["password created"],
        },
    }]})


def test_compiled_skill_predicts_current_action_without_reading_its_gold_label() -> None:
    conversation = {
        "convo_id": "1",
        "scenario": {"personal": {"username": "hidden-scenario-value"}},
        "original": [
            ["customer", "My username is alice"],
            ["action", "Account found"],
            ["agent", "I will create a password"],
            ["action", "Password created"],
        ],
        "delexed": [
            {"speaker": "customer", "text": "My username is alice", "targets": ["x", None, None, [], -1]},
            # Deliberately conflicting gold action and slot. The skill must
            # still predict from the visible prefix and its own code.
            {"speaker": "action", "text": "Account found", "targets": ["x", "take_action", "verify-identity", ["secret"], -1]},
            {"speaker": "agent", "text": "I will create a password", "targets": ["x", "retrieve_utterance", None, [], -1]},
            {"speaker": "action", "text": "Password created", "targets": ["x", "take_action", "verify-identity", ["secret"], -1]},
        ],
    }

    class BaseAgent:
        def predict_all_turns(self, _conversation, *, turn_index, **_kwargs):
            return [{"convo_id": "1", "turn_index": turn_index,
                     "target_type": "utterance", "prediction": "okay",
                     "predicted_action": "", "predicted_slots": []}]

    prompts = []

    def fake_chat(messages, **_kwargs):
        prompts.append(messages[0]["content"])
        return json.dumps({"skill_name": "recover_account", "arguments": {"username": "alice"}})

    with patch("skill_disco.abcd_runtime.create_skill_disco_abcd_agent", return_value=BaseAgent()), \
         patch("skill_disco.abcd_runtime.chat", side_effect=fake_chat):
        agent = CompiledSkillDiscoABCDAgent(_library(), "verified guidance")
        rows = agent.generate_all_turn_predictions([conversation], predict_actions=True,
                                                   verbose=False)

    action_rows = [row for row in rows if row.get("target_type") == "action"]
    assert [row["predicted_action"] for row in action_rows] == ["pull-up-account", "make-password"]
    assert action_rows[0]["predicted_slots"] == ["alice"]
    assert action_rows[1]["skill_invocation"]["matched_prior_actions"] == 1
    assert all("hidden-scenario-value" not in prompt for prompt in prompts)
    assert "Account found" not in prompts[0]
    assert "Password created" not in prompts[1]
    usage = summarize_skill_invocations(rows)
    assert usage["skill_invoked_turns"] == 2
    assert usage["base_fallback_turns"] == 0


def test_unified_evaluator_loads_compiled_skill_artifact(tmp_path) -> None:
    (tmp_path / "SKILL.md").write_text("### Skill: recover_account", encoding="utf-8")
    (tmp_path / "generation_artifact.json").write_text(json.dumps({
        "compiled_skills": [{
            "status": "verified", "implementation": _SOURCE,
            "contract": {
                "skill_name": "recover_account", "description": "Recover an account",
                "parameters": [{"name": "username", "required": True}],
            },
        }],
    }), encoding="utf-8")
    with patch("skill_disco.abcd_runtime.create_skill_disco_abcd_agent", return_value=object()):
        agent = _build_agent("skill_disco", tmp_path, "deepseek-chat", None)
    assert isinstance(agent, CompiledSkillDiscoABCDAgent)
    assert agent.library.tool_specs()[0]["name"] == "recover_account"
