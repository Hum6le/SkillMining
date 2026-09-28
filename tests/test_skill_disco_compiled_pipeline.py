from __future__ import annotations

import json
from unittest.mock import patch

from skill_disco.compiled_pipeline import (
    run_compiled_abcd_pipeline, split_induction_and_verification,
)
from skill_disco.callable_runtime import CompiledSkillLibrary
from skill_disco.operation_extraction import SemanticOperation
from skill_disco.skill_specification import SkillContract, SkillParameter


def _conversation(index: int) -> dict:
    username = f"user{index}"
    return {
        "convo_id": str(index), "scenario": {"personal": {"username": username}},
        "original": [["customer", f"My username is {username}"],
                     ["action", "Found"], ["action", "Created"]],
        "delexed": [
            {"speaker": "customer", "text": f"My username is {username}", "targets": ["x", None, None, [], -1]},
            {"speaker": "action", "text": "Found", "targets": ["x", "take_action", "pull-up-account", [username], -1]},
            {"speaker": "action", "text": "Created", "targets": ["x", "take_action", "make-password", [], -1]},
        ],
    }


def test_compiled_pipeline_keeps_verification_cases_out_of_discovery() -> None:
    conversations = [_conversation(index) for index in range(5)]
    induction, verification = split_induction_and_verification(conversations)
    source_operation_id = f"{induction[0]['convo_id']}:0-1:recover"
    contract = SkillContract(
        cluster_id="cluster_000", skill_name="recover_account",
        description="Recover account", docstring="Use supplied username",
        parameters=[SkillParameter("username", "str", "Customer username", True, None)],
        return_type="dict", preconditions=[], postconditions=["password_created"],
        side_effects=["account_accessed"],
        canonical_action_sequence=["pull-up-account(username)", "make-password()"],
        abstraction_level="composite", estimated_actions_saved=1,
        confidence_score=0.8, supporting_conversations=[induction[0]["convo_id"]],
        source_operation_ids=[source_operation_id],
    )
    operation = SemanticOperation(
        operation_id=source_operation_id, conversation_id=induction[0]["convo_id"],
        name="recover", description="Recover account", action_start_index=0,
        action_end_index=1, action_turn_indices=[1, 2],
        action_sequence=contract.canonical_action_sequence,
        preconditions=[], postconditions=[], control_flow="fixed_sequence",
        parameters=["username"], supporting_event_turns=[0],
        completion_evidence="Created", code_snippet="env.step('make-password', [])",
    )
    discovered = {"stages": ["trace_normalization", "skill_specification"],
                  "traces": [{"operations": [operation.to_dict()]}],
                  "contracts": [contract.to_dict()], "skill_library": "old"}
    implementation = (
        "def recover_account(username):\n"
        "    process_trace = []\n"
        "    observation, available_actions = env.step('pull-up-account', [username])\n"
        "    process_trace.append(('pull-up-account', observation))\n"
        "    observation, available_actions = env.step('make-password', [])\n"
        "    process_trace.append(('make-password', observation))\n"
        "    return {'success': True, 'observation': observation, "
        "'available_actions': available_actions, 'process_trace': process_trace}"
    )
    seen = []

    def fake_discovery(rows, *_args, **_kwargs):
        seen.extend(row["convo_id"] for row in rows)
        return discovered.copy()

    with patch("skill_disco.compiled_pipeline.run_offline_pseudocode_pipeline", fake_discovery):
        artifact = run_compiled_abcd_pipeline(
            conversations, lambda *_args, **_kwargs: json.dumps({"implementation": implementation}),
        )
    assert set(seen) == {row["convo_id"] for row in induction}
    assert not set(seen) & {row["convo_id"] for row in verification}
    assert artifact["compiled_skills"][0]["status"] == "verified"
    assert len(artifact["verified_contracts"]) == 1
    assert "### Skill: recover_account" in artifact["skill_library"]

    class LiveEnvironment:
        def __init__(self):
            self.calls = []

        def step(self, action, slots):
            self.calls.append((action, slots))
            return f"done:{action}", ["pull-up-account", "make-password"]

    live = LiveEnvironment()
    library = CompiledSkillLibrary(artifact)
    result = library.invoke("recover_account", {"username": "charlie"}, live)
    assert result["success"] is True
    assert live.calls == [
        ("pull-up-account", ["charlie"]), ("make-password", []),
    ]
    assert len(library.tool_specs()) == 1


def test_all_five_stages_run_with_a_scripted_model() -> None:
    conversations = [_conversation(index) for index in range(3)]
    induction, heldout = split_induction_and_verification(conversations)
    responses = []
    for row in induction:
        responses.extend([
            {"events": [
                {"turn_index": 0, "dialogue_act": "provide_username", "intent": "recover_account",
                 "state_updates": ["username_available"], "parameters": ["username"], "control_signal": "start"},
                {"turn_index": 1, "dialogue_act": "backend_action", "intent": "recover_account",
                 "state_updates": ["account_found"], "parameters": ["username"], "control_signal": "advance"},
                {"turn_index": 2, "dialogue_act": "backend_action", "intent": "recover_account",
                 "state_updates": ["password_created"], "parameters": [], "control_signal": "complete"},
            ]},
            {"operations": [{
                "name": "recover", "description": "Recover the account", "start_action_index": 0,
                "end_action_index": 1, "preconditions": ["username_available"],
                "postconditions": ["password_created"], "control_flow": "fixed_sequence",
                "parameters": ["username"], "supporting_event_turns": [0],
                "completion_evidence": "Created", "succeeded": True,
            }]},
        ])
    operation_ids = [f"{row['convo_id']}:0-1:recover" for row in induction]
    responses.extend([
        {"groups": [{"name": "recover", "description": "Account recovery",
                     "operation_ids": operation_ids}]},
        {"clusters": [{"name": "recover", "description": "Account recovery",
                       "group_ids": ["batch0000_group000"]}]},
        {"skill_name": "recover_account", "description": "Recover the account",
         "docstring": "Use the observed username", "parameters": [
             {"name": "username", "type": "str", "description": "Current username",
              "required": True, "default": None}],
         "preconditions": ["username_available"], "postconditions": ["password_created"],
         "side_effects": ["account_accessed"],
         "canonical_action_sequence": ["pull-up-account(username)", "make-password()"],
         "abstraction_level": "composite"},
    ])
    implementation = (
        "def recover_account(username):\n"
        "    process_trace = []\n"
        "    observation, available_actions = env.step('pull-up-account', [username])\n"
        "    process_trace.append(('pull-up-account', observation))\n"
        "    observation, available_actions = env.step('make-password', [])\n"
        "    process_trace.append(('make-password', observation))\n"
        "    return {'success': True, 'observation': observation, "
        "'available_actions': available_actions, 'process_trace': process_trace}"
    )
    responses.append({"implementation": implementation})

    def chat(*_args, **_kwargs):
        return json.dumps(responses.pop(0))

    artifact = run_compiled_abcd_pipeline(conversations, chat)
    assert not responses
    assert artifact["split"]["verification_ids"] == [heldout[0]["convo_id"]]
    assert artifact["compiled_skills"][0]["status"] == "verified"
    assert len(artifact["verified_contracts"]) == 1
