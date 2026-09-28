from __future__ import annotations

import json

from skill_disco.skill_specification import SkillContract, SkillParameter
from skill_disco.synthesis import VerificationResult, synthesize_and_verify, validate_skill_source
from skill_disco.abcd_verification import build_replay_cases, verify_abcd_replay


def _contract() -> SkillContract:
    return SkillContract(
        cluster_id="cluster_000", skill_name="recover_account",
        description="Recover the account", docstring="Use the username from the dialogue",
        parameters=[SkillParameter("username", "str", "Account username", True, None)],
        return_type="dict", preconditions=["username_available"],
        postconditions=["account_found"], side_effects=["account_accessed"],
        canonical_action_sequence=["pull-up-account(username)", "make-password()"],
        abstraction_level="composite", estimated_actions_saved=1,
        confidence_score=1.0, supporting_conversations=["1", "2"],
        source_operation_ids=["1:0-1", "2:0-1"],
    )


def test_synthesis_retries_with_verification_feedback() -> None:
    prompts: list[str] = []
    code = (
        "def recover_account(username):\n"
        "    observation, available_actions = env.step('pull-up-account', [username])\n"
        "    return {'success': True, 'observation': observation, "
        "'available_actions': available_actions, 'process_trace': []}"
    )

    def chat(prompt: str, **_kwargs: object) -> str:
        prompts.append(prompt)
        return json.dumps({"implementation": code})

    checks = 0

    def verify(_source: str, _contract: SkillContract) -> VerificationResult:
        nonlocal checks
        checks += 1
        return VerificationResult(
            passed=checks == 2, runtime_correct=True,
            postconditions_met=checks == 2, actions_saved=1,
            cases_passed=int(checks == 2), cases_total=1,
            feedback="missing second action",
        )

    result = synthesize_and_verify(
        _contract(), [], chat, verify, environment_note="ABCD replay", max_attempts=2,
    )
    assert result["status"] == "verified"
    assert len(result["attempts"]) == 2
    assert "missing second action" in prompts[1]


def test_synthesis_rejects_code_without_env_step() -> None:
    def chat(_prompt: str, **_kwargs: object) -> str:
        return json.dumps({"implementation": "def recover_account(username):\n    return {}"})

    def verify(_source: str, _contract: SkillContract) -> VerificationResult:
        raise AssertionError("invalid source must never reach verification")

    result = synthesize_and_verify(
        _contract(), [], chat, verify, environment_note="ABCD replay", max_attempts=1,
    )
    assert result["status"] == "discarded"
    assert "env.step" in result["attempts"][0]["error"]
    try:
        validate_skill_source(
            "def recover_account(username):\n    env.reset()\n    return env.step('x', [])",
            "recover_account",
        )
    except ValueError as error:
        assert "only step" in str(error)
    else:
        raise AssertionError("non-step environment calls must be rejected")


def test_abcd_replay_checks_actions_slots_and_return_protocol() -> None:
    conversation = {
        "convo_id": "heldout-1", "scenario": {"personal": {"username": "alice"}},
        "original": [["customer", "My username is alice"], ["action", "Found"],
                     ["action", "Created"]],
        "delexed": [
            {"speaker": "customer", "text": "My username is alice", "targets": ["x", None, None, [], -1]},
            {"speaker": "action", "text": "Found", "targets": ["x", "take_action", "pull-up-account", ["alice"], -1]},
            {"speaker": "action", "text": "Created", "targets": ["x", "take_action", "make-password", [], -1]},
        ],
    }
    cases = build_replay_cases(_contract(), [conversation])
    assert len(cases) == 1
    source = (
        "def recover_account(username):\n"
        "    process_trace = []\n"
        "    observation, available_actions = env.step('pull-up-account', [username])\n"
        "    process_trace.append(('pull-up-account', observation))\n"
        "    observation, available_actions = env.step('make-password', [])\n"
        "    process_trace.append(('make-password', observation))\n"
        "    return {'success': True, 'observation': observation, "
        "'available_actions': available_actions, 'process_trace': process_trace}"
    )
    assert verify_abcd_replay(source, _contract(), cases).passed
    assert not verify_abcd_replay(source.replace("'make-password'", "'verify-identity'"),
                                  _contract(), cases).passed
