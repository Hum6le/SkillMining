from __future__ import annotations

import json
import unittest

from skill_disco.pipeline import run_offline_pseudocode_pipeline


def _conversation(conversation_id: str, username: str) -> dict:
    return {
        "convo_id": conversation_id,
        "scenario": {"personal": {"username": username}},
        "original": [
            ["customer", f"I forgot my password. My username is {username}."],
            ["action", "Account found."],
            ["action", "A password was generated."],
        ],
        "delexed": [
            {"speaker": "customer", "text": "I forgot my password.", "targets": ["hidden_label", None, None, [], -1]},
            {"speaker": "action", "text": "Account found.", "targets": ["hidden_label", "take_action", "pull-up-account", [username], -1]},
            {"speaker": "action", "text": "A password was generated.", "targets": ["hidden_label", "take_action", "make-password", [], -1]},
        ],
    }


class MockPipelineTest(unittest.TestCase):
    def test_end_to_end_pipeline_uses_only_mocked_model_outputs(self) -> None:
        responses = [
            {"events": [
                {"turn_index": 0, "dialogue_act": "request_password_reset", "intent": "password_recovery", "state_updates": ["reset_requested", "username_available"], "parameters": ["username"], "control_signal": "start"},
                {"turn_index": 1, "dialogue_act": "backend_action", "intent": "password_recovery", "state_updates": ["account_found"], "parameters": ["username"], "control_signal": "advance"},
                {"turn_index": 2, "dialogue_act": "backend_action", "intent": "password_recovery", "state_updates": ["password_generated"], "parameters": [], "control_signal": "complete"},
            ]},
            {"operations": [{"name": "recover_account_password", "description": "Recover a password.", "start_action_index": 0, "end_action_index": 1, "preconditions": ["reset_requested"], "postconditions": ["password_generated"], "control_flow": "fixed_sequence", "parameters": ["username"], "supporting_event_turns": [0], "completion_evidence": "A password was generated.", "succeeded": True}]},
            {"events": [
                {"turn_index": 0, "dialogue_act": "request_password_reset", "intent": "password_recovery", "state_updates": ["reset_requested", "username_available"], "parameters": ["username"], "control_signal": "start"},
                {"turn_index": 1, "dialogue_act": "backend_action", "intent": "password_recovery", "state_updates": ["account_found"], "parameters": ["username"], "control_signal": "advance"},
                {"turn_index": 2, "dialogue_act": "backend_action", "intent": "password_recovery", "state_updates": ["password_generated"], "parameters": [], "control_signal": "complete"},
            ]},
            {"operations": [{"name": "recover_account_password", "description": "Recover a password.", "start_action_index": 0, "end_action_index": 1, "preconditions": ["reset_requested"], "postconditions": ["password_generated"], "control_flow": "fixed_sequence", "parameters": ["username"], "supporting_event_turns": [0], "completion_evidence": "A password was generated.", "succeeded": True}]},
            {"groups": [{"name": "password_recovery", "description": "Recover account passwords.", "operation_ids": ["mock-1:0-1:recover_account_password", "mock-2:0-1:recover_account_password"]}]},
            {"clusters": [{"name": "password_recovery", "description": "Recover account passwords.", "group_ids": ["batch0000_group000"]}]},
            {"skill_name": "recover_account_password", "description": "Recover an account password.", "docstring": "Recover a password using a known username.", "parameters": [{"name": "username", "type": "str", "description": "Customer username.", "required": True, "default": None}], "preconditions": ["reset_requested"], "postconditions": ["password_generated"], "side_effects": ["account_accessed", "password_generated"], "canonical_action_sequence": ["pull-up-account(username)", "make-password()"], "abstraction_level": "composite"},
        ]
        calls = []
        def mock_chat(*_args, **_kwargs):
            calls.append(1)
            return json.dumps(responses.pop(0))

        artifact = run_offline_pseudocode_pipeline(
            [_conversation("mock-1", "alice"), _conversation("mock-2", "bob")], mock_chat
        )
        self.assertEqual(len(calls), 7)
        self.assertEqual(len(artifact["contracts"]), 1)
        self.assertIn("recover_account_password(username: str)", artifact["skill_library"])
        self.assertNotIn("alice", artifact["skill_library"])
        self.assertNotIn("bob", artifact["skill_library"])
        self.assertNotIn("No explicit precondition", artifact["skill_library"])

    def test_skips_trace_after_exhausted_llm_transport_retries(self) -> None:
        calls = []

        def unavailable_chat(*_args, **_kwargs):
            calls.append(1)
            raise RuntimeError("Workflow HTTP error 502")

        artifact = run_offline_pseudocode_pipeline(
            [_conversation("mock-1", "alice")], unavailable_chat
        )

        self.assertEqual(len(calls), 3)
        self.assertEqual(artifact["traces"][0]["status"], "skipped_semantic_abstraction")
        self.assertEqual(artifact["contracts"], [])
        self.assertIn("Procedural Skill Library", artifact["skill_library"])


if __name__ == "__main__":
    unittest.main()
