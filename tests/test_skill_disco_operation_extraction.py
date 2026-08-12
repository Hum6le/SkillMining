from __future__ import annotations

import json
import unittest

from skill_disco import (
    SemanticEventAnnotation,
    build_operation_extraction_prompt,
    extract_trace_operations,
    normalize_abcd_conversation,
)


class OperationExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = normalize_abcd_conversation(
            {
                "convo_id": 11,
                "scenario": {
                    "flow": "account",
                    "subflow": "recover_password",
                    "personal": {"username": "ada"},
                },
                "original": [
                    ["customer", "I forgot my password. My username is ada."],
                    ["action", "Account found."],
                    ["agent", "I will reset it."],
                    ["action", "Password reset started."],
                    ["action", "A password was generated."],
                ],
                "delexed": [
                    {"speaker": "customer", "text": "I forgot my password. My username is ada.", "targets": ["recover_password", None, None, [], -1]},
                    {"speaker": "action", "text": "Account found.", "targets": ["recover_password", "take_action", "pull-up-account", ["ada"], -1]},
                    {"speaker": "agent", "text": "I will reset it.", "targets": ["recover_password", "retrieve_utterance", None, [], 1]},
                    {"speaker": "action", "text": "Password reset started.", "targets": ["recover_password", "take_action", "reset-password", ["ada"], -1]},
                    {"speaker": "action", "text": "A password was generated.", "targets": ["recover_password", "take_action", "make-password", [], -1]},
                ],
            }
        )
        self.annotations = [
            SemanticEventAnnotation(event.turn_index, "backend_action" if event.event_type == "backend_action" else "inform", "password_recovery", [], ["username"] if "{username}" in event.parameterized_text else [], "advance")
            for event in self.trace.events
        ]

    def test_prompt_hides_subflow_and_uses_parameterized_actions(self) -> None:
        prompt = build_operation_extraction_prompt(self.trace, self.annotations)
        self.assertNotIn("recover_password", prompt)
        self.assertIn("pull-up-account(username)", prompt)
        self.assertIn("slot-filling behavior", prompt)

    def test_reconstructs_action_sequence_and_rejects_single_action_candidate(self) -> None:
        response = json.dumps(
            {
                "operations": [
                    {
                        "name": "recover_account_password",
                        "description": "Find the account and generate a replacement password.",
                        "start_action_index": 0,
                        "end_action_index": 2,
                        "preconditions": ["reset_requested"],
                        "postconditions": ["password_generated"],
                        "control_flow": "fixed_sequence",
                        "parameters": ["username", "not_allowed"],
                        "supporting_event_turns": [0, 99],
                        "completion_evidence": "A password was generated.",
                        "succeeded": True,
                    },
                    {
                        "name": "invalid_one_step",
                        "description": "Invalid primitive.",
                        "start_action_index": 1,
                        "end_action_index": 1,
                        "succeeded": True,
                    },
                ]
            }
        )

        operations, rejected, _ = extract_trace_operations(
            self.trace, self.annotations, lambda *_args, **_kwargs: response
        )

        self.assertEqual(len(operations), 1)
        self.assertEqual(
            operations[0].action_sequence,
            ["pull-up-account(username)", "reset-password(username)", "make-password()"],
        )
        self.assertEqual(operations[0].parameters, ["username"])
        self.assertEqual(operations[0].supporting_event_turns, [0])
        self.assertIn("observe_customer", operations[0].code_snippet)
        self.assertIn("env.step(action, slots)", operations[0].code_snippet)
        self.assertEqual(rejected[0]["reason"], "invalid_or_single_action_span")

    def test_retries_once_after_invalid_json(self) -> None:
        valid_response = json.dumps({
            "operations": [{
                "name": "recover_account_password", "description": "Recover a password.",
                "start_action_index": 0, "end_action_index": 2, "preconditions": [],
                "postconditions": ["password_generated"], "control_flow": "fixed_sequence",
                "parameters": ["username"], "supporting_event_turns": [0],
                "completion_evidence": "A password was generated.", "succeeded": True,
            }],
        })
        prompts: list[str] = []

        def chat(prompt: str, **_kwargs: object) -> str:
            prompts.append(prompt)
            return '{"operations": [}' if len(prompts) == 1 else valid_response

        operations, _, raw_output = extract_trace_operations(
            self.trace, self.annotations, chat
        )

        self.assertEqual(len(prompts), 2)
        self.assertIn("not valid JSON", prompts[1])
        self.assertEqual(len(operations), 1)
        self.assertEqual(raw_output, valid_response)


if __name__ == "__main__":
    unittest.main()
