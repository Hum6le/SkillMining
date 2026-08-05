from __future__ import annotations

import json
import unittest

from skill_disco import (
    annotate_trace_semantics,
    build_semantic_abstraction_prompt,
    normalize_abcd_conversation,
)


class SemanticAbstractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = normalize_abcd_conversation(
            {
                "convo_id": 9,
                "scenario": {
                    "flow": "account",
                    "subflow": "recover_password",
                    "personal": {"username": "ada"},
                },
                "original": [
                    ["customer", "My username is ada and I forgot my password."],
                    ["action", "Password reset started."],
                ],
                "delexed": [
                    {"speaker": "customer", "text": "My username is ada and I forgot my password.", "targets": ["recover_password", None, None, [], -1]},
                    {"speaker": "action", "text": "Password reset started.", "targets": ["recover_password", "take_action", "reset-password", ["ada"], -1]},
                ],
            }
        )

    def test_prompt_hides_subflow_but_exposes_parameterized_events(self) -> None:
        prompt = build_semantic_abstraction_prompt(self.trace)
        self.assertNotIn("recover_password", prompt)
        self.assertIn("{username}", prompt)
        self.assertIn("reset-password(username)", prompt)

    def test_parses_complete_annotations_from_injected_llm(self) -> None:
        response = json.dumps(
            {
                "events": [
                    {
                        "turn_index": 0,
                        "dialogue_act": "request_password_reset",
                        "intent": "password_recovery",
                        "state_updates": ["reset_requested", "username_available"],
                        "parameters": ["username", "not_allowed"],
                        "control_signal": "start",
                    },
                    {
                        "turn_index": 1,
                        "dialogue_act": "backend_action",
                        "intent": "password_recovery",
                        "state_updates": ["password_reset_started"],
                        "parameters": ["username"],
                        "control_signal": "advance",
                    },
                ]
            }
        )

        annotations, raw_output = annotate_trace_semantics(
            self.trace, lambda *_args, **_kwargs: response
        )

        self.assertEqual(raw_output, response)
        self.assertEqual(len(annotations), 2)
        self.assertEqual(annotations[0].dialogue_act, "request_password_reset")
        self.assertEqual(annotations[0].parameters, ["username"])
        self.assertEqual(annotations[1].control_signal, "advance")


if __name__ == "__main__":
    unittest.main()
