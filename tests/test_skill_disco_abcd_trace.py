from __future__ import annotations

import unittest

from skill_disco import normalize_abcd_conversation


class NormalizeABCDTraceTest(unittest.TestCase):
    def test_parameterizes_known_slots_and_keeps_prefix_only_context(self) -> None:
        conversation = {
            "convo_id": 7,
            "scenario": {
                "flow": "account",
                "subflow": "recover_password",
                "personal": {"customer_name": "Ada Lovelace", "username": "ada"},
            },
            "original": [
                ["customer", "Please reset my password."],
                ["agent", "What is your username?"],
                ["customer", "My username is ada and my account ID is 000123."],
                ["action", "Password reset started."],
            ],
            "delexed": [
                {"speaker": "customer", "text": "Please reset my password.", "targets": ["recover_password", None, None, [], -1]},
                {"speaker": "agent", "text": "What is your username?", "targets": ["recover_password", "retrieve_utterance", None, [], 1]},
                {"speaker": "customer", "text": "My username is ada and my account ID is 000123.", "targets": ["recover_password", None, None, [], -1]},
                {"speaker": "action", "text": "Password reset started.", "targets": ["recover_password", "take_action", "reset-password", ["ada", "email-token"], -1]},
            ],
        }

        trace = normalize_abcd_conversation(conversation)

        self.assertEqual(trace.action_count, 1)
        step = trace.steps[0]
        self.assertEqual(step.parameter_names, ["username", "arg_2"])
        self.assertEqual(step.parameterized_action, "reset-password(username, arg_2)")
        self.assertEqual(step.pre_context[-1], "My username is ada and my account ID is 000123.")
        self.assertNotIn("Password reset started.", step.pre_context)
        self.assertEqual(trace.events[2].event_type, "customer_observation")
        self.assertEqual(
            trace.events[2].parameterized_text,
            "My username is {username} and my account ID is {account_id}.",
        )
        self.assertEqual(trace.events[3].event_type, "backend_action")
        self.assertIn("reset-password", trace.to_program())

    def test_ignores_non_action_turns(self) -> None:
        trace = normalize_abcd_conversation(
            {"convo_id": 8, "scenario": {}, "delexed": [{"speaker": "agent", "text": "How can I help?", "targets": ["x", "retrieve_utterance", None, [], 0]}]}
        )
        self.assertEqual(trace.action_count, 0)
        self.assertEqual(trace.events[0].event_type, "agent_response")

    def test_parameterizes_generated_password_in_agent_response(self) -> None:
        trace = normalize_abcd_conversation(
            {
                "convo_id": 10,
                "scenario": {},
                "delexed": [
                    {
                        "speaker": "agent",
                        "text": "Your new password is Zx92Ab.",
                        "targets": ["x", "retrieve_utterance", None, [], 0],
                    }
                ],
            }
        )
        self.assertEqual(
            trace.events[0].parameterized_text,
            "Your new password is {password}.",
        )

    def test_keeps_an_empty_action_slot_as_a_positional_parameter(self) -> None:
        trace = normalize_abcd_conversation(
            {
                "convo_id": 12,
                "scenario": {"flow": "manage", "subflow": "manage_cancel"},
                "delexed": [
                    {
                        "speaker": "action",
                        "text": "A refund was offered.",
                        "targets": [
                            "manage_cancel", "take_action", "offer-refund", [""], -1
                        ],
                    }
                ],
            }
        )

        self.assertEqual(trace.steps[0].slot_values, [""])
        self.assertEqual(trace.steps[0].parameter_names, ["arg_1"])
        self.assertEqual(trace.steps[0].parameterized_action, "offer-refund(arg_1)")


if __name__ == "__main__":
    unittest.main()
