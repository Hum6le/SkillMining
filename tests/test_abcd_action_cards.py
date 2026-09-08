import unittest
from unittest.mock import patch

from eval_tod.abcd.agent import ABCDAgent


class ActionCardRuntimeTest(unittest.TestCase):
    @patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""})
    def test_action_card_merges_rule_and_slots_and_uses_explicit_action(self, _config):
        agent = ABCDAgent(
            action_rules_text=(
                "# Action Rules\n\n#### `send-link`\nSend the requested link.\n\n"
                "#### `pull-up-account`\nRetrieve account context first.\n"
            ),
            slot_policies_text=(
                "# Slot Policies\n\n#### `send-link`\nUse the current request's link target.\n\n"
                "#### `pull-up-account`\nUse only the current account identifier.\n"
            ),
        )

        cards = agent._retrieve_action_cards(["send-link"])

        self.assertIn("#### `send-link`", cards)
        self.assertIn("Action rule:", cards)
        self.assertIn("Send the requested link.", cards)
        self.assertIn("Slot binding policy:", cards)
        self.assertIn("current request's link target", cards)
        self.assertNotIn("pull-up-account", cards)
        self.assertEqual(agent._last_action_card_lookup["selected_actions"], ["send-link"])

    @patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""})
    def test_action_card_is_not_injected_without_a_candidate_action(self, _config):
        agent = ABCDAgent(
            action_rules_text="#### `send-link`\nSend the requested link.\n",
            slot_policies_text="#### `send-link`\nUse the current request target.\n",
        )

        prompt = agent._build_system_prompt({}, "Customer asks for help.")

        self.assertNotIn('<retrieved_action_card tool="retrieve_action_card">', prompt)

    @patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""})
    def test_two_stage_grounding_locks_action_and_retrieves_its_card(self, _config):
        agent = ABCDAgent(
            action_rules_text="#### `send-link`\nSend the requested account link.\n",
            slot_policies_text="#### `send-link`\nNo slot values are allowed.\n",
        )
        # Keep the fixture independent of whichever local ABCD data subset is
        # available when this unit test is run.
        agent.action_schema = {
            "actions": {"send-link", "wrong-action"},
            "slot_counts": {"send-link": {0}, "wrong-action": {1}},
        }
        agent._plan_reference_lookup = lambda *_args, **_kwargs: {
            "thought": "Find account-link evidence.",
            "query_text": "account link",
            "top_k": 1,
            "messages": [],
            "raw_output": "",
            "tool_call": {},
            "fallback_used": False,
            "candidate_actions": ["wrong-action"],
        }
        agent._lookup_reference = lambda *_args, **_kwargs: {
            "query": {"query": "account link", "top_k": 1},
            "executed": True,
            "status": "matched",
            "observation": "Reference: send the requested account link.",
            "selected_sections": [],
        }
        conversation = {
            "convo_id": "two-stage-fixture",
            "scenario": {},
            "delexed": [
                {"speaker": "customer", "text": "Please send me the account link."},
                {
                    "speaker": "action",
                    "text": "Send the account link.",
                    "targets": ["", "take_action", "send-link", []],
                },
            ],
        }

        with patch(
            "llm.chat",
            side_effect=[
                '{"action":"send-link"}',
                '{"action":"wrong-action","slots":[],"response":"I sent the link."}',
            ],
        ):
            results = agent.predict_all_turns(conversation, predict_actions=True)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["predicted_action"], "send-link")
        self.assertEqual(result["predicted_slots"], [])
        self.assertEqual(result["action_card_lookup"]["selected_actions"], ["send-link"])
        self.assertTrue(result["action_selection"]["two_stage_applied"])
        self.assertTrue(result["action_selection"]["action_locked"])
        self.assertEqual(result["action_selection"]["stage2_reported_action"], "wrong-action")
        self.assertIn("Return `slots: []` exactly", result["react_trace"][-1]["action_input"]["messages"][1]["content"])
        self.assertEqual(
            [step["action"] for step in result["react_trace"][-2:]],
            ["llm_select_action", "llm_ground_slots_and_response"],
        )


if __name__ == "__main__":
    unittest.main()
