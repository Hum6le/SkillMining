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


if __name__ == "__main__":
    unittest.main()
