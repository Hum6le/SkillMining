import unittest
from unittest.mock import patch

from eval_tod.abcd.agent import ABCDAgent, summarize_action_selection_runtime


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
    def test_action_selection_candidates_merge_planner_and_reference_targets(self, _config):
        agent = ABCDAgent(
            action_rules_text=(
                "#### `send-link`\nSend a link.\n\n"
                "#### `make-password`\nGenerate a password.\n\n"
                "#### `verify-identity`\nVerify the customer.\n"
            ),
        )
        agent.action_schema = {
            "actions": {"send-link", "make-password", "verify-identity", "pull-up-account"},
            "slot_counts": {},
        }

        candidates = agent._action_selection_candidates(
            {"candidate_actions": ["verify-identity", "pull-up-account"]},
            {"selected_sections": [
                {"transition_title": "enter-details -> make-password"},
                {"transition_title": "enter-details -> send-link"},
            ]},
        )

        self.assertEqual(
            candidates, ["verify-identity", "make-password", "send-link"],
        )

    @patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""})
    def test_stage1_compares_cards_without_restricting_selected_action(self, _config):
        agent = ABCDAgent(
            action_rules_text=(
                "#### `send-link`\nSend a link only after verification.\n\n"
                "#### `make-password`\nGenerate a password after identity checks.\n"
            ),
        )
        agent.action_schema = {
            "actions": {"send-link", "make-password", "verify-identity"},
            "slot_counts": {},
        }

        with patch("llm.chat", return_value='{"action":"verify-identity"}'):
            selected, _raw, meta = agent._select_action_for_grounding(
                {}, "[Customer] I need help accessing my account.", "No reference.",
                candidate_actions=["send-link", "make-password"],
            )

        self.assertEqual(selected, "verify-identity")
        self.assertEqual(
            meta["candidate_card_lookup"]["selected_actions"],
            ["send-link", "make-password"],
        )
        self.assertIn("not a\nclosed action set", meta["messages"][1]["content"])

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
        self.assertTrue(result["action_selection"]["attempted"])
        self.assertTrue(result["action_selection"]["action_locked"])
        self.assertEqual(result["action_selection"]["stage2_reported_action"], "wrong-action")
        self.assertEqual(result["action_selection"]["candidate_actions"], [])
        self.assertIn("Return `slots: []` exactly", result["react_trace"][-1]["action_input"]["messages"][1]["content"])
        self.assertEqual(
            [step["action"] for step in result["react_trace"][-4:]],
            [
                "retrieve_candidate_action_cards", "llm_select_action",
                "retrieve_action_card", "llm_ground_slots_and_response",
            ],
        )

    def test_runtime_summary_separates_candidate_hits_misses_and_empty_lists(self):
        conversation = {
            "convo_id": "metrics-fixture",
            "delexed": [
                {"targets": ["", "take_action", "a", []]},
                {"targets": ["", "take_action", "b", []]},
                {"targets": ["", "take_action", "c", []]},
            ],
        }
        rows = [
            {
                "convo_id": "metrics-fixture", "turn_index": 0,
                "action_selection": {
                    "attempted": True, "two_stage_applied": True,
                    "candidate_actions": ["a", "x"], "selected_action": "a",
                },
            },
            {
                "convo_id": "metrics-fixture", "turn_index": 1,
                "action_selection": {
                    "attempted": True, "two_stage_applied": True,
                    "candidate_actions": ["x"], "selected_action": "b",
                },
            },
            {
                "convo_id": "metrics-fixture", "turn_index": 2,
                "action_selection": {
                    "attempted": True, "two_stage_applied": False,
                    "candidate_actions": [], "selected_action": "x",
                },
            },
        ]

        summary = summarize_action_selection_runtime([conversation], rows)

        self.assertEqual(summary["num_stage1_attempts"], 3)
        self.assertEqual(summary["num_two_stage_applied"], 2)
        self.assertEqual(summary["num_fallback_no_selected_card"], 1)
        self.assertEqual(summary["candidate_recall"], 0.5)
        self.assertEqual(summary["selection_accuracy"], 0.6667)
        self.assertEqual(summary["selection_accuracy_candidate_hit"], 1.0)
        self.assertEqual(summary["selection_accuracy_candidate_miss"], 1.0)
        self.assertEqual(summary["selection_accuracy_empty_candidates"], 0.0)
        self.assertEqual(summary["correct_outside_shortlist"], 1)


if __name__ == "__main__":
    unittest.main()
