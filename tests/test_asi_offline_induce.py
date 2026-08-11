from __future__ import annotations

import unittest

from asi_offline import (
    build_induction_episode,
    build_episode_induction_messages,
    induce_episode,
)


def _conversation() -> dict:
    return {
        "convo_id": "induce-1",
        "scenario": {"subflow": "recover_password"},
        "original": [
            ["customer", "I forgot my password. My username is alice."],
            ["action", "Account found."],
            ["action", "A password was generated."],
            ["action", "Password reset confirmed."],
        ],
        "delexed": [
            {"speaker": "customer", "text": "I forgot my password.", "targets": ["hidden", None, None, [], -1]},
            {"speaker": "action", "text": "Account found.", "targets": ["hidden", "take_action", "pull-up-account", ["alice"], -1]},
            {"speaker": "action", "text": "A password was generated.", "targets": ["hidden", "take_action", "make-password", [], -1]},
            {"speaker": "action", "text": "Password reset confirmed.", "targets": ["hidden", "take_action", "confirm-reset", [], -1]},
        ],
    }


class ASIOfflineInduceTest(unittest.TestCase):
    def test_renders_original_style_single_example_prompt(self) -> None:
        episode = build_induction_episode(_conversation())
        messages = build_episode_induction_messages(episode)

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("### Example 1 (induce-1)", messages[1]["content"])
        self.assertIn("take_action('pull-up-account', ['alice'])", messages[1]["content"])
        self.assertIn("## Reusable Functions", messages[1]["content"])
        self.assertNotIn("recover_password", messages[1]["content"])
        self.assertIn("alice", messages[1]["content"])

    def test_records_raw_model_response_without_validating_or_replaying(self) -> None:
        episode = build_induction_episode(_conversation())
        calls = []

        def mock_chat(messages, **kwargs):
            calls.append((messages, kwargs))
            return "```python\ndef recover_password(username):\n    pass\n```"

        artifact = induce_episode(episode, mock_chat, temperature=0.7)
        self.assertEqual(len(calls), 1)
        self.assertEqual(artifact.episode_id, "induce-1")
        self.assertEqual(artifact.action_count, 3)
        self.assertIn("def recover_password", artifact.raw_response)


if __name__ == "__main__":
    unittest.main()
