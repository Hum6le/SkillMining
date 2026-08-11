from __future__ import annotations

import unittest

from asi_offline import (
    build_induction_corpus,
    build_induction_episode,
    build_induction_prompt,
    parse_candidate_output,
    rewrite_episode_actions,
)


def _conversation(conversation_id: str, username: str) -> dict:
    return {
        "convo_id": conversation_id,
        "scenario": {"personal": {"username": username}},
        "original": [
            ["customer", f"I forgot my password. My username is {username}."],
            ["action", "Account found."],
            ["action", "A password was generated."],
            ["action", "The password reset was confirmed."],
        ],
        "delexed": [
            {
                "speaker": "customer",
                "text": "I forgot my password.",
                "targets": ["hidden", None, None, [], -1],
            },
            {
                "speaker": "action",
                "text": "Account found.",
                "targets": ["hidden", "take_action", "pull-up-account", [username], -1],
            },
            {
                "speaker": "action",
                "text": "A password was generated.",
                "targets": ["hidden", "take_action", "make-password", [], -1],
            },
            {
                "speaker": "action",
                "text": "The password reset was confirmed.",
                "targets": ["hidden", "take_action", "confirm-reset", [], -1],
            },
        ],
    }


class ASIOfflineInductionTest(unittest.TestCase):
    def test_builds_parameterized_single_trace_episode(self) -> None:
        episode = build_induction_episode(_conversation("trace-1", "alice"))

        self.assertEqual(episode.source_split, "train")
        self.assertTrue(episode.eligible_for_induction)
        self.assertEqual(
            [item["parameterized_action"] for item in episode.primitive_actions],
            ["pull-up-account(username)", "make-password()", "confirm-reset()"],
        )
        self.assertIn("take_action('pull-up-account', ['username'])", episode.parameterized_program)
        self.assertNotIn("alice", episode.parameterized_program)

    def test_rejects_non_train_induction_and_short_traces(self) -> None:
        with self.assertRaisesRegex(ValueError, "train split"):
            build_induction_episode(_conversation("trace-1", "alice"), source_split="test")

        short = _conversation("trace-2", "bob")
        short["delexed"] = short["delexed"][:2]
        self.assertEqual(build_induction_corpus([short]), [])

    def test_reconstructs_candidate_program_and_rewritten_trace(self) -> None:
        episode = build_induction_episode(_conversation("trace-3", "carol"))
        raw_output = """{
          "skills": [{
            "name": "recover_account_password",
            "description": "Recover an account password.",
            "start_action_index": 0,
            "end_action_index": 2,
            "parameters": ["username"]
          }]
        }"""
        candidates, rejected = parse_candidate_output(raw_output, episode)
        self.assertEqual(rejected, [])
        self.assertEqual(len(candidates), 1)
        self.assertIn("take_action('pull-up-account', [username])", candidates[0].function_source)
        self.assertIn("take_action('make-password', [])", candidates[0].function_source)
        self.assertIn("take_action('confirm-reset', [])", candidates[0].function_source)
        rewritten = rewrite_episode_actions(episode, candidates)
        self.assertEqual(rewritten[0]["kind"], "skill_call")
        self.assertEqual(rewritten[0]["replaces_action_indices"], [0, 1, 2])
        self.assertIn("Backend action table", build_induction_prompt(episode))


if __name__ == "__main__":
    unittest.main()
