from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from asi_offline import (
    build_online_episode,
    build_online_induction_prompt,
    parse_online_candidate_output,
    validate_online_candidates,
    ASIOnlineLibraryManager,
    decide_asi_update,
)


def _conversation() -> dict:
    return {
        "convo_id": "online-1",
        "original": [
            ["customer", "I need help."],
            ["action", "Account found."],
            ["action", "Identity verified."],
            ["action", "A reset link was sent."],
        ],
        "delexed": [
            {"speaker": "customer", "text": "I need help.", "targets": ["x", "none", None, [], -1]},
            {"speaker": "action", "text": "Account found.", "targets": ["x", "take_action", "pull-up-account", ["alice"], -1]},
            {"speaker": "action", "text": "Identity verified.", "targets": ["x", "take_action", "verify-identity", [], -1]},
            {"speaker": "action", "text": "A reset link was sent.", "targets": ["x", "take_action", "send-link", [], -1]},
        ],
    }


class ASIOnlineEpisodeTest(unittest.TestCase):
    def test_successful_rollout_is_eligible(self) -> None:
        episode = build_online_episode(
            _conversation(),
            [
                {"convo_id": "online-1", "turn_index": 1, "target_type": "action", "predicted_action": "pull-up-account", "predicted_slots": ["alice"]},
                {"convo_id": "online-1", "turn_index": 2, "target_type": "action", "predicted_action": "verify-identity", "predicted_slots": []},
                {"convo_id": "online-1", "turn_index": 3, "target_type": "action", "predicted_action": "send-link", "predicted_slots": []},
            ],
            {"ast_score": 1.0, "action_total": 3, "action_correct": 3},
        )
        self.assertTrue(episode.eligible_for_induction)
        self.assertEqual(episode.eligibility_reason, "eligible_successful_rollout")
        self.assertEqual([x["action"] for x in episode.primitive_actions], [
            "pull-up-account", "verify-identity", "send-link",
        ])

    def test_wrong_slot_is_not_an_online_success(self) -> None:
        episode = build_online_episode(
            _conversation(),
            [
                {"convo_id": "online-1", "turn_index": 1, "target_type": "action", "predicted_action": "pull-up-account", "predicted_slots": ["bob"]},
                {"convo_id": "online-1", "turn_index": 2, "target_type": "action", "predicted_action": "verify-identity", "predicted_slots": []},
                {"convo_id": "online-1", "turn_index": 3, "target_type": "action", "predicted_action": "send-link", "predicted_slots": []},
            ],
            {"ast_score": 0.66, "action_total": 3, "action_correct": 2},
        )
        self.assertFalse(episode.eligible_for_induction)
        self.assertEqual(episode.eligibility_reason, "rollout_action_or_slot_mismatch")

    def test_online_candidate_uses_positional_slot_parameters(self) -> None:
        episode = build_online_episode(
            _conversation(),
            [
                {"convo_id": "online-1", "turn_index": 1, "target_type": "action", "predicted_action": "pull-up-account", "predicted_slots": ["alice"]},
                {"convo_id": "online-1", "turn_index": 2, "target_type": "action", "predicted_action": "verify-identity", "predicted_slots": []},
                {"convo_id": "online-1", "turn_index": 3, "target_type": "action", "predicted_action": "send-link", "predicted_slots": []},
            ],
            {"ast_score": 1.0, "action_total": 3, "action_correct": 3},
        )
        prompt = build_online_induction_prompt(episode)
        self.assertIn("slot_1", prompt)
        candidates, rejected = parse_online_candidate_output(
            '{"skills": [{"name": "verify_account", "description": "Verify an account and send a link.", "start_action_index": 0, "end_action_index": 2, "parameters": ["slot_1"]}]}',
            episode,
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(candidates), 1)
        self.assertIn("take_action('pull-up-account', [slot_1])", candidates[0].function_source)
        validation = validate_online_candidates(episode, candidates)
        self.assertTrue(validation.replay_valid)
        self.assertEqual(validation.accepted_candidates, ["verify_account"])

    def test_short_candidate_span_is_rejected(self) -> None:
        episode = build_online_episode(
            _conversation(),
            [
                {"convo_id": "online-1", "turn_index": 1, "target_type": "action", "predicted_action": "pull-up-account", "predicted_slots": ["alice"]},
                {"convo_id": "online-1", "turn_index": 2, "target_type": "action", "predicted_action": "verify-identity", "predicted_slots": []},
                {"convo_id": "online-1", "turn_index": 3, "target_type": "action", "predicted_action": "send-link", "predicted_slots": []},
            ],
            {"ast_score": 1.0, "action_total": 3, "action_correct": 3},
        )
        candidates, parse_rejected = parse_online_candidate_output(
            '{"skills": [{"name": "first", "description": "First.", "start_action_index": 0, "end_action_index": 2, "parameters": ["slot_1"]}, {"name": "second", "description": "Second.", "start_action_index": 1, "end_action_index": 2, "parameters": []}]}',
            episode,
        )
        validation = validate_online_candidates(episode, candidates)
        self.assertTrue(validation.replay_valid)
        self.assertEqual(validation.accepted_candidates, ["first"])
        self.assertEqual(parse_rejected[-1]["reason"], "invalid_or_duplicate_span")

    def test_online_library_stages_accepts_and_rolls_back_versions(self) -> None:
        episode = build_online_episode(
            _conversation(),
            [
                {"convo_id": "online-1", "turn_index": 1, "target_type": "action", "predicted_action": "pull-up-account", "predicted_slots": ["alice"]},
                {"convo_id": "online-1", "turn_index": 2, "target_type": "action", "predicted_action": "verify-identity", "predicted_slots": []},
                {"convo_id": "online-1", "turn_index": 3, "target_type": "action", "predicted_action": "send-link", "predicted_slots": []},
            ],
            {"ast_score": 1.0, "action_total": 3, "action_correct": 3},
        )
        candidates, _ = parse_online_candidate_output(
            '{"skills": [{"name": "verify_account", "description": "Verify an account and send a link.", "start_action_index": 0, "end_action_index": 2, "parameters": ["slot_1"]}]}',
            episode,
        )
        validation = validate_online_candidates(episode, candidates)
        with TemporaryDirectory() as directory:
            manager = ASIOnlineLibraryManager(Path(directory))
            update = manager.stage(candidates, validation, ast_before=0.5)
            self.assertNotIn("verify_account", manager.current_library_path().read_text(encoding="utf-8"))
            manager.accept(update, ast_after=0.6)
            self.assertIn("verify_account", manager.current_library_path().read_text(encoding="utf-8"))
            second = manager.stage(candidates, validation, ast_before=0.6)
            manager.rollback(second, ast_after=0.55)
            self.assertIn("verify_account", manager.current_library_path().read_text(encoding="utf-8"))

    def test_update_decision_requires_ast_gain_without_action_or_slot_regression(self) -> None:
        baseline = {"ast_cds": {"ast_joint": 0.50, "ast_action_name": 0.70, "ast_slot_value": 0.60}}
        improved = {"ast_cds": {"ast_joint": 0.60, "ast_action_name": 0.70, "ast_slot_value": 0.60}}
        decision = decide_asi_update(baseline, improved)
        self.assertTrue(decision.accepted)
        regressed = {"ast_cds": {"ast_joint": 0.70, "ast_action_name": 0.65, "ast_slot_value": 0.60}}
        self.assertFalse(decide_asi_update(baseline, regressed).accepted)


if __name__ == "__main__":
    unittest.main()
