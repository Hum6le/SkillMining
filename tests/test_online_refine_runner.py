import unittest
from unittest.mock import patch

from scripts.run_backbone_online_refine import _replay_candidate


class ReplayGateTest(unittest.TestCase):
    def test_replay_uses_action_turn_gold_and_counts_missing_rows(self):
        samples = [
            {
                "sample_id": "c1:2", "conversation_id": "c1", "turn_index": 2,
                "target_action": "send-link", "gold_slots": ["email"],
                "conversation": {"convo_id": "c1"},
            },
            {
                "sample_id": "c2:1", "conversation_id": "c2", "turn_index": 1,
                "target_action": "make-password", "gold_slots": [],
                "conversation": {"convo_id": "c2"},
            },
        ]
        rows = [{
            "convo_id": "c1", "turn_index": 2,
            "predicted_action": "send-link", "predicted_slots": ["email"],
        }]
        with patch("scripts.run_backbone_online_refine._build_agent", return_value=object()), \
             patch("scripts.run_backbone_online_refine._rollout_online_batch", return_value=rows):
            result = _replay_candidate(
                object(), "skill", {}, samples, {"c1:2", "c2:1"},
                "reference", "rules", "policies", None, None,
            )
        self.assertEqual(result["num_samples"], 2)
        self.assertEqual(result["action_correct"], 1)
        self.assertEqual(result["joint_correct"], 1)


if __name__ == "__main__":
    unittest.main()
