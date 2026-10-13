import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.run_backbone_online_refine import _checkpoint, _replay_candidate
from skill_mining.online_refinement import initialize_skill_dag


class ReplayGateTest(unittest.TestCase):
    def test_checkpoint_persists_promoted_branch_when_action_section_is_missing(self):
        state = initialize_skill_dag({
            "nodes": [{"id": "a", "label": "enter-details"}, {"id": "b", "label": "make-password"}],
            "edges": [{"source": "a", "target": "b", "support": 2}],
            "backbone": {"compilation_order": ["a"], "edges": []},
        }, "account_access")
        state["edges"]["a=>b"].update(kind="promoted_branch", visibility="skill", guard="password reset requested")
        skill = "# Skill\n## Workflow\nKeep workflow.\n## Slot Policies\nKeep slots.\n"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            merged = _checkpoint(output, state, skill, "Reference evidence.")
            self.assertIn("#### `make-password`", merged)
            self.assertIn("## Slot Policies\nKeep slots.", merged)
            self.assertEqual((output / "working_skill.md").read_text(encoding="utf-8"), merged)
            self.assertEqual((output / "skill.md").read_text(encoding="utf-8"), merged)
            self.assertTrue((output / "skill_dag_state.json").exists())

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
