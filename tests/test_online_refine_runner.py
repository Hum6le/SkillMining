import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.run_backbone_online_refine import _checkpoint, _replay_candidate, _restore_rollout_schedule
from skill_mining.online_refinement import initialize_skill_dag


class ReplayGateTest(unittest.TestCase):
    def test_resume_restores_saved_action_turn_schedule_and_unit(self):
        train = [{"convo_id": "c1"}, {"convo_id": "c2"}]
        saved = {"refinement_mode": "standard", "selection_unit": "action_turn",
                 "batches": [{"samples": [{"conversation_id": "c2", "turn_index": 7, "sample_id": "c2:7"}]},
                             {"samples": [{"conversation_id": "c1", "turn_index": 1, "sample_id": "c1:1"}]}]}
        batches, unit = _restore_rollout_schedule(saved, train, "standard")
        self.assertEqual(unit, "action_turn")
        self.assertEqual([batch[0]["sample_id"] for batch in batches], ["c2:7", "c1:1"])
        self.assertIs(batches[0][0]["conversation"], train[1])
        self.assertNotIn("conversation", saved["batches"][0]["samples"][0])

    def test_resume_legacy_schedule_without_unit_has_mode_default(self):
        for mode, expected in (("standard", "action_turn"),
                               ("constrained-repair", "action_turn_uncertainty_local"),
                               ("trace2skill-hybrid", "conversation_sequence_similarity")):
            with self.subTest(mode=mode):
                batches, unit = _restore_rollout_schedule({"batches": [{"conversation_ids": ["c2", "c1"]}]},
                                                          [{"convo_id": "c1"}, {"convo_id": "c2"}], mode)
                self.assertEqual(unit, expected)
                self.assertEqual([item["convo_id"] for item in batches[0]], ["c2", "c1"])

    def test_resume_checks_missing_sessions_in_both_schedule_formats(self):
        for item in ({"samples": [{"conversation_id": "missing", "turn_index": 1}]},
                     {"conversation_ids": ["missing"]}):
            with self.subTest(item=item), self.assertRaisesRegex(RuntimeError, "absent from this train split"):
                _restore_rollout_schedule({"batches": [item]}, [], "standard")

    def test_resume_rejects_changed_refinement_mode(self):
        with self.assertRaisesRegex(ValueError, "refinement mode differs"):
            _restore_rollout_schedule({"refinement_mode": "trace2skill-hybrid", "batches": []}, [], "standard")

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
