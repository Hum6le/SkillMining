import json
import tempfile
import unittest
from pathlib import Path

from scripts.restore_online_refine_and_evaluate import _failed_edits


class RestoreOnlineRefineTest(unittest.TestCase):
    def test_collects_failed_dynamic_edits_from_autonomous_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            reflection_dir = run_dir / "autonomous_reflection"
            reflection_dir.mkdir()
            payload = {
                "proposed_skill_operations": [{
                    "operation_id": "fix-route", "op": "replace",
                    "match_text": "old rule", "new_text": "correct rule",
                }],
                "skill_operations": [{
                    "operation_id": "fix-route", "op": "replace",
                    "match_text": "old rule", "new_text": "correct rule",
                    "error": "match_text was not found in current skill",
                }],
            }
            (reflection_dir / "batch_0001.json").write_text(
                json.dumps(payload), encoding="utf-8",
            )
            dynamic, semantic, successful, unreplayable = _failed_edits(run_dir)
            self.assertEqual(len(dynamic), 1)
            self.assertEqual(dynamic[0]["operation_id"], "fix-route")
            self.assertEqual(semantic, [])
            self.assertEqual(successful, [])
            self.assertEqual(unreplayable, [])

    def test_reads_group_reflection_applied_operation_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            group_dir = run_dir / "iterative_refinement" / "wave_0001" / "group_reflections"
            group_dir.mkdir(parents=True)
            payload = {
                "skill_operations": [{
                    "operation_id": "edit-1", "op": "replace",
                    "match_text": "before", "new_text": "after",
                }],
                "applied_skill_operations": [{
                    "operation_id": "edit-1", "op": "replace",
                    "match_text": "before", "new_text": "after",
                    "applied": True,
                }],
            }
            (group_dir / "reflection_0001.json").write_text(
                json.dumps(payload), encoding="utf-8",
            )
            dynamic, semantic, successful, unreplayable = _failed_edits(run_dir)
            self.assertEqual(dynamic, [])
            self.assertEqual(semantic, [])
            self.assertEqual(len(successful), 1)
            self.assertEqual(unreplayable, [])


if __name__ == "__main__":
    unittest.main()
