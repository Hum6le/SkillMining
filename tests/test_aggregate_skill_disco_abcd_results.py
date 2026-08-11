from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.aggregate_skill_disco_abcd_results import aggregate_records, collect_records


class AggregateSkillDiscoABCDResultsTest(unittest.TestCase):
    def _write_result(
        self, root: Path, subflow: str, *, sessions: int, action_turns: int,
        action_acc: float, slot_acc: float,
    ) -> Path:
        run_dir = root / f"skill_disco_abcd_subflow_{subflow}_2026-08-11_12-00-00"
        evaluation_dir = run_dir / "evaluation"
        evaluation_dir.mkdir(parents=True)
        (run_dir / "manifest.txt").write_text(f"subflow={subflow}\n", encoding="utf-8")
        result = {
            "num_conversations": sessions,
            "text": {"num_samples": sessions, "bert_f1": action_acc},
            "ast_cds": {
                "num_action_turns": action_turns,
                "ast_joint": action_acc,
                "ast_action_name": action_acc,
                "ast_slot_value": slot_acc,
                "cds_overall": slot_acc,
            },
        }
        path = evaluation_dir / "result.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        return path

    def test_uses_correct_metric_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._write_result(
                root, "first", sessions=2, action_turns=10, action_acc=0.5, slot_acc=0.2
            )
            second = self._write_result(
                root, "second", sessions=3, action_turns=30, action_acc=0.8, slot_acc=0.7
            )
            aggregate = aggregate_records(collect_records([first, second]))

        self.assertEqual(aggregate["weights"], {"test_sessions": 5, "text_samples": 5, "action_turns": 40})
        self.assertEqual(aggregate["metrics"]["ast_action_name"], 0.725)
        self.assertEqual(aggregate["metrics"]["ast_slot_value"], 0.575)
        self.assertEqual(aggregate["metrics"]["cds_overall"], 0.5)


if __name__ == "__main__":
    unittest.main()
