import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_a2_abcd as a2
from scripts.abcd_runtime_utils import RUNTIME_DEFAULTS, abcd_runtime_options
from scripts.run_backbone_online_refine import _build_agent


def _args(root, **changes):
    values = dict(model="deepseek-chat", arms=a2.ARMS, skip_arms=("full",), resume=False,
                  output_dir=str(root), max_train=None, max_test=None, max_batches=None,
                  batch_size=8, refinement_mode="standard", skip_utterance_eval=True,
                  **RUNTIME_DEFAULTS)
    values.update(changes)
    return argparse.Namespace(**values)


class A2RuntimeContractTest(unittest.TestCase):
    def test_offline_and_full_use_identical_agent_class_and_runtime_options(self):
        for overrides in ({}, {"reference_top_k": 2, "reference_max_chars": 1200,
                               "disable_competitive_action_cards": True, "action_selection_candidate_limit": 4}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                offline = root / "offline_artifact"
                offline.mkdir()
                a2._json(offline / "complete.json", {"subflow": "f", "max_train": None})
                a2._json(offline / "mining_usage.json", {"total": {"calls": 3}})
                for name, text in (("skill.md", "OFFLINE_SKILL"), ("reference.md", "OFFLINE_REFERENCE"),
                                   ("action_rules.md", "OFFLINE_ACTION_RULES"), ("slot_policies.md", "OFFLINE_SLOT_POLICIES")):
                    (offline / name).write_text(text, encoding="utf-8")
                args = _args(root, arms=("offline_only",), skip_arms=(), **overrides)
                usage = {"total": {"calls": 1}}
                with patch("scripts.run_subflow_eval.load_subflow_data", return_value=([{"convo_id": "train"}], [{"convo_id": "test"}])), \
                        patch("scripts.run_subflow_eval.mine_subflow_skill_backbone") as mine, \
                        patch("scripts.run_subflow_eval.evaluate_agent_on_subflow", return_value={"ast_cds": {}}), \
                        patch("scripts.llm_usage_utils.get_usage", return_value=usage), \
                        patch("eval_tod.abcd.agent.ABCDAgent") as constructor:
                    a2._run_direct(args, "f", root, ["ID1"])
                    offline_kwargs = copy.copy(constructor.call_args.kwargs)
                    _build_agent(args, "REFINED_SKILL", "OFFLINE_REFERENCE", "OFFLINE_ACTION_RULES",
                                 "OFFLINE_SLOT_POLICIES", {}, workflow_id="ID1")
                    full_kwargs = constructor.call_args.kwargs
                    self.assertEqual(constructor.call_count, 2)
                    for key, value in abcd_runtime_options(args).items():
                        self.assertEqual(offline_kwargs[key], value)
                        self.assertEqual(full_kwargs[key], value)
                    self.assertEqual(offline_kwargs["workflow_id"], full_kwargs["workflow_id"])
                    self.assertEqual(offline_kwargs["slot_policies_text"], "OFFLINE_SLOT_POLICIES")
                    self.assertTrue(full_kwargs["slot_policies_text"].startswith("OFFLINE_SLOT_POLICIES"))
                    self.assertEqual(offline_kwargs["reference_text"], "OFFLINE_REFERENCE")
                    mine.assert_not_called()
                result = a2._read(root / "offline_only" / "result.json")
                self.assertEqual(result["runtime_protocol"], abcd_runtime_options(args))

    def test_online_subprocess_receives_same_runtime_knobs(self):
        args = _args(Path("unused"), reference_top_k=2, reference_max_chars=1200,
                     disable_competitive_action_cards=True, action_selection_candidate_limit=4)
        command = a2._online_command(args, "f", "full", Path("unused/full"), Path("unused/offline"), [])
        for flag, value in (("--reference-top-k", "2"), ("--reference-max-chars", "1200"),
                            ("--action-selection-candidate-limit", "4")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--disable-competitive-action-cards", command)

    def test_legacy_args_keep_original_default_protocol(self):
        options = abcd_runtime_options(argparse.Namespace(model="deepseek-chat"))
        self.assertTrue(options["competitive_action_cards"])
        self.assertEqual(options["reference_top_k"], 3)
        self.assertEqual(options["reference_max_chars"], 1800)
        self.assertEqual(options["action_selection_candidate_limit"], 3)
        self.assertIsNone(options["workflow_max_chars"])
        self.assertFalse(options["expose_scenario_labels"])
