import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_a2_abcd as a2


def _args(output, **overrides):
    values = dict(arms=a2.ARMS, skip_arms=("full",), output_dir=str(output),
                  resume=True, model="deepseek-chat", max_train=None,
                  max_test=None, max_batches=None, batch_size=8,
                  refinement_mode="standard", skip_utterance_eval=False)
    values.update(overrides)
    return argparse.Namespace(**values)


class A2SkipArmsTest(unittest.TestCase):
    def test_execution_filter_does_not_change_configured_arms(self):
        args = _args("unused")
        self.assertEqual(a2._execution_arms(args), a2.ARMS[:-1])
        self.assertEqual(args.arms, a2.ARMS)
        args.skip_arms = ()
        self.assertEqual(a2._execution_arms(args), a2.ARMS)

    def test_flow_skips_full_and_leaves_full_artifacts_untouched(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory)
            full = root / "account_access" / "full"
            full.mkdir(parents=True)
            original = b'{"existing_full_artifact": true}\n'
            (full / "a2_result.json").write_bytes(original)
            with patch.object(a2, "_run_direct") as direct, patch.object(a2, "_run_online") as online:
                a2._run_flow(_args(root), "account_access", ["ID1"])
            direct.assert_called_once()
            self.assertEqual([call.args[2] for call in online.call_args_list], ["online_only"])
            self.assertEqual((full / "a2_result.json").read_bytes(), original)

    def test_completed_online_only_is_not_rerun_on_resume(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory)
            output = root / "account_access" / "online_only"
            output.mkdir(parents=True)
            (output / "online_refine_result.json").write_text("{}", encoding="utf-8")
            with patch.object(a2, "_run_direct"), patch.object(a2, "_run_online") as online:
                a2._run_flow(_args(root), "account_access", [])
            online.assert_not_called()

    def test_direct_does_not_mine_for_skipped_full(self):
        # Only the skipped full arm needs offline data in this configuration.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root, arms=("online_only", "full"))
            a2._run_direct(args, "account_access", root, [])
            self.assertFalse((root / "offline_artifact").exists())

    def test_resume_forwards_skip_to_workers_preserves_config_and_aggregates_existing_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "a2_config.json"
            args = _args(root)
            prior = {"arguments": {key: value for key, value in vars(args).items()
                                   if key != "skip_arms"}}
            config.write_text(json.dumps(prior), encoding="utf-8")
            before = config.read_bytes()
            full = root / "account_access" / "full"
            full.mkdir(parents=True)
            a2._json(full / "a2_result.json", {"ast_cds": {
                "ast_joint": 0.5, "ast_action_name": 0.6,
                "ast_slot_value": 0.7, "num_action_turns": 10}})
            argv = ["run_a2_abcd.py", "--subflow", "account_access", "--output-dir", str(root),
                    "--resume", "--skip-arms", "full", "--workflow-ids", "ID1"]
            with patch.object(sys, "argv", argv), patch.object(a2, "_subflows", return_value=["account_access"]), \
                    patch.object(a2.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                run.return_value.returncode = 0
                self.assertEqual(a2.main(), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--skip-arms") + 1], "full")
            self.assertEqual(command[command.index("--arms") + 1], ",".join(a2.ARMS))
            self.assertIn("--resume", command)
            self.assertEqual(config.read_bytes(), before)
            aggregate = a2._read(root / "a2_aggregate.json")
            self.assertEqual(aggregate["skipped_arms"], ["full"])
            self.assertEqual(aggregate["execution_arms"], list(a2.ARMS[:-1]))
            self.assertEqual(aggregate["aggregate"]["full"]["joint_ast"], 0.5)

    def test_skip_does_not_bypass_other_resume_configuration_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a2._json(root / "a2_config.json", {"arguments": {
                **vars(_args(root)), "arms": list(a2.ARMS), "model": "original-model"}})
            argv = ["run_a2_abcd.py", "--subflow", "account_access", "--output-dir", str(root),
                    "--resume", "--skip-arms", "full"]
            with patch.object(sys, "argv", argv), patch.object(a2, "_subflows", return_value=["account_access"]), \
                    patch.object(a2.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                a2.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_invalid_skip_arguments_are_rejected(self):
        for skip in ("unknown", "full,full", ",".join(a2.ARMS)):
            with self.subTest(skip=skip), patch.object(sys, "argv", [
                "run_a2_abcd.py", "--subflow", "account_access", "--skip-arms", skip]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                a2.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
