import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_a2_abcd as a2
from scripts import abcd_offline_awm as experiment
from scripts.abcd_runtime_utils import RUNTIME_DEFAULTS, abcd_runtime_options
from skill_mining.online_refinement import build_action_turn_samples


def _conversation(cid):
    return {"convo_id": cid, "scenario": {"flow": "SECRET_FLOW", "subflow": "SECRET_LABEL"}, "delexed": [
        {"speaker": "customer", "text": "Help me", "targets": ["", "", "", [], -1]},
        {"speaker": "action", "text": "Observed action", "targets": ["", "take_action", "enter-details", [], -1]},
        {"speaker": "customer", "text": "Please send a link", "targets": ["", "", "", [], -1]},
        {"speaker": "action", "text": "Observed action", "targets": ["", "take_action", "send-link", ["FUTURE_VALUE"], -1]},
    ]}


def _args(root, **changes):
    values = dict(model="deepseek-chat", resume=False, max_train=None, max_test=None, max_batches=None,
                  batch_size=1, refinement_mode="standard", skip_utterance_eval=True, **RUNTIME_DEFAULTS)
    values.update(changes)
    return argparse.Namespace(**values)


class OfflineAWMTest(unittest.TestCase):
    def test_plain_markdown_workflow_does_not_require_json_escaping(self):
        from eval_tod.abcd.agent import _parse_awm_card_update, _AWM_CARD_START, _AWM_CARD_END
        workflow = '### Rule\nUse "quoted" cues and C:\\local\\path.\n' + "long learned rule\n" * 400
        text = workflow + _AWM_CARD_START + '\n[{"action":"send-link","slot_policy":"Use the verified current email."}]\n' + _AWM_CARD_END
        parsed, updates = _parse_awm_card_update(text, {"send-link"})
        self.assertEqual(parsed, workflow.strip())
        self.assertEqual(updates[0]["action"], "send-link")
        self.assertGreater(len(parsed), 4000)
        for invalid in (workflow + _AWM_CARD_START + "\n[]", text + " trailing",
                        text + "\n" + _AWM_CARD_END, workflow + _AWM_CARD_END + _AWM_CARD_START,
                        workflow + _AWM_CARD_START + '\n[{"slot_policy":"cut' + _AWM_CARD_END):
            with self.subTest(text=invalid[-100:]), self.assertRaises(ValueError):
                _parse_awm_card_update(invalid, {"send-link"})

    def test_invalid_json_retry_gets_error_feedback_and_preserves_raw_response(self):
        from awm import WorkflowStore
        from eval_tod.abcd.agent import ABCDAgent, _AWM_CARD_START, _AWM_CARD_END
        from eval_tod.response_logger import ResponseLogger
        workflow = WorkflowStore()
        workflow.replace("OLD")
        bad = '{\n"workflow":"truncated'
        good = '### Fixed workflow\nUse "literal" cues.\n' + _AWM_CARD_START + '\n[]\n' + _AWM_CARD_END
        with tempfile.TemporaryDirectory() as directory, \
             patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""}), \
             patch.object(ABCDAgent, "_chat", side_effect=[bad, good]) as chat, \
             patch("eval_tod.abcd.agent.time.sleep"):
            agent = ABCDAgent(model="test", workflow=workflow, response_logger=ResponseLogger(directory))
            result = agent.induce([], [], [], editable_action_cards=True)
            self.assertEqual(chat.call_count, 2)
            self.assertIn("Previous Attempt Rejected", chat.call_args_list[1].args[0])
            self.assertIn("Unterminated string", chat.call_args_list[1].args[0])
            files = list(Path(directory).glob("*awm_induction_validation_error_response.json"))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text(encoding="utf-8"))["raw_response"], bad)
            self.assertEqual(len(list(Path(directory).glob("*awm_induction_output_response.json"))), 2)
            self.assertTrue(result.startswith("### Fixed workflow"))
            self.assertEqual(agent.last_awm_card_updates, [])

    def test_resume_diagnostics_distinguish_protocol_runtime_and_schedule(self):
        saved = {"protocol": "v2", "runtime": {"model": "old", "max_turns": 1},
                 "schedule": [["a", "b"]], "train_hash": "unchanged"}
        current = {"protocol": "v3", "runtime": {"model": "new", "max_turns": 1},
                   "schedule": [["b", "a"]], "train_hash": "unchanged"}
        diff = experiment._scope_differences(saved, current)
        self.assertEqual(set(diff), {"protocol", "runtime", "schedule"})
        self.assertEqual(diff["protocol"], {"saved": "v2", "current": "v3"})
        self.assertEqual(set(diff["runtime"]), {"model"})
        self.assertEqual(diff["schedule"]["saved"], {"batches": 1, "action_turns": 2})
        self.assertEqual(experiment._scope_differences(current, current), {})

    def test_unbounded_induction_keeps_late_targets_without_react_or_repeated_context(self):
        from eval_tod.abcd.agent import _format_abcd_turn_trajectory
        conv = _conversation("train")
        history = [{"speaker": "customer", "text": f"HISTORY_{i}_" + "x" * 200,
                    "targets": ["", "", "", [], -1]} for i in range(40)]
        conv["delexed"] = history + conv["delexed"]
        conv["original"] = [[t["speaker"], t["text"]] for t in conv["delexed"]]
        target = len(conv["delexed"]) - 1
        row = {"convo_id": "train", "turn_index": target, "target_type": "action",
               "context": "FULL_REACT_CONTEXT_SENTINEL", "predicted_action": "send-link",
               "predicted_slots": ["WRONG_VALUE"], "prediction": "AGENT_FINAL_PREDICTION_" + "y" * 800,
               "react_trace": [{"messages": "REACT_MESSAGES_SENTINEL", "observation": "TOOL_DUMP_SENTINEL"}],
               "action_selection": {"messages": "SELECTION_MESSAGES_SENTINEL"}}
        prefix = experiment.feedback_prefixes(build_action_turn_samples([conv])[-1:])[0]
        text = _format_abcd_turn_trajectory(prefix, [row], max_chars=None)
        self.assertGreater(len(text), 4000)
        self.assertIn(f"turn_index: {target}", text)
        self.assertIn('gold_slots: ["FUTURE_VALUE"]', text)
        self.assertIn('predicted_slots: ["WRONG_VALUE"]', text)
        self.assertIn("slot_correct: False", text)
        self.assertIn(row["prediction"], text)
        for i in range(40):
            self.assertEqual(text.count(f"HISTORY_{i}_"), 1)
        for forbidden in ("FULL_REACT_CONTEXT_SENTINEL", "REACT_MESSAGES_SENTINEL", "TOOL_DUMP_SENTINEL", "SELECTION_MESSAGES_SENTINEL"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(text.count("predicted_action:"), 1)
        self.assertNotIn("gold_action: enter-details", text)  # unsampled history

    def test_native_induction_does_not_drop_large_batch_summaries(self):
        from awm import WorkflowStore
        from eval_tod.abcd.agent import ABCDAgent
        from eval_tod.schemas import Prediction
        conversations = [_conversation(f"train-{i}") for i in range(30)]
        predictions = [Prediction(dialogue_id=str(i), inform_slots={}, request_slots={},
            booking={}, response_text="response") for i in range(30)]
        workflow = WorkflowStore()
        with patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""}), \
             patch.object(ABCDAgent, "_chat", return_value="### Updated pattern") as chat:
            agent = ABCDAgent(model="test", workflow=workflow, expose_scenario_labels=False)
            agent.induce(conversations, predictions, [{} for _ in conversations], turn_results=[])
        self.assertIn("### Conversation train-29", chat.call_args.args[0])

    def test_feedback_is_prefix_only_and_only_sampled_turns_have_gold(self):
        conv = _conversation("train")
        before = copy.deepcopy(conv)
        samples = build_action_turn_samples([conv])
        first = experiment.feedback_prefixes(samples[:1])[0]
        self.assertEqual(len(first["delexed"]), 2)
        self.assertNotIn("FUTURE_VALUE", json.dumps(first))
        second = experiment.feedback_prefixes(samples[1:])[0]
        self.assertEqual(second["delexed"][1]["targets"][1], "observed_history")
        self.assertEqual(second["delexed"][3]["targets"][1], "take_action")
        self.assertNotIn("scenario", second)
        self.assertEqual(conv, before)

    def test_resource_description_is_idempotent_and_uses_real_runtime_interface(self):
        text = experiment._with_resource_use("OFFLINE K0")
        self.assertEqual(experiment._with_resource_use(text), text)
        self.assertIn("retrieve_reference", text)
        self.assertIn("retrieve_action_card", text)
        self.assertIn("sampled action turns", text)

    def test_damaged_resource_markers_are_repaired_without_losing_patterns(self):
        cases = [
            f"### Pattern\nKeep this rule.\n{experiment.RESOURCE_START}",
            f"{experiment.RESOURCE_END}\n### Pattern\nKeep this rule.",
            f"{experiment.RESOURCE_END}\n### Pattern\nKeep this rule.\n{experiment.RESOURCE_START}",
            experiment.RESOURCE_USE + "\n### Pattern\nKeep this rule.\n" + experiment.RESOURCE_USE,
            f"{experiment.RESOURCE_START}\n## Offline and Online Resources\n- stale resource guide\n### Pattern\nKeep this rule.",
            f"{experiment.RESOURCE_START}\n### Pattern\nKeep this rule.\n{experiment.RESOURCE_END}",
        ]
        for text in cases:
            with self.subTest(text=text):
                result = experiment._with_resource_use(text)
                self.assertIn("### Pattern\nKeep this rule.", result)
                self.assertEqual(result.count(experiment.RESOURCE_START), 1)
                self.assertEqual(result.count(experiment.RESOURCE_END), 1)
                self.assertEqual(result.count("## Offline and Online Resources"), 1)
                self.assertEqual(experiment._with_resource_use(result), result)
                self.assertNotIn("stale resource guide", result)

    def _run_case(self, root, fail_once=False, card_updates=False, migrate_v2=False, legacy_completed=0):
        offline = root / "offline_artifact"
        offline.mkdir()
        for name, content in (("skill.md", "OFFLINE K0"), ("reference.md", "BASE REFERENCE"),
                              ("action_rules.md", "BASE ACTION RULES"), ("slot_policies.md", "BASE SLOT POLICIES")):
            (offline / name).write_text(content, encoding="utf-8")
        experiment._save(offline / "mining_usage.json", {"total": {"calls": 3, "usage_available": True, "total_tokens": 30}})
        original = {path.name: path.read_bytes() for path in offline.iterdir()}
        train, test = [_conversation("train")], [_conversation("test")]
        calls, events, fail = [0], [], [fail_once]

        def usage():
            return {"total": {"calls": calls[0], "total_tokens": calls[0] * 10, "usage_available": True}}

        class FakeAgent:
            def __init__(self, **kwargs):
                self.workflow, self.memory, self.kwargs = kwargs["workflow"], kwargs["memory"], kwargs
                self.last_awm_card_updates = []

            def predict_all_turns(self, conversation, *, turn_index, **kwargs):
                calls[0] += 1
                events.append(("predict", conversation["convo_id"], turn_index))
                if card_updates and turn_index == 3:
                    assert "EVOLVED SLOT POLICY" in self.kwargs["slot_policies_text"]
                targets = conversation["delexed"][turn_index]["targets"]
                return [{"convo_id": conversation["convo_id"], "turn_index": turn_index, "target_type": "action",
                         "predicted_action": targets[2], "predicted_slots": targets[3], "prediction": "Predicted action."}]

            def induce(self, views, predictions, metrics, *, turn_results, resource_context, editable_action_cards):
                assert editable_action_cards is True
                assert experiment.RESOURCE_START not in self.workflow.text
                assert experiment.RESOURCE_END not in self.workflow.text
                calls[0] += 1
                events.append(("induce", copy.deepcopy(views)))
                assert all(conv["convo_id"] == "train" for conv in views)
                assert "BASE SLOT POLICIES" in resource_context and "BASE ACTION RULES" in resource_context
                if fail[0] and int(turn_results[0]["turn_index"]) == (3 if legacy_completed else 1):
                    from eval_tod.abcd.agent import _AWM_WORKFLOW_RESOURCE_SECTION
                    fail[0] = False
                    self.workflow.replace(_AWM_WORKFLOW_RESOURCE_SECTION)
                    return _AWM_WORKFLOW_RESOURCE_SECTION
                # Deliberate malformed echo: must not crash the whole A2 worker.
                self.workflow.replace("### AWM learned workflow\nWhen a link is requested, use dialogue values.\n" + experiment.RESOURCE_START)
                if card_updates:
                    self.last_awm_card_updates = [{"action": "send-link", "slot_policy": "EVOLVED SLOT POLICY"}]
                return self.workflow.text

            def update_memory(self, views, predictions, metrics, **kwargs):
                self.memory.exemplars.append({"dialogue_id": "train", "domains": [], "trajectory": "verified prefix"})

        def evaluate(agent, conversations, label, subflow, **kwargs):
            calls[0] += 1
            events.append(("test",))
            self.assertEqual(conversations, test)
            self.assertIn(experiment.RESOURCE_START, agent.workflow.text)
            if card_updates:
                self.assertIn("EVOLVED SLOT POLICY", agent.kwargs["slot_policies_text"])
            else:
                self.assertEqual(agent.kwargs["slot_policies_text"], "BASE SLOT POLICIES")
            self.assertEqual(len(agent.memory.exemplars), 2)
            return {"ast_cds": {"ast_joint": .5, "ast_action_name": .6, "ast_slot_value": .7, "num_action_turns": 2}}

        args = _args(root)
        with patch("scripts.run_subflow_eval.load_subflow_data", return_value=(train, test)), \
                patch.object(experiment, "build_abcd_runtime_agent", side_effect=lambda args, **kwargs: FakeAgent(**kwargs)), \
                patch("scripts.run_subflow_eval.evaluate_agent_on_subflow", side_effect=evaluate), \
                patch("scripts.llm_usage_utils.get_usage", side_effect=usage), \
                patch("scripts.llm_usage_utils.reset_usage", side_effect=lambda: calls.__setitem__(0, 0)):
            if fail_once:
                with self.assertRaisesRegex(RuntimeError, "retaining previous workflow"):
                    experiment.run_offline_awm(args, "f", root, ["ID1", "ID2"])
                state = experiment._read(root / "offline_awm" / "awm_state.json")
                self.assertEqual(state["completed_batches"], legacy_completed)
                if not legacy_completed:
                    self.assertIn("OFFLINE K0", state["workflow"])
                if migrate_v2:
                    state["scope"]["protocol"] = experiment.V2_PROTOCOL
                    state.pop("cards")
                    experiment._save(root / "offline_awm" / "awm_state.json", state)
                    path = root / "offline_awm" / "batches" / f"batch_{legacy_completed + 1:04d}.json"
                    cached = experiment._read(path)
                    cached["identity"]["scope_hash"] = experiment._hash(state["scope"])
                    cached["identity"].pop("cards_hash")
                    experiment._save(path, cached)
                    args.migrate_offline_awm_v2 = True
                args.resume = True
            experiment.run_offline_awm(args, "f", root, ["ID1", "ID2"])
            count = len(events)
            args.resume = True
            experiment.run_offline_awm(args, "f", root, ["ID1", "ID2"])
            self.assertEqual(len(events), count)
        self.assertEqual({path.name: path.read_bytes() for path in offline.iterdir()}, original)
        self.assertEqual([event for event in events if event[0] == "predict"], [("predict", "train", 1), ("predict", "train", 3)])
        result = experiment._read(root / "offline_awm" / "result.json")
        self.assertEqual(result["training_action_turns"], 2)
        self.assertEqual(result["runtime_protocol"], abcd_runtime_options(args))
        self.assertEqual(result["llm_usage"]["generation"]["total"]["calls"], 8 if fail_once else 7)
        self.assertEqual(result["llm_usage"]["testing"]["total"]["calls"], 1)
        if migrate_v2:
            self.assertEqual(result["training_lineage"], "mixed-v2-to-v3")
            self.assertEqual(result["migration"]["completed_v2_batches"], legacy_completed)
            self.assertEqual(len(list((root / "offline_awm" / "migration_backups").glob("*/awm_state.json"))), 1)

    def test_online_training_frozen_test_and_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_case(Path(directory))

    def test_failed_induction_resumes_cached_rollout_without_losing_k0_or_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_case(Path(directory), fail_once=True)

    def test_updated_cards_reach_next_rollout_and_frozen_test(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_case(Path(directory), card_updates=True)
            root = Path(directory) / "offline_awm"
            self.assertIn("EVOLVED SLOT POLICY", (root / "slot_policies.md").read_text(encoding="utf-8"))
            self.assertIn("EVOLVED SLOT POLICY", experiment._read(root / "awm_state.json")["cards"]["slot_policies.md"])
            self.assertEqual(len(list((root / "card_updates").glob("*.json"))), 2)

    def test_v2_migration_reuses_boundary_rollout_and_preserves_failed_attempt_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_case(Path(directory), fail_once=True, card_updates=True, migrate_v2=True)

    def test_v2_migration_skips_completed_batches_and_preserves_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_case(Path(directory), fail_once=True, migrate_v2=True, legacy_completed=1)

    def test_v2_migration_preserves_completed_workflow_memory_and_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "awm_state.json"
            scope = {"protocol": experiment.PROTOCOL, "schedule": [["a"], ["b"], ["c"]],
                     "runtime": {"model": "test"}, "resource_hash": "same"}
            old = {"scope": {**scope, "protocol": experiment.V2_PROTOCOL}, "completed_batches": 2,
                   "workflow": "LEARNED V2", "exemplars": [{"dialogue_id": "kept"}],
                   "usage": {"total": {"calls": 999, "total_tokens": 12345}}}
            experiment._save(path, old)
            original = path.read_bytes()
            resources = {"action_rules.md": "K0 rules", "slot_policies.md": "K0 slots"}
            migrated = experiment.migrate_v2_checkpoint(old, scope, resources, path)
            self.assertEqual(migrated["completed_batches"], 2)
            for key in ("workflow", "exemplars", "usage"):
                self.assertEqual(migrated[key], old[key])
            self.assertEqual(migrated["cards"], resources)
            backup = Path(migrated["migration"]["backup_dir"])
            self.assertEqual((backup / "awm_state.json").read_bytes(), original)
            self.assertEqual(migrated["migration"]["planned_v3_batches"], 1)
            self.assertEqual(migrated["migration"]["first_v3_batch_number"], 3)
            self.assertEqual(experiment._read(path), migrated)

    def test_migration_refuses_nonprotocol_drift_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "awm_state.json"
            scope = {"protocol": experiment.PROTOCOL, "schedule": [["a"]], "train_hash": "new"}
            old = {"scope": {**scope, "protocol": experiment.V2_PROTOCOL, "train_hash": "old"},
                   "completed_batches": 0, "workflow": "keep", "exemplars": []}
            experiment._save(path, old)
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "protocol-only"):
                experiment.migrate_v2_checkpoint(old, scope, {}, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((path.parent / "migration_backups").exists())

    def test_card_replacement_preserves_other_actions_and_removes_duplicates(self):
        cards = {"action_rules.md": "# Rules\n#### `send-link`\nold\n#### `enter-details`\nkeep\n",
                 "slot_policies.md": "# Slots\n#### `send-link`\nwrong\n#### `enter-details`\nkeep-slot\n#### `send-link`\nobsolete\n"}
        updated = experiment.update_card_resources(cards, [{"action": "send-link", "slot_policy": "new-order"}])
        self.assertEqual(updated["action_rules.md"], cards["action_rules.md"])
        self.assertEqual(updated["slot_policies.md"].count("#### `send-link`"), 1)
        self.assertIn("keep-slot", updated["slot_policies.md"])
        self.assertNotIn("obsolete", updated["slot_policies.md"])
        self.assertEqual(experiment.update_card_resources(updated, [{"action": "send-link", "slot_policy": "new-order"}]), updated)

    def test_native_awm_card_proposal_validates_and_keeps_single_induction(self):
        from awm import WorkflowStore
        from eval_tod.abcd.agent import ABCDAgent, _parse_awm_card_update
        workflow = WorkflowStore()
        workflow.replace("OLD")
        raw = json.dumps({"workflow": "### Updated workflow", "action_card_updates": [
            {"action": "send-link", "slot_policy": "Use the validated email from the current dialogue."}]})
        with patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""}), \
             patch.object(ABCDAgent, "_chat", return_value=raw) as chat:
            agent = ABCDAgent(model="test", workflow=workflow, expose_scenario_labels=False)
            result = agent.induce([], [], [], resource_context="CURRENT CARDS", editable_action_cards=True)
        self.assertEqual(chat.call_count, 1)
        self.assertIn("Writable Action Cards", chat.call_args.args[0])
        self.assertEqual(agent.last_awm_card_updates[0]["action"], "send-link")
        self.assertTrue(result.startswith("### Updated workflow"))
        for updates in ([{"action": "not-valid", "slot_policy": "bad"}],
                        [{"action": "send-link", "slot_policy": "#### `enter-details`\nhijack"}],
                        [{"action": "send-link", "slot_policy": ""}],
                        [{"action": "send-link", "slot_policy": "a"}, {"action": "send-link", "slot_policy": "b"}]):
            with self.assertRaises(ValueError):
                _parse_awm_card_update(json.dumps({"workflow": "ok", "action_card_updates": updates}), {"send-link"})

    def test_invalid_card_response_does_not_mutate_workflow(self):
        from awm import WorkflowStore
        from eval_tod.abcd.agent import ABCDAgent
        workflow = WorkflowStore()
        workflow.replace("OLD")
        with patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""}), \
             patch.object(ABCDAgent, "_chat", return_value='{"workflow":"bad","action_card_updates":[{"action":"BAD"}]}'), \
             patch("eval_tod.abcd.agent.time.sleep"):
            agent = ABCDAgent(model="test", workflow=workflow)
            with self.assertRaisesRegex(RuntimeError, "no update applied"):
                agent.induce([], [], [], editable_action_cards=True)
        self.assertEqual(workflow.text, "OLD")
        self.assertEqual(agent.last_awm_card_updates, [])

    def test_native_awm_receives_read_only_resource_context(self):
        from awm import WorkflowStore
        from eval_tod.abcd.agent import ABCDAgent
        from eval_tod.schemas import Prediction
        workflow = WorkflowStore()
        workflow.replace("OFFLINE K0")
        with patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""}), \
                patch.object(ABCDAgent, "_chat", return_value="### Learned pattern\nUse current dialogue values.") as chat:
            agent = ABCDAgent(model="test", workflow=workflow, expose_scenario_labels=False)
            agent.induce([_conversation("train")], [Prediction(dialogue_id="train", inform_slots={}, request_slots={}, booking={}, response_text="")],
                         [{"action_total": 2, "action_correct": 1}], resource_context="OFFLINE SLOT RESOURCE")
        prompt = chat.call_args.args[0]
        self.assertIn("OFFLINE K0", prompt)
        self.assertIn("OFFLINE SLOT RESOURCE", prompt)
        self.assertNotIn("SECRET_LABEL", prompt)

    def test_add_arm_preserves_old_resume_config_and_runs_only_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = dict(arms=list(a2.ARMS), model="deepseek-chat", max_train=None, max_test=None,
                            max_batches=None, batch_size=8, refinement_mode="standard", skip_utterance_eval=True)
            a2._json(root / "a2_config.json", {"arguments": previous})
            original = (root / "a2_config.json").read_bytes()
            argv = ["run_a2_abcd.py", "--subflow", "f", "--output-dir", str(root), "--resume", "--skip-utterance-eval",
                    "--add-arms", "offline_awm", "--skip-arms", ",".join(a2.ARMS), "--migrate-offline-awm-v2"]
            with patch.object(sys, "argv", argv), patch.object(a2, "_subflows", return_value=["f"]), \
                    patch.object(a2.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                run.return_value.returncode = 0
                self.assertEqual(a2.main(), 0)
            self.assertEqual((root / "a2_config.json").read_bytes(), original)
            command = run.call_args.args[0]
            self.assertIn("offline_awm", command[command.index("--arms") + 1])
            self.assertIn("--migrate-offline-awm-v2", command)
            aggregate = a2._read(root / "a2_aggregate.json")
            self.assertIn("offline_awm", aggregate["arms"])
            self.assertEqual(aggregate["execution_arms"], ["offline_awm"])


if __name__ == "__main__":
    unittest.main()
