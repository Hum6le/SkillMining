import argparse
import contextlib
import copy
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_a3_abcd as a3


def _graph():
    return {
        "nodes": [{"id": "f:a", "label": "enter-details"}, {"id": "f:b", "label": "send-link"},
                  {"id": "f:c", "label": "make-password"}],
        "backbone": {"root": "<START>", "compilation_order": ["f:a", "f:b", "f:c"],
                     "edges": [{"source": "<START>", "target": "f:a"}, {"source": "f:a", "target": "f:b"},
                               {"source": "f:b", "target": "f:c"}]},
        "edges": [{"source": "f:a", "target": "f:b"}, {"source": "f:b", "target": "f:c"},
                  {"source": "f:a", "target": "f:c"}, {"source": "f:c", "target": "f:c"},
                  {"source": "f:c", "target": "f:a"}],
        "local_transitions": {"f:a": [{"source": "f:a", "target": "f:b"}, {"source": "f:a", "target": "f:c"}],
                              "f:b": [{"source": "f:b", "target": "f:c"}],
                              "f:c": [{"source": "f:c", "target": "f:c"}]},
        "transition_induction": {"rules_by_source": {
            "f:a": [{"target": "f:b", "condition": "common-guard", "status": "resolved"},
                    {"target": "f:c", "condition": "residual-only-guard", "status": "resolved"}],
            "f:b": [{"target": "f:c", "condition": "second-tree-guard", "status": "resolved"}],
            "f:c": [{"target": "f:c", "condition": "retry-only-guard", "status": "underspecified"},
                    {"target": "f:a", "condition": "observed-unselected-guard", "status": "resolved"}]}}
    }


def _args(root, **changes):
    values = dict(output_dir=str(root), offline_root=str(root / "source"), arms=a3.ARMS, skip_arms=(),
                  resume=False, max_train=None, max_test=None, model="deepseek-chat", workflow_max_chars=8000,
                  reference_top_k=3, reference_max_chars=1800, action_selection_candidate_limit=3,
                  skip_utterance_eval=True, random_seeds=a3.DEFAULT_RANDOM_SEEDS)
    values.update(changes)
    return argparse.Namespace(**values)


def _source(root):
    source = root / "source" / "f" / "offline_artifact"
    source.mkdir(parents=True)
    a3._json(source / "subgraph.json", _graph())
    a3._json(source / "complete.json", {"subflow": "f", "max_train": None})
    a3._json(source / "mining_usage.json", {"total": {"calls": 3, "usage_available": True, "total_tokens": 50}})
    (source / "action_rules.md").write_text("# Action Rules\n#### `enter-details`\nShared procedure.", encoding="utf-8")
    (source / "slot_policies.md").write_text("# Slot Policies\n#### `enter-details`\nShared slots.", encoding="utf-8")
    return source


class A3VariantsTest(unittest.TestCase):
    def test_same_cards_and_conditions_different_layout(self):
        graph = _graph()
        original = copy.deepcopy(graph)
        variants = a3.build_variants("f", graph, "cards", "slots")
        for variant in variants.values():
            self.assertEqual(variant["resources"]["action_rules.md"], "cards")
            self.assertEqual(variant["resources"]["slot_policies.md"], "slots")
        self.assertEqual(a3._pairs(variants["flat"]["edges"]), a3._pairs(variants["backbone_residual"]["edges"]))
        for condition in ("common-guard", "residual-only-guard", "retry-only-guard"):
            for arm in ("flat", "backbone_residual"):
                self.assertIn(condition, variants[arm]["resources"]["reference.md"])
        self.assertNotIn("## Backbone", variants["flat"]["resources"]["skill.md"])
        self.assertNotIn("Placement:", variants["flat"]["resources"]["reference.md"])
        self.assertIn("## Backbone", variants["backbone_residual"]["resources"]["skill.md"])
        self.assertEqual(graph, original)
        self.assertEqual(variants, a3.build_variants("f", graph, "cards", "slots"))

    def test_full_graph_keeps_observed_edges_while_flat_keeps_only_selected(self):
        variants = a3.build_variants("f", _graph(), "cards", "slots")
        self.assertNotIn("backbone_only", variants)
        self.assertEqual(len(variants), 6)
        self.assertEqual(len(variants["flat"]["edges"]), 4)
        self.assertEqual(len(variants["full_graph_flat"]["edges"]), 5)
        self.assertNotIn("observed-unselected-guard", variants["flat"]["resources"]["reference.md"])
        self.assertIn("observed-unselected-guard", variants["full_graph_flat"]["resources"]["reference.md"])
        self.assertEqual(variants["flat"]["resources"]["skill.md"], variants["full_graph_flat"]["resources"]["skill.md"])

    def test_random_backbones_are_real_rooted_trees_with_same_semantics(self):
        import networkx as nx
        graph = _graph()
        variants = a3.build_variants("f", graph, "cards", "slots", random_seeds=range(10))
        selected = set(a3._pairs(variants["backbone_residual"]["edges"]))
        changed = []
        for seed in range(10):
            variant = variants[f"random_backbone/seed_{seed}"]
            backbone = variant["layout_backbone"]
            pairs = set(a3._pairs(backbone["edges"]))
            network = nx.DiGraph(list(pairs))
            self.assertTrue(nx.is_arborescence(network))
            self.assertEqual(network.in_degree("<START>"), 0)
            self.assertEqual(set(network), {"<START>", "f:a", "f:b", "f:c"})
            self.assertTrue((pairs - {("<START>", "f:a")}) <= selected)
            self.assertEqual(set(a3._pairs(variant["edges"])), selected)
            for condition in ("common-guard", "second-tree-guard", "residual-only-guard", "retry-only-guard"):
                self.assertIn(condition, variant["resources"]["reference.md"])
            changed.append(variant["randomization"]["changed"])
        self.assertTrue(any(changed))

    def test_unique_tree_is_reported_unchanged_not_rewired(self):
        graph = _graph()
        graph["local_transitions"] = {"f:a": [{"source": "f:a", "target": "f:b"}],
                                      "f:b": [{"source": "f:b", "target": "f:c"}]}
        variants = a3.build_variants("f", graph, "cards", "slots")
        for seed in a3.DEFAULT_RANDOM_SEEDS:
            info = variants[f"random_backbone/seed_{seed}"]["randomization"]
            self.assertFalse(info["changed"])
            self.assertEqual(info["tree_edge_change_fraction"], 0)
            self.assertEqual(variants[f"random_backbone/seed_{seed}"]["resources"], variants["backbone_residual"]["resources"])

    def test_full_graph_requires_guards_for_unselected_edges_too(self):
        graph = _graph()
        graph["transition_induction"]["rules_by_source"]["f:c"] = graph["transition_induction"]["rules_by_source"]["f:c"][:1]
        with self.assertRaisesRegex(ValueError, "common induction pool"):
            a3.build_variants("f", graph, "cards", "slots")
        variants = a3.build_variants("f", graph, "cards", "slots", arms=("flat",))
        self.assertEqual(list(variants), ["flat"])

    def test_random_tree_cannot_add_root_shortcut_to_disconnected_action(self):
        graph = _graph()
        graph["nodes"].append({"id": "f:isolated", "label": "isolated"})
        with self.assertRaisesRegex(ValueError, "spanning arborescence"):
            a3.build_variants("f", graph, "cards", "slots", arms=("random_backbone",))

    def test_missing_induction_is_error_not_silent_omission(self):
        graph = _graph()
        graph["transition_induction"]["rules_by_source"]["f:c"] = []
        with self.assertRaisesRegex(ValueError, "lacks induced"):
            a3.build_variants("f", graph, "cards", "slots")

    def test_graph_metrics_have_explicit_denominators(self):
        variants = a3.build_variants("f", _graph(), "cards", "slots")
        conversations = [{"seq": ["f:a", "f:b"]}, {"seq": ["f:a", "f:c", "f:c"]}, {"seq": ["f:c"]}]
        with patch("scripts.eval_offline_graph_10flow.action_sequence", side_effect=lambda flow, conv: conv["seq"]):
            full = a3.graph_metrics("f", variants["backbone_residual"], conversations)
            random = a3.graph_metrics("f", variants["random_backbone/seed_0"], conversations)
        self.assertEqual(full["edges"], 4)
        self.assertEqual(full["transition_events"], 3)
        self.assertEqual(full["complete_route_rate"], 1)
        self.assertEqual(random["transition_coverage"], full["transition_coverage"])
        self.assertEqual(random["complete_route_rate"], full["complete_route_rate"])
        self.assertEqual(random["action_coverage"], 1)


class A3RunnerTest(unittest.TestCase):
    def test_frozen_eval_shared_cache_resume_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory)
            source = _source(root)
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            args = _args(root)
            train, test = [{"convo_id": "train-only"}], [{"convo_id": "test-only"}]
            usage = {"total": {"calls": 2, "total_tokens": 100, "usage_available": True}}
            result = {"ast_cds": {"ast_joint": .5, "ast_action_name": .6, "ast_slot_value": .7, "num_action_turns": 2}}
            with patch("scripts.run_subflow_eval.load_subflow_data", return_value=(train, test)), \
                    patch("scripts.run_subflow_eval.mine_subflow_skill_backbone") as mine, \
                    patch("scripts.run_subflow_eval.evaluate_agent_on_subflow", side_effect=lambda *a, **k: copy.deepcopy(result)) as evaluate, \
                    patch("eval_tod.abcd.agent.ABCDAgent") as agent, \
                    patch("scripts.llm_usage_utils.get_usage", return_value=usage), \
                    patch.object(a3, "graph_metrics", return_value={"edges": 3}):
                a3._run_flow(args, "f", ["ID1"])
                self.assertEqual(evaluate.call_count, 6)
                mine.assert_not_called()
                for call in evaluate.call_args_list:
                    self.assertEqual(call.args[1], test)
                for call in agent.call_args_list:
                    self.assertEqual(call.kwargs["reference_max_chars"], 1800)
                    self.assertFalse(call.kwargs["expose_scenario_labels"])
                args.resume = True
                a3._run_flow(args, "f", ["ID1"])
                self.assertEqual(evaluate.call_count, 6)
                (source / "action_rules.md").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed resources"):
                    a3._run_flow(args, "f", ["ID1"])
                (source / "action_rules.md").write_bytes(before["action_rules.md"])
            self.assertEqual({path.name: path.read_bytes() for path in source.iterdir()}, before)
            saved = a3._read(root / "f" / "flat" / "result.json")
            self.assertFalse(saved["online_refinement"])
            self.assertEqual(saved["llm_usage"]["generation"]["total"]["calls"], 3)

    def test_cached_training_scope_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source(root)
            with self.assertRaisesRegex(ValueError, "max_train"):
                a3._prepare(_args(root, max_train=20), "f", [], root / "f")

    def test_aggregate_weighting_and_missing_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for flow, turns, score in (("f", 2, .5), ("g", 6, 1)):
                a3._json(root / flow / "flat" / "result.json", {
                    "ast_cds": {"num_action_turns": turns, "ast_joint": score, "ast_action_name": score, "ast_slot_value": score},
                    "graph_metrics": {"edges": 3, "nodes": 4, "covered_transitions": 1, "transition_events": 2,
                                      "complete_routes": 1, "route_dialogues": 3, "covered_actions": 2, "action_events": 3}})
            a3._aggregate(_args(root), ["f", "g"])
            aggregate = a3._read(root / "a3_aggregate.json")["aggregate"]
            self.assertEqual(aggregate["flat"]["joint_ast"], .875)
            self.assertEqual(aggregate["flat"]["total_edges"], 6)
            self.assertEqual(aggregate["flat"]["mean_edges"], 3)
            self.assertIsNone(aggregate["flat"]["runtime_tokens"])
            self.assertFalse(aggregate["backbone_residual"]["complete"])
            self.assertIsNone(aggregate["backbone_residual"]["joint_ast"])
            self.assertIsNone(aggregate["random_backbone"]["joint_ast"])

    def test_parent_forwards_runtime_and_reuse_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["run_a3_abcd.py", "--subflow", "f", "--output-dir", str(root / "out"),
                    "--offline-root", str(root / "source"), "--workflow-ids", "ID1,ID2", "--skip-arms", "flat",
                    "--skip-utterance-eval", "--reference-max-chars", "2000"]
            with patch.object(sys, "argv", argv), patch.object(a3, "_subflows", return_value=["f"]), \
                    patch.object(a3.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                run.return_value.returncode = 0
                self.assertEqual(a3.main(), 0)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--reference-max-chars") + 1], "2000")
                self.assertEqual(command[command.index("--workflow-ids") + 1], "ID1,ID2")
                self.assertEqual(command[command.index("--skip-arms") + 1], "flat")
                self.assertIn("--offline-root", command)
                self.assertIn("--skip-utterance-eval", command)
                self.assertEqual(command[command.index("--random-seeds") + 1], "0,1,2")

    def test_random_seed_aggregation_does_not_pool_duplicate_turns_or_hide_missing_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root, random_seeds=(0, 1))
            for seed in args.random_seeds:
                for flow, turns, score in (("f", 2, .5 - seed * .25), ("g", 6, 1 - seed * .25)):
                    a3._json(root / flow / "random_backbone" / f"seed_{seed}" / "result.json", {
                        "ast_cds": {"num_action_turns": turns, "ast_joint": score, "ast_action_name": score, "ast_slot_value": score},
                        "graph_metrics": {"edges": 3, "nodes": 4, "covered_transitions": 1, "transition_events": 2,
                                          "complete_routes": 1, "route_dialogues": 3, "covered_actions": 2, "action_events": 3},
                        "llm_usage": {"testing": {"total": {"calls": 2, "total_tokens": 100, "usage_available": True}}}})
            a3._aggregate(args, ["f", "g"])
            summary = a3._read(root / "a3_aggregate.json")["aggregate"]["random_backbone"]
            self.assertEqual(summary["joint_ast"], .75)
            self.assertAlmostEqual(summary["joint_ast_std"], .1767766952966369)
            self.assertEqual(summary["action_turns_per_seed"], 8)
            self.assertEqual(summary["total_edges"], 6)
            self.assertEqual(summary["runtime_tokens"], 200)
            self.assertEqual(summary["total_testing_tokens_all_seeds"], 400)
            args.random_seeds = (0, 1, 2)
            a3._aggregate(args, ["f", "g"])
            summary = a3._read(root / "a3_aggregate.json")["aggregate"]["random_backbone"]
            self.assertFalse(summary["complete"])
            self.assertIsNone(summary["joint_ast"])
            self.assertEqual(summary["completed_seeds"], 2)

    def test_removed_arm_and_bad_seed_flags_are_rejected(self):
        for flag, value in (("--arms", "backbone_only"), ("--random-seeds", "0,0"),
                            ("--random-seeds", ""), ("--random-seeds", "a")):
            with self.subTest(flag=flag, value=value), patch.object(sys, "argv", [
                "run_a3_abcd.py", "--subflow", "f", flag, value]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                a3.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
