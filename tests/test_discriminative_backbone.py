import unittest

from skill_mining.backbone_workflow_mining import (
    mine_backbone_workflow,
    mine_backbone_workflow_heuristics,
    mine_backbone_workflow_motif_trace_cover,
    mine_backbone_workflow_observable_trace_cover,
    mine_backbone_workflow_structural_trace_cover,
    mine_backbone_workflow_session_coverage,
    mine_backbone_workflow_trace_cover,
)


def _conversation(convo_id, actions):
    turns = []
    original = []
    for action in actions:
        turns.append({
            "speaker": "action",
            "text": "",
            "targets": ["", "take_action", action, []],
        })
        original.append(["action", ""])
    return {"convo_id": str(convo_id), "delexed": turns, "original": original}


class DiscriminativeBackboneTest(unittest.TestCase):
    def setUp(self):
        self.conversations = [
            _conversation("a1", ["pull-up-account", "verify-identity", "send-link"]),
            _conversation("a2", ["pull-up-account", "verify-identity", "send-link"]),
            _conversation("a3", ["pull-up-account", "verify-identity", "send-link"]),
            _conversation("b1", ["pull-up-account", "verify-identity", "make-password"]),
            _conversation("b2", ["pull-up-account", "verify-identity", "make-password"]),
            _conversation("b3", ["pull-up-account", "verify-identity", "make-password"]),
        ]

    def test_discriminative_metadata_is_recorded(self):
        result = mine_backbone_workflow("account_access", self.conversations)
        graph = result["subgraph"]
        self.assertEqual(graph["mining_method"], "discriminative_backbone")
        self.assertIn("cohort_reweighting", graph)
        edge = next(item for item in graph["edges"] if item["target"].endswith("send-link"))
        self.assertIn("base_weight", edge)
        self.assertIn("discriminative_log_odds", edge)
        self.assertIn("final_backbone_weight", edge)
        self.assertGreater(edge["discriminative_log_odds"], 0.0)

    def test_coverage_alias_has_identical_discriminative_result(self):
        direct = mine_backbone_workflow("account_access", self.conversations)
        alias = mine_backbone_workflow_session_coverage("account_access", self.conversations)
        self.assertEqual(direct["subgraph"]["mining_method"], "discriminative_backbone")
        self.assertEqual(direct["subgraph"]["backbone"], alias["subgraph"]["backbone"])
        self.assertEqual(direct["subgraph"]["edges"], alias["subgraph"]["edges"])

    def test_heuristics_baseline_records_dependency(self):
        result = mine_backbone_workflow_heuristics("account_access", self.conversations)
        graph = result["subgraph"]
        self.assertEqual(graph["mining_method"], "heuristics_dependency_backbone")
        self.assertTrue(all("dependency" in edge for edge in graph["edges"]))

    def test_trace_cover_reaches_target_with_compact_residual_set(self):
        result = mine_backbone_workflow_trace_cover(
            "account_access",
            self.conversations,
            trace_coverage_target=0.8,
            corpus_fitness_target=0.95,
        )
        graph = result["subgraph"]
        metadata = graph["trace_cover"]
        self.assertEqual(graph["mining_method"], "saturated_trace_cover")
        self.assertGreaterEqual(metadata["final_fitness"], 0.95)
        retained = {
            (edge["source"], edge["target"])
            for edges in graph["local_transitions"].values()
            for edge in edges
        }
        self.assertLessEqual(len(retained), len(graph["edges"]))

    def test_observable_trace_cover_emits_routing_modes(self):
        conversations = []
        for index in range(5):
            conversation = _conversation(
                f"email-{index}", ["pull-up-account", "verify-identity", "send-link"],
            )
            conversation["original"].insert(2, ["customer", "please send an email link"])
            conversation["delexed"].insert(2, {
                "speaker": "customer", "text": "please send an email link", "targets": [],
            })
            conversations.append(conversation)
        for index in range(5):
            conversation = _conversation(
                f"password-{index}", ["pull-up-account", "verify-identity", "make-password"],
            )
            conversation["original"].insert(2, ["customer", "I need a new password"])
            conversation["delexed"].insert(2, {
                "speaker": "customer", "text": "i need a new password", "targets": [],
            })
            conversations.append(conversation)
        result = mine_backbone_workflow_observable_trace_cover(
            "account_access", conversations,
            routing_mode_min_leaf_support=2,
        )
        graph = result["subgraph"]
        self.assertEqual(graph["mining_method"], "observable_trace_cover")
        self.assertGreater(graph["observable_router"]["num_modes"], 0)
        self.assertGreater(
            graph["trace_cover"]["mean_selected_pair_distinguishability"], 0.0,
        )

    def test_structural_trace_cover_splits_only_on_action_history(self):
        conversations = [
            _conversation(f"left-{index}", ["pull-up-account", "verify-identity", "send-link"])
            for index in range(8)
        ] + [
            _conversation(f"right-{index}", ["make-password", "verify-identity", "pull-up-account"])
            for index in range(8)
        ]
        result = mine_backbone_workflow_structural_trace_cover(
            "account_access", conversations,
            state_min_leaf_support=2,
            state_node_penalty=0.0,
            state_edge_penalty=0.0,
        )
        graph = result["subgraph"]
        refined = graph["structural_refinement"]
        router = refined["routers"]["account_access:verify-identity"]
        self.assertEqual(graph["mining_method"], "structural_trace_cover")
        self.assertGreater(refined["extra_state_nodes"], 0)
        self.assertLess(refined["refined_routing_entropy"], refined["coarse_routing_entropy"])
        split_features = []
        stack = [router["tree"]]
        while stack:
            node = stack.pop()
            if node.get("kind") == "split":
                split_features.append(node["feature"])
                stack.extend([node["present"], node["absent"]])
        self.assertTrue(split_features)
        self.assertTrue(all(feature.startswith(("lag", "seen:", "start:", "position_", "current_")) for feature in split_features))

    def test_motif_trace_cover_splits_same_action_into_process_roles(self):
        conversations = [
            _conversation(f"left-{index}", ["pull-up-account", "verify-identity", "send-link"])
            for index in range(8)
        ] + [
            _conversation(f"right-{index}", ["make-password", "verify-identity", "pull-up-account"])
            for index in range(8)
        ]
        result = mine_backbone_workflow_motif_trace_cover(
            "account_access", conversations,
            motif_history_order=1,
            motif_min_support=2,
            motif_state_node_penalty=0.0,
            motif_state_edge_penalty=0.0,
        )
        graph = result["subgraph"]
        refined = graph["motif_refinement"]
        router = refined["routers"]["account_access:verify-identity"]
        self.assertEqual(graph["mining_method"], "motif_trace_cover")
        self.assertEqual(len(router["modes"]), 2)
        self.assertGreater(refined["extra_state_nodes"], 0)
        self.assertLess(refined["refined_routing_entropy"], refined["coarse_routing_entropy"])
        self.assertTrue(all("path=" in key for key in router["signature_to_mode"]))


if __name__ == "__main__":
    unittest.main()
