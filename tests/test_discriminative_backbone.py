import unittest

from skill_mining.backbone_workflow_mining import (
    mine_backbone_workflow,
    mine_backbone_workflow_session_coverage,
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


if __name__ == "__main__":
    unittest.main()
