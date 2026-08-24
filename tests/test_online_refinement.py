import unittest
from unittest.mock import patch

from skill_mining.online_refinement import (
    RefinementPolicy,
    apply_refinement_patches,
    build_guard_induction_context,
    edge_confidence,
    initialize_skill_dag,
    propose_refinement_patches,
    render_online_resources,
    schedule_contrastive_batches,
)


def _subgraph():
    return {
        "nodes": [
            {"id": "a", "label": "enter-details"},
            {"id": "b", "label": "send-link"},
            {"id": "c", "label": "make-password"},
        ],
        "edges": [
            {"source": "a", "target": "b", "support": 10, "num_sessions": 8},
            {"source": "a", "target": "c", "support": 5, "num_sessions": 4},
            {"source": "b", "target": "b", "support": 2, "num_sessions": 2},
        ],
        "backbone": {
            "compilation_order": ["a", "b", "c"],
            "edges": [{"source": "a", "target": "b"}],
        },
    }


class OnlineRefinementTest(unittest.TestCase):
    def test_initialization_separates_backbone_branch_and_retry(self):
        state = initialize_skill_dag(_subgraph(), "account_access")
        self.assertEqual(state["edges"]["a=>b"]["kind"], "backbone")
        self.assertEqual(state["edges"]["a=>b"]["visibility"], "skill")
        self.assertEqual(state["edges"]["a=>c"]["kind"], "candidate_branch")
        self.assertEqual(state["edges"]["b=>b"]["kind"], "retry")

    @patch("skill_mining.online_refinement._actions")
    def test_scheduler_pairs_competing_targets_and_limits_duplicates(self, actions):
        actions.side_effect = lambda conv: conv["actions"]
        state = initialize_skill_dag(_subgraph(), "account_access")
        conversations = [
            {"convo_id": "1", "actions": ["enter-details", "send-link"]},
            {"convo_id": "2", "actions": ["enter-details", "send-link"]},
            {"convo_id": "3", "actions": ["enter-details", "make-password"]},
            {"convo_id": "4", "actions": ["enter-details", "make-password", "send-link"]},
        ]
        batches = schedule_contrastive_batches(conversations, state, batch_size=4, per_transition_cap=1)
        first = {item["convo_id"] for item in batches[0]}
        self.assertTrue(first & {"1", "2"})
        self.assertTrue(first & {"3", "4"})
        self.assertEqual({item["convo_id"] for batch in batches for item in batch}, {"1", "2", "3", "4"})

    def test_promotion_requires_resolved_guard_and_deferred_resource_is_rendered(self):
        state = initialize_skill_dag(_subgraph(), "account_access")
        edge = state["edges"]["a=>c"]
        edge.update({"gold_support": 5, "rollout_success": 5, "guard_status": "pending"})
        patches = propose_refinement_patches(state, RefinementPolicy(min_gold_support=3, min_confidence=0.6))
        promotion = next(patch for patch in patches if patch["operation"] == "promote_to_skill")
        self.assertTrue(any(patch["operation"] == "induce_guard" for patch in patches))
        apply_refinement_patches(state, [promotion])
        self.assertEqual(state["edges"]["a=>c"]["visibility"], "reference")
        state["edges"]["a=>c"]["guard"] = "The customer is creating a new password."
        state["edges"]["a=>c"]["guard_status"] = "resolved"
        second = propose_refinement_patches(state, RefinementPolicy(min_gold_support=3, min_confidence=0.6))
        apply_refinement_patches(state, [next(patch for patch in second if patch["operation"] == "promote_to_skill")])
        self.assertEqual(state["edges"]["a=>c"]["visibility"], "skill")
        self.assertAlmostEqual(edge_confidence(state["edges"]["a=>c"]), 6 / 7)
        skill, reference = render_online_resources(state)
        self.assertIn("enter-details", skill)
        self.assertNotIn("enter-details -> make-password", reference)

    def test_guard_context_contains_siblings_and_failure_evidence(self):
        state = initialize_skill_dag(_subgraph(), "account_access")
        state["edges"]["a=>c"]["evidence"] = [{
            "conversation_id": "case-1", "action_success": False,
            "context": "Customer forgot a password", "react_trace": [],
        }]
        context = build_guard_induction_context(state, "a=>c")
        self.assertEqual(context["target_edge"]["edge_id"], "a=>c")
        self.assertEqual(context["sibling_edges"][0]["edge_id"], "a=>b")
        self.assertEqual(context["target_edge"]["negative_cases"][0]["conversation_id"], "case-1")


if __name__ == "__main__":
    unittest.main()
