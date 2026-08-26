import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from skill_mining.online_refinement import (
    RefinementPolicy,
    apply_refinement_patches,
    build_guard_induction_context,
    edge_confidence,
    initialize_skill_dag,
    load_skill_dag,
    induce_joint_refinement_patches,
    autonomous_resource_reflection,
    localize_rollout_batch,
    propose_refinement_patches,
    render_online_resources,
    render_online_slot_policies,
    schedule_contrastive_batches,
    summarize_refinement_state,
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
    def test_scheduler_pairs_competing_targets_without_full_dataset_backfill(self, actions):
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
        selected = {item["convo_id"] for batch in batches for item in batch}
        self.assertEqual(len(selected), 2)
        self.assertLess(len(selected), len(conversations))

    @patch("skill_mining.online_refinement._actions")
    def test_scheduler_keeps_three_sibling_targets_in_each_round(self, actions):
        actions.side_effect = lambda conv: conv["actions"]
        subgraph = _subgraph()
        subgraph["nodes"].append({"id": "d", "label": "verify-identity"})
        subgraph["edges"].append({"source": "a", "target": "d", "support": 4, "num_sessions": 3})
        state = initialize_skill_dag(subgraph, "account_access")
        conversations = [
            {"convo_id": "b1", "actions": ["enter-details", "send-link", "send-link-context-1"]},
            {"convo_id": "b2", "actions": ["enter-details", "send-link", "send-link-context-2"]},
            {"convo_id": "c1", "actions": ["enter-details", "make-password", "password-context-1"]},
            {"convo_id": "c2", "actions": ["enter-details", "make-password", "password-context-2"]},
            {"convo_id": "d1", "actions": ["enter-details", "verify-identity", "identity-context-1"]},
            {"convo_id": "d2", "actions": ["enter-details", "verify-identity", "identity-context-2"]},
        ]
        batches = schedule_contrastive_batches(conversations, state, batch_size=8, per_transition_cap=2)
        self.assertEqual(len(batches), 1)
        for batch in batches:
            targets = {item["actions"][1] for item in batch}
            self.assertEqual(targets, {"send-link", "make-password", "verify-identity"})

    @patch("skill_mining.online_refinement._actions")
    def test_scheduler_targets_thirty_percent_of_training_sessions(self, actions):
        actions.side_effect = lambda conv: conv["actions"]
        state = initialize_skill_dag(_subgraph(), "account_access")
        conversations = []
        for target in ("send-link", "make-password"):
            for index in range(10):
                conversations.append({
                    "convo_id": f"{target}-{index}",
                    "actions": ["enter-details", target, f"context-{target}-{index}"],
                })
        batches = schedule_contrastive_batches(
            conversations, state, batch_size=8, per_transition_cap=1,
            target_selection_rate=0.30,
        )
        selected = {item["convo_id"] for batch in batches for item in batch}
        self.assertGreaterEqual(len(selected), 6)
        self.assertLessEqual(len(selected), 7)

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

    def test_localization_scores_the_target_decision_not_both_endpoints(self):
        state = initialize_skill_dag(_subgraph(), "account_access")
        conversation = {
            "convo_id": "case-1",
            "delexed": [
                {"targets": ["", "take_action", "enter-details", []]},
                {"targets": ["", "take_action", "send-link", []]},
            ],
        }
        rows = [
            {"convo_id": "case-1", "turn_index": 0, "predicted_action": "make-password", "predicted_slots": [], "context": "", "react_trace": []},
            {"convo_id": "case-1", "turn_index": 1, "predicted_action": "send-link", "predicted_slots": [], "context": "", "react_trace": []},
        ]
        localize_rollout_batch([conversation], rows, state)
        edge = state["edges"]["a=>b"]
        self.assertEqual(edge["rollout_success"], 1)
        self.assertEqual(edge["rollout_failure"], 0)

    def test_slot_failures_are_action_conditioned_and_request_policy_patch(self):
        state = initialize_skill_dag(_subgraph(), "account_access")
        conversation = {
            "convo_id": "case-1",
            "delexed": [
                {"targets": ["", "take_action", "enter-details", []]},
                {"targets": ["", "take_action", "send-link", ["gold"]]},
            ],
        }
        rows = [
            {"convo_id": "case-1", "turn_index": 0, "predicted_action": "enter-details", "predicted_slots": [], "context": "", "react_trace": []},
            {"convo_id": "case-1", "turn_index": 1, "predicted_action": "send-link", "predicted_slots": ["wrong"], "context": "customer supplied a value", "react_trace": []},
        ]
        for _ in range(3):
            localize_rollout_batch([conversation], rows, state)
        patches = propose_refinement_patches(state, RefinementPolicy(min_slot_support=3))
        self.assertTrue(any(patch["operation"] == "induce_slot_policy" for patch in patches))
        state["slot_policies"]["send-link"].update({"policy": "Use the current customer-provided value after confirmation.", "status": "resolved"})
        self.assertIn("#### `send-link`", render_online_slot_policies(state))

    def test_load_legacy_slot_policy_record_hydrates_rollout_counters(self):
        state = initialize_skill_dag(_subgraph(), "account_access")
        state["slot_policies"]["send-link"] = {"action": "send-link", "policy": "old", "status": "resolved"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skill_dag_state.json"
            path.write_text(json.dumps(state), encoding="utf-8")
            restored = load_skill_dag(path)
        policy = restored["slot_policies"]["send-link"]
        self.assertEqual(policy["slot_total"], 0)
        self.assertEqual(policy["slot_success"], 0)
        self.assertEqual(policy["slot_failures"], 0)

    @patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""})
    @patch("llm.chat")
    def test_joint_reflection_updates_guard_and_slot_policy_together(self, chat, _config):
        state = initialize_skill_dag(_subgraph(), "account_access")
        state["edges"]["a=>c"]["evidence"] = [{
            "conversation_id": "case", "action_success": False, "slot_success": False,
            "context": "Customer forgot their password", "react_trace": [],
        }]
        state["slot_policies"]["make-password"] = {
            "action": "make-password", "slot_total": 3, "slot_success": 1,
            "slot_failures": 2, "evidence": [], "policy": "", "status": "pending",
        }
        chat.return_value = '''{
          "guards":[{"edge_id":"a=>c","guard":"The customer is creating a password.","status":"resolved","rationale":"distinct goal"}],
          "slot_policies":[{"action":"make-password","policy":"Use only the newly confirmed customer value.","status":"resolved"}]
        }'''
        result = induce_joint_refinement_patches(state, [
            {"operation": "induce_guard", "edge_id": "a=>c"},
            {"operation": "induce_slot_policy", "action": "make-password"},
        ], "# skill", "# slots", "test")
        self.assertEqual(chat.call_count, 1)
        self.assertIn("gold_action", chat.call_args.args[0][0]["content"])
        self.assertEqual(state["edges"]["a=>c"]["guard_status"], "resolved")
        self.assertEqual(state["slot_policies"]["make-password"]["status"], "resolved")
        self.assertEqual(len(result["guards"]), 1)
        self.assertEqual(len(result["slot_policies"]), 1)

    @patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""})
    @patch("llm.chat")
    def test_autonomous_reflection_selects_valid_resources_itself(self, chat, _config):
        state = initialize_skill_dag(_subgraph(), "account_access")
        chat.side_effect = [
            '''{"lookups":[{"resource":"slot_policies","query":"make-password","top_k":1}]}''',
            '''{"decision":"update","updates":[
          {"resource":"transition_guard","edge_id":"a=>c","content":"The customer is creating a password.","status":"resolved","rationale":"gold mismatch"},
          {"resource":"slot_policy","action":"make-password","content":"Use the newly confirmed value only.","status":"resolved","rationale":"slot mismatch"},
          {"resource":"reference","content":"Keep rare recovery evidence in reference.","status":"uncertain","rationale":"limited support"}
        ]}'''
        ]
        result = autonomous_resource_reflection(state, [{"gold_action": "make-password", "predicted_action": "send-link"}], "skill", "reference", "rules", "slots", "test")
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(result["prompt_chars"], len(result["prompt"]))
        self.assertNotIn('"react_trace"', result["prompt"])
        self.assertEqual(result["lookups"][0]["resource"], "slot_policies")
        self.assertIn("<current_skill>skill</current_skill>", result["prompt"])
        self.assertEqual(result["model_decision"], "update")
        self.assertEqual(len(result["accepted"]), 3)
        self.assertEqual(state["edges"]["a=>c"]["guard_status"], "resolved")
        self.assertEqual(state["slot_policies"]["make-password"]["status"], "resolved")
        self.assertEqual(len(state["reference_notes"]), 1)

    def test_summary_names_branch_blockers(self):
        state = initialize_skill_dag(_subgraph(), "account_access")
        summary = summarize_refinement_state(state, RefinementPolicy())
        branch = next(row for row in summary["branches"] if row["edge_id"] == "a=>c")
        self.assertIn("insufficient_gold_support", branch["blockers"])
        self.assertIn("guard_unresolved", branch["blockers"])


if __name__ == "__main__":
    unittest.main()
