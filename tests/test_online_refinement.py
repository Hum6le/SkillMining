import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from skill_mining.online_refinement import (
    RefinementPolicy,
    apply_refinement_patches,
    apply_dynamic_skill_operations,
    apply_working_skill_operations,
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
    def test_dynamic_skill_operations_do_not_require_compiler_anchors(self):
        skill = "# Skill\n\n## Workflow\n- Route A.\n\n## Reference\n- Retrieve details.\n"
        updated, operations = apply_dynamic_skill_operations(skill, [
            {
                "op": "upsert", "match_text": "- Route A.",
                "new_text": "- Route B when the customer rejects the first option.",
                "rationale": "Clarify the existing route using rollout evidence.",
            },
            {
                "op": "upsert", "match_text": "## Reference",
                "new_text": "## Reference\n- Use transition evidence for uncertain alternatives.",
                "rationale": "Add compatible retrieval guidance without changing the workflow.",
            },
            {"op": "delete", "match_text": "- Retrieve details.\n", "new_text": "",
             "rationale": "The old retrieval line is duplicated by the new guidance."},
        ])
        self.assertFalse(any("error" in operation for operation in operations))
        self.assertIn("Route B when", updated)
        self.assertIn("transition evidence", updated)
        self.assertNotIn("Retrieve details", updated)

    def test_dynamic_skill_operations_require_disambiguation_for_repeated_text(self):
        skill = "# Skill\n\n- Ask for details.\n- Ask for details.\n"
        unchanged, rejected = apply_dynamic_skill_operations(skill, [{
            "op": "upsert", "match_text": "- Ask for details.",
            "new_text": "- Ask for confirmed account details.",
            "rationale": "Clarify a local rule.",
        }])
        self.assertEqual(unchanged, skill)
        self.assertIn("occurs multiple times", rejected[0]["error"])

        updated, applied = apply_dynamic_skill_operations(skill, [{
            "op": "upsert", "match_text": "- Ask for details.",
            "occurrence": 2,
            "new_text": "- Ask for confirmed account details.",
            "rationale": "Clarify the second local rule.",
        }])
        self.assertTrue(applied[0]["applied"])
        self.assertEqual(updated.count("- Ask for details."), 1)
        self.assertIn("- Ask for confirmed account details.", updated)

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

    def test_slot_evidence_covers_first_and_single_action_turns(self):
        state = initialize_skill_dag(_subgraph(), "account_access")
        conversations = [
            {
                "convo_id": "single",
                "delexed": [{"targets": ["", "take_action", "send-link", ["gold"]]}],
            },
            {
                "convo_id": "first",
                "delexed": [
                    {"targets": ["", "take_action", "enter-details", []]},
                    {"targets": ["", "take_action", "send-link", ["gold"]]},
                ],
            },
        ]
        rows = [
            {"convo_id": "single", "turn_index": 0, "predicted_action": "send-link", "predicted_slots": ["wrong"], "context": "", "react_trace": []},
            {"convo_id": "first", "turn_index": 0, "predicted_action": "enter-details", "predicted_slots": [], "context": "", "react_trace": []},
            {"convo_id": "first", "turn_index": 1, "predicted_action": "send-link", "predicted_slots": ["gold"], "context": "", "react_trace": []},
        ]

        localized = localize_rollout_batch(conversations, rows, state)

        self.assertEqual(localized["num_slot_events"], 3)
        self.assertEqual(state["slot_policies"]["send-link"]["slot_total"], 2)
        self.assertEqual(state["slot_policies"]["send-link"]["slot_success"], 1)
        self.assertEqual(state["slot_policies"]["send-link"]["slot_failures"], 1)
        self.assertEqual(state["slot_policies"]["enter-details"]["slot_total"], 1)

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

    def test_working_skill_operations_edit_anchor_sections_in_place(self):
        state = initialize_skill_dag(_subgraph(), "account_access")
        skill = '''# Skill
<!-- ACTION_RULES_START -->
#### `enter-details`
Old action rule.
<!-- ACTION_RULES_END -->
<!-- TRANSITION_RULES_START -->
<!-- TRANSITION_RULES_END -->
'''
        updated, operations = apply_working_skill_operations(skill, state, [
            {"resource": "action_rule", "op": "upsert", "action": "enter-details", "content": "New action rule."},
            {"resource": "transition_guard", "op": "upsert", "edge_id": "a=>c", "content": "The customer is creating a password."},
        ])
        self.assertIn("New action rule.", updated)
        self.assertNotIn("Old action rule.", updated)
        self.assertIn("EDGE_RULE:a=>c", updated)
        self.assertIn("creating a password", updated)
        self.assertFalse(any("error" in operation for operation in operations))

    def test_working_skill_operations_support_current_routing_compiler_format(self):
        state = initialize_skill_dag(_subgraph(), "account_access")
        skill = '''# Skill
## Workflow
### Routing Policies
<!-- ROUTING_SECTION_START -->
<!-- ROUTE_SOURCE:a -->
#### Routing after `enter-details`
- Continue along the observed workflow.
<!-- ROUTING_SECTION_END -->

### Action Rules
#### `enter-details`
- Gather required information.

## Slot Discipline
- Use dialogue values only.
'''
        updated, operations = apply_working_skill_operations(skill, state, [
            {"resource": "transition_guard", "op": "upsert", "edge_id": "a=>c",
             "content": "the customer asks to create or reset a password"},
            {"resource": "action_rule", "op": "upsert", "action": "enter-details",
             "content": "Collect only details supplied in the dialogue."},
        ])
        self.assertFalse(any("error" in operation for operation in operations))
        self.assertIn("<!-- ROUTE_EDGE:a=>c -->", updated)
        self.assertIn("create or reset a password", updated)
        self.assertIn("<!-- ACTION_RULES_START -->", updated)
        self.assertIn("Collect only details supplied in the dialogue.", updated)

    @patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""})
    @patch("llm.chat")
    def test_threshold_diagnostics_do_not_revert_autonomous_skill_edit(self, chat, _config):
        state = initialize_skill_dag(_subgraph(), "account_access")
        chat.side_effect = [
            '''{"lookups":[]}''',
            '''{"decision":"update","updates":[{"resource":"transition_guard","edge_id":"a=>c","op":"upsert","content":"The customer is creating a password.","status":"resolved","rationale":"gold mismatch"}],"skill_operations":[{"op":"upsert","edge_id":"a=>c","match_text":"<!-- TRANSITION_RULES_START -->","new_text":"<!-- TRANSITION_RULES_START -->\\n- Password route.","rationale":"integrate guard"}]}''',
        ]
        reflection = autonomous_resource_reflection(state, [{"gold": {"gold_action": "make-password"}}], "# Skill\n<!-- ACTION_RULES_START -->\n<!-- ACTION_RULES_END -->\n<!-- TRANSITION_RULES_START -->\n<!-- TRANSITION_RULES_END -->", "", "", "", "test")
        skill, operations = apply_working_skill_operations(reflection["prompt"].split("<current_skill>", 1)[1].split("</current_skill>", 1)[0], state, reflection["accepted"])
        self.assertEqual(state["edges"]["a=>c"]["visibility"], "skill")
        diagnostics = propose_refinement_patches(state, RefinementPolicy())
        self.assertTrue(any(item["operation"] == "sink_to_reference" for item in diagnostics))
        self.assertEqual(state["edges"]["a=>c"]["visibility"], "skill")
        self.assertIn("EDGE_RULE:a=>c", skill)
        self.assertFalse(any("error" in item for item in operations))

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
          {"resource":"reference","edge_id":"a=>c","content":"Keep rare recovery evidence in reference.","status":"uncertain","rationale":"limited support"}
        ],"skill_operations":[{"op":"upsert","edge_id":"a=>c","match_text":"skill","new_text":"updated skill","rationale":"integrate guard"}]}'''
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
        self.assertEqual(state["edges"]["a=>c"]["visibility"], "skill")
        online_skill, _ = render_online_resources(state)
        self.assertIn("enter-details", online_skill)
        self.assertIn("make-password", online_skill)
        self.assertEqual(state["slot_policies"]["make-password"]["status"], "resolved")
        self.assertEqual(len(state["reference_notes"]), 1)
        self.assertEqual(state["reference_notes"][0]["edge_id"], "a=>c")

    def test_summary_names_branch_blockers(self):
        state = initialize_skill_dag(_subgraph(), "account_access")
        summary = summarize_refinement_state(state, RefinementPolicy())
        branch = next(row for row in summary["branches"] if row["edge_id"] == "a=>c")
        self.assertIn("insufficient_gold_support", branch["blockers"])
        self.assertIn("guard_unresolved", branch["blockers"])


if __name__ == "__main__":
    unittest.main()
