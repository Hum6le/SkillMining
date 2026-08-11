from __future__ import annotations

import json
import unittest

from skill_disco.consolidation import SkillCluster
from skill_disco.operation_extraction import SemanticOperation
from skill_disco.skill_specification import build_skill_specification_prompt, specify_skill_contract


class SkillSpecificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.operation = SemanticOperation(
            operation_id="1:0-1:recover", conversation_id="1", name="recover", description="Recover a password.",
            action_start_index=0, action_end_index=1, action_turn_indices=[1, 3],
            action_sequence=["pull-up-account(username)", "make-password()"], preconditions=["reset_requested"],
            postconditions=["password_generated"], control_flow="fixed_sequence", parameters=["username"],
            supporting_event_turns=[0], completion_evidence="Password generated.", code_snippet="action = 'make-password'",
            grounded_actions=[{
                "action_index": 1,
                "action": "pull-up-account",
                "slot_values": ["ada"],
                "pre_action_evidence": ["My username is ada."],
            }],
        )
        self.cluster = SkillCluster(
            cluster_id="cluster_000", name="password_recovery", description="Recover passwords.", group_ids=["g0"],
            operation_ids=[self.operation.operation_id], supporting_conversations=["1", "2"],
            reusability_score=0.5, representative_action_sequence=self.operation.action_sequence,
        )

    def test_contract_prompt_and_local_metadata(self) -> None:
        prompt = build_skill_specification_prompt(self.cluster, [self.operation])
        self.assertNotIn("recover_password", prompt)
        self.assertIn("pull-up-account(username)", prompt)
        self.assertIn('"observed_action_slots"', prompt)
        self.assertIn('"ada"', prompt)
        self.assertIn("My username is ada.", prompt)
        self.assertIn("semantic role", prompt)
        response = json.dumps({
            "skill_name": "recover_account_password", "description": "Recover a password.",
            "docstring": "Recover a customer password from an account.",
            "parameters": [{"name": "username", "type": "str", "description": "Account username.", "required": True, "default": None}],
            "return_type": "wrong_value", "preconditions": ["reset_requested"], "postconditions": ["password_generated"],
            "side_effects": ["account accessed", "password generated"],
            "canonical_action_sequence": ["pull-up-account(username)", "make-password()"], "abstraction_level": "composite",
        })
        contract, _ = specify_skill_contract(self.cluster, [self.operation], lambda *_args, **_kwargs: response)
        self.assertEqual(contract.return_type, "SkillResult")
        self.assertEqual(contract.confidence_score, 0.5)
        self.assertEqual(contract.estimated_actions_saved, 1)
        self.assertEqual(contract.parameters[0].name, "username")


if __name__ == "__main__":
    unittest.main()
