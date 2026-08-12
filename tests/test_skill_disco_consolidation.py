from __future__ import annotations

import json
import unittest

from skill_disco.consolidation import (
    consolidate_groups,
    group_operation_batch,
    parse_grouping_output,
)
from skill_disco.operation_extraction import SemanticOperation


def _operation(conversation_id: str, name: str) -> SemanticOperation:
    return SemanticOperation(
        operation_id=f"{conversation_id}:0-1:{name}", conversation_id=conversation_id,
        name=name, description="Reset a password.", action_start_index=0, action_end_index=1,
        action_turn_indices=[2, 4], action_sequence=["pull-up-account(username)", "make-password()"],
        preconditions=["reset_requested"], postconditions=["password_generated"],
        control_flow="fixed_sequence", parameters=["username"], supporting_event_turns=[0],
        completion_evidence="Password generated.", code_snippet="action = 'make-password'",
    )


class ConsolidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.operations = [_operation("1", "recover_password"), _operation("2", "recover_password")]

    def test_two_pass_partition_and_reusability_score(self) -> None:
        groups, _ = group_operation_batch(
            self.operations,
            lambda *_args, **_kwargs: json.dumps({"groups": [{"name": "password_recovery", "description": "Reset password.", "operation_ids": [operation.operation_id for operation in self.operations]}]}),
            batch_index=0,
        )
        clusters, _ = consolidate_groups(
            groups, {operation.operation_id: operation for operation in self.operations}, 2,
            lambda *_args, **_kwargs: json.dumps({"clusters": [{"name": "password_recovery", "description": "Recover account password.", "group_ids": [groups[0].group_id]}]}),
        )
        self.assertEqual(clusters[0].supporting_conversations, ["1", "2"])
        self.assertEqual(clusters[0].reusability_score, 1.0)

    def test_repairs_non_partitioning_group_response_at_runtime(self) -> None:
        incomplete = json.dumps({"groups": [{"name": "only_one", "description": "Incomplete.", "operation_ids": [self.operations[0].operation_id]}]})
        with self.assertRaisesRegex(ValueError, "partition"):
            parse_grouping_output(incomplete, self.operations, batch_index=0)
        with self.assertWarnsRegex(RuntimeWarning, "Recovered an invalid"):
            groups, _ = group_operation_batch(
                self.operations,
                lambda *_args, **_kwargs: incomplete,
                batch_index=0,
            )
        self.assertEqual(
            [operation_id for group in groups for operation_id in group.operation_ids],
            [operation.operation_id for operation in self.operations],
        )
        self.assertEqual(groups[-1].name, "unassigned_operation_000")

    def test_recovers_unknown_ids_and_unparseable_grouping_json(self) -> None:
        unknown_ids = json.dumps({"groups": [{
            "name": "password_recovery", "description": "Contains an invalid ID.",
            "operation_ids": [self.operations[0].operation_id, "unknown-operation"],
        }]})
        with self.assertWarnsRegex(RuntimeWarning, "unknown ID"):
            groups, _ = group_operation_batch(
                self.operations, lambda *_args, **_kwargs: unknown_ids, batch_index=0
            )
        self.assertEqual(
            {operation_id for group in groups for operation_id in group.operation_ids},
            {operation.operation_id for operation in self.operations},
        )

        with self.assertWarnsRegex(RuntimeWarning, "unusable"):
            groups, _ = group_operation_batch(
                self.operations, lambda *_args, **_kwargs: '{"groups": [', batch_index=0
            )
        self.assertEqual(len(groups), len(self.operations))
        self.assertTrue(all(len(group.operation_ids) == 1 for group in groups))

    def test_recovers_unparseable_consolidation_json(self) -> None:
        groups, _ = group_operation_batch(
            self.operations,
            lambda *_args, **_kwargs: json.dumps({"groups": [{
                "name": "password_recovery", "description": "Reset password.",
                "operation_ids": [operation.operation_id for operation in self.operations],
            }]}),
            batch_index=0,
        )
        with self.assertWarnsRegex(RuntimeWarning, "unusable"):
            clusters, _ = consolidate_groups(
                groups, {operation.operation_id: operation for operation in self.operations}, 2,
                lambda *_args, **_kwargs: '{"clusters": [',
            )
        self.assertEqual(len(clusters), len(groups))
        self.assertEqual(clusters[0].operation_ids, groups[0].operation_ids)


if __name__ == "__main__":
    unittest.main()
