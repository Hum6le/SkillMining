from __future__ import annotations

import unittest

from asi_offline import build_induction_episode, validate_asi_response


def _episode():
    conversation = {
        "convo_id": "validate-1",
        "scenario": {"personal": {"username": "alice"}},
        "original": [
            ["customer", "I forgot my password. My username is alice."],
            ["action", "Account alice found."],
            ["action", "Details for alice entered."],
            ["action", "Password generated."],
        ],
        "delexed": [
            {"speaker": "customer", "text": "I forgot my password.", "targets": ["hidden", None, None, [], -1]},
            {"speaker": "action", "text": "Account found.", "targets": ["hidden", "take_action", "pull-up-account", ["alice"], -1]},
            {"speaker": "action", "text": "Details entered.", "targets": ["hidden", "take_action", "enter-details", ["alice"], -1]},
            {"speaker": "action", "text": "Password generated.", "targets": ["hidden", "take_action", "make-password", [], -1]},
        ],
    }
    return build_induction_episode(conversation)


class OfflineASIValidationTest(unittest.TestCase):
    def test_accepts_parameterized_function_and_exact_rewrite(self) -> None:
        raw = '''```python
def recover_password(username):
    """Recover a password.

    Args:
        username: Account username.
    Returns:
        Completion status.
    Examples:
        recover_password("alice")
    """
    take_action("pull-up-account", [username])
    take_action("enter-details", [username])
    take_action("make-password", [])
```

## Rewritten Trajectory
```python
recover_password("alice")
```'''
        result = validate_asi_response(raw, _episode())
        self.assertTrue(result.rewritten_trajectory_valid, result.rewritten_trajectory_errors)
        self.assertEqual([item.name for item in result.accepted_functions], ["recover_password"])
        self.assertEqual([item["action"] for item in result.expanded_actions], ["pull-up-account", "enter-details", "make-password"])

    def test_rejects_hard_coded_training_value_in_function_body(self) -> None:
        raw = '''```python
def recover_password(username):
    take_action("pull-up-account", ["alice"])
    take_action("enter-details", [username])
    take_action("make-password", [])
```

## Rewritten Trajectory
```python
recover_password("alice")
```'''
        result = validate_asi_response(raw, _episode())
        self.assertEqual(result.accepted_functions, [])
        self.assertIn("function_body_hardcodes_or_derives_a_slot_value", [item["reason"] for item in result.rejected_functions])
        self.assertFalse(result.rewritten_trajectory_valid)


if __name__ == "__main__":
    unittest.main()
