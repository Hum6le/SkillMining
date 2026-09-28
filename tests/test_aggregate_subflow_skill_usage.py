from scripts.aggregate_subflow_results import _weighted_average


def test_skill_usage_is_aggregated_over_action_turns() -> None:
    rows = [
        {"method": "skill_disco", "phase": "final", "subflow": "a",
         "text_samples": 4, "action_turns": 4, "test_sessions": 2,
         "metrics": {"ast_joint": 0.5},
         "skill_usage": {"action_target_turns": 4, "skill_invoked_turns": 3}},
        {"method": "skill_disco", "phase": "final", "subflow": "b",
         "text_samples": 6, "action_turns": 6, "test_sessions": 3,
         "metrics": {"ast_joint": 0.5},
         "skill_usage": {"action_target_turns": 6, "skill_invoked_turns": 2}},
    ]
    usage = _weighted_average(rows)["skill_disco:final"]["skill_usage"]
    assert usage["skill_invoked_turns"] == 5
    assert usage["base_fallback_turns"] == 5
    assert usage["skill_invocation_rate"] == 0.5
