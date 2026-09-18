from skill_mining.online_refinement import _classify_online_evidence


def test_online_diagnosis_exposes_case_and_permutation():
    item = {
        "action_success": True,
        "slot_success": False,
        "gold_slots": ["david williams", "(322) 976-2201", "1592"],
        "predicted_slots": ["David Williams", "1592", "(322) 976-2201"],
    }

    errors = _classify_online_evidence(item)

    assert "case_plus_slot_order" in errors
    assert item["slot_match_profile"]["casefold_unordered"] is True
    assert item["slot_error_bucket"] == "case_or_whitespace_plus_permutation"


def test_online_diagnosis_does_not_call_a_true_value_error_permutation():
    item = {
        "action_success": True,
        "slot_success": False,
        "gold_slots": ["a", "b", "c"],
        "predicted_slots": ["a", "b", "d"],
    }

    errors = _classify_online_evidence(item)

    assert "wrong_slot_value" in errors
    assert "wrong_slot_order" not in errors
