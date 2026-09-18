from eval_tod.abcd.metrics import slot_error_bucket, slot_match_profile


def test_slot_profile_separates_permutation_and_case():
    gold = ["david williams", "(322) 976-2201", "1592"]
    predicted = ["David Williams", "1592", "(322) 976-2201"]

    profile = slot_match_profile(gold, predicted)

    assert not profile["strict_ordered"]
    assert not profile["exact_unordered"]
    assert not profile["casefold_ordered"]
    assert profile["casefold_unordered"]
    assert slot_error_bucket(gold, predicted) == "case_or_whitespace_plus_permutation"


def test_slot_profile_keeps_duplicates_in_unordered_comparison():
    assert not slot_match_profile(["a", "a", "b"], ["a", "b", "b"])["exact_unordered"]


def test_slot_profile_reports_punctuation_separately():
    gold = ["(322) 976-2201"]
    predicted = ["3229762201"]

    profile = slot_match_profile(gold, predicted)

    assert not profile["casefold_ordered"]
    assert profile["punctuation_insensitive_ordered"]
    assert slot_error_bucket(gold, predicted) == "punctuation_or_formatting_only"
