import json
from copy import deepcopy
from unittest.mock import patch
import pytest
from awm import WorkflowStore
from eval_tod.abcd.agent import ABCDAgent
from skill_mining.awm_edit_tools import apply_awm_edit_plan, AWMEditError, build_edit_catalog, render_edit_catalog
from skill_mining.awm_edit_tools import render_id_instructions


def files():
    return {"skill.md": "# Skill\n## Route\nOld route.\n## Other\nKeep.\n",
            "action_rules.md": "# Rules\n#### `send-link`\nOld rule.\n",
            "slot_policies.md": "# Slots\n#### `send-link`\nOld binding.\n"}


def test_display_brackets_accept_exact_current_ids_and_ranges_only():
    source = files()
    catalog = build_edit_catalog(source)
    ids = [key for key, row in catalog.items() if row["file"] == "skill.md"]
    ops = [{"file": "skill.md", "target_id": f"[{ids[1]}]",
            "end_target_id": f"[{ids[2]}]", "op": "replace", "new_text": "## Fixed\nNew route."}]
    edited, logs = apply_awm_edit_plan(json.dumps({"operations": ops}), source, {"send-link"})
    assert "## Fixed\nNew route." in edited["skill.md"]
    assert logs[0]["target_id"] == f"[{ids[1]}]"  # Original response retained for audit.
    for target in (f"[[{ids[1]}]]", "[0000000000000000:L2]", f"`{ids[1]}`"):
        with pytest.raises(ValueError):
            apply_awm_edit_plan(json.dumps({"operations": [{**ops[0], "target_id": target}]}), source, {"send-link"})
    with pytest.raises(ValueError):
        apply_awm_edit_plan(json.dumps({"operations": [{**ops[0], "file": "slot_policies.md"}]}), source, {"send-link"})
    with pytest.raises(ValueError, match="overlapping"):
        apply_awm_edit_plan(json.dumps({"operations": ops * 2}), source, {"send-link"})


def operation(**changes):
    return {"file": "skill.md", "section": "## Route", "op": "replace",
            "match_text": "Old route.", "new_text": "New route.", "rationale": "Observed failure", **changes}


def test_id_examples_are_current_and_stale_end_id_gets_explicit_feedback():
    source = files()
    catalog = build_edit_catalog(source)
    prompt = render_id_instructions(source)
    assert "COPY_ACTUAL_ID" not in prompt and "5d9cc6170bb3ef49" not in prompt
    import re
    assert all(key in catalog for key in re.findall(r"[0-9a-f]{16}:L[1-9][0-9]*", prompt))
    target = next(k for k, row in catalog.items() if row["text"] == "Old route.")
    plan = {"operations": [{"file": "skill.md", "target_id": target,
                            "end_target_id": "0000000000000000:L3", "op": "delete"}]}
    with pytest.raises(AWMEditError, match="end_target_id=.*unknown_revision") as caught:
        apply_awm_edit_plan(json.dumps(plan), source, {"send-link"})
    checks = caught.value.repair_context["address_checks"]
    assert checks[0]["reason"] == "valid" and checks[1]["reason"] == "unknown_revision"
    assert checks[1]["same_line_number_candidates_not_auto_applied"] == [target]
    assert source == files()


@pytest.mark.parametrize("addressing", ["id", "legacy"])
def test_delete_may_omit_new_text_without_mutating_plan(addressing):
    source = files()
    if addressing == "id":
        target = next(k for k, row in build_edit_catalog(source).items() if row["text"] == "Old route.")
        op = {"file": "skill.md", "target_id": target, "op": "delete"}
    else:
        op = {"file": "skill.md", "match_text": "Old route.", "op": "delete"}
    original = deepcopy(source)
    edited, logs = apply_awm_edit_plan(json.dumps({"operations": [op]}), source, {"send-link"})
    assert "Old route." not in edited["skill.md"] and "Keep." in edited["skill.md"]
    assert "new_text" not in op and source == original
    assert "delete_new_text_defaulted_empty" in logs[0]["warnings"]
    for bad in (None, 123, "not empty"):
        with pytest.raises(ValueError):
            apply_awm_edit_plan(json.dumps({"operations": [{**op, "new_text": bad}]}), source, {"send-link"})
    for other in ("replace", "insert_before", "insert_after"):
        with pytest.raises(ValueError, match="new_text"):
            apply_awm_edit_plan(json.dumps({"operations": [{**op, "op": other}]}), source, {"send-link"})


def test_id_validation_error_uses_original_response_index_after_reverse_sort():
    source = files()
    catalog = build_edit_catalog(source)
    targets = {row["text"]: key for key, row in catalog.items()}
    ops = [{"file": "skill.md", "target_id": targets["Old route."], "op": "replace", "new_text": "New."},
           {"file": "skill.md", "target_id": targets["Keep."], "op": "replace"}]
    with pytest.raises(AWMEditError, match=r"Edit 2 \(skill.md\): missing/non-text new_text") as caught:
        apply_awm_edit_plan(json.dumps({"operations": ops}), source, {"send-link"})
    assert caught.value.repair_context["edit_index"] == 2
    assert caught.value.repair_context["execution_index"] == 1
    assert source == files()


def test_duplicate_card_rejection_provides_original_heading_ids():
    source = files()
    catalog = build_edit_catalog(source)
    target = next(k for k, row in catalog.items() if row["text"] == "Old rule.")
    heading = next(k for k, row in catalog.items() if row["file"] == "action_rules.md" and row["text"] == "#### `send-link`")
    plan = {"operations": [{"file": "action_rules.md", "target_id": target, "op": "replace",
                            "new_text": "#### `send-link`\nNew rule."}]}
    with pytest.raises(AWMEditError, match="duplicate action cards") as caught:
        apply_awm_edit_plan(json.dumps(plan), source, {"send-link"})
    assert caught.value.repair_context["existing_card_heading_ids"] == [heading]
    assert caught.value.repair_context["before_heading_count"] == 1
    assert caught.value.repair_context["after_heading_count"] == 2
    assert source == files()


def test_catalog_ids_disambiguate_duplicates_and_do_not_shift_within_plan():
    source = files()
    source["skill.md"] = "# Skill\n## Route\nSame.\nMiddle.\nSame.\n"
    catalog = build_edit_catalog(source)
    same = [key for key, row in catalog.items() if row["text"] == "Same."]
    assert len(same) == 2 and same[0] != same[1]
    ops = [{"file": "skill.md", "target_id": same[0], "op": "insert_before", "new_text": "New first line.\n"},
           {"file": "skill.md", "target_id": same[1], "op": "replace", "new_text": "Changed second copy."}]
    edited, logs = apply_awm_edit_plan(json.dumps({"operations": ops}), source, {"send-link"})
    assert edited["skill.md"] == "# Skill\n## Route\nNew first line.\nSame.\nMiddle.\nChanged second copy.\n"
    assert [log["target_id"] for log in logs] == same
    assert render_edit_catalog(source).count("Same.") == 2
    # Stale/wrong-file IDs and conflicting ranges cannot silently redirect.
    for bad in ([{**ops[0], "file": "slot_policies.md"}], [ops[0], ops[0]],
                [{**ops[0], "target_id": "bad-id"}]):
        with pytest.raises(ValueError):
            apply_awm_edit_plan(json.dumps({"operations": bad}), source, {"send-link"})
    changed = {**source, "skill.md": source["skill.md"] + "Changed snapshot.\n"}
    with pytest.raises(ValueError, match="stale"):
        apply_awm_edit_plan(json.dumps({"operations": ops}), changed, {"send-link"})


@pytest.mark.parametrize("display_brackets", [False, True])
def test_native_id_induction_uses_the_same_addressed_prior_once(display_brackets):
    source = files()
    workflow = WorkflowStore()
    workflow.replace(source["skill.md"])
    calls = []
    def chat(self, prompt, **kwargs):
        calls.append(prompt)
        target = next(k for k, row in build_edit_catalog({**source, "skill.md": workflow.text}).items()
                      if row["text"] == "Old route.")
        if display_brackets:
            target = f"[{target}]"
        return json.dumps({"operations": [{"file": "skill.md", "target_id": target,
                                           "op": "replace", "new_text": "Updated route."}]})
    with patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""}), \
         patch.object(ABCDAgent, "_chat", chat):
        agent = ABCDAgent(model="test", workflow=workflow)
        updated = agent.induce([], [], [], editable_action_cards=True, edit_mode="tools", editable_resources=source)
    assert "Updated route." in updated and len(calls) == 1
    assert calls[0].count("Old route.") == 1
    assert "Current Editable Snapshot" in calls[0]


def test_atomic_shared_edits_across_files():
    source = files()
    original = deepcopy(source)
    ops = [operation(), operation(file="slot_policies.md", section="#### `send-link`",
        match_text="Old binding.", new_text="New binding.")]
    edited, logs = apply_awm_edit_plan(json.dumps({"operations": ops}), source, {"send-link"})
    assert "New route." in edited["skill.md"] and "New binding." in edited["slot_policies.md"]
    assert edited["action_rules.md"] == source["action_rules.md"]
    assert source == original and len(logs) == 2
    assert logs[1]["file"] == "slot_policies.md"


@pytest.mark.parametrize("bad", [
    {"file": "reference.md"}, {"file": "/etc/passwd"}, {"new_text": ""},
    {"match_text": "not present"}, {"op": "append"}, {"occurrence": True},
    {"op": "delete", "new_text": "invalid"}, {"match_text": ""},
])
def test_failed_plan_never_changes_input(bad):
    source = files()
    with pytest.raises(ValueError):
        apply_awm_edit_plan(json.dumps({"operations": [operation(), operation(**bad)]}), source, {"send-link"})
    assert source == files()


@pytest.mark.parametrize("metadata", [{}, {"rationale": None}, {"section": "Route"},
    {"section": "## Other"}, {"section": "not an exact heading", "rationale": ""}])
def test_metadata_does_not_block_exact_text_edits(metadata):
    op = {k: v for k, v in operation().items() if k not in {"section", "rationale"}}
    op.update(metadata)
    edited, logs = apply_awm_edit_plan(json.dumps({"operations": [op]}), files(), {"send-link"})
    assert "New route." in edited["skill.md"]
    assert logs[0]["section"] == "## Route"
    assert logs[0]["section_path"] == ["# Skill", "## Route"]
    assert "rationale_missing" in logs[0]["warnings"]
    if metadata.get("section") == "## Other":
        assert "section_hint_mismatch_exact_anchor_used" in logs[0]["warnings"]


def test_duplicate_headings_are_safe_with_unique_anchor_but_not_ambiguous_anchor():
    source = files()
    source["skill.md"] = "# Skill\n## Route\nOld route.\n## Route\nKeep.\n"
    edited, _ = apply_awm_edit_plan(json.dumps({"operations": [operation()]}), source, {"send-link"})
    assert "New route." in edited["skill.md"] and "Keep." in edited["skill.md"]
    source["skill.md"] += "Old route.\n"
    with pytest.raises(ValueError, match="ambiguous"):
        apply_awm_edit_plan(json.dumps({"operations": [operation()]}), source, {"send-link"})
    edited, _ = apply_awm_edit_plan(json.dumps({"operations": [operation(occurrence=2)]}), source, {"send-link"})
    assert edited["skill.md"].count("Old route.") == 1


def test_five_edit_case_keeps_missing_rationale_and_repeated_section_safe():
    source = files()
    source["skill.md"] = "# Skill\n## Route\n" + "\n".join(f"Rule {i}." for i in range(4)) + "\n"
    ops = [operation(section="Route", match_text=f"Rule {i}.", new_text=f"Fixed {i}.") for i in range(4)]
    last = operation(file="slot_policies.md", section="send-link", match_text="Old binding.", new_text="New binding.")
    last.pop("rationale")
    edited, logs = apply_awm_edit_plan(json.dumps({"operations": [*ops, last]}), source, {"send-link"})
    assert len(logs) == 5 and "New binding." in edited["slot_policies.md"]
    assert all(f"Fixed {i}." in edited["skill.md"] for i in range(4))


def test_missing_anchor_feedback_supplies_real_text_without_fuzzy_applying():
    source = files()
    with pytest.raises(AWMEditError) as caught:
        apply_awm_edit_plan(json.dumps({"operations": [operation(match_text="Old routing.")]}), source, {"send-link"})
    context = caught.value.repair_context
    assert context["exact_match_count"] == 0 and "0 exact matches" in str(caught.value)
    assert context["file"] == "skill.md" and context["edit_index"] == 1
    assert any(c["match_text"] == "Old route." for c in context["literal_candidates"])
    assert all(c["match_text"] in source["skill.md"] for c in context["literal_candidates"])
    assert source == files()


def test_ambiguous_anchor_feedback_exposes_occurrences_and_preview_only_prefix():
    source = files()
    source["skill.md"] += "Old route.\n"
    with pytest.raises(AWMEditError) as caught:
        apply_awm_edit_plan(json.dumps({"operations": [operation()]}), source, {"send-link"})
    assert caught.value.repair_context["exact_match_count"] == 2
    assert [c["occurrence"] for c in caught.value.repair_context["literal_candidates"]] == [1, 2]
    original = files()
    with pytest.raises(AWMEditError) as caught:
        apply_awm_edit_plan(json.dumps({"operations": [operation(), operation(match_text="absent")]}), original, {"send-link"})
    assert caught.value.repair_context["validated_prefix_edits"] == 1
    assert original == files()


def test_native_induction_retries_exact_anchor_error_without_partial_state():
    source = files()
    workflow = WorkflowStore()
    workflow.replace(source["skill.md"])
    bad = json.dumps({"operations": [operation(match_text="absent")]})
    good = json.dumps({"operations": [operation(), operation(file="slot_policies.md",
        section="#### `send-link`", match_text="Old binding.", new_text="Use verified email.")]})
    with patch("llm.resolve_config", return_value={"model": "test", "api_key": "", "base_url": ""}), \
         patch.object(ABCDAgent, "_chat", side_effect=[bad, good]) as chat, \
         patch("eval_tod.abcd.agent.time.sleep"):
        agent = ABCDAgent(model="test", workflow=workflow)
        result = agent.induce([], [], [], editable_action_cards=True, edit_mode="tools", editable_resources=source)
    assert chat.call_count == 2 and "New route." in result
    assert "apply_text_edits" in chat.call_args_list[0].args[0]
    assert "anchor missing" in chat.call_args_list[1].args[0]
    assert "<rejected_edit_plan>" in chat.call_args_list[1].args[0]
    assert bad in chat.call_args_list[1].args[0]
    assert '"exact_match_count": 0' in chat.call_args_list[1].args[0]
    assert "literal_candidates" in chat.call_args_list[1].args[0]
    assert "Use verified email." in agent.last_awm_edited_resources["slot_policies.md"]
    assert len(agent.last_awm_edit_operations) == 2
