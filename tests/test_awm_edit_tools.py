import json
from copy import deepcopy
from unittest.mock import patch
import pytest
from awm import WorkflowStore
from eval_tod.abcd.agent import ABCDAgent
from skill_mining.awm_edit_tools import apply_awm_edit_plan


def files():
    return {"skill.md": "# Skill\n## Route\nOld route.\n## Other\nKeep.\n",
            "action_rules.md": "# Rules\n#### `send-link`\nOld rule.\n",
            "slot_policies.md": "# Slots\n#### `send-link`\nOld binding.\n"}


def operation(**changes):
    return {"file": "skill.md", "section": "## Route", "op": "replace",
            "match_text": "Old route.", "new_text": "New route.", "rationale": "Observed failure", **changes}


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
    assert "Use verified email." in agent.last_awm_edited_resources["slot_policies.md"]
    assert len(agent.last_awm_edit_operations) == 2
