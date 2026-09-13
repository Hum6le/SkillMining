from __future__ import annotations

import json

from eval_tod.abcd.agent import _parse_reference_sections
from scripts.run_trace2skill_abcd import _run_direct_memory_update


def test_direct_memory_update_preserves_and_deduplicates_evidence(tmp_path):
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text("---\nname: test\n---\n\n# Test skill\n", encoding="utf-8")

    error_path = tmp_path / "error.json"
    success_path = tmp_path / "success.json"
    error_path.write_text(json.dumps([{
        "instance_id": "case-1",
        "items": [{
            "type": "failure_memory",
            "title": "Verify before account lookup",
            "description": "Identity has not been verified",
            "content": "Complete identity verification before pulling up the account.",
        }],
    }]), encoding="utf-8")
    success_path.write_text(json.dumps([{
        "instance_id": "case-2",
        "items": [{
            "type": "success_memory",
            "title": "Reuse established account name",
            "description": "The customer name is already established",
            "content": "Bind the established customer name as the only ordered value.",
        }],
    }]), encoding="utf-8")

    first = _run_direct_memory_update(error_path, success_path, skill_path)
    second = _run_direct_memory_update(error_path, success_path, skill_path)

    reference = (skill_dir / "references" / "direct_analysis_memory.md").read_text(encoding="utf-8")
    ledger = json.loads((skill_dir / "direct_analysis_memory.json").read_text(encoding="utf-8"))
    skill = skill_path.read_text(encoding="utf-8")

    assert "added 2 deduplicated item(s)" in first[0]
    assert "added 0 deduplicated item(s)" in second[0]
    assert len(ledger) == 2
    assert "Evidence type: `success_memory`" in reference
    assert "Evidence type: `failure_memory`" in reference
    assert skill.count("## Direct analysis memory") == 1
    assert len(_parse_reference_sections(reference)) == 2
