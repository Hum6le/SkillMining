from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_backbone_online_refine import _install_hybrid_evolver_resources
from scripts.audit_trace2skill_hybrid import audit_run
from skill_evolver.parallel_evolving_agent import ParallelSkillEvolver, Patch, PatchEdit
from skill_evolver.skill_evolving_agent import SkillEvolver


def test_hybrid_resources_are_in_trace2skill_skill_state(tmp_path: Path):
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    skill = _install_hybrid_evolver_resources(
        skill_dir,
        "---\nname: test\n---\n\n# Test skill\n",
        "graph reference",
        "action rule",
        "slot policy",
    )
    state = SkillEvolver.read_skill_state(SimpleNamespace(skill_dir=skill_dir))

    assert "references/tod_reference.md" in state
    assert "references/tod_action_rules.md" in state
    assert "references/tod_slot_policies.md" in state
    assert "references/tod_action_rules.md" in skill
    assert "references/tod_slot_policies.md" in skill


def test_failed_reduce_preserves_all_input_patches(tmp_path: Path):
    evolver = object.__new__(ParallelSkillEvolver)
    evolver.merge_batch_size = 5
    evolver.max_merge_levels = 1
    evolver.max_workers = 1
    evolver.output_dir = None
    patches = [
        Patch(
            reasoning=f"reason-{index}",
            edits=[PatchEdit(file="SKILL.md", op="append_to_section", content=f"edit-{index}")],
            changelog_entries=[f"change-{index}"],
            batch_index=index,
            raw_json={},
        )
        for index in range(5)
    ]

    with patch.object(evolver, "_run_single_merge", return_value=[]):
        result = evolver.run_reduce_phase({}, patches)

    assert result is not None
    assert [edit.content for edit in result.edits] == [f"edit-{index}" for index in range(5)]
    assert result.changelog_entries == [f"change-{index}" for index in range(5)]


def test_hybrid_audit_detects_prompt_wiring_and_forbidden_fields(tmp_path: Path):
    batch = tmp_path / "trace2skill_hybrid_batches" / "batch_0001"
    (batch / "error_analysis" / "abcd-1").mkdir(parents=True)
    (batch / "evolution" / "prompt_samples" / "map").mkdir(parents=True)
    (batch / "evolution" / "prompt_samples" / "merge_level_1").mkdir(parents=True)
    (tmp_path / "trace2skill_hybrid_skill" / "references").mkdir(parents=True)
    (batch / "trajectory_evidence.json").write_text(json.dumps([{
        "conversation_id": "1",
        "trajectory": [{
            "context": "prefix", "prediction": "current prediction",
            "react_trace": [{"messages": ["complete internal prompt"]}],
        }],
        "graph_context": {"nodes": [{"id": "n1"}]},
    }]), encoding="utf-8")
    logger_dir = tmp_path / "llm_responses"
    logger_dir.mkdir()
    (logger_dir / "0001_trace2skill_error_analysis_prompt.json").write_text(json.dumps({
        "call_tag": "trace2skill_error_analysis",
        "messages": [{"role": "user", "content": "conversation_id: 1 graph_context gold_action gold_slots"}],
    }), encoding="utf-8")
    (batch / "error_analysis_parsed.json").write_text(json.dumps([{
        "instance_id": "abcd-1", "ast_evidence": [{"gold_action": "act"}],
        "items": [{"title": "lesson"}],
    }]), encoding="utf-8")
    (batch / "evolution" / "prompt_samples" / "map" / "batch_0001.md").write_text(
        "references/tod_reference.md", encoding="utf-8"
    )
    (batch / "evolution" / "prompt_samples" / "merge_level_1" / "batch_0001.md").write_text(
        "merge sample", encoding="utf-8"
    )
    (tmp_path / "trace2skill_hybrid_skill" / "references" / "tod_reference.md").write_text(
        "reference body", encoding="utf-8"
    )

    report = audit_run(tmp_path)
    item = report["batches"][0]
    assert item["evidence"]["forbidden_trajectory_fields"] == ["react_trace"]
    assert item["analysis"]["saved_raw_prompt_calls_matched"] == 1
    assert item["analysis"]["calls"][0]["has_graph_context_marker"] is True
    assert item["analysis"]["calls"][0]["has_gold_slots"] is True
    assert item["analysis"]["parsed_analysis_artifacts"][0]["records_with_ast_evidence"] == 1
    assert item["map"]["resource_names_mentioned"]["tod_reference.md"] is True
    assert item["map"]["resource_audit"]["tod_reference.md"]["source_file_exists"] is True
    assert item["reduce"]["prompt_sample_count"] == 1
