import json
from pathlib import Path

import pytest

from scripts.analyze_offline_online_contribution import (
    AuditFileTools,
    _format_model_report,
    _json_safe,
    analyze,
    render_markdown,
)


def _make_run(root: Path, *, action: str, slots: list[str], card: list[str]) -> None:
    root.mkdir()
    (root / "online_refine_result.json").write_text(json.dumps({
        "ast_cds": {"ast_joint": 0.5, "ast_action_name": 0.8, "ast_slot_value": 0.6},
    }), encoding="utf-8")
    (root / "online_refined_predictions.json").write_text(json.dumps([{
        "convo_id": "c1", "turn_index": 2, "target_type": "action",
        "predicted_action": action, "predicted_slots": slots,
        "action_card_lookup": {"executed": True, "selected_actions": card},
    }]), encoding="utf-8")
    (root / "action_rules.md").write_text("rule", encoding="utf-8")
    (root / "slot_policies.md").write_text("policy", encoding="utf-8")
    (root / "autonomous_reflection").mkdir()
    (root / "autonomous_reflection" / "batch_0001.json").write_text(json.dumps({
        "model_decision": "update", "accepted": [{"resource": "action_rules"}],
        "rejected": [], "skill_operations": [],
    }), encoding="utf-8")


def test_audit_reports_resource_update_card_and_paired_prediction_changes(tmp_path: Path):
    primary = tmp_path / "primary"
    comparison = tmp_path / "comparison"
    _make_run(primary, action="send-link", slots=["email"], card=["send-link"])
    _make_run(comparison, action="send-link", slots=["phone"], card=["send-link"])

    report = analyze(primary, comparison_dir=comparison)

    assert report["resources"]["action_rules.md"]["nonempty"]
    assert report["online_updates"]["accepted_by_resource"] == {"action_rules": 1}
    assert report["card_runtime"]["card_lookup_executed"] == 1
    paired = report["comparison"]["paired_turns"]
    assert paired["shared_turns"] == 1
    assert paired["prediction_changes"]["slots_changed"] == 1
    assert "offline contribution" in report["causal_warning"]
    assert "Primary run" in render_markdown(report)
    assert "c1|2" in _json_safe(report)["card_runtime"]["per_turn"]


def test_hybrid_batch_inventory_separates_update_samples_from_eval_rows(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    batch_root = run / "trace2skill_hybrid_batches"
    first = batch_root / "batch_0001"
    second = batch_root / "batch_0002"
    for path, count in ((first, 2), (second, 1)):
        path.mkdir(parents=True)
        (path / "batch_summary.json").write_text(json.dumps({
            "num_conversations": count, "num_turns": count * 3,
            "failed_cases": count, "successful_cases": 0, "changelog": ["updated"],
        }), encoding="utf-8")
        (path / "trajectory_evidence.json").write_text(json.dumps([
            {"conversation_id": f"c{i}", "trajectory": [{}, {}, {}]}
            for i in range(count)
        ]), encoding="utf-8")
    (run / "rollout_schedule.json").write_text(json.dumps({
        "num_selected_samples": 3, "batch_size": 2, "batches": [{}, {}],
    }), encoding="utf-8")

    report = analyze(run)

    hybrid = report["trace2skill_hybrid"]
    assert hybrid["batch_count"] == 2
    assert hybrid["scheduled_selected_samples"] == 3
    assert [row["trajectory_evidence_turns"] for row in hybrid["batches"]] == [6, 3]
    rendered = render_markdown(report)
    assert "autonomous-reflection" in rendered
    assert "held-out prediction/action-turn counts are different denominators" in rendered


def test_plan_execute_tools_are_read_only_and_confined(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "evidence.md").write_text("offline rule", encoding="utf-8")
    tools = AuditFileTools(run_dir, None)

    result = tools.execute({"action": "read_file", "path": "primary/evidence.md"})
    assert "offline rule" in result["content"]
    with pytest.raises(ValueError):
        tools.execute({"action": "read_file", "path": "primary/../../secret.txt"})
    with pytest.raises(ValueError):
        tools.execute({"action": "read_file", "path": "comparison/result.json"})


def test_json_llm_response_is_rendered_as_markdown():
    report = _format_model_report('{"findings":["Cards are not always selected"],"next_step":"Compare matched turns"}')
    assert report.startswith("# LLM Analysis")
    assert "## Findings" in report
    assert "Compare matched turns" in report
