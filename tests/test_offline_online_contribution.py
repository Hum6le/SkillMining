import json
from pathlib import Path

from scripts.analyze_offline_online_contribution import analyze, render_markdown


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
