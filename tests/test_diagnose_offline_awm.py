import json

from scripts.diagnose_offline_awm import catalog_from_prompt, checkpoint_report, digest, inspect_plan, diagnose


def test_snapshot_excludes_read_only_and_rejected_ids():
    prompt = ('## Current Editable Snapshot\n## skill.md\n[aaaaaaaaaaaaaaaa:L1] # Skill\n'
              '\n\n## Read-only Reference\n[bbbbbbbbbbbbbbbb:L1] ignore\n'
              '<rejected_edit_plan>cccccccccccccccc:L1</rejected_edit_plan>')
    assert catalog_from_prompt(prompt) == {"aaaaaaaaaaaaaaaa:L1": "skill.md"}


def test_distinguishes_id_failure_classes():
    ids = {"aaaaaaaaaaaaaaaa:L1": "skill.md"}
    targets = [("aaaaaaaaaaaaaaaa:L1", "slot_policies.md"),
               ("aaaaaaaaaaaaaaaa:L2", "skill.md"), ("bbbbbbbbbbbbbbbb:L1", "skill.md"),
               ("cccccccccccccccc:L1", "skill.md"), ("COPY_ACTUAL_ID", "skill.md")]
    plan = {"operations": [{"target_id": target, "file": name} for target, name in targets]}
    result = inspect_plan(json.dumps(plan), ids, {"bbbbbbbbbbbbbbbb:L1"})
    assert [i["reason"] for i in result["issues"]] == [
        "wrong_file", "unknown_line_in_current_revision", "id_seen_in_other_logged_snapshot",
        "unknown_revision", "malformed_or_placeholder_id"]
    assert inspect_plan('{"operations": [', ids, set())["status"] == "invalid_json"


def test_tail_cache_verification(tmp_path):
    state = {"scope": {}, "workflow": "RULE\n", "cards": {}, "completed_batches": 4,
             "failed_batches": [{"batch_index": 2}, {"batch_index": 3}]}
    (tmp_path / "batches").mkdir()
    cache = tmp_path / "batches" / "batch_0003.json"
    cache.write_text(json.dumps({"identity": {"scope_hash": digest({}), "workflow_hash": digest("RULE"),
                                             "cards_hash": digest({}), "index": 2}}))
    assert checkpoint_report(tmp_path, state)["tail_restart_precheck_passed"]
    state["cards"] = {"changed": "yes"}
    assert not checkpoint_report(tmp_path, state)["tail_restart_precheck_passed"]
    state["failed_batches"] = [{"batch_index": 1}, {"batch_index": 3}]
    assert not checkpoint_report(tmp_path, state)["failures_are_contiguous_processed_tail"]


def test_diagnose_is_read_only_and_does_not_echo_private_text(tmp_path):
    (tmp_path / "awm_state.json").write_text(json.dumps({"completed_batches": 0}))
    logs = tmp_path / "llm_responses" / "run"
    logs.mkdir(parents=True)
    prompt = '## Current Editable Snapshot\n## skill.md\n[aaaaaaaaaaaaaaaa:L1] PRIVATE_TEXT\n\n## Read-only Reference\n'
    (logs / "0001_awm_induction_validation_error_prompt.json").write_text(json.dumps({
        "messages": [{"content": prompt}], "extra": {"error": "invalid id", "attempt": 1}}))
    (logs / "0001_awm_induction_validation_error_response.json").write_text(json.dumps({
        "raw_response": json.dumps({"operations": [{"file": "skill.md", "target_id": "BAD", "new_text": "PRIVATE_TEXT"}]})}))
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = diagnose(tmp_path)
    assert report["logged_rejected_attempts"] == 1
    assert "PRIVATE_TEXT" not in json.dumps(report)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
