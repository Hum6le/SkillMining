"""Fairness gates and paired outcomes for A4's cached-feedback protocol."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.run_a4_abcd import (aggregate_results, bootstrap_command, correctness, generic_update,
                                 paired, run_flow, source_paths, validate_source, workflow_buckets)
from scripts.run_a4_abcd import merge_recorded_usage
from scripts.run_subflow_eval import load_subflow_data
from scripts.abcd_runtime_utils import abcd_runtime_options
from skill_mining.online_refinement import build_action_turn_samples


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_source(tmp_path):
    flow = "account_access"
    train, test = load_subflow_data(flow)
    train_sample = build_action_turn_samples(train[:1])[0]
    test_sample = build_action_turn_samples(test[:1])[0]
    root = tmp_path / flow
    offline, full, k0 = root / "offline_artifact", root / "full", root / "offline_only"
    offline.mkdir(parents=True)
    full.mkdir(parents=True)
    k0.mkdir(parents=True)
    (offline / "skill.md").write_text("# K0\n", encoding="utf-8")
    (full / "base_skill.md").write_text("# K0\n", encoding="utf-8")
    for name in ("reference", "action_rules", "slot_policies"):
        (offline / f"{name}.md").write_text(f"# {name}\n", encoding="utf-8")
        (full / f"base_{name}.md").write_text(f"# {name}\n", encoding="utf-8")
    dump(offline / "complete.json", {"subflow": flow, "max_train": 1})
    dump(offline / "subgraph.json", {"nodes": [], "edges": []})
    arguments = {
        "subflow": flow, "refinement_mode": "standard", "max_train": 1,
        "max_test": 1, "batch_size": 8, "model": "test-model"}
    dump(full / "runner_config.json", {"arguments": arguments})
    dump(full / "rollout_schedule.json", {"refinement_mode": "standard",
        "selection_unit": "action_turn", "batches": [{"samples": [{
            "sample_id": train_sample["sample_id"]}]}]})
    dump(full / "post_rollout_records.json", [{"sample": {
        "sample_id": train_sample["sample_id"],
        "target_action": train_sample["target_action"],
        "gold_slots": train_sample["gold_slots"]}, "result": {}}])
    dump(full / "skill_dag_state.json", {"posthoc_reflection_complete": True})
    samples = build_action_turn_samples(test[:1])
    result = {"runtime_protocol": abcd_runtime_options(SimpleNamespace(**arguments)),
              "ast_cds": {"num_action_turns": len(samples), "ast_joint": 1.0,
                          "ast_action_name": 1.0, "ast_slot_value": 1.0}}
    dump(full / "online_refine_result.json", result)
    dump(k0 / "result.json", result)
    predictions = [{"convo_id": s["conversation_id"], "turn_index": s["turn_index"],
                    "target_type": "action", "predicted_action": s["target_action"],
                    "predicted_slots": s["gold_slots"]} for s in samples]
    dump(k0 / "offline_only_predictions.json", predictions)
    dump(full / "online_refined_predictions.json", predictions)
    dump(full / "batch_root_causes" / "evidence_batch_0001.json", {
        "status": "ok", "batch_id": "evidence_batch_0001", "sample_ids": [train_sample["sample_id"]]})
    dump(full / "group_reflections" / "reflection_0001.json", {
        "status": "ok", "batch_ids": ["evidence_batch_0001"]})
    return tmp_path, flow, train_sample, test_sample


def test_source_checks_cached_gold_and_k0(tmp_path):
    root, flow, sample, _ = fixture_source(tmp_path)
    config, records, _ = validate_source(root, flow)
    assert config["max_train"] == 1 and len(records) == 1
    path = root / flow / "full" / "post_rollout_records.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[0]["sample"]["target_action"] = "WRONG"
    dump(path, payload)
    with pytest.raises(ValueError, match="target differs"):
        validate_source(root, flow)


def test_paired_counts_and_strict_slot_order(tmp_path):
    root, flow, _, test_sample = fixture_source(tmp_path)
    prediction = root / flow / "offline_only" / "offline_only_predictions.json"
    # One conversation may have more than one action turn; use the complete
    # gold action sequence to make the paired denominator exact.
    _, test = load_subflow_data(flow)
    samples = build_action_turn_samples(test[:1])
    dump(prediction, [{"convo_id": s["conversation_id"], "turn_index": s["turn_index"],
                       "target_type": "action", "predicted_action": s["target_action"],
                       "predicted_slots": s["gold_slots"]} for s in samples])
    before = correctness(prediction, test[:1])
    assert len(before) == len(samples) and all(before.values())
    rows = json.loads(prediction.read_text(encoding="utf-8"))
    rows[0]["predicted_action"] = "WRONG"
    dump(prediction, rows)
    after = correctness(prediction, test[:1])
    report = paired(before, after)
    assert report["regressed"] == 1 and report["fixed"] == 0
    assert report["net_gain"] == -1 / len(samples)


def test_generic_updater_uses_shared_edit_applicator(tmp_path, monkeypatch):
    import skill_mining.online_refinement as refinement

    root, flow, _, _ = fixture_source(tmp_path)
    config, records, _ = validate_source(root, flow)
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(kwargs["call_tag"])
        if kwargs["call_tag"] == "a4_generic_batch_analysis":
            return {"summary": "No supported change", "observations": [],
                    "supported_rules": []}, "{}", "", 1
        return {"summary": "Improve the candidate skill", "updates": [],
                "skill_operations": [{"op": "replace", "match_text": "# K0",
                                      "new_text": "# K1", "rationale": "mock edit"}]}, "{}", "", 1

    monkeypatch.setattr(refinement, "_chat_for_json_with_retries", fake_chat)
    skill, _, _ = generic_update(config, records, source_paths(root, flow),
                                 root / flow / "generic", [], retries=1)
    assert skill.startswith("# K1")
    assert calls == ["a4_generic_batch_analysis", "a4_generic_global_rewrite"]


@pytest.mark.parametrize("mutation,reason", [
    ("missing_runtime", "runtime_protocol"), ("wrong_runtime", "runtime contracts"),
    ("failed_group", "skipped analysis"), ("resource_drift", "resource differs"),
    ("report_coverage", "cover exactly"), ("missing_predictions", "cover"),
    ("metric_drift", "Joint AST/count")])
def test_source_rejects_invalid_comparison(tmp_path, mutation, reason):
    root, flow, _, _ = fixture_source(tmp_path)
    paths = source_paths(root, flow)
    full = paths["full"]
    if mutation in ("missing_runtime", "wrong_runtime", "metric_drift"):
        path = full / "online_refine_result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "missing_runtime":
            result.pop("runtime_protocol")
        elif mutation == "wrong_runtime":
            result["runtime_protocol"]["max_turns"] = 99
            dump(paths["k0_result"], result)  # Two identical wrong protocols still fail.
        else:
            result["ast_cds"]["ast_joint"] = 0.0
        dump(path, result)
    elif mutation == "failed_group":
        dump(full / "skill_dag_state.json", {"posthoc_reflection_complete": True,
                                            "posthoc_failed_reflection_groups": [1]})
    elif mutation == "resource_drift":
        (full / "base_action_rules.md").write_text("changed", encoding="utf-8")
    elif mutation == "report_coverage":
        dump(full / "batch_root_causes" / "evidence_batch_0001.json", {
            "batch_id": "evidence_batch_0001", "sample_ids": []})
    else:
        dump(paths["k0_predictions"], [])
    with pytest.raises(ValueError, match=reason):
        validate_source(root, flow)


def test_single_flow_keeps_all_workflow_ids():
    assert workflow_buckets(["account_access"], ["ID1", "ID2", "ID3", "ID4"]) == {
        "ID1,ID2,ID3,ID4": ["account_access"]}
    buckets = workflow_buckets(["a", "b", "c"], ["ID1", "ID2"])
    assert buckets == {"ID1": ["a", "c"], "ID2": ["b"]}


def test_absent_usage_tracker_is_not_zero_cost():
    usage = merge_recorded_usage({"schema_version": 0, "usage_available": False})
    assert usage["calls_available"] is False
    assert merge_recorded_usage(usage, {"total": {"calls": 1}})["calls_available"] is False


def test_bootstrap_preserves_existing_a2_and_skips_unrelated_arms(tmp_path):
    prior = {"arms": ["schema_only", "offline_only", "online_only", "full"],
             "model": "original-model", "batch_size": 4, "max_train": 20,
             "max_test": 5, "max_batches": 2, "refinement_mode": "standard",
             "skip_utterance_eval": True}
    dump(tmp_path / "a2_config.json", {"arguments": prior})
    args = SimpleNamespace(a2_root=str(tmp_path), max_train=None, max_test=None, max_batches=None)
    command = bootstrap_command(args, ["ID1", "ID2"], ["account_access"])
    assert command[command.index("--model") + 1] == "original-model"
    assert command[command.index("--skip-arms") + 1] == "schema_only,online_only"
    assert "--resume" in command
    args.max_train = 10
    with pytest.raises(ValueError, match="scope differs"):
        bootstrap_command(args, [], ["account_access"])


def test_failed_rewrite_resume_reuses_analysis_and_retains_usage(tmp_path, monkeypatch):
    import skill_mining.online_refinement as refinement
    import scripts.llm_usage_utils as usage
    root, flow, _, _ = fixture_source(tmp_path)
    config, records, _ = validate_source(root, flow)
    out = tmp_path / "generic"
    calls = []
    current = {"calls": 0}
    monkeypatch.setattr(usage, "reset_usage", lambda: current.update(calls=0))
    monkeypatch.setattr(usage, "get_usage", lambda: {"total": {"calls": current["calls"],
        "total_tokens": current["calls"] * 100, "usage_available": True}})

    def fake_chat(messages, **kwargs):
        tag = kwargs["call_tag"]
        current["calls"] += 1
        calls.append(tag)
        if tag.endswith("batch_analysis"):
            return {"summary": "ok", "observations": [], "supported_rules": []}, "{}", "", 1
        if calls.count(tag) == 1:
            return {}, "invalid", "bad JSON", 1
        return {"summary": "ok", "updates": [], "skill_operations": []}, "{}", "", 1

    monkeypatch.setattr(refinement, "_chat_for_json_with_retries", fake_chat)
    with pytest.raises(RuntimeError, match="rewrite failed"):
        generic_update(config, records, source_paths(root, flow), out, ["ID1"], retries=1)
    _, _, result_usage = generic_update(config, records, source_paths(root, flow), out,
                                         ["ID2"], retries=1, resume=True)
    assert calls.count("a4_generic_batch_analysis") == 1
    assert calls.count("a4_generic_global_rewrite") == 2
    assert result_usage["generation"]["total"]["calls"] == 3
    assert result_usage["generation"]["total"]["total_tokens"] == 300
    (root / flow / "offline_artifact" / "reference.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="different K0"):
        generic_update(config, records, source_paths(root, flow), out, [], retries=1, resume=True)


def test_pooled_metrics_unknown_cost_and_partial_failures(tmp_path):
    for flow, n, fixed, wrong in (("a", 10, 1, 2), ("b", 90, 9, 90)):
        outcomes = {"action_turns": n, "fixed": fixed, "regressed": 0,
                    "k0_wrong": wrong, "k0_correct": n - wrong}
        row = {"subflow": flow}
        for arm in ("frozen_k0", "graph_local", "generic_global"):
            row[arm] = {"num_action_turns": n, "ast_joint": fixed / n,
                        "ast_action_name": 0.5, "ast_slot_value": 0.2}
        for key in ("graph_local_vs_k0", "generic_global_vs_k0", "graph_local_vs_generic"):
            row[key] = outcomes
        dump(tmp_path / flow / "a4_result.json", row)
    result = aggregate_results(tmp_path, ["a", "b"])
    assert result["complete"]
    assert result["paired"]["graph_local_vs_k0"]["fix_rate"] == 10 / 92
    assert result["aggregate"]["graph_local"]["ast_joint"] == pytest.approx(0.1)
    assert result["cost"]["graph_local"]["updater"]["calls"] is None
    partial = aggregate_results(tmp_path, ["a", "b"], [{"subflow": "b", "exit_code": 1}])
    assert not partial["complete"] and partial["missing_subflows"] == ["b"]
    assert partial["aggregate"]["graph_local"]["action_turns"] == 10


def test_mock_end_to_end_is_readonly_and_resume_validates_artifacts(tmp_path, monkeypatch):
    import skill_mining.online_refinement as refinement
    import scripts.run_subflow_eval as evaluation
    # Constructor configuration only; every model/evaluation call is mocked.
    monkeypatch.setenv("OPENAI_API_KEY", "mock-not-a-real-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://unused.invalid")
    root, flow, _, _ = fixture_source(tmp_path / "source")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(kwargs["call_tag"])
        if kwargs["call_tag"].endswith("batch_analysis"):
            return {"summary": "ok", "observations": [], "supported_rules": []}, "{}", "", 1
        return {"summary": "edit", "updates": [], "skill_operations": [
            {"op": "replace", "match_text": "# K0", "new_text": "# K1"}]}, "{}", "", 1

    def fake_evaluate(agent, test, method, subflow, *, save_dir, **kwargs):
        # Real held-out ABCD action turns, synthetic predictions: plumbing sanity only.
        samples = build_action_turn_samples(test)
        dump(save_dir / "a4_generic_predictions.json", [{"convo_id": s["conversation_id"],
            "turn_index": s["turn_index"], "target_type": "action", "predicted_action": s["target_action"],
            "predicted_slots": s["gold_slots"]} for s in samples])
        return {"ast_cds": {"num_action_turns": len(samples), "ast_joint": 1.0,
                             "ast_action_name": 1.0, "ast_slot_value": 1.0}}

    monkeypatch.setattr(refinement, "_chat_for_json_with_retries", fake_chat)
    monkeypatch.setattr(evaluation, "evaluate_agent_on_subflow", fake_evaluate)
    args = SimpleNamespace(a2_root=str(root), output_dir=str(tmp_path / "output"), retries=1, resume=False)
    result = run_flow(args, flow, ["ID1", "ID2"])
    assert result["generic_global"]["ast_joint"] == 1.0
    assert result["graph_local_vs_k0"]["regressed"] == 0
    assert all(p.read_bytes() == content for p, content in before.items())
    args.resume = True
    run_flow(args, flow, ["ID3"])
    assert len(calls) == 2  # Neither training rollout nor updater/test rerun.
    (tmp_path / "output" / flow / "generic" / "skill.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifacts have changed"):
        run_flow(args, flow, ["ID3"])


def test_parent_continues_after_worker_failure_and_writes_partial_aggregate(tmp_path, monkeypatch):
    import scripts.run_a4_abcd as a4
    source, output = tmp_path / "source", tmp_path / "output"
    flows = ["account_access", "shipping_issue"]
    for flow in flows:
        for relative in ("full/online_refine_result.json", "full/post_rollout_records.json",
                         "offline_only/result.json", "offline_artifact/complete.json"):
            dump(source / flow / relative, {})
    monkeypatch.setattr(a4, "validate_source", lambda *args: ({}, [], {}))
    monkeypatch.setattr(a4.sys, "argv", ["a4", "--subflows", ",".join(flows),
        "--a2-root", str(source), "--output-dir", str(output), "--workflow-ids", "ID1"])
    seen = []

    def fake_worker(cmd, **kwargs):
        flow = cmd[cmd.index("--worker-subflow") + 1]
        seen.append(flow)
        assert cmd[cmd.index("--workflow-ids") + 1] == "ID1"
        if flow == flows[0]:
            return SimpleNamespace(returncode=1)
        row = {"subflow": flow}
        for arm in ("frozen_k0", "graph_local", "generic_global"):
            row[arm] = {"num_action_turns": 1, "ast_joint": 1.0,
                        "ast_action_name": 1.0, "ast_slot_value": 1.0}
        for key in ("graph_local_vs_k0", "generic_global_vs_k0", "graph_local_vs_generic"):
            row[key] = paired({("x", 1): True}, {("x", 1): True})
        dump(output / flow / "a4_result.json", row)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(a4.subprocess, "run", fake_worker)
    assert a4.main() == 1
    assert seen == flows
    result = json.loads((output / "a4_aggregate.json").read_text(encoding="utf-8"))
    assert not result["complete"] and result["completed_subflows"] == [flows[1]]
