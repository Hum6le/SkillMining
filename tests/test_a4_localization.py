"""Controlled native replay sanity checks; no actual LLM calls."""
from types import SimpleNamespace
from copy import deepcopy
from threading import Barrier

import pytest

from test_a4_abcd import fixture_source, dump
from scripts import abcd_a4_localization as a4
from scripts.run_a4_abcd import read, validate_source, aggregate_results
from skill_mining import online_refinement as native


def source(tmp_path):
    root, flow, sample, _ = fixture_source(tmp_path)
    root_action = sample["source_action"]
    edges = [{"source": root_action, "target": sample["target_action"], "support": 10}]
    nodes = [{"id": root_action, "label": root_action},
             {"id": sample["target_action"], "label": sample["target_action"]}]
    # A real disconnected control of the same size, never invented by the sampler.
    nodes += [{"id": n, "label": n} for n in ("other-a", "other-b")]
    edges += [{"source": "other-a", "target": "other-b", "support": 4}]
    dump(root / flow / "offline_artifact" / "subgraph.json", {
        "nodes": nodes, "edges": edges, "backbone": {"edges": edges,
        "compilation_order": [n["id"] for n in nodes]}})
    dump(root / flow / "full" / "post_rollout_records.json", [{
        "sample": {k: v for k, v in sample.items() if k != "conversation"},
        "result": {"predicted_action": sample["target_action"], "predicted_slots": sample["gold_slots"]}}])
    config, records, context = validate_source(root, flow)
    return root, flow, config, records, context, a4.prepare_plan(config, records, context)


def args(root, output, arms="graph_local,no_graph,random_graph"):
    return SimpleNamespace(a2_root=str(root), output_dir=str(output), retries=1,
                           resume=False, seed=0, arms=arms)


def fake_llm(monkeypatch, *, fail_first=False):
    import scripts.llm_usage_utils as usage
    current, calls = {"calls": 0}, []
    monkeypatch.setattr(usage, "reset_usage", lambda: current.update(calls=0))
    monkeypatch.setattr(usage, "get_usage", lambda: {"total": {
        "calls": current["calls"], "total_tokens": current["calls"] * 100, "usage_available": True}})

    def chat(messages, **kwargs):
        current["calls"] += 1
        calls.append((kwargs["call_tag"], messages[0]["content"], kwargs.get("workflow_id")))
        if fail_first and len(calls) == 1:
            return {}, "bad", "empty_or_invalid_json_response", 1
        if kwargs["call_tag"] == "online_batch_root_cause":
            return {"summary": "mock", "root_causes": [], "candidate_updates": []}, "{}", "", 1
        return {"summary": "mock", "updates": [], "skill_operations": [
            {"op": "replace", "match_text": "# K0", "new_text": "# K1"},
            {"op": "replace", "match_text": "not-present", "new_text": "bad"}]}, "{}", "", 1

    monkeypatch.setattr(native, "_chat_for_json_with_retries", chat)
    return calls


def fake_test(monkeypatch):
    import scripts.run_subflow_eval as evaluation
    monkeypatch.setenv("OPENAI_API_KEY", "mock-unused")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://unused.invalid")
    calls = []

    def evaluate(agent, test, label, flow, *, save_dir, **kwargs):
        calls.append(label)
        samples = native.build_action_turn_samples(test)
        dump(save_dir / f"{label}_predictions.json", [{
            "convo_id": s["conversation_id"], "turn_index": s["turn_index"], "target_type": "action",
            "predicted_action": s["target_action"], "predicted_slots": s["gold_slots"]} for s in samples])
        return {"ast_cds": {"num_action_turns": len(samples), "ast_joint": 1.,
                            "ast_action_name": 1., "ast_slot_value": 1.}}

    monkeypatch.setattr(evaluation, "evaluate_agent_on_subflow", evaluate)
    return calls


def test_shared_plan_and_real_size_matched_controls(tmp_path):
    _, _, config, records, context, plan = source(tmp_path)
    assert plan == a4.prepare_plan(config, records, context)
    assert [bid for group in plan["groups"] for bid in group] == [j["batch_id"] for j in plan["jobs"]]
    job = plan["jobs"][0]
    actual = job["local_graph"]
    control = a4.graph_input("random_graph", actual, plan["state"], job["nodes"], 0, "job")
    assert len(control["nodes"]) == len(actual["nodes"]) == 2
    assert len(control["edges"]) == len(actual["edges"]) == 1
    assert {n["id"] for n in control["nodes"]} == {"other-a", "other-b"}
    assert control == a4.graph_input("random_graph", actual, plan["state"], job["nodes"], 0, "job")
    assert a4.graph_input("no_graph", actual, plan["state"], [], 0, "job") == {"nodes": [], "edges": []}
    assert set(a4.audit_graph_inputs(plan, a4.ARMS, 0)) == set(a4.ARMS)
    with pytest.raises(ValueError, match="size-matched"):
        a4.graph_input("random_graph", actual, plan["state"], [n["id"] for n in plan["state"]["nodes"]], 0, "job")


def test_three_arms_fresh_native_replay_resume_and_edit_metrics(tmp_path, monkeypatch):
    root, flow, _, _, _, _ = source(tmp_path / "source")
    original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    calls, tests = fake_llm(monkeypatch), fake_test(monkeypatch)
    options = args(root, tmp_path / "output")
    result = a4.run_flow(options, flow, ["ID1", "ID2", "ID3", "ID4"])
    assert result["complete"] and len(calls) == 6 and len(tests) == 3
    for arm in options.arms.split(","):
        assert result[arm]["ast_joint"] == 1.
        assert result["edit_metrics"][arm]["text_applied"] == 1
        assert result["edit_metrics"][arm]["text_failed"] == 1
        assert result["usage"][arm]["generation"]["total"]["calls"] == 2
    assert all(p.read_bytes() == content for p, content in original.items())
    # Identical diagnosis prompt except the explicit neighborhood JSON.
    graph = read(tmp_path / "output" / flow / "graph_local" / "batch_reports" / "evidence_batch_0001.json")["local_graph"]
    import json
    assert calls[0][1].replace(json.dumps(graph, ensure_ascii=False), json.dumps({"nodes": [], "edges": []})) == calls[2][1]
    options.resume = True
    assert a4.run_flow(options, flow, ["REPLACEMENT_ID"])["complete"]
    assert len(calls) == 6 and len(tests) == 3
    (tmp_path / "output" / flow / "graph_local" / "skill.md").write_text("tampered", encoding="utf-8")
    bad = a4.run_flow(options, flow, ["ID1"])
    assert not bad["complete"] and "artifact changed" in bad["arm_status"]["graph_local"]["error"]


def test_failed_case_continues_and_resume_retries_only_failed_stage(tmp_path, monkeypatch):
    root, flow, _, _, _, _ = source(tmp_path / "source")
    calls, tests = fake_llm(monkeypatch, fail_first=True), fake_test(monkeypatch)
    options = args(root, tmp_path / "output", "graph_local,no_graph")
    result = a4.run_flow(options, flow, ["ID1"])
    assert not result["complete"]
    assert not result["arm_status"]["graph_local"]["complete"]
    assert result["arm_status"]["no_graph"]["complete"]
    assert len(calls) == 3
    options.resume = True
    repaired = a4.run_flow(options, flow, ["ID2"])
    assert repaired["complete"] and len(calls) == 5
    assert repaired["usage"]["graph_local"]["generation"]["total"]["calls"] == 3
    assert len(tests) == 3  # Graph-local K1 changed; no-graph test is reused.


def test_reflection_failure_rolls_back_and_retries(tmp_path, monkeypatch):
    root, flow, config, records, context, plan = source(tmp_path / "source")
    calls = fake_llm(monkeypatch)
    import scripts.run_backbone_online_refine as runner
    original = runner._checkpoint
    checkpoints = []

    def failing(out, state, skill, *a, **kw):
        checkpoints.append(skill)
        if len(checkpoints) == 1:
            raise ValueError("mock checkpoint error")
        return original(out, state, skill, *a, **kw)

    monkeypatch.setattr(runner, "_checkpoint", failing)
    options = args(root, tmp_path / "output", "graph_local")
    out = tmp_path / "output" / flow / "graph_local"
    skill, state, _, errors = a4.refine_arm(config, context, records, plan, "graph_local", out, ["ID1"], options)
    assert errors and skill.startswith("# K0")
    assert not state.get("applied_skill_operation_ids", [])
    assert read(out / "refinement_status.json")["edit_metrics"]["rolled_back_groups"] == 1
    options.resume = True
    monkeypatch.setattr(runner, "_checkpoint", original)
    skill, _, usage, errors = a4.refine_arm(config, context, records, plan, "graph_local", out, ["ID1"], options)
    assert not errors and skill.startswith("# K1") and len(calls) == 3
    assert usage["total"]["calls"] == 3


def test_new_parent_keeps_workflows_and_excludes_incomplete_results(tmp_path, monkeypatch):
    import scripts.run_a4_abcd as parent
    root, flow, _, _, _, _ = source(tmp_path / "source")
    output = tmp_path / "output"
    monkeypatch.setattr(parent.sys, "argv", ["a4", "--subflow", flow, "--a2-root", str(root),
        "--output-dir", str(output), "--workflow-ids", "ID1,ID2,ID3,ID4"])
    seen = []

    def worker(cmd, **kwargs):
        seen.append(cmd)
        assert cmd[cmd.index("--workflow-ids") + 1] == "ID1,ID2,ID3,ID4"
        assert cmd[cmd.index("--experiment") + 1] == "graph-localization"
        dump(output / flow / "a4_result.json", {"subflow": flow, "complete": False})
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(parent.subprocess, "run", worker)
    assert parent.main() == 1 and len(seen) == 1
    assert read(output / "a4_aggregate.json")["aggregate"]["no_graph"]["action_turns"] == 0
    assert not aggregate_results(output, [flow], arms=["frozen_k0", "graph_local", "no_graph"],
                                 comparison_keys=[], source_graph_local=False)["complete"]


def test_diagnosis_runs_concurrently_on_distinct_workflows(tmp_path, monkeypatch):
    root, flow, config, records, context, plan = source(tmp_path / "source")
    calls = fake_llm(monkeypatch)
    plan["jobs"] = [dict(deepcopy(plan["jobs"][0]), batch_id=f"evidence_batch_{i:04d}") for i in range(1, 5)]
    plan["groups"] = [[j["batch_id"] for j in plan["jobs"]]]
    plan["grouping_mode"] = "graph_aware"
    barrier = Barrier(4)
    original = native.diagnose_rollout_batch

    def simultaneous(*a, **kw):
        barrier.wait(timeout=10)
        return original(*a, **kw)

    monkeypatch.setattr(native, "diagnose_rollout_batch", simultaneous)
    options = args(root, tmp_path / "output", "no_graph")
    _, _, usage, errors = a4.refine_arm(config, context, records, plan, "no_graph",
        tmp_path / "output" / flow / "no_graph", ["ID1", "ID2", "ID3", "ID4"], options)
    assert not errors
    assert {wid for tag, _, wid in calls if tag == "online_batch_root_cause"} == {"ID1", "ID2", "ID3", "ID4"}
    assert usage["total"]["calls"] == 5


def test_factorial_arms_native_replay_and_resume(tmp_path, monkeypatch):
    root, flow, _, _, _, _ = source(tmp_path / "source")
    calls, tests = fake_llm(monkeypatch), fake_test(monkeypatch)
    options = args(root, tmp_path / "output", a4.DEFAULT_ARMS)
    result = a4.run_flow(options, flow, ["ID1", "ID2"])
    assert result["complete"] and len(calls) == 8 and len(tests) == 4
    assert result["grouping"]["graph_local"]["mode"] == "graph_aware"
    assert result["grouping"]["no_graph_grouping"]["mode"] == "semantic_only"
    assert result["grouping"]["no_context_and_grouping"]["mode"] == "semantic_only"
    assert "no_graph_vs_no_context_and_grouping" in result
    assert "no_graph_grouping_vs_no_context_and_grouping" in result
    options.resume = True
    assert a4.run_flow(options, flow, ["ID3"])["complete"]
    assert len(calls) == 8 and len(tests) == 4


def test_grouping_changes_both_levels_with_exact_coverage_and_resource_exposure(tmp_path):
    _, _, config, _, context, _ = source(tmp_path / "source")
    from scripts.run_subflow_eval import load_subflow_data
    train, _ = load_subflow_data(config["subflow"])
    context = {**context, "train": train[:12]}
    samples = native.build_action_turn_samples(context["train"])
    assert len(samples) > 1
    records = [{"sample": {k: v for k, v in s.items() if k != "conversation"},
                "result": {"context": f"shared visible context {index % 2}",
                           "predicted_action": s["target_action"], "predicted_slots": s["gold_slots"]}}
               for index, s in enumerate(samples)]
    plan = a4.prepare_plan({**config, "batch_size": 1}, records, context, seed=7)
    graph, semantic = a4.arm_plan(plan, "graph_local"), a4.arm_plan(plan, "no_graph_grouping")
    assert [len(j["records"]) for j in graph["jobs"]] == [len(j["records"]) for j in semantic["jobs"]]
    assert [len(g) for g in graph["groups"]] == [len(g) for g in semantic["groups"]]
    expected = sorted(s["sample_id"] for s in samples)
    for selected in (graph, semantic):
        assert sorted(sid for j in selected["jobs"] for sid in j["sample_ids"]) == expected
        assert sorted(bid for group in selected["groups"] for bid in group) == sorted(j["batch_id"] for j in selected["jobs"])
        for job in selected["jobs"]:
            assert job["available_resources"] == [section for sid in job["sample_ids"]
                                                   for section in plan["sample_resources"][str(sid)]]
    assert [j["sample_ids"] for j in graph["jobs"]] != [j["sample_ids"] for j in semantic["jobs"]]
    assert semantic == a4.arm_plan(a4.prepare_plan({**config, "batch_size": 1}, records, context, seed=7), "no_graph_grouping")


def test_graph_free_packing_does_not_use_labels_or_topology():
    records = [{"sample_id": i, "context": "apple fruit" if i % 2 else "bus travel",
                "source_action": f"source-{i}", "target_action": f"target-{i}"} for i in range(12)]
    features = lambda r: set(r["context"].split())
    before = a4.semantic_partition(records, [3, 3, 2, 4], features, 9, "diagnosis")
    changed = [{**r, "source_action": "same", "target_action": "same", "graph": "different"} for r in records]
    after = a4.semantic_partition(changed, [3, 3, 2, 4], features, 9, "diagnosis")
    assert [[r["sample_id"] for r in g] for g in before] == [[r["sample_id"] for r in g] for g in after]
    assert sorted(r["sample_id"] for group in before for r in group) == list(range(12))
    with pytest.raises(ValueError, match="capacities"):
        a4.semantic_partition(records, [3], features, 9, "bad")
