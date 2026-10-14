"""Controlled replay of the native updater with different local graph inputs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
import random
from threading import Lock
from types import SimpleNamespace
from uuid import uuid4

from scripts.run_a4_abcd import (read, write, digest, value_digest, validate_source,
                                correctness, paired, merge_recorded_usage)
from skill_mining import online_refinement as native

PROTOCOL = "a4_graph_context_grouping_ablation_v2"
ARMS = ("graph_local", "no_graph", "no_graph_grouping", "no_context_and_grouping", "random_graph")
DEFAULT_ARMS = "graph_local,no_graph,no_graph_grouping,no_context_and_grouping"


def comparison_pairs(arms):
    pairs = [("graph_local", arm) for arm in arms if arm != "graph_local"] if "graph_local" in arms else []
    pairs += [(left, right) for left, right in (
        ("no_graph", "no_context_and_grouping"),
        ("no_graph_grouping", "no_context_and_grouping")) if left in arms and right in arms]
    return pairs


def semantic_partition(items, sizes, features, seed, key, outcome=None):
    """Graph-free, capacity-matched greedy packing; seeded tie/order control.

    Features must come from dialogue text, never action IDs/prefixes/topology.
    The capacities are shared budget constraints, not grouping signals.
    """
    if sum(sizes) != len(items) or any(size <= 0 for size in sizes):
        raise ValueError("Grouping capacities do not cover the input exactly")
    pending = list(items)
    random.Random(int(value_digest([seed, key])[:16], 16)).shuffle(pending)
    groups = []
    for size in sizes:
        first = pending.pop(0)
        left = features(first)

        def similarity(item):
            right = features(item)
            score = len(left & right) / max(1, len(left | right))
            if outcome is not None:
                score += .10 if outcome(first) == outcome(item) else .16
            return score

        ranked = sorted(pending, key=similarity, reverse=True)
        chosen = ranked[:size - 1]
        selected = {id(item) for item in chosen}
        pending = [item for item in pending if id(item) not in selected]
        groups.append([first, *chosen])
    return groups


def arm_plan(plan, arm):
    """Keep evidence/state fixed; change both packing levels only in grouping arms."""
    if plan.get("grouping_mode"):
        return plan
    mode = "semantic_only" if arm in {"no_graph_grouping", "no_context_and_grouping"} else "graph_aware"
    selected = plan.get("grouping_controls", {}).get(mode)
    if selected is None:
        if mode != "graph_aware":
            raise ValueError("Shared plan lacks the graph-free grouping control")
        selected = {"jobs": plan["jobs"], "groups": plan["groups"]}
    return {**plan, **selected, "grouping_mode": mode}


def prepare_plan(config, records, context, seed=0):
    """Rebuild pre-refinement evidence once, then freeze all shared inputs."""
    offline = context["paths"]["offline"]
    state = native.initialize_skill_dag(read(offline / "subgraph.json"), config["subflow"])
    samples = {str(s["sample_id"]): s for s in native.build_action_turn_samples(context["train"])}
    for index, record in enumerate(records, 1):
        sample = samples[str(record["sample"]["sample_id"])]
        row = dict(record["result"])
        row.update(convo_id=sample["conversation_id"], turn_index=sample["turn_index"], target_type="action")
        localized = native.localize_rollout_batch([sample], [row], state)
        native.accumulate_online_evidence(state, native.build_online_evidence_packets(localized), index)
    resources = {name: (offline / f"{name}.md").read_text(encoding="utf-8")
                 for name in ("reference", "action_rules", "slot_policies")}
    available_text = {
        "reference": resources["reference"] + "\n" + native.render_online_resources(state)[1],
        "action_rules": resources["action_rules"] + "\n" + native.render_online_action_rules(state),
        "slot_policies": resources["slot_policies"] + "\n" + native.render_online_slot_policies(state),
    }
    batches = native.build_post_rollout_batches(records, max_batch_size=int(config.get("batch_size") or 8))
    # Freeze retrieval per sample, not per batch: regrouping cannot change what
    # evidence a case brings into diagnosis. Do not deduplicate across cases,
    # which would introduce a composition-dependent resource/token budget.
    sample_resources = {}
    for record in records:
        query = " ".join(str(record["sample"].get(k) or "") for k in ("source_action", "target_action"))
        sample_resources[str(record["sample"]["sample_id"])] = [
            section for name, content in available_text.items()
            for section in native._resource_lookup_sections(name, content, query, 2)]

    def make_jobs(batches):
        jobs = []
        for index, batch in enumerate(batches, 1):
            nodes = sorted({str(r["sample"][key]) for r in batch for key in ("source_action", "target_action")
                            if r["sample"].get(key)})
            jobs.append({"batch_id": f"evidence_batch_{index:04d}", "records": batch,
                         "sample_ids": [r["sample"]["sample_id"] for r in batch], "nodes": nodes,
                         "local_graph": native.lookup_graph_neighborhood(state, nodes, radius=1),
                         "available_resources": [s for r in batch for s in sample_resources[str(r["sample"]["sample_id"])]]})
        return jobs

    jobs = make_jobs(batches)
    # Group using training-derived footprints, not the LLM reports of one arm.
    # Context-only arms share groups; grouping arms change membership, not capacities.
    footprints = [{"batch_id": j["batch_id"], "summary": "", "local_graph": j["local_graph"],
                   "graph_footprint": {"nodes": j["nodes"]}} for j in jobs]
    groups = [[r["batch_id"] for r in group] for group in native.group_batch_reports(footprints, max_reports=16)]
    def text_features(record):
        return native._tokenize_for_lookup(str(record.get("result", {}).get("context", "")))

    semantic_batches = semantic_partition(records, [len(b) for b in batches], text_features, seed, "diagnosis",
        outcome=lambda r: r.get("result", {}).get("predicted_action") == r["sample"].get("target_action"))
    semantic_jobs = make_jobs(semantic_batches)
    semantic_groups = semantic_partition(semantic_jobs, [len(g) for g in groups],
        lambda j: set().union(*(text_features(r) for r in j["records"])), seed, "reflection")
    semantic_groups = [[j["batch_id"] for j in group] for group in semantic_groups]
    catalog = {"actions": sorted({n["label"] for n in state["nodes"]}), "edge_ids": sorted(state["edges"])}
    return {"state": state, "resources": resources, "jobs": jobs, "groups": groups, "catalog": catalog,
            "grouping_seed": seed, "sample_resources": sample_resources,
            "grouping_controls": {"graph_aware": {"jobs": jobs, "groups": groups},
                                  "semantic_only": {"jobs": semantic_jobs, "groups": semantic_groups}},
            "base_skill": (offline / "skill.md").read_text(encoding="utf-8")}


def graph_input(arm, graph, state, anchors, seed, key):
    if arm in {"graph_local", "no_graph_grouping"}:
        return deepcopy(graph)
    if arm in {"no_graph", "no_context_and_grouping"}:
        return {"nodes": [], "edges": []}
    if arm != "random_graph":
        raise ValueError(f"Unknown A4 arm: {arm}")
    labels = {str(n["label"]): str(n["id"]) for n in state["nodes"]}
    excluded = {labels.get(str(a), str(a)) for a in anchors}
    eligible = {str(n["id"]): n for n in state["nodes"] if str(n["id"]) not in excluded}
    n, m = len(graph["nodes"]), len(graph["edges"])
    if len(eligible) < n:
        raise ValueError(f"No size-matched unrelated graph for {key}: need {n} nodes, have {len(eligible)}")
    universe = native.lookup_graph_neighborhood(state, list(eligible), radius=0)["edges"]
    if len(universe) < m:
        raise ValueError(f"No size-matched unrelated graph for {key}: need {m} edges, have {len(universe)}")
    rng = random.Random(int(value_digest([seed, key])[:16], 16))
    for _ in range(256):
        edges = rng.sample(universe, m)
        selected = {str(e[k]) for e in edges for k in ("source", "target")}
        if len(selected) > n:
            continue
        selected.update(rng.sample(sorted(set(eligible) - selected), n - len(selected)))
        return {"nodes": [deepcopy(eligible[k]) for k in sorted(selected)], "edges": deepcopy(edges)}
    raise ValueError(f"No size-matched unrelated graph found for {key} after 256 seeded attempts")


def audit_graph_inputs(plan, arms, seed):
    """Check all controls before paying for an updater; never shrink a control."""
    if len(arms) > 1:
        return {arm: audit_graph_inputs(arm_plan(plan, arm), [arm], seed)[arm] for arm in arms}
    plan = arm_plan(plan, arms[0])
    contexts = [(j["batch_id"], j["local_graph"], j["nodes"]) for j in plan["jobs"]]
    jobs = {j["batch_id"]: j for j in plan["jobs"]}
    for index, group in enumerate(plan["groups"], 1):
        anchors = sorted({n for bid in group for n in jobs[bid]["nodes"]})
        contexts.append((f"reflection_{index}",
                         native.lookup_graph_neighborhood(plan["state"], anchors, radius=1), anchors))
    return {arm: [{"key": key, "nodes": len(control["nodes"]), "edges": len(control["edges"])}
                  for key, graph, anchors in contexts
                  for control in [graph_input(arm, graph, plan["state"], anchors, seed, key)]]
            for arm in arms}


def controlled_template(template, catalog, resources=None):
    def literal(value):
        return json.dumps(value, ensure_ascii=False).replace("{", "{{").replace("}", "}}")
    result = template + "\n<shared_edit_catalog>" + literal(catalog) + "</shared_edit_catalog>\n"
    if resources is not None:
        result += "<fixed_available_resources>" + literal(resources) + "</fixed_available_resources>\n"
    return result


def _manifest(config, context, records, plan, arm, args):
    paths = context["paths"]
    root = Path(__file__).resolve().parents[1]
    return {"protocol": PROTOCOL, "arm": arm, "seed": args.seed, "retries": args.retries,
            "source_config": digest(paths["full"] / "runner_config.json"),
            "feedback_sha256": digest(paths["full"] / "post_rollout_records.json"),
            "sample_ids": [r["sample"]["sample_id"] for r in records],
            "k0_sha256": value_digest(plan["base_skill"]), "plan_sha256": value_digest(plan),
            "train_sha256": value_digest(context["train"]), "test_sha256": value_digest(context["test"]),
            "model": config["model"], "runtime_protocol": read(paths["k0_result"])["runtime_protocol"],
            "implementation_sha256": digest(Path(__file__)), "native_sha256": digest(Path(native.__file__)),
            "dependency_sha256": {name: digest(root / name) for name in (
                "scripts/run_a4_abcd.py", "scripts/run_backbone_online_refine.py",
                "scripts/abcd_runtime_utils.py", "skill_mining/skill_writer.py",
                "eval_tod/abcd/agent.py", "scripts/run_subflow_eval.py", "scripts/llm_usage_utils.py")}}


def _check_cache(path, identity, resume):
    if resume and path.is_file():
        item = read(path)
        if item.get("identity") == identity and item.get("status") != "error" and not item.get("error"):
            return item
    return None


def refine_arm(config, context, records, plan, arm, out, ids, args):
    from eval_tod.response_logger import ResponseLogger
    from scripts.llm_usage_utils import get_usage, reset_usage
    from scripts.run_backbone_online_refine import _checkpoint
    plan = arm_plan(plan, arm)
    manifest = _manifest(config, context, records, plan, arm, args)
    manifest_path = out / "a4_manifest.json"
    if manifest_path.is_file() and read(manifest_path) != manifest:
        raise ValueError(f"A4 {arm} source, plan, or implementation changed; use a fresh output")
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError(f"A4 {arm} output is nonempty; use --resume")
    if out.exists() and any(out.iterdir()) and not manifest_path.is_file():
        raise ValueError(f"A4 {arm} output has no provenance manifest")
    if not manifest_path.is_file():
        write(manifest_path, manifest)
    completed_path = out / "result.json"
    if args.resume and completed_path.is_file() and read(completed_path).get("refinement_complete"):
        completed = read(completed_path)
        if completed.get("manifest_hash") != value_digest(manifest) or not completed.get("artifact_hashes"):
            raise ValueError(f"A4 {arm} completed result has incompatible provenance")
        for name, checksum in completed["artifact_hashes"].items():
            if not (out / name).is_file() or digest(out / name) != checksum:
                raise ValueError(f"A4 {arm} completed artifact changed: {name}")
        return (out / "skill.md").read_text(encoding="utf-8"), native.load_skill_dag(out / "skill_dag_state.json"), completed["llm_usage"]["generation"], []
    state, skill = deepcopy(plan["state"]), plan["base_skill"]
    usage_path = out / "generation_usage.json"
    previous_usage = read(usage_path) if args.resume and usage_path.is_file() else None
    reset_usage()
    logger, usage_lock = ResponseLogger(out / "llm_responses" / uuid4().hex), Lock()
    errors, reports = [], {}

    def persist():
        with usage_lock:
            write(usage_path, merge_recorded_usage(previous_usage, get_usage()))

    def analyze(job, wid):
        path = out / "batch_reports" / (job["batch_id"] + ".json")
        identity = {"manifest_hash": value_digest(manifest), "input_hash": value_digest(job)}
        report = _check_cache(path, identity, args.resume)
        if report is None:
            try:
                graph = graph_input(arm, job["local_graph"], plan["state"], job["nodes"], args.seed, job["batch_id"])
                report = native.diagnose_rollout_batch(job["batch_id"], job["records"], state, skill, config["model"],
                    response_logger=logger, workflow_id=wid, max_retries=args.retries,
                    prompt_template=controlled_template(native._BATCH_ROOT_CAUSE_PROMPT, plan["catalog"]),
                    graph_context=graph, available_resources_override=job["available_resources"])
            except Exception as exc:
                report = {"status": "error", "error": repr(exc)}
            report.update(identity=identity, batch_id=job["batch_id"], sample_ids=job["sample_ids"],
                          arm=arm, fixed_resource_hash=value_digest(job["available_resources"]))
            write(path, report)
        persist()
        print(f"A4 {arm}: diagnosis {job['batch_id']} status={report.get('status')}", flush=True)
        return report

    buckets = [[] for _ in (ids or [None])]
    for index, job in enumerate(plan["jobs"]):
        buckets[index % len(buckets)].append(job)
    def worker(index):
        return [analyze(job, ids[index] if ids else None) for job in buckets[index]]
    # One serial request stream per workflow. Only parent applies edits.
    with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
        collected = [item for group in pool.map(worker, range(len(buckets))) for item in group]
    for report in collected:
        reports[report["batch_id"]] = report
        if report.get("status") == "error" or report.get("error"):
            errors.append({"phase": "diagnosis", "batch_id": report["batch_id"], "error": report.get("error")})
    applied_ids, reflected_results = set(), []
    jobs = {j["batch_id"]: j for j in plan["jobs"]}
    for index, batch_ids in enumerate(plan["groups"], 1):
        if any(reports[bid].get("status") == "error" or reports[bid].get("error") for bid in batch_ids):
            errors.append({"phase": "reflection_skipped", "group_index": index, "error": "missing_successful_diagnosis"})
            continue
        path = out / "group_reflections" / f"reflection_{index:04d}.json"
        group = [reports[bid] for bid in batch_ids]
        anchors = sorted({node for bid in batch_ids for node in jobs[bid]["nodes"]})
        true_graph = native.lookup_graph_neighborhood(plan["state"], anchors, radius=1)
        fixed_resources = [s for bid in batch_ids for s in jobs[bid]["available_resources"]]
        identity = {"manifest_hash": value_digest(manifest), "skill_hash": value_digest(skill),
                    "state_hash": value_digest(state), "reports_hash": value_digest(group)}
        reflection = _check_cache(path, identity, args.resume)
        before = (deepcopy(state), skill, set(applied_ids))
        try:
            if reflection is None:
                graph = graph_input(arm, true_graph, plan["state"], anchors, args.seed, f"reflection_{index}")
                reflection = native.reflect_batch_report_group(group, state, skill, config["model"],
                    response_logger=logger, workflow_id=ids[(index - 1) % len(ids)] if ids else None,
                    max_retries=args.retries, graph_context=graph,
                    prompt_template=controlled_template(native._GROUP_REFLECTION_PROMPT, plan["catalog"], fixed_resources))
                reflection.update(identity=identity, arm=arm, fixed_resource_hash=value_digest(fixed_resources))
            if reflection.get("status") == "error" or reflection.get("error"):
                raise RuntimeError(str(reflection.get("error")))
            accepted, rejected = native.apply_reflection_updates_to_state(state, reflection.get("updates", []))
            skill, operations = native.apply_dynamic_skill_operations(skill, reflection.get("skill_operations", []), applied_ids)
            state["applied_skill_operation_ids"] = sorted(applied_ids)
            skill = _checkpoint(out, state, skill, plan["resources"]["reference"],
                base_slot_policies=plan["resources"]["slot_policies"], base_action_rules=plan["resources"]["action_rules"])
            reflection.update(accepted_updates=accepted, rejected_updates=rejected, applied_skill_operations=operations)
        except Exception as exc:
            state, skill, applied_ids = before
            reflection = reflection or {}
            reflection.update(status="error", error=repr(exc), identity=identity, rolled_back=True)
            errors.append({"phase": "reflection", "group_index": index, "error": repr(exc)})
        write(path, reflection)
        reflected_results.append(reflection)
        persist()
        print(f"A4 {arm}: reflection {index}/{len(plan['groups'])} status={reflection.get('status')}", flush=True)
    skill = _checkpoint(out, state, skill, plan["resources"]["reference"],
        base_slot_policies=plan["resources"]["slot_policies"], base_action_rules=plan["resources"]["action_rules"])
    persist()
    edits = {"accepted_updates": 0, "rejected_updates": 0, "text_applied": 0,
             "text_failed": 0, "text_skipped": 0, "rolled_back_groups": 0}
    for report in reflected_results:
        if report.get("rolled_back"):
            edits["rolled_back_groups"] += 1
            continue
        edits["accepted_updates"] += len(report.get("accepted_updates", []))
        edits["rejected_updates"] += len(report.get("rejected_updates", []))
        for operation in report.get("applied_skill_operations", []):
            key = "text_failed" if operation.get("error") else "text_skipped" if operation.get("skipped") else "text_applied"
            edits[key] += 1
    write(out / "refinement_status.json", {"complete": not errors, "errors": errors,
          "diagnosis_batches": len(plan["jobs"]), "reflection_groups": len(plan["groups"]), "edit_metrics": edits,
          "grouping_mode": plan["grouping_mode"], "batch_sizes": [len(j["records"]) for j in plan["jobs"]],
          "group_sizes": [len(g) for g in plan["groups"]]})
    return skill, state, read(usage_path), errors


def evaluate_arm(config, context, plan, arm, out, ids, args, skill, state, generation, errors):
    from awm import WorkflowStore
    from eval_tod.response_logger import ResponseLogger
    from scripts.abcd_runtime_utils import build_abcd_runtime_agent, abcd_runtime_options
    from scripts.llm_usage_utils import get_usage, reset_usage, split_usage_summary
    from scripts.run_subflow_eval import evaluate_agent_on_subflow
    label = f"a4_{arm}"
    prediction_path = out / f"{label}_predictions.json"
    manifest_hash = value_digest(read(out / "a4_manifest.json"))
    result_path = out / "result.json"
    previous = read(result_path) if args.resume and result_path.is_file() else None
    # Partial-refinement evaluation is reused only if its exact K1 is unchanged.
    reusable = previous and previous.get("manifest_hash") == manifest_hash and previous.get("artifact_hashes") and all(
        (out / name).is_file() and digest(out / name) == checksum for name, checksum in previous["artifact_hashes"].items())
    if reusable:
        result = previous
        testing = previous["llm_usage"]["testing"]
    else:
        usage_path = out / "testing_attempt_usage.json"
        prior_usage = read(usage_path) if args.resume and usage_path.is_file() else None
        reset_usage()
        workflow = WorkflowStore()
        workflow.update(skill)
        runtime_args = SimpleNamespace(**config)
        resources = plan["resources"]
        agent = build_abcd_runtime_agent(runtime_args, workflow=workflow,
            reference_text=resources["reference"] + "\n" + native.render_online_resources(state)[1],
            action_rules_text=resources["action_rules"] + "\n" + native.render_online_action_rules(state),
            slot_policies_text=resources["slot_policies"] + "\n" + native.render_online_slot_policies(state),
            response_logger=ResponseLogger(out / "test_llm_responses" / uuid4().hex), workflow_id=ids[0] if ids else None)
        parallel_ids = ids if len(ids) > 1 and "fork" in mp.get_all_start_methods() else None
        try:
            result = evaluate_agent_on_subflow(agent, context["test"], label, config["subflow"],
                save_dir=out, skip_utterance_eval=bool(config.get("skip_utterance_eval")), eval_workflow_ids=parallel_ids)
        except Exception:
            testing = merge_recorded_usage(prior_usage, get_usage())
            if parallel_ids or (prior_usage or {}).get("incomplete_parallel_accounting"):
                testing.update(incomplete_parallel_accounting=True, calls_available=False)
                testing["total"]["usage_available"] = False
            write(usage_path, testing)
            raise
        testing = merge_recorded_usage(prior_usage, get_usage(), *result.pop("_evaluation_worker_usage", []))
        if (prior_usage or {}).get("incomplete_parallel_accounting"):
            testing.update(incomplete_parallel_accounting=True, calls_available=False)
            testing["total"]["usage_available"] = False
        write(usage_path, testing)
        result["runtime_protocol"] = abcd_runtime_options(runtime_args)
    decisions = correctness(prediction_path, context["test"])
    ast = result["ast_cds"]
    if ast["num_action_turns"] != len(decisions) or abs(ast["ast_joint"] - sum(decisions.values()) / max(1, len(decisions))) > .000051:
        raise ValueError("A4 reported AST differs from held-out ground-truth scoring")
    result.update(protocol=PROTOCOL, arm=arm, manifest_hash=manifest_hash,
                  refinement_complete=not errors, refinement_errors=errors,
                  llm_usage=split_usage_summary(generation, testing),
                  edit_metrics=read(out / "refinement_status.json")["edit_metrics"])
    result["artifact_hashes"] = {name: digest(out / name) for name in (
        "skill.md", "skill_dag_state.json", "reference.md", "action_rules.md", "slot_policies.md", prediction_path.name)}
    write(out / "llm_usage.json", result["llm_usage"])
    write(result_path, result)
    return result, decisions


def run_flow(args, flow, ids):
    config, records, context = validate_source(Path(args.a2_root), flow)
    plan = prepare_plan(config, records, context, args.seed)
    out = Path(args.output_dir) / flow
    shared_path = out / "shared_replay_plan.json"
    if shared_path.is_file() and read(shared_path) != json.loads(json.dumps(plan)):
        raise ValueError("A4 shared feedback/resource/group plan changed")
    if not shared_path.is_file():
        write(shared_path, plan)
    k0 = correctness(context["paths"]["k0_predictions"], context["test"])
    results = {"protocol": PROTOCOL, "subflow": flow, "complete": True,
               "frozen_k0": read(context["paths"]["k0_result"])["ast_cds"],
               "resource_lineage": context["resource_lineage"], "arm_status": {}, "edit_metrics": {}, "usage": {
                   "frozen_k0": read(context["paths"]["k0_result"]).get("llm_usage")},
               "feedback_samples": len(records), "shared_plan_sha256": value_digest(plan), "grouping": {}}
    decisions = {}
    for arm in args.arms.split(","):
        arm_out = out / arm
        try:
            audit_graph_inputs(plan, [arm], args.seed)
            selected = arm_plan(plan, arm)
            results["grouping"][arm] = {"mode": selected["grouping_mode"],
                "batch_sizes": [len(j["records"]) for j in selected["jobs"]], "group_sizes": [len(g) for g in selected["groups"]],
                "plan_sha256": value_digest(selected)}
            skill, state, generation, errors = refine_arm(config, context, records, selected, arm, arm_out, ids, args)
            result, scores = evaluate_arm(config, context, selected, arm, arm_out, ids, args, skill, state, generation, errors)
            results[arm] = result["ast_cds"]
            results["usage"][arm] = result["llm_usage"]
            results["edit_metrics"][arm] = result["edit_metrics"]
            results[f"{arm}_vs_k0"] = paired(k0, scores)
            results["arm_status"][arm] = {"complete": not errors, "errors": errors}
            decisions[arm] = scores
            results["complete"] &= not errors
        except Exception as exc:
            results["complete"] = False
            results["arm_status"][arm] = {"complete": False, "error": repr(exc)}
            print(f"FAILED A4 {flow}/{arm}: {exc}", flush=True)
        write(out / "a4_result.json", results)
    for left, right in comparison_pairs(list(decisions)):
        results[f"{left}_vs_{right}"] = paired(decisions[right], decisions[left])
    write(out / "a4_result.json", results)
    return results
