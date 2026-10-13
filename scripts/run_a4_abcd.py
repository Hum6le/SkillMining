#!/usr/bin/env python3
"""A4: compare graph-local and generic updates on identical cached K0 feedback.

The A2 ``full`` arm is the graph-local updater. Its standard-mode training
feedback is saved in ``post_rollout_records.json`` (the executable skill stays
at K0 during collection; auxiliary evidence may accumulate).
This runner verifies that cache, starts a generic updater from the same K0,
and evaluates both frozen K1 artifacts against the same held-out split.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_a2_abcd import SPLITS, _ids
from scripts.abcd_runtime_utils import abcd_runtime_options

PROTOCOL = "a4_same_cached_k0_feedback_v2"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def value_digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def merge_recorded_usage(*snapshots) -> dict:
    from scripts.llm_usage_utils import merge_usage_summaries
    valid = [item for item in snapshots if isinstance(item, dict)]
    combined = merge_usage_summaries(*valid)
    combined["calls_available"] = bool(valid) and all(
        "calls" in item.get("total", {}) and item.get("calls_available", True) for item in valid)
    return combined


def selected_flows(args) -> list[str]:
    index = read(SPLITS / "INDEX.json")
    flows = sorted(index) if args.all else ([x.strip() for x in args.subflows.split(",") if x.strip()]
                                      if args.subflows else [args.subflow])
    if not flows or len(flows) != len(set(flows)) or any(flow not in index for flow in flows):
        raise ValueError("Choose unique existing ABCD subflows")
    return flows


def source_paths(a2_root: Path, flow: str) -> dict[str, Path]:
    root = a2_root / flow
    return {
        "offline": root / "offline_artifact",
        "full": root / "full",
        "k0_result": root / "offline_only" / "result.json",
        "k0_predictions": root / "offline_only" / "offline_only_predictions.json",
    }


def validate_source(a2_root: Path, flow: str) -> tuple[dict, list[dict], dict]:
    paths = source_paths(a2_root, flow)
    offline, full = paths["offline"], paths["full"]
    required = [
        offline / "complete.json", offline / "subgraph.json", offline / "skill.md",
        offline / "reference.md", offline / "action_rules.md", offline / "slot_policies.md",
        full / "base_skill.md", full / "post_rollout_records.json",
        full / "runner_config.json", full / "rollout_schedule.json",
        full / "skill_dag_state.json", full / "online_refine_result.json",
        full / "online_refined_predictions.json", paths["k0_result"],
        paths["k0_predictions"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("A4 needs completed A2 offline_only and full arms: " + ", ".join(missing))
    config = read(full / "runner_config.json").get("arguments", {})
    if config.get("subflow") != flow or config.get("refinement_mode") != "standard":
        raise ValueError(f"A2 full must be a standard-mode run for {flow}")
    if digest(offline / "skill.md") != digest(full / "base_skill.md"):
        raise ValueError("A2 full K0 differs from offline_only K0")
    meta = read(offline / "complete.json")
    if meta.get("subflow") != flow or meta.get("max_train") != config.get("max_train"):
        raise ValueError("A2 offline and full training scopes differ")
    schedule = read(full / "rollout_schedule.json")
    if schedule.get("refinement_mode") != "standard" or schedule.get("selection_unit") != "action_turn":
        raise ValueError("A4 requires frozen K0 action-turn feedback")
    state = read(full / "skill_dag_state.json")
    if not state.get("posthoc_reflection_complete"):
        raise ValueError("A2 full refinement is incomplete")
    if state.get("posthoc_failed_batch_ids") or state.get("posthoc_failed_reflection_groups"):
        raise ValueError("A2 full has skipped analysis/reflection groups; repair them before same-feedback comparison")
    from scripts.run_subflow_eval import load_subflow_data
    from skill_mining.online_refinement import build_action_turn_samples
    train, test = load_subflow_data(flow)
    if config.get("max_train") is not None:
        train = train[:int(config["max_train"])]
    if config.get("max_test") is not None:
        test = test[:int(config["max_test"])]
    source_samples = {str(s["sample_id"]): s for s in build_action_turn_samples(train)}
    scheduled = [str(s["sample_id"]) for b in schedule.get("batches", [])
                 for s in b.get("samples", [])]
    records = read(full / "post_rollout_records.json")
    actual = [str(r.get("sample", {}).get("sample_id")) for r in records]
    if not records or len(scheduled) != len(set(scheduled)) or len(actual) != len(set(actual)) or actual != scheduled:
        raise ValueError("Cached feedback does not match the frozen rollout schedule")
    for record in records:
        sample = record["sample"]
        truth = source_samples.get(str(sample["sample_id"]))
        if truth is None or (sample.get("target_action"), sample.get("gold_slots")) != (
                truth.get("target_action"), truth.get("gold_slots")):
            raise ValueError("Cached feedback target differs from current training split")
        for key in ("conversation_id", "turn_index", "prefix_action_sequence"):
            if key in sample and sample[key] != truth[key]:
                raise ValueError(f"Cached feedback sample identity differs: {key}")
        result = record.get("result", {})
        if not isinstance(result, dict):
            raise ValueError("Cached feedback result must be an object")
        if "convo_id" in result and str(result["convo_id"]) != str(truth["conversation_id"]):
            raise ValueError("Cached result belongs to a different conversation")
        if "turn_index" in result and int(result["turn_index"]) != int(truth["turn_index"]):
            raise ValueError("Cached result belongs to a different turn")
    k0 = read(paths["k0_result"])
    full_result = read(full / "online_refine_result.json")
    expected_runtime = abcd_runtime_options(SimpleNamespace(**config))
    if not k0.get("runtime_protocol") or not full_result.get("runtime_protocol"):
        raise ValueError("A4 needs recorded runtime_protocol on both A2 arms; legacy missing metadata is not proof of equality")
    if k0["runtime_protocol"] != full_result["runtime_protocol"] or full_result["runtime_protocol"] != expected_runtime:
        raise ValueError("A2 frozen K0 and graph-local K1 runtime contracts differ")
    lineage = {"skill_verified": True, "reference_verified": False,
               "action_rules_verified": False, "slot_policies_verified": False}
    for name in ("reference", "action_rules", "slot_policies"):
        base = full / f"base_{name}.md"
        if base.is_file():
            if digest(base) != digest(offline / f"{name}.md"):
                raise ValueError(f"A2 full K0 resource differs: {name}")
            lineage[name + "_verified"] = True
    source_groups = sorted((full / "group_reflections").glob("reflection_*.json"))
    if not source_groups:
        raise ValueError("A2 full is missing graph-local reflection groups")
    for path in source_groups:
        reflection = read(path)
        if reflection.get("status") == "error" or reflection.get("error"):
            raise ValueError(f"A2 full reflection is not successfully completed: {path}")
    source_reports = sorted((full / "batch_root_causes").glob("evidence_batch_*.json"))
    if not source_reports:
        raise ValueError("A4 needs source root-cause reports to audit the graph-local feedback coverage")
    report_ids, used_ids = [], []
    for path in source_reports:
        report = read(path)
        if report.get("status") == "error" or report.get("error"):
            raise ValueError("A2 full root-cause analysis has failed artifacts")
        report_ids.append(str(report.get("batch_id")))
        used_ids.extend(str(value) for value in report.get("sample_ids", []))
    grouped_ids = [str(value) for path in source_groups for value in read(path).get("batch_ids", [])]
    if (len(report_ids) != len(set(report_ids)) or len(used_ids) != len(set(used_ids)) or set(used_ids) != set(actual)
            or len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(report_ids)):
        raise ValueError("A2 graph-local reports/reflections do not cover exactly the cached feedback")
    # Do not pay for generic updates before discovering stale/incomplete source
    # predictions. Check their exact coverage and rounded reported Joint AST.
    for prediction_path, reported in ((paths["k0_predictions"], k0), (full / "online_refined_predictions.json", full_result)):
        decisions = correctness(prediction_path, test)
        ast = reported.get("ast_cds", {})
        score = sum(decisions.values()) / len(decisions) if decisions else 0.0
        if ast.get("num_action_turns") != len(decisions) or abs(float(ast.get("ast_joint", -1)) - score) > 0.000051:
            raise ValueError("A2 reported Joint AST/count differs from current held-out predictions")
    return config, records, {"test": test, "train": train, "paths": paths, "schedule": schedule,
                             "resource_lineage": lineage}


def feedback_view(record: dict) -> dict:
    """Only cached training supervision, never hidden test or prompt traces."""
    from skill_mining.online_refinement import _react_model_output_projection
    sample, result = record["sample"], record["result"]
    return {
        "sample_id": sample["sample_id"],
        "prefix_actions": sample.get("prefix_action_sequence", []),
        "context": str(result.get("context", "")),
        "gold_action": sample.get("target_action"),
        "gold_slots": sample.get("gold_slots", []),
        "predicted_action": result.get("predicted_action"),
        "predicted_slots": result.get("predicted_slots", []),
        "prediction": result.get("prediction", ""),
        "full_trajectory": [{key: item.get(key) for key in
                             ("turn_index", "target_type", "context", "prediction", "predicted_action", "predicted_slots")}
                            for item in record.get("trajectory", [])],
        **_react_model_output_projection(result.get("react_trace")),
    }


def generic_update(config: dict, records: list[dict], paths: dict[str, Path],
                   out: Path, workflow_ids: list[str], *, retries: int = 3,
                   resume: bool = False) -> tuple[str, dict, dict]:
    from eval_tod.response_logger import ResponseLogger
    from scripts.llm_usage_utils import get_usage, reset_usage
    from skill_mining.online_refinement import (
        _chat_for_json_with_retries, apply_dynamic_skill_operations,
        apply_reflection_updates_to_state, build_post_rollout_batches,
        initialize_skill_dag,
        render_online_resources, render_online_action_rules, render_online_slot_policies,
    )
    from scripts.run_backbone_online_refine import _checkpoint
    from scripts.run_subflow_eval import load_subflow_data

    offline = paths["offline"]
    base_skill = (offline / "skill.md").read_text(encoding="utf-8")
    resources = {name: (offline / f"{name}.md").read_text(encoding="utf-8")
                 if (offline / f"{name}.md").is_file() else ""
                 for name in ("reference", "action_rules", "slot_policies")}
    subgraph = read(offline / "subgraph.json")
    state = initialize_skill_dag(subgraph, config["subflow"])
    batches = build_post_rollout_batches(records, max_batch_size=int(config.get("batch_size") or 8))
    train, test = load_subflow_data(config["subflow"])
    manifest = {
        "protocol": PROTOCOL, "updater": "generic_global",
        "k0_sha256": digest(offline / "skill.md"),
        "feedback_sha256": digest(paths["full"] / "post_rollout_records.json"),
        "sample_ids": [str(r["sample"]["sample_id"]) for r in records],
        "model": config["model"], "batch_size": config.get("batch_size"),
        "runtime_protocol": read(paths["k0_result"])["runtime_protocol"],
        "resource_hashes": {name: digest(offline / f"{name}.md") for name in resources},
        "graph_sha256": digest(offline / "subgraph.json"),
        "source_config_sha256": digest(paths["full"] / "runner_config.json"),
        "source_schedule_sha256": digest(paths["full"] / "rollout_schedule.json"),
        "split_hashes": {"train": value_digest(train[:config.get("max_train")]),
                         "test": value_digest(test[:config.get("max_test")])},
        "source_reflection_hashes": {path.name: digest(path) for path in sorted((paths["full"] / "group_reflections").glob("reflection_*.json"))},
        "retries": retries,
    }
    manifest_path = out / "a4_manifest.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError("A4 output belongs to different K0, feedback, or runtime")
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError("A4 generic output is nonempty; use --resume or a fresh directory")
    if not manifest_path.is_file():
        write(manifest_path, manifest)
    if resume and (out / "a4_generic_result.json").is_file():
        completed = read(out / "a4_generic_result.json")
        if completed.get("manifest_hash") != value_digest(manifest):
            raise ValueError("Completed A4 generic result has an incompatible manifest")
        if not completed.get("artifact_hashes") or any(not (out / name).is_file() or digest(out / name) != checksum
                                                       for name, checksum in completed["artifact_hashes"].items()):
            raise ValueError("Completed A4 generic artifacts have changed or are incomplete")
        return (out / "skill.md").read_text(encoding="utf-8"), read(out / "skill_dag_state.json"), read(out / "llm_usage.json")

    logger = ResponseLogger(out / "llm_responses")
    generation_path = out / "generation_usage.json"
    previous_usage = read(generation_path) if resume and generation_path.is_file() else None
    reset_usage()

    def cached_artifact(path, identity):
        if not resume or not path.is_file():
            return None
        payload = read(path)
        if payload.get("identity") == identity and not payload.get("error"):
            return payload
        return None

    def query(prompt, tag, workflow_id):
        try:
            attempts = 0
            while attempts < retries:
                parsed, raw, error, used = _chat_for_json_with_retries(
                    [{"role": "user", "content": prompt}], model=config["model"],
                    response_logger=logger, call_tag=tag, workflow_id=workflow_id, max_retries=retries - attempts)
                attempts += max(1, used)
                parsed = dict(parsed) if isinstance(parsed, dict) else {}
                if error:
                    break
                arrays = ("observations", "supported_rules") if tag == "a4_generic_batch_analysis" else ("updates", "skill_operations")
                if not isinstance(parsed.get("summary"), str) or any(not isinstance(parsed.get(key), list) for key in arrays):
                    error = "invalid_generic_response_schema"
                elif tag != "a4_generic_batch_analysis" and any(not isinstance(item, dict) for key in arrays for item in parsed[key]):
                    error = "generic_edits_must_be_objects"
                if not error:
                    break
            parsed.update(raw_response=raw, error=error, attempts=attempts)
            return parsed
        finally:
            # Native retry calls and failed responses count too. Replaying
            # cached proposals on resume must not zero out their prior cost.
            write(generation_path, merge_recorded_usage(previous_usage, get_usage()))
    reports = []
    for index, batch in enumerate(batches, 1):
        path = out / "batch_reports" / f"batch_{index:04d}.json"
        prompt = (
                "Analyze these training executions under a candidate ToD skill. "
                "Identify useful corrections and behavior supported by successes. "
                "Do not use a graph neighborhood or preassign errors to edit targets. "
                "Return JSON with summary (string), observations (array), and supported_rules (array).\n\n"
                f"<skill>\n{base_skill}\n</skill>\n"
                f"<base_resources>\n{json.dumps(resources, ensure_ascii=False)}\n</base_resources>\n"
                f"<feedback>\n{json.dumps([feedback_view(r) for r in batch], ensure_ascii=False)}\n</feedback>"
            )
        identity = {"manifest_hash": value_digest(manifest), "prompt_hash": value_digest(prompt),
                    "sample_ids": [r["sample"]["sample_id"] for r in batch]}
        report = cached_artifact(path, identity)
        if report is None:
            print(f"A4 generic analysis {index}/{len(batches)}", flush=True)
            report = query(prompt, "a4_generic_batch_analysis",
                           workflow_ids[(index - 1) % len(workflow_ids)] if workflow_ids else None)
            report.update({"batch_id": f"batch_{index:04d}",
                           "sample_ids": [r["sample"]["sample_id"] for r in batch],
                           "identity": identity})
            write(path, report)
        if report.get("error"):
            raise RuntimeError(f"Generic batch analysis failed at {path}: {report['error']}")
        reports.append(report)

    # The global baseline sees every report, the whole K0, all base resources,
    # and the complete edge catalog. No evidence-to-edge target is supplied.
    edge_catalog = [{"edge_id": key, "source": edge["source_action"],
                     "target": edge["target_action"]} for key, edge in state["edges"].items()]
    action_catalog = sorted({str(node.get("label")) for node in state.get("nodes", [])})
    applied_ids: set[str] = set()
    working_skill = base_skill
    source_group_count = len(list((paths["full"] / "group_reflections").glob("reflection_*.json")))
    if not source_group_count:
        raise ValueError("A2 full is missing graph-local reflection groups")
    # Match the number of global rewrite decisions to the source local
    # updater, while preserving all batch reports in their frozen order.
    group_count = min(source_group_count, len(reports))
    groups = [reports[(i * len(reports)) // group_count:
                      ((i + 1) * len(reports)) // group_count]
              for i in range(group_count)]
    for group_index, report_group in enumerate(groups, 1):
        group = [{key: report.get(key) for key in
                  ("batch_id", "sample_ids", "summary", "observations", "supported_rules")}
                 for report in report_group]
        path = out / "group_reflections" / f"reflection_{group_index:04d}.json"
        current_resources = {"reference": resources["reference"] + "\n" + render_online_resources(state)[1],
                             "action_rules": resources["action_rules"] + "\n" + render_online_action_rules(state),
                             "slot_policies": resources["slot_policies"] + "\n" + render_online_slot_policies(state)}
        prompt = (
                "Globally revise this ToD skill from the training feedback summaries. "
                "You may edit the executable skill, reference notes, action rules, "
                "slot policies, and transition guards. Preserve behavior supported "
                "by successful executions. Review the whole artifact; no local "
                "target edge or action/slot error category has been selected for you. "
                "For text edits, copy match_text exactly from current_skill. "
                "Return JSON: {\"summary\":string, \"updates\":[{\"resource\":\"transition_guard|action_node|action_rule|slot_policy|reference\",\"op\":\"upsert|delete\",\"edge_id\":string,\"action\":string,\"content\":string,\"status\":\"resolved|uncertain\"}], "
                "\"skill_operations\":[{\"op\":\"replace|insert_before|insert_after|delete\",\"match_text\":string,\"new_text\":string,\"rationale\":string}]}. "
                "Use [] for a field when no supported edit is needed. Use edge_id "
                "only from edge_catalog. Never invent actions, edges, or policy facts.\n\n"
                f"<current_skill>\n{working_skill}\n</current_skill>\n"
                f"<current_resources>\n{json.dumps(current_resources, ensure_ascii=False)}\n</current_resources>\n"
                f"<edge_catalog>\n{json.dumps(edge_catalog, ensure_ascii=False)}\n</edge_catalog>\n"
                f"<action_catalog>\n{json.dumps(action_catalog, ensure_ascii=False)}\n</action_catalog>\n"
                f"<training_summaries>\n{json.dumps(group, ensure_ascii=False)}\n</training_summaries>"
            )
        identity = {"manifest_hash": value_digest(manifest), "prompt_hash": value_digest(prompt),
                    "batch_ids": [r["batch_id"] for r in group]}
        reflection = cached_artifact(path, identity)
        if reflection is None:
            print(f"A4 generic rewrite {group_index}/{len(groups)}", flush=True)
            reflection = query(prompt, "a4_generic_global_rewrite",
                               workflow_ids[(group_index - 1) % len(workflow_ids)] if workflow_ids else None)
            reflection.update(identity=identity, batch_ids=identity["batch_ids"])
        if reflection.get("error"):
            write(path, reflection)
            raise RuntimeError(f"Generic global rewrite failed at {path}: {reflection['error']}")
        updates, operations = reflection.get("updates", []), reflection.get("skill_operations", [])
        if not isinstance(updates, list) or not isinstance(operations, list):
            raise ValueError("Generic updater returned invalid edit arrays")
        accepted, rejected = apply_reflection_updates_to_state(state, updates)
        working_skill, applied = apply_dynamic_skill_operations(working_skill, operations, applied_ids)
        reflection.update({"accepted_updates": accepted, "rejected_updates": rejected,
                           "applied_skill_operations": applied})
        write(path, reflection)
        # Match the native updater: preserve failed edit diagnostics while
        # keeping valid edits and continuing with later groups.
    working_skill = _checkpoint(out, state, working_skill, resources["reference"],
                                base_slot_policies=resources["slot_policies"],
                                base_action_rules=resources["action_rules"])
    usage = merge_recorded_usage(previous_usage, get_usage())
    write(generation_path, usage)
    write(out / "llm_usage.json", {"generation": usage})
    return working_skill, state, {"generation": usage}


def correctness(predictions: Path, test: list[dict]) -> dict[tuple[str, int], bool]:
    from eval_tod.abcd.metrics import slot_match_profile
    from skill_mining.online_refinement import build_action_turn_samples
    gold = {(str(s["conversation_id"]), int(s["turn_index"])): s
            for s in build_action_turn_samples(test)}
    rows = read(predictions)
    seen: dict[tuple[str, int], bool] = {}
    for row in rows:
        if row.get("target_type") != "action":
            continue
        key = (str(row.get("convo_id")), int(row.get("turn_index", -1)))
        if key not in gold or key in seen:
            raise ValueError(f"Invalid or duplicated held-out prediction: {key}")
        sample = gold[key]
        seen[key] = (str(row.get("predicted_action", "")) == str(sample["target_action"])
                     and bool(slot_match_profile(sample.get("gold_slots", []),
                                                 row.get("predicted_slots", []))["strict_ordered"]))
    if set(seen) != set(gold):
        raise ValueError(f"Held-out predictions cover {len(seen)}/{len(gold)} action turns")
    return seen


def paired(before: dict, after: dict) -> dict:
    if set(before) != set(after):
        raise ValueError("Paired evaluation keys differ")
    fixed = sum(not before[k] and after[k] for k in before)
    regressed = sum(before[k] and not after[k] for k in before)
    wrong = sum(not v for v in before.values())
    correct = len(before) - wrong
    return {"action_turns": len(before), "k0_wrong": wrong, "k0_correct": correct,
            "fixed": fixed, "regressed": regressed,
            "fix_rate": fixed / wrong if wrong else None,
            "regression_rate": regressed / correct if correct else None,
            "net_gain": (fixed - regressed) / len(before) if before else None}


def run_flow(args, flow: str, workflow_ids: list[str]) -> dict:
    from awm import WorkflowStore
    from eval_tod.response_logger import ResponseLogger
    from scripts.abcd_runtime_utils import build_abcd_runtime_agent
    from scripts.llm_usage_utils import get_usage, reset_usage, split_usage_summary
    from scripts.run_subflow_eval import evaluate_agent_on_subflow
    from skill_mining.online_refinement import load_skill_dag, render_online_action_rules, render_online_resources, render_online_slot_policies

    config, records, context = validate_source(Path(args.a2_root), flow)
    paths = context["paths"]
    out = Path(args.output_dir) / flow
    generic_dir = out / "generic"
    generic_path = generic_dir / "a4_generic_result.json"
    # Always validate the manifest, including completed results. A matching
    # filename alone never authorizes reusing another source run's K1.
    skill, state, usage = generic_update(config, records, paths, generic_dir, workflow_ids,
                                         retries=args.retries, resume=args.resume)
    manifest_hash = value_digest(read(generic_dir / "a4_manifest.json"))
    if not generic_path.is_file():
        testing_path = generic_dir / "testing_attempt_usage.json"
        previous_testing = read(testing_path) if args.resume and testing_path.is_file() else None
        reset_usage()
        _, online_reference = render_online_resources(state)
        offline = paths["offline"]
        base = {key: (offline / f"{key}.md").read_text(encoding="utf-8")
                if (offline / f"{key}.md").is_file() else ""
                for key in ("reference", "action_rules", "slot_policies")}
        workflow = WorkflowStore()
        workflow.update(skill)
        runtime_args = SimpleNamespace(**config)
        agent = build_abcd_runtime_agent(
            runtime_args, workflow=workflow,
            reference_text=base["reference"].rstrip() + "\n\n" + online_reference,
            action_rules_text=base["action_rules"].rstrip() + "\n\n" + render_online_action_rules(state),
            slot_policies_text=base["slot_policies"].rstrip() + "\n\n" + render_online_slot_policies(state),
            response_logger=ResponseLogger(generic_dir / "test_llm_responses"),
            workflow_id=workflow_ids[0] if workflow_ids else None,
        )
        parallel_ids = workflow_ids if len(workflow_ids) > 1 and "fork" in mp.get_all_start_methods() else None
        print(f"A4 {flow}: generic K1 frozen evaluation started", flush=True)
        try:
            result = evaluate_agent_on_subflow(
                agent, context["test"], "a4_generic", flow, save_dir=generic_dir,
                skip_utterance_eval=bool(config.get("skip_utterance_eval")), eval_workflow_ids=parallel_ids)
        except Exception:
            partial = merge_recorded_usage(previous_testing, get_usage())
            if parallel_ids or (previous_testing or {}).get("incomplete_parallel_accounting"):
                partial["incomplete_parallel_accounting"] = True
                partial["calls_available"] = False
                partial["total"]["usage_available"] = False
                partial["note"] = "Unreturned failed test-worker usage is incomplete; known counts are lower bounds"
            write(testing_path, partial)
            raise
        worker_usage = result.pop("_evaluation_worker_usage", [])
        testing = merge_recorded_usage(previous_testing, get_usage(), *worker_usage)
        if (previous_testing or {}).get("incomplete_parallel_accounting"):
            testing["incomplete_parallel_accounting"] = True
            testing["calls_available"] = False
            testing["total"]["usage_available"] = False
        write(testing_path, testing)
        result["llm_usage"] = split_usage_summary(usage["generation"], testing)
        result["runtime_protocol"] = abcd_runtime_options(runtime_args)
        result["manifest_hash"] = manifest_hash
        result["artifact_hashes"] = {name: digest(generic_dir / name) for name in
                                     ("skill.md", "skill_dag_state.json", "reference.md", "action_rules.md",
                                      "slot_policies.md", "a4_generic_predictions.json")}
        write(generic_dir / "llm_usage.json", result["llm_usage"])
        write(generic_path, result)
    k0 = correctness(paths["k0_predictions"], context["test"])
    ours = correctness(paths["full"] / "online_refined_predictions.json", context["test"])
    generic = correctness(generic_dir / "a4_generic_predictions.json", context["test"])
    generic_ast = read(generic_path)["ast_cds"]
    if (generic_ast["num_action_turns"] != len(generic)
            or abs(float(generic_ast["ast_joint"]) - sum(generic.values()) / max(len(generic), 1)) > 0.000051):
        raise ValueError("Generic reported Joint AST/count differs from held-out predictions")
    results = {"subflow": flow,
               "k0_sha256": digest(paths["offline"] / "skill.md"),
               "feedback_sha256": digest(paths["full"] / "post_rollout_records.json"),
               "feedback_samples": len(records),
               "frozen_k0": read(paths["k0_result"])["ast_cds"],
               "graph_local": read(paths["full"] / "online_refine_result.json")["ast_cds"],
               "generic_global": read(generic_path)["ast_cds"],
               "graph_local_vs_k0": paired(k0, ours),
               "generic_global_vs_k0": paired(k0, generic),
               "graph_local_vs_generic": paired(generic, ours),
               "resource_lineage": context["resource_lineage"],
               "usage": {
                   "frozen_k0": read(paths["k0_result"]).get("llm_usage"),
                   "graph_local": read(paths["full"] / "online_refine_result.json").get("llm_usage"),
                   "generic_global": read(generic_path).get("llm_usage"),
               },
               "source_a2": str(Path(args.a2_root) / flow),
               "generic_result": str(generic_path)}
    write(out / "a4_result.json", results)
    return results


def workflow_buckets(flows: list[str], ids: list[str]) -> dict[str, list[str]]:
    if len(flows) == 1 or not ids:
        return {",".join(ids): flows}
    buckets = {wid: [] for wid in ids}
    for index, flow in enumerate(flows):
        buckets[ids[index % len(ids)]].append(flow)
    return {wid: names for wid, names in buckets.items() if names}


def aggregate_results(root: Path, flows: list[str], failures: list[dict] | None = None) -> dict:
    """Aggregate completed flows only; ratios use pooled counts, not mean rates."""
    failed = {item["subflow"] for item in failures or []}
    rows = [read(root / flow / "a4_result.json") for flow in flows
            if flow not in failed and (root / flow / "a4_result.json").is_file()]
    aggregate, comparisons, cost = {}, {}, {}
    for arm in ("frozen_k0", "generic_global", "graph_local"):
        count = sum(row[arm]["num_action_turns"] for row in rows)
        aggregate[arm] = {"subflows": len(rows), "action_turns": count, **{
            key: sum(row[arm][key] * row[arm]["num_action_turns"] for row in rows) / count
            if count else None for key in ("ast_joint", "ast_action_name", "ast_slot_value")}}
        cost[arm] = {}
        for phase in ("generation", "testing", "updater"):
            buckets = []
            for row in rows:
                usage = row.get("usage", {}).get(arm) or {}
                snapshot = usage.get("generation" if phase == "updater" else phase) or {}
                if phase == "updater" and arm == "frozen_k0":
                    buckets.append({"calls": 0, "total_tokens": 0, "usage_available": True})
                elif phase == "updater" and arm == "graph_local":
                    tags = snapshot.get("by_call_tag", {})
                    selected = [tags[tag] for tag in ("online_batch_root_cause", "online_group_reflection") if tag in tags]
                    # Missing tag accounting is unknown, never zero updater cost.
                    buckets.append({"calls": sum(x.get("calls", 0) for x in selected),
                                    "total_tokens": sum(x.get("total_tokens", 0) for x in selected),
                                    "usage_available": len(selected) == 2 and all(x.get("usage_available") for x in selected)}
                                   if len(selected) == 2 else {})
                else:
                    bucket = dict(snapshot.get("total", {}))
                    if snapshot.get("calls_available") is False:
                        bucket.pop("calls", None)
                    buckets.append(bucket)
            known_calls = [bucket["calls"] for bucket in buckets if "calls" in bucket]
            tokens = [bucket["total_tokens"] for bucket in buckets if bucket.get("usage_available") and "total_tokens" in bucket]
            cost[arm][phase] = {"calls": sum(known_calls) if rows and len(known_calls) == len(rows) else None,
                                "total_tokens": sum(tokens) if rows and len(tokens) == len(rows) else None,
                                "known_calls": sum(known_calls), "known_tokens": sum(tokens),
                                "flows_with_calls": len(known_calls), "flows_with_tokens": len(tokens)}
    for key in ("graph_local_vs_k0", "generic_global_vs_k0", "graph_local_vs_generic"):
        counts = {field: sum(row[key][field] for row in rows) for field in
                  ("action_turns", "k0_wrong", "k0_correct", "fixed", "regressed")}
        comparisons[key] = {**counts,
                            "fix_rate": counts["fixed"] / counts["k0_wrong"] if counts["k0_wrong"] else None,
                            "regression_rate": counts["regressed"] / counts["k0_correct"] if counts["k0_correct"] else None,
                            "net_gain": (counts["fixed"] - counts["regressed"]) / counts["action_turns"] if counts["action_turns"] else None}
    completed = {row["subflow"] for row in rows}
    return {"protocol": PROTOCOL, "complete": len(rows) == len(flows),
            "expected_subflows": flows, "completed_subflows": sorted(completed),
            "missing_subflows": [flow for flow in flows if flow not in completed],
            "failures": failures or [], "rows": rows, "aggregate": aggregate,
            "paired": comparisons, "cost": cost,
            "cost_note": "K0/graph_local source costs are reused, not new A4 charges. Graph-local generation includes training rollout; updater uses diagnosis/reflection tags only. Unknown costs are null."}


def bootstrap_command(args, ids: list[str], flows: list[str]) -> list[str]:
    """Preserve an existing A2 root's experiment configuration on resume."""
    prior_path = Path(args.a2_root) / "a2_config.json"
    prior = read(prior_path).get("arguments", {}) if prior_path.is_file() else {}
    if prior and prior.get("refinement_mode") != "standard":
        raise ValueError("A4 bootstrap cannot reuse a non-standard A2 root")
    for key in ("max_train", "max_test", "max_batches"):
        if prior and getattr(args, key) is not None and getattr(args, key) != prior.get(key):
            raise ValueError(f"A4 bootstrap scope differs from existing A2: {key}")
    arms = prior.get("arms") or ["offline_only", "full"]
    cmd = [sys.executable, str(ROOT / "scripts" / "run_a2_abcd.py"),
           "--subflows", ",".join(flows), "--workflow-ids", ",".join(ids),
           "--output-dir", args.a2_root, "--arms", ",".join(arms), "--refinement-mode", "standard"]
    added = [arm for arm in ("offline_only", "full") if arm not in arms]
    skipped = [arm for arm in arms if arm not in ("offline_only", "full")]
    if added:
        cmd += ["--add-arms", ",".join(added)]
    if skipped:
        cmd += ["--skip-arms", ",".join(skipped)]
    for key in ("model", "batch_size", "max_train", "max_test", "max_batches",
                "reference_top_k", "reference_max_chars", "action_selection_candidate_limit"):
        value = prior.get(key) if prior else getattr(args, key, None)
        if value is not None:
            cmd += ["--" + key.replace("_", "-"), str(value)]
    for key in ("skip_utterance_eval", "disable_action_selection_cards"):
        if prior.get(key) if prior else getattr(args, key, False):
            cmd += ["--" + key.replace("_", "-")]
    if prior:
        cmd += ["--resume"]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subflow")
    group.add_argument("--subflows")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--a2-root", required=True, help="A2 offline_only/full result root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workflow-ids", default=os.environ.get("SKILLMINING_WORKFLOW_ID", ""))
    parser.add_argument("--bootstrap-a2", action="store_true", help="Run missing A2 offline_only/full arms first")
    parser.add_argument("--max-train", type=int, help="Only for --bootstrap-a2")
    parser.add_argument("--max-test", type=int, help="Only for --bootstrap-a2")
    parser.add_argument("--max-batches", type=int, help="Only for --bootstrap-a2")
    parser.add_argument("--batch-size", type=int, default=8, help="Only for --bootstrap-a2")
    parser.add_argument("--model", default="deepseek-chat", help="Only for --bootstrap-a2")
    parser.add_argument("--skip-utterance-eval", action="store_true", help="Only for --bootstrap-a2")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Audit completed A2 sources without model calls or writes")
    parser.add_argument("--worker-subflow", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.retries <= 0 or any(value is not None and value <= 0 for value in
                                (args.max_train, args.max_test, args.max_batches, args.batch_size)):
        parser.error("Limits and retries must be positive")
    args.a2_root = str(Path(args.a2_root).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    source_root, output_root = Path(args.a2_root), Path(args.output_dir)
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        parser.error("--a2-root and --output-dir must be separate, non-nested directories")
    flows = selected_flows(args)
    ids = _ids(args.workflow_ids) if args.workflow_ids else []
    if not ids:
        import config as project_config
        llm_config = getattr(project_config, "LLM_CONFIG", {})
        candidate = str(llm_config.get("workflow_id", ""))
        if llm_config.get("provider") == "workflow" and candidate and set(candidate.lower()) != {"x"}:
            ids = [candidate]
    if any(set(wid.lower()) == {"x"} for wid in ids):
        parser.error("--workflow-ids contains a placeholder, not a configured endpoint")
    if args.worker_subflow:
        if args.worker_subflow not in flows:
            parser.error("Worker subflow must match the selected subflow")
        run_flow(args, args.worker_subflow, ids)
        return 0
    missing = [flow for flow in flows if any(not path.is_file() for path in (
        Path(args.a2_root) / flow / "full" / "online_refine_result.json",
        Path(args.a2_root) / flow / "full" / "post_rollout_records.json",
        Path(args.a2_root) / flow / "offline_only" / "result.json",
        Path(args.a2_root) / flow / "offline_artifact" / "complete.json",
    ))]
    plan = {"flows": flows, "workflow_ids": ids, "a2_root": args.a2_root,
            "output_dir": args.output_dir, "missing_a2": missing,
            "bootstrap_a2": args.bootstrap_a2, "assignments": workflow_buckets(flows, ids)}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    if args.validate_only:
        for flow in flows:
            _, records, context = validate_source(source_root, flow)
            print(json.dumps({"subflow": flow, "feedback_samples": len(records),
                              "resource_lineage": context["resource_lineage"]}), flush=True)
        return 0
    config_path = output_root / "a4_config.json"
    experiment_config = {"protocol": PROTOCOL, "a2_root": args.a2_root, "flows": flows, "retries": args.retries}
    if output_root.exists() and any(output_root.iterdir()):
        if not args.resume:
            parser.error("Output directory is nonempty; use --resume or a fresh --output-dir")
        if not config_path.is_file() or read(config_path) != experiment_config:
            parser.error("A4 resume configuration differs or provenance is missing; use a fresh output directory")
    if missing:
        if not args.bootstrap_a2:
            raise FileNotFoundError("A2 full results missing; provide --bootstrap-a2 or an existing --a2-root")
        cmd = bootstrap_command(args, ids, flows)
        subprocess.run(cmd, cwd=ROOT, check=True)
    # Validate every selected source before paying for any generic updates.
    for flow in flows:
        validate_source(source_root, flow)
    if not config_path.is_file():
        write(config_path, experiment_config)
    # Distinct workflow IDs are serialized per worker, as in A2.
    buckets = workflow_buckets(flows, ids)
    def worker(wid: str, names: list[str]):
        failures = []
        for flow in names:
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--worker-subflow", flow, "--subflow", flow,
                   "--a2-root", args.a2_root, "--output-dir", args.output_dir,
                   "--workflow-ids", wid, "--retries", str(args.retries)]
            if args.resume:
                cmd.append("--resume")
            output = Path(args.output_dir) / flow
            output.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if wid:
                env["SKILLMINING_WORKFLOW_ID"] = wid.split(",")[0]
            try:
                with (output / "a4_worker.log").open("a", encoding="utf-8") as log:
                    log.write(f"\nRUN {datetime.now(timezone.utc).isoformat()} COMMAND: {subprocess.list2cmdline(cmd)}\n")
                    log.flush()
                    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, check=False)
                exit_code = proc.returncode
            except OSError as exc:
                failures.append({"subflow": flow, "error": str(exc), "log": str(output / "a4_worker.log")})
                print(f"FAILED to start A4 {flow}: {exc}", flush=True)
                continue
            if exit_code:
                failures.append({"subflow": flow, "exit_code": exit_code, "log": str(output / "a4_worker.log")})
                print(f"FAILED A4 {flow}; see {output / 'a4_worker.log'}", flush=True)
            else:
                print(f"Completed A4 {flow}", flush=True)
        return failures
    failures = []
    with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
        futures = [pool.submit(worker, wid, names) for wid, names in buckets.items() if names]
        for future in as_completed(futures):
            failures.extend(future.result())
    report = aggregate_results(output_root, flows, failures)
    write(output_root / "a4_aggregate.json", report)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
