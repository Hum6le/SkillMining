#!/usr/bin/env python3
"""Frozen ABCD K1: full-context flat graph vs flat ReAct vs organized ReAct.

Default: prepare flat resources and evaluate the two controls, NOT Ours. Never
mine, refine, repair, or write to the supplied K1 directory. Each subflow runs
in its own process; multiple workflows shard evaluation *within* a subflow.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from scripts.run_a2_abcd import _ids, _subflows
from scripts.abcd_runtime_utils import abcd_runtime_options
from skill_mining.k1_flatten import (RESOURCE_NAMES, VERSION, digest, flatten_resources,
                                    graph_flat_progressive_resources)

ARMS = ("flat_full_context", "flat_react", "ours")
PROTOCOL = "abcd-k1-context-v3"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_directory(args, flow):
    if args.k1_dir:
        return Path(args.k1_dir).resolve()
    base = Path(args.source_root).resolve() / flow
    # A2 root/<flow>/full, or full-run root/<flow>.
    return base / "full" if (base / "full").is_dir() else base


def load_source(directory, flow):
    missing = [name for name in RESOURCE_NAMES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"K1 resource bundle incomplete at {directory}: {missing}")
    resources = {name: (directory / name).read_text(encoding="utf-8") for name in RESOURCE_NAMES}
    working = directory / "working_skill.md"
    if working.is_file() and working.read_text(encoding="utf-8") != resources["skill.md"]:
        raise ValueError("working_skill.md differs from skill.md; finish/checkpoint the source run first")
    config_path, state_path = directory / "runner_config.json", directory / "skill_dag_state.json"
    config = read(config_path).get("arguments", {}) if config_path.is_file() else {}
    if config.get("subflow") and config["subflow"] != flow:
        raise ValueError("Source runner_config subflow differs")
    state = read(state_path) if state_path.is_file() else {}
    failures = {key: state[key] for key in ("posthoc_failed_batch_ids", "posthoc_failed_reflection_groups") if state.get(key)}
    result_path = directory / "online_refine_result.json"
    final = read(result_path) if result_path.is_file() else None
    verified = bool(config_path.is_file() and state_path.is_file() and final is not None)
    if config.get("refinement_mode") == "standard" and not state.get("posthoc_reflection_complete"):
        verified = False
    return resources, state, config, {
        "source": str(directory), "source_hashes": {n: digest(resources[n]) for n in RESOURCE_NAMES},
        "state_hash": digest(state) if state_path.is_file() else None,
        "runner_config_hash": digest(config) if config_path.is_file() else None,
        "completion_verified": verified and not failures, "source_failures": failures,
        "source_result_hash": digest(final) if final is not None else None,
        "source_result": str(result_path) if result_path.is_file() else None,
        "note": "K1 exported resource files are the authority; state is used only for node aliases.",
    }


def runtime_args(args, source_config):
    values = {"model": args.model or source_config.get("model", "deepseek-chat")}
    for name, default in (("reference_top_k", 3), ("reference_max_chars", 1800),
                          ("action_selection_candidate_limit", 3), ("disable_competitive_action_cards", False)):
        values[name] = source_config.get(name, default)
    # Frozen evaluation overrides, not modifications to source/training settings.
    values["reference_top_k"] = getattr(args, "reference_top_k", 10)
    values["action_selection_candidate_limit"] = getattr(args, "action_selection_candidate_limit", 8)
    for key in ("action_card_max_chars", "reference_max_chars", "reference_section_max_chars"):
        values[key] = getattr(args, key, 0) or None
    if values["disable_competitive_action_cards"]:
        raise ValueError("This experiment requires two-stage action/card inference; source disabled competitive cards")
    return SimpleNamespace(**values)


def prepare(directory, output, flow):
    resources, state, config, provenance = load_source(directory, flow)
    flat = flatten_resources(resources, state)
    progressive = graph_flat_progressive_resources(resources, flat)
    output.mkdir(parents=True, exist_ok=True)
    for name, text in flat["resources"].items():
        (output / ("flat_" + name)).write_text(text, encoding="utf-8")
    (output / "flat_graph.txt").write_text(flat["full_context"], encoding="utf-8")
    write(output / "flat_graph.json", flat["records"])
    write(output / "flatten_audit.json", flat["audit"])
    write(output / "source_manifest.json", provenance)
    progressive_dir = output / "flat_react_resources"
    progressive_dir.mkdir(parents=True, exist_ok=True)
    for name, text in progressive["resources"].items():
        if name == "skill.md":
            (progressive_dir / name).write_text(text, encoding="utf-8")
        else:
            shutil.copyfile(directory / name, progressive_dir / name)
    for name, text in progressive["inline_policies"].items():
        (progressive_dir / ("inline_" + name)).write_text(text, encoding="utf-8")
    write(progressive_dir / "projection_audit.json", progressive["audit"])
    counts = {key: len(value) for key, value in flat["records"].items()}
    print(f"{flow}: flattened {counts}; chars={len(flat['full_context'])}; "
          f"scoped-rule warnings={len(flat['audit']['warnings'])}", flush=True)
    return resources, flat, config, provenance


def create_agent(arm, runtime, original, flat, workflow_id=None, logger=None):
    from awm import MemoryStore, WorkflowStore
    from eval_tod.abcd.agent import ABCDAgent
    from eval_tod.abcd.k1_context_agent import K1FullContextAgent
    progressive = graph_flat_progressive_resources(original, flat) if arm == "flat_react" else None
    selected = (original if arm == "ours" else progressive["resources"] if progressive else flat["resources"])
    workflow = WorkflowStore()
    workflow.replace(flat["full_context"] if arm == "flat_full_context" else selected["skill.md"])
    constructor = K1FullContextAgent if arm == "flat_full_context" else ABCDAgent
    agent = constructor(**abcd_runtime_options(runtime), workflow=workflow, memory=MemoryStore(),
        reference_text="" if arm == "flat_full_context" else selected["reference.md"],
        action_rules_text=selected["action_rules.md"], slot_policies_text=selected["slot_policies.md"],
        workflow_id=workflow_id, response_logger=logger)
    if progressive:
        from eval_tod.abcd.agent import _parse_action_policy_sections
        # Retain rules unique to original skill.md, but expose them only through
        # the exact same bounded action-card mechanism, not as flattened cards.
        for role, name in (("action_rule", "action_rules.md"), ("slot_policy", "slot_policies.md")):
            for row in _parse_action_policy_sections(progressive["inline_policies"][name]):
                card = agent.action_cards.setdefault(row["action"], {})
                previous = card.get(role, "")
                if row["body"] not in previous:
                    card[role] = (previous + "\n\n" + row["body"]).strip()
    return agent


def target_keys(test):
    return {(str(c.get("convo_id", "?")), i) for c in test for i, t in enumerate(c.get("delexed", []))
            if len(t.get("targets", [])) >= 3 and t["targets"][1] == "take_action"}


def reuse_ours(directory, test, runtime, provenance, skip_utterance_eval=True):
    """Conservative reuse: verify exact action-target set and observable inputs.

    Older exports have no historical resource hashes. We explicitly disclose
    that limitation instead of presenting a fresh matched-runtime run.
    """
    if not provenance.get("completion_verified"):
        return None, "source training completion unverified; Ours not reused or run"
    result_path, predictions_path = directory / "online_refine_result.json", directory / "online_refined_predictions.json"
    if not result_path.is_file() or not predictions_path.is_file():
        return None, "source result/predictions unavailable; Ours not run"
    result = read(result_path)
    if result.get("runtime_protocol") != abcd_runtime_options(runtime):
        return None, "source runtime/model metadata differs or is missing; Ours not run"
    rows = read(predictions_path)
    action_rows = [r for r in rows if r.get("target_type") == "action"]
    keys = [(str(r.get("convo_id")), r.get("turn_index")) for r in action_rows]
    expected = target_keys(test)
    if len(keys) != len(set(keys)) or set(keys) != expected or result["ast_cds"]["num_action_turns"] != len(expected):
        return None, "source evaluation target set differs; Ours not run"
    from eval_tod.abcd.agent import _get_original_turn
    by_id = {str(c.get("convo_id", "?")): c for c in test}
    for row in action_rows:
        conv = by_id[str(row["convo_id"])]
        lines = []
        for i, turn in enumerate(conv["delexed"][:row["turn_index"]]):
            speaker, text = _get_original_turn(conv, i, turn)
            if text:
                lines.append(f"[{ {'agent': 'Agent', 'customer': 'Customer', 'action': 'System'}.get(speaker, speaker)}] {text}")
        if row.get("context") != "\n".join(lines):
            return None, "source prediction dialogue inputs differ; Ours not run"
    # Recompute AST using current ground truth, rather than trusting summary.
    from eval_tod.abcd.agent import turn_results_to_abcd_predictions
    from eval_tod.abcd.data import extract_ground_truth
    from eval_tod.abcd.metrics import evaluate_abcd
    scored = evaluate_abcd([extract_ground_truth(c) for c in test], turn_results_to_abcd_predictions(rows, test))
    for key, value in (("ast_joint", scored.ast.joint_accuracy), ("ast_action_name", scored.ast.action_name_accuracy),
                       ("ast_slot_value", scored.ast.slot_value_accuracy)):
        if abs(result["ast_cds"][key] - value) > .000051:
            return None, "source summary does not match predictions/current gold; Ours not run"
    reused = dict(result)
    reused.update(protocol=PROTOCOL, arm="ours", reused=True, source_manifest=provenance,
                  evaluation_test_hash=digest(test),
                  reuse_audit={"runtime_verified": True, "target_set_verified": True, "dialogue_inputs_verified": True,
                               "metrics_recomputed": True, "historical_resource_hashes_verified": False,
                               "limitation": "Legacy source results do not bind predictions to exported resource hashes; audit source provenance before formal comparison."})
    if skip_utterance_eval and any(r.get("target_type") == "utterance" for r in rows):
        reused["source_llm_usage"] = reused.pop("llm_usage", None)
        reused["reuse_audit"]["testing_cost_comparable"] = False
        reused["reuse_audit"]["cost_limitation"] = "Historical usage includes utterance evaluation; not comparable to action-only controls."
    return reused, "source result reused; no Ours inference calls (historical resource hashes unverified)"


def run_flow(args, flow, workflow_ids):
    directory = source_directory(args, flow)
    output = Path(args.output_dir) / flow
    # Protect user/source artifacts, including ancestor/descendant overlap.
    if directory == output.resolve() or directory in output.resolve().parents or output.resolve() in directory.parents:
        raise ValueError("Source and experiment output directories must not overlap")
    os.environ["ABCD_OUTPUT_DIR"] = str(output / "eval_aux")
    if workflow_ids:
        os.environ["SKILLMINING_WORKFLOW_ID"] = workflow_ids[0]
    original, flat, config, provenance = prepare(directory, output, flow)
    if args.prepare_only:
        return
    if not provenance["completion_verified"] and not args.allow_unverified_source:
        raise ValueError("K1 completion is unverified/has failures; inspect source_manifest.json. "
                         "Use --allow-unverified-source only for explicitly labelled diagnostic evaluation.")
    from scripts.run_subflow_eval import load_subflow_data, evaluate_agent_on_subflow
    from scripts.llm_usage_utils import get_usage, reset_usage, merge_usage_summaries
    from eval_tod.response_logger import ResponseLogger
    _, test = load_subflow_data(flow)
    maximum = (None if getattr(args, "full_test", False) else
               args.max_test if args.max_test is not None else config.get("max_test"))
    if maximum is not None:
        test = test[:maximum]
    runtime = runtime_args(args, config)
    progressive = graph_flat_progressive_resources(original, flat)
    write(output / "evaluation_scope.json", {"test_hash": digest(test), "source_manifest": provenance,
          "runtime": abcd_runtime_options(runtime), "skip_utterance_eval": args.skip_utterance_eval})
    failures = []
    for arm in ARMS:
        arm_dir = output / arm
        if arm in args.skip_arms:
            if arm == "ours":
                if (arm_dir / "result.json").is_file() and not read(arm_dir / "result.json").get("reused"):
                    print(f"{flow}/ours: existing locally evaluated Ours result retained", flush=True)
                    continue
                reused, reason = reuse_ours(directory, test, runtime, provenance, args.skip_utterance_eval)
                write(output / "ours_reuse_status.json", {"reused": reused is not None, "reason": reason})
                if reused is not None:
                    write(arm_dir / "result.json", reused)
                    write(arm_dir / "status.json", {"status": "complete", "reused": True})
                print(f"{flow}/ours: {reason}", flush=True)
            continue
        manifest = {"protocol": PROTOCOL, "flatten_version": VERSION, "arm": arm,
                    "source_manifest": provenance, "test_hash": digest(test),
                    "runtime": abcd_runtime_options(runtime), "skip_utterance_eval": args.skip_utterance_eval,
                    "flat_records_hash": flat["audit"]["records_hash"]}
        if arm == "flat_react":
            manifest["progressive_projection_hash"] = digest(progressive["audit"])
        if (arm_dir / "result.json").is_file():
            previous = read(arm_dir / "result.json")
            replace_reused = arm == "ours" and previous.get("reused")
            if replace_reused:
                write(arm_dir / "source_reused_result.json", previous)
            elif args.resume and (arm_dir / "manifest.json").is_file() and read(arm_dir / "manifest.json") == manifest:
                print(f"{flow}/{arm}: completed result reused", flush=True)
                continue
            else:
                raise ValueError(f"Result already exists with missing/different manifest: {arm_dir}; use a fresh output directory")
        arm_dir.mkdir(parents=True, exist_ok=True)
        write(arm_dir / "manifest.json", manifest)
        write(arm_dir / "status.json", {"status": "running", "manifest": manifest})
        attempt_dir = arm_dir / "attempts" / uuid4().hex
        attempt_dir.mkdir(parents=True)
        reset_usage()
        agent = create_agent(arm, runtime, original, flat, workflow_ids[0] if workflow_ids else None,
                             ResponseLogger(attempt_dir / "llm_responses"))
        parallel = workflow_ids if len(workflow_ids) > 1 and "fork" in mp.get_all_start_methods() else None
        if len(workflow_ids) > 1 and parallel is None:
            print("WARNING: no fork support; evaluation uses the first workflow serially", flush=True)
        try:
            print(f"{flow}/{arm}: frozen evaluation ({len(test)} dialogues)", flush=True)
            result = evaluate_agent_on_subflow(agent, test, arm, flow, save_dir=attempt_dir,
                eval_workflow_ids=parallel, skip_utterance_eval=args.skip_utterance_eval)
            testing = merge_usage_summaries(get_usage(), *result.pop("_evaluation_worker_usage", []))
            result.update(protocol=PROTOCOL, subflow=flow, arm=arm, reused=False,
                evaluation_test_hash=digest(test),
                evaluation_artifacts=str(attempt_dir),
                test_dialogues=len(test), full_test=getattr(args, "full_test", False),
                source_manifest=provenance, runtime_protocol=abcd_runtime_options(runtime),
                resource_access_mode=arm, online_refinement=False, initialization="frozen K1",
                llm_usage={"testing": testing, "note": "New evaluation cost only; shared K1 training cost not charged again."},
                flatten_records_hash=flat["audit"]["records_hash"],
                projected_knowledge_counts={key: len(value) for key, value in flat["records"].items()},
                flat_full_context_chars=len(flat["full_context"]) if arm == "flat_full_context" else None)
            if arm == "flat_react":
                result["progressive_projection"] = progressive["audit"]
            write(arm_dir / "result.json", result)
            write(arm_dir / "status.json", {"status": "complete", "manifest": manifest})
            ast = result["ast_cds"]
            print(f"RESULT {flow}/{arm}: joint={ast['ast_joint']:.4f} "
                  f"action={ast['ast_action_name']:.4f} slot={ast['ast_slot_value']:.4f}", flush=True)
        except Exception as exc:
            error = {"error": str(exc), "traceback": traceback.format_exc(),
                     "llm_usage_parent_only": get_usage(), "manifest": manifest,
                     "evaluation_artifacts": str(attempt_dir)}
            write(arm_dir / "error.json", error)
            write(attempt_dir / "error.json", error)
            write(arm_dir / "status.json", {"status": "error", "error": str(exc), "manifest": manifest})
            failures.append(arm)
            print(f"ERROR {flow}/{arm}: {exc}; continuing other arms", flush=True)
    if failures:
        raise RuntimeError("Some evaluations failed; saved results remain available: " + ", ".join(failures))


def aggregate(output, flows):
    rows, summary = [], {}
    for arm in ARMS:
        group = []
        for flow in flows:
            path = output / flow / arm / "result.json"
            if path.is_file():
                row = read(path)
                if row.get("protocol") and row["protocol"] != PROTOCOL:
                    print(f"Old protocol excluded: {path}", flush=True)
                    continue
                status_path = path.parent / "status.json"
                if status_path.is_file() and read(status_path).get("status") != "complete":
                    print(f"Incomplete current attempt excluded: {path}", flush=True)
                    continue
                scope_path = output / flow / "evaluation_scope.json"
                if scope_path.is_file():
                    scope = read(scope_path)
                    if (row.get("source_manifest") != scope["source_manifest"]
                            or row.get("evaluation_test_hash") != scope["test_hash"]
                            or row.get("runtime_protocol") != scope["runtime"]):
                        print(f"STALE result excluded from aggregate: {path}", flush=True)
                        continue
                group.append(row)
                rows.append({"subflow": flow, "arm": arm, "result": str(path), **row["ast_cds"],
                             "reused": row.get("reused", False), "reuse_audit": row.get("reuse_audit"),
                             "source_completion_verified": row.get("source_manifest", {}).get("completion_verified")})
        denom = sum(r["ast_cds"]["num_action_turns"] for r in group)
        entry = {"completed_subflows": len(group), "expected_subflows": len(flows),
                 "complete": len(group) == len(flows), "action_turns": denom}
        for label, key in (("joint_ast", "ast_joint"), ("action", "ast_action_name"), ("slot", "ast_slot_value")):
            entry[label] = sum(r["ast_cds"][key] * r["ast_cds"]["num_action_turns"] for r in group) / denom if denom else None
        buckets = [r.get("llm_usage", {}).get("testing", {}).get("total", {}) for r in group]
        available = bool(group) and all(b.get("usage_available") for b in buckets)
        entry["testing_tokens"] = sum(b["total_tokens"] for b in buckets) if available else None
        entry["tokens_per_action_turn"] = entry["testing_tokens"] / denom if available and denom else None
        entry["testing_calls"] = sum(b["calls"] for b in buckets) if buckets and all("calls" in b for b in buckets) else None
        entry["testing_prompt_tokens"] = sum(b["prompt_tokens"] for b in buckets) if available and all("prompt_tokens" in b for b in buckets) else None
        entry["testing_completion_tokens"] = sum(b["completion_tokens"] for b in buckets) if available and all("completion_tokens" in b for b in buckets) else None
        entry["usage_sources"] = sorted({b.get("usage_source", "unavailable") for b in buckets})
        entry["cost_note"] = "Reused Ours cost is historical, not newly incurred; missing usage is null. Text evaluation, if enabled, adds calls."
        summary[arm] = entry
    payload = {"protocol": PROTOCOL, "aggregate": summary, "rows": rows}
    write(output / "k1_context_aggregate.json", payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return payload


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", help="Existing A2/full-experiment root; read-only")
    source.add_argument("--k1-dir", help="Exact K1 bundle directory; requires --subflow")
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--subflow")
    selection.add_argument("--subflows")
    selection.add_argument("--all", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--workflow-ids", default=os.getenv("SKILLMINING_WORKFLOW_ID", ""))
    p.add_argument("--model", help="Default inherits the K1 source model")
    p.add_argument("--max-test", type=int)
    p.add_argument("--full-test", action="store_true", help="Use the full test split, ignoring source smoke max_test")
    p.add_argument("--reference-top-k", type=int, default=10)
    p.add_argument("--action-selection-candidate-limit", type=int, default=8)
    p.add_argument("--action-card-max-chars", type=int, default=0, help="0: no card character cap")
    p.add_argument("--reference-max-chars", type=int, default=0, help="0: no total retrieved reference character cap")
    p.add_argument("--reference-section-max-chars", type=int, default=0, help="0: no per-section reference character cap")
    p.add_argument("--skip-arms", default="ours", help="Default skips Ours inference; compatible existing result is reused")
    p.add_argument("--include-ours", action="store_true", help="Explicitly reevaluate organized Ours; no training")
    p.add_argument("--prepare-only", action="store_true", help="Flatten and audit only; no llm import or API calls")
    p.add_argument("--allow-unverified-source", action="store_true")
    p.add_argument("--skip-utterance-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker-subflow", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.skip_arms = tuple(s.strip() for s in args.skip_arms.split(",") if s.strip())
    if args.include_ours:
        args.skip_arms = tuple(s for s in args.skip_arms if s != "ours")
    if set(args.skip_arms) - set(ARMS) or len(set(args.skip_arms)) != len(args.skip_arms) or len(args.skip_arms) == len(ARMS):
        p.error("--skip-arms must be a unique subset leaving at least one arm")
    if args.k1_dir and not args.subflow:
        p.error("--k1-dir requires a single --subflow")
    if args.max_test is not None and args.max_test < 1:
        p.error("--max-test must be positive")
    if args.full_test and args.max_test is not None:
        p.error("--full-test cannot be combined with --max-test")
    for key, minimum in (("reference_top_k", 0), ("action_selection_candidate_limit", 1),
                         ("action_card_max_chars", 0), ("reference_max_chars", 0), ("reference_section_max_chars", 0)):
        if getattr(args, key) < minimum:
            p.error(f"--{key.replace('_', '-')} must be >= {minimum}")
    if 0 < args.reference_max_chars < 200:
        p.error("--reference-max-chars must be 0 (unlimited) or >= 200")
    flows, ids = _subflows(args), _ids(args.workflow_ids)
    args.output_dir = str(Path(args.output_dir).resolve())
    for key in ("source_root", "k1_dir"):
        if getattr(args, key):
            setattr(args, key, str(Path(getattr(args, key)).resolve()))
    if args.worker_subflow:
        run_flow(args, args.worker_subflow, ids)
        return 0
    print(json.dumps({"subflows": flows, "run_arms": [a for a in ARMS if a not in args.skip_arms],
                      "prepare_only": args.prepare_only, "workflows_within_subflow": ids,
                      "online_training": False, "subflows_run_sequentially": True}, indent=2), flush=True)
    if args.dry_run:
        return 0
    output = Path(args.output_dir)
    for flow in flows:
        source = source_directory(args, flow)
        if source == output or source in output.parents or output in source.parents:
            p.error("Source and output roots must not overlap")
    settings = {k: v for k, v in vars(args).items() if k not in
                {"workflow_ids", "resume", "dry_run", "worker_subflow", "prepare_only", "skip_arms", "include_ours"}}
    settings.update(protocol=PROTOCOL)  # skip/include arms are invocation choices.
    config = output / "k1_context_config.json"
    if config.is_file() and read(config) != settings:
        p.error("Output config differs; choose a new directory")
    if output.exists() and any(output.iterdir()) and not args.resume:
        p.error("Output exists; use --resume or a fresh directory")
    if output.exists() and any(output.iterdir()) and not config.is_file():
        p.error("Nonempty output has no matching experiment config")
    write(config, settings)
    failures = []
    for flow in flows:
        command = [sys.executable, str(Path(__file__).resolve()), "--subflow", flow,
                   "--worker-subflow", flow, "--output-dir", str(output),
                   "--workflow-ids", ",".join(ids), "--skip-arms", ",".join(args.skip_arms)]
        for key in ("source_root", "k1_dir", "model", "max_test", "reference_top_k",
                    "action_selection_candidate_limit", "action_card_max_chars", "reference_max_chars",
                    "reference_section_max_chars"):
            if getattr(args, key) is not None:
                command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        for key in ("prepare_only", "allow_unverified_source", "resume", "full_test"):
            if getattr(args, key):
                command.append("--" + key.replace("_", "-"))
        command.append("--skip-utterance-eval" if args.skip_utterance_eval else "--no-skip-utterance-eval")
        directory = output / flow
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "k1_context_worker.log"
        print(f"START {flow}; log={log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
            log.flush()
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            failures.append(flow)
            print(f"ERROR {flow}; inspect {log_path}; continuing other subflows", flush=True)
        else:
            print(f"DONE {flow}", flush=True)
        if not args.prepare_only:
            aggregate(output, flows)
    if args.prepare_only:
        print(f"Flat resources prepared under {output}; no model calls made", flush=True)
    else:
        print(f"Results: {output / 'k1_context_aggregate.json'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
