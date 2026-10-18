#!/usr/bin/env python3
"""Replay standard refinement from frozen TRAIN rollout and saved diagnoses."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_tod.multiwoz_policy import MultiWOZStandardSkillAgent, load_hotel_pilot, ontology_from_train, predict_policy
from eval_tod.sgd_adapter import gold_policy, score_policies
from eval_tod.sgd_errors import error_summary, record_error
from eval_tod.sgd_llm import sgd_chat_with_retry
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_skill_agent import _json_object
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase
from skill_mining.online_refinement import build_post_rollout_batches, group_batch_reports


def repair_report(report, model):
    original = report["raw_response"]
    feedback = ""
    for attempt in range(1, 4):
        raw = sgd_chat_with_retry(
            "Repair JSON syntax only. Preserve the original diagnosis fields and semantic decisions. "
            "Escape embedded quotes and newlines correctly. Return exactly one JSON object, no markdown.\n"
            + feedback + "\nOriginal response:\n" + original,
            model=model, temperature=0, call_tag="multiwoz_diagnosis_format_repair")
        try:
            payload = _json_object(raw)
            if not isinstance(payload.get("summary"), str):
                raise ValueError("repaired diagnosis omitted summary")
            result = copy.deepcopy(report)
            result.update(payload)
            result.update(status="ok", error="", raw_response=raw,
                          format_repair_attempts=attempt)
            return result
        except ValueError as exc:
            feedback = "Previous repair parse error: " + str(exc)
    raise ValueError("diagnosis JSON remained invalid after 3 format repairs")


def combine_usage(previous, replay):
    result = {"note": "Includes the previous attempt plus this replay; reused rollout and diagnoses are not recharged.",
              "accounting_complete": previous.get("accounting_complete", False) and replay.get("accounting_complete", False)}
    keys = ("calls", "successful_calls", "failed_calls", "calls_with_usage", "prompt_tokens",
            "completion_tokens", "total_tokens", "exact_calls", "estimated_calls")
    for phase in ("generation", "testing", "total"):
        left = previous.get(phase, {}).get("total", {})
        right = replay.get(phase, {}).get("total", {})
        total = {key: left.get(key, 0) + right.get(key, 0) for key in keys}
        total["usage_source"] = "exact" if total["estimated_calls"] == 0 else "mixed"
        result[phase] = {"total": total}
    return result


def restore_failed_groups(agent, source, output, groups, indices):
    """Keep all committed groups and append only requested rolled-back groups."""
    if len(set(indices)) != len(indices) or any(i < 1 or i > len(groups) for i in indices):
        raise ValueError("Retry group indices must be unique and in range")
    training = json.loads((source / "ours_training.json").read_text(encoding="utf-8"))
    if training["reflection_groups"] != len(groups) or training["final_sha256"] != agent.resource_hash():
        raise ValueError("Source refinement checkpoint is incomplete or changed")
    committed = {entry.get("group_index") for entry in agent.ledger
                 if entry.get("phase") == "online_reflection"}
    inherited, failures = [], []
    for index in range(1, len(groups) + 1):
        success = source / f"reflection_{index:04d}.json"
        failure = source / f"reflection_{index:04d}_error.json"
        if index in indices:
            if success.exists() or index in committed or not failure.exists():
                raise ValueError(f"Group {index} was not fully rolled back; refusing duplicate application")
            failures.append(json.loads(failure.read_text(encoding="utf-8")))
        else:
            if not success.exists() or index not in committed:
                raise ValueError(f"Unrequested group {index} has no committed result")
            report = json.loads(success.read_text(encoding="utf-8"))
            if report.get("status") != "ok":
                raise ValueError(f"Unrequested group {index} did not succeed")
            inherited.append(index)
    # Validate the full recovery plan before copying any inherited results.
    for index in inherited:
        shutil.copy2(source / f"reflection_{index:04d}.json", output / f"reflection_{index:04d}.json")
    history = {"source_run": str(source), "source_checkpoint_sha256": training["final_sha256"],
               "retry_groups": indices, "inherited_groups": inherited,
               "application_order": [entry["group_index"] for entry in agent.ledger
                   if entry.get("phase") == "online_reflection"] + indices,
               "historical_errors": failures,
               "note": "Retries are rebased after all previously committed groups; no rollout accumulation repeated."}
    agent._write(output / "reflection_recovery.json", history)
    return [groups[i - 1] for i in indices]


def run(args):
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    manifest = json.loads((source / "data_manifest.json").read_text(encoding="utf-8"))
    data_source = Path(manifest["source"])
    if hashlib.sha256(data_source.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Dataset source hash changed")
    data = load_hotel_pilot(data_source, seed=manifest["seed"], fraction=manifest["fraction"])
    for split in ("train", "validation"):
        if [d["dialogue_id"] for d in data[split]] != manifest["selected_ids"][split]:
            raise ValueError("Replay dataset IDs differ from original run")
    records = [json.loads(line) for line in (source / "ours_training_turns.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != manifest["policy_turns"]["train"] or any("sgd_evidence" not in r for r in records):
        raise ValueError("Replay requires a complete successful frozen rollout")
    output.mkdir(parents=True)
    shutil.copytree(source / "offline", output / "offline")
    for name in ("ours_training_turns.jsonl", "online_schedule.json"):
        shutil.copy2(source / name, output / name)
    retry_indices = getattr(args, "retry_groups", None)
    if retry_indices:
        args.structured_reflection = True
    manifest.update(replay_from=str(source), runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), refinement_executor=(
        "semantic_resources_aliases_retry_v5" if retry_indices else
        "semantic_resources_compiler_v3" if args.structured_reflection else "validated_edits_with_rebase_v2"))
    if retry_indices:
        manifest["retry_groups"] = retry_indices
    (output / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    configure_usage(output)
    agent = MultiWOZStandardSkillAgent(family="hotel", model=manifest["model"],
              ontology=ontology_from_train(data["train"]), predictor=predict_policy, mining_method="trace_cover")
    agent.parallel_workers = args.workers
    agent.runtime_diagnostics_dir = output
    agent.load(source if retry_indices else output / "offline")
    if args.structured_reflection:
        agent.structured_reflection = True
        agent.reflection_prompt = """Reflect on neighboring TRAIN diagnosis reports.
Resolve conflicting proposals using failures AND successes. Current skill and
graph include earlier groups. Keep fixed action-head definitions and backbone.
No API/database tools or extra observations exist. Use prior dialogue context;
never memorize instance values. Hotel scenarios include other domains.
<current_skill>{skill}</current_skill>
<local_graph>{local_graph}</local_graph>
<batch_reports>{batch_reports}</batch_reports>
Return one JSON object with decision, summary, merged_root_causes, updates,
skill_operations, rejected_candidates, unresolved_questions.
Express ALL edits through semantic resource updates. The compiler places them
in the correct inline section or external card. skill_operations MUST be [].
Ignore stale match_text/candidate_skill_operations in reports; translate their
supported intent into updates. Do not copy or emit HTML markers or whole sections.
Each update has resource (transition_guard|action_rule|slot_policy|reference),
op (upsert|delete), status (resolved|uncertain), content (plain rule prose), rationale.
transition_guard requires exact edge_id from local_graph; action_rule/slot_policy
requires exact action label. Never invent IDs or change node-to-head mappings.
Use upsert for slot_policy/reference. Resolved action_rule updates are compiled
inline AND into the action card; slot_policy updates stay in the payload card.
Resolved guards are compiled into routing; uncertain evidence belongs in reference.
Preserve earlier useful rule content when replacing a resource.
REQUIRED JSON SHAPE (replace placeholders with exact IDs from local_graph):
{{"decision":"update","summary":"...","updates":[
{{"resource":"action_rule","action":"EXACT_NODE_LABEL","op":"upsert","status":"resolved","content":"Rule prose only","rationale":"..."}},
{{"resource":"transition_guard","edge_id":"EXACT_EDGE_ID","op":"upsert","status":"resolved","content":"Condition prose only","rationale":"..."}}],
"skill_operations":[],"rejected_candidates":[],"unresolved_questions":[]}}
The action and edge_id keys are MANDATORY for their respective resource types.
Never put IDs only in headings inside content. No HTML comments in content.
Action labels are policy-HASH; node IDs are hotel:policy-HASH; edge IDs join two
existing node IDs with =>. A known node pair is NOT necessarily an existing edge.
For an unobserved edge, keep the proposal in reference or rejected_candidates,
or give node-level advice as an action_rule. Never invent a transition_guard ID.
"""
    initial_hash = agent.resource_hash()
    if not retry_indices:
        agent._accumulate(records)
    agent.stage = "refining"
    if not retry_indices:
        agent.save(output)
    reports = []
    set_phase("online_diagnosis")
    batches = build_post_rollout_batches(records, max_batch_size=8)
    repaired = 0
    for index, batch in enumerate(batches, 1):
        report = json.loads((source / f"diagnosis_{index:04d}.json").read_text(encoding="utf-8"))
        if report["sample_ids"] != [r["sample"]["sample_id"] for r in batch]:
            raise ValueError(f"Diagnosis {index} belongs to a different rollout batch")
        if report["status"] != "ok":
            report = repair_report(report, agent.model)
            repaired += 1
        agent._write(output / f"diagnosis_{index:04d}.json", report)
        reports.append(report)
    groups = group_batch_reports(reports, max_reports=16)
    print(f"Reused {len(records)} rollout turns; {len(reports)} diagnoses ({repaired} repaired); {len(groups)} reflection groups", flush=True)
    if retry_indices:
        selected_groups = restore_failed_groups(agent, source, output, groups, retry_indices)
        print(f"Retrying only groups {retry_indices}; all other committed groups reused", flush=True)
        agent.reflect_groups(selected_groups, output, edit_attempts=4, group_indices=retry_indices)
    else:
        agent.reflect_groups(groups, output)
    errors = error_summary(output)
    agent.stage = "refined_with_errors" if errors["total"] else "refined"
    agent.save(output)
    agent._write(output / "ours_training.json", {"rollout_turns": len(records), "reports": len(reports),
                "reflection_groups": len(groups), "repaired_diagnoses": repaired,
                "replayed_group_indices": retry_indices,
                "snapshot_sha256": initial_hash, "final_sha256": agent.resource_hash(), "case_errors": errors})
    if retry_indices and errors["total"]:
        raise RuntimeError("Targeted reflection retries failed; checkpoint saved, evaluation not started")
    set_phase("testing")
    cases = [(d, i) for d in data["validation"] for i in d["target_indices"]["policy_turn_indices"]]
    print(f"Replay validation: {len(cases)} turns", flush=True)
    predictions = ordered_cases(lambda case: agent.predict(*case), cases,
                                workers=args.workers, directory=output, phase="testing")
    gold_rows, pred_rows = [], []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for outcome, (dialogue, index) in zip(predictions, cases):
            gold = gold_policy(dialogue, index)
            try:
                prediction, diagnostic = require_value(outcome)
            except Exception as exc:
                record_error(output, "testing", exc, dialogue_id=dialogue["dialogue_id"], turn_index=index)
                prediction, diagnostic = {"call": None, "acts": [], "_execution_error": True}, {"status": "error"}
            gold_rows.append(gold)
            pred_rows.append(prediction)
            stream.write(json.dumps({"dialogue_id": dialogue["dialogue_id"], "turn_index": index,
                        "gold": gold, "prediction": prediction, "diagnostic": diagnostic}, ensure_ascii=False) + "\n")
            stream.flush()
    replay_usage = persist_usage()
    source_summary = source / "summary.json"
    if source_summary.exists():
        previous_usage = json.loads(source_summary.read_text(encoding="utf-8"))["llm_usage"]
    else:
        previous_usage = json.loads((source / "llm_usage.json").read_text(encoding="utf-8"))
        previous_usage["accounting_complete"] = False
        previous_usage["note"] = "Source run is still in progress; cumulative historical usage is incomplete."
    errors = error_summary(output)
    summary = {**manifest, **score_policies(gold_rows, pred_rows),
               "graph_nodes": len(agent.graph["nodes"]), "graph_edges": len(agent.graph["edges"]),
               "skill_stage": agent.stage, "case_errors": errors,
               "llm_usage": combine_usage(previous_usage, replay_usage),
               "previous_attempt_usage": previous_usage, "replay_usage": replay_usage,
               "execution_status": "completed_with_errors" if errors["total"] else "completed"}
    agent._write(output / "summary.json", summary)
    from scripts.audit_multiwoz_empty_values import audit
    audit(output)
    print(json.dumps({"turn_joint_ast": summary["turn_joint_ast"], "errors": errors["total"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--structured-reflection", action="store_true")
    parser.add_argument("--retry-groups", type=int, nargs="+",
                        help="Load final source checkpoint and replay only these rolled-back groups")
    run(parser.parse_args())
