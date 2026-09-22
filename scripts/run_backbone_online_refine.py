#!/usr/bin/env python3
"""Online refinement for an offline graph-compiled ABCD skill.

The runner intentionally keeps the offline arborescence immutable. Training
rollouts update edge evidence on the training split only; selected local guards
are appended to the runtime skill only after an evidence-based promotion.
"""
from __future__ import annotations

import argparse
import atexit
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from awm import MemoryStore, WorkflowStore
from eval_tod.abcd.agent import ABCDAgent
from eval_tod.abcd.metrics import slot_match_profile
from skill_mining.online_refinement import (
    RefinementPolicy,
    _actions,
    autonomous_resource_reflection,
    apply_dynamic_skill_operations,
    apply_working_skill_operations,
    accumulate_online_evidence,
    build_online_evidence_packets,
    initialize_skill_dag,
    load_skill_dag,
    localize_rollout_batch,
    propose_refinement_patches,
    render_online_resources,
    render_online_action_rules,
    merge_online_skill_additions,
    render_online_slot_policies,
    save_skill_dag,
    schedule_contrastive_batches,
    schedule_constrained_repair_batches,
    schedule_trace2skill_conversation_batches,
    schedule_action_turn_batches,
    build_action_turn_samples,
    build_post_rollout_batches,
    diagnose_rollout_batch,
    group_batch_reports,
    lookup_graph_neighborhood,
    reflect_batch_report_group,
    trace2skill_hybrid_reflect_report_group,
    apply_reflection_updates_to_state,
    summarize_refinement_state,
)
from scripts.run_subflow_eval import (
    MODEL,
    evaluate_agent_on_subflow,
    load_subflow_data,
    mine_subflow_skill_backbone,
)
from scripts import run_trace2skill_abcd as trace2skill_impl


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _trace2skill_turn_view(row: dict) -> dict:
    """Project a rollout row to the trajectory Trace2Skill is allowed to see.

    The trajectory is the dialogue prefix plus the current-turn prediction.
    Runtime ReAct/tool diagnostics are deliberately excluded: ``react_trace``,
    reference-planning messages, action-selection messages, and prompt-budget
    fields can contain the complete LLM prompt (including the whole skill).
    """
    allowed = (
        "convo_id", "turn_index", "agent_turn_num", "target_type",
        "total_agent_turns", "subflow", "flow", "context", "context_view",
        "prediction", "predicted_action",
        "predicted_slots", "action_schema_validation",
    )
    return {
        key: row.get(key)
        for key in allowed
        if key in row
    }


def _run_trace2skill_hybrid_batch(
    *, batch_index: int, conversations: list[dict], turns: list[dict],
    out_dir: Path, working_skill: str, base_reference: str,
    action_rules: str, slot_policies: str, model: str, response_logger,
    state: dict, rollout_workers: int = 1,
    analysis_batch_size: int = 8, map_batch_size: int = 8,
) -> tuple[str, dict]:
    """Run the original Trace2Skill analysis/evolution path on one batch.

    The online runner only adapts the resource context. Case construction,
    success/failure analysis, MAP/REDUCE/TRANSLATE/APPLY, and patch parsing are
    delegated to the existing Trace2Skill implementation.
    """
    batch_root = out_dir / "trace2skill_hybrid_batches" / f"batch_{batch_index:04d}"
    batch_root.mkdir(parents=True, exist_ok=True)
    skill_dir = out_dir / "trace2skill_hybrid_skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(working_skill, encoding="utf-8")
    # Make graph-compiled resources available to the original evolver through
    # the same linked-resource files it already understands.
    (skill_dir / "reference.md").write_text(base_reference, encoding="utf-8")
    (skill_dir / "action_rules.md").write_text(action_rules, encoding="utf-8")
    (skill_dir / "slot_policies.md").write_text(slot_policies, encoding="utf-8")
    # Keep the full prefix/current-prediction trajectory, but never pass the
    # runtime's complete ReAct trace into the transplanted Trace2Skill path.
    trace_turns = [_trace2skill_turn_view(row) for row in turns]
    ast_scores = trace2skill_impl.compute_ast_from_turn_results(conversations, trace_turns)
    failures = trace2skill_impl._build_ast_failure_cases(
        conversations, trace_turns, ast_scores,
        log_dir=batch_root / "failure_logs", hide_scenario_labels=False,
    )
    successes = trace2skill_impl._build_ast_success_cases(
        conversations, trace_turns, ast_scores,
        log_dir=batch_root / "success_logs", hide_scenario_labels=False,
    )
    # Attach a typed bridge record to the Trace2Skill cases. The trajectory
    # passed to the analyzer is exactly the prefix/current-prediction view;
    # graph evidence is an additional compact ToD resource, not a ReAct dump.
    turns_by_convo: dict[str, list[dict]] = defaultdict(list)
    for row in trace_turns:
        turns_by_convo[str(row.get("convo_id", "?"))].append(row)
    evidence_records = []
    for conv, ast in zip(conversations, ast_scores):
        convo_id = str(conv.get("convo_id", "?"))
        conv_turns = sorted(turns_by_convo.get(convo_id, []), key=lambda row: int(row.get("turn_index", -1)))
        mismatches, _ = trace2skill_impl._build_ast_mismatch_report(conv, conv_turns)
        first = mismatches[0] if mismatches else None
        actions = _actions(conv)
        nodes = [value for value in (
            first.get("gold_action") if first else None,
            first.get("predicted_action") if first else None,
            actions[-1] if actions else None,
        ) if value]
        graph_context = lookup_graph_neighborhood(state, [str(value) for value in nodes], radius=1)
        protected = [
            {"conversation_id": str(item.get("convo_id", "?")), "ast_score": float(score.get("ast_score", 0.0))}
            for item, score in zip(conversations, ast_scores)
            if float(score.get("ast_score", 0.0)) >= 1.0
        ][:8]
        record = {
            "conversation_id": convo_id,
            "trajectory": conv_turns,
            "first_divergence": first,
            "error_type": (
                "route" if first and first.get("predicted_action") != first.get("gold_action")
                else "slot" if first else "none"
            ),
            "action_sequence": actions,
            "graph_context": graph_context,
            "protected_successes": protected,
        }
        evidence_records.append(record)
    _write(batch_root / "trajectory_evidence.json", json.dumps(evidence_records, indent=2, ensure_ascii=False))
    evidence_by_id = {item["conversation_id"]: item for item in evidence_records}
    # The case trajectory above is the only trajectory sent to Trace2Skill:
    # complete dialogue prefixes plus current-turn predictions.  The runtime
    # ReAct/tool records are intentionally absent from this evidence bridge.
    def _analysis_evidence(record: dict) -> dict:
        first = record.get("first_divergence")
        if isinstance(first, dict):
            first = {
                key: first.get(key)
                for key in (
                    "action_turn_index", "source_agent_turn_index",
                    "predicted_action", "predicted_slots", "gold_action",
                    "gold_slots", "action_match", "slots_match",
                    "source_agent_response", "reference_agent_response",
                )
                if key in first
            }
        return {
            "conversation_id": record.get("conversation_id"),
            "first_divergence": first,
            "error_type": record.get("error_type"),
            "action_sequence": record.get("action_sequence", []),
            "graph_context": record.get("graph_context", {}),
            "protected_successes": record.get("protected_successes", []),
        }

    for case in failures + successes:
        cid = str(case.get("dialogue_id", case.get("instance_id", "")).removeprefix("abcd-"))
        evidence = evidence_by_id.get(cid)
        if not evidence:
            continue
        prompt_evidence = _analysis_evidence(evidence)
        case["hybrid_evidence"] = prompt_evidence
        case["trajectory"] = (
            str(case.get("trajectory", ""))
            + "\n\n## Typed ToD Decision Evidence\n"
            + json.dumps(prompt_evidence, ensure_ascii=False, indent=2)
        )
    error_path = None
    success_path = None
    if failures and successes:
        # Error and success analyses are independent and write separate
        # artifact trees, so overlap their network-bound LLM calls.
        with ThreadPoolExecutor(max_workers=2) as executor:
            error_future = executor.submit(
                trace2skill_impl._run_error_analysis,
                failures, batch_root / "error_analysis", model, response_logger,
                batch_size=analysis_batch_size,
            )
            success_future = executor.submit(
                trace2skill_impl._run_success_analysis,
                successes, batch_root / "success_analysis", model, response_logger,
                batch_size=analysis_batch_size,
            )
            error_path = error_future.result()
            success_path = success_future.result()
    elif failures:
        error_path = trace2skill_impl._run_error_analysis(
            failures, batch_root / "error_analysis", model, response_logger,
            batch_size=analysis_batch_size,
        )
    elif successes:
        success_path = trace2skill_impl._run_success_analysis(
            successes, batch_root / "success_analysis", model, response_logger,
            batch_size=analysis_batch_size,
        )
    changelog = []
    if error_path or success_path:
        changelog = trace2skill_impl._run_skill_evolution(
            error_path, success_path, skill_path,
            batch_root / "evolution", model, response_logger,
            map_batch_size=map_batch_size,
        )
    updated_skill = skill_path.read_text(encoding="utf-8")
    summary = {
        "batch_index": batch_index,
        "num_conversations": len(conversations),
        "num_turns": len(turns),
        "rollout_workers": rollout_workers,
        "analysis_workers": 2 if failures and successes else 1,
        "failed_cases": len(failures),
        "successful_cases": len(successes),
        "changelog": changelog,
        "skill_path": str(skill_path),
    }
    _write(batch_root / "batch_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    return updated_skill, summary


def _load_successful_posthoc_artifact(path: Path, expected_ids: list[str]) -> dict | None:
    """Load a completed artifact whose evidence identity still matches."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("status") == "error" or payload.get("error"):
        return None
    actual_ids = payload.get("sample_ids", payload.get("batch_ids", []))
    if [str(value) for value in actual_ids] != [str(value) for value in expected_ids]:
        return None
    return payload


def _rollout_online_batch(agent: ABCDAgent, batch: list) -> list[dict]:
    """Run one online training batch in order under the current policy state."""
    rows: list[dict] = []
    for item in batch:
        conversation = item.get("conversation", item) if isinstance(item, dict) else item
        turn_index = item.get("turn_index") if isinstance(item, dict) and "turn_index" in item else None
        rows.extend(agent.predict_all_turns(conversation, predict_actions=True,
                                            action_only=True, verbose=False,
                                            turn_index=turn_index))
    return rows


def _run_parallel_online_wave(
    args, batches: list[list[dict]], workflow_ids: list[str],
    working_skill: str, base_reference: str, action_rules: str,
    slot_policies: str, state: dict, response_logger,
) -> list[dict]:
    """Roll out different batches concurrently against one frozen skill.

    This is deliberately limited to rollout. Localization, evidence-pool
    mutation, reflection, and skill writes remain in the parent process.
    """
    frozen_state = copy.deepcopy(state)
    ids = workflow_ids or [None]

    def run_one(index: int, batch: list[dict], workflow_id: str | None) -> dict:
        agent = _build_agent(
            args, working_skill, base_reference, action_rules, slot_policies,
            copy.deepcopy(frozen_state), response_logger=response_logger,
            workflow_id=workflow_id,
        )
        turns = _rollout_online_batch(agent, batch)
        return {
            "batch_index": index,
            "workflow_id": workflow_id,
            "batch": batch,
            "turns": turns,
        }

    # A transplanted Trace2Skill batch must remain one semantic unit: all
    # conversations are rolled out against the same frozen skill and then fed
    # to one analysis/evolution pass. Split only the rollout work across
    # workflow workers and merge the turn rows before returning.
    if args.refinement_mode == "trace2skill-hybrid" and len(batches) == 1 and len(ids) > 1:
        batch = batches[0]
        chunks = [
            batch[start:start + (len(batch) + len(ids) - 1) // len(ids)]
            for start in range(0, len(batch), (len(batch) + len(ids) - 1) // len(ids))
        ]
        chunks = [chunk for chunk in chunks if chunk]
        results: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=min(len(ids), len(chunks))) as executor:
            futures = {
                executor.submit(run_one, index, chunk, ids[index % len(ids)]): index
                for index, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                item = future.result()
                results[item["batch_index"]] = item
        merged_turns = [
            row for index in sorted(results) for row in results[index]["turns"]
        ]
        return [{
            "batch_index": 0,
            "workflow_id": ",".join(str(item["workflow_id"]) for item in results.values()),
            "batch": batch,
            "turns": merged_turns,
            "parallel_rollout_workers": len(chunks),
        }]

    if len(batches) == 1:
        return [run_one(0, batches[0], ids[0])]
    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=min(len(ids), len(batches))) as executor:
        futures = {
            executor.submit(run_one, index, batch, ids[index % len(ids)]): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            item = future.result()
            results[item["batch_index"]] = item
    return [results[index] for index in sorted(results)]


def _merge_evidence_packets(packet_parts: list[dict]) -> dict:
    """Combine one wave's bounded packet views for the single reflection call."""
    merged = {"transition": {}, "action_card": {}, "counts": {}}
    for packets in packet_parts:
        for group_name in ("transition", "action_card"):
            for key, buckets in (packets.get(group_name, {}) or {}).items():
                target = merged[group_name].setdefault(key, {})
                for bucket, values in (buckets or {}).items():
                    target.setdefault(bucket, []).extend(values or [])
    merged["counts"] = {
        "transition_edges": len(merged["transition"]),
        "actions": len(merged["action_card"]),
        "raw_examples": sum(int((p.get("counts") or {}).get("raw_examples", 0) or 0)
                             for p in packet_parts),
        "examples": sum(len(values) for group in merged.values() if isinstance(group, dict)
                         for buckets in group.values() if isinstance(buckets, dict)
                         for values in buckets.values() if isinstance(values, list)),
    }
    return merged


def _repair_failed_skill_operations(
    out_dir: Path, state: dict, working_skill: str,
) -> tuple[str, list[dict]]:
    """Replay only historically failed in-place skill edits after an anchor migration."""
    repaired: list[dict] = []
    reflection_dir = out_dir / "autonomous_reflection"
    if not reflection_dir.exists():
        return working_skill, repaired
    for path in sorted(reflection_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        previous = payload.get("skill_operations", [])
        if not any(isinstance(item, dict) and item.get("error") for item in previous):
            continue
        dynamic_updates = payload.get("proposed_skill_operations", [])
        if dynamic_updates:
            working_skill, replayed = apply_dynamic_skill_operations(working_skill, dynamic_updates)
        else:
            # Pre-content-addressed runs only stored semantic resource updates.
            updates = [
                item for item in payload.get("accepted", [])
                if isinstance(item, dict) and item.get("resource") in {"action_rule", "transition_guard"}
            ]
            if not updates:
                continue
            working_skill, replayed = apply_working_skill_operations(working_skill, state, updates)
        payload["resume_repair_skill_operations"] = replayed
        _write(path, json.dumps(payload, indent=2, ensure_ascii=False))
        repaired.append({"batch": path.stem, "operations": replayed})
    return working_skill, repaired


def _batch_rollout_supervision(conversations: list[dict], turn_results: list[dict]) -> list[dict]:
    """Align every rollout target with gold action/slot supervision when present."""
    gold_by_turn = {}
    for item in conversations:
        conversation = item.get("conversation", item) if isinstance(item, dict) else item
        convo_id = str(conversation.get("convo_id", "?"))
        for turn_index, turn in enumerate(conversation.get("delexed") or []):
            targets = turn.get("targets") or []
            if len(targets) >= 3 and targets[1] == "take_action":
                gold_by_turn[(convo_id, turn_index)] = {
                    "gold_action": targets[2],
                    "gold_slots": targets[3] if len(targets) > 3 and isinstance(targets[3], list) else [],
                }
    rows = []
    for row in turn_results:
        key = (str(row.get("convo_id", "?")), int(row.get("turn_index", -1)))
        rows.append({
            "conversation_id": key[0], "turn_index": key[1], "target_type": row.get("target_type"),
            "context": row.get("context", ""), "prediction": row.get("prediction", ""),
            "predicted_action": row.get("predicted_action", ""),
            "predicted_slots": row.get("predicted_slots", []),
            "gold": gold_by_turn.get(key), "react_trace": row.get("react_trace", []),
            "gold_response": (
                row.get("reference_original") or row.get("reference", "")
                if row.get("target_type") == "utterance" else ""
            ),
        })
    return rows


def _batch_conversations(batch: list[dict]) -> list[dict]:
    """Unwrap action-turn samples while retaining one copy per sample target."""
    result, seen = [], set()
    for item in batch:
        conversation = item.get("conversation", item) if isinstance(item, dict) else item
        key = str(conversation.get("convo_id", "?"))
        if key not in seen:
            result.append(conversation); seen.add(key)
    return result


def _build_agent(args, working_skill: str, base_reference: str, action_rules: str,
                 slot_policies: str, state: dict, response_logger=None,
                 workflow_id: str | None = None) -> ABCDAgent:
    _, online_reference = render_online_resources(state)
    workflow = WorkflowStore()
    workflow.update(working_skill)
    return ABCDAgent(
        model=args.model,
        workflow=workflow,
        # Keep the full evolving skill visible to the runtime. Trace2Skill's
        # agent uses the same unbounded workflow prompt; the ABCDAgent default
        # of 8000 chars would silently truncate later online additions.
        workflow_max_chars=None,
        memory=MemoryStore(),
        reference_text=base_reference.rstrip() + "\n\n" + online_reference,
        action_rules_text=action_rules.rstrip() + "\n\n" + render_online_action_rules(state),
        slot_policies_text=slot_policies.rstrip() + "\n\n" + render_online_slot_policies(state),
        response_logger=response_logger,
        reference_top_k=args.reference_top_k,
        reference_max_chars=args.reference_max_chars,
        expose_scenario_labels=False,
        workflow_id=workflow_id,
    )


def _checkpoint(
    out_dir: Path, state: dict, working_skill: str, base_reference: str,
    policy: RefinementPolicy | None = None, base_slot_policies: str = "", base_action_rules: str = "",
) -> str:
    working_skill = merge_online_skill_additions(working_skill, state)
    online_skill, online_reference = render_online_resources(state)
    save_skill_dag(state, out_dir / "skill_dag_state.json")
    _write(out_dir / "online_transition_guards.md", online_skill)
    _write(out_dir / "online_reference.md", online_reference)
    _write(out_dir / "online_slot_policies.md", render_online_slot_policies(state))
    _write(out_dir / "online_action_rules.md", render_online_action_rules(state))
    _write(out_dir / "working_skill.md", working_skill)
    _write(out_dir / "skill.md", working_skill)
    _write(out_dir / "reference.md", base_reference.rstrip() + "\n\n" + online_reference)
    _write(out_dir / "slot_policies.md", base_slot_policies.rstrip() + "\n\n" + render_online_slot_policies(state))
    _write(out_dir / "action_rules.md", base_action_rules.rstrip() + "\n\n" + render_online_action_rules(state))
    if policy is not None:
        _write(
            out_dir / "refinement_summary.json",
            json.dumps(summarize_refinement_state(state, policy), indent=2, ensure_ascii=False),
        )
    return working_skill


def _replay_candidate(
    args, candidate_skill: str, candidate_state: dict, source_samples: list[dict],
    sample_ids: set[str], base_reference: str, action_rules: str,
    slot_policies: str, response_logger, workflow_id: str | None,
) -> dict:
    """Replay only the candidate's local evidence and return compact metrics."""
    selected = [sample for sample in source_samples
                if str(sample.get("sample_id")) in sample_ids]
    if not selected:
        return {"num_samples": 0, "action_correct": 0, "joint_correct": 0,
                "slot_correct": 0, "error": "no_replay_samples"}
    agent = _build_agent(args, candidate_skill, base_reference, action_rules,
                         slot_policies, candidate_state,
                         response_logger=response_logger, workflow_id=workflow_id)
    rows = _rollout_online_batch(agent, selected)
    def row_key(item: dict) -> tuple[str, int]:
        try:
            turn_index = int(item.get("turn_index", -1))
        except (TypeError, ValueError):
            turn_index = -1
        return (str(item.get("conversation_id", item.get("convo_id", "?"))), turn_index)

    # The replay API may return fewer rows after a transient/model failure.
    # Keep those samples in the denominator and score them as incorrect; a
    # candidate must never pass by silently dropping difficult examples.
    returned = {row_key(row): row for row in rows}
    action_correct = joint_correct = slot_correct = 0
    total = len(selected)
    for sample in selected:
        row = returned.get(row_key(sample))
        if row is None:
            continue
        action_ok = str(row.get("predicted_action", "")) == str(sample.get("target_action", ""))
        profile = slot_match_profile(
            [str(value) for value in sample.get("gold_slots", [])],
            [str(value) for value in row.get("predicted_slots", [])],
        )
        slot_ok = bool(profile.get("strict_ordered"))
        action_correct += int(action_ok)
        slot_correct += int(slot_ok)
        joint_correct += int(action_ok and slot_ok)
    return {
        "num_samples": total,
        "action_correct": action_correct,
        "joint_correct": joint_correct,
        "slot_correct": slot_correct,
        "action_accuracy": action_correct / max(total, 1),
        "joint_accuracy": joint_correct / max(total, 1),
        "slot_accuracy": slot_correct / max(total, 1),
    }


def _run_refinement_pass(
    *, artifact_root: Path, pass_id: str, records: list[dict], state: dict,
    working_skill: str, base_reference: str, action_rules: str,
    slot_policies: str, model: str, response_logger, workflow_ids: list[str],
    batch_size: int, refinement_mode: str, hybrid_map_batch_size: int,
    max_retries: int, ledger_path: Path, log: logging.Logger,
    runner_args=None, replay_source: list[dict] | None = None,
) -> tuple[str, dict]:
    """Diagnose and apply one frozen-snapshot refinement pass."""
    evidence_batches = build_post_rollout_batches(records, max_batch_size=batch_size)
    log.info("Refinement pass=%s mode=%s evidence_batches=%d",
             pass_id, refinement_mode, len(evidence_batches))
    reports = []
    replay_samples_by_id = {}
    if refinement_mode == "constrained-repair" and replay_source:
        # Action-turn samples are the supervision-bearing source of truth. Do
        # this once per pass; raw conversations do not contain sample_id or
        # target_action fields suitable for replay gating.
        replay_samples_by_id = {
            str(sample.get("sample_id")): sample
            for sample in build_action_turn_samples(replay_source)
        }
    failed_batch_ids = []
    for report_index, evidence_batch in enumerate(evidence_batches, start=1):
        batch_id = f"evidence_batch_{report_index:04d}"
        report_path = artifact_root / "batch_root_causes" / f"{batch_id}.json"
        sample_ids = [str(item["sample"]["sample_id"]) for item in evidence_batch]
        report = _load_successful_posthoc_artifact(report_path, sample_ids)
        if report is not None:
            reports.append(report)
            log.info("Root-cause pass=%s batch=%d/%d reused", pass_id,
                     report_index, len(evidence_batches))
            continue
        workflow_id = workflow_ids[(report_index - 1) % len(workflow_ids)] if workflow_ids else None
        log.info("Root-cause pass=%s batch=%d/%d started records=%d", pass_id,
                 report_index, len(evidence_batches), len(evidence_batch))
        report = diagnose_rollout_batch(
            batch_id, evidence_batch, state, working_skill, model,
            response_logger=response_logger, workflow_id=workflow_id,
            reference=base_reference + "\n" + render_online_resources(state)[1],
            action_rules=action_rules + "\n" + render_online_action_rules(state),
            slot_policies=slot_policies + "\n" + render_online_slot_policies(state),
            max_retries=max_retries,
        )
        _write(report_path, json.dumps(report, indent=2, ensure_ascii=False))
        if report.get("status") == "error":
            failed_batch_ids.append(batch_id)
            log.error("Root-cause pass=%s batch=%d/%d skipped after %d attempts: %s",
                      pass_id, report_index, len(evidence_batches),
                      report.get("attempts", 0), report.get("error", "unknown error"))
            continue
        reports.append(report)
        log.info("Root-cause pass=%s batch=%d/%d completed", pass_id,
                 report_index, len(evidence_batches))

    reflection_groups = group_batch_reports(reports, max_reports=16)
    applied_ids = set(state.get("applied_skill_operation_ids", []))
    failed_reflection_groups = []
    for reflection_index, report_group in enumerate(reflection_groups, start=1):
        reflection_path = artifact_root / "group_reflections" / f"reflection_{reflection_index:04d}.json"
        group_batch_ids = [str(report.get("batch_id")) for report in report_group]
        reflection = _load_successful_posthoc_artifact(reflection_path, group_batch_ids)
        workflow_id = workflow_ids[(reflection_index - 1) % len(workflow_ids)] if workflow_ids else None
        if reflection is not None:
            log.info("Reflection pass=%s group=%d/%d reused", pass_id,
                     reflection_index, len(reflection_groups))
        else:
            log.info("Reflection pass=%s group=%d/%d started mode=%s reports=%d",
                     pass_id, reflection_index, len(reflection_groups),
                     refinement_mode, len(report_group))
            if refinement_mode in {"trace2skill-hybrid", "constrained-repair"}:
                reflection = trace2skill_hybrid_reflect_report_group(
                    report_group, state, working_skill, model,
                    response_logger=response_logger, workflow_id=workflow_id,
                    map_batch_size=hybrid_map_batch_size,
                    max_retries=max_retries,
                )
            else:
                reflection = reflect_batch_report_group(
                    report_group, state, working_skill, model,
                    response_logger=response_logger, workflow_id=workflow_id,
                    max_retries=max_retries,
                )
            _write(reflection_path, json.dumps(reflection, indent=2, ensure_ascii=False))
        if reflection.get("status") == "error":
            failed_reflection_groups.append(reflection_index)
            log.error("Reflection pass=%s group=%d/%d skipped after retries: %s",
                      pass_id, reflection_index, len(reflection_groups),
                      reflection.get("error", "unknown error"))
            continue
        candidate_state = copy.deepcopy(state)
        candidate_accepted, candidate_rejected = apply_reflection_updates_to_state(
            candidate_state, reflection.get("updates", []),
        )
        candidate_skill, candidate_text_operations = apply_dynamic_skill_operations(
            working_skill, reflection.get("skill_operations", []), set(applied_ids),
        )
        accepted, rejected = candidate_accepted, candidate_rejected
        text_operations = candidate_text_operations
        replay = None
        replay_decision = "apply"
        if refinement_mode == "constrained-repair" and runner_args is not None and replay_source:
            replay_ids = {
                str(sample_id)
                for report in report_group
                for sample_id in report.get("sample_ids", [])
            }
            # Build the local action-turn set from the conversations, but use
            # the derived samples as the sole source of truth.  The frozen
            # wave records are diagnosis evidence only; they are not a valid
            # baseline for a sequential candidate decision.
            replay_samples = [replay_samples_by_id[sample_id]
                              for sample_id in sorted(replay_ids)
                              if sample_id in replay_samples_by_id]
            baseline = _replay_candidate(
                runner_args, working_skill, state, replay_samples, replay_ids,
                base_reference, action_rules, slot_policies,
                response_logger,
                workflow_ids[(reflection_index - 1) % len(workflow_ids)]
                if workflow_ids else None,
            )
            replay = _replay_candidate(
                runner_args, candidate_skill, candidate_state,
                replay_samples, replay_ids,
                base_reference, action_rules, slot_policies,
                response_logger,
                workflow_ids[(reflection_index - 1) % len(workflow_ids)]
                if workflow_ids else None,
            )
            replay["baseline"] = baseline
            replay["comparison"] = {
                "type": "sequential_paired_replay",
                "sample_ids": sorted(replay_ids),
                "baseline_skill_sha256": hashlib.sha256(working_skill.encode("utf-8")).hexdigest(),
                "candidate_skill_sha256": hashlib.sha256(candidate_skill.encode("utf-8")).hexdigest(),
            }
            # A candidate must not lose action correctness; among non-degrading
            # candidates, apply only if it improves action or joint accuracy.
            replay_decision = (
                "apply" if replay.get("action_correct", 0) >= baseline.get("action_correct", 0)
                and replay.get("joint_correct", 0) >= baseline.get("joint_correct", 0)
                and (replay.get("action_correct", 0) > baseline.get("action_correct", 0)
                     or replay.get("joint_correct", 0) > baseline.get("joint_correct", 0))
                else "reject"
            )
            if replay_decision == "reject":
                accepted, rejected = [], [
                    {"update": update, "reason": "candidate_replay_no_improvement"}
                    for update in reflection.get("updates", [])
                ]
                text_operations = []
            else:
                state.clear(); state.update(candidate_state)
                working_skill = candidate_skill
                applied_ids.update(
                    str(item.get("operation_id")) for item in text_operations
                    if item.get("applied") and item.get("operation_id")
                )
            log.info(
                "Replay gate pass=%s group=%d/%d decision=%s baseline(action=%d joint=%d) "
                "candidate(action=%d joint=%d) samples=%d",
                pass_id, reflection_index, len(reflection_groups), replay_decision,
                baseline.get("action_correct", 0), baseline.get("joint_correct", 0),
                replay.get("action_correct", 0), replay.get("joint_correct", 0),
                baseline.get("num_samples", 0),
            )
        else:
            accepted, rejected = apply_reflection_updates_to_state(state, reflection.get("updates", []))
            working_skill, text_operations = apply_dynamic_skill_operations(
                working_skill, reflection.get("skill_operations", []), applied_ids,
            )
        reflection["accepted_updates"] = accepted
        reflection["rejected_updates"] = rejected
        reflection["applied_skill_operations"] = text_operations
        reflection["candidate_replay"] = replay
        reflection["candidate_replay_decision"] = replay_decision
        _write(reflection_path, json.dumps(reflection, indent=2, ensure_ascii=False))
        _append_jsonl(ledger_path, {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pass_id": pass_id,
            "reflection_index": reflection_index,
            "batch_ids": reflection.get("batch_ids", []),
            "root_cause_summary": reflection.get("summary", ""),
            "accepted_updates": accepted,
            "rejected_updates": rejected,
            "skill_operations": text_operations,
            "refinement_mode": refinement_mode,
            "candidate_replay": reflection.get("candidate_replay"),
            "candidate_replay_decision": reflection.get("candidate_replay_decision", "apply"),
        })
        log.info("Reflection pass=%s group=%d/%d completed accepted=%d rejected=%d text_ops=%d",
                 pass_id, reflection_index, len(reflection_groups), len(accepted),
                 len(rejected), len(text_operations))
    state["applied_skill_operation_ids"] = sorted(applied_ids)
    return working_skill, {
        "pass_id": pass_id,
        "num_evidence_batches": len(evidence_batches),
        "num_successful_reports": len(reports),
        "num_reflection_groups": len(reflection_groups),
        "failed_batch_ids": failed_batch_ids,
        "failed_reflection_groups": failed_reflection_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-calibrated online refinement for a backbone skill")
    parser.add_argument("--subflow", required=True, help="One existing ABCD split directory")
    parser.add_argument("--output-dir", required=True, help="Run directory; also used by --resume")
    parser.add_argument("--offline-dir", default=None,
                        help="Existing offline backbone artifact directory; skip offline re-mining")
    parser.add_argument("--resume", action="store_true", help="Resume completed online batches from skill_dag_state.json")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Rollout batch size for the standard online-refine path (default: 8).",
    )
    parser.add_argument(
        "--trace2skill-batch-size", type=int, default=25,
        help="Complete conversations per transplanted Trace2Skill batch (default: 25).",
    )
    parser.add_argument("--per-transition-cap", type=int, default=3)
    parser.add_argument(
        "--target-selection-rate", type=float, default=0.30,
        help="Target fraction of train sessions selected for online refinement (default: 0.30).",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument(
        "--analysis-batch-size", type=int, default=8,
        help="Trace2Skill-compatible success/error analysis batch size (default: 8).",
    )
    parser.add_argument(
        "--skip-utterance-eval", action="store_true",
        help="Final held-out evaluation only predicts/evaluates action turns; text metrics are left empty.",
    )
    parser.add_argument("--reference-top-k", type=int, default=3)
    parser.add_argument("--reference-max-chars", type=int, default=1800)
    parser.add_argument("--min-gold-support", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--min-conflict-count", type=int, default=2)
    parser.add_argument("--max-skill-branches-per-source", type=int, default=3)
    parser.add_argument("--guard-retries", type=int, default=3,
                        help="Attempts for each guard, diagnosis, MAP, REDUCE, or reflection call")
    parser.add_argument(
        "--eval-workflow-ids", default="",
        help="Comma-separated workflow IDs used only for parallel held-out evaluation. "
             "Use --refine-workflow-ids for parallel online training batches.",
    )
    parser.add_argument(
        "--refine-workflow-ids", default="",
        help="Comma-separated workflow IDs for parallel online batches in one wave. "
             "Their evidence is merged before one reflection/update.",
    )
    parser.add_argument(
        "--refinement-mode", choices=("standard", "trace2skill-hybrid", "constrained-repair"),
        default="standard",
        help="Post-rollout synthesis: direct reflection, Trace2Skill hybrid, or constrained local repair.",
    )
    parser.add_argument(
        "--hybrid-map-batch-size", type=int, default=8,
        help="Trace2Skill MAP records per call in trace2skill-hybrid mode (default: 8).",
    )
    parser.add_argument("--skip-guard-llm", action="store_true",
                        help="Only collect graph evidence and deterministic patches")
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Fail immediately on an empty LLM response; useful for diagnosing workflow failures.",
    )
    args = parser.parse_args()
    if args.hybrid_map_batch_size <= 0:
        parser.error("--hybrid-map-batch-size must be positive")
    if args.analysis_batch_size <= 0:
        parser.error("--analysis-batch-size must be positive")
    if args.trace2skill_batch_size <= 0:
        parser.error("--trace2skill-batch-size must be positive")
    if args.guard_retries <= 0:
        parser.error("--guard-retries must be positive")
    if args.stop_on_error:
        os.environ["SKILLMINING_STOP_ON_ERROR"] = "1"
    eval_workflow_ids = [value.strip() for value in args.eval_workflow_ids.split(",") if value.strip()]
    refine_workflow_ids = [value.strip() for value in args.refine_workflow_ids.split(",") if value.strip()]
    if refine_workflow_ids and not os.environ.get("SKILLMINING_WORKFLOW_ID", "").strip():
        # Some offline/mining helpers and older llm.py code resolve the
        # endpoint from this compatibility variable. Explicit per-agent
        # workflow routing remains authoritative for rollout/reflection.
        os.environ["SKILLMINING_WORKFLOW_ID"] = refine_workflow_ids[0]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep usage accounting process-local and persist it even when an online
    # batch fails, so partial runs remain auditable and resumable.
    from scripts.llm_usage_utils import (
        get_usage,
        merge_usage_summaries,
        reset_usage,
        split_usage_summary,
    )
    from eval_tod.response_logger import ResponseLogger

    usage_path = out_dir / "llm_usage.json"

    def _load_previous_generation_usage() -> dict:
        """Recover generation usage from a partial/completed run on resume.

        A resumed run must not discard calls already made by completed online
        batches. Testing usage is deliberately excluded because a failed test
        phase may be rerun from scratch.
        """
        if not args.resume or not usage_path.exists():
            return split_usage_summary(None, None)["generation"]
        try:
            payload = json.loads(usage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return split_usage_summary(None, None)["generation"]
        generation = payload.get("generation") if isinstance(payload, dict) else None
        if isinstance(generation, dict):
            return generation
        # Compatibility with a pre-phase-split usage file from an interrupted
        # legacy run: treat its calls as generation, never as testing.
        if isinstance(payload, dict) and "total" in payload:
            return payload
        return split_usage_summary(None, None)["generation"]

    previous_generation_usage = _load_previous_generation_usage()
    reset_usage()

    usage_phase = "generation"
    generation_usage_snapshot = previous_generation_usage

    def _persist_usage_snapshot() -> None:
        """Persist a phase-split snapshot even on interruption or API failure."""
        if usage_phase == "generation":
            generation = merge_usage_summaries(
                previous_generation_usage, get_usage(),
            )
            testing = split_usage_summary(None, None)["testing"]
        else:
            generation = generation_usage_snapshot
            testing = get_usage()
        usage_path.write_text(
            json.dumps(
                split_usage_summary(generation, testing),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    usage_fallback = _persist_usage_snapshot
    atexit.register(usage_fallback)
    _persist_usage_snapshot()
    response_logger = ResponseLogger(out_dir / "llm_responses")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "online_refine.log"), logging.StreamHandler()],
    )
    log = logging.getLogger("online_refine")
    log.info(
        "Configured workflow routing: refine=%s eval=%s refinement_mode=%s hybrid_map_batch_size=%d",
        ",".join(refine_workflow_ids) if refine_workflow_ids else "config.py",
        ",".join(eval_workflow_ids) if eval_workflow_ids else "config.py",
        args.refinement_mode, args.hybrid_map_batch_size,
    )
    train, test = load_subflow_data(args.subflow)
    if args.max_train:
        train = train[:args.max_train]
    if args.max_test:
        test = test[:args.max_test]

    state_path = out_dir / "skill_dag_state.json"
    schedule_path = out_dir / "rollout_schedule.json"
    if args.resume and args.offline_dir:
        raise ValueError("--resume and --offline-dir cannot be used together")

    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot resume: {state_path} does not exist")
        state = load_skill_dag(state_path)
        base_skill = (out_dir / "base_skill.md").read_text(encoding="utf-8")
        working_path = out_dir / "working_skill.md"
        working_skill = working_path.read_text(encoding="utf-8") if working_path.exists() else (out_dir / "skill.md").read_text(encoding="utf-8")
        base_reference = (out_dir / "base_reference.md").read_text(encoding="utf-8")
        action_rules = (out_dir / "action_rules.md").read_text(encoding="utf-8")
        slot_policies = (out_dir / "slot_policies.md").read_text(encoding="utf-8")
        log.info("Resuming after %d batches", state.get("batches_processed", 0))
        working_skill, repaired_operations = _repair_failed_skill_operations(
            out_dir, state, working_skill,
        )
        if repaired_operations:
            _write(out_dir / "working_skill.md", working_skill)
            _write(out_dir / "skill.md", working_skill)
            repaired_count = sum(len(item["operations"]) for item in repaired_operations)
            failed_count = sum(
                1 for item in repaired_operations for operation in item["operations"]
                if operation.get("error")
            )
            log.info(
                "Resume repaired %d historical skill operations across %d batches (%d still failed)",
                repaired_count, len(repaired_operations), failed_count,
            )
        # Re-materialize state-level promotions before the next rollout. This
        # also repairs runs produced before promoted branches were rendered
        # into the executable skill file.
        working_skill = merge_online_skill_additions(working_skill, state)
        _write(out_dir / "working_skill.md", working_skill)
        _write(out_dir / "skill.md", working_skill)
    else:
        offline_dir = Path(args.offline_dir) if args.offline_dir else None
        if offline_dir is not None:
            log.info("Loading existing offline backbone artifacts from %s", offline_dir)
            subgraph_path = offline_dir / "subgraph.json"
            skill_path = offline_dir / "skill.md"
            reference_path = offline_dir / "reference.md"
            if not subgraph_path.exists() or not skill_path.exists():
                raise FileNotFoundError(
                    "--offline-dir must contain at least subgraph.json and skill.md: "
                    f"{offline_dir}"
                )
            subgraph = json.loads(subgraph_path.read_text(encoding="utf-8"))
            state = initialize_skill_dag(subgraph, args.subflow)
            base_skill = skill_path.read_text(encoding="utf-8")
            base_reference = reference_path.read_text(encoding="utf-8") if reference_path.exists() else ""
            action_path = offline_dir / "action_rules.md"
            slot_path = offline_dir / "slot_policies.md"
            action_rules = action_path.read_text(encoding="utf-8") if action_path.exists() else ""
            slot_policies = slot_path.read_text(encoding="utf-8") if slot_path.exists() else ""
            _write(out_dir / "subgraph.json", json.dumps(subgraph, indent=2, ensure_ascii=False))
        else:
            log.info("Offline mining initial backbone for %s (%d train sessions)", args.subflow, len(train))
            mined = mine_subflow_skill_backbone(args.subflow, train, artifact_dir=out_dir / "offline_mining")
            state = initialize_skill_dag(mined["subgraph"], args.subflow)
            base_skill = mined["skill_md"]
            base_reference = mined["reference_md"]
            action_rules = mined.get("action_rules_md", "")
            slot_policies = mined.get("slot_policies_md", "")
            _write(out_dir / "subgraph.json", json.dumps(mined["subgraph"], indent=2, ensure_ascii=False))
        _write(out_dir / "base_skill.md", base_skill)
        working_skill = base_skill
        _write(out_dir / "base_reference.md", base_reference)
        _write(out_dir / "action_rules.md", action_rules)
        _write(out_dir / "slot_policies.md", slot_policies)
        working_skill = _checkpoint(
            out_dir, state, working_skill, base_reference,
            base_slot_policies=slot_policies, base_action_rules=action_rules,
        )

    if args.resume:
        if not schedule_path.exists():
            raise FileNotFoundError(f"Cannot resume safely: {schedule_path} does not exist")
        by_id = {str(conversation.get("convo_id", "?")): conversation for conversation in train}
        saved_schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        batches = []
        for item in saved_schedule.get("batches", []):
            samples = item.get("samples", [])
            if samples:
                batches.append([{**sample, "conversation": by_id[str(sample["conversation_id"])]} for sample in samples])
                continue
            ids = [str(value) for value in item.get("conversation_ids", [])]
            missing = [sid for sid in ids if sid not in by_id]
            if missing:
                raise RuntimeError(f"Saved schedule references sessions absent from this train split: {missing[:3]}")
            batches.append([by_id[sid] for sid in ids])
    else:
        # Roll out action-turn samples under one frozen skill snapshot.  The
        # constrained mode uses an uncertainty/locality budget; other modes
        # retain the historical full action-turn schedule.
        if args.refinement_mode == "trace2skill-hybrid":
            batches = schedule_trace2skill_conversation_batches(
                train, batch_size=args.trace2skill_batch_size, max_batches=args.max_batches,
            )
            schedule_unit = "conversation_sequence_similarity"
        elif args.refinement_mode == "constrained-repair":
            batches = schedule_constrained_repair_batches(
                train, state, batch_size=args.batch_size,
                per_transition_cap=args.per_transition_cap,
                target_selection_rate=args.target_selection_rate,
                max_batches=args.max_batches,
            )
            schedule_unit = "action_turn_uncertainty_local"
        else:
            samples = build_action_turn_samples(train)
            batches = [samples[index:index + args.batch_size]
                       for index in range(0, len(samples), args.batch_size)]
            schedule_unit = "action_turn"
        if args.max_batches is not None:
            batches = batches[:args.max_batches]
        _write(schedule_path, json.dumps({
            "subflow": args.subflow,
            "num_train_sessions": len(train),
            "num_selected_samples": sum(len(batch) for batch in batches),
            "batch_size": args.batch_size,
            "trace2skill_batch_size": args.trace2skill_batch_size,
            "per_transition_cap": args.per_transition_cap,
            "target_selection_rate": args.target_selection_rate,
            "selection_unit": schedule_unit,
            "refinement_mode": args.refinement_mode,
            "max_batches": args.max_batches,
            "batches": [
                (
                    {"batch_index": index, "conversation_ids": [
                        str(item.get("convo_id", "?")) for item in batch
                    ]}
                    if args.refinement_mode == "trace2skill-hybrid"
                    else {"batch_index": index, "samples": [
                        {k: value for k, value in item.items() if k != "conversation"}
                        for item in batch
                    ]}
                )
                for index, batch in enumerate(batches, start=1)
            ],
        }, indent=2, ensure_ascii=False))
    selected_sessions = sum(len(batch) for batch in batches)
    log.info(
        "Rollout schedule: selected=%d/%d sessions (%.1f%%; target=%.1f%%), batches=%d, unit=%s",
        selected_sessions, len(train), 100 * selected_sessions / max(len(train), 1),
        100 * args.target_selection_rate, len(batches), schedule_unit,
    )
    policy = RefinementPolicy(
        min_gold_support=args.min_gold_support,
        min_confidence=args.min_confidence,
        min_conflict_count=args.min_conflict_count,
        max_skill_branches_per_source=args.max_skill_branches_per_source,
    )
    completed = int(state.get("batches_processed", 0))
    if completed > len(batches):
        raise RuntimeError(f"State has {completed} completed batches, but current schedule has {len(batches)}")

    # Rollouts may run concurrently on independent workflow agents. The
    # parent process still localizes and applies updates in batch order.
    # The transplanted Trace2Skill path updates the skill after each complete
    # conversation batch, so batches must be serialized even with workflow
    # sharding. Standard online refinement retains its parallel waves.
    wave_size = (
        1 if args.refinement_mode == "trace2skill-hybrid"
        else (max(1, len(refine_workflow_ids)) if refine_workflow_ids else 1)
    )
    remaining_batches = batches[completed:]
    rollout_record_log = out_dir / "post_rollout_records.jsonl"
    all_rollout_records: list[dict] = []
    if rollout_record_log.exists():
        all_rollout_records = [json.loads(line) for line in
                               rollout_record_log.read_text(encoding="utf-8").splitlines()
                               if line.strip()]
    pending_hybrid = state.get("pending_hybrid_refinement")
    if (args.refinement_mode in {"trace2skill-hybrid", "constrained-repair"} and not args.skip_guard_llm
            and isinstance(pending_hybrid, dict)):
        pending_records_path = out_dir / str(pending_hybrid.get("records_path", ""))
        if not pending_records_path.is_file():
            raise FileNotFoundError(
                f"Pending hybrid refinement records are missing: {pending_records_path}"
            )
        pending_records = json.loads(pending_records_path.read_text(encoding="utf-8"))
        pass_id = str(pending_hybrid["pass_id"])
        log.info("Resuming pending iterative hybrid pass=%s records=%d",
                 pass_id, len(pending_records))
        working_skill, pass_summary = _run_refinement_pass(
            artifact_root=out_dir / "iterative_refinement" / pass_id,
            pass_id=pass_id, records=pending_records, state=state,
            working_skill=working_skill, base_reference=base_reference,
            action_rules=action_rules, slot_policies=slot_policies,
            model=args.model, response_logger=response_logger,
            workflow_ids=refine_workflow_ids, batch_size=args.batch_size,
            refinement_mode=args.refinement_mode,
            hybrid_map_batch_size=args.hybrid_map_batch_size,
            max_retries=args.guard_retries,
            ledger_path=out_dir / "refinement_ledger.jsonl", log=log,
            runner_args=args, replay_source=train,
        )
        state.setdefault("iterative_refinement_history", []).append(pass_summary)
        state["hybrid_refined_through_batch"] = int(pending_hybrid["wave_end"])
        state.pop("pending_hybrid_refinement", None)
        working_skill = _checkpoint(
            out_dir, state, working_skill, base_reference, policy,
            slot_policies, action_rules,
        )
    posthoc_group_reflection = True
    for wave_offset in range(0, len(remaining_batches), wave_size):
        wave_batches = remaining_batches[wave_offset:wave_offset + wave_size]
        wave_start = completed + wave_offset + 1
        wave_end = wave_start + len(wave_batches) - 1
        active_workflow_ids = (
            refine_workflow_ids[:len(wave_batches)]
            if refine_workflow_ids else ["config.py"]
        )
        log.info("Online wave %d-%d/%d: %d batches, workflows=%s",
                 wave_start, wave_end, len(batches), len(wave_batches),
                 ",".join(active_workflow_ids))
        log.info(
            "Workflow schedule: configured=%d, active=%d",
            len(refine_workflow_ids) if refine_workflow_ids else 1,
            len(active_workflow_ids),
        )
        rollout_items = _run_parallel_online_wave(
            args, wave_batches, refine_workflow_ids, working_skill,
            base_reference, action_rules, slot_policies, state, response_logger,
        )
        localized_parts, supervision_parts, packet_parts = [], [], []
        wave_rollout_records: list[dict] = []
        for item in rollout_items:
            batch_index = completed + item["batch_index"] + 1
            batch = item["batch"]
            turns = item["turns"]
            if args.refinement_mode == "trace2skill-hybrid":
                working_skill, hybrid_summary = _run_trace2skill_hybrid_batch(
                    batch_index=batch_index, conversations=batch, turns=turns,
                    out_dir=out_dir, working_skill=working_skill,
                    base_reference=base_reference,
                    action_rules=action_rules + "\n" + render_online_action_rules(state),
                    slot_policies=slot_policies + "\n" + render_online_slot_policies(state),
                    state=state,
                    rollout_workers=item.get("parallel_rollout_workers", 1),
                    model=args.model, response_logger=response_logger,
                    analysis_batch_size=args.analysis_batch_size,
                    map_batch_size=args.hybrid_map_batch_size,
                )
                state["batches_processed"] = batch_index
                state.setdefault("trace2skill_hybrid_history", []).append(hybrid_summary)
                working_skill = _checkpoint(
                    out_dir, state, working_skill, base_reference, policy,
                    slot_policies, action_rules,
                )
                log.info(
                    "Trace2Skill hybrid batch=%d conversations=%d rollout_workers=%d failures=%d successes=%d changes=%d",
                    batch_index, hybrid_summary["num_conversations"],
                    item.get("parallel_rollout_workers", 1),
                    hybrid_summary["failed_cases"], hybrid_summary["successful_cases"],
                    len(hybrid_summary["changelog"]),
                )
                continue
            # Trace2Skill-hybrid rolls out complete conversations. Preserve the
            # full turn trajectory on every action record so diagnosis/MAP can
            # reason over the same evidence as Trace2Skill's analyzers.
            if args.refinement_mode == "trace2skill-hybrid" and batch and "turn_index" not in batch[0]:
                conversations_by_id = {
                    str(conv.get("convo_id", "?")): conv for conv in batch
                }
                turns_by_id = defaultdict(list)
                for row in turns:
                    turns_by_id[str(row.get("convo_id", "?"))].append(row)
                for row in turns:
                    if row.get("target_type") != "action":
                        continue
                    convo_id = str(row.get("convo_id", "?"))
                    conversation = conversations_by_id.get(convo_id, {})
                    turn_index = int(row.get("turn_index", -1))
                    delexed = conversation.get("delexed", [])
                    target_turn = delexed[turn_index] if 0 <= turn_index < len(delexed) else {}
                    targets = target_turn.get("targets", []) if isinstance(target_turn, dict) else []
                    prefix_actions = [
                        str(turn.get("targets", [None, None, ""])[2])
                        for turn in delexed[:turn_index]
                        if isinstance(turn, dict)
                        and len(turn.get("targets", [])) >= 3
                        and turn.get("targets", [None, None])[1] == "take_action"
                    ]
                    source_action = prefix_actions[-1] if prefix_actions else "ROOT"
                    sample = {
                        "sample_id": f"{convo_id}:{turn_index}",
                        "conversation_id": convo_id,
                        "convo_id": convo_id,
                        "turn_index": turn_index,
                        "source_action": source_action,
                        "source_turn": None,
                        "prefix_action_sequence": prefix_actions,
                        "target_action": str(targets[2]) if len(targets) >= 3 else "",
                        "gold_slots": targets[3] if len(targets) > 3 and isinstance(targets[3], list) else [],
                        "conversation": conversation,
                    }
                    rollout_record = {
                        "sample": {key: value for key, value in sample.items() if key != "conversation"},
                        "result": row,
                        "trajectory": turns_by_id[convo_id],
                    }
                    all_rollout_records.append(rollout_record)
                    wave_rollout_records.append(rollout_record)
                    _append_jsonl(rollout_record_log, rollout_record)
            else:
                result_by_key = {
                    (str(row.get("convo_id", "?")), int(row.get("turn_index", -1))): row
                    for row in turns
                }
                for sample in batch:
                    result = result_by_key.get((str(sample.get("conversation_id", "?")), int(sample.get("turn_index", -1))))
                    if result is not None:
                        rollout_record = {
                            "sample": {key: value for key, value in sample.items() if key != "conversation"},
                            "result": result,
                        }
                        all_rollout_records.append(rollout_record)
                        wave_rollout_records.append(rollout_record)
                        _append_jsonl(rollout_record_log, rollout_record)
            _write(out_dir / "rollouts" / f"batch_{batch_index:04d}.json",
                   json.dumps(turns, indent=2, ensure_ascii=False))
            localized = localize_rollout_batch(batch, turns, state)
            rollout_supervision = _batch_rollout_supervision(batch, turns)
            evidence_packets = build_online_evidence_packets(localized)
            accumulate_online_evidence(state, evidence_packets, batch_index)
            localized_parts.append(localized)
            supervision_parts.extend(rollout_supervision)
            packet_parts.append(evidence_packets)
            _write(out_dir / "online_evidence" / f"batch_{batch_index:04d}.json", json.dumps({
                "batch_index": batch_index,
                "workflow_id": item.get("workflow_id"),
                "conversation_ids": [str(value.get("conversation_id", value.get("convo_id", "?"))) for value in batch],
                "rollout_supervision": rollout_supervision,
                "localized": localized,
                "packets": evidence_packets,
            }, indent=2, ensure_ascii=False))
        batch_index = wave_end
        batch = [conversation for item in rollout_items for conversation in item["batch"]]
        rollout_supervision = supervision_parts
        localized = {
            "events": [event for part in localized_parts for event in part["events"]],
            "num_events": sum(part["num_events"] for part in localized_parts),
            "slot_events": [event for part in localized_parts for event in part["slot_events"]],
            "num_slot_events": sum(part["num_slot_events"] for part in localized_parts),
            "action_events": [event for part in localized_parts for event in part["action_events"]],
            "num_action_events": sum(part["num_action_events"] for part in localized_parts),
        }
        evidence_packets = _merge_evidence_packets(packet_parts)
        # Retain threshold-based proposals only as diagnostics. In autonomous
        # mode they must not mutate visibility or override the optimizer's
        # resource decision.
        patches = propose_refinement_patches(state, policy)
        reflection = {"accepted": [], "rejected": []}
        # Reflection is intentionally deferred until every rollout from this
        # frozen skill snapshot has completed and post-hoc batches exist.
        if not posthoc_group_reflection and not args.skip_guard_llm:
            online_skill, online_reference = render_online_resources(state)
            reflection = autonomous_resource_reflection(
                state, rollout_supervision, working_skill,
                base_reference + "\n" + online_reference,
                action_rules + "\n" + render_online_action_rules(state),
                slot_policies + "\n" + render_online_slot_policies(state),
                args.model, max_retries=args.guard_retries,
                response_logger=response_logger,
                evidence_packets=evidence_packets,
                workflow_id=refine_workflow_ids[0] if refine_workflow_ids else None,
            )
            skill_before_sha256 = hashlib.sha256(working_skill.encode("utf-8")).hexdigest()
            proposed_skill_operations = reflection.get("proposed_skill_operations", [])
            working_skill, skill_operations = apply_dynamic_skill_operations(
                working_skill, proposed_skill_operations,
            )
            materialized = {
                (
                    str(item.get("resource", "")),
                    str(item.get("edge_id", "")),
                    str(item.get("action", "")),
                )
                for item in skill_operations
                if item.get("applied")
            }
            semantic_fallbacks = []
            for update in reflection.get("accepted", []):
                resource = str(update.get("resource", ""))
                if resource not in {"transition_guard", "action_node", "action_rule"}:
                    continue
                materialized_resource = "action_rule" if resource == "action_node" else resource
                key = (
                    materialized_resource,
                    str(update.get("edge_id", "")),
                    str(update.get("action", "")),
                )
                edge_was_materialized = (
                    resource == "transition_guard"
                    and any(item[1] == key[1] for item in materialized if item[1])
                )
                if key not in materialized and not edge_was_materialized:
                    fallback = dict(update)
                    if resource == "action_node":
                        # The existing editor stores node placement/role in the
                        # action-card region. Keep the semantic decision in
                        # the ledger while materializing it through that API.
                        fallback["resource"] = "action_rule"
                    semantic_fallbacks.append(fallback)
            if semantic_fallbacks:
                working_skill, fallback_operations = apply_working_skill_operations(
                    working_skill, state, semantic_fallbacks,
                )
                skill_operations.extend(fallback_operations)
            reflection["requested_skill_operations"] = proposed_skill_operations
            reflection["semantic_fallback_updates"] = semantic_fallbacks
            reflection["skill_operations"] = skill_operations
            reflection["working_skill_before_sha256"] = skill_before_sha256
            reflection["working_skill_after_sha256"] = hashlib.sha256(working_skill.encode("utf-8")).hexdigest()
            _write(out_dir / "autonomous_reflection" / f"batch_{batch_index:04d}.json",
                   json.dumps(reflection, indent=2, ensure_ascii=False))
            _append_jsonl(out_dir / "refinement_ledger.jsonl", {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "batch_index": batch_index,
                "conversation_ids": [str(item.get("conversation_id", item.get("convo_id", "?"))) for item in batch],
                "model_decision": reflection.get("model_decision", ""),
                "no_update_reason": reflection.get("model_no_update_reason", ""),
                "lookups": reflection.get("lookups", []),
                "retrieved_sections": [
                    {"resource": item.get("resource"), "title": item.get("title")}
                    for item in reflection.get("retrieved_resources", [])
                ],
                "accepted_updates": reflection.get("accepted", []),
                "rejected_updates": reflection.get("rejected", []),
                "skill_operations": reflection.get("skill_operations", []),
                "working_skill_before_sha256": reflection.get("working_skill_before_sha256", ""),
                "working_skill_after_sha256": reflection.get("working_skill_after_sha256", ""),
                "error": reflection.get("error", "") or reflection.get("planner_error", ""),
            })
            if reflection.get("error"):
                io_debug = reflection.get("reflection_io", {})
                log.warning(
                    "  autonomous reflection failed after retries "
                    "(input_chars=%s input_tokens~=%s output_chars=%s output_tokens~=%s): %s",
                    io_debug.get("input_chars", reflection.get("prompt_chars", "?")),
                    io_debug.get("input_token_estimate", "?"),
                    io_debug.get("last_output_chars", "?"),
                    io_debug.get("last_output_token_estimate", "?"),
                    reflection["error"],
                )
            else:
                log.info(
                    "  autonomous reflection planner_lookups=%d retrieved_sections=%d "
                    "accepted=%d rejected=%d decision=%s reason=%s prompt_chars=%d",
                    len(reflection.get("lookups", [])), len(reflection.get("retrieved_resources", [])),
                    len(reflection.get("accepted", [])), len(reflection.get("rejected", [])),
                    reflection.get("model_decision", "missing"),
                    reflection.get("model_no_update_reason", "")[:180], reflection.get("prompt_chars", 0),
                )
        if (args.refinement_mode in {"trace2skill-hybrid", "constrained-repair"} and not args.skip_guard_llm
                and wave_rollout_records):
            pass_id = f"wave_{wave_start:04d}_{wave_end:04d}"
            pass_root = out_dir / "iterative_refinement" / pass_id
            records_path = pass_root / "records.json"
            _write(records_path, json.dumps(wave_rollout_records, indent=2, ensure_ascii=False))
            state["pending_hybrid_refinement"] = {
                "pass_id": pass_id,
                "wave_start": wave_start,
                "wave_end": wave_end,
                "records_path": str(records_path.relative_to(out_dir)),
            }
            # Persist rollout progress before LLM refinement so resume never
            # needs to repeat this frozen-snapshot wave.
            working_skill = _checkpoint(
                out_dir, state, working_skill, base_reference, policy,
                slot_policies, action_rules,
            )
            working_skill, pass_summary = _run_refinement_pass(
                artifact_root=pass_root, pass_id=pass_id,
                records=wave_rollout_records, state=state,
                working_skill=working_skill, base_reference=base_reference,
                action_rules=action_rules, slot_policies=slot_policies,
                model=args.model, response_logger=response_logger,
                workflow_ids=refine_workflow_ids, batch_size=args.batch_size,
                refinement_mode=args.refinement_mode,
                hybrid_map_batch_size=args.hybrid_map_batch_size,
                max_retries=args.guard_retries,
                ledger_path=out_dir / "refinement_ledger.jsonl", log=log,
                runner_args=args, replay_source=train,
            )
            state.setdefault("iterative_refinement_history", []).append(pass_summary)
            state["hybrid_refined_through_batch"] = wave_end
            state.pop("pending_hybrid_refinement", None)

        working_skill = _checkpoint(
            out_dir, state, working_skill, base_reference, policy,
            slot_policies, action_rules,
        )
        _persist_usage_snapshot()
        _write(out_dir / "batch_diagnostics" / f"batch_{batch_index:04d}.json", json.dumps({
            "batch_index": batch_index,
            "conversation_ids": [str(item.get("convo_id", "?")) for item in batch],
            "localization": localized,
            "evidence_packets": evidence_packets,
            "patches": patches,
            "autonomous_reflection": reflection,
        }, indent=2, ensure_ascii=False))
        summary = summarize_refinement_state(state, policy)
        log.info(
            "  localized=%d diagnostic_patches=%d autonomous_updates=%d candidate_branches=%d blockers=%s",
            localized["num_events"], len(patches), len(reflection["accepted"]),
            summary["num_candidate_branches"], summary["blocker_counts"],
        )

    # Post-hoc evidence organization: outcomes, trajectory prefixes and local
    # graph structure are now all available. Batch reports diagnose causes;
    # group reflections (<=16 reports) make the final edits.
    if all_rollout_records:
        _write(out_dir / "post_rollout_records.json", json.dumps(
            all_rollout_records, indent=2, ensure_ascii=False))
    elif (out_dir / "post_rollout_records.json").exists():
        all_rollout_records = json.loads(
            (out_dir / "post_rollout_records.json").read_text(encoding="utf-8"))

    # Compatibility for a hybrid run created before iterative wave refinement
    # existed. Its completed rollouts cannot be replayed historically without
    # new model calls, so resume them once as a catch-up pass and continue.
    if (args.refinement_mode in {"trace2skill-hybrid", "constrained-repair"} and all_rollout_records
            and not args.skip_guard_llm
            and not state.get("iterative_refinement_history")
            and not state.get("posthoc_reflection_complete", False)):
        log.info("Legacy hybrid resume detected; running one catch-up refinement pass")
        working_skill, pass_summary = _run_refinement_pass(
            artifact_root=out_dir, pass_id="legacy_resume_catchup",
            records=all_rollout_records, state=state, working_skill=working_skill,
            base_reference=base_reference, action_rules=action_rules,
            slot_policies=slot_policies, model=args.model,
            response_logger=response_logger, workflow_ids=refine_workflow_ids,
            batch_size=args.batch_size, refinement_mode=args.refinement_mode,
            hybrid_map_batch_size=args.hybrid_map_batch_size,
            max_retries=args.guard_retries,
            ledger_path=out_dir / "refinement_ledger.jsonl", log=log,
            runner_args=args, replay_source=train,
        )
        state.setdefault("iterative_refinement_history", []).append(pass_summary)
        state["posthoc_reflection_complete"] = True
        working_skill = _checkpoint(
            out_dir, state, working_skill, base_reference, policy,
            slot_policies, action_rules,
        )

    if (args.refinement_mode == "standard" and all_rollout_records
            and not args.skip_guard_llm
            and not state.get("posthoc_reflection_complete", False)):
        working_skill, pass_summary = _run_refinement_pass(
            artifact_root=out_dir, pass_id="posthoc", records=all_rollout_records,
            state=state, working_skill=working_skill,
            base_reference=base_reference, action_rules=action_rules,
            slot_policies=slot_policies, model=args.model,
            response_logger=response_logger, workflow_ids=refine_workflow_ids,
            batch_size=args.batch_size, refinement_mode=args.refinement_mode,
            hybrid_map_batch_size=args.hybrid_map_batch_size,
            max_retries=args.guard_retries,
            ledger_path=out_dir / "refinement_ledger.jsonl", log=log,
            runner_args=args, replay_source=train,
        )
        state["posthoc_failed_batch_ids"] = pass_summary["failed_batch_ids"]
        state["posthoc_failed_reflection_groups"] = pass_summary["failed_reflection_groups"]
        state["posthoc_reflection_complete"] = True
        working_skill = _checkpoint(
            out_dir, state, working_skill, base_reference, policy,
            slot_policies, action_rules,
        )

    # Freeze all mining/refinement calls before the held-out evaluation so the
    # two budgets remain auditable in the final artifact.
    generation_usage = merge_usage_summaries(
        previous_generation_usage, get_usage(),
    )
    generation_usage_snapshot = generation_usage
    reset_usage()
    usage_phase = "testing"
    _persist_usage_snapshot()
    log.info("Frozen test evaluation on %d held-out sessions", len(test))
    final_agent = _build_agent(args, working_skill, base_reference, action_rules, slot_policies, state,
                               response_logger=response_logger)
    result = evaluate_agent_on_subflow(
        final_agent, test, "online_refined", args.subflow, save_dir=out_dir,
        eval_workflow_ids=eval_workflow_ids,
        skip_utterance_eval=args.skip_utterance_eval,
    )
    log.info("Held-out evaluation merged successfully; writing final result artifacts")
    # Parallel evaluation runs in forked processes. Merge their isolated
    # usage snapshots with the parent process' mining/refinement usage.
    worker_usage = result.pop("_evaluation_worker_usage", [])
    _write(out_dir / "online_refine_result.json", json.dumps(result, indent=2, ensure_ascii=False))
    testing_usage = get_usage()
    if worker_usage:
        from scripts.llm_usage_utils import merge_usage_summaries
        testing_usage = merge_usage_summaries(testing_usage, *worker_usage)
    usage = split_usage_summary(generation_usage, testing_usage)
    atexit.unregister(usage_fallback)
    usage_path.write_text(
        json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result["llm_usage"] = usage
    _write(out_dir / "online_refine_result.json", json.dumps(result, indent=2, ensure_ascii=False))
    working_skill = _checkpoint(
        out_dir, state, working_skill, base_reference, policy,
        slot_policies, action_rules,
    )
    log.info("Final AST=%.4f action=%.4f slot=%.4f", result["ast_cds"]["ast_joint"], result["ast_cds"]["ast_action_name"], result["ast_cds"]["ast_slot_value"])
    # Keep single-subflow runs consistent with the full launcher: expose the
    # same weighted aggregate schema even when there is only one record.
    try:
        from scripts.aggregate_subflow_results import _records_from_summary, _weighted_average
        records = _records_from_summary(out_dir / "online_refine_result.json")
        if records:
            aggregate_payload = {
                "protocol": "independent_subflow_runs",
                "summary_files": [str(out_dir / "online_refine_result.json")],
                "records": records,
                "aggregate": _weighted_average(records),
                "llm_usage": usage,
            }
            _write(out_dir / "aggregate_online_refine.json",
                   json.dumps(aggregate_payload, indent=2, ensure_ascii=False))
            log.info("Aggregate written to %s", out_dir / "aggregate_online_refine.json")
    except Exception as exc:
        log.warning("Could not write single-subflow aggregate: %s", exc)


if __name__ == "__main__":
    main()
