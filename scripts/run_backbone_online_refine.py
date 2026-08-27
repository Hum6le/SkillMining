#!/usr/bin/env python3
"""Online refinement for an offline graph-compiled ABCD skill.

The runner intentionally keeps the offline arborescence immutable. Training
rollouts update edge evidence on the training split only; selected local guards
are appended to the runtime skill only after an evidence-based promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from awm import MemoryStore, WorkflowStore
from eval_tod.abcd.agent import ABCDAgent
from skill_mining.online_refinement import (
    RefinementPolicy,
    autonomous_resource_reflection,
    apply_dynamic_skill_operations,
    apply_working_skill_operations,
    initialize_skill_dag,
    load_skill_dag,
    localize_rollout_batch,
    propose_refinement_patches,
    render_online_resources,
    render_online_action_rules,
    render_online_slot_policies,
    save_skill_dag,
    schedule_contrastive_batches,
    summarize_refinement_state,
)
from scripts.run_subflow_eval import (
    MODEL,
    evaluate_agent_on_subflow,
    load_subflow_data,
    mine_subflow_skill_backbone,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    for conversation in conversations:
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


def _build_agent(args, working_skill: str, base_reference: str, action_rules: str,
                 slot_policies: str, state: dict) -> ABCDAgent:
    _, online_reference = render_online_resources(state)
    workflow = WorkflowStore()
    workflow.update(working_skill)
    return ABCDAgent(
        model=args.model,
        workflow=workflow,
        memory=MemoryStore(),
        reference_text=base_reference.rstrip() + "\n\n" + online_reference,
        action_rules_text=action_rules.rstrip() + "\n\n" + render_online_action_rules(state),
        slot_policies_text=slot_policies.rstrip() + "\n\n" + render_online_slot_policies(state),
        reference_top_k=args.reference_top_k,
        reference_max_chars=args.reference_max_chars,
        expose_scenario_labels=False,
    )


def _checkpoint(
    out_dir: Path, state: dict, working_skill: str, base_reference: str,
    policy: RefinementPolicy | None = None, base_slot_policies: str = "", base_action_rules: str = "",
) -> None:
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
        help="Representative rollout sessions per online update batch (default: 8).",
    )
    parser.add_argument("--per-transition-cap", type=int, default=3)
    parser.add_argument(
        "--target-selection-rate", type=float, default=0.30,
        help="Target fraction of train sessions selected for online refinement (default: 0.30).",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--reference-top-k", type=int, default=3)
    parser.add_argument("--reference-max-chars", type=int, default=1800)
    parser.add_argument("--min-gold-support", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--min-conflict-count", type=int, default=2)
    parser.add_argument("--max-skill-branches-per-source", type=int, default=3)
    parser.add_argument("--guard-retries", type=int, default=3,
                        help="Retries for one local guard-induction call")
    parser.add_argument("--skip-guard-llm", action="store_true",
                        help="Only collect graph evidence and deterministic patches")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "online_refine.log"), logging.StreamHandler()],
    )
    log = logging.getLogger("online_refine")
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
        _checkpoint(out_dir, state, working_skill, base_reference, base_slot_policies=slot_policies, base_action_rules=action_rules)

    if args.resume:
        if not schedule_path.exists():
            raise FileNotFoundError(f"Cannot resume safely: {schedule_path} does not exist")
        by_id = {str(conversation.get("convo_id", "?")): conversation for conversation in train}
        saved_schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        batches = []
        for item in saved_schedule.get("batches", []):
            ids = [str(value) for value in item.get("conversation_ids", [])]
            missing = [sid for sid in ids if sid not in by_id]
            if missing:
                raise RuntimeError(f"Saved schedule references sessions absent from this train split: {missing[:3]}")
            batches.append([by_id[sid] for sid in ids])
    else:
        batches = schedule_contrastive_batches(
            train, state, batch_size=args.batch_size,
            per_transition_cap=args.per_transition_cap,
            target_selection_rate=args.target_selection_rate,
            max_batches=args.max_batches,
        )
        _write(schedule_path, json.dumps({
            "subflow": args.subflow,
            "num_train_sessions": len(train),
            "num_selected_sessions": sum(len(batch) for batch in batches),
            "selection_rate": round(sum(len(batch) for batch in batches) / max(len(train), 1), 6),
            "batch_size": args.batch_size,
            "per_transition_cap": args.per_transition_cap,
            "target_selection_rate": args.target_selection_rate,
            "max_batches": args.max_batches,
            "batches": [
                {"batch_index": index, "conversation_ids": [str(item.get("convo_id", "?")) for item in batch]}
                for index, batch in enumerate(batches, start=1)
            ],
        }, indent=2, ensure_ascii=False))
    selected_sessions = sum(len(batch) for batch in batches)
    log.info(
        "Contrastive rollout schedule: selected=%d/%d sessions (%.1f%%; target=%.1f%%), batches=%d, per_transition_cap=%d",
        selected_sessions, len(train), 100 * selected_sessions / max(len(train), 1),
        100 * args.target_selection_rate, len(batches), args.per_transition_cap,
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

    for batch_index, batch in enumerate(batches[completed:], start=completed + 1):
        log.info("Online batch %d/%d: %d sessions", batch_index, len(batches), len(batch))
        agent = _build_agent(args, working_skill, base_reference, action_rules, slot_policies, state)
        turns = []
        for conversation in batch:
            turns.extend(agent.predict_all_turns(conversation, predict_actions=True, verbose=False))
        _write(out_dir / "rollouts" / f"batch_{batch_index:04d}.json", json.dumps(turns, indent=2, ensure_ascii=False))
        localized = localize_rollout_batch(batch, turns, state)
        # Retain threshold-based proposals only as diagnostics. In autonomous
        # mode they must not mutate visibility or override the optimizer's
        # resource decision.
        patches = propose_refinement_patches(state, policy)
        reflection = {"accepted": [], "rejected": []}
        if not args.skip_guard_llm:
            online_skill, online_reference = render_online_resources(state)
            reflection = autonomous_resource_reflection(
                state, _batch_rollout_supervision(batch, turns), working_skill,
                base_reference + "\n" + online_reference,
                action_rules + "\n" + render_online_action_rules(state),
                slot_policies + "\n" + render_online_slot_policies(state),
                args.model, max_retries=args.guard_retries,
            )
            skill_before_sha256 = hashlib.sha256(working_skill.encode("utf-8")).hexdigest()
            proposed_skill_operations = reflection.get("proposed_skill_operations", [])
            working_skill, skill_operations = apply_dynamic_skill_operations(
                working_skill, proposed_skill_operations,
            )
            reflection["requested_skill_operations"] = proposed_skill_operations
            reflection["skill_operations"] = skill_operations
            reflection["working_skill_before_sha256"] = skill_before_sha256
            reflection["working_skill_after_sha256"] = hashlib.sha256(working_skill.encode("utf-8")).hexdigest()
            _write(out_dir / "autonomous_reflection" / f"batch_{batch_index:04d}.json",
                   json.dumps(reflection, indent=2, ensure_ascii=False))
            _append_jsonl(out_dir / "refinement_ledger.jsonl", {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "batch_index": batch_index,
                "conversation_ids": [str(item.get("convo_id", "?")) for item in batch],
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
                log.warning(
                    "  autonomous reflection failed after retries (prompt_chars=%s): %s",
                    reflection.get("prompt_chars", "?"), reflection["error"],
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
        _checkpoint(out_dir, state, working_skill, base_reference, policy, slot_policies, action_rules)
        _write(out_dir / "batch_diagnostics" / f"batch_{batch_index:04d}.json", json.dumps({
            "batch_index": batch_index,
            "conversation_ids": [str(item.get("convo_id", "?")) for item in batch],
            "localization": localized,
            "patches": patches,
            "autonomous_reflection": reflection,
        }, indent=2, ensure_ascii=False))
        summary = summarize_refinement_state(state, policy)
        log.info(
            "  localized=%d diagnostic_patches=%d autonomous_updates=%d candidate_branches=%d blockers=%s",
            localized["num_events"], len(patches), len(reflection["accepted"]),
            summary["num_candidate_branches"], summary["blocker_counts"],
        )

    log.info("Frozen test evaluation on %d held-out sessions", len(test))
    final_agent = _build_agent(args, working_skill, base_reference, action_rules, slot_policies, state)
    result = evaluate_agent_on_subflow(final_agent, test, "online_refined", args.subflow, save_dir=out_dir)
    _write(out_dir / "online_refine_result.json", json.dumps(result, indent=2, ensure_ascii=False))
    _checkpoint(out_dir, state, working_skill, base_reference, policy, slot_policies, action_rules)
    log.info("Final AST=%.4f action=%.4f slot=%.4f", result["ast_cds"]["ast_joint"], result["ast_cds"]["ast_action_name"], result["ast_cds"]["ast_slot_value"])


if __name__ == "__main__":
    main()
