"""Offline K0 + native ABCD AWM Online, with the shared ours runtime.

Updates are sequential; target rollouts inside a batch can use independent
workflows in parallel. No ours diagnosis/reflection or graph updater is used.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing as mp
import re
import shutil
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

from scripts.abcd_runtime_utils import abcd_runtime_options, build_abcd_runtime_agent

PROTOCOL = "offline-awm-online-v3-writable-action-cards"
V2_PROTOCOL = "offline-awm-online-v2-target-complete-feedback"
RESOURCE_START = "<!-- OFFLINE_AWM_RESOURCE_START -->"
RESOURCE_END = "<!-- OFFLINE_AWM_RESOURCE_END -->"
RESOURCE_USE = f"""{RESOURCE_START}
## Offline and Online Resources
- The workflow starts from a trace-mined offline skill and is maintained by AWM Online.
- Propose plausible canonical candidate actions in the resource-planning step.
  The runtime's `retrieve_action_card` supplies unified Action Cards containing
  procedures from action_rules.md and ordered binding rules from slot_policies.md.
  Select the action before grounding its arguments. Do not invent separate
  retrieve_action_rule or retrieve_slot_policy calls: these are not exposed tools.
- Use `retrieve_reference` for uncertain transitions and supporting dialogue evidence.
  Queries must be concise and grounded in the current dialogue.
- The shared runtime also retrieves AWM successful-prefix exemplars from memory.
  Only sampled action turns in each stored prefix were jointly validated; do not
  assume every historical turn was evaluated. Examples are procedural evidence,
  never default names, emails, order IDs or other private slot values.
- AWM updates may refine both the workflow and Action Cards using training feedback. Preserve the
  canonical action schema and obtain every argument from the current dialogue.
{RESOURCE_END}
"""


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _scope_differences(saved, current):
    """Readable resume diagnostics without dumping dialogue/resource contents."""
    differences = {}
    for key in sorted(set(saved) | set(current)):
        old, new = saved.get(key), current.get(key)
        if old == new:
            continue
        if key == "schedule":
            def summary(schedule):
                return {"batches": len(schedule or []), "action_turns": sum(len(b) for b in schedule or [])}
            differences[key] = {"saved": summary(old), "current": summary(new),
                                "note": "Batch membership/order differs even when counts match"}
        elif key == "runtime" and isinstance(old, dict) and isinstance(new, dict):
            differences[key] = {name: {"saved": old.get(name), "current": new.get(name)}
                                for name in sorted(set(old) | set(new)) if old.get(name) != new.get(name)}
        else:
            differences[key] = {"saved": old, "current": new}
    return differences


def migrate_v2_checkpoint(state, scope, resources, checkpoint_path):
    """Explicit, backed-up continuation, not retroactive v3 training."""
    differences = _scope_differences(state["scope"], scope)
    if state["scope"].get("protocol") != V2_PROTOCOL or set(differences) != {"protocol"}:
        raise ValueError("v2->v3 migration requires v2 and protocol-only differences: "
                         + json.dumps(differences, ensure_ascii=False))
    completed = state.get("completed_batches")
    if (type(completed) is not int or not 0 <= completed <= len(scope["schedule"])
            or not isinstance(state.get("workflow"), str) or not state["workflow"].strip()
            or not isinstance(state.get("exemplars"), list)):
        raise ValueError("Invalid v2 checkpoint batch/workflow/memory state")
    old_scope_hash = _hash(state["scope"])
    result_path = checkpoint_path.parent / "result.json"
    if result_path.is_file() and _read(result_path).get("scope_hash") != old_scope_hash:
        raise ValueError("Existing v2 result does not match the migration source checkpoint")
    backup = checkpoint_path.parent / "migration_backups" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint_path, backup / checkpoint_path.name)
    # Preserve any prior final artifact before re-evaluation can replace it.
    for path in checkpoint_path.parent.iterdir():
        if path.is_file() and path != checkpoint_path:
            shutil.copy2(path, backup / path.name)
    cards = {name: resources[name] for name in ("action_rules.md", "slot_policies.md")}
    record = {"source_protocol": V2_PROTOCOL, "target_protocol": PROTOCOL,
              "source_scope_hash": old_scope_hash, "target_scope_hash": _hash(scope),
              "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "source_workflow_hash": _hash(state["workflow"].strip()), "initial_cards_hash": _hash(cards),
              "completed_v2_batches": completed,
              "completed_v2_action_turns": sum(len(b) for b in scope["schedule"][:completed]),
              "first_v3_batch_index": completed, "first_v3_batch_number": completed + 1,
              "planned_v3_batches": len(scope["schedule"]) - completed,
              "historical_usage": copy.deepcopy(state.get("usage")),
              "backup_dir": str(backup), "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
              "interpretation": "mixed v2->v3 continuation; prior batches were not retrained"}
    updated = copy.deepcopy(state)
    updated.update(scope=scope, cards=cards, migration=record)
    _save(backup / "migration.json", record)
    _save(checkpoint_path, updated)  # Commit only after the backup is complete.
    print(f"Migrated offline_awm v2->v3 after {completed} batches; backup={backup}", flush=True)
    return updated


def cached_rollout_matches(cached, identity, state):
    if not cached:
        return False
    if cached.get("identity") == identity:
        return True
    migration = state.get("migration") or {}
    # The boundary rollout used the same v2 workflow, original cards, data and
    # runtime. Only its subsequent induction changes; reuse at this boundary only.
    if (identity["index"] == migration.get("first_v3_batch_index")
            and identity["workflow_hash"] == migration.get("source_workflow_hash")
            and identity["cards_hash"] == migration.get("initial_cards_hash")):
        legacy = {"scope_hash": migration["source_scope_hash"],
                  "workflow_hash": identity["workflow_hash"], "index": identity["index"]}
        return cached.get("identity") == legacy
    return False


def _without_resource_use(text):
    """Remove adapter-owned scaffolding, not learned rules between bad markers.

    Markers are metadata, not model-authored edit anchors. In particular, never
    delete everything after an unmatched START or between reversed markers.
    """
    text = re.sub(r"<!--\s*OFFLINE_AWM_RESOURCE_(?:START|END)\s*-->", "", text)
    # The reserved section contains adapter instructions only. Stop at ANY new
    # markdown heading, preserving subsequent learned patterns even with ###.
    text = re.sub(r"(?ms)^## Offline and Online Resources[^\n]*\n.*?(?=^#{1,6}\s|\Z)", "", text)
    return text.strip()


def _with_resource_use(text):
    # Always reconstruct exactly one compiler-owned footer. LLM formatting
    # damage must not invalidate otherwise usable workflow/card induction.
    return _without_resource_use(text) + "\n\n" + RESOURCE_USE


def feedback_prefixes(batch):
    """Native AWM sees gold only for sampled targets, not unsampled future turns.

    Keep original turn indices and observed historical utterances. Historical
    action annotations outside this batch are not counted as feedback targets.
    """
    grouped = {}
    for sample in batch:
        cid = str(sample["conversation_id"])
        grouped.setdefault(cid, {"conversation": sample["conversation"], "indices": set()})["indices"].add(int(sample["turn_index"]))
    prefixes = []
    for item in grouped.values():
        indices = item["indices"]
        conversation = copy.deepcopy(item["conversation"])
        end = max(indices) + 1
        conversation["delexed"] = conversation.get("delexed", [])[:end]
        if isinstance(conversation.get("original"), list):
            conversation["original"] = conversation["original"][:end]
        for index, turn in enumerate(conversation["delexed"]):
            targets = turn.get("targets") or []
            if index not in indices and len(targets) >= 3 and targets[1] == "take_action":
                turn["targets"] = [targets[0], "observed_history", "", [], *targets[4:]]
        conversation.pop("scenario", None)  # no scenario-label/answer side channel
        prefixes.append(conversation)
    return prefixes


def update_card_resources(cards, updates):
    """Deterministic action-keyed replacement, independent of ours graph writers."""
    result = dict(cards)
    for update in updates:
        action = update["action"]
        for key, filename in (("action_rule", "action_rules.md"), ("slot_policy", "slot_policies.md")):
            if key not in update:
                continue
            text = result[filename]
            matches = list(re.finditer(r"(?m)^####\s+`([^`]+)`\s*$", text))
            spans = [(m.start(), matches[i + 1].start() if i + 1 < len(matches) else len(text))
                     for i, m in enumerate(matches) if m.group(1).strip() == action]
            replacement = f"#### `{action}`\n{update[key].strip()}\n\n"
            if spans:
                # Replace the first section and remove obsolete duplicate cards.
                for i in range(len(spans) - 1, -1, -1):
                    start, end = spans[i]
                    text = text[:start] + (replacement if i == 0 else "") + text[end:]
            else:
                text = text.rstrip() + "\n\n" + replacement
            result[filename] = text.rstrip() + "\n"
    return result


def run_offline_awm(args, subflow, flow_dir, workflow_ids):
    from awm import MemoryStore, WorkflowStore
    from eval_tod.abcd.agent import compute_ast_from_turn_results, _AWM_WORKFLOW_RESOURCE_SECTION
    from eval_tod.schemas import Prediction
    from eval_tod.response_logger import ResponseLogger
    from skill_mining.online_refinement import build_action_turn_samples
    from scripts.run_backbone_online_refine import _rollout_online_batch
    from scripts.run_subflow_eval import load_subflow_data, evaluate_agent_on_subflow
    from scripts.llm_usage_utils import get_usage, reset_usage, merge_usage_summaries, split_usage_summary

    output, offline = flow_dir / "offline_awm", flow_dir / "offline_artifact"
    train, test = load_subflow_data(subflow)
    if args.max_train is not None:
        train = train[:args.max_train]
    if args.max_test is not None:
        test = test[:args.max_test]
    resources = {name: (offline / name).read_text(encoding="utf-8")
                 for name in ("skill.md", "reference.md", "action_rules.md", "slot_policies.md")}
    samples = build_action_turn_samples(train)
    batches = [samples[index:index + args.batch_size] for index in range(0, len(samples), args.batch_size)]
    if args.max_batches is not None:
        batches = batches[:args.max_batches]
    scope = {"protocol": PROTOCOL, "subflow": subflow, "runtime": abcd_runtime_options(args),
             "resource_hash": _hash(resources), "train_hash": _hash(train), "test_hash": _hash(test),
             "schedule": [[sample["sample_id"] for sample in batch] for batch in batches]}
    checkpoint_path = output / "awm_state.json"
    if args.resume and checkpoint_path.is_file():
        state = _read(checkpoint_path)
        if state["scope"] != scope:
            if getattr(args, "migrate_offline_awm_v2", False):
                state = migrate_v2_checkpoint(state, scope, resources, checkpoint_path)
        if state["scope"] != scope:
            differences = _scope_differences(state["scope"], scope)
            advice = ("For an explicit mixed v2->v3 continuation use --resume --migrate-offline-awm-v2; otherwise start a fresh arm."
                      if "protocol" in differences else
                      "Restore the original data/resources/runtime/batch settings; do not bypass checkpoint validation.")
            raise ValueError("Offline-AWM checkpoint resources/data/runtime/schedule differ: "
                             + json.dumps(differences, ensure_ascii=False) + ". " + advice)
    else:
        if checkpoint_path.exists():
            raise ValueError("Offline-AWM checkpoint exists; use --resume")
        state = {"scope": scope, "completed_batches": 0,
                 "workflow": _with_resource_use(resources["skill.md"]), "exemplars": [], "usage": None,
                 "cards": {name: resources[name] for name in ("action_rules.md", "slot_policies.md")}}
    if set(state.get("cards", {})) != {"action_rules.md", "slot_policies.md"}:
        raise ValueError("Offline-AWM checkpoint is missing its evolving Action Cards")
    if args.resume and (output / "result.json").is_file():
        result_scope = _read(output / "result.json").get("scope_hash")
        if result_scope == _hash(scope):
            return
        if result_scope != (state.get("migration") or {}).get("source_scope_hash"):
            raise ValueError("Completed offline-AWM result has incompatible scope")
        print("Existing v2 test result is backed up; evaluating the migrated continuation anew", flush=True)
    if not 0 <= int(state["completed_batches"]) <= len(batches):
        raise ValueError("Offline-AWM completed batch count is outside the saved schedule")
    output.mkdir(parents=True, exist_ok=True)
    old_mode = state.get("induction_edit_mode", "rewrite")
    edit_mode = getattr(args, "offline_awm_edit_mode", None) or old_mode
    if edit_mode != old_mode:
        if checkpoint_path.is_file():
            backup = output / "edit_mode_backups" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
            backup.mkdir(parents=True)
            shutil.copy2(checkpoint_path, backup / checkpoint_path.name)
        else:
            backup = None
        state.setdefault("edit_interface_history", []).append({"from": old_mode, "to": edit_mode,
            "first_batch_index": state["completed_batches"], "first_batch_number": state["completed_batches"] + 1,
            "backup_dir": str(backup) if backup else None, "changed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(f"AWM edit interface {old_mode}->{edit_mode} from batch {state['completed_batches'] + 1}", flush=True)
    state["induction_edit_mode"] = edit_mode
    _save(checkpoint_path, state)
    workflow, memory = WorkflowStore(), MemoryStore()
    workflow.replace(state["workflow"])
    memory.exemplars = copy.deepcopy(state["exemplars"])
    ids = workflow_ids or [None]
    live_resources = {**resources, **state["cards"]}
    response_logger = ResponseLogger(output / "llm_responses" / uuid4().hex)

    def make_agent(workflow_id, snapshot=False):
        current_workflow, current_memory = workflow, memory
        if snapshot:
            current_workflow, current_memory = WorkflowStore(), MemoryStore()
            current_workflow.replace(workflow.text)
            current_memory.exemplars = copy.deepcopy(memory.exemplars)
        return build_abcd_runtime_agent(args, workflow=current_workflow, memory=current_memory,
                 reference_text=live_resources["reference.md"], action_rules_text=live_resources["action_rules.md"],
                 slot_policies_text=live_resources["slot_policies.md"], workflow_id=workflow_id,
                 response_logger=response_logger)

    previous_usage = state.get("usage")
    reset_usage()
    for index in range(state["completed_batches"], len(batches)):
        batch = batches[index]
        print(f"offline_awm {subflow}: batch {index + 1}/{len(batches)} action-turns={len(batch)}", flush=True)
        before = workflow.text
        before_state = copy.deepcopy(state)
        batch_path = output / "batches" / f"batch_{index + 1:04d}.json"
        try:
            identity = {"scope_hash": _hash(scope), "workflow_hash": _hash(before),
                        "cards_hash": _hash(state["cards"]), "index": index}
            cached = _read(batch_path) if args.resume and batch_path.is_file() else None
            if args.resume and cached_rollout_matches(cached, identity, state):
                rows = cached["rows"]
                print(f"offline_awm {subflow}: reused cached rollout batch {index + 1}", flush=True)
            else:
                shards = [batch[offset::len(ids)] for offset in range(len(ids))]
                tasks = [(wid, shard) for wid, shard in zip(ids, shards) if shard]
                def predict(item):
                    wid, shard = item
                    return _rollout_online_batch(make_agent(wid, snapshot=True), shard)
                with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                    rows = [row for result in pool.map(predict, tasks) for row in result]
                _save(batch_path, {"identity": identity, "rows": rows})
            expected = {(str(item["conversation_id"]), int(item["turn_index"])) for item in batch}
            actual = [(str(row["convo_id"]), int(row["turn_index"])) for row in rows]
            if set(actual) != expected or len(actual) != len(expected):
                raise ValueError("Offline-AWM rollout must return every sampled action turn exactly once")
            views = feedback_prefixes(batch)
            metrics = compute_ast_from_turn_results(views, rows)
            last_rows = {}
            for row in sorted(rows, key=lambda row: int(row["turn_index"])):
                last_rows[str(row["convo_id"])] = row
            predictions = [Prediction(dialogue_id=f"abcd-{conv['convo_id']}", inform_slots={}, request_slots={},
                           booking={}, response_text=str(last_rows[str(conv["convo_id"])].get("prediction") or "")) for conv in views]
            updater = make_agent(ids[index % len(ids)])
            resource_context = "\n\n".join(f"## {name}\n{text}" for name, text in live_resources.items() if name != "skill.md")
            # Hide compiler-owned metadata from the global AWM rewrite. Runtime
            # rollout still gets the complete footer; reconstruct it afterward.
            workflow.replace(_without_resource_use(before))
            try:
                options = {"resource_context": resource_context, "editable_action_cards": True}
                if edit_mode == "tools":
                    options.update(edit_mode="tools", editable_resources=state["cards"])
                updated = updater.induce(views, predictions, metrics, turn_results=rows, **options)
            finally:
                workflow.replace(before)
            if not updated.strip() or updated.strip() == _AWM_WORKFLOW_RESOURCE_SECTION.strip():
                workflow.replace(before)
                raise RuntimeError("AWM induction failed; retaining previous workflow and cached rollout for resume")
            # Replace the legacy resource-only footer with this adapter's
            # accurate resource contract (hidden scenario labels, prefix memory).
            updated = updated.replace(_AWM_WORKFLOW_RESOURCE_SECTION.strip(), "")
            cards = ({name: updater.last_awm_edited_resources[name] for name in state["cards"]}
                     if edit_mode == "tools" else update_card_resources(state["cards"], updater.last_awm_card_updates))
            workflow.replace(_with_resource_use(updated))
            updater.update_memory(views, predictions, metrics, turn_results=rows)
            state.update(completed_batches=index + 1, workflow=workflow.text, exemplars=copy.deepcopy(memory.exemplars), cards=cards)
            state["usage"] = merge_usage_summaries(previous_usage, get_usage())
            _save(output / "card_updates" / f"batch_{index + 1:04d}.json", {
                "identity": identity, "edit_mode": edit_mode, "updates": updater.last_awm_card_updates,
                "file_edits": updater.last_awm_edit_operations if edit_mode == "tools" else [],
                "cards_before_hash": _hash(before_state["cards"]), "cards_after_hash": _hash(cards)})
            _save(checkpoint_path, state)
            live_resources.update(cards)
        except Exception:
            state = before_state
            workflow.replace(state["workflow"])
            memory.exemplars = copy.deepcopy(state["exemplars"])
            live_resources.update(state["cards"])
            state["usage"] = merge_usage_summaries(previous_usage, get_usage())
            _save(checkpoint_path, state)
            raise
    generation = merge_usage_summaries(_read(offline / "mining_usage.json"), previous_usage, get_usage())
    (output / "skill.md").write_text(workflow.text, encoding="utf-8")
    _save(output / "exemplars.json", memory.exemplars)
    for name in ("reference.md", "action_rules.md", "slot_policies.md"):
        (output / name).write_text(live_resources[name], encoding="utf-8")
    reset_usage()
    agent = make_agent(ids[0])
    testing_path = output / "testing_attempt_usage.json"
    previous_testing = _read(testing_path) if args.resume and testing_path.is_file() else None
    parallel_test = workflow_ids if len(workflow_ids) > 1 and "fork" in mp.get_all_start_methods() else None
    try:
        result = evaluate_agent_on_subflow(agent, test, "offline_awm", subflow, save_dir=output,
                 skip_utterance_eval=args.skip_utterance_eval, eval_workflow_ids=parallel_test)
    except Exception:
        partial = merge_usage_summaries(previous_testing, get_usage())
        if parallel_test or (previous_testing or {}).get("incomplete_parallel_accounting"):
            partial["incomplete_parallel_accounting"] = True
            partial["total"]["usage_available"] = False
            partial["note"] = "Failed test sharding may have unreturned child-process usage; known counts are lower bounds"
        _save(testing_path, partial)
        raise
    workers = result.pop("_evaluation_worker_usage", [])
    testing = merge_usage_summaries(previous_testing, get_usage(), *workers)
    if (previous_testing or {}).get("incomplete_parallel_accounting"):
        testing["incomplete_parallel_accounting"] = True
        testing["total"]["usage_available"] = False
        testing["note"] = previous_testing.get("note")
    _save(testing_path, testing)
    result.update(arm="offline_awm", subflow=subflow, runtime_protocol=abcd_runtime_options(args),
                  scope_hash=_hash(scope), llm_usage=split_usage_summary(generation, testing),
                  training_protocol="sequential AWM updates on standard action-turn batches; sampled-prefix exemplars",
                  training_action_turns=sum(len(batch) for batch in batches), training_batches=len(batches),
                  protocol=PROTOCOL, writable_action_cards=True, cards_hash=_hash(state["cards"]),
                  training_lineage=("mixed-v2-to-v3" if state.get("migration") else
                      "mixed-edit-interface" if any(h["first_batch_index"] > 0 for h in state.get("edit_interface_history", [])) else
                      "v3-tools-from-start" if edit_mode == "tools" else "v3-from-start"),
                  induction_edit_mode=edit_mode, edit_interface_history=state.get("edit_interface_history", []),
                  migration=state.get("migration"),
                  exemplar_count=len(memory))
    _save(output / "result.json", result)
