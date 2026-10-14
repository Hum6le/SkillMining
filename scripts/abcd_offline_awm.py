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
from pathlib import Path

from scripts.abcd_runtime_utils import abcd_runtime_options, build_abcd_runtime_agent

PROTOCOL = "offline-awm-online-v2-target-complete-feedback"
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
- AWM workflow updates may refine the prior using training feedback. Preserve the
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


def _with_resource_use(text):
    # Compiler-owned instructions are restored deterministically after global
    # AWM rewriting, without touching the model's learned workflow patterns.
    if RESOURCE_START in text:
        if text.count(RESOURCE_START) != 1 or text.count(RESOURCE_END) != 1:
            raise ValueError("Malformed offline-AWM resource-use markers")
        start, end = text.index(RESOURCE_START), text.index(RESOURCE_END)
        if end < start:
            raise ValueError("Offline-AWM resource-use markers are reversed")
        text = text[:start] + text[end + len(RESOURCE_END):]
    elif RESOURCE_END in text:
        raise ValueError("Malformed offline-AWM resource-use markers")
    return text.rstrip() + "\n\n" + RESOURCE_USE


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


def run_offline_awm(args, subflow, flow_dir, workflow_ids):
    from awm import MemoryStore, WorkflowStore
    from eval_tod.abcd.agent import compute_ast_from_turn_results, _AWM_WORKFLOW_RESOURCE_SECTION
    from eval_tod.schemas import Prediction
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
            raise ValueError("Offline-AWM checkpoint resources/data/runtime/schedule differ")
    else:
        if checkpoint_path.exists():
            raise ValueError("Offline-AWM checkpoint exists; use --resume")
        state = {"scope": scope, "completed_batches": 0,
                 "workflow": _with_resource_use(resources["skill.md"]), "exemplars": [], "usage": None}
    if args.resume and (output / "result.json").is_file():
        if _read(output / "result.json").get("scope_hash") != _hash(scope):
            raise ValueError("Completed offline-AWM result has incompatible scope")
        return
    if not 0 <= int(state["completed_batches"]) <= len(batches):
        raise ValueError("Offline-AWM completed batch count is outside the saved schedule")
    output.mkdir(parents=True, exist_ok=True)
    _save(checkpoint_path, state)
    workflow, memory = WorkflowStore(), MemoryStore()
    workflow.replace(state["workflow"])
    memory.exemplars = copy.deepcopy(state["exemplars"])
    ids = workflow_ids or [None]

    def make_agent(workflow_id, snapshot=False):
        current_workflow, current_memory = workflow, memory
        if snapshot:
            current_workflow, current_memory = WorkflowStore(), MemoryStore()
            current_workflow.replace(workflow.text)
            current_memory.exemplars = copy.deepcopy(memory.exemplars)
        return build_abcd_runtime_agent(args, workflow=current_workflow, memory=current_memory,
                 reference_text=resources["reference.md"], action_rules_text=resources["action_rules.md"],
                 slot_policies_text=resources["slot_policies.md"], workflow_id=workflow_id)

    previous_usage = state.get("usage")
    reset_usage()
    for index in range(state["completed_batches"], len(batches)):
        batch = batches[index]
        print(f"offline_awm {subflow}: batch {index + 1}/{len(batches)} action-turns={len(batch)}", flush=True)
        before = workflow.text
        batch_path = output / "batches" / f"batch_{index + 1:04d}.json"
        try:
            identity = {"scope_hash": _hash(scope), "workflow_hash": _hash(before), "index": index}
            cached = _read(batch_path) if args.resume and batch_path.is_file() else None
            if cached and cached.get("identity") == identity:
                rows = cached["rows"]
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
            resource_context = "\n\n".join(f"## {name}\n{text}" for name, text in resources.items() if name != "skill.md")
            updated = updater.induce(views, predictions, metrics, turn_results=rows,
                                      resource_context=resource_context)
            if not updated.strip() or updated.strip() == _AWM_WORKFLOW_RESOURCE_SECTION.strip():
                workflow.replace(before)
                raise RuntimeError("AWM induction failed; retaining previous workflow and cached rollout for resume")
            # Replace the legacy resource-only footer with this adapter's
            # accurate resource contract (hidden scenario labels, prefix memory).
            updated = updated.replace(_AWM_WORKFLOW_RESOURCE_SECTION.strip(), "")
            workflow.replace(_with_resource_use(updated))
            updater.update_memory(views, predictions, metrics, turn_results=rows)
            state.update(completed_batches=index + 1, workflow=workflow.text, exemplars=copy.deepcopy(memory.exemplars))
            state["usage"] = merge_usage_summaries(previous_usage, get_usage())
            _save(checkpoint_path, state)
        except Exception:
            state["usage"] = merge_usage_summaries(previous_usage, get_usage())
            _save(checkpoint_path, state)
            raise
    generation = merge_usage_summaries(_read(offline / "mining_usage.json"), previous_usage, get_usage())
    (output / "skill.md").write_text(workflow.text, encoding="utf-8")
    _save(output / "exemplars.json", memory.exemplars)
    for name in ("reference.md", "action_rules.md", "slot_policies.md"):
        (output / name).write_text(resources[name], encoding="utf-8")
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
                  exemplar_count=len(memory))
    _save(output / "result.json", result)
