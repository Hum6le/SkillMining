#!/usr/bin/env python3
"""Resumable corrected evaluation of frozen MultiWOZ AWM/T2S/Ours resources."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_tod.multiwoz_matched_executor import (
    PROTOCOL, CONTRACT, bind_payload, digest, plan_observation, request_json,
    select_heads, write_json,
)
from eval_tod.multiwoz_policy import MultiWOZStandardSkillAgent, load_hotel_pilot, ontology_from_train
from eval_tod.sgd_adapter import action_counter, gold_policy, score_policies, visible_context
from eval_tod.sgd_agent import SGDAWMAgent
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_skill_agent import policy_token
from eval_tod.sgd_usage import configure_usage, persist_usage
from eval_tod.reference_lookup import load_trace2skill_references
from scripts.llm_usage_utils import merge_usage_summaries

METHODS = ("awm", "trace2skill", "ours")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cached(path, signature):
    path = Path(path)
    if not path.exists():
        return None
    value = read_json(path)
    if value.get("signature") != signature or value.get("protocol") != PROTOCOL:
        raise ValueError(f"Cache input/protocol mismatch: {path}")
    return value


def compile_card_job(job):
    path, signature, profiles, model = job
    if cached(path, signature):
        return str(path)
    ids = {profile["node_id"] for profile in profiles}

    def validate(payload):
        cards = payload.get("cards")
        if not isinstance(cards, list) or len(cards) != len(ids):
            raise ValueError("One complete card per exact requested node_id is required")
        if {card.get("node_id") for card in cards if isinstance(card, dict)} != ids:
            raise ValueError("Missing, duplicated or invented node_id; copy IDs exactly")
        for card in cards:
            for field in ("action_rule", "slot_policy"):
                if not isinstance(card.get(field), str) or len(card[field].strip()) < 20:
                    raise ValueError("Both a specific action_rule and slot_policy are mandatory for EVERY node")
                if card[field].strip() in {
                    "Use named slots; ground values in prior context or visible API observation.",
                    "Choose this unordered action set only when supported by dialogue context."}:
                    raise ValueError("Unchanged generic placeholder is not a completed card")
                if card[field].lstrip().startswith("MISSING"):
                    raise ValueError("Missing-field instruction was echoed instead of a completed rule")
                if "<!--" in card[field]:
                    raise ValueError("Cards contain prose, no compiler markers")
        return cards

    prompt_profiles = copy.deepcopy(profiles)
    for profile in prompt_profiles:
        for field in ("current_action_rule", "current_slot_policy"):
            if not profile.get(field) or profile[field].startswith(("Use named slots; ground values", "Choose this unordered action set only")):
                profile[field] = "MISSING. Write a concrete rule for this node's exact heads and observed slot variants from the TRAIN cases."
    prompt = ("Compile complete Action Cards from labeled TRAIN evidence only. "
        "TRAIN gold acts below are permitted for compilation; no evaluation labels are provided. "
        "At runtime, agents see prior dialogue and read-only public DB results, never future/gold acts. "
        "One node is a complete UNORDERED head bundle, not an individual act. "
        "Return JSON {\"cards\":[{\"node_id\":\"COPY_EXACT_ID\",\"action_rule\":\"when this bundle applies\","
        "\"slot_policy\":\"context-dependent slot choices and value sources\"}]}. "
        "Return EVERY requested node. Never use primitive intent/domain/slot expressions as node_id. "
        "Keep each field under 180 words. A MISSING field must be newly written, never echoed as a placeholder. "
        "Do not return generic 'choose this action when supported' or 'use named slots' rules. "
        "Name the actual intents/domains and relevant slots of each node. "
        "Do not memorize instance entity values or training IDs. "
        "Explain which requested slots to emit, when to include entity name, when no slot is needed, "
        "and whether values come from dialogue, DB, or an unavailable booking result. "
        "Slot inventories are supported alternatives, not mandatory fields to emit together. "
        "Different slots can share the same head. Omit values for request/global acts. "
        "Available read-only DB queries now supply entity facts. Preserve supported existing guidance "
        "but correct old assumptions that no DB observations exist. No booking tool exists.\n"
        "TRAIN node profiles:\n" + json.dumps(prompt_profiles, ensure_ascii=False))
    cards, diagnostic = request_json(prompt, model, "matched_train_card_completion", validate)
    write_json(path, {"protocol": PROTOCOL, "signature": signature, "cards": cards, "diagnostic": diagnostic})
    return str(path)


def observation_job(job):
    path, signature, dialogue, turn, db_dir, model = job
    if cached(path, signature):
        return str(path)
    observation = plan_observation(dialogue, turn, kb_dir=db_dir, model=model)
    write_json(path, {"protocol": PROTOCOL, "signature": signature, "observation": observation})
    return str(path)


@lru_cache(maxsize=4)
def load_bundle(path):
    return read_json(path)


@lru_cache(maxsize=2)
def awm_resources(directory, model):
    agent = SGDAWMAgent(family="hotel", model=model, ontology={}, predictor=None)
    agent.load(Path(directory))
    return agent


def method_knowledge(method, dialogue, turn, observation, bundle):
    context = visible_context(dialogue, turn)[-12000:]
    if method == "awm":
        agent = awm_resources(bundle["awm_directory"], bundle["model"])
        text = ("AWM workflow:\n" + agent.workflow.format_prompt(query_text=context, max_chars=8000)
                + "\nVerified TRAIN exemplars:\n" + agent.memory.format_prompt(["hotel"], max_chars=3000))
        return text, {"kind": "workflow_and_verified_exemplars"}
    if method == "trace2skill":
        return bundle["trace2skill_skill"] + "\nReferences:\n" + bundle["trace2skill_reference"], {"kind": "evolved_skill"}
    from skill_mining.online_refinement import _resource_lookup_sections
    query = observation["plan"]["query_text"] or context[-1800:]
    tokens = set(query.casefold().split())
    cards = sorted(bundle["ours_cards"].values(), key=lambda card: (
        -len(tokens & set(card["action_rule"].casefold().split())), -card["frequency"], card["node_id"]))[:3]
    reference = _resource_lookup_sections("reference", bundle["ours_reference"], query, 2)
    text = (bundle["ours_skill"] + "\nCandidate Action Cards:\n" + json.dumps(cards, ensure_ascii=False)
            + "\nRetrieved reference:\n" + json.dumps(reference, ensure_ascii=False)[:1800])
    return text, {"kind": "backbone_cards_and_reference", "candidate_ids": [c["node_id"] for c in cards]}


def evaluation_job(job):
    path, signature, method, dialogue, turn, observation, bundle_path, ontology = job
    if cached(path, signature):
        return str(path)
    bundle = load_bundle(str(bundle_path))
    selection_knowledge, resource_diagnostic = method_knowledge(method, dialogue, turn, observation, bundle)
    heads, action_diagnostic = select_heads(dialogue, turn, observation=observation,
        knowledge=selection_knowledge, model=bundle["model"])
    payload_knowledge = selection_knowledge
    selected_card_ids = []
    if method == "ours":
        chosen = {tuple(head) for head in heads}
        cards = [card for card in bundle["ours_cards"].values()
                 if {tuple(head) for head in card["heads"]} == chosen]
        selected_card_ids = [card["node_id"] for card in cards]
        payload_knowledge = "Selected complete Action Card:\n" + json.dumps(cards, ensure_ascii=False)
        if not cards:
            payload_knowledge += "\nUnseen action bundle: use the common schema and visible facts; no unrelated card substituted."
    prediction, payload_diagnostic = bind_payload(dialogue, turn, heads=heads, observation=observation,
        knowledge=payload_knowledge, ontology=ontology, model=bundle["model"])
    # Gold is read only after inference, for the saved scorer input.
    record = {"protocol": PROTOCOL, "signature": signature, "method": method,
        "dialogue_id": dialogue["dialogue_id"], "turn_index": turn,
        "gold": gold_policy(dialogue, turn), "prediction": prediction,
        "diagnostic": {"selected_action_heads": heads, "selected_card_ids": selected_card_ids,
            "resources": resource_diagnostic, "action_selection": action_diagnostic,
            "payload": payload_diagnostic, "shared_observation_sha256": digest(observation),
            "knowledge_sha256": bundle["knowledge_sha256"]}}
    write_json(path, record)
    return str(path)


def recover_logged_usage(root):
    """Rebuild known usage once from append-only request logs after a restart."""
    by_phase = {}
    for path in root.rglob("llm_requests.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase, tag = row["phase"], row["call_tag"]
            by_phase[phase] = merge_usage_summaries(by_phase.get(phase, {"total": {}}),
                {"total": row["total"], "by_call_tag": {tag: row["total"]}})
    testing = merge_usage_summaries({"total": {}}, by_phase.get("testing", {}))
    generation = merge_usage_summaries({"total": {}}, *[
        value for phase, value in by_phase.items() if phase != "testing"])
    interrupted = (root / "supervisor_interruptions.jsonl").exists()
    write_json(root / "llm_usage.json", {"schema_version": 3, "by_phase": by_phase,
        "testing": testing, "generation": generation, "total": merge_usage_summaries(testing, generation),
        "incomplete_phases": ["interrupted_requests"] if interrupted else [],
        "accounting_complete": not interrupted})


def run_jobs(root, name, jobs, function, workers):
    pending = [job for job in jobs if cached(job[0], job[1]) is None]
    total = len(jobs)
    for retry_round in range(1, 4):
        write_json(root / "progress.json", {"time": datetime.now().isoformat(), "phase": name,
            "total": total, "completed": total - len(pending), "pending": len(pending), "round": retry_round})
        if not pending:
            return
        print(f"{name}: {total-len(pending)}/{total} cached; {len(pending)} pending; workers={workers}", flush=True)
        outcomes = ordered_cases(function, pending, workers=workers, directory=root,
                                 phase="testing" if name == "matched_evaluation" else name)
        for outcome, job in zip(outcomes, pending):
            try:
                require_value(outcome)
            except Exception as exc:
                with (root / "retry_history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"phase": name, "path": str(job[0]), "round": retry_round,
                        "error": str(exc)[:5000], "time": datetime.now().isoformat()}) + "\n")
            completed = sum(Path(item[0]).exists() for item in jobs)
            write_json(root / "progress.json", {"time": datetime.now().isoformat(), "phase": name,
                "total": total, "completed": completed, "pending": total-completed, "round": retry_round})
        pending = [job for job in pending if cached(job[0], job[1]) is None]
        if not pending:
            return
    if name == "matched_evaluation":
        # A model's exhausted output/schema failure is a failed evaluation
        # turn, not a reason to discard the complete comparison for all methods.
        for job in pending:
            path, signature, method, dialogue, turn, observation, bundle_path, _ = job
            bundle = load_bundle(str(bundle_path))
            write_json(path, {"protocol": PROTOCOL, "signature": signature, "method": method,
                "dialogue_id": dialogue["dialogue_id"], "turn_index": turn,
                "gold": gold_policy(dialogue, turn),
                "prediction": {"call": None, "acts": [], "_execution_error": True},
                "diagnostic": {"status": "error", "error": "Output validation exhausted bounded retries; see retry_history.jsonl",
                    "selected_action_heads": [], "shared_observation_sha256": digest(observation),
                    "knowledge_sha256": bundle["knowledge_sha256"]}})
        print(f"{name}: {len(pending)} terminal failures retained in the full evaluation denominator", flush=True)
        return
    raise RuntimeError(f"{name}: {len(pending)} jobs still failed; successes cached for supervisor recovery")


def build_profiles(train, agent):
    node_lookup = {node["label"]: node for node in agent.graph["nodes"]}
    evidence = defaultdict(list)
    slots = defaultdict(Counter)
    for dialogue in train:
        for turn in dialogue["target_indices"]["policy_turn_indices"]:
            gold = gold_policy(dialogue, turn)
            label = policy_token(gold)
            slots[label].update((act["intent"], act["domain"], act["slot"], "value" in act) for act in gold["acts"])
            if len(evidence[label]) < 4:
                evidence[label].append({"context": visible_context(dialogue, turn)[-2200:], "gold": gold})
    profiles = []
    for label, node in node_lookup.items():
        if not evidence[label]:
            raise ValueError(f"No TRAIN evidence for node {label}")
        current = agent.action_card(node)
        profiles.append({"node_id": node["id"], "heads": node["action_heads"], "frequency": node["frequency"],
            "observed_slot_variants": [{"intent": k[0], "domain": k[1], "slot": k[2],
                "value_present": k[3], "count": n} for k,n in slots[label].most_common()],
            "current_action_rule": current["rule"], "current_slot_policy": current["payload_policy"],
            "train_cases": evidence[label]})
    return profiles


def final_reports(root, manifest, evaluation_jobs, full):
    per_method = {}
    for method in METHODS:
        jobs = [job for job in evaluation_jobs if job[2] == method]
        records = [read_json(job[0]) for job in jobs]
        directory = root / method
        directory.mkdir(exist_ok=True)
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        gold = [row["gold"] for row in records]
        predicted = [row["prediction"] for row in records]
        scores = score_policies(gold, predicted)
        def semantic(policy):
            policy = copy.deepcopy(policy)
            for act in policy["acts"]:
                if act.get("value") == "":
                    act.pop("value")
                elif "value" in act:
                    act["value"] = " ".join(act["value"].casefold().split())
            return policy
        semantic_scores = score_policies([semantic(p) for p in gold], [semantic(p) for p in predicted])
        result = {**manifest, "method": method, **scores, "semantic_value_audit": semantic_scores,
                  "execution_status": ("completed_with_errors" if scores["counts"]["execution_failed_turns"] else "completed")
                      if full else "smoke_completed"}
        write_json(directory / ("summary.json" if full else "smoke_summary.json"), result)
        per_method[method] = result
    filename = "final_comparison" if full else "smoke_comparison"
    usage = persist_usage()
    write_json(root / (filename + ".json"), {"protocol": PROTOCOL, "methods": per_method, "llm_usage": usage})
    lines = ["# Corrected MultiWOZ hotel 10% evaluation", "",
        "Frozen trained AWM/T2S/Ours knowledge; Ours cards completed using TRAIN only. This is an evaluation protocol revision, not complete retraining under the revised executor.", "",
        "All methods share context-only DB planning/observations, action-head selection, fixed-head slot generation and output validation. No gold DB pointers or belief states. Public entity DB cannot create booking references.", "",
        "| Metric | AWM | Trace2Skill | Ours |", "|---|---:|---:|---:|"]
    for label,key in [("Action exact","action_set_exact"),("Action micro F1","action_micro_f1"),
                      ("Slot exact conditional on actions","slot_exact_given_action"),("Joint exact","turn_joint_ast")]:
        lines.append("| " + label + " | " + " | ".join(f"{per_method[m][key]*100:.2f}%" for m in METHODS) + " |")
    lines.append("| Joint with case/whitespace value normalization (audit) | " + " | ".join(
        f"{per_method[m]['semantic_value_audit']['turn_joint_ast']*100:.2f}%" for m in METHODS) + " |")
    lines += ["", "Every requested case stays in the denominator. Exhausted output failures are explicit execution-error predictions; retries are retained in retry_history.jsonl. Primary scoring is unchanged; the semantic-value audit is separate.", "",
        "Shared DB planning and TRAIN card-completion costs are tracked separately from testing in llm_usage.json. Prior training expenditure is excluded from this new run's usage. Interrupted provider requests can have unknown usage."]
    (root / (filename + ".md")).write_text("\n".join(lines)+"\n", encoding="utf-8")
    if full:
        (root / "RESULTS_READY.md").write_text("Completed. See [comparison](final_comparison.md) and final_comparison.json.\n", encoding="utf-8")
    return per_method


def run(args):
    root, source_root = args.output_root.resolve(), args.source_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    recover_logged_usage(root)
    configure_usage(root)
    original = read_json(source_root / "ours_retry_11_12/data_manifest.json")
    source = Path(original["source"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != original["source_sha256"]:
        raise ValueError("Source dataset changed")
    data = load_hotel_pilot(source, seed=original["seed"], fraction=original["fraction"])
    for split in ("train", "validation"):
        if [d["dialogue_id"] for d in data[split]] != original["selected_ids"][split]:
            raise ValueError("Data selection changed")
    ontology = ontology_from_train(data["train"])
    agent = MultiWOZStandardSkillAgent(family="hotel", model=original["model"], ontology=ontology, predictor=None)
    agent.load(source_root / "ours_retry_11_12")
    full_cases = [(d,i) for d in data["validation"] for i in d["target_indices"]["policy_turn_indices"]]
    cases = full_cases[:args.max_eval_cases] if args.max_eval_cases else full_cases
    manifest = {"protocol": PROTOCOL, "source_root": str(source_root), "source_sha256": original["source_sha256"],
        "seed": original["seed"], "fraction": original["fraction"], "model": original["model"],
        "train_dialogues": len(data["train"]), "train_turns": 1818, "validation_turns": len(full_cases),
        "workers_total": args.workers, "training_mode": "frozen_source_resources_plus_train_only_card_completion",
        "selected_ids": original["selected_ids"], "ours_source_checkpoint": agent.resource_hash()}
    signature = digest(manifest)
    old_path = root / "manifest.json"
    if old_path.exists() and read_json(old_path) != manifest:
        raise ValueError("Run manifest changed; use a new output root")
    write_json(old_path, manifest)
    profiles = build_profiles(data["train"], agent)
    card_jobs = []
    for start in range(0, len(profiles), 4):
        group = profiles[start:start+4]
        card_jobs.append((root / "card_completion" / f"batch_{start//4:04d}.json",
                          digest([signature, group]), group, original["model"]))
    run_jobs(root, "train_card_completion", card_jobs, compile_card_job, args.workers)
    cards = {}
    nodes = {node["id"]: node for node in agent.graph["nodes"]}
    profiles_by_id = {profile["node_id"]: profile for profile in profiles}
    for job in card_jobs:
        for card in read_json(job[0])["cards"]:
            cards[card["node_id"]] = {**card, "heads": nodes[card["node_id"]]["action_heads"],
                "frequency": nodes[card["node_id"]]["frequency"],
                "supported_train_slot_variants": profiles_by_id[card["node_id"]]["observed_slot_variants"]}
    if set(cards) != set(nodes):
        raise ValueError("Incomplete corrected action-card coverage")
    write_json(root / "corrected_action_cards.json", {"coverage": len(cards), "total_nodes": len(nodes), "cards": cards})
    # Freeze AWM files locally before workers load them.
    awm_dir = root / "frozen_awm"
    awm_dir.mkdir(exist_ok=True)
    for name in ("awm_workflow.txt", "awm_exemplars.json"):
        value = (source_root / "awm" / name).read_bytes()
        path = awm_dir / name
        if path.exists() and path.read_bytes() != value:
            raise ValueError("AWM source resources changed")
        path.write_bytes(value)
    bundle = {"model": original["model"], "awm_directory": str(awm_dir), "ours_cards": cards,
        "ours_skill": agent.runtime_skill_text().replace("Use only prior dialogue utterances at prediction time;",
            "Use prior dialogue utterances and the provided read-only DB observation;"),
        "ours_reference": (source_root / "ours_retry_11_12/reference.md").read_text(encoding="utf-8"),
        "trace2skill_skill": (source_root / "trace2skill/evolved_skill/SKILL.md").read_text(encoding="utf-8"),
        "trace2skill_reference": load_trace2skill_references(source_root / "trace2skill/evolved_skill")}
    bundle["knowledge_sha256"] = digest(bundle)
    bundle_path = root / "frozen_knowledge.json"
    if bundle_path.exists() and read_json(bundle_path) != bundle:
        raise ValueError("Frozen knowledge changed")
    write_json(bundle_path, bundle)
    db_dir = source.parent
    db_hash = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(db_dir.glob("*_db.json"))})
    observation_jobs = []
    for index,(dialogue,turn) in enumerate(cases):
        observation_jobs.append((root / "observations" / f"case_{index:04d}.json",
            digest([signature, db_hash, dialogue["dialogue_id"], turn, visible_context(dialogue,turn)]),
            dialogue, turn, str(db_dir), original["model"]))
    run_jobs(root, "shared_db_planning", observation_jobs, observation_job, args.workers)
    evaluation_jobs = []
    for index,((dialogue,turn),observation_case) in enumerate(zip(cases,observation_jobs)):
        observation = read_json(observation_case[0])["observation"]
        for method in METHODS:
            evaluation_jobs.append((root / method / "cases" / f"case_{index:04d}.json",
                digest([signature,bundle["knowledge_sha256"],method,observation_case[1]]),
                method,dialogue,turn,observation,bundle_path,ontology))
    run_jobs(root, "matched_evaluation", evaluation_jobs, evaluation_job, args.workers)
    final_reports(root, manifest, evaluation_jobs, full=not args.max_eval_cases)
    write_json(root / "progress.json", {"time": datetime.now().isoformat(),
        "phase": "completed" if not args.max_eval_cases else "smoke_completed", "total": len(evaluation_jobs),
        "completed": len(evaluation_jobs), "pending": 0})
    print("MATCHED EVALUATION COMPLETE" if not args.max_eval_cases else "SMOKE COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-eval-cases", type=int, default=0)
    run(parser.parse_args())
