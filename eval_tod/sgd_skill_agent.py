"""Backbone/residual skill mining and post-rollout local refinement for SGD.

Reuse the ABCD discriminative graph algorithm, not its ordered-slot prompts.
One graph node is an unordered turn-level action-head bundle, with service
versions abstracted to their family. Function names/arguments remain payloads.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

from eval_tod.sgd_adapter import action_counter, gold_policy, score_policies, visible_context
from eval_tod.sgd_llm import sgd_chat_with_retry
from eval_tod.sgd_usage import set_phase

ROOT = "<START>"


def _family(service: str) -> str:
    return service.rsplit("_", 1)[0] if service.rsplit("_", 1)[-1].isdigit() else service


def policy_heads(policy: dict) -> list[list[str]]:
    return [list(item) for item in sorted({
        (kind, _family(service)) for kind, service in action_counter(policy)
    })]


def policy_token(policy: dict) -> str:
    value = json.dumps(policy_heads(policy), ensure_ascii=False, separators=(",", ":"))
    return "policy-" + hashlib.sha256(value.encode()).hexdigest()[:16]


def previous_node(dialogue: dict, turn_index: int, family: str) -> str:
    indices = [index for index in dialogue["target_indices"]["policy_turn_indices"] if index < turn_index]
    return f"{family}:{policy_token(gold_policy(dialogue, indices[-1]))}" if indices else ROOT


def _json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("SGD skill synthesis must return a JSON object")
    return result


class SGDSkillAgent:
    def __init__(self, *, family: str, model: str, ontology: dict, predictor: Callable,
                 candidate_limit: int = 3, reference_max_chars: int = 1800,
                 max_outgoing_edges: int = 3, min_branch_support: int = 2,
                 discriminative_lambda: float = 1.0, discriminative_clip: float = 3.0,
                 mining_method: str = "discriminative", trace_coverage_target: float = .8,
                 corpus_fitness_target: float = .95, routing_complexity_weight: float = 2.,
                 seed: int = 42):
        self.family, self.model = family, model
        self.ontology, self.predictor = ontology, predictor
        self.candidate_limit, self.reference_max_chars = candidate_limit, reference_max_chars
        self.miner_options = dict(max_outgoing_edges=max_outgoing_edges,
                                  min_branch_support=min_branch_support,
                                  discriminative_lambda=discriminative_lambda,
                                  discriminative_clip=discriminative_clip)
        self.seed = seed
        if mining_method not in {"discriminative", "trace_cover"}:
            raise ValueError("Unknown SGD graph mining method")
        self.mining_method = mining_method
        self.trace_cover_options = dict(min_branch_support=min_branch_support,
            trace_coverage_target=trace_coverage_target, corpus_fitness_target=corpus_fitness_target,
            routing_complexity_weight=routing_complexity_weight)
        self.stage = "uninitialized"
        self.graph: dict = {}
        self.resources: dict = {"action_rules": {}, "slot_policies": {}, "transition_guards": {}}
        self.ledger: list[dict] = []

    def mine(self, dialogues: list[dict], directory: Path, *, cases_per_node: int = 3,
             compile_batch_size: int = 8) -> None:
        from skill_mining.backbone_workflow_mining import (
            mine_backbone_workflow_discriminative, mine_backbone_workflow_trace_cover,
        )

        set_phase("offline_compile")
        projected, patterns, cases = [], {}, defaultdict(list)
        for dialogue in dialogues:
            delexed = [{"speaker": "customer" if turn["speaker"] == "user" else "agent",
                        "text": turn.get("utterance", ""), "targets": []}
                       for turn in dialogue["turns"]]
            for index in dialogue["target_indices"]["policy_turn_indices"]:
                gold = gold_policy(dialogue, index)
                token = policy_token(gold)
                patterns[token] = policy_heads(gold)
                # Atomic bundle: no artificial edges between simultaneous acts.
                delexed[index] = {"speaker": "action", "text": "",
                                  "targets": [None, "take_action", token, []]}
                node = f"{self.family}:{token}"
                if len(cases[node]) < cases_per_node:
                    observation = (dialogue["turns"][index].get("db_results") or {}).get(gold["call"]["service"]) if gold["call"] else None
                    cases[node].append({"dialogue_id": dialogue["dialogue_id"], "turn_index": index,
                                        "context": visible_context(dialogue, index)[-1800:], "gold": gold,
                                        "api_observation_after_gold_call": json.dumps(observation, ensure_ascii=False)[:2000]})
            projected.append({"convo_id": dialogue["dialogue_id"], "delexed": delexed})
        if not patterns:
            raise ValueError(f"No training policy turns for {self.family}")
        if self.mining_method == "trace_cover":
            mined = mine_backbone_workflow_trace_cover(self.family, projected, **self.trace_cover_options)
        else:
            mined = mine_backbone_workflow_discriminative(self.family, projected, **self.miner_options)
        self.graph = mined["subgraph"]
        self.stage = "offline"
        self.graph["protocol"] = "sgd_unordered_action_bundle_family_v1"
        # Discard ABCD-only state/positional-slot diagnostics: these must not
        # masquerade as SGD named-argument contracts.
        for node in self.graph["nodes"]:
            node["action_heads"] = patterns[node["label"]]
            for key in ("slot_contract", "slot_examples", "observed_slot_counts"):
                node.pop(key, None)
            key = node["id"]
            self.resources["action_rules"][key] = "Choose this unordered action set only when supported by dialogue context."
            self.resources["slot_policies"][key] = "Use named slots; ground values in prior context or visible API observation."
        edge_lists = [self.graph["edges"], self.graph["backbone"]["edges"], self.graph["residual_edges"]]
        edge_lists.extend(self.graph["local_transitions"].values())
        for edges in edge_lists:
            for edge in edges:
                edge.pop("condition", None)
                key = self.edge_key(edge)
                self.resources["transition_guards"].setdefault(key, "Observed training transition; not a mandatory execution constraint.")
        self.save(directory)
        nodes = self.graph["nodes"]
        groups = [nodes[start:start + compile_batch_size] for start in range(0, len(nodes), compile_batch_size)]
        while groups:
            group = groups.pop(0)
            ids = {node["id"] for node in group}
            evidence = {node: rows for node, rows in cases.items() if node in ids}
            edges = [edge for edge in self.graph["edges"] if edge["target"] in ids]
            payload = {"nodes": [{"id": node["id"], "action_heads": node["action_heads"],
                                   "frequency": node["frequency"]} for node in group],
                       "edges": [{key: edge[key] for key in ("source", "target", "support", "score")}
                                 for edge in edges], "cases": evidence}
            encoded = json.dumps(payload, ensure_ascii=False)
            if len(encoded) > 48000:
                if len(group) > 1:
                    midpoint = len(group) // 2
                    groups[:0] = [group[:midpoint], group[midpoint:]]
                    continue
                raise ValueError("One-node offline evidence exceeds prompt budget; reduce --ours-cases-per-node")
            prompt = (
                "Compile local SGD skill rules from TRAIN traces, without any agent rollout. "
                "Nodes are unordered action-head bundles (intent, service FAMILY), not an ordering "
                "of simultaneous acts. CALL function names and named arguments are payloads. "
                "Actual service versions and legal methods must come from the runtime schema. "
                "Infer observable routing conditions and named-slot grounding rules; "
                "never memorize private example values or rely on future/current system utterances. "
                "Return JSON {\"updates\":[{\"resource\":\"action_rules|slot_policies|transition_guards\","
                "\"key\":\"exact node ID or exact source => target\",\"text\":\"concise general rule\"}]}. "
                "Only use listed node/edge IDs; unsupported distinctions should remain uncertain.\n"
                + encoded
            )
            raw = sgd_chat_with_retry(prompt, model=self.model, temperature=0, call_tag="sgd_ours_offline_compile")
            payload = _json_object(raw)
            allowed = {"action_rules": ids, "slot_policies": ids,
                       "transition_guards": {self.edge_key(edge) for edge in edges}}
            self.apply_updates(payload.get("updates", []), phase="offline_compile", allowed=allowed)
            self.save(directory)
        self.save(directory)

    @staticmethod
    def edge_key(edge: dict) -> str:
        return f"{edge['source']} => {edge['target']}"

    def apply_updates(self, updates: list, *, phase: str, allowed: dict | None = None) -> None:
        if not isinstance(updates, list):
            raise ValueError("Skill updates must be a list")
        for update in updates:
            resource, key, text = (update.get("resource"), update.get("key"), update.get("text")) if isinstance(update, dict) else (None, None, None)
            valid = (resource in self.resources and key in self.resources.get(resource, {})
                     and isinstance(text, str) and bool(text.strip())
                     and (allowed is None or key in allowed.get(resource, set())))
            row = {"phase": phase, "update": update, "applied": bool(valid)}
            if valid:
                row["before"] = self.resources[resource][key]
                self.resources[resource][key] = text.strip()
            else:
                row["reason"] = "unknown_target_or_empty_rule"
            self.ledger.append(row)

    def runtime_resources(self, dialogue: dict, turn_index: int) -> dict:
        source = previous_node(dialogue, turn_index, self.family)
        nodes = {node["id"]: node for node in self.graph["nodes"]}
        if source == ROOT:
            selected = [edge for edge in self.graph["backbone"]["edges"] if edge["source"] == ROOT]
        else:
            selected = self.graph["local_transitions"].get(source, [])
        selected = sorted(selected, key=lambda edge: (edge.get("priority", 0), -edge.get("score", 0)))
        context_tokens = set(re.findall(r"\w+", visible_context(dialogue, turn_index).lower()))
        fallback = sorted(nodes, key=lambda key: (-len(context_tokens & set(re.findall(
            r"\w+", self.resources["action_rules"][key].lower()))), -nodes[key]["frequency"], key))
        candidate_ids = list(dict.fromkeys([edge["target"] for edge in selected] + fallback))[:self.candidate_limit]
        cards = [{"id": key, "heads": nodes[key]["action_heads"],
                  "rule": self.resources["action_rules"][key],
                  "slots": self.resources["slot_policies"][key]} for key in candidate_ids]
        # Residual text is retrieved locally, rather than injecting all cases.
        residual = [edge for edge in self.graph["residual_edges"] if edge["source"] == source]
        references = [{"edge": self.edge_key(edge), "guard": self.resources["transition_guards"][self.edge_key(edge)],
                       "heads": nodes[edge["target"]]["action_heads"]} for edge in residual]
        guards = [{"edge": self.edge_key(edge), "guard": self.resources["transition_guards"][self.edge_key(edge)]}
                  for edge in selected[:self.candidate_limit]]
        skill = (
            "SGD backbone/residual skill: action nodes represent UNORDERED turn action sets. "
            "Family names are templates, not literal service IDs; bind to current service schema. "
            "Candidates and graph order are advisory: do not discard valid unlisted actions. "
            "Use local routing rules and named-slot policies; do not copy training values.\n"
            + json.dumps({"source": source, "candidate_cards": cards, "guards": guards}, ensure_ascii=False)
        )
        return {"skill_text": skill, "reference_text": json.dumps(references, ensure_ascii=False)[:self.reference_max_chars],
                "source": source, "candidate_ids": candidate_ids}

    def predict(self, dialogue: dict, turn_index: int) -> tuple[dict, dict]:
        resources = self.runtime_resources(dialogue, turn_index)
        prediction, diagnostic = self.predictor(
            dialogue, turn_index, model=self.model, ontology=self.ontology, frequency_graph=None,
            skill_text=resources["skill_text"], reference_text=resources["reference_text"],
        )
        diagnostic.update(source_node=resources["source"], candidate_ids=resources["candidate_ids"],
                          resource_sha256=self.resource_hash())
        return prediction, diagnostic

    def refine(self, dialogues: list[dict], directory: Path, *, selection_rate: float = .30,
               batch_size: int = 8, max_batches: int | None = None, per_transition_cap: int = 3,
               analysis_batch_size: int = 8) -> dict:
        # Stratified, deterministic dialogue selection covers sibling pairs
        # first; the remaining quota is random, independent of evaluation.
        rng = random.Random(self.seed)
        transitions = defaultdict(list)
        for position, dialogue in enumerate(dialogues):
            seen = set()
            for index in dialogue["target_indices"]["policy_turn_indices"]:
                seen.add((previous_node(dialogue, index, self.family), policy_token(gold_policy(dialogue, index))))
            for pair in sorted(seen):
                transitions[pair].append(position)
        budget = min(len(dialogues), math.ceil(len(dialogues) * selection_rate))
        chosen = []
        pools = [rng.sample(values, len(values))[:per_transition_cap] for _, values in sorted(transitions.items())]
        rng.shuffle(pools)
        for offset in range(per_transition_cap):
            for pool in pools:
                if offset < len(pool) and pool[offset] not in chosen and len(chosen) < budget:
                    chosen.append(pool[offset])
        remaining = [index for index in range(len(dialogues)) if index not in chosen]
        rng.shuffle(remaining)
        chosen += remaining[:max(0, budget - len(chosen))]
        if max_batches is not None:
            chosen = chosen[:max_batches * batch_size]
        initial_hash = self.resource_hash()
        topology_hash = hashlib.sha256(json.dumps(self.graph, sort_keys=True).encode()).hexdigest()
        rows = []
        set_phase("train_rollout")
        with (directory / "ours_training_turns.jsonl").open("w", encoding="utf-8") as stream:
            for position in chosen:
                dialogue = dialogues[position]
                for index in dialogue["target_indices"]["policy_turn_indices"]:
                    prediction, diagnostic = self.predict(dialogue, index)
                    gold = gold_policy(dialogue, index)
                    scores = score_policies([gold], [prediction])
                    error = ("action" if not scores["action_set_exact"] else
                             "slot" if not scores["turn_joint_ast"] else "success")
                    row = {"dialogue_id": dialogue["dialogue_id"], "turn_index": index,
                           "context": visible_context(dialogue, index)[-1800:], "gold": gold,
                           "prediction": prediction, "error_type": error,
                           "source_node": diagnostic["source_node"],
                           "api_observation_after_predicted_call": json.dumps(diagnostic.get("api_observation"), ensure_ascii=False)[:2000],
                           "target_node": f"{self.family}:{policy_token(gold)}"}
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.resource_hash() != initial_hash:
            raise AssertionError("Rollout must use one frozen resource snapshot")
        set_phase("online_diagnosis")
        reports = []
        # Diagnosis operates on bounded policy-event groups, not unbounded
        # full-dialogue dumps. Rollout still retains every selected event.
        for start in range(0, len(rows), analysis_batch_size):
            evidence = rows[start:start + analysis_batch_size]
            ids = {row["source_node"] for row in evidence} | {row["target_node"] for row in evidence}
            local_resources = {
                "action_rules": {key: value for key, value in self.resources["action_rules"].items() if key in ids},
                "slot_policies": {key: value for key, value in self.resources["slot_policies"].items() if key in ids},
                "transition_guards": {key: value for key, value in self.resources["transition_guards"].items()
                                      if any(key.startswith(source + " => ") for source in ids)},
            }
            prompt = (
                "Diagnose TRAINING SGD rollouts for local skill refinement. Use only observable "
                "pre-turn context and gold feedback provided here. If action-set wrong, localize "
                "routing/action rules, not wrong-action argument bindings. If action-set right "
                "but payload wrong, localize named-slot/function-binding rules. Use successes to "
                "protect correct alternatives. Do not edit graph nodes/edges, memorize values, "
                "or turn simultaneous acts into a sequence. Return JSON {\"updates\":[]}; each "
                "update is {resource: action_rules|slot_policies|transition_guards, key: exact ID, text: rule}.\n"
                + json.dumps({"resources": local_resources, "evidence": evidence}, ensure_ascii=False)
            )
            raw = sgd_chat_with_retry(prompt, model=self.model, temperature=0, call_tag="sgd_ours_diagnosis")
            report = _json_object(raw)
            allowed = {resource: set() for resource in self.resources}
            for row in evidence:
                target = row["target_node"]
                if row["error_type"] in {"action", "success"}:
                    allowed["action_rules"].add(target)
                    allowed["transition_guards"].update(
                        key for key in local_resources["transition_guards"]
                        if key.startswith(row["source_node"] + " => "))
                if row["error_type"] in {"slot", "success"}:
                    allowed["slot_policies"].add(target)
            proposed = report.get("updates", [])
            if not isinstance(proposed, list):
                raise ValueError("Diagnosis updates must be a list")
            accepted = []
            for update in proposed:
                resource, key = (update.get("resource"), update.get("key")) if isinstance(update, dict) else (None, None)
                if key in allowed.get(resource, set()) and key in self.resources.get(resource, {}):
                    accepted.append(update)
                else:
                    self.ledger.append({"phase": "online_diagnosis", "update": update, "applied": False,
                                        "reason": "target_outside_action_slot_error_localization"})
            report["updates"] = accepted
            reports.append(report)
            (directory / f"diagnosis_{len(reports):04d}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if reports:
            set_phase("online_reflection")
            # One final proposal per stable target. Conflicting proposals are
            # consolidated before any write, eliminating stale text anchors.
            grouped = defaultdict(list)
            for report in reports:
                for update in report.get("updates", []):
                    if isinstance(update, dict):
                        grouped[(str(update.get("resource")), str(update.get("key")))].append(update)
            final_updates = []
            targets = sorted(grouped)
            for start in range(0, len(targets), 16):
                group = targets[start:start + 16]
                prompt = (
                    "Reflect and consolidate TRAIN-derived SGD local rule proposals. Return JSON "
                    "{\"updates\":[]}, at most ONE complete rule per listed (resource,key). "
                    "Preserve supported existing behavior and successes, resolve contradictions, "
                    "and make guards observable. Keep graph topology fixed; no memorized values. "
                    "Omit unsupported proposals. Only listed stable keys can be changed.\n"
                    + json.dumps({"targets": [{"resource": resource, "key": key,
                                    "existing": self.resources.get(resource, {}).get(key),
                                    "proposals": grouped[(resource, key)]}
                                   for resource, key in group]}, ensure_ascii=False)
                )
                raw = sgd_chat_with_retry(prompt, model=self.model, temperature=0, call_tag="sgd_ours_reflection")
                updates = _json_object(raw).get("updates", [])
                if not isinstance(updates, list):
                    raise ValueError("Reflection updates must be a list")
                seen = set()
                for update in updates:
                    target = (update.get("resource"), update.get("key")) if isinstance(update, dict) else (None, None)
                    if target not in group or target in seen:
                        raise ValueError("Reflection returned unlisted or duplicate target")
                    seen.add(target)
                    final_updates.append(update)
                (directory / f"reflection_{start // 16 + 1:04d}.json").write_text(
                    json.dumps({"targets": group, "updates": updates}, ensure_ascii=False, indent=2), encoding="utf-8")
            self.apply_updates(final_updates, phase="online_reflection")
        assert topology_hash == hashlib.sha256(json.dumps(self.graph, sort_keys=True).encode()).hexdigest()
        self.stage = "refined"
        self.save(directory)
        summary = {"selected_dialogue_ids": [dialogues[index]["dialogue_id"] for index in chosen],
                   "rollout_turns": len(rows), "reports": len(reports),
                   "snapshot_sha256": initial_hash, "final_sha256": self.resource_hash(),
                   "protocol": "frozen_train_rollout_then_local_diagnosis_then_single_writer_reflection"}
        (directory / "ours_training.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def resource_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.resources, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"protocol": "sgd_backbone_online_v1", "family": self.family, "stage": self.stage,
                   "graph": self.graph, "resources": self.resources, "ledger": self.ledger}
        path = directory / "sgd_skill_state.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        (directory / "subgraph.json").write_text(json.dumps(self.graph, ensure_ascii=False, indent=2), encoding="utf-8")
        for resource, filename in (("action_rules", "action_rules.md"), ("slot_policies", "slot_policies.md"),
                                   ("transition_guards", "reference.md")):
            text = "\n\n".join(f"## {key}\n{value}" for key, value in self.resources[resource].items())
            (directory / filename).write_text(text + "\n", encoding="utf-8")
        order = self.graph.get("backbone", {}).get("compilation_order", [])
        nodes = {node["id"]: node for node in self.graph.get("nodes", [])}
        skill = "# SGD backbone skill\nGraph order is organization, not an execution constraint.\n"
        for key in order:
            skill += f"\n## {key}\nHeads: {json.dumps(nodes[key]['action_heads'])}\n{self.resources['action_rules'][key]}\n"
        (directory / "skill.md").write_text(skill, encoding="utf-8")
        (directory / "refinement_ledger.json").write_text(json.dumps(self.ledger, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, directory: Path) -> None:
        payload = json.loads((directory / "sgd_skill_state.json").read_text(encoding="utf-8"))
        if payload.get("protocol") != "sgd_backbone_online_v1" or payload.get("family") != self.family:
            raise ValueError("SGD skill protocol/domain mismatch")
        self.graph, self.resources = payload["graph"], payload["resources"]
        self.stage = payload.get("stage", "unknown")
        self.ledger = copy.deepcopy(payload.get("ledger", []))
