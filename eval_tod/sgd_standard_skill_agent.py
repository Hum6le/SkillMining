"""SGD task adapter for the ABCD standard post-hoc refinement protocol.

Shared clustering, reflection, DAG updates, text editor and renderers are
imported, not reimplemented. Only targets, slot diagnostics and prompts differ.
The original simplified port remains available as the explicit prototype.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

from eval_tod.sgd_adapter import gold_policy, score_policies, slot_signature, visible_context
from eval_tod.sgd_llm import sgd_chat_with_retry
from eval_tod.sgd_skill_agent import SGDSkillAgent, policy_token, previous_node, _json_object
from eval_tod.sgd_usage import set_phase
from eval_tod.sgd_errors import record_error, error_summary
from eval_tod.sgd_parallel import ordered_cases, require_value, diagnostics_directory
from skill_mining.online_refinement import (
    initialize_skill_dag, build_post_rollout_batches, group_batch_reports,
    diagnose_rollout_batch, reflect_batch_report_group,
    apply_reflection_updates_to_state, apply_dynamic_skill_operations,
    merge_online_skill_additions, render_online_resources,
    render_online_action_rules, render_online_slot_policies, save_skill_dag,
)

log = logging.getLogger(__name__)


def normalize_action_heads(selection, services):
    """Normalize representation only; never guess an action or service version."""
    heads = selection.get("action_heads")
    if not isinstance(heads, list):
        raise ValueError("action_heads must be a list")
    intents = {"call", "request", "confirm", "inform", "inform_count", "offer", "offer_intent",
               "notify_success", "notify_failure", "req_more", "goodbye"}
    normalized = set()
    for head in heads:
        if isinstance(head, dict):
            if "intent" not in head or not ("domain" in head or "service" in head):
                raise ValueError("head object needs intent and domain/service")
            head = [head["intent"], head.get("domain", head.get("service"))]
        if not isinstance(head, list) or len(head) != 2 or not isinstance(head[0], str):
            raise ValueError("each head must be [intent, service] or an intent/domain object")
        intent = head[0].strip().lower()
        if intent not in intents:
            raise ValueError(f"unknown intent: {head[0]!r}")
        service = "" if head[1] is None else head[1]
        if not isinstance(service, str):
            raise ValueError("service must be a string or null for a global act")
        service = service.strip()
        if service:
            matches = [item for item in services if item.casefold() == service.casefold()]
            if not matches:
                from eval_tod.sgd_skill_agent import _family
                matches = [item for item in services if _family(item).casefold() == service.casefold()]
            if len(matches) != 1:
                raise ValueError(f"unknown or ambiguous service: {service!r}; allowed={services!r}")
            service = matches[0]
        if intent == "call" and not service:
            raise ValueError("CALL needs an actual service ID")
        normalized.add((intent, service))
    if sum(intent == "call" for intent, _ in normalized) > 1:
        raise ValueError("multiple distinct CALL services are not supported")
    return [list(head) for head in sorted(normalized)]


_SCHEMA = """
Return JSON only, using native resource names and exact graph labels/edge IDs.
Resource updates: {{"resource":"transition_guard|action_node|action_rule|slot_policy|reference",
"edge_id":"optional exact edge ID","action":"optional exact action label",
"op":"upsert|delete","content":"reusable rule","status":"resolved|uncertain","rationale":"..."}}.
Skill operations: {{"operation_id":"stable unique ID","op":"replace|insert_before|insert_after|delete",
"match_text":"exact unique excerpt of CURRENT skill","new_text":"replacement; empty for delete","rationale":"..."}}.
"""
_TASK = """
SGD actions are UNORDERED turn-level sets of (dialogue-act type, service family),
including CALL. Never invent intra-turn ordering. Concrete API method, named
parameters, act slot names and values are payloads. Service families are
templates; bind actual service IDs to the available schema. No positional-slot
rules apply. Deterministic action_correct and slot_exact_correct are authoritative.
For action failures diagnose routing/action selection, not wrong-action payloads;
for action-correct payload failures diagnose function selection, named-slot
grounding, omissions or extras. Contrast successes and failures and preserve
supported alternatives. Use pre-turn context, not future system text. Do not
memorize instance values or invent unobserved nodes/edges.
"""
DIAGNOSIS_PROMPT = """Diagnose a compact TRAIN rollout batch from a frozen skill.
""" + _TASK + """
Explain causal deficiencies in skill text, graph routing, resource query/retrieval,
ignored resources, action selection or payload grounding. Changes are proposals;
do not write yet. Include reusable successes and unresolved evidence.
<current_skill>{skill}</current_skill>
<local_graph>{local_graph}</local_graph>
<available_local_resources>{available_resources}</available_local_resources>
<batch_rollouts>{rollouts}</batch_rollouts>
""" + _SCHEMA + """
Top-level object: {{"summary":"detailed analysis","root_causes":[],
"graph_footprint":{{"nodes":[],"edges":[]}},"candidate_updates":[],
"candidate_skill_operations":[],"unresolved_questions":[]}}.
"""
REFLECTION_PROMPT = """Reflect on at most 16 neighboring TRAIN diagnosis reports.
""" + _TASK + """
Resolve conflicting proposals; preserve supported successful behavior. The
CURRENT skill/state includes earlier applied reflection groups. Rebase edits
against this current text, never against stale report excerpts. Return one
coherent local change; uncertain knowledge belongs in reference, not an
executable guard. Diagnose why resources were insufficient or ignored.
<current_skill>{skill}</current_skill>
<local_graph>{local_graph}</local_graph>
<batch_reports>{batch_reports}</batch_reports>
""" + _SCHEMA + """
Top-level object: {{"decision":"update|no_update","summary":"synthesis",
"merged_root_causes":[],"updates":[],"skill_operations":[],
"rejected_candidates":[],"unresolved_questions":[]}}.
"""


class SGDStandardSkillAgent(SGDSkillAgent):
    """Same standard scheduling/refinement machinery, SGD-specific targets."""
    protocol = "sgd_backbone_online_standard_v2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = {}
        self.working_skill = ""
        self.base_reference = ""
        self.base_action_rules = ""
        self.base_slot_policies = ""
        self.runtime_diagnostics_dir = None

    def mine(self, dialogues, directory, **kwargs):
        # Keep the discriminative miner and SGD evidence compiler. The latter
        # must not reuse ABCD's account/credential/positional-slot prompts.
        super().mine(dialogues, directory, **kwargs)
        self.state = initialize_skill_dag(self.graph, self.family)
        labels = {node["id"]: node["label"] for node in self.graph["nodes"]}
        for key, rule in self.resources["action_rules"].items():
            self.state["action_rules"][labels[key]] = {"action": labels[key], "rule": rule, "status": "resolved"}
        for key, policy in self.resources["slot_policies"].items():
            self.state["slot_policies"][labels[key]] = {"action": labels[key], "policy": policy, "status": "resolved"}
        for edge in self.state["edges"].values():
            edge["guard"] = self.resources["transition_guards"].get(self.edge_key(edge), "")
        self.base_action_rules = render_online_action_rules(self.state)
        self.base_slot_policies = render_online_slot_policies(self.state)
        # Offline cards stay in progressive-disclosure resources. The native
        # online state starts with empty refinements, as in the ABCD runner.
        self.state["action_rules"] = {}
        self.state["slot_policies"] = {}
        _, self.base_reference = render_online_resources(self.state)
        self.working_skill = self._seed_skill()
        self.save(directory)

    def _seed_skill(self):
        from skill_mining.skill_writer import _render_backbone_tree, _render_backbone_edge_table
        rows = ["# SGD backbone skill", _TASK.strip(),
                "Graph order organizes knowledge; it does not constrain legal actions.",
                "## Workflow structure", "### Backbone Tree", _render_backbone_tree(self.graph),
                "### Backbone Edges", _render_backbone_edge_table(self.graph),
                "### Routing Policies", "<!-- ROUTING_SECTION_START -->"]
        for source in dict.fromkeys(edge["source"] for edge in self.state["edges"].values()):
            rows.append(f"<!-- ROUTE_SOURCE:{source} -->")
            for edge in self.state["edges"].values():
                if edge["source"] == source and edge["visibility"] == "skill" and edge["guard"]:
                    rows.append(f"<!-- ROUTE_EDGE:{edge['source']}=>{edge['target']} -->")
                    rows.append(f"- {edge['source_action']} -> {edge['target_action']}: {edge['guard']}")
        rows.append("<!-- ROUTING_SECTION_END -->")
        rows.append("## Action semantics")
        for node in self.graph["nodes"]:
            rows.append(f"- `{node['label']}` / `{node['id']}`: {json.dumps(node['action_heads'])}")
        rows.append("Compare retrieved Action Cards before selection; bind payloads only after selecting heads.")
        # Native graph promotions need an editable region, even though offline
        # action cards remain deferred resources rather than inline skill text.
        rows.extend(["### Action Rules", "<!-- ACTION_RULES_START -->", "<!-- ACTION_RULES_END -->"])
        return "\n".join(rows)

    def runtime_resources(self, dialogue, turn_index):
        local = super().runtime_resources(dialogue, turn_index)
        if not self.state:
            return local  # During the inherited offline compiler's save.
        guards, reference = render_online_resources(self.state)
        # The full working skill is now the authoritative runtime artifact;
        # state-level updates are supplied too, even without text operations.
        local["skill_text"] = "\n\n".join((self.working_skill, guards,
            render_online_action_rules(self.state), render_online_slot_policies(self.state)))
        from skill_mining.online_refinement import _resource_lookup_sections
        query = visible_context(dialogue, turn_index)[-1800:] + " " + local["source"]
        retrieved = _resource_lookup_sections("reference", self.base_reference + "\n" + reference, query, 2)
        local["reference_text"] = json.dumps(retrieved, ensure_ascii=False)[:self.reference_max_chars]
        return local

    def resource_hash(self):
        value = {"resources": self.resources, "state": self.state, "skill": self.working_skill,
                 "reference": self.base_reference, "action_rules": self.base_action_rules,
                 "slot_policies": self.base_slot_policies}
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _chat(self, messages, **kwargs):
        # Explicit dependency injection: no global monkeypatch, thread leakage
        # or modification of the server's workflow-aware llm.py.
        return sgd_chat_with_retry(messages, model=kwargs["model"], temperature=0,
                                   call_tag=kwargs["call_tag"])

    def predict(self, dialogue, turn_index):
        context = visible_context(dialogue, turn_index)[-12000:]
        resources = self.runtime_resources(dialogue, turn_index)
        nodes = {node["label"]: node for node in self.graph["nodes"]}
        plan_prompt = ("Plan one read-only resource lookup before SGD policy selection. "
            "Return JSON {\"query_text\":\"...\",\"candidate_actions\":[\"known node labels\"]}. "
            "Candidates are advisory, not an output constraint. No payload values yet. "
            "Keep query_text short (at most 300 characters), return at most 3 candidate labels. "
            "Do not reproduce skill text or give explanations; return only the JSON object.\n"
            + self.working_skill + "\nDialogue so far:\n" + context)
        plan, plan_meta = self._request_validated_json(plan_prompt, dialogue, turn_index,
            stage="reference_plan", validator=self._validate_reference_plan)
        from skill_mining.online_refinement import _resource_lookup_sections
        _, online_reference = render_online_resources(self.state)
        query = str(plan.get("query_text", "")) or context[-1800:]
        reference = _resource_lookup_sections("reference", self.base_reference + "\n" + online_reference, query, 2)
        candidates = plan.get("candidate_actions", [])
        if not isinstance(candidates, list):
            candidates = []
        candidate_ids = list(dict.fromkeys([nodes[label]["id"] for label in candidates if isinstance(label, str) and label in nodes]
                                          + resources["candidate_ids"]))[:self.candidate_limit]
        by_id = {node["id"]: node for node in self.graph["nodes"]}
        cards = []
        for key in candidate_ids:
            label = by_id[key]["label"]
            rule = self.state["action_rules"].get(label, {})
            policy = self.state["slot_policies"].get(label, {})
            cards.append({"label": label, "heads": by_id[key]["action_heads"],
                "rule": rule.get("rule", "") if rule.get("status") == "resolved" else self.resources["action_rules"][key],
                "payload_policy": policy.get("policy", "") if policy.get("status") == "resolved" else self.resources["slot_policies"][key]})
        reference_text = json.dumps(reference, ensure_ascii=False)[:self.reference_max_chars]
        selection_prompt = ("Select the next SGD system turn's UNORDERED action-head set. "
            "Return JSON {\"action_heads\":[[\"intent\",\"actual service ID or empty for global act\"]]}. "
            "Allowed intents: call, request, confirm, inform, inform_count, offer, offer_intent, "
            "notify_success, notify_failure, req_more, goodbye. Do not bind function/slots/values yet. "
            "Compare candidate action cards by context and rules, not by ease of slot filling. "
            "You may choose valid unlisted combinations. At most one distinct CALL. "
            "Use lower-case intents and EXACT actual service IDs from Services below, "
            "not the family names shown in mined cards. Global acts use an empty string.\nServices: "
            + json.dumps(dialogue["services"]) + "\nSkill:\n" + resources["skill_text"]
            + "\nCandidate cards:\n" + json.dumps(cards, ensure_ascii=False)
            + "\nRetrieved reference:\n" + reference_text + "\nDialogue so far:\n" + context)
        heads, selection_meta = self._select_heads(selection_prompt, dialogue, turn_index)
        prediction, diagnostic = self.predictor(dialogue, turn_index, model=self.model,
            ontology=self.ontology, frequency_graph=None,
            skill_text=resources["skill_text"] + "\nRetrieved Action Cards:\n" + json.dumps(cards, ensure_ascii=False),
            reference_text=reference_text, selected_action_heads=heads)
        diagnostic.update(source_node=resources["source"], candidate_ids=candidate_ids,
            resource_sha256=self.resource_hash(), reference_plan=plan, retrieved_reference=reference,
            selected_action_heads=heads, action_selection=selection_meta, reference_plan_diagnostic=plan_meta)
        return prediction, diagnostic

    def _select_heads(self, prompt, dialogue, turn_index):
        return self._request_validated_json(prompt, dialogue, turn_index, stage="action_selection",
            validator=lambda payload: normalize_action_heads(payload, dialogue["services"]),
            result_field="action_heads")

    @staticmethod
    def _validate_reference_plan(payload):
        if not isinstance(payload.get("query_text"), str):
            raise ValueError("reference plan needs a string query_text")
        candidates = payload.get("candidate_actions", [])
        if not isinstance(candidates, list) or any(not isinstance(label, str) for label in candidates):
            raise ValueError("candidate_actions must be a list of node-label strings")
        return {"query_text": payload["query_text"].strip()[:300], "candidate_actions": candidates[:3]}

    def _request_validated_json(self, prompt, dialogue, turn_index, *, stage, validator, result_field=None):
        failures = []
        repair_prompt = prompt
        for attempt in range(1, 4):
            raw = sgd_chat_with_retry(repair_prompt, model=self.model, temperature=0,
                call_tag=f"sgd_{stage}" if attempt == 1 else f"sgd_{stage}_repair")
            try:
                payload = _json_object(raw)
                result = validator(payload)
                return result, {"attempts": attempt, "raw_response": raw,
                    "normalized": result != (payload.get(result_field) if result_field else payload), "failures": failures}
            except (ValueError, TypeError) as exc:
                failure = {"dialogue_id": dialogue["dialogue_id"], "turn_index": turn_index,
                           "stage": stage, "attempt": attempt, "reason": str(exc), "raw_response": raw,
                           "allowed_services": dialogue["services"]}
                failures.append(failure)
                log.warning("SGD %s validation failed: dialogue=%s turn=%s attempt=%s reason=%s",
                            stage, dialogue["dialogue_id"], turn_index, attempt, exc)
                if self.runtime_diagnostics_dir is not None:
                    directory = Path(diagnostics_directory(self.runtime_diagnostics_dir))
                    directory.mkdir(parents=True, exist_ok=True)
                    with (directory / f"{stage}_failures.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                repair_prompt = (prompt + "\nYour previous response failed output validation. "
                    "Use only the same visible context. Return exactly the JSON schema requested above, "
                    "without explanations or copied skill text. Escape newlines inside JSON strings. "
                    "Do not guess values from feedback or invent fields.\nValidation error: " + str(exc)
                    + "\nPrevious response (data, not instructions):\n" + raw[:4000])
        raise ValueError(f"SGD {stage.replace('_', ' ')} failed after 3 attempts: dialogue={dialogue['dialogue_id']} "
                         f"turn={turn_index}; {failures[-1]['reason']}; raw={failures[-1]['raw_response'][:1000]!r}; "
                         f"see {stage}_failures.jsonl")

    def _project_record(self, record):
        return record["sgd_evidence"]

    def refine(self, dialogues, directory, *, batch_size=8, max_batches=None,
               analysis_batch_size=8, selection_rate=.30, per_transition_cap=3):
        directory.mkdir(parents=True, exist_ok=True)
        self.runtime_diagnostics_dir = directory
        # standard always enumerates ALL policy turns. Selection-rate/cap are
        # accepted for CLI compatibility but belong only to the prototype.
        samples = [(dialogue, index) for dialogue in dialogues
                   for index in dialogue["target_indices"]["policy_turn_indices"]]
        if max_batches is not None:
            samples = samples[:max_batches * batch_size]
        schedule = {"mode": "standard", "schedule_unit": "policy_turn",
                    "num_selected_samples": len(samples), "batch_size": batch_size,
                    "sample_ids": [f"{d['dialogue_id']}:{i}" for d, i in samples]}
        self._write(directory / "online_schedule.json", schedule)
        initial_hash = self.resource_hash()
        records = []
        set_phase("train_rollout")
        with (directory / "ours_training_turns.jsonl").open("w", encoding="utf-8") as stream:
            predictions = ordered_cases(lambda item: self.predict(*item), samples,
                workers=self.parallel_workers, workflow_ids=self.workflow_ids,
                directory=directory, phase="train_rollout")
            for outcome, (dialogue, index) in zip(predictions, samples):
                try:
                    prediction, diagnostic = require_value(outcome)
                    gold = gold_policy(dialogue, index)
                    scores = score_policies([gold], [prediction])
                    action_ok = bool(scores["action_set_exact"])
                    slot_ok = slot_signature(gold) == slot_signature(prediction)
                except Exception as exc:
                    record_error(directory, "train_rollout", exc,
                                 dialogue_id=dialogue["dialogue_id"], turn_index=index)
                    stream.write(json.dumps({"sample_id": f"{dialogue['dialogue_id']}:{index}",
                                             "status": "error", "error": str(exc)}, ensure_ascii=False) + "\n")
                    stream.flush()
                    continue
                source = previous_node(dialogue, index, self.family)
                target, predicted = policy_token(gold), policy_token(prediction)
                prefix = [policy_token(gold_policy(dialogue, i)) for i in
                          dialogue["target_indices"]["policy_turn_indices"] if i < index]
                sample_id = f"{dialogue['dialogue_id']}:{index}"
                evidence = {"sample_id": sample_id, "conversation_id": dialogue["dialogue_id"],
                    "turn_index": index, "source_action": source, "gold_target": target,
                    "gold_policy": gold, "predicted_policy": prediction,
                    "action_correct": action_ok, "slot_exact_correct": slot_ok,
                    "slot_error_bucket": "strict_correct" if slot_ok else "named_payload_mismatch",
                    "context": visible_context(dialogue, index)[-1800:], "runtime_diagnostic": diagnostic}
                record = {"sample": {"sample_id": sample_id, "conversation_id": dialogue["dialogue_id"],
                    "source_action": source, "target_action": target, "prefix_action_sequence": prefix},
                    "result": {"predicted_action": predicted, "context": evidence["context"]},
                    "sgd_evidence": evidence}
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        assert self.resource_hash() == initial_hash, "Rollout resources must stay frozen"
        self.stage = "refining"
        self._accumulate(records)
        self.save(directory)
        reports = []
        set_phase("online_diagnosis")
        batches = list(enumerate(build_post_rollout_batches(records, max_batch_size=analysis_batch_size), 1))
        def diagnose(item):
            batch_index, batch = item
            return diagnose_rollout_batch(str(batch_index), batch, self.state, self.working_skill,
                self.model, reference=self.base_reference, action_rules=self.base_action_rules,
                slot_policies=self.base_slot_policies, record_projector=self._project_record,
                prompt_template=DIAGNOSIS_PROMPT, chat_fn=self._chat)
        diagnoses = ordered_cases(diagnose, batches, workers=self.parallel_workers,
            workflow_ids=self.workflow_ids, directory=directory, phase="online_diagnosis")
        for outcome, (batch_index, batch) in zip(diagnoses, batches):
            try:
                report = require_value(outcome)
                self._write(directory / f"diagnosis_{batch_index:04d}.json", report)
                if report["status"] == "error":
                    raise RuntimeError(f"SGD diagnosis failed: {report['error']}")
            except Exception as exc:
                record_error(directory, "online_diagnosis", exc, batch_index=batch_index,
                             sample_ids=[r["sample"]["sample_id"] for r in batch])
                continue
            reports.append(report)
        groups = group_batch_reports(reports, max_reports=16)
        applied_ids = set(self.state.get("applied_skill_operation_ids", []))
        set_phase("online_reflection")
        for group_index, group in enumerate(groups, 1):
            # Reflection groups consume earlier writes, so only independent
            # rollout/diagnosis cases run in parallel.
            if self.workflow_ids:
                import os
                os.environ["SKILLMINING_WORKFLOW_ID"] = self.workflow_ids[(group_index - 1) % len(self.workflow_ids)]
            before = (copy.deepcopy(self.state), self.working_skill, copy.deepcopy(self.ledger), set(applied_ids))
            try:
                reflection = reflect_batch_report_group(group, self.state, self.working_skill,
                    self.model, prompt_template=REFLECTION_PROMPT, chat_fn=self._chat)
                if reflection["status"] == "error":
                    self._write(directory / f"reflection_{group_index:04d}.json", reflection)
                    raise RuntimeError(f"SGD reflection failed: {reflection['error']}")
                accepted, rejected = apply_reflection_updates_to_state(self.state, reflection.get("updates", []))
                self.working_skill, operations = apply_dynamic_skill_operations(
                    self.working_skill, reflection.get("skill_operations", []), applied_ids)
                failed_operations = [op for op in operations if op.get("error")]
                if failed_operations:
                    raise RuntimeError("SGD skill text edits failed: " + json.dumps(failed_operations, ensure_ascii=False))
                self.state["applied_skill_operation_ids"] = sorted(applied_ids)
                reflection.update(accepted_updates=accepted, rejected_updates=rejected,
                                  applied_skill_operations=operations)
                self.ledger.append({"phase": "online_reflection", "group_index": group_index,
                                    "accepted_updates": accepted, "rejected_updates": rejected,
                                    "skill_operations": operations})
                self.save(directory)  # Validate/materialize before committing proposal diagnostics.
                self._write(directory / f"reflection_{group_index:04d}.json", reflection)
            except Exception as exc:
                self.state, self.working_skill, self.ledger, applied_ids = before
                failure = record_error(directory, "online_reflection", exc, group_index=group_index,
                                       batch_ids=[r.get("batch_id") for r in group], rolled_back=True)
                self._write(directory / f"reflection_{group_index:04d}_error.json", failure)
                # A structural edit error cannot poison the next group. Disk
                # write failures are also recorded; in-memory K1 stays valid.
                try:
                    self.save(directory)
                except Exception as rollback_exc:
                    record_error(directory, "checkpoint_rollback", rollback_exc, group_index=group_index)
        failures = error_summary(directory)
        self.state["refinement_errors"] = failures
        self.stage = "refined_with_errors" if failures["total"] else "refined"
        self.save(directory)
        summary = {"protocol": self.protocol, "schedule_unit": "policy_turn",
                   "rollout_turns": len(records), "scheduled_turns": len(samples),
                   "failed_rollout_turns": len(samples) - len(records),
                   "reports": len(reports), "reflection_groups": len(groups),
                   "case_errors": failures,
                   "selected_dialogue_ids": list(dict.fromkeys(d["dialogue_id"] for d, _ in samples)),
                   "snapshot_sha256": initial_hash, "final_sha256": self.resource_hash()}
        self._write(directory / "ours_training.json", summary)
        return summary

    def _accumulate(self, records):
        labels = {node["id"]: node["label"] for node in self.graph["nodes"]}
        for record in records:
            sample, row = record["sample"], record["sgd_evidence"]
            source = sample["source_action"]
            target_id = f"{self.family}:{sample['target_action']}"
            edge = self.state["edges"].setdefault(f"{source}=>{target_id}", {
                "source": source, "target": target_id, "source_action": labels.get(source, source),
                "target_action": sample["target_action"], "kind": "candidate_branch", "visibility": "reference",
                "offline_support": 0, "offline_sessions": 0, "gold_support": 0,
                "rollout_success": 0, "rollout_failure": 0, "slot_total": 0, "slot_success": 0,
                "slot_failures": 0, "competing_targets": Counter(), "guard": "", "guard_status": "pending", "evidence": []})
            edge["gold_support"] += 1
            edge["rollout_success" if row["action_correct"] else "rollout_failure"] += 1
            if not row["action_correct"]:
                edge["competing_targets"][record["result"]["predicted_action"]] += 1
            if len(edge["evidence"]) < 12:
                edge["evidence"].append(copy.deepcopy(row))
            if row["action_correct"]:
                edge["slot_total"] += 1
                edge["slot_success" if row["slot_exact_correct"] else "slot_failures"] += 1
                policy = self.state["slot_policies"].setdefault(labels[target_id], {
                    "action": labels[target_id], "policy": "", "status": "pending", "evidence": []})
                policy["slot_total"] = policy.get("slot_total", 0) + 1
                key = "slot_success" if row["slot_exact_correct"] else "slot_failures"
                policy[key] = policy.get(key, 0) + 1
                if len(policy["evidence"]) < 12:
                    policy["evidence"].append(copy.deepcopy(row))
        self.state["batches_processed"] = len(records)

    @staticmethod
    def _write(path, value):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def save(self, directory):
        if self.state:
            # Fail before overwriting any checkpoint with an invalid edit.
            self.working_skill = merge_online_skill_additions(self.working_skill, self.state)
        super().save(directory)
        if not self.state:
            return
        save_skill_dag(self.state, directory / "skill_dag_state.json")
        guards, reference = render_online_resources(self.state)
        for name, text in {"skill.md": self.working_skill, "working_skill.md": self.working_skill,
                "online_transition_guards.md": guards, "reference.md": self.base_reference + "\n" + reference,
                "action_rules.md": self.base_action_rules + "\n" + render_online_action_rules(self.state),
                "slot_policies.md": self.base_slot_policies + "\n" + render_online_slot_policies(self.state)}.items():
            (directory / name).write_text(text, encoding="utf-8")
        self._write(directory / "sgd_standard_state.json", {"protocol": self.protocol,
            "family": self.family, "state": self.state, "working_skill": self.working_skill,
            "base_reference": self.base_reference, "base_action_rules": self.base_action_rules,
            "base_slot_policies": self.base_slot_policies})

    def load(self, directory):
        path = directory / "sgd_standard_state.json"
        if not path.exists():
            raise ValueError("Standard requires a standard checkpoint; prototype resources are not equivalent")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != self.protocol or payload.get("family") != self.family:
            raise ValueError("SGD standard protocol/domain mismatch")
        super().load(directory)
        self.state = payload["state"]
        for edge in self.state["edges"].values():
            edge["competing_targets"] = Counter(edge.get("competing_targets", {}))
        for key in ("working_skill", "base_reference", "base_action_rules", "base_slot_policies"):
            setattr(self, key, payload[key])
