"""Turn-level MultiWOZ action/slot projection for the two-stage skill pilot."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from eval_tod.sgd_adapter import canonical_acts, visible_context
from eval_tod.sgd_agent import SGDAWMAgent
from eval_tod.sgd_llm import sgd_chat_with_retry
from eval_tod.sgd_skill_agent import _json_object
from eval_tod.sgd_standard_skill_agent import (
    DIAGNOSIS_PROMPT, REFLECTION_PROMPT, SGDStandardSkillAgent, _TASK,
)


INTENTS = ("book", "bye", "greet", "inform", "nobook", "nooffer", "offerbook",
           "recommend", "reqmore", "request", "select", "welcome")
TASK = """
MultiWOZ targets are unordered sets of system dialogue acts. One action node is
the complete set of (intent, domain) heads emitted at a single system turn.
Slot names and values are payloads, not graph nodes. There is no CALL action in
this benchmark protocol. Use only prior dialogue utterances at prediction time;
the current system utterance, gold acts and future turns are hidden. Never
invent an order between simultaneous acts. Diagnose action-set errors separately
from named-slot/value errors. Preserve supported alternatives, generalize from
training examples, and do not memorize instance values.
"""


def _domain_key(dialogue: dict) -> str:
    return next((domain for domain in dialogue.get("domains", []) if domain != "general"), "general")


def _acts(turn: dict) -> list[dict]:
    return canonical_acts([
        {"intent": act["intent"], "domain": "" if act.get("domain") == "general" else act.get("domain", ""),
         "slot": act.get("slot") or "", **({"value": act["value"]} if "value" in act else {})}
        for group in (turn.get("dialogue_acts") or {}).values()
        for act in group if isinstance(act, dict) and act.get("intent")
    ])


def project_dialogue(dialogue: dict, family: str) -> dict:
    services = [domain for domain in dialogue.get("domains", []) if domain != "general"]
    turns = []
    indices = []
    for index, turn in enumerate(dialogue["turns"]):
        acts = _acts(turn) if turn.get("speaker") == "system" else []
        if acts:
            indices.append(index)
        annotation_types = [
            {"intent": act["intent"], "domain": "" if act.get("domain") == "general" else act.get("domain", ""),
             "slot": act.get("slot") or "", "type": kind}
            for kind, group in (turn.get("dialogue_acts") or {}).items()
            for act in group if isinstance(act, dict) and act.get("intent")
        ] if turn.get("speaker") == "system" else []
        turns.append({"speaker": turn.get("speaker", ""),
                      "utterance": turn.get("utterance", ""),
                      "dialogue_acts": {"all": acts}, "act_annotation_types": annotation_types,
                      "service_call": {}, "db_results": {}})
    return {"dialogue_id": dialogue["dialogue_id"], "original_id": dialogue.get("original_id"),
            "domain_family": family, "services": services,
            # The SGD scorer includes service-free acts only for one family.
            # Here all acts belong to this complete-dialogue scenario.
            "all_services": [family],
            "target_indices": {"policy_turn_indices": indices,
                               "api_turn_indices": [], "domain_act_turn_indices": indices,
                               "global_act_turn_indices": []},
            "turns": turns}


def load_hotel_pilot(source: Path, *, seed: int = 42, fraction: float = .1) -> dict[str, list[dict]]:
    if not 0 < fraction <= 1 or not math.isfinite(fraction):
        raise ValueError("fraction must be finite and in (0, 1]")
    raw = json.loads(source.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for dialogue in raw:
        if _domain_key(dialogue) != "hotel":
            continue
        grouped[dialogue["data_split"]].append(dialogue)
    result = {}
    for split in ("train", "validation", "test"):
        rows = grouped[split]
        ranked = sorted(rows, key=lambda item: (
            hashlib.sha256(f"{seed}\0hotel\0{item['dialogue_id']}".encode()).digest(),
            item["dialogue_id"],
        ))
        result[split] = [project_dialogue(item, "hotel")
                         for item in ranked[:math.ceil(len(rows) * fraction)]]
    return result


def ontology_from_train(dialogues: list[dict]) -> dict:
    domains: dict[str, set[str]] = defaultdict(set)
    annotation_types: dict[tuple, set[str]] = defaultdict(set)
    for dialogue in dialogues:
        for domain in dialogue["services"]:
            domains[domain]
        for index in dialogue["target_indices"]["policy_turn_indices"]:
            for annotation in dialogue["turns"][index].get("act_annotation_types", []):
                annotation_types[(annotation["intent"], annotation["domain"], annotation["slot"])].add(annotation["type"])
            for act in dialogue["turns"][index]["dialogue_acts"]["all"]:
                if act["domain"] and act["slot"]:
                    domains[act["domain"]].add(act["slot"])
    return {"domains": {domain: {"slots": sorted(slots)} for domain, slots in sorted(domains.items())},
            # Learn value-free shapes from TRAIN annotation kinds, never from
            # missing values in an evaluation label or an entity's DB value.
            # Mixed-kind keys keep their values: there is no safe global rule.
            "value_free_acts": [list(key) for key, kinds in sorted(annotation_types.items())
                                if kinds == {"binary"}]}


def _parse_acts(raw: str) -> list[dict]:
    payload = _json_object(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("acts"), list):
        raise ValueError("MultiWOZ prediction requires JSON object with acts list")
    return canonical_acts(payload["acts"])


def predict_policy(dialogue: dict, turn_index: int, *, model: str, ontology: dict,
                   frequency_graph=None, workflow_text: str = "", exemplar_text: str = "",
                   skill_text: str = "", reference_text: str = "",
                   selected_action_heads: list | None = None) -> tuple[dict, dict]:
    context = visible_context(dialogue, turn_index)[-12000:]
    services = dialogue["services"]
    schema = {domain: ontology["domains"].get(domain, {}) for domain in services}
    instructions = (
        "Predict the next system turn's MultiWOZ dialogue acts from prior utterances. "
        "Return JSON only: {\"acts\":[{\"intent\":\"inform\",\"domain\":\"hotel\","
        "\"slot\":\"name\",\"value\":\"example\"}]}. "
        "An act may omit value. Multiple acts with different slots may share a head. "
        "Use only these intents: " + ", ".join(INTENTS) + ". "
        "Use domain names exactly as listed; service-free general acts use empty domain. "
        "Do not include an API CALL or natural-language response. "
        "Do not read or infer the current system utterance or future turns. "
    )
    if selected_action_heads is not None:
        instructions += ("Action heads are FIXED; bind only their named slots and values. "
                         "Emit each selected head at least once and no other head. "
                         + json.dumps(selected_action_heads, ensure_ascii=False) + "\n")
    prompt = (instructions + "\nDomains and observed TRAIN slot names: "
              + json.dumps(schema, ensure_ascii=False)
              + ("\nAWM workflow:\n" + workflow_text if workflow_text else "")
              + ("\nVerified examples:\n" + exemplar_text if exemplar_text else "")
              + ("\nSkill:\n" + skill_text if skill_text else "")
              + ("\nRetrieved reference:\n" + reference_text if reference_text else "")
              + "\nDialogue so far:\n" + context)
    raw = sgd_chat_with_retry(prompt, model=model, temperature=0,
                              call_tag="multiwoz_act_payload")
    try:
        acts = _parse_acts(raw)
    except (ValueError, json.JSONDecodeError):
        # Some gateways occasionally return a list or prose despite a JSON
        # instruction. Retry the one malformed prediction with a strict schema.
        repair = (prompt + "\n\nThe previous response did not match the required JSON "
                  "object schema. Return only one JSON object with an acts array. "
                  "No markdown, commentary, or top-level array.\nPrevious response:\n"
                  + raw[-3000:])
        acts = _parse_acts(sgd_chat_with_retry(repair, model=model, temperature=0,
                                                call_tag="multiwoz_act_payload_repair"))
    allowed = set(services)
    acts = [act for act in acts if act["domain"] in allowed or not act["domain"]]
    if selected_action_heads is not None:
        heads = {tuple(head) for head in selected_action_heads}
        acts = [act for act in acts if (act["intent"], act["domain"]) in heads]
        observed = {(act["intent"], act["domain"]) for act in acts}
        for intent, domain in sorted(heads - observed):
            acts.append({"intent": intent, "domain": domain, "slot": ""})
    return {"call": None, "acts": canonical_acts(acts)}, {"status": "ok"}


class MultiWOZAWMAgent(SGDAWMAgent):
    dataset_name = "MultiWOZ"
    induction_task_contract = (
        "Each policy turn has an unordered set of (intent, domain) dialogue acts. "
        "There is no API CALL action. Learn when to choose action heads and how to bind "
        "their named slots and values from prior utterances. Generalize across dialogues; "
        "do not memorize entity values or use future system utterances."
    )


class MultiWOZStandardSkillAgent(SGDStandardSkillAgent):
    protocol = "multiwoz21_hotel_action_slot_standard_v3"
    dataset_name = "MultiWOZ"
    task_prompt = TASK
    diagnosis_prompt = DIAGNOSIS_PROMPT.replace(_TASK, TASK)
    reflection_prompt = REFLECTION_PROMPT.replace(_TASK, TASK) + """
Adapter details: 'hotel' selects a complete-dialogue scenario; the same dialogue
can contain train, restaurant, attraction and other domains. Those observed
nodes are valid. Preserve the fixed node-to-action-head definitions and the
backbone. No database query tool or extra API observation is available here.
Only Routing Policies and Action Rules are editable inline at runtime.
Deferred Action Cards are external resources: update them by action_rule or
slot_policy upsert, never by matching their prose inside current_skill.
"""
    selection_intents = ", ".join(INTENTS)
    selection_call_rule = "Do not select CALL. "
    selection_target_label = "domain name"

    def payload_resources(self, heads, cards, resources, reference_text):
        # The action shortlist is advisory. The selected action can therefore
        # be absent from it; fetch its complete card from the full resource pool.
        selected = {tuple(head) for head in heads}
        matches = [node for node in self.graph["nodes"]
                   if {tuple(head) for head in node["action_heads"]} == selected]
        selected_cards = [self.action_card(node) for node in matches]
        text = (self.task_prompt.strip()
                + "\nAction selection is complete. Bind payloads under the fixed heads "
                "using the selected Action Card's procedure and payload_policy.\n"
                "Selected Action Card:\n" + json.dumps(selected_cards, ensure_ascii=False))
        if not matches:
            text += "\nNo saved card exists for this exact action bundle. Use the declared slot schema and visible context."
        lookup = {"selected_action_heads": heads, "selected_node_ids": [node["id"] for node in matches],
                  "exact_card_found": bool(matches), "candidate_cards_reused": False,
                  "payload_skill_chars": len(text)}
        return text, reference_text, {"action_card_lookup": lookup,
            "executor_protocol": "multiwoz_action_then_selected_card_v1"}

    def runtime_skill_text(self):
        """Publish one compact backbone view while keeping editable markup in working_skill."""
        if not self.graph or not self.state:
            return self.working_skill
        labels = {str(node["id"]): str(node["label"]) for node in self.graph["nodes"]}
        rows = [f"# {self.dataset_name} backbone skill", self.task_prompt.strip(),
                "Graph order is advisory. Retrieve deferred branches from reference when needed.",
                "## Backbone nodes and action heads"]
        for node in self.graph["nodes"]:
            heads = ", ".join(f"{intent}:{domain or '*'}" for intent, domain in node["action_heads"])
            rows.append(f"- {node['label']} = {heads}")
        rows.append("## Backbone edges")
        for edge in self.graph["backbone"]["edges"]:
            source = labels.get(str(edge["source"]), "ROOT")
            target = labels.get(str(edge["target"]), str(edge["target"]))
            rows.append(f"- {source} > {target}")
        for edge in self.state["edges"].values():
            if edge["kind"] == "promoted_branch" and edge["visibility"] == "skill":
                rows.append(f"- {edge['source_action']} > {edge['target_action']}")
        rows.append("## Routing policies")
        routing = self.working_skill.split("### Routing Policies", 1)
        if len(routing) == 2:
            body = routing[1].split("## Action semantics", 1)[0]
            rows.extend(line for line in body.splitlines()
                        if line.strip() and not line.lstrip().startswith("<!--"))
        action_rules = self.working_skill.split("### Action Rules", 1)
        if len(action_rules) == 2:
            rules = [line for line in action_rules[1].splitlines()
                     if line.strip() and not line.lstrip().startswith("<!--")]
            if rules:
                rows.extend(["## Online action rules", *rules])
        return "\n".join(rows)
    
    def runtime_resources(self, dialogue, turn_index):
        resources = super().runtime_resources(dialogue, turn_index)
        if self.state:
            # Accepted updates are materialized in working_skill; selected
            # action cards carry their local action/slot policy. Repeating all
            # guards in both downstream calls adds no new runtime evidence.
            resources["skill_text"] = self.runtime_skill_text()
        return resources
    runtime_task_intro = (
        "MultiWOZ backbone/residual skill: nodes represent unordered system-turn "
        "action sets. Domains are the exact names listed for this dialogue. "
        "Candidates and graph order are advisory; use local routing and named-slot "
        "rules without copying training values.\n"
    )
    offline_task_intro = (
        "Compile local MultiWOZ skill rules from TRAIN traces without rollout. "
        "Nodes are unordered sets of (intent, domain) heads emitted at one system turn. "
        "Named slots and values are payloads. There is no CALL action. "
        "Infer observable routing and slot-grounding rules from prior dialogue context; "
        "never rely on the current or future system utterance or memorize instance values. "
    )

    def _select_heads(self, prompt, dialogue, turn_index):
        allowed_domains = set(dialogue["services"]) | {""}
        allowed_intents = set(INTENTS)

        def validate(payload):
            heads = payload.get("action_heads")
            if not isinstance(heads, list):
                raise ValueError("action_heads must be a list")
            normalized = set()
            for head in heads:
                if isinstance(head, dict):
                    head = [head.get("intent"), head.get("domain")]
                if not isinstance(head, list) or len(head) != 2:
                    raise ValueError("each action head must be [intent, domain]")
                intent = str(head[0]).strip().lower()
                domain = "" if head[1] is None else str(head[1]).strip()
                if intent not in allowed_intents or domain not in allowed_domains:
                    raise ValueError(f"unknown MultiWOZ head: {(intent, domain)!r}")
                normalized.add((intent, domain))
            if not normalized:
                raise ValueError("at least one action head is required")
            return [list(head) for head in sorted(normalized)]

        return self._request_validated_json(prompt, dialogue, turn_index,
                                            stage="action_selection", validator=validate,
                                            result_field="action_heads")
