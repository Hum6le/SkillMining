"""No-skill, two-stage ABCD baseline (not an interactive environment ReAct).

Uses the shared dialogue view, interface schema, parser and workflow-aware
LLM client. No graph, mined cards, routing, examples or retrieval are exposed.
"""
from __future__ import annotations

import hashlib
import json
import re

from .agent import ABCDAgent, _SYSTEM_PROMPT, _get_original_turn, _parse_action_response
from .action_schema import action_contract_prompt, canonical_action_name, canonicalize_prediction


SELECT_PROMPT = """<dialogue_history>
{context}
</dialogue_history>
Select the next backend action from the allowed vocabulary using only the
conversation and the supplied background facts. Do not fill slots yet.
Return only a JSON object: {{"action": "canonical-action-name"}}.
There is no live environment, knowledge lookup or backend execution here.
"""

GROUND_PROMPT = """<dialogue_history>
{context}
</dialogue_history>
The selected action is fixed: {action}
{slot_contract}
Fill its ordered slot values from the conversation or compatible background
facts, then write a brief customer-facing response. Do not change the action.
Do not invent missing values. No backend action is actually executed.
Return only JSON: {{"action": "{action}", "slots": [], "response": "..."}}.
"""

UTTERANCE_PROMPT = """<dialogue_history>
{context}
</dialogue_history>
Write only the next helpful customer-facing utterance. Do not continue the
conversation or pretend to execute a backend action.
"""


class ReactBaseAgent(ABCDAgent):
    """Explicitly resource-free protocol; unsupported resource kwargs fail closed."""

    def __init__(self, model="deepseek-chat", api_key=None, base_url=None,
                 workflow_id=None, response_logger=None):
        super().__init__(model=model, api_key=api_key, base_url=base_url,
                         workflow_id=workflow_id, response_logger=response_logger,
                         expose_scenario_labels=False, reference_top_k=0,
                         competitive_action_cards=False)

    def _assert_no_resources(self):
        if (self.workflow.text.strip() or len(self.memory) or self.reference_text.strip()
                or self.reference_sections or self.action_cards or self.action_rule_sections
                or self.slot_policy_sections or self.observable_router
                or self.structural_router or self.motif_router or self.expose_scenario_labels):
            raise ValueError("react_base cannot load skills, memory, routers or task labels")

    def protocol(self):
        """Fingerprint actual prompts/schema, including train-derived arities."""
        schema = {"actions": sorted(self.action_schema["actions"]), "slot_counts": {
            key: sorted(value) for key, value in sorted(self.action_schema["slot_counts"].items())}}
        prompts = [_SYSTEM_PROMPT, action_contract_prompt(self.action_schema),
                   SELECT_PROMPT, GROUND_PROMPT, UTTERANCE_PROMPT]
        digest = lambda value: hashlib.sha256(json.dumps(
            value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return {"name": "abcd-react-base-v1", "model": self.model,
                "stages": ["action_selection", "action_slot_grounding"],
                "temperatures": [0.0, 0.0], "action_locked": True,
                "knowledge_resources": [], "retrieval_calls": 0,
                "history": "original-prefix-teacher-forced",
                "scenario_labels": False, "scenario_facts": "shared-abcd-runtime",
                "schema_source": "shared global guideline action vocabulary + train actions/arities",
                "schema_sha256": digest(schema), "prompt_sha256": digest(prompts),
                "on_case_error": "record-and-continue; no joint-generation fallback"}

    def _build_system_prompt(self, scenario, context="", candidate_actions=None):
        self._assert_no_resources()
        return (self._build_task_system_prompt(scenario) + "\n\n<action_slot_contract>\n"
                + action_contract_prompt(self.action_schema) + "\n</action_slot_contract>")

    def _fixed_action_slot_contract(self, action):
        if not self.action_schema.get("slot_counts", {}).get(action):
            return "Return the ordered real slot values; no named-slot schema is available."
        return super()._fixed_action_slot_contract(action)

    def _call_stage(self, system, prompt, tag, trace):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        step = {"turn": len(trace) + 1, "action": tag,
                "action_input": {"messages": messages}, "executed": True}
        trace.append(step)
        try:
            raw = self._chat(messages, model=self.model, api_key=self.api_key,
                             base_url=self.base_url, temperature=0.0,
                             response_logger=self._response_logger, call_tag=tag).strip()
            step["observation"] = raw
            if not raw:
                raise ValueError("Empty LLM response")
            return raw
        except Exception as exc:
            step["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def predict_all_turns(self, conversation, verbose=False, predict_actions=False,
                          action_only=False, turn_index=None):
        self._assert_no_resources()
        delexed = conversation.get("delexed", [])
        scenario = conversation.get("scenario", {})
        convo_id = str(conversation.get("convo_id", "?"))
        action_indices = {i for i, turn in enumerate(delexed)
                          if len(turn.get("targets", [])) >= 3
                          and turn["targets"][1] == "take_action"}
        agent_indices = [i for i, turn in enumerate(delexed)
                         if turn.get("speaker") == "agent" and turn.get("text", "").strip()]
        targets = sorted(action_indices if action_only else action_indices | set(agent_indices))
        if turn_index is not None:
            targets = [turn_index] if turn_index in targets else []
        results = []
        for index in targets:
            # Exactly the same observable, original-history view as ABCDAgent.
            # Target text/labels below are stored only for evaluation, never prompts.
            context_lines = []
            for previous in range(index):
                speaker, text = _get_original_turn(conversation, previous, delexed[previous])
                if text:
                    label = {"agent": "Agent", "customer": "Customer", "action": "System"}.get(speaker, speaker)
                    context_lines.append(f"[{label}] {text}")
            context = "\n".join(context_lines)
            _, reference_original = _get_original_turn(conversation, index, delexed[index])
            is_action = index in action_indices
            row = {"convo_id": convo_id, "turn_index": index,
                   "agent_turn_num": agent_indices.index(index) + 1 if index in agent_indices else None,
                   "target_type": "action" if is_action else "utterance",
                   "subflow": str(scenario.get("subflow", "")), "flow": str(scenario.get("flow", "")),
                   "context": context, "context_view": "original",
                   "reference": str(delexed[index].get("text", "")),
                   "reference_original": reference_original, "prediction": "",
                   "workflow_injected": False, "workflow_chars": 0,
                   "workflow_injected_chars": 0, "exemplar_injected_chars": 0,
                   "memory_exemplars": 0, "resource_access_mode": "none",
                   "react_trace": [], "errors": []}
            selected = ""
            selection_raw = ""
            selection = {"attempted": False, "two_stage_applied": False,
                         "action_locked": False, "selected_action_card_injected": None,
                         "candidate_actions": [], "grounding_resource_mode": "interface-only"}
            row["action_selection"] = selection
            if predict_actions:
                row.update(predicted_action="", predicted_slots=[])
            try:
                system = self._build_system_prompt(scenario, context)
                row["system_prompt_chars"] = len(system)
                if predict_actions and is_action:
                    selection["attempted"] = True
                    selection_raw = self._call_stage(system, SELECT_PROMPT.format(context=context),
                                                      "action_selection", row["react_trace"])
                    payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", selection_raw.strip(),
                                     flags=re.IGNORECASE).strip()
                    parsed = json.loads(payload)
                    selected, _ = canonical_action_name(
                        parsed.get("action", "") if isinstance(parsed, dict) else "",
                        self.action_schema["actions"])
                    if selected not in self.action_schema["actions"]:
                        selected = ""
                        raise ValueError("Invalid stage-1 action; no fallback or slot call applied")
                    # Keep a valid stage-1 action even if stage 2 fails.
                    row["predicted_action"] = selected
                    selection.update(selected_action=selected, action_locked=True, two_stage_applied=True)
                    raw = self._call_stage(system, GROUND_PROMPT.format(
                        context=context, action=selected,
                        slot_contract=self._fixed_action_slot_contract(selected)),
                        "action_slot_grounding", row["react_trace"])
                    raw_action, slots, response = _parse_action_response(raw)
                    # Keep the same JSON/fenced-JSON/line-format parser as Ours;
                    # do not reject otherwise usable outputs with a stricter parser.
                    action, values, validation = canonicalize_prediction(selected, slots, self.action_schema)
                    row.update(predicted_action=action, predicted_slots=values,
                               action_schema_validation=validation, prediction=response)
                    selection["stage2_reported_action"] = raw_action
                else:
                    row["prediction"] = self._call_stage(system, UTTERANCE_PROMPT.format(context=context),
                                                         "react_base_utterance", row["react_trace"])
            except Exception as exc:
                error = {"stage": row["react_trace"][-1]["action"] if row["react_trace"] else "input",
                         "type": type(exc).__name__, "message": str(exc)}
                row["errors"].append(error)
                print(f"[react_base] convo={convo_id} turn={index} {error['stage']}: {exc}", flush=True)
            selection.update(raw_output=selection_raw, selected_action=selected)
            if selection["attempted"] and row["react_trace"]:
                selection["messages"] = row["react_trace"][0]["action_input"]["messages"]
            results.append(row)
        return results
