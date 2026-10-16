"""Full-context K1 control, reusing ABCD history, selection, parser and metrics."""
from __future__ import annotations

import re

from .agent import (ABCDAgent, _AWM_RESOURCE_POLICY_PROMPT, _POLICY_RESOURCE_PROMPT,
                    _REFERENCE_TOOL_PROMPT)


class K1FullContextAgent(ABCDAgent):
    """No retrieval/planning call; all flat K1 knowledge is in both stages.

    The inherited turn loop still selects an action then locks it for ordered
    slot grounding. Invalid selections follow the same fallback as Ours.
    """

    def _plan_reference_lookup(self, context, scenario, verbose=False):
        return {"thought": "All K1 records are already visible; no tool planning.",
                "query_text": "", "top_k": 0, "raw_output": "", "tool_call": {},
                "messages": [], "fallback_used": False, "candidate_actions": []}

    def _lookup_reference(self, query_text, context, scenario, top_k=None):
        return {"tool": None, "query": {}, "executed": False, "status": "full_context",
                "selected_sections": [], "observation": "All flat K1 resources are already in the system message."}

    def _action_selection_candidates(self, *args, **kwargs):
        return []  # Full vocabulary stays available, not an artificial shortlist.

    def _retrieve_action_cards(self, candidate_actions, max_chars=2200):
        self._last_action_card_lookup = {"tool": None, "executed": False,
                                        "status": "full_context", "query": [], "selected_actions": []}
        return ""  # Cards are present in full in workflow; never bounded here.

    def _build_system_prompt(self, scenario, context="", candidate_actions=None):
        system = super()._build_system_prompt(scenario, context, candidate_actions)
        for text in (_REFERENCE_TOOL_PROMPT, _POLICY_RESOURCE_PROMPT):
            system = system.replace(text, "")
        system = system.replace(_AWM_RESOURCE_POLICY_PROMPT,
            "All learned graph records, action rules and slot policies are provided in full below. "
            "No resource tools or retrieval planning are available or necessary. "
            "Use the complete action definition already in context for each selected action. "
            "Legacy file/tool references inside copied rules denote supporting evidence already "
            "provided inline, not instructions to access unavailable files or tools.")
        self._last_prompt_budget["system_chars"] = len(system)
        return system

    def _chat(self, messages, **kwargs):
        # Keep recorded messages identical to the actual provider request.
        for message in messages:
            if message.get("role") != "user":
                continue
            text = re.sub(r'<retrieved_reference tool="retrieve_reference">.*?</retrieved_reference>\s*',
                          "", message["content"], flags=re.S)
            text = text.replace("stage will retrieve the exact action card", "stage will use the action definition already in context")
            text = text.replace("Use the exact retrieved action card", "Use the exact action definition already provided")
            text = text.replace("The retrieved\nreference is optional evidence", "The flat graph is procedural evidence")
            message["content"] = text
        return super()._chat(messages, **kwargs)

    def predict_all_turns(self, *args, **kwargs):
        rows = super().predict_all_turns(*args, **kwargs)
        for row in rows:
            row["resource_access_mode"] = "flat_full_context"
            row["client_resource_truncated"] = False
            row["provider_context_truncation"] = "unknown"
            row["inference_trace"] = [step for step in row.get("react_trace", [])
                                      if step.get("action", "").startswith("llm_")
                                      and step.get("action") != "llm_plan_reference_query"]
            row["react_trace"] = []  # Do not mislabel the two-stage predictor as ReAct.
        return rows
