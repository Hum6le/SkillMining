"""Dialogue-conditioned router over several graph-compiled ABCD skills."""
from __future__ import annotations

import json
import re
from typing import Any

from .agent import ABCDAgent
from awm.memory import WorkflowStore


class SemanticSkillRouterAgent:
    """Select one latent skill once per session, then use the normal agent."""

    def __init__(self, skills: dict[str, dict[str, str]], cards_prompt: str,
                 model: str = "deepseek-chat", response_logger=None):
        self.skills = skills
        self.cards_prompt = cards_prompt
        self.model = model
        self.response_logger = response_logger
        self.agents = {}
        for skill_id, payload in skills.items():
            workflow = WorkflowStore()
            workflow.replace(payload.get("skill", ""))
            self.agents[skill_id] = ABCDAgent(
                model=model, workflow=workflow,
                reference_text=payload.get("reference", ""),
                expose_scenario_labels=False, response_logger=response_logger,
            )
        self.selection_log: list[dict[str, Any]] = []

    @staticmethod
    def _context(conversation: dict[str, Any]) -> str:
        rows = []
        for turn in conversation.get("delexed") or []:
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            speaker = {"agent": "Agent", "customer": "Customer", "action": "System"}.get(
                turn.get("speaker", ""), turn.get("speaker", ""))
            rows.append(f"[{speaker}] {text}")
        return "\n".join(rows)

    def _select(self, conversation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        from llm import chat
        prompt = (
            "Select the best workflow skill for this dialogue.\n"
            "Do not use dataset labels or hidden state. Shared actions alone are not enough; "
            "use user language and distinctive transition evidence. Return JSON only with "
            "selected_skill, confidence, evidence.\n\n" + self.cards_prompt +
            "\n\n<dialogue_context>\n" + self._context(conversation)[:10000] +
            "\n</dialogue_context>"
        )
        parsed: dict[str, Any] = {}
        try:
            raw = chat([
                {"role": "system", "content": "You are a conservative skill router. Return JSON only."},
                {"role": "user", "content": prompt},
            ], model=self.model, temperature=0.0, response_logger=self.response_logger)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                value = json.loads(match.group(0))
                if isinstance(value, dict):
                    parsed = value
        except Exception as exc:
            parsed = {"error": str(exc)}
        selected = str(parsed.get("selected_skill", ""))
        if selected not in self.skills:
            selected = next(iter(self.skills), "")
            parsed["fallback_used"] = True
        parsed["selected_skill"] = selected
        return selected, parsed

    def predict_all_turns(self, conversation: dict[str, Any], verbose: bool = False,
                          predict_actions: bool = False) -> list[dict]:
        selected, decision = self._select(conversation)
        self.selection_log.append({"convo_id": str(conversation.get("convo_id", "?")), **decision})
        if not selected:
            return []
        rows = self.agents[selected].predict_all_turns(
            conversation, verbose=verbose, predict_actions=predict_actions)
        for row in rows:
            row["router"] = {"selected_skill": selected, **decision}
            row.setdefault("react_trace", []).insert(
                0, {"turn": 0, "action": "select_skill",
                    "action_input": {"available_skills": list(self.skills)},
                    "observation": decision})
        return rows
