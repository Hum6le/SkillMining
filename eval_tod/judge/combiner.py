"""Combiner —— 综合多位专业评判官的评分，给出最终综合分数。"""

from __future__ import annotations

from typing import Any

from .config import COMBINER_DEFINITION
from .prompts import build_combiner_prompt, build_combiner_user_message
from .llm_client import call_llm_structured


class Combiner:
    """Senior evaluator that synthesizes multiple specialist judges' scores.

    The Combiner acts like a QA lead: it reviews each judge's independent
    scores and reasoning, identifies consensus and divergence points, and
    produces a final balanced set of scores.
    """

    def __init__(
        self,
        role: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.role = role or COMBINER_DEFINITION["role"]
        self.model = model or "deepseek-chat"
        self.api_key = api_key
        self.base_url = base_url
        self._system_prompt = build_combiner_prompt(self.role)

    def combine(
        self,
        goal_description: str,
        turns_text: str,
        inform_slots: dict,
        request_slots: dict,
        booking: dict,
        judge_results: list[dict],
    ) -> dict:
        """Synthesize multiple judge evaluations into final scores.

        Args:
            goal_description: Natural-language user goal.
            turns_text: Formatted dialogue turns.
            inform_slots: Agent-predicted inform slots.
            request_slots: Agent-predicted request slots.
            booking: Agent-predicted booking info.
            judge_results: List of per-judge result dicts with keys
                ``judge_name``, ``focus_dimension``, ``scores``, ``reasoning``.

        Returns:
            Dict with ``scores``, ``reasoning``, ``judge_agreement``.
        """
        user_message = build_combiner_user_message(
            goal_description, turns_text, inform_slots, request_slots, booking,
            judge_results,
        )

        result = call_llm_structured(
            system_prompt=self._system_prompt,
            user_message=user_message,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        return result
