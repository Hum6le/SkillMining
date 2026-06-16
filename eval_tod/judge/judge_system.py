"""Multi-agent LLM Judge orchestrator for ToD evaluation.

Ported and adapted from ``LLMasaJudge/judge_system.py``.

Usage::

    from eval_tod.judge import MultiAgentJudge

    judge = MultiAgentJudge(model="deepseek-chat")
    result = judge.evaluate_single(
        goal_description="...",
        turns_text="[user] ...\n[system] ...",
        inform_slots={"hotel": {"type": "hotel"}},
        request_slots={"hotel": ["address"]},
        booking={"hotel": {"reference": "ABC123"}},
    )
    print(result["scores"])  # {task_completion: 4, slot_accuracy: 3, ...}
"""

from __future__ import annotations

import time
from typing import Any

from .base import JudgeAgent, JudgeResult, create_judges_from_config
from .combiner import Combiner
from .config import COMBINER_DEFINITION, JUDGE_DEFINITIONS


class MultiAgentJudge:
    """Multi-agent judge system orchestrator.

    Workflow:
    1. Multiple specialist judges evaluate each dialogue independently
    2. A combiner synthesizes their scores into a final evaluation
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or "deepseek-chat"
        self.api_key = api_key
        self.base_url = base_url
        self.judges: list[JudgeAgent] = []
        self.combiner: Combiner | None = None
        self._init_judges()
        self._init_combiner()

    def _init_judges(self) -> None:
        """Create all judge agents from config."""
        self.judges = create_judges_from_config(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _init_combiner(self) -> None:
        """Create the combiner agent."""
        self.combiner = Combiner(
            role=COMBINER_DEFINITION["role"],
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def evaluate_single(
        self,
        goal_description: str,
        turns_text: str,
        inform_slots: dict,
        request_slots: dict,
        booking: dict,
    ) -> dict:
        """Evaluate a single dialogue through all judges + combiner.

        Args:
            goal_description: Natural-language user goal.
            turns_text: Formatted dialogue turns for judge reading.
            inform_slots: Agent-predicted inform slots.
            request_slots: Agent-predicted request slots.
            booking: Agent-predicted booking info.

        Returns:
            Dict with ``scores``, ``reasoning``, ``judge_agreement``,
            ``individual_scores``, ``judge_reasonings``.
        """
        # Phase 1: Each judge evaluates independently
        judge_results: list[JudgeResult] = []
        judge_reasonings: dict[str, str] = {}

        for judge in self.judges:
            result = judge.evaluate(
                goal_description=goal_description,
                turns_text=turns_text,
                inform_slots=inform_slots,
                request_slots=request_slots,
                booking=booking,
            )
            judge_results.append(result)
            judge_reasonings[judge.name] = result.reasoning

        # Phase 2: Combiner synthesizes
        judge_dicts = [
            {
                "judge_name": jr.judge_name,
                "focus_dimension": jr.focus_dimension,
                "scores": jr.scores,
                "reasoning": jr.reasoning,
            }
            for jr in judge_results
        ]

        final = self.combiner.combine(
            goal_description=goal_description,
            turns_text=turns_text,
            inform_slots=inform_slots,
            request_slots=request_slots,
            booking=booking,
            judge_results=judge_dicts,
        )

        final["judge_reasonings"] = judge_reasonings
        final["individual_scores"] = {
            jr.judge_name: jr.scores for jr in judge_results
        }

        return final

    def evaluate_batch(
        self,
        dialogues: list[dict],
        delay: float = 0.3,
    ) -> list[dict]:
        """Evaluate a batch of dialogues.

        Each dialogue dict must have keys: ``goal_description``,
        ``turns_text``, ``inform_slots``, ``request_slots``, ``booking``.

        Args:
            dialogues: List of dialogue dicts.
            delay: Seconds between evaluations (rate limiting).

        Returns:
            List of result dicts with ``scores``, ``reasoning``, etc.
        """
        results: list[dict] = []
        total = len(dialogues)

        for i, item in enumerate(dialogues):
            print(f"  LLM Judge evaluating {i+1}/{total}...")

            try:
                result = self.evaluate_single(
                    goal_description=item.get("goal_description", ""),
                    turns_text=item.get("turns_text", ""),
                    inform_slots=item.get("inform_slots", {}),
                    request_slots=item.get("request_slots", {}),
                    booking=item.get("booking", {}),
                )
                results.append({
                    "dialogue_id": item.get("dialogue_id", ""),
                    "scores": result.get("scores", {}),
                    "reasoning": result.get("reasoning", ""),
                    "judge_agreement": result.get("judge_agreement", ""),
                    "individual_scores": result.get("individual_scores", {}),
                    "judge_reasonings": result.get("judge_reasonings", {}),
                })
            except Exception as exc:
                print(f"    ERROR: {exc}")
                results.append({
                    "dialogue_id": item.get("dialogue_id", ""),
                    "scores": {},
                    "reasoning": f"Error: {exc}",
                    "error": str(exc),
                })

            if i < total - 1:
                time.sleep(delay)

        return results
