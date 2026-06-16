"""JudgeAgent —— 每位评判官从特定专业视角独立评估对话。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import JUDGE_DEFINITIONS, SCORING_DIMENSIONS
from .prompts import build_judge_prompt, build_judge_user_message
from .llm_client import call_llm_structured


@dataclass
class JudgeResult:
    """Single judge's evaluation result."""

    judge_name: str
    focus_dimension: str
    scores: dict[str, int]  # dimension -> score
    reasoning: str
    raw_response: str = ""


class JudgeAgent:
    """A specialist judge that evaluates dialogue quality from a specific lens.

    Each judge is configured with a **focus dimension** and a **role**
    description.  They receive the dialogue text plus the agent's
    predictions and return scores on all dimensions, with special
    attention to their focus area.
    """

    def __init__(
        self,
        name: str,
        focus: str,
        role: str,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.name = name
        self.focus = focus  # The dimension this judge specializes in
        self.role = role    # Role description injected into system prompt
        self.model = model or "deepseek-chat"
        self.api_key = api_key
        self.base_url = base_url
        self._system_prompt = build_judge_prompt(self.role, self.focus)

    def evaluate(
        self,
        goal_description: str,
        turns_text: str,
        inform_slots: dict,
        request_slots: dict,
        booking: dict,
    ) -> JudgeResult:
        """Evaluate a single dialogue and return scores + reasoning.

        Args:
            goal_description: Natural-language user goal.
            turns_text: Formatted dialogue turns.
            inform_slots: Agent-predicted inform slots.
            request_slots: Agent-predicted request slots.
            booking: Agent-predicted booking info.
        """
        user_message = build_judge_user_message(
            goal_description, turns_text, inform_slots, request_slots, booking
        )

        raw = call_llm_structured(
            system_prompt=self._system_prompt,
            user_message=user_message,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        return JudgeResult(
            judge_name=self.name,
            focus_dimension=self.focus,
            scores=raw.get("scores", {}),
            reasoning=raw.get("reasoning", ""),
            raw_response=str(raw),
        )

    def to_dict(self) -> dict[str, str]:
        """Export judge config for serialization."""
        return {"name": self.name, "focus": self.focus, "role": self.role}


def create_judges_from_config(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[JudgeAgent]:
    """Create all judge agents from JUDGE_DEFINITIONS config."""
    judges: list[JudgeAgent] = []
    for judge_id, config in JUDGE_DEFINITIONS.items():
        judges.append(JudgeAgent(
            name=config["name"],
            focus=config["focus"],
            role=config["role"],
            model=model,
            api_key=api_key,
            base_url=base_url,
        ))
    return judges
