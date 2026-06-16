"""Multi-agent LLM Judge subpackage for ToD evaluation.

Provides a multi-agent judge system that evaluates dialogue quality
through multiple specialist perspectives and synthesizes their scores.

Adapted from ``D:\\paper\\LLMasaJudge``.
"""

from .judge_system import MultiAgentJudge
from .base import JudgeAgent, JudgeResult, create_judges_from_config
from .combiner import Combiner
from .config import (
    SCORING_DIMENSIONS,
    JUDGE_DEFINITIONS,
    COMBINER_DEFINITION,
    LLM_CONFIG,
)
from .prompts import (
    format_dialogue_for_judge,
    build_judge_user_message,
    build_combiner_user_message,
)

__all__ = [
    # Main orchestrator
    "MultiAgentJudge",
    # Components
    "JudgeAgent",
    "JudgeResult",
    "Combiner",
    "create_judges_from_config",
    # Config
    "SCORING_DIMENSIONS",
    "JUDGE_DEFINITIONS",
    "COMBINER_DEFINITION",
    "LLM_CONFIG",
    # Utilities
    "format_dialogue_for_judge",
    "build_judge_user_message",
    "build_combiner_user_message",
]
