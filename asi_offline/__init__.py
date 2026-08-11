"""Offline ASI-style skill induction utilities for task-oriented dialogue.

The package follows the ``ASIoffline`` protocol reported by Skill-Disco:
build a skill library from a fixed induction split, freeze it, and evaluate it
without any test-time skill updates.
"""

from .abcd_induction import (
    ASIOfflineEpisode,
    build_induction_corpus,
    build_induction_episode,
)
from .candidate_induction import (
    ASISkillCandidate,
    build_induction_prompt,
    parse_candidate_output,
    rewrite_episode_actions,
)
from .induce import (
    ASIInductionArtifact,
    build_episode_induction_messages,
    episode_from_dict,
    induce_episode,
)
from .offline_validation import (
    OfflineValidationResult,
    StaticASIFunction,
    extract_static_functions,
    validate_asi_response,
    validate_rewritten_trajectory,
)

__all__ = [
    "ASIOfflineEpisode",
    "build_induction_corpus",
    "build_induction_episode",
    "ASISkillCandidate",
    "build_induction_prompt",
    "parse_candidate_output",
    "rewrite_episode_actions",
    "ASIInductionArtifact",
    "build_episode_induction_messages",
    "episode_from_dict",
    "induce_episode",
    "OfflineValidationResult",
    "StaticASIFunction",
    "extract_static_functions",
    "validate_asi_response",
    "validate_rewritten_trajectory",
]
