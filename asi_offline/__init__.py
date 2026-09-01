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
from .library import ASILibrary, render_asi_library
from .runtime import build_asi_workflow, create_asi_offline_abcd_agent, load_asi_library
from .online import (
    ASIOnlineEpisode,
    build_online_episode,
    build_online_episode_batch,
    build_online_induction_prompt,
    induce_online_episode,
    online_episode_to_induction_episode,
    parse_online_candidate_output,
    successful_online_episodes,
)
from .online_validation import ASIOnlineValidationResult, validate_online_candidates
from .online_library import (
    ASILibraryUpdate,
    ASIOnlineLibraryManager,
    candidate_to_library_record,
)
from .online_evaluation import ASIUpdateDecision, decide_asi_update, evaluate_asi_library

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
    "ASILibrary",
    "render_asi_library",
    "build_asi_workflow",
    "create_asi_offline_abcd_agent",
    "load_asi_library",
    "ASIOnlineEpisode",
    "build_online_episode",
    "build_online_episode_batch",
    "build_online_induction_prompt",
    "induce_online_episode",
    "online_episode_to_induction_episode",
    "parse_online_candidate_output",
    "successful_online_episodes",
    "ASIOnlineValidationResult",
    "validate_online_candidates",
    "ASILibraryUpdate",
    "ASIOnlineLibraryManager",
    "candidate_to_library_record",
    "ASIUpdateDecision",
    "decide_asi_update",
    "evaluate_asi_library",
]
