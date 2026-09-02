"""Local replay checks for candidates induced from ABCD online rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .candidate_induction import ASISkillCandidate, rewrite_episode_actions
from .online import ASIOnlineEpisode, online_episode_to_induction_episode


@dataclass(frozen=True)
class ASIOnlineValidationResult:
    """Result of replaying one candidate set against its source rollout."""

    episode_id: str
    accepted_candidates: list[str]
    rejected_candidates: list[dict[str, Any]]
    replay_valid: bool
    replay_errors: list[str]
    rewritten_trajectory: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_expansion(
    candidate: ASISkillCandidate,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand a candidate while checking parameter bindings are consistent."""
    bindings: dict[str, str] = {}
    errors: list[str] = []
    expanded: list[dict[str, Any]] = []
    for action in candidate.primitive_actions:
        values = [str(value) for value in (action.get("slot_values") or [])]
        parameters = [str(value) for value in (action.get("parameter_names") or [])]
        if len(values) != len(parameters):
            errors.append(f"slot_parameter_count_mismatch:{candidate.skill_name}")
            continue
        for parameter, value in zip(parameters, values):
            previous = bindings.setdefault(parameter, value)
            if previous != value:
                errors.append(
                    f"inconsistent_parameter_binding:{candidate.skill_name}:{parameter}"
                )
        expanded.append({"action": str(action["action"]), "slot_values": values})
    return expanded, errors


def validate_online_candidates(
    episode: ASIOnlineEpisode,
    candidates: list[ASISkillCandidate],
) -> ASIOnlineValidationResult:
    """Replay non-overlapping candidates and compare them to the rollout.

    This check validates preservation of the source action sequence. It does
    not claim that a candidate generalizes to unseen conversations; that is a
    separate held-out evaluation stage.
    """
    if not episode.eligible_for_induction:
        return ASIOnlineValidationResult(
            episode_id=episode.conversation_id,
            accepted_candidates=[],
            rejected_candidates=[{
                "reason": "episode_has_no_safe_local_success_span",
                "eligibility_reason": episode.eligibility_reason,
            }],
            replay_valid=False,
            replay_errors=["episode_not_eligible"],
            rewritten_trajectory=[],
        )

    source_episode = online_episode_to_induction_episode(episode)
    rejected: list[dict[str, Any]] = []
    valid_candidates: list[ASISkillCandidate] = []
    for candidate in candidates:
        expanded, errors = _candidate_expansion(candidate)
        if errors:
            rejected.append({"skill_name": candidate.skill_name, "reason": errors})
            continue
        expected = [
            {
                "action": str(item["action"]),
                "slot_values": [str(value) for value in (item.get("slot_values") or [])],
            }
            for item in source_episode.primitive_actions[
                candidate.action_start_index : candidate.action_end_index + 1
            ]
        ]
        if expanded != expected:
            rejected.append({
                "skill_name": candidate.skill_name,
                "reason": "candidate_does_not_expand_to_source_span",
            })
            continue
        valid_candidates.append(candidate)

    rewritten = rewrite_episode_actions(source_episode, valid_candidates)
    used_names = {
        str(item["skill_name"])
        for item in rewritten
        if item.get("kind") == "skill_call"
    }
    for candidate in valid_candidates:
        if candidate.skill_name not in used_names:
            rejected.append({
                "skill_name": candidate.skill_name,
                "reason": "overlapping_candidate_not_selected",
            })
    replayed: list[dict[str, Any]] = []
    for item in rewritten:
        if item["kind"] == "primitive_action":
            action = source_episode.primitive_actions[item["action_index"]]
            replayed.append({
                "action": str(action["action"]),
                "slot_values": [str(value) for value in (action.get("slot_values") or [])],
            })
        else:
            candidate = next(
                item_candidate
                for item_candidate in valid_candidates
                if item_candidate.skill_name == item["skill_name"]
            )
            expanded, _ = _candidate_expansion(candidate)
            replayed.extend(expanded)

    source = [
        {
            "action": str(item["action"]),
            "slot_values": [str(value) for value in (item.get("slot_values") or [])],
        }
        for item in source_episode.primitive_actions
    ]
    replay_errors = [] if replayed == source else [
        "rewritten_trajectory_does_not_match_source_rollout"
    ]
    if not valid_candidates:
        replay_errors.append("no_candidate_passed_local_replay")
    return ASIOnlineValidationResult(
        episode_id=episode.conversation_id,
        accepted_candidates=sorted(used_names),
        rejected_candidates=rejected,
        replay_valid=not replay_errors,
        replay_errors=replay_errors,
        rewritten_trajectory=rewritten,
    )
