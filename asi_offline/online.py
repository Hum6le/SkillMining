"""ABCD online-ASI episode preparation.

The original ASI implementation induces from successful rollouts.  ABCD has
no executable environment, so we use a local success criterion: a rollout can
    contribute any individually correct action turn. A globally imperfect
    conversation can therefore still provide useful evidence, while incorrect
    actions and slots are never persisted into an induced function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .abcd_induction import ASIOfflineEpisode
from .candidate_induction import (
    ASISkillCandidate,
    build_induction_prompt,
    parse_candidate_output,
)


@dataclass(frozen=True)
class ASIOnlineEpisode:
    """One successful ABCD rollout prepared for later ASI induction."""

    conversation_id: str
    source_split: str
    ast_score: float
    action_total: int
    action_correct: int
    events: list[dict[str, Any]]
    primitive_actions: list[dict[str, Any]]
    eligible_for_induction: bool
    eligibility_reason: str
    success_source: str = "abcd_online_rollout"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gold_action_rows(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(conversation.get("delexed", [])):
        targets = turn.get("targets", [])
        if len(targets) < 4 or targets[1] != "take_action" or not targets[2]:
            continue
        slots = targets[3] if isinstance(targets[3], list) else []
        rows.append({
            "turn_index": turn_index,
            "action": str(targets[2]),
            "slot_values": [str(value) for value in slots],
        })
    return rows


def _rollout_action_rows(
    turn_results: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sorted(
        (item for item in turn_results if item.get("target_type") == "action"),
        key=lambda item: int(item.get("turn_index", 0)),
    ):
        action = str(row.get("predicted_action") or "").strip()
        if not action:
            continue
        rows.append({
            "turn_index": int(row.get("turn_index", 0)),
            "action": action,
            "slot_values": [str(value) for value in (row.get("predicted_slots") or [])],
        })
    return rows


def _events_from_conversation(
    conversation: dict[str, Any],
    turn_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_by_turn = {
        int(row["turn_index"]): row
        for row in turn_results
        if isinstance(row, dict) and "turn_index" in row
    }
    original = conversation.get("original") or []
    events: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(conversation.get("delexed", [])):
        raw = original[turn_index] if turn_index < len(original) else None
        if isinstance(raw, dict):
            speaker = str(raw.get("speaker", turn.get("speaker", "unknown")))
            text = str(raw.get("text", ""))
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            speaker, text = str(raw[0]), str(raw[1])
        else:
            speaker, text = str(turn.get("speaker", "unknown")), str(turn.get("text", ""))
        row = result_by_turn.get(turn_index, {})
        event_type = "backend_action" if _is_action_turn(turn) else speaker
        event: dict[str, Any] = {
            "turn_index": turn_index,
            "speaker": speaker,
            "event_type": event_type,
            "content": text.strip(),
        }
        if event_type == "backend_action":
            event["predicted_action"] = str(row.get("predicted_action") or "")
            event["predicted_slots"] = [
                str(value) for value in (row.get("predicted_slots") or [])
            ]
        events.append(event)
    return events


def _is_action_turn(turn: dict[str, Any]) -> bool:
    targets = turn.get("targets", [])
    return len(targets) >= 3 and targets[1] == "take_action" and bool(targets[2])


def build_online_episode(
    conversation: dict[str, Any],
    turn_results: Iterable[dict[str, Any]],
    ast_result: dict[str, Any],
    *,
    source_split: str = "train",
    min_actions: int = 1,
) -> ASIOnlineEpisode:
    """Prepare one rollout and retain one locally successful action.

    Ground truth is used only as an eligibility check.  The induction payload
    contains the rollout values from the selected span, never reconstructed
    gold values.  This prevents a wrong slot outside the span from poisoning
    the induced function and avoids the old whole-dialogue ``AST == 1`` gate.
    """
    if min_actions < 1:
        raise ValueError("min_actions must be positive")
    conversation_id = str(conversation.get("convo_id", "?"))
    rows = [dict(row) for row in turn_results]
    gold = _gold_action_rows(conversation)
    rollout = _rollout_action_rows(rows)
    ast_score = float(ast_result.get("ast_score", 0.0) or 0.0)
    action_total = int(ast_result.get("action_total", len(gold)) or 0)
    action_correct = int(ast_result.get("action_correct", 0) or 0)

    rollout_by_turn = {row["turn_index"]: row for row in rollout}
    flags = []
    for gold_row in gold:
        predicted = rollout_by_turn.get(gold_row["turn_index"])
        flags.append(
            predicted is not None
            and predicted["action"] == gold_row["action"]
            and predicted["slot_values"] == gold_row["slot_values"]
        )

    correct_indices = [index for index, is_correct in enumerate(flags) if is_correct]

    if not gold:
        reason = "no_gold_action_turns"
    elif not rollout:
        reason = "no_predicted_action_turns"
    elif not correct_indices:
        reason = "no_locally_correct_action"
    else:
        reason = "eligible_local_success_action"

    eligible = reason == "eligible_local_success_action"
    if eligible:
        if len(correct_indices) == len(gold):
            # Preserve the original ASI behavior for a genuinely successful
            # rollout: the complete successful trajectory is available for
            # multi-action induction.
            primitive_source = rollout
        else:
            # A partial rollout contributes one verified turn only. Do not
            # stitch non-adjacent correct turns or expose neighboring bad
            # slots to the induction model.
            selected_gold = [gold[correct_indices[0]]]
            primitive_source = [rollout_by_turn[selected_gold[0]["turn_index"]]]
    else:
        primitive_source = rollout
    primitive_actions = [
        {
            "action_index": index,
            "turn_index": row["turn_index"],
            "action": row["action"],
            "slot_values": row["slot_values"],
        }
        for index, row in enumerate(primitive_source)
    ]
    return ASIOnlineEpisode(
        conversation_id=conversation_id,
        source_split=source_split,
        ast_score=ast_score,
        action_total=action_total,
        action_correct=action_correct,
        events=_events_from_conversation(conversation, rows),
        primitive_actions=primitive_actions,
        eligible_for_induction=eligible,
        eligibility_reason=reason,
    )


def build_online_episode_batch(
    conversations: list[dict[str, Any]],
    turn_results: list[dict[str, Any]],
    ast_results: list[dict[str, Any]],
    *,
    source_split: str = "train",
    min_actions: int = 3,
) -> list[ASIOnlineEpisode]:
    """Build auditable online episodes from one rollout batch."""
    by_conversation: dict[str, list[dict[str, Any]]] = {}
    for row in turn_results:
        by_conversation.setdefault(str(row.get("convo_id", "")), []).append(row)
    if len(conversations) != len(ast_results):
        raise ValueError("conversations and ast_results must have equal length")
    return [
        build_online_episode(
            conversation,
            by_conversation.get(str(conversation.get("convo_id", "")), []),
            ast_result,
            source_split=source_split,
            min_actions=min_actions,
        )
        for conversation, ast_result in zip(conversations, ast_results)
    ]


def successful_online_episodes(
    episodes: Iterable[ASIOnlineEpisode],
) -> list[ASIOnlineEpisode]:
    """Return only episodes that passed the strict online induction gate."""
    return [episode for episode in episodes if episode.eligible_for_induction]


def online_episode_to_induction_episode(
    episode: ASIOnlineEpisode,
) -> ASIOfflineEpisode:
    """Convert a successful online rollout to the existing ASI IR.

    ABCD action annotations expose slot values rather than parameter names.
    The adapter therefore assigns positional names (``slot_1``, ``slot_2``,
    ...).  These names are only an induction-time abstraction; their values
    are never persisted in the generated function body.
    """
    if not episode.eligible_for_induction:
        raise ValueError(
            f"episode {episode.conversation_id!r} is not eligible: "
            f"{episode.eligibility_reason}"
        )
    primitive_actions: list[dict[str, Any]] = []
    for action in episode.primitive_actions:
        parameter_names = [
            f"slot_{index}"
            for index in range(1, len(action.get("slot_values", [])) + 1)
        ]
        primitive_actions.append({
            **action,
            "parameter_names": parameter_names,
            "parameterized_action": (
                f"{action['action']}({', '.join(parameter_names)})"
            ),
            "pre_context": "",
            "observation": next(
                (
                    str(event.get("content", ""))
                    for event in episode.events
                    if event.get("turn_index") == action.get("turn_index")
                ),
                "",
            ),
            "parameterized_observation": next(
                (
                    str(event.get("content", ""))
                    for event in episode.events
                    if event.get("turn_index") == action.get("turn_index")
                ),
                "",
            ),
        })
    return ASIOfflineEpisode(
        conversation_id=episode.conversation_id,
        source_split=episode.source_split,
        events=episode.events,
        primitive_actions=primitive_actions,
        parameterized_program="",
        eligible_for_induction=True,
        success_source=episode.success_source,
    )


def build_online_induction_prompt(episode: ASIOnlineEpisode) -> str:
    """Build the ASI candidate prompt for one verified online action."""
    induction_episode = online_episode_to_induction_episode(episode)
    prompt = build_induction_prompt(induction_episode, min_actions=1)
    return (
        prompt
        + "\n\nOnline rollout constraints:\n"
        + "- The optimization target is ABCD joint AST, not action accuracy alone. AST requires the canonical primitive action name and the complete ordered slot-value list to be correct on the same action turn.\n"
        + "- A candidate is useful only if it can improve both action selection and dialogue-grounded slot binding; do not simplify away slot arguments merely to preserve the action sequence.\n"
        + "- This trajectory contains one locally successful action; generalize only the verified action in the action table.\n"
        + "- Ignore every other action or slot value from the source dialogue; never copy unverified behavior into a function.\n"
        + "- Infer one or more reusable contiguous procedures from the rollout; do not invent actions.\n"
        + "- `slot_1`, `slot_2`, etc. denote positional values only. Reuse a parameter only when it refers to the same entity/value in the dialogue.\n"
        + "- Never place a concrete customer value from this rollout into a function body.\n"
    )


def parse_online_candidate_output(
    raw_output: str,
    episode: ASIOnlineEpisode,
    *,
    min_actions: int = 1,
    max_actions: int = 10,
) -> tuple[list[ASISkillCandidate], list[dict[str, Any]]]:
    """Parse and validate candidates generated from an online rollout."""
    induction_episode = online_episode_to_induction_episode(episode)
    return parse_candidate_output(
        raw_output,
        induction_episode,
        min_actions=min_actions,
        max_actions=max_actions,
    )


def induce_online_episode(
    episode: ASIOnlineEpisode,
    chat_fn,
    *,
    temperature: float = 1.0,
) -> tuple[str, list[ASISkillCandidate], list[dict[str, Any]]]:
    """Call the LLM once and return raw output plus validated candidates."""
    prompt = build_online_induction_prompt(episode)
    raw_output = chat_fn(
        [{"role": "user", "content": prompt}],
        temperature=temperature,
    ) or ""
    candidates, rejected = parse_online_candidate_output(raw_output, episode)
    return raw_output, candidates, rejected
