"""Train-only ABCD trace preparation for an ASIoffline-style library.

This module deliberately prepares one parameterized *episode* at a time.
Unlike SKILL-DISCO, it does not extract operations across the corpus, cluster
them, or consolidate duplicates: original ASI induces candidate programs from
individual successful trajectories.

The input is an ABCD expert trajectory.  The action labels are therefore part
of the induction corpus, and this stage is offline trace supervision rather
than a label-free self-improvement procedure.  The module prevents accidental
construction from ``dev`` or ``test`` by accepting only ``source_split=train``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from skill_disco.abcd_trace import normalize_abcd_conversation


_INDUCTION_SPLIT = "train"
_MIN_INDUCTION_ACTIONS = 3


def _safe_parameter_names(raw_names: list[str]) -> list[str]:
    """Convert trace placeholders to stable identifiers for generated APIs.

    ABCD templates occasionally contain nested placeholders such as
    ``arg_{num_products}``.  They are useful in the source trace, but invalid
    as a Python/DSL parameter name.  Keep their semantic tokens while making
    the resulting skill contract syntactically safe.
    """
    names: list[str] = []
    for position, raw_name in enumerate(raw_names, start=1):
        name = re.sub(r"[^A-Za-z0-9_]", "_", str(raw_name)).strip("_").lower()
        name = re.sub(r"_+", "_", name)
        if not name:
            name = f"arg_{position}"
        if name[0].isdigit():
            name = f"arg_{name}"
        names.append(name)
    return names


@dataclass(frozen=True)
class ASIOfflineEpisode:
    """One prompt-ready successful trace used for per-trajectory induction."""

    conversation_id: str
    source_split: str
    events: list[dict[str, Any]]
    primitive_actions: list[dict[str, Any]]
    parameterized_program: str
    eligible_for_induction: bool
    success_source: str = "abcd_expert_train_trace"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_induction_episode(
    conversation: dict[str, Any],
    *,
    source_split: str = _INDUCTION_SPLIT,
) -> ASIOfflineEpisode:
    """Convert one ABCD training conversation into a single ASI episode.

    The output retains both a grounded successful trace (the original ASI
    induction input) and an instance-independent action IR (for later static
    checks). Scenario/subflow labels are not passed to a later induction
    prompt or deployed skill-selection policy.
    """
    if source_split != _INDUCTION_SPLIT:
        raise ValueError(
            "ASIoffline induction is restricted to the train split; "
            f"got source_split={source_split!r}"
        )

    trace = normalize_abcd_conversation(conversation)
    events = [
        {
            "turn_index": event.turn_index,
            "speaker": event.speaker,
            "event_type": event.event_type,
            "content": event.parameterized_text,
            "grounded_content": event.text,
        }
        for event in trace.events
    ]
    parameterized_event_by_turn = {
        event["turn_index"]: event["content"]
        for event in events
        if event["event_type"] == "backend_action"
    }
    primitive_actions = []
    for action_index, step in enumerate(trace.steps):
        parameter_names = _safe_parameter_names(step.parameter_names)
        primitive_actions.append(
            {
                "action_index": action_index,
                "turn_index": step.turn_index,
                "action": step.action_name,
                "parameter_names": parameter_names,
                "parameterized_action": (
                    f"{step.action_name}({', '.join(parameter_names)})"
                ),
                "slot_values": step.slot_values,
                "pre_context": step.pre_context,
                "observation": step.observation,
                "parameterized_observation": parameterized_event_by_turn[step.turn_index],
            }
        )

    program_lines = [
        "# Parameterized ASIoffline induction episode",
        "context = []",
    ]
    for event in events:
        if event["event_type"] == "backend_action":
            action = next(
                item for item in primitive_actions
                if item["turn_index"] == event["turn_index"]
            )
            program_lines.extend(
                [
                    f"# turn {action['turn_index']}: {event['content']!r}",
                    f"take_action({action['action']!r}, {action['parameter_names']!r})",
                ]
            )
        else:
            program_lines.append(
                f"context.append(({event['event_type']!r}, {event['content']!r}))"
            )

    return ASIOfflineEpisode(
        conversation_id=trace.conversation_id,
        source_split=source_split,
        events=events,
        primitive_actions=primitive_actions,
        parameterized_program="\n".join(program_lines),
        eligible_for_induction=len(primitive_actions) >= _MIN_INDUCTION_ACTIONS,
    )


def build_induction_corpus(
    conversations: list[dict[str, Any]],
    *,
    source_split: str = _INDUCTION_SPLIT,
    min_actions: int = _MIN_INDUCTION_ACTIONS,
) -> list[ASIOfflineEpisode]:
    """Build train-only per-trajectory induction episodes.

    ``min_actions`` is an eligibility threshold only.  It does not alter the
    recorded episode, so later induction remains auditable against the source
    trace.
    """
    if min_actions < 1:
        raise ValueError("min_actions must be positive")

    episodes = [
        build_induction_episode(conversation, source_split=source_split)
        for conversation in conversations
    ]
    return [
        episode
        for episode in episodes
        if len(episode.primitive_actions) >= min_actions
    ]
