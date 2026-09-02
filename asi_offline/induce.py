"""Original-style, per-trajectory action induction for ASIoffline.

The official ASI implementation calls ``induce_actions.py`` after each
successful trajectory.  This module preserves that granularity for a frozen
ABCD induction corpus.  It records the model's complete response verbatim;
offline structural validation and library construction are intentionally
separate stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .abcd_induction import ASIOfflineEpisode


_SYSTEM_PROMPT = """You are a proficient software engineer inducing reusable
programmatic skills from one successful task-oriented dialogue trajectory.
Your task is to (1) summarize a reusable multi-action procedure as an API and
(2) rewrite the same trajectory using the API you generated.

The dialogue context explains why each action is appropriate, but it is not a
literal program specification. Infer a procedure only when the action order
and the dialogue evidence support it. Preserve the source action order and do
not invent actions, tools, preconditions, or outcomes.

The downstream ABCD objective is joint AST, not action accuracy alone. For
every action turn, AST is correct only when both the canonical primitive action
name and the complete ordered slot-value list are correct. Therefore infer
slot binding together with the action sequence: preserve which dialogue value
fills each positional argument and preserve argument order. Do not trade
correct slot binding for a more abstract action description.

Each generated function must contain 3 to 10 take_action calls. A function
may call only take_action(action_name, args), using canonical action names from
the example. Replace concrete customer values with meaningful parameters.
Slot parameters are placeholders for values that must be bound from the
current dialogue at runtime; they are never literal output values. Never
hard-code names, emails, phones, account IDs, order IDs, or other instance
values in a function body. Do not confuse a slot name with a slot value, and
preserve the ordered slot contract of every primitive action.

The generated function is a reusable plan, not the action emitted at runtime.
The runtime must identify the next applicable primitive step, bind its ordered
slot values from the current dialogue, and output that primitive action only.
Include Args, Returns, and Examples in each function docstring, while keeping
examples illustrative and free of private training values.

After the function definitions, provide ## Rewritten Trajectory followed by a
Python code block. The rewritten trajectory must preserve the source action
order and call at least one generated function. You may generate zero, one, or
multiple functions when justified by the trajectory."""


@dataclass(frozen=True)
class ASIInductionArtifact:
    """Auditable result of one original-style ASI induction call."""

    episode_id: str
    action_count: int
    messages: list[dict[str, str]]
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def episode_from_dict(payload: dict[str, Any]) -> ASIOfflineEpisode:
    """Rehydrate an episode written by the train-only preparation stage."""
    required = {
        "conversation_id",
        "source_split",
        "events",
        "primitive_actions",
        "parameterized_program",
        "eligible_for_induction",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"induction episode is missing fields: {missing}")
    return ASIOfflineEpisode(
        conversation_id=str(payload["conversation_id"]),
        source_split=str(payload["source_split"]),
        events=list(payload["events"]),
        primitive_actions=list(payload["primitive_actions"]),
        parameterized_program=str(payload["parameterized_program"]),
        eligible_for_induction=bool(payload["eligible_for_induction"]),
        success_source=str(payload.get("success_source", "abcd_expert_train_trace")),
    )


def build_episode_induction_messages(episode: ASIOfflineEpisode) -> list[dict[str, str]]:
    """Render the prompt shape used by ASI's ``induce_actions.py``.

    The original prompt contains one successful example and ends at
    ``## Reusable Functions``. ABCD backend labels replace browser actions;
    scenario/subflow metadata are absent. Concrete training values remain in
    the grounded trace so the model can infer function arguments, but the
    generated function body must parameterize them; validation rejects any
    hard-coded instance value.
    """
    if not episode.eligible_for_induction:
        raise ValueError(
            f"episode {episode.conversation_id!r} has fewer than 3 primitive actions"
        )

    trajectory = [
        f"### Example 1 ({episode.conversation_id})",
        "Grounded dialogue context (evidence, not a slot-value template):",
    ]
    for event in episode.events:
        if event["event_type"] != "backend_action":
            trajectory.append(
                f"- {event['speaker']}: "
                f"{event.get('grounded_content', event['content'])}"
            )
    trajectory.extend([
        "",
        "Successful backend-action trajectory (preserve this order):",
    ])
    for action in episode.primitive_actions:
        arguments = repr(action["slot_values"])
        trajectory.extend([
            (
                f"# action {action['action_index']}; observation: "
                f"{action['observation']}"
            ),
            f"take_action({action['action']!r}, {arguments})",
        ])
    trajectory.extend(["", "## Reusable Functions"])

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(trajectory)},
    ]


def induce_episode(
    episode: ASIOfflineEpisode,
    chat_fn: Callable[..., str],
    *,
    temperature: float = 1.0,
) -> ASIInductionArtifact:
    """Run exactly one ASI induction request over a fixed successful trace."""
    messages = build_episode_induction_messages(episode)
    raw_response = chat_fn(messages, temperature=temperature)
    return ASIInductionArtifact(
        episode_id=episode.conversation_id,
        action_count=len(episode.primitive_actions),
        messages=messages,
        raw_response=raw_response or "",
    )
