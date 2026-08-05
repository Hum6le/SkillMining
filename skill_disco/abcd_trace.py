"""Normalize ABCD expert dialogues into structural action traces.

This is the deterministic foundation for the trace-normalization stage of
SKILL-DISCO.  It retains the observed backend-action sequence and parameter
bindings while replacing instance values in the reusable program view with
stable parameter names.  Branch and loop recovery is intentionally deferred
to the later LLM-assisted normalization pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


def _normalize_value(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _original_turn_text(conversation: dict[str, Any], turn_index: int, fallback: dict[str, Any]) -> str:
    original = conversation.get("original") or []
    if 0 <= turn_index < len(original):
        raw_turn = original[turn_index]
        if isinstance(raw_turn, (list, tuple)) and len(raw_turn) >= 2:
            return str(raw_turn[1]).strip()
        if isinstance(raw_turn, dict):
            return str(raw_turn.get("text", "")).strip()
    return str(fallback.get("text", "")).strip()


def _scenario_bindings(scenario: dict[str, Any]) -> dict[str, str]:
    """Map normalized scenario values to stable, human-readable parameters."""
    bindings: dict[str, str] = {}
    for section in ("personal", "order", "product"):
        values = scenario.get(section, {})
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            parameter = str(key).strip().lower().replace(" ", "_")
            items = value if isinstance(value, list) else [value]
            for item in items:
                normalized = _normalize_value(item)
                if normalized and normalized not in bindings:
                    bindings[normalized] = parameter
    return bindings


def _parameterize_text(text: str, bindings: dict[str, str]) -> str:
    """Replace grounded entities with their stable parameter placeholders."""
    result = str(text)
    for value, parameter in sorted(bindings.items(), key=lambda item: -len(item[0])):
        if not value:
            continue
        result = re.sub(
            re.escape(value),
            "{" + parameter + "}",
            result,
            flags=re.IGNORECASE,
        )
    # Dialogue turns can contain customer-provided identifiers that are not
    # present in the scenario metadata or in any annotated backend slot.
    # Preserve their role without allowing concrete identifiers to fragment
    # the structural representation.
    result = re.sub(
        r"(\b(?:account|acount)\s*(?:id)?\s*(?:is|:)\s*)([A-Za-z0-9_-]+)",
        r"\1{account_id}",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"(\border\s*(?:id)?\s*(?:is|:)\s*)([A-Za-z0-9_-]+)",
        r"\1{order_id}",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"(\b(?:my\s+)?username\s*(?:is|:)\s*)([A-Za-z0-9_-]+)",
        r"\1{username}",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"(\b(?:new\s+)?password\s*(?:is|:)\s*)([A-Za-z0-9_-]+)",
        r"\1{password}",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "{email}",
        result,
        flags=re.IGNORECASE,
    )
    return result


def _add_action_slot_bindings(
    delexed: list[dict[str, Any]], bindings: dict[str, str]
) -> None:
    """Name values only seen in action slots so user turns can refer to them."""
    for turn in delexed:
        targets = turn.get("targets", [])
        if len(targets) < 4 or targets[1] != "take_action":
            continue
        values = targets[3] if isinstance(targets[3], list) else []
        for position, value in enumerate(values, start=1):
            normalized = _normalize_value(value)
            if normalized and normalized not in bindings:
                bindings[normalized] = f"arg_{position}"


@dataclass(frozen=True)
class NormalizedDialogueEvent:
    """One observable dialogue event in its raw and parameterized forms."""

    turn_index: int
    speaker: str
    event_type: str
    text: str
    parameterized_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedActionStep:
    """One observed ABCD backend action in reusable and grounded forms."""

    turn_index: int
    action_name: str
    slot_values: list[str]
    parameter_names: list[str]
    parameterized_action: str
    pre_context: list[str]
    observation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedABCDTrace:
    """A normalized action trace suitable for offline skill distillation."""

    conversation_id: str
    flow: str
    subflow: str
    events: list[NormalizedDialogueEvent] = field(default_factory=list)
    steps: list[NormalizedActionStep] = field(default_factory=list)

    @property
    def action_count(self) -> int:
        return len(self.steps)

    def to_program(self) -> str:
        """Render an executable-style, instance-independent IR for auditing."""
        lines = [
            f"# ABCD normalized trace: {self.conversation_id}",
            "context = []",
        ]
        actions_by_turn = {step.turn_index: step for step in self.steps}
        for event in self.events:
            if event.event_type == "backend_action":
                step = actions_by_turn[event.turn_index]
                lines.append(f"# turn {event.turn_index}; observed: {event.parameterized_text!r}")
                lines.append(f"action = {step.action_name!r}")
                lines.append(f"slots = {step.parameter_names!r}")
                lines.append("context.append((action, slots))")
            else:
                lines.append(
                    f"context.append(({event.event_type!r}, {event.parameterized_text!r}))"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "flow": self.flow,
            "subflow": self.subflow,
            "event_count": len(self.events),
            "action_count": self.action_count,
            "events": [event.to_dict() for event in self.events],
            "steps": [step.to_dict() for step in self.steps],
            "program": self.to_program(),
        }


def normalized_trace_from_dict(data: dict[str, Any]) -> NormalizedABCDTrace:
    """Reconstruct a normalized trace loaded from a JSON artifact."""
    return NormalizedABCDTrace(
        conversation_id=str(data.get("conversation_id", "")),
        flow=str(data.get("flow", "unknown")),
        subflow=str(data.get("subflow", "unknown")),
        events=[
            NormalizedDialogueEvent(
                turn_index=int(event["turn_index"]),
                speaker=str(event["speaker"]),
                event_type=str(event["event_type"]),
                text=str(event["text"]),
                parameterized_text=str(event["parameterized_text"]),
            )
            for event in data.get("events", [])
        ],
        steps=[
            NormalizedActionStep(
                turn_index=int(step["turn_index"]),
                action_name=str(step["action_name"]),
                slot_values=[str(value) for value in step.get("slot_values", [])],
                parameter_names=[str(value) for value in step.get("parameter_names", [])],
                parameterized_action=str(step["parameterized_action"]),
                pre_context=[str(value) for value in step.get("pre_context", [])],
                observation=str(step.get("observation", "")),
            )
            for step in data.get("steps", [])
        ],
    )


def normalize_abcd_conversation(conversation: dict[str, Any]) -> NormalizedABCDTrace:
    """Convert one ABCD dialogue into a parameterized backend-action trace.

    Only ``take_action`` turns become primitive operations.  The context of a
    step consists solely of earlier turns, so artifacts can later support
    prefix-only offline replay without exposing future targets.
    """
    scenario = conversation.get("scenario", {})
    bindings = _scenario_bindings(scenario)
    delexed = conversation.get("delexed", [])
    _add_action_slot_bindings(delexed, bindings)
    events: list[NormalizedDialogueEvent] = []
    steps: list[NormalizedActionStep] = []

    for turn_index, turn in enumerate(delexed):
        speaker = str(turn.get("speaker", "unknown"))
        text = _original_turn_text(conversation, turn_index, turn)
        targets = turn.get("targets", [])
        is_action = len(targets) >= 4 and targets[1] == "take_action"
        event_type = (
            "backend_action" if is_action
            else "customer_observation" if speaker == "customer"
            else "agent_response" if speaker == "agent"
            else "other"
        )
        if text:
            events.append(
                NormalizedDialogueEvent(
                    turn_index=turn_index,
                    speaker=speaker,
                    event_type=event_type,
                    text=text,
                    parameterized_text=_parameterize_text(text, bindings),
                )
            )
        if not is_action:
            continue

        action_name = str(targets[2] or "").strip()
        if not action_name:
            continue
        raw_slots = targets[3] if isinstance(targets[3], list) else []
        slot_values = [str(value) for value in raw_slots]
        parameter_names = [bindings[_normalize_value(value)] for value in slot_values]
        parameterized_action = f"{action_name}({', '.join(parameter_names)})"

        pre_context = [
            _original_turn_text(conversation, index, previous)
            for index, previous in enumerate(delexed[:turn_index])
            if _original_turn_text(conversation, index, previous)
        ]
        observation = _original_turn_text(conversation, turn_index, turn)
        steps.append(
            NormalizedActionStep(
                turn_index=turn_index,
                action_name=action_name,
                slot_values=slot_values,
                parameter_names=parameter_names,
                parameterized_action=parameterized_action,
                pre_context=pre_context,
                observation=observation,
            )
        )

    return NormalizedABCDTrace(
        conversation_id=str(conversation.get("convo_id", "")),
        flow=str(scenario.get("flow", "unknown")),
        subflow=str(scenario.get("subflow", "unknown")),
        events=events,
        steps=steps,
    )
