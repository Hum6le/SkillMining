"""Offline, turn-level semantic abstraction for normalized ABCD traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Callable

from .abcd_trace import NormalizedABCDTrace


@dataclass(frozen=True)
class SemanticEventAnnotation:
    """Semantic state transition attributed to one normalized dialogue event."""

    turn_index: int
    dialogue_act: str
    intent: str
    state_updates: list[str]
    parameters: list[str]
    control_signal: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _allowed_parameters(trace: NormalizedABCDTrace) -> list[str]:
    names: set[str] = set()
    for event in trace.events:
        names.update(re.findall(r"\{([^{}]+)\}", event.parameterized_text))
    for step in trace.steps:
        names.update(step.parameter_names)
    return sorted(name for name in names if name)


def build_semantic_abstraction_prompt(trace: NormalizedABCDTrace) -> str:
    """Build the Stage-1b prompt without exposing scenario/subflow labels."""
    actions = {step.turn_index: step for step in trace.steps}
    events = []
    for event in trace.events:
        record: dict[str, Any] = {
            "turn_index": event.turn_index,
            "speaker": event.speaker,
            "event_type": event.event_type,
            "content": event.parameterized_text,
        }
        if event.turn_index in actions:
            step = actions[event.turn_index]
            record["backend_action"] = step.parameterized_action
        events.append(record)

    return """You normalize task-oriented dialogue traces into reusable procedural state transitions.
Analyze every event below independently while respecting prior context. The task may contain
different customers and products; use the event content, never any hidden scenario label.

For each event emit:
- dialogue_act: a concise snake_case communicative act, such as request_password_reset,
  provide_account_id, ask_for_identity, confirm, deny, acknowledge, backend_action, or close.
- intent: a concise snake_case local service intent, or "none" for greetings and generic fillers.
- state_updates: zero or more symbolic snake_case facts made true by this event, such as
  reset_requested, account_id_available, identity_requested, identity_verified, or password_generated.
- parameters: only names from the allowed parameter list that this event supplies, requests, or uses.
- control_signal: exactly one of start, advance, branch, confirm, reject, complete, or none.

Do not copy concrete values. Do not output the dataset subflow name. Backend actions must use
dialogue_act="backend_action" and describe their observable state change.
For every parameter used by a backend action, infer its semantic role and whether its value is supplied
by the dialogue or current task state before that action. Preserve parameter order through the action
trace so later skill induction can learn how to fill each slot. Generalize roles and sources; never copy
concrete customer values into the annotations.

Return exactly one JSON object and no Markdown:
{"events": [{"turn_index": 0, "dialogue_act": "...", "intent": "...",
"state_updates": ["..."], "parameters": ["..."], "control_signal": "..."}]}

Allowed parameters:
""" + json.dumps(_allowed_parameters(trace)) + "\n\nEvents:\n" + json.dumps(events, ensure_ascii=False, indent=2)


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    if not fenced and not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        candidate = candidate[start : end + 1] if start >= 0 and end > start else ""
    if not candidate:
        raise ValueError("LLM output does not contain a JSON object")
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON output must be an object")
    return parsed


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def parse_semantic_abstraction_output(
    raw_output: str, trace: NormalizedABCDTrace
) -> list[SemanticEventAnnotation]:
    """Parse and validate a complete per-event LLM annotation response."""
    payload = _extract_json_object(raw_output)
    records = payload.get("events")
    if not isinstance(records, list):
        raise ValueError("LLM JSON must contain an events list")

    expected_turns = {event.turn_index for event in trace.events}
    allowed_parameters = set(_allowed_parameters(trace))
    annotations: dict[int, SemanticEventAnnotation] = {}
    valid_signals = {"start", "advance", "branch", "confirm", "reject", "complete", "none"}
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            turn_index = int(record.get("turn_index"))
        except (TypeError, ValueError):
            continue
        if turn_index not in expected_turns or turn_index in annotations:
            continue
        signal = str(record.get("control_signal", "none")).strip().lower()
        annotations[turn_index] = SemanticEventAnnotation(
            turn_index=turn_index,
            dialogue_act=str(record.get("dialogue_act", "other")).strip() or "other",
            intent=str(record.get("intent", "none")).strip() or "none",
            state_updates=_string_list(record.get("state_updates")),
            parameters=[
                value for value in _string_list(record.get("parameters"))
                if value in allowed_parameters
            ],
            control_signal=signal if signal in valid_signals else "none",
        )

    missing = sorted(expected_turns - annotations.keys())
    if missing:
        raise ValueError(f"LLM output is missing event annotations for turns {missing}")
    return [annotations[event.turn_index] for event in trace.events]


def annotate_trace_semantics(
    trace: NormalizedABCDTrace,
    chat_fn: Callable[..., str],
    *,
    model: str = "deepseek-chat",
) -> tuple[list[SemanticEventAnnotation], str]:
    """Run one offline semantic-abstraction call and return parsed annotations.

    The caller decides how to persist or handle a failure. This function does
    not retry, refine, or access any held-out evaluation trajectory.
    """
    prompt = build_semantic_abstraction_prompt(trace)
    raw_output = chat_fn(prompt, model=model, temperature=0.0)
    if not raw_output.strip():
        raise ValueError("semantic abstraction returned an empty response")
    return parse_semantic_abstraction_output(raw_output, trace), raw_output
