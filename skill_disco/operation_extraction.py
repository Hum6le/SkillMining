"""Stage-2 subgoal-level operation extraction for offline SKILL-DISCO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Callable

from .abcd_trace import NormalizedABCDTrace
from .semantic_abstraction import SemanticEventAnnotation


@dataclass(frozen=True)
class SemanticOperation:
    """A contiguous, multi-action subgoal grounded in one expert trace."""

    operation_id: str
    conversation_id: str
    name: str
    description: str
    action_start_index: int
    action_end_index: int
    action_turn_indices: list[int]
    action_sequence: list[str]
    preconditions: list[str]
    postconditions: list[str]
    control_flow: str
    parameters: list[str]
    supporting_event_turns: list[int]
    completion_evidence: str
    code_snippet: str
    # Concrete gold action arguments from successful induction trajectories.
    # They are carried only to the Stage-4 induction prompt.
    grounded_actions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def semantic_operation_from_dict(data: dict[str, Any]) -> SemanticOperation:
    """Reconstruct an operation stored in a Stage-2 JSON artifact."""
    return SemanticOperation(
        operation_id=str(data["operation_id"]),
        conversation_id=str(data["conversation_id"]),
        name=str(data["name"]),
        description=str(data.get("description", "")),
        action_start_index=int(data["action_start_index"]),
        action_end_index=int(data["action_end_index"]),
        action_turn_indices=[int(value) for value in data.get("action_turn_indices", [])],
        action_sequence=[str(value) for value in data.get("action_sequence", [])],
        preconditions=[str(value) for value in data.get("preconditions", [])],
        postconditions=[str(value) for value in data.get("postconditions", [])],
        control_flow=str(data.get("control_flow", "fixed_sequence")),
        parameters=[str(value) for value in data.get("parameters", [])],
        supporting_event_turns=[int(value) for value in data.get("supporting_event_turns", [])],
        completion_evidence=str(data.get("completion_evidence", "")),
        code_snippet=str(data.get("code_snippet", "")),
        grounded_actions=[
            dict(value) for value in data.get("grounded_actions", [])
            if isinstance(value, dict)
        ],
    )


def _allowed_parameters(trace: NormalizedABCDTrace) -> set[str]:
    return {
        parameter
        for step in trace.steps
        for parameter in step.parameter_names
        if parameter
    }


def build_operation_extraction_prompt(
    trace: NormalizedABCDTrace,
    annotations: list[SemanticEventAnnotation],
) -> str:
    """Build a no-label prompt for extracting trace-local reusable operations."""
    semantic_by_turn = {annotation.turn_index: annotation for annotation in annotations}
    action_table = [
        {
            "action_index": index,
            "turn_index": step.turn_index,
            "action": step.parameterized_action,
            "observation": re.sub(r"\{[^{}]+\}", "<param>", step.observation),
            "semantic_state": (
                semantic_by_turn[step.turn_index].state_updates
                if step.turn_index in semantic_by_turn
                else []
            ),
        }
        for index, step in enumerate(trace.steps)
    ]
    event_table = [
        {
            "turn_index": annotation.turn_index,
            "dialogue_act": annotation.dialogue_act,
            "intent": annotation.intent,
            "state_updates": annotation.state_updates,
            "control_signal": annotation.control_signal,
        }
        for annotation in annotations
    ]
    return """You extract reusable procedural operations from one successful task-oriented dialogue trace.
An operation is a coherent subgoal made of a contiguous span of backend actions in action_index order.
It may include intervening customer/agent events, whose semantic annotations are provided below.

Extract every high-value operation that:
1. contains at least two backend actions;
2. has one coherent goal and a reusable boundary;
3. ends with observable evidence that the subgoal completed.

Do not emit single actions, arbitrary non-contiguous actions, the whole trace when it contains unrelated
subgoals, concrete user values, scenario names, or any action absent from the action table. Prefer the
largest coherent operation when two candidates are redundant. Use snake_case names.
For every parameterized action in an extracted operation, preserve the ordered parameter roles and infer
how each concrete slot should be obtained from dialogue or task-state evidence available before that
action. Summarize this slot-filling behavior in the operation description, preconditions, or postconditions
without copying concrete customer values.

Return exactly one JSON object and no Markdown:
{"operations": [{"name": "...", "description": "...", "start_action_index": 0,
"end_action_index": 1, "preconditions": ["..."], "postconditions": ["..."],
"control_flow": "fixed_sequence|conditional_branch|loop", "parameters": ["..."],
"supporting_event_turns": [0, 2], "completion_evidence": "...", "succeeded": true}]}

Allowed parameters:
""" + json.dumps(sorted(_allowed_parameters(trace))) + "\n\nBackend action table:\n" + json.dumps(
        action_table, ensure_ascii=False, indent=2
    ) + "\n\nInteraction semantics:\n" + json.dumps(event_table, ensure_ascii=False, indent=2)


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
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def _int_list(value: Any, allowed: set[int]) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            converted = int(item)
        except (TypeError, ValueError):
            continue
        if converted in allowed and converted not in result:
            result.append(converted)
    return result


def _render_code_snippet(
    trace: NormalizedABCDTrace,
    annotations: dict[int, SemanticEventAnnotation],
    selected_steps: list,
    preconditions: list[str],
    supporting_event_turns: list[int],
) -> str:
    """Render a parameterized program view of one validated action span."""
    start_turn = selected_steps[0].turn_index
    end_turn = selected_steps[-1].turn_index
    steps_by_turn = {step.turn_index: step for step in selected_steps}
    events_by_turn = {event.turn_index: event for event in trace.events}
    included_turns = set(supporting_event_turns)
    included_turns.update(
        turn_index for turn_index in events_by_turn if start_turn <= turn_index <= end_turn
    )
    lines = ["# Reusable subgoal operation"]
    for condition in preconditions:
        lines.append(f"# precondition: {condition}")
    for turn_index in sorted(included_turns):
        event = events_by_turn.get(turn_index)
        if event is None:
            continue
        annotation = annotations.get(turn_index)
        semantic_comment = (
            f"  # {annotation.dialogue_act}; {annotation.control_signal}"
            if annotation is not None
            else ""
        )
        if turn_index in steps_by_turn:
            step = steps_by_turn[turn_index]
            lines.append(f"action = {step.action_name!r}{semantic_comment}")
            lines.append(f"slots = {step.parameter_names!r}")
            lines.append("observation = env.step(action, slots)")
        elif event.event_type == "customer_observation":
            lines.append(
                f"observe_customer({event.parameterized_text!r}){semantic_comment}"
            )
        elif event.event_type == "agent_response":
            lines.append(
                f"agent_response({event.parameterized_text!r}){semantic_comment}"
            )
        else:
            lines.append(f"observe({event.parameterized_text!r}){semantic_comment}")
    return "\n".join(lines)


def _grounding_evidence(step: Any, max_lines: int = 8) -> list[str]:
    """Select compact pre-action dialogue evidence for observed slot values.

    This is training-only trace serialization for Stage-4. It does not infer a
    new slot policy or access future turns: every returned line predates the
    action. Exact value matches are preferred; when a value is derived rather
    than spoken verbatim, retain the recent context needed to infer it.
    """
    slot_values = [
        " ".join(str(value).lower().split())
        for value in step.slot_values
        if str(value).strip()
    ]
    relevant = [
        text for text in step.pre_context
        if any(value in " ".join(str(text).lower().split()) for value in slot_values)
    ]
    evidence = relevant if relevant else step.pre_context[-max_lines:]
    return [str(text)[:600] for text in evidence[-max_lines:]]


def parse_operation_extraction_output(
    raw_output: str,
    trace: NormalizedABCDTrace,
    annotations: list[SemanticEventAnnotation],
) -> tuple[list[SemanticOperation], list[dict[str, Any]]]:
    """Validate LLM candidates and reconstruct every action sequence locally."""
    payload = _extract_json_object(raw_output)
    candidates = payload.get("operations")
    if not isinstance(candidates, list):
        raise ValueError("LLM JSON must contain an operations list")

    operations: list[SemanticOperation] = []
    rejected: list[dict[str, Any]] = []
    parameters = _allowed_parameters(trace)
    turn_indices = {event.turn_index for event in trace.events}
    annotations_by_turn = {annotation.turn_index: annotation for annotation in annotations}
    seen_spans: set[tuple[int, int]] = set()

    for candidate in candidates:
        if not isinstance(candidate, dict):
            rejected.append({"reason": "candidate_not_object"})
            continue
        try:
            start = int(candidate.get("start_action_index"))
            end = int(candidate.get("end_action_index"))
        except (TypeError, ValueError):
            rejected.append({"reason": "missing_action_span", "candidate": candidate})
            continue
        if not candidate.get("succeeded", False):
            rejected.append({"reason": "not_succeeded", "candidate": candidate})
            continue
        if start < 0 or end >= len(trace.steps) or end - start + 1 < 2:
            rejected.append({"reason": "invalid_or_single_action_span", "candidate": candidate})
            continue
        if (start, end) in seen_spans:
            rejected.append({"reason": "duplicate_span", "candidate": candidate})
            continue
        name = str(candidate.get("name", "")).strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            rejected.append({"reason": "invalid_name", "candidate": candidate})
            continue
        control_flow = str(candidate.get("control_flow", "fixed_sequence")).strip()
        if control_flow not in {"fixed_sequence", "conditional_branch", "loop"}:
            control_flow = "fixed_sequence"

        selected = trace.steps[start : end + 1]
        seen_spans.add((start, end))
        preconditions = _string_list(candidate.get("preconditions"))
        supporting_event_turns = _int_list(
            candidate.get("supporting_event_turns"), turn_indices
        )
        operations.append(
            SemanticOperation(
                operation_id=f"{trace.conversation_id}:{start}-{end}:{name}",
                conversation_id=trace.conversation_id,
                name=name,
                description=str(candidate.get("description", "")).strip(),
                action_start_index=start,
                action_end_index=end,
                action_turn_indices=[step.turn_index for step in selected],
                action_sequence=[step.parameterized_action for step in selected],
                preconditions=preconditions,
                postconditions=_string_list(candidate.get("postconditions")),
                control_flow=control_flow,
                parameters=[
                    value for value in _string_list(candidate.get("parameters"))
                    if value in parameters
                ],
                supporting_event_turns=supporting_event_turns,
                completion_evidence=str(candidate.get("completion_evidence", "")).strip(),
                code_snippet=_render_code_snippet(
                    trace,
                    annotations_by_turn,
                    selected,
                    preconditions,
                    supporting_event_turns,
                ),
                grounded_actions=[
                    {
                        "action_index": action_index,
                        "action": step.action_name,
                        "slot_values": step.slot_values,
                        "pre_action_evidence": _grounding_evidence(step),
                    }
                    for action_index, step in enumerate(selected, start=1)
                ],
            )
        )
    return operations, rejected


def _build_json_retry_prompt(prompt: str, error: json.JSONDecodeError) -> str:
    """Request one format-only retry after an otherwise unusable model response."""
    return (
        f"{prompt}\n\n"
        "Your previous response was not valid JSON and could not be parsed "
        f"({error.msg}). Repeat the task using the same evidence. Return exactly "
        "one syntactically valid JSON object that follows the requested schema, "
        "with no Markdown or explanatory text."
    )


def extract_trace_operations(
    trace: NormalizedABCDTrace,
    annotations: list[SemanticEventAnnotation],
    chat_fn: Callable[..., str],
    *,
    model: str = "deepseek-chat",
) -> tuple[list[SemanticOperation], list[dict[str, Any]], str]:
    """Run Stage-2 extraction, with one JSON-format retry when necessary."""
    prompt = build_operation_extraction_prompt(trace, annotations)
    raw_output = chat_fn(prompt, model=model, temperature=0.0)
    if not raw_output.strip():
        raise ValueError("operation extraction returned an empty response")
    try:
        operations, rejected = parse_operation_extraction_output(
            raw_output, trace, annotations
        )
    except json.JSONDecodeError as error:
        raw_output = chat_fn(
            _build_json_retry_prompt(prompt, error), model=model, temperature=0.0
        )
        if not raw_output.strip():
            raise ValueError("operation extraction JSON retry returned an empty response")
        operations, rejected = parse_operation_extraction_output(
            raw_output, trace, annotations
        )
    return operations, rejected, raw_output
