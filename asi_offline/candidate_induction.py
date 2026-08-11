"""Per-trajectory candidate-program induction for ASIoffline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any

from .abcd_induction import ASIOfflineEpisode


@dataclass(frozen=True)
class ASISkillCandidate:
    """A trace-grounded, callable program candidate from one episode."""

    episode_id: str
    skill_name: str
    description: str
    parameters: list[str]
    action_start_index: int
    action_end_index: int
    primitive_actions: list[dict[str, Any]]
    function_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _episode_action_parameters(episode: ASIOfflineEpisode, start: int, end: int) -> list[str]:
    parameters: list[str] = []
    for action in episode.primitive_actions[start : end + 1]:
        for parameter in action["parameter_names"]:
            if parameter not in parameters:
                parameters.append(parameter)
    return parameters


def _render_function_source(skill_name: str, description: str, parameters: list[str], primitive_actions: list[dict[str, Any]]) -> str:
    signature = ", ".join(f"{parameter}: str" for parameter in parameters)
    lines = [f"def {skill_name}({signature}):", f'    """{description}"""']
    for action in primitive_actions:
        slots = ", ".join(action["parameter_names"])
        slot_expression = f"[{slots}]" if slots else "[]"
        lines.append(f"    take_action({action['action']!r}, {slot_expression})")
    lines.append("    return {'success': True}")
    return "\n".join(lines)


def build_induction_prompt(episode: ASIOfflineEpisode) -> str:
    """Build the label-hidden ASI-style per-trajectory abstraction prompt."""
    action_table = [
        {
            "action_index": action["action_index"],
            "turn_index": action["turn_index"],
            "action": action["parameterized_action"],
            "observation": action["parameterized_observation"],
        }
        for action in episode.primitive_actions
    ]
    instruction_lines = [
        "You induce reusable programmatic skills from one successful task-oriented dialogue trajectory.",
        "Each candidate replaces one contiguous span of three or more backend actions with a callable skill.",
        "Each candidate must have one coherent purpose and only use parameters present in that span.",
        "Do not copy concrete customer values or select a whole trace with unrelated goals.",
        "Do not write arbitrary Python: the runtime rebuilds each body from the selected source span.",
        "Return exactly one JSON object and no Markdown:",
        '{"skills": [{"name": "snake_case", "description": "...", "start_action_index": 0, "end_action_index": 1, "parameters": ["parameter_name"]}]}',
        "",
        "Parameterized dialogue events:",
    ]
    return "\n".join(instruction_lines) + "\n" + json.dumps(
        episode.events, ensure_ascii=False, indent=2
    ) + "\n\nBackend action table:\n" + json.dumps(action_table, ensure_ascii=False, indent=2)


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    if not fenced and not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        candidate = candidate[start : end + 1] if start >= 0 and end > start else ""
    if not candidate:
        raise ValueError("candidate induction output does not contain a JSON object")
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("candidate induction output must be a JSON object")
    return payload


def parse_candidate_output(raw_output: str, episode: ASIOfflineEpisode, *, min_actions: int = 3, max_actions: int = 10) -> tuple[list[ASISkillCandidate], list[dict[str, Any]]]:
    """Validate LLM-selected spans and rebuild their executable DSL bodies."""
    if min_actions < 3 or max_actions < min_actions:
        raise ValueError("invalid action-span bounds")
    payload = _extract_json_object(raw_output)
    records = payload.get("skills")
    if not isinstance(records, list):
        raise ValueError("candidate induction JSON must contain a skills list")
    candidates: list[ASISkillCandidate] = []
    rejected: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_spans: set[tuple[int, int]] = set()
    action_count = len(episode.primitive_actions)
    for record in records:
        if not isinstance(record, dict):
            rejected.append({"reason": "candidate_not_object"})
            continue
        name = str(record.get("name", "")).strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in seen_names:
            rejected.append({"reason": "invalid_or_duplicate_name", "candidate": record})
            continue
        try:
            start = int(record.get("start_action_index"))
            end = int(record.get("end_action_index"))
        except (TypeError, ValueError):
            rejected.append({"reason": "missing_action_span", "candidate": record})
            continue
        span_length = end - start + 1
        if start < 0 or end >= action_count or span_length < min_actions or span_length > max_actions or (start, end) in seen_spans:
            rejected.append({"reason": "invalid_or_duplicate_span", "candidate": record})
            continue
        allowed_parameters = _episode_action_parameters(episode, start, end)
        requested_parameters = record.get("parameters", [])
        if not isinstance(requested_parameters, list):
            rejected.append({"reason": "parameters_not_list", "candidate": record})
            continue
        parameters = [str(item).strip() for item in requested_parameters if str(item).strip()]
        if len(parameters) != len(set(parameters)) or any(parameter not in allowed_parameters for parameter in parameters):
            rejected.append({"reason": "unsupported_parameters", "candidate": record})
            continue
        primitive_actions = [dict(action) for action in episode.primitive_actions[start : end + 1]]
        description = str(record.get("description", "")).strip() or "Reusable backend action sequence."
        candidates.append(ASISkillCandidate(
            episode_id=episode.conversation_id,
            skill_name=name,
            description=description,
            parameters=parameters,
            action_start_index=start,
            action_end_index=end,
            primitive_actions=primitive_actions,
            function_source=_render_function_source(name, description, parameters, primitive_actions),
        ))
        seen_names.add(name)
        seen_spans.add((start, end))
    return candidates, rejected


def rewrite_episode_actions(episode: ASIOfflineEpisode, candidates: list[ASISkillCandidate]) -> list[dict[str, Any]]:
    """Replace non-overlapping candidate spans with explicit skill calls."""
    selected: list[ASISkillCandidate] = []
    next_available = 0
    for candidate in sorted(candidates, key=lambda item: (item.action_start_index, -item.action_end_index)):
        if candidate.episode_id != episode.conversation_id:
            raise ValueError("candidate belongs to a different induction episode")
        if candidate.action_start_index < next_available:
            continue
        selected.append(candidate)
        next_available = candidate.action_end_index + 1
    by_start = {candidate.action_start_index: candidate for candidate in selected}
    rewritten: list[dict[str, Any]] = []
    action_index = 0
    while action_index < len(episode.primitive_actions):
        candidate = by_start.get(action_index)
        if candidate is None:
            action = episode.primitive_actions[action_index]
            rewritten.append({"kind": "primitive_action", "action_index": action_index, "action": action["parameterized_action"]})
            action_index += 1
            continue
        rewritten.append({
            "kind": "skill_call",
            "skill_name": candidate.skill_name,
            "arguments": candidate.parameters,
            "replaces_action_indices": list(range(candidate.action_start_index, candidate.action_end_index + 1)),
        })
        action_index = candidate.action_end_index + 1
    return rewritten
