"""Stage-3 two-pass consolidation of trace-local operations into skill clusters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Callable

from .operation_extraction import SemanticOperation


@dataclass(frozen=True)
class OperationGroup:
    group_id: str
    name: str
    description: str
    operation_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillCluster:
    cluster_id: str
    name: str
    description: str
    group_ids: list[str]
    operation_ids: list[str]
    supporting_conversations: list[str]
    reusability_score: float
    representative_action_sequence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_object(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    if not fenced and not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        candidate = candidate[start : end + 1] if start >= 0 and end > start else ""
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output must be a JSON object")
    return parsed


def _snake_name(value: Any) -> str:
    name = str(value or "").strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"invalid snake_case name: {name!r}")
    return name


def build_grouping_prompt(operations: list[SemanticOperation]) -> str:
    """First-pass prompt: partition one operation batch into reusable groups."""
    records = [
        {
            "operation_id": operation.operation_id,
            "description": operation.description,
            "action_sequence": operation.action_sequence,
            "preconditions": operation.preconditions,
            "postconditions": operation.postconditions,
            "control_flow": operation.control_flow,
        }
        for operation in operations
    ]
    return """You identify reusable procedural operation patterns. Partition every input operation into
non-overlapping groups. Operations belong together only when they have the same parameterized action
structure and subgoal, even when their concrete customer context differs. Use action_sequence and
state transitions as the primary evidence; names and wording are secondary.

Prefer fewer, larger, uniform groups. Do not use scenario labels or concrete values. Every operation_id
must appear exactly once across the output groups.

Return exactly one JSON object and no Markdown:
{"groups": [{"name": "snake_case_name", "description": "...", "operation_ids": ["..."]}]}

Operations:
""" + json.dumps(records, ensure_ascii=False, indent=2)


def parse_grouping_output(raw_output: str, operations: list[SemanticOperation], batch_index: int) -> list[OperationGroup]:
    payload = _json_object(raw_output)
    records = payload.get("groups")
    if not isinstance(records, list) or not records:
        raise ValueError("LLM JSON must contain a non-empty groups list")
    expected = {operation.operation_id for operation in operations}
    assigned: list[str] = []
    groups: list[OperationGroup] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("group must be a JSON object")
        ids = [str(value) for value in record.get("operation_ids", [])]
        if not ids or any(value not in expected for value in ids):
            raise ValueError("group contains missing or unknown operation IDs")
        assigned.extend(ids)
        groups.append(OperationGroup(
            group_id=f"batch{batch_index:04d}_group{index:03d}",
            name=_snake_name(record.get("name")),
            description=str(record.get("description", "")).strip(),
            operation_ids=ids,
        ))
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError("groups must partition every operation exactly once")
    return groups


def group_operation_batch(
    operations: list[SemanticOperation],
    chat_fn: Callable[..., str],
    *,
    batch_index: int,
    model: str = "deepseek-chat",
) -> tuple[list[OperationGroup], str]:
    prompt = build_grouping_prompt(operations)
    raw_output = chat_fn(prompt, model=model, temperature=0.0)
    if not raw_output.strip():
        raise ValueError("operation grouping returned an empty response")
    return parse_grouping_output(raw_output, operations, batch_index), raw_output


def build_consolidation_prompt(groups: list[OperationGroup], operation_by_id: dict[str, SemanticOperation]) -> str:
    """Second-pass prompt: merge batch groups into the minimal global skill set."""
    records = []
    for group in groups:
        examples = [operation_by_id[operation_id].action_sequence for operation_id in group.operation_ids[:3]]
        records.append({
            "group_id": group.group_id,
            "name": group.name,
            "description": group.description,
            "num_operations": len(group.operation_ids),
            "action_examples": examples,
        })
    return """Merge candidate operation groups into a minimal, non-redundant procedural skill set.
Groups belong in one skill cluster when their action structure, induced state change, and reusable goal
are equivalent. Do not merge merely because they are in the same customer-service domain. Every group_id
must appear exactly once across the output clusters.

Return exactly one JSON object and no Markdown:
{"clusters": [{"name": "snake_case_name", "description": "...", "group_ids": ["..."]}]}

Groups:
""" + json.dumps(records, ensure_ascii=False, indent=2)


def parse_consolidation_output(
    raw_output: str,
    groups: list[OperationGroup],
    operation_by_id: dict[str, SemanticOperation],
    total_conversations: int,
) -> list[SkillCluster]:
    payload = _json_object(raw_output)
    records = payload.get("clusters")
    if not isinstance(records, list) or not records:
        raise ValueError("LLM JSON must contain a non-empty clusters list")
    group_by_id = {group.group_id: group for group in groups}
    expected = set(group_by_id)
    assigned: list[str] = []
    clusters: list[SkillCluster] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("cluster must be a JSON object")
        group_ids = [str(value) for value in record.get("group_ids", [])]
        if not group_ids or any(value not in expected for value in group_ids):
            raise ValueError("cluster contains missing or unknown group IDs")
        assigned.extend(group_ids)
        operation_ids = [
            operation_id
            for group_id in group_ids
            for operation_id in group_by_id[group_id].operation_ids
        ]
        conversations = sorted({operation_by_id[operation_id].conversation_id for operation_id in operation_ids})
        representative = operation_by_id[operation_ids[0]].action_sequence
        clusters.append(SkillCluster(
            cluster_id=f"cluster_{index:03d}",
            name=_snake_name(record.get("name")),
            description=str(record.get("description", "")).strip(),
            group_ids=group_ids,
            operation_ids=operation_ids,
            supporting_conversations=conversations,
            reusability_score=round(len(conversations) / max(total_conversations, 1), 6),
            representative_action_sequence=representative,
        ))
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError("clusters must partition every group exactly once")
    return clusters


def consolidate_groups(
    groups: list[OperationGroup],
    operation_by_id: dict[str, SemanticOperation],
    total_conversations: int,
    chat_fn: Callable[..., str],
    *,
    model: str = "deepseek-chat",
) -> tuple[list[SkillCluster], str]:
    prompt = build_consolidation_prompt(groups, operation_by_id)
    raw_output = chat_fn(prompt, model=model, temperature=0.0)
    if not raw_output.strip():
        raise ValueError("group consolidation returned an empty response")
    return parse_consolidation_output(
        raw_output, groups, operation_by_id, total_conversations
    ), raw_output
