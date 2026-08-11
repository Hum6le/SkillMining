"""Stage-4 skill contract definition for offline SKILL-DISCO clusters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Callable
import warnings

from .consolidation import SkillCluster
from .operation_extraction import SemanticOperation


@dataclass(frozen=True)
class SkillParameter:
    name: str
    type: str
    description: str
    required: bool
    default: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillContract:
    """A typed, evidence-grounded contract before any code synthesis."""

    cluster_id: str
    skill_name: str
    description: str
    docstring: str
    parameters: list[SkillParameter]
    return_type: str
    preconditions: list[str]
    postconditions: list[str]
    side_effects: list[str]
    canonical_action_sequence: list[str]
    abstraction_level: str
    estimated_actions_saved: int
    confidence_score: float
    supporting_conversations: list[str]
    source_operation_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }


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


def _string_list(value: Any) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def _supported_parameters(operations: list[SemanticOperation]) -> set[str]:
    return {parameter for operation in operations for parameter in operation.parameters}


def _derived_action_savings(operations: list[SemanticOperation]) -> int:
    lengths = [len(operation.action_sequence) for operation in operations]
    return max(round(sum(lengths) / max(len(lengths), 1)) - 1, 1)


def build_skill_specification_prompt(cluster: SkillCluster, operations: list[SemanticOperation]) -> str:
    """Adapt Appendix B.4's contract prompt to ABCD's offline action adapter."""
    examples = [
        {
            "operation_id": operation.operation_id,
            "description": operation.description,
            "preconditions": operation.preconditions,
            "postconditions": operation.postconditions,
            "action_sequence": operation.action_sequence,
            "observed_action_slots": operation.grounded_actions,
            "code_snippet": operation.code_snippet,
        }
        for operation in operations[:5]
    ]
    allowed_actions = sorted({action for operation in operations for action in operation.action_sequence})
    return """You design reusable procedural skill APIs for task-oriented dialogue agents.
Define one typed skill contract from the successful operation cluster below. This contract will later
be compiled, so it must be concrete and evidence-grounded rather than a generic policy description.

Design principles: (1) generalize using parameters, never concrete values; (2) single responsibility;
(3) action selection, not construction: use only exact supported action templates; (4) declare every
backend state change in side_effects and canonical_action_sequence; (5) state preconditions explicitly;
(6) derive only from successful traces and prefer the shortest supported sequence.

The ``observed_action_slots`` records contain the ordered, concrete gold arguments used by successful
training actions and the dialogue evidence available before each action. Use each value-evidence pair to
learn parameter order, semantic role, and slot-filling behavior. State how every action parameter should
be obtained from current dialogue/state evidence in the existing skill description or docstring. Generalize
the observed values into a reusable procedure; never copy a concrete training value into the skill output.

Return exactly one JSON object and no Markdown:
{"skill_name": "snake_case", "description": "...", "docstring": "...",
 "parameters": [{"name": "...", "type": "str|list[str]|bool|int", "description": "...", "required": true, "default": null}],
 "return_type": "SkillResult", "preconditions": ["..."], "postconditions": ["..."], "side_effects": ["..."],
 "canonical_action_sequence": ["exact action template"], "abstraction_level": "primitive|composite|workflow"}

Supported parameter names:
""" + json.dumps(sorted(_supported_parameters(operations))) + "\n\nSupported action templates:\n" + json.dumps(
        allowed_actions, ensure_ascii=False, indent=2
    ) + "\n\nCluster evidence:\n" + json.dumps({
        "name": cluster.name, "description": cluster.description,
        "num_operations": len(operations), "num_conversations": len(cluster.supporting_conversations),
        "reusability_score": cluster.reusability_score,
        "representative_action_sequence": cluster.representative_action_sequence,
    }, ensure_ascii=False, indent=2) + "\n\nRepresentative successful operations:\n" + json.dumps(examples, ensure_ascii=False, indent=2)


def parse_skill_contract_output(raw_output: str, cluster: SkillCluster, operations: list[SemanticOperation]) -> SkillContract:
    """Validate an LLM contract and locally attach non-negotiable metadata."""
    payload = _json_object(raw_output)
    skill_name = str(payload.get("skill_name", "")).strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", skill_name):
        raise ValueError("skill_name must be snake_case")
    allowed_parameters = _supported_parameters(operations)
    parameters: list[SkillParameter] = []
    for raw_parameter in payload.get("parameters", []):
        if not isinstance(raw_parameter, dict):
            continue
        name = str(raw_parameter.get("name", "")).strip()
        if name not in allowed_parameters or any(parameter.name == name for parameter in parameters):
            continue
        parameter_type = str(raw_parameter.get("type", "str")).strip()
        parameters.append(SkillParameter(
            name=name, type=parameter_type if parameter_type in {"str", "list[str]", "bool", "int"} else "str",
            description=str(raw_parameter.get("description", "")).strip(),
            required=bool(raw_parameter.get("required", True)),
            default=None if raw_parameter.get("default") is None else str(raw_parameter.get("default")),
        ))
    allowed_actions = {action for operation in operations for action in operation.action_sequence}
    canonical = _string_list(payload.get("canonical_action_sequence"))
    if not canonical or any(action not in allowed_actions for action in canonical):
        canonical = list(cluster.representative_action_sequence)
        if not canonical or any(action not in allowed_actions for action in canonical):
            raise ValueError("cluster has no valid representative action sequence")
        warnings.warn(
            "Recovered an invalid LLM skill contract action sequence by using "
            "the cluster's representative successful action sequence.",
            RuntimeWarning,
            stacklevel=2,
        )
    level = str(payload.get("abstraction_level", "composite")).strip()
    if level not in {"primitive", "composite", "workflow"}:
        level = "composite"
    return SkillContract(
        cluster_id=cluster.cluster_id, skill_name=skill_name,
        description=str(payload.get("description", "")).strip(), docstring=str(payload.get("docstring", "")).strip(),
        parameters=parameters, return_type="SkillResult", preconditions=_string_list(payload.get("preconditions")),
        postconditions=_string_list(payload.get("postconditions")), side_effects=_string_list(payload.get("side_effects")),
        canonical_action_sequence=canonical, abstraction_level=level,
        estimated_actions_saved=_derived_action_savings(operations), confidence_score=cluster.reusability_score,
        supporting_conversations=cluster.supporting_conversations, source_operation_ids=cluster.operation_ids,
    )


def specify_skill_contract(cluster: SkillCluster, operations: list[SemanticOperation], chat_fn: Callable[..., str], *, model: str = "deepseek-chat") -> tuple[SkillContract, str]:
    """Run one offline Stage-4 contract-definition call without refinement."""
    prompt = build_skill_specification_prompt(cluster, operations)
    raw_output = chat_fn(prompt, model=model, temperature=0.0)
    if not raw_output.strip():
        raise ValueError("skill specification returned an empty response")
    return parse_skill_contract_output(raw_output, cluster, operations), raw_output
