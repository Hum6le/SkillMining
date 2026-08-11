"""Static substitute for ASI's environment-based induced-action tests.

The official ASI code appends model-generated functions to the action space and
replays a rewritten trajectory in WebArena.  In the frozen offline ABCD setup
we cannot interact with an environment.  Instead, we accept a candidate only
when its constrained DSL body is trace-grounded and the model's rewritten
trajectory symbolically expands to the exact expert action/argument sequence.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import re
from typing import Any

from .abcd_induction import ASIOfflineEpisode


_REWRITTEN_MARKER = re.compile(r"##\s*Rewritten\s+Trajector(?:y|ies)", re.IGNORECASE)
_PYTHON_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class StaticASIFunction:
    """A generated ASI function that maps to one source action span."""

    episode_id: str
    name: str
    parameters: list[str]
    action_start_index: int
    action_end_index: int
    function_source: str
    action_template: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OfflineValidationResult:
    episode_id: str
    accepted_functions: list[StaticASIFunction]
    rejected_functions: list[dict[str, Any]]
    rewritten_trajectory_valid: bool
    rewritten_trajectory_errors: list[str]
    expanded_actions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["accepted_functions"] = [
            candidate.to_dict() for candidate in self.accepted_functions
        ]
        return payload


def _code_blocks(text: str) -> list[str]:
    return [match.group(1).strip() for match in _PYTHON_BLOCK.finditer(text) if match.group(1).strip()]


def _action_template(function: ast.FunctionDef) -> tuple[list[dict[str, Any]], str | None]:
    """Extract a minimal safe DSL body, rejecting arbitrary generated Python."""
    parameters = [argument.arg for argument in function.args.args]
    template: list[dict[str, Any]] = []
    for statement in function.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue  # docstring or harmless string literal
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)):
            return [], "function_body_contains_non_action_statement"
        call = statement.value
        if not (isinstance(call.func, ast.Name) and call.func.id == "take_action"):
            return [], "function_body_calls_undefined_tool"
        if len(call.args) != 2 or call.keywords:
            return [], "take_action_requires_exactly_two_positional_arguments"
        if not (isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)):
            return [], "action_name_must_be_a_string_literal"
        slot_node = call.args[1]
        if not isinstance(slot_node, (ast.List, ast.Tuple)):
            return [], "action_arguments_must_be_a_list_or_tuple"
        slot_parameters: list[str] = []
        for item in slot_node.elts:
            if not isinstance(item, ast.Name) or item.id not in parameters:
                return [], "function_body_hardcodes_or_derives_a_slot_value"
            slot_parameters.append(item.id)
        template.append({"action": call.args[0].value, "arguments": slot_parameters})
    if not template:
        return [], "function_has_no_take_action_calls"
    return template, None


def _matching_span(
    episode: ASIOfflineEpisode, template: list[dict[str, Any]]
) -> tuple[int, int] | None:
    """Find an unambiguous source span with the generated action shape."""
    matches: list[tuple[int, int]] = []
    size = len(template)
    for start in range(0, len(episode.primitive_actions) - size + 1):
        source = episode.primitive_actions[start : start + size]
        if all(
            generated["action"] == action["action"]
            and len(generated["arguments"]) == len(action["slot_values"])
            for generated, action in zip(template, source)
        ):
            matches.append((start, start + size - 1))
    return matches[0] if len(matches) == 1 else None


def extract_static_functions(
    raw_response: str, episode: ASIOfflineEpisode
) -> tuple[list[StaticASIFunction], list[dict[str, Any]]]:
    """Mirror ``write_actions`` while enforcing a safe trace-grounded DSL."""
    accepted: list[StaticASIFunction] = []
    rejected: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    function_blocks = _code_blocks(raw_response)
    for block_index, block in enumerate(function_blocks):
        try:
            module = ast.parse(block)
        except SyntaxError as exc:
            rejected.append({"block": block_index, "reason": f"invalid_python:{exc.msg}"})
            continue
        for node in module.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not re.fullmatch(r"[a-z][a-z0-9_]*", node.name) or node.name in seen_names:
                rejected.append({"name": node.name, "reason": "invalid_or_duplicate_function_name"})
                continue
            if node.decorator_list or node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
                rejected.append({"name": node.name, "reason": "unsupported_function_signature"})
                continue
            template, error = _action_template(node)
            if error:
                rejected.append({"name": node.name, "reason": error})
                continue
            if not 3 <= len(template) <= 10:
                rejected.append({"name": node.name, "reason": "action_count_outside_original_asi_bounds"})
                continue
            span = _matching_span(episode, template)
            if span is None:
                rejected.append({"name": node.name, "reason": "action_sequence_not_an_unambiguous_source_span"})
                continue
            accepted.append(
                StaticASIFunction(
                    episode_id=episode.conversation_id,
                    name=node.name,
                    parameters=[argument.arg for argument in node.args.args],
                    action_start_index=span[0],
                    action_end_index=span[1],
                    function_source=ast.unparse(node),
                    action_template=template,
                )
            )
            seen_names.add(node.name)
    return accepted, rejected


def _literal_list(node: ast.AST) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[str] = []
    for element in node.elts:
        try:
            value = ast.literal_eval(element)
        except (ValueError, TypeError):
            return None
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            return None
        values.append(str(value))
    return values


def _rewritten_module(raw_response: str) -> ast.Module | None:
    marker = _REWRITTEN_MARKER.search(raw_response)
    if marker is None:
        return None
    blocks = _code_blocks(raw_response[marker.end() :])
    if not blocks:
        return None
    try:
        return ast.parse(blocks[0])
    except SyntaxError:
        return None


def validate_rewritten_trajectory(
    raw_response: str,
    episode: ASIOfflineEpisode,
    functions: list[StaticASIFunction],
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    """Symbolically expand a rewritten trajectory and compare it to the expert."""
    module = _rewritten_module(raw_response)
    if module is None:
        return False, ["missing_or_invalid_rewritten_trajectory_code_block"], []
    by_name = {function.name: function for function in functions}
    expanded: list[dict[str, Any]] = []
    errors: list[str] = []
    used_function = False
    for statement in module.body:
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)):
            errors.append("rewritten_trajectory_contains_non_call_statement")
            continue
        call = statement.value
        if not isinstance(call.func, ast.Name) or call.keywords:
            errors.append("rewritten_trajectory_contains_unsupported_call")
            continue
        if call.func.id == "take_action":
            if len(call.args) != 2 or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
                errors.append("invalid_direct_take_action")
                continue
            arguments = _literal_list(call.args[1])
            if arguments is None:
                errors.append("direct_take_action_arguments_are_not_grounded_literals")
                continue
            expanded.append({"action": call.args[0].value, "slot_values": arguments})
            continue
        function = by_name.get(call.func.id)
        if function is None:
            errors.append(f"rewritten_trajectory_calls_unknown_function:{call.func.id}")
            continue
        if len(call.args) != len(function.parameters):
            errors.append(f"wrong_argument_count_for:{function.name}")
            continue
        bound: dict[str, str] = {}
        for parameter, argument in zip(function.parameters, call.args):
            try:
                value = ast.literal_eval(argument)
            except (ValueError, TypeError):
                errors.append(f"function_call_argument_is_not_grounded_literal:{function.name}")
                break
            if not isinstance(value, (str, int, float, bool)) and value is not None:
                errors.append(f"function_call_argument_is_not_scalar:{function.name}")
                break
            bound[parameter] = str(value)
        else:
            used_function = True
            for action in function.action_template:
                expanded.append({
                    "action": action["action"],
                    "slot_values": [bound[name] for name in action["arguments"]],
                })
    expected = [
        {"action": action["action"], "slot_values": [str(value) for value in action["slot_values"]]}
        for action in episode.primitive_actions
    ]
    if not used_function:
        errors.append("rewritten_trajectory_does_not_use_an_induced_function")
    if expanded != expected:
        errors.append("rewritten_trajectory_does_not_expand_to_the_expert_action_sequence")
    return not errors, errors, expanded


def validate_asi_response(raw_response: str, episode: ASIOfflineEpisode) -> OfflineValidationResult:
    """Perform all offline checks for one original-style ASI response."""
    accepted, rejected = extract_static_functions(raw_response, episode)
    valid, errors, expanded = validate_rewritten_trajectory(raw_response, episode, accepted)
    return OfflineValidationResult(
        episode_id=episode.conversation_id,
        accepted_functions=accepted,
        rejected_functions=rejected,
        rewritten_trajectory_valid=valid,
        rewritten_trajectory_errors=errors,
        expanded_actions=expanded,
    )
