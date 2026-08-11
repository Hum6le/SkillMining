"""Freeze validated ASI functions into a label-free ABCD runtime library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ASILibrary:
    """The original ASI append-only action library in prompt-ready form."""

    functions: list[dict[str, Any]]
    duplicate_names: list[dict[str, Any]]
    rendered_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "functions": self.functions,
            "duplicate_names": self.duplicate_names,
            "rendered_text": self.rendered_text,
        }


def _render_function(record: dict[str, Any]) -> str:
    name = str(record["name"])
    parameters = [str(parameter) for parameter in record.get("parameters", [])]
    signature = ", ".join(parameters)
    lines = [
        f"## Induced Action: {name}({signature})",
        "",
        "Primitive expansion:",
    ]
    for index, action in enumerate(record["action_template"], start=1):
        arguments = ", ".join(str(item) for item in action["arguments"])
        lines.append(f"{index}. take_action({action['action']!r}, [{arguments}])")
    lines.extend([
        "",
        "Runtime rule: bind these arguments from the current dialogue only. "
        "For an individual ABCD target turn, output the next applicable "
        "primitive action rather than this composite function name.",
    ])
    return "\n".join(lines)


def render_asi_library(functions: list[dict[str, Any]]) -> ASILibrary:
    """Apply ASI's first-definition-wins append policy deterministically."""
    accepted: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in functions:
        name = str(record.get("name", "")).strip()
        template = record.get("action_template")
        if not name or not isinstance(template, list):
            raise ValueError("validated ASI function has no name or action template")
        if name in seen:
            duplicates.append({
                "name": name,
                "episode_id": str(record.get("episode_id", "")),
                "reason": "first_definition_already_frozen",
            })
            continue
        seen.add(name)
        accepted.append({
            "episode_id": str(record.get("episode_id", "")),
            "name": name,
            "parameters": [str(parameter) for parameter in record.get("parameters", [])],
            "action_start_index": int(record.get("action_start_index", 0)),
            "action_end_index": int(record.get("action_end_index", -1)),
            "action_template": [
                {
                    "action": str(action["action"]),
                    "arguments": [str(argument) for argument in action.get("arguments", [])],
                }
                for action in template
            ],
        })
    header = """# ASIoffline Programmatic Action Library

The following composite actions were induced once from fixed successful training
trajectories and are now frozen. Infer when a procedure applies from the current
dialogue only. Never copy values from an induction example. ABCD evaluation
requires a primitive action and its ordered real slot values at each target turn;
use these functions to plan that primitive prediction, not as an output action.
""".strip()
    rendered = [_render_function(record) for record in accepted]
    return ASILibrary(
        functions=accepted,
        duplicate_names=duplicates,
        rendered_text=header + ("\n\n" + "\n\n".join(rendered) if rendered else "\n"),
    )
