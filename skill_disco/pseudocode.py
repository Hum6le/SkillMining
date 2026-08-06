"""Deterministic rendering of Stage-4 contracts into a prompt-ready skill library."""

from __future__ import annotations

from .skill_specification import SkillContract


def render_skill_pseudocode(contract: SkillContract) -> str:
    """Render one contract without adding new procedural content."""
    signature = ", ".join(
        f"{parameter.name}: {parameter.type}" for parameter in contract.parameters
    )
    lines = [
        f"## Skill: {contract.skill_name}",
        f"Signature: {contract.skill_name}({signature})",
        "",
        "Purpose:",
        contract.description or contract.docstring or "Reusable procedural subgoal.",
        "",
        "Use when:",
    ]
    if contract.preconditions:
        lines.extend(f"- {item}" for item in contract.preconditions)
    else:
        lines.append("- No explicit precondition.")
    lines.extend(["", "Procedure:"])
    lines.extend(
        f"{index}. {action}" for index, action in enumerate(contract.canonical_action_sequence, start=1)
    )
    lines.extend(["", "Expected state:"])
    if contract.postconditions:
        lines.extend(f"- {item}" for item in contract.postconditions)
    else:
        lines.append("- No declared postcondition.")
    lines.extend([
        "",
        "Metadata:",
        f"- abstraction_level: {contract.abstraction_level}",
        f"- estimated_actions_saved: {contract.estimated_actions_saved}",
        f"- confidence_score: {contract.confidence_score:.6f}",
    ])
    return "\n".join(lines)


def render_skill_library(contracts: list[SkillContract]) -> str:
    """Render a stable, prompt-ready library from successful contracts."""
    header = """# ABCD Procedural Skill Library

Use a skill only when its preconditions are supported by the current dialogue.
Bind parameters from the current dialogue; never copy values from examples or another task.
Follow the listed backend actions in order, then return to ordinary dialogue reasoning.
""".strip()
    rendered = [render_skill_pseudocode(contract) for contract in sorted(contracts, key=lambda item: item.skill_name)]
    return header + ("\n\n" + "\n\n".join(rendered) if rendered else "\n")
