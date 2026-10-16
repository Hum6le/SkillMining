"""Identical knowledge-unit text for full exposure and progressive disclosure."""
from __future__ import annotations


def render_action_card(action: str, card: dict[str, str]) -> str:
    blocks = [f"\n#### `{action}`"]
    if card.get("action_rule"):
        blocks.extend(["Action rule:", card["action_rule"]])
    if card.get("slot_policy"):
        blocks.extend(["Slot binding policy:", card["slot_policy"]])
    return "\n".join(blocks)


def render_reference_unit(title: str, body: str) -> str:
    return f"\n### {title}\n{body}"
