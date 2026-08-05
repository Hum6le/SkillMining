"""ABCD action/slot schema normalization.

ABCD action targets contain an action name and an ordered list of slot values.
They do not contain slot names. This module keeps that contract explicit at
the boundary between LLM text and ABCD evaluation.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_SCHEMA_TOKEN_RE = re.compile(
    r"^(?:<[^>]+>|\[[^\]]+\]|"
    r"(?:zip[_ -]?code|postal[_ -]?code|phone(?:[_ -]?number)?|"
    r"email(?:[_ -]?address)?|e[-_ ]?mail|name|full[_ -]?name|"
    r"account[_ -]?id|order[_ -]?id|username|address|amount|"
    r"member[_ -]?level|payment[_ -]?method|reason))$",
    re.IGNORECASE,
)

def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"[^a-z0-9:-]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-")

def canonical_action_name(value: Any, vocabulary: set[str] | None = None) -> tuple[str, list[str]]:
    """Return canonical action and any legacy action suffix slot."""
    raw = str(value or "").strip().lower()
    raw_base, raw_suffix = (raw.split(":", 1) + [""])[:2] if ":" in raw else (raw, "")
    action = _slug(raw_base)
    if action in {"", "none", "null", "no-action", "no_action"}:
        return "", []
    suffix_slots: list[str] = []
    if raw_suffix:
        if action and raw_suffix and (vocabulary is None or action in vocabulary):
            suffix_slots.append(raw_suffix.strip())
    aliases = {
        "pull-up-account-information": "pull-up-account",
        "verify-identity-information": "verify-identity",
    }
    return aliases.get(action, action), suffix_slots

def _guideline_actions(guidelines: dict[str, Any]) -> set[str]:
    actions: set[str] = set()
    for product in guidelines.values():
        for subflow in (product or {}).get("subflows", {}).values():
            for item in (subflow or {}).get("actions", []):
                button = item.get("button") if isinstance(item, dict) else None
                if button and str(button).strip().lower() != "n/a":
                    actions.add(_slug(button))
    return actions

@lru_cache(maxsize=4)
def load_action_schema(data_dir: str | None = None) -> dict[str, Any]:
    """Load official action vocabulary and observed train slot counts."""
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "eval" / "abcd" / "data"
    vocab: set[str] = set()
    guideline_path = root / "guidelines.json"
    if guideline_path.exists():
        vocab |= _guideline_actions(json.loads(guideline_path.read_text(encoding="utf-8")))
    counts: dict[str, set[int]] = {}
    data_path = root / "abcd_v1.1.json"
    if data_path.exists():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        for item in data.get("train", []):
            for turn in item.get("delexed", []):
                targets = turn.get("targets", [])
                if len(targets) >= 4 and targets[1] == "take_action" and targets[2]:
                    action = _slug(targets[2])
                    slots = targets[3] if isinstance(targets[3], list) else []
                    vocab.add(action)
                    counts.setdefault(action, set()).add(len(slots))
    return {"actions": vocab, "slot_counts": counts}

def action_contract_prompt(schema: dict[str, Any]) -> str:
    actions = ", ".join(sorted(schema.get("actions", set())))
    return (
        "## ABCD Action/Slot Contract\n"
        "Predict one canonical ABCD action name. Allowed action vocabulary:\n"
        f"{actions}\n"
        "Slots are an ordered list of REAL VALUES in the exact action order. "
        "They are not slot names or schema labels. Never output zip_code, "
        "phone_number, email_address, name, placeholders such as <zip_code> "
        "or [phone], or key=value strings. Use only values grounded in the "
        "current dialogue/scenario; do not copy private values from examples. "
        "If no action is needed, use action=\"\" and slots=[]."
    )

def clean_slot_values(slots: Any) -> tuple[list[str], list[str]]:
    if not isinstance(slots, list):
        slots = [] if slots is None else [slots]
    accepted: list[str] = []
    rejected: list[str] = []
    for value in slots:
        text = str(value).strip()
        if not text or text.lower() in {"none", "null"}:
            continue
        normalized = text.lower().replace(" ", "_")
        if _SCHEMA_TOKEN_RE.match(text) or "=" in text or normalized.endswith("_name"):
            rejected.append(text)
        else:
            accepted.append(text)
    return accepted, rejected

def canonicalize_prediction(
    action: Any, slots: Any, schema: dict[str, Any] | None = None,
) -> tuple[str, list[str], dict[str, Any]]:
    schema = schema or {"actions": set(), "slot_counts": {}}
    canonical, suffix_slots = canonical_action_name(action, schema.get("actions"))
    raw_slots = list(slots) if isinstance(slots, list) else ([] if slots is None else [slots])
    cleaned, rejected = clean_slot_values(raw_slots)
    if suffix_slots:
        cleaned = suffix_slots + cleaned
    allowed_counts = sorted(schema.get("slot_counts", {}).get(canonical, set()))
    count_ok = not allowed_counts or len(cleaned) in allowed_counts
    action_ok = not canonical or canonical in schema.get("actions", set())
    return canonical, cleaned, {
        "raw_action": str(action or ""),
        "raw_slots": [str(x) for x in raw_slots],
        "canonical_action": canonical,
        "rejected_slots": rejected,
        "slot_count": len(cleaned),
        "allowed_slot_counts": allowed_counts,
        "slot_count_valid": count_ok,
        "action_valid": action_ok,
        "valid": action_ok and count_ok and not rejected,
    }
