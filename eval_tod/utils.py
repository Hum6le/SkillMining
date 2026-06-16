"""Slot normalization, fuzzy matching, and goal extraction helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


# ── Slot value normalization ──────────────────────────────────────

_NUMBER_WORDS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def normalize_slot_value(value: str) -> str:
    """Lowercase, strip whitespace, normalize common variants.

    Handles:
    - Number words -> digits
    - Trailing punctuation in values like "yes." -> "yes"
    - "don't care" and "doesn't care" -> "dontcare"
    """
    if not isinstance(value, str):
        value = str(value)

    v = value.lower().strip().rstrip(".,;:!?")

    if v in ("dont care", "doesn't care", "do not care"):
        return "dontcare"

    if v in _NUMBER_WORDS:
        return _NUMBER_WORDS[v]

    # Collapse multiple spaces
    v = " ".join(v.split())

    return v


def match_value(pred: str, gt: str) -> bool:
    """Check if pred_value matches gt_value.

    Handles pipe-delimited alternatives in gt_value.  A prediction
    matches if it equals *any* one of the alternatives after normalization.
    """
    pred_normalized = normalize_slot_value(pred)

    alternatives = [normalize_slot_value(a) for a in gt.split("|")]

    return pred_normalized in alternatives


# ── Goal extraction helpers ────────────────────────────────────────

# Slots in the goal that are booking sub-attributes (not "inform" slots
# in the traditional sense).  We split these out for separate handling.
_BOOKING_SLOT_PREFIXES = ("book ",)


def is_booking_slot(slot_name: str) -> bool:
    """Return True if slot_name is a booking sub-slot (e.g. 'book stay')."""
    return any(slot_name.startswith(p) for p in _BOOKING_SLOT_PREFIXES)


def extract_inform_slots(goal_inform: Dict[str, Dict[str, str]]) -> List[Tuple[str, str, str]]:
    """Extract non-booking inform slots from goal into flat list.

    Returns:
        List of (domain, slot_name, gt_value) tuples.

    Booking sub-slots (book stay, book day, etc.) are excluded because
    they are evaluated separately via the booking reference check.
    """
    slots: List[Tuple[str, str, str]] = []
    for domain, slot_dict in goal_inform.items():
        for slot_name, slot_value in slot_dict.items():
            if not is_booking_slot(slot_name):
                slots.append((domain, slot_name, slot_value))
    return slots


def extract_booking_domains(goal_inform: Dict[str, Dict[str, str]]) -> Set[str]:
    """Return set of domain names that have at least one booking sub-slot."""
    booking_domains: Set[str] = set()
    for domain, slot_dict in goal_inform.items():
        for slot_name in slot_dict:
            if is_booking_slot(slot_name):
                booking_domains.add(domain)
                break
    return booking_domains


def extract_request_slots(goal_request: Dict[str, Dict[str, str]]) -> List[Tuple[str, str]]:
    """Extract request slots from goal into flat list.

    Empty dict values (``{}``) are skipped because they mean "give me
    all available info" and we cannot enumerate what that entails
    without access to the result database.

    Returns:
        List of (domain, slot_name) tuples.
    """
    slots: List[Tuple[str, str]] = []
    for domain, slot_dict in goal_request.items():
        # Empty dict = user wants "all info" for this domain → skip
        if not slot_dict:
            continue
        for slot_name in slot_dict:
            if slot_name:
                slots.append((domain, slot_name))
    return slots


# ── Prediction extraction helpers ──────────────────────────────────


def get_pred_inform_value(
    prediction_inform: Dict[str, Dict[str, str]],
    domain: str,
    slot_name: str,
) -> str | None:
    """Get the predicted value for an inform slot, or None if missing."""
    domain_slots = prediction_inform.get(domain, {})
    return domain_slots.get(slot_name)


def get_pred_request_slots(
    prediction_request: Dict[str, List[str]],
    domain: str,
) -> Set[str]:
    """Get the set of request slot names predicted for a domain."""
    slots = prediction_request.get(domain, [])
    return {normalize_slot_value(s) for s in slots}


def get_booking_reference(
    prediction_booking: Dict[str, Dict[str, str]],
    domain: str,
) -> str | None:
    """Get the booking reference for a domain, or None if missing."""
    booking = prediction_booking.get(domain, {})
    ref = booking.get("reference", "")
    return ref if ref else None


# ── Dialogue formatting ────────────────────────────────────────────


def format_dialogue_text(turns: List[Dict[str, Any]]) -> str:
    """Format dialogue turns as readable text for LLM judge prompt.

    Args:
        turns: List of turn dicts with 'speaker' and 'utterance' keys.

    Returns:
        Multi-line string of the dialogue.
    """
    lines: List[str] = []
    for turn in turns:
        speaker = turn.get("speaker", "unknown")
        utterance = turn.get("utterance", "")
        lines.append(f"[{speaker}] {utterance}")
    return "\n".join(lines)
