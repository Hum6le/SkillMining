"""ABCD dataset loader.

Loads the Action-Based Conversations Dataset from local JSON files,
extracts ground-truth turn annotations, and provides utterance lookups.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .schemas import ABCDGroundTruth

# ── Default paths ──────────────────────────────────────────────

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "eval", "abcd", "data"
)


def _resolve(path: str | None) -> str:
    """Resolve a path relative to the abcd data directory."""
    if path is None:
        path = _DEFAULT_DATA_DIR
    return os.path.normpath(path)


# ── Main loader ────────────────────────────────────────────────

def load_abcd_data(
    split: str = "test",
    data_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Load ABCD conversations for a given split.

    Args:
        split: One of ``"train"``, ``"dev"``, ``"test"``.
        data_dir: Path to the directory containing ``abcd_v1.1.json``.
            Defaults to ``data/eval/abcd/data/``.

    Returns:
        List of conversation dicts, each with keys:
        ``convo_id``, ``scenario``, ``original``, ``delexed``.
    """
    base = _resolve(data_dir)
    data_path = os.path.join(base, "abcd_v1.1.json")

    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if split not in all_data:
        raise ValueError(
            f"Unknown split '{split}'. Available: {list(all_data.keys())}"
        )
    return all_data[split]


# ── Ground-truth extraction ────────────────────────────────────

def extract_ground_truth(conversation: dict[str, Any]) -> list[ABCDGroundTruth]:
    """Extract per-turn ground truth from one ABCD conversation.

    Args:
        conversation: A single conversation dict from ``load_abcd_data``.

    Returns:
        Ordered list of ``ABCDGroundTruth``, one per turn.
    """
    truths: list[ABCDGroundTruth] = []

    for i, turn in enumerate(conversation["delexed"]):
        targets = turn["targets"]
        # targets = [subflow, action_type, next_action, slot_values, utterance_id]
        action_type = targets[1]
        utterance_id = targets[4]

        # Determine speaker / turn_type
        speaker = turn["speaker"]
        if action_type == "retrieve_utterance":
            turn_type = "utterance"
        elif action_type == "take_action":
            turn_type = "action"
        else:
            turn_type = "customer"

        truths.append(
            ABCDGroundTruth(
                turn_index=i,
                speaker=speaker,
                turn_type=turn_type,
                utterance_id=utterance_id if utterance_id >= 0 else None,
                action_name=targets[2] if turn_type == "action" else None,
                slot_values=list(targets[3]) if turn_type == "action" else None,
                text=turn.get("text", ""),
                candidates=list(turn.get("candidates", [])),
            )
        )

    return truths


# ── Utterance lookup ───────────────────────────────────────────

_UTTERANCE_CACHE: list[str] | None = None


def _load_utterances(data_dir: str | None = None) -> list[str]:
    """Load the utterance pool (cached)."""
    global _UTTERANCE_CACHE
    if _UTTERANCE_CACHE is not None:
        return _UTTERANCE_CACHE

    base = _resolve(data_dir)
    path = os.path.join(base, "utterances.json")
    with open(path, "r", encoding="utf-8") as f:
        _UTTERANCE_CACHE = json.load(f)
    return _UTTERANCE_CACHE


def get_utterance_text(utterance_id: int, data_dir: str | None = None) -> str:
    """Look up the text of an utterance by its pool id.

    Args:
        utterance_id: 0-based index into the utterance pool.
        data_dir: Path to the ABCD data directory.

    Returns:
        The utterance text, or ``"<unknown>"`` if out of range.
    """
    pool = _load_utterances(data_dir)
    if 0 <= utterance_id < len(pool):
        return pool[utterance_id]
    return "<unknown>"


def get_utterance_pool_size(data_dir: str | None = None) -> int:
    """Return the total number of utterances in the pool."""
    return len(_load_utterances(data_dir))


# ── Convenience ────────────────────────────────────────────────

def load_abcd_with_truth(
    split: str = "test",
    data_dir: str | None = None,
) -> list[tuple[dict[str, Any], list[ABCDGroundTruth]]]:
    """Load ABCD conversations with pre-extracted ground truth.

    Returns:
        List of ``(conversation, ground_truths)`` tuples.
    """
    conversations = load_abcd_data(split, data_dir)
    return [(conv, extract_ground_truth(conv)) for conv in conversations]


def last_agent_response_text(conversation: dict[str, Any]) -> str:
    """Return the final delexed agent utterance for dialogue-level text eval."""
    for turn in reversed(conversation.get("delexed", [])):
        if turn.get("speaker") == "agent":
            return str(turn.get("text", "")).strip()
    return ""


def last_agent_response_texts(conversations: list[dict[str, Any]]) -> list[str]:
    """Return one dialogue-level text reference per ABCD conversation."""
    return [last_agent_response_text(conv) for conv in conversations]
