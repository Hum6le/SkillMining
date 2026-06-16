"""Load and normalize ToD datasets (MultiWOZ 2.1, etc.)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import Dialogue, Goal, Turn


# ── MultiWOZ 2.1 loader ──────────────────────────────────────────

def _to_goal(raw_goal: Dict[str, Any]) -> Goal:
    """Convert raw MultiWOZ goal dict to Goal dataclass."""
    return Goal(
        description=raw_goal.get("description", ""),
        inform=raw_goal.get("inform", {}),
        request=raw_goal.get("request", {}),
    )


def _to_turn(raw_turn: Dict[str, Any]) -> Turn:
    """Convert raw MultiWOZ turn dict to Turn dataclass."""
    return Turn(
        speaker=raw_turn.get("speaker", ""),
        utterance=raw_turn.get("utterance", ""),
        utt_idx=raw_turn.get("utt_idx", 0),
        dialogue_acts=raw_turn.get("dialogue_acts", {}),
        state=raw_turn.get("state"),
        booked=raw_turn.get("booked"),
    )


def _to_dialogue(raw: Dict[str, Any]) -> Dialogue:
    """Convert raw MultiWOZ dialogue dict to Dialogue dataclass."""
    return Dialogue(
        dataset=raw.get("dataset", ""),
        data_split=raw.get("data_split", ""),
        dialogue_id=raw.get("dialogue_id", ""),
        original_id=raw.get("original_id", ""),
        domains=list(raw.get("domains", [])),
        goal=_to_goal(raw.get("goal", {})),
        turns=[_to_turn(t) for t in raw.get("turns", [])],
    )


def load_multiwoz21(
    data_path: str,
    split: Optional[str] = None,
) -> List[Dialogue]:
    """Load MultiWOZ 2.1 dialogues from the unified JSON format.

    Args:
        data_path: Path to the MultiWOZ 2.1 data directory (contains
                   ``data/data/dialogues.json``) or path to a specific
                   JSON file.
        split: Filter by ``data_split``: ``"train"``, ``"validation"``,
               ``"test"``, or ``None`` to load all.

    Returns:
        List of ``Dialogue`` objects.
    """
    # Resolve the data file
    if os.path.isfile(data_path):
        data_file = data_path
    else:
        candidates = [
            os.path.join(data_path, "data", "data", "dialogues.json"),
            os.path.join(data_path, "data.json"),
            os.path.join(data_path, "dataset.json"),
        ]
        data_file = None
        for cand in candidates:
            if os.path.exists(cand):
                data_file = cand
                break
        if data_file is None:
            # Try to find any JSON/JSONL file
            if os.path.isdir(data_path):
                for fname in os.listdir(data_path):
                    if fname.endswith((".json", ".jsonl")):
                        data_file = os.path.join(data_path, fname)
                        break
        if data_file is None:
            raise FileNotFoundError(
                f"Could not find dialogues.json in {data_path}. "
                f"Checked: {candidates}"
            )

    with open(data_file, "r", encoding="utf-8") as handle:
        raw_list: List[Dict[str, Any]] = json.load(handle)

    dialogues = [_to_dialogue(raw) for raw in raw_list]

    if split is not None:
        dialogues = [d for d in dialogues if d.data_split == split]

    return dialogues


# ── Dataset dispatcher ────────────────────────────────────────────

_LOADERS: Dict[str, callable] = {
    "multiwoz21": load_multiwoz21,
}


def load_dataset(
    name: str,
    data_path: str,
    split: Optional[str] = None,
) -> List[Dialogue]:
    """Load dialogues for a supported ToD dataset.

    Args:
        name: Dataset name. Currently supported: ``"multiwoz21"``.
        data_path: Path to the dataset directory or JSON file.
        split: Data split filter (``"train"``, ``"validation"``,
               ``"test"``, or ``None`` for all).

    Returns:
        List of ``Dialogue`` objects.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    loader = _LOADERS.get(name)
    if loader is None:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {list(_LOADERS.keys())}"
        )
    return loader(data_path, split)


# ── Prediction loader ────────────────────────────────────────────

def load_predictions(pred_path: str) -> List[Dict[str, Any]]:
    """Load agent predictions from a JSON file.

    Expected format: a JSON array of prediction objects, each with keys:
    ``dialogue_id``, ``inform_slots``, ``request_slots``, ``booking``.

    Also accepts JSONL (one JSON object per line).

    Args:
        pred_path: Path to the predictions file (.json or .jsonl).

    Returns:
        List of prediction dicts.
    """
    with open(pred_path, "r", encoding="utf-8") as handle:
        if pred_path.endswith(".jsonl"):
            predictions = [
                json.loads(line) for line in handle if line.strip()
            ]
        else:
            predictions = json.load(handle)

    if not isinstance(predictions, list):
        raise ValueError(
            f"Predictions file must contain a JSON array, "
            f"got {type(predictions).__name__}"
        )

    return predictions


def list_available_splits(data_path: str) -> Dict[str, int]:
    """List available data splits and their dialogue counts.

    Args:
        data_path: Path to the MultiWOZ data directory or file.

    Returns:
        Dict mapping split name to dialogue count.
    """
    dialogues = load_multiwoz21(data_path)
    counts: Dict[str, int] = {}
    for d in dialogues:
        counts[d.data_split] = counts.get(d.data_split, 0) + 1
    return counts
