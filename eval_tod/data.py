"""Dataset loading, splitting, and normalization for ToD evaluation.

Provides a unified interface for loading and splitting MultiWOZ (and
future) datasets.  All functions are framework-agnostic — just pass in
a dataset name and data path.

Usage::

    from eval_tod.data import (
        load_dataset, list_available_splits,
        split_train_val, build_batches, load_and_split,
    )

    # List splits
    counts = list_available_splits("data/eval/multiwoz21/...")

    # Load one split
    dialogues = load_dataset("multiwoz21", data_path, split="test")

    # Split train set into batches
    batches = build_batches(train_dialogues, batch_size=50)

    # Or do everything at once
    train, val, test, counts = load_and_split(
        dataset_name="multiwoz21",
        data_path="data/eval/multiwoz21/...",
        split="train",
        val_split="validation",
        test_split="test",
    )
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import Dialogue, Goal, Turn

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# MultiWOZ 2.1 loader
# ══════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════
# Dataset dispatcher
# ══════════════════════════════════════════════════════════════════

_LOADERS: Dict[str, callable] = {
    "multiwoz21": load_multiwoz21,
    # "multiwoz22": load_multiwoz22,
    # "abcd":      load_abcd,
    # "csds":      load_csds,
    # "taskmaster": load_taskmaster,
    # "woz2":      load_woz2,
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


def register_loader(name: str, loader: callable) -> None:
    """Register a new dataset loader.

    Args:
        name: Dataset key (e.g. ``"multiwoz22"``).
        loader: Callable with signature ``(data_path, split) -> list[Dialogue]``.
    """
    _LOADERS[name] = loader


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


# ══════════════════════════════════════════════════════════════════
# Prediction I/O
# ══════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════
# Splitting utilities
# ══════════════════════════════════════════════════════════════════

_MULTIWOZ_DOMAINS = [
    "attraction", "hotel", "restaurant", "train",
    "taxi", "hospital", "police",
]


def _scenario_key(dialogue) -> tuple:
    """Unique scenario key: sorted tuple of non-general domains.

    e.g. ``('hotel', 'train')`` or ``('restaurant',)``.
    Each dialogue maps to exactly one scenario.
    """
    return tuple(sorted(d for d in dialogue.domains if d != "general"))


def split_train_val(
    dialogues: list,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """Randomly split dialogues into train and validation sets.

    Framework-agnostic: works with any list of Dialogue objects.

    Args:
        dialogues: List of Dialogue objects.
        val_fraction: Fraction to hold out for val (0 < val_fraction < 1).
        seed: Random seed for reproducibility.

    Returns:
        (train_dialogues, val_dialogues) tuple.
    """
    rng = random.Random(seed)
    indices = list(range(len(dialogues)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_fraction))
    val_indices = set(indices[:n_val])
    train_indices = indices[n_val:]
    train = [dialogues[i] for i in sorted(train_indices)]
    val = [dialogues[i] for i in sorted(val_indices)]
    return train, val


def split_by_scenario(
    dialogues: list,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[tuple, dict[str, list]]:
    """Split dialogues by unique domain-combination scenario.

    Each dialogue belongs to **exactly one** scenario (defined by its
    domain combination, e.g. ``('hotel','train')``).  Within each
    scenario, dialogues are randomly assigned to train/val/test with
    the given fractions.  Tiny scenarios (< 3 dialogues) are placed
    entirely in train.

    Args:
        dialogues: List of Dialogue objects.
        train_frac: Train fraction (default 0.8).
        val_frac: Val fraction (default 0.1).
        test_frac: Test fraction (default 0.1).
        seed: Random seed.

    Returns:
        Dict of ``{scenario_tuple: {"train": [...], "val": [...], "test": [...]}}``.
        Each dialogue appears in exactly one scenario's split.

    Example::

        splits = split_by_scenario(dialogues)
        # Train on hotel+train scenario
        ht_train = splits[("hotel", "train")]["train"]
        ht_test  = splits[("hotel", "train")]["test"]
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 0.001

    from collections import defaultdict
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, d in enumerate(dialogues):
        groups[_scenario_key(d)].append(i)

    rng = random.Random(seed)
    result: dict[tuple, dict[str, list]] = {}

    for scenario, indices in sorted(groups.items()):
        n = len(indices)
        rng.shuffle(indices)

        if n < 3:
            # Too small to split — put all in train
            result[scenario] = {
                "train": [dialogues[i] for i in sorted(indices)],
                "val": [],
                "test": [],
            }
            continue

        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_val = max(1, n - n_train - n_test)
        if n_val < 1:
            n_val = 1
            n_train = n - n_val - n_test

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        result[scenario] = {
            "train": [dialogues[i] for i in sorted(train_idx)],
            "val":   [dialogues[i] for i in sorted(val_idx)],
            "test":  [dialogues[i] for i in sorted(test_idx)],
        }

    return result


def build_batches(
    dialogues: list,
    batch_size: int,
    max_batches: int | None = None,
) -> list[list]:
    """Split dialogues into fixed-size batches.

    Framework-agnostic: works with any list of Dialogue objects.

    Args:
        dialogues: List of Dialogue objects.
        batch_size: Number of dialogues per batch.
        max_batches: If set, only return the first N batches.

    Returns:
        List of batches (each batch is a list of Dialogue objects).
    """
    batches = [
        dialogues[i:i + batch_size]
        for i in range(0, len(dialogues), batch_size)
    ]
    if max_batches is not None:
        batches = batches[:max_batches]
    return batches


def load_and_split(
    dataset_name: str,
    data_path: str,
    split: str = "train",
    val_split: str | None = None,
    test_split: str | None = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    start: int = 0,
    end: int | None = None,
) -> tuple[list, list, list, dict]:
    """Load a dataset and split into train/val/test in one call.

    Strategy:
    1. If explicit ``test_split`` exists in data, load from that split.
       Otherwise, hold out ``val_fraction`` of the main pool as test.
    2. If explicit ``val_split`` exists, load from that split.
       Otherwise, hold out ``val_fraction`` of remaining data as val.
    3. The rest is train.

    Framework-agnostic.  Callable from any pipeline without importing
    PipelineConfig.

    Args:
        dataset_name: e.g. ``"multiwoz21"``.
        data_path: Path to dataset directory or JSON file.
        split: Main data split to load (default ``"train"``).
        val_split: Explicit validation split name (e.g. ``"validation"``).
        test_split: Explicit test split name (e.g. ``"test"``).
        val_fraction: Fraction for hold-out when no explicit split exists.
        seed: Random seed for reproducibility.
        start: Start index for slicing.
        end: End index for slicing (None = all).

    Returns:
        (train_dialogues, val_dialogues, test_dialogues, split_counts)
    """
    split_counts = list_available_splits(data_path)
    avail = list(split_counts.keys())

    main_split = split if split in split_counts else None

    all_dialogues = load_dataset(dataset_name, data_path, split=main_split)
    all_dialogues = all_dialogues[start:(end or len(all_dialogues))]

    # ── Test set ──
    if test_split and test_split in split_counts and test_split != (main_split or ""):
        test_dialogues = load_dataset(dataset_name, data_path, split=test_split)
        test_dialogues = test_dialogues[start:(end or len(test_dialogues))]
        remaining = all_dialogues
    else:
        test_dialogues, remaining = split_train_val(all_dialogues, val_fraction, seed)

    # ── Validation set ──
    if val_split and val_split in split_counts and val_split != (main_split or ""):
        val_dialogues = load_dataset(dataset_name, data_path, split=val_split)
        val_dialogues = val_dialogues[start:(end or len(val_dialogues))]
        train_raw = remaining
    else:
        train_raw, val_dialogues = split_train_val(remaining, val_fraction, seed)

    return train_raw, val_dialogues, test_dialogues, split_counts
