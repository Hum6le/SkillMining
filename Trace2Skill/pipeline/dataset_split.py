"""Dataset loading, splitting, and checkpoint utilities.

Provides:
- ``_split_train_val`` -- random train/val split
- ``_build_batches`` -- split dialogues into fixed-size batches
- ``_stage0_load_and_split`` -- full data loading with train/val/test separation
- ``_save_batch_checkpoint`` -- checkpoint current skill state
- ``_resume_from_checkpoint`` -- restore from a checkpoint
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

from .config import PipelineConfig

log = logging.getLogger(__name__)

# Re-export _PROJECT_ROOT from config for convenience
from .config import _TRACE2SKILL, _PROJECT_ROOT


def _split_train_val(
    dialogues: list, val_fraction: float, seed: int,
) -> tuple[list, list]:
    """Randomly split dialogues into train and validation sets.

    Args:
        dialogues: List of Dialogue objects.
        val_fraction: Fraction to hold out for validation (0 < val_fraction < 1).
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


def _build_batches(
    dialogues: list, batch_size: int, max_batches: int | None = None,
) -> list[list]:
    """Split dialogues into batches of batch_size.

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


def _save_batch_checkpoint(
    skill_dir: Path,
    checkpoint_root: Path,
    step: int,
    metrics: dict,
    changelog: list[str] | None = None,
    val_metrics: dict | None = None,
) -> Path:
    """Save a checkpoint of the current skill state.

    Copies the skill directory to ``checkpoint_root/step_{step:04d}/``
    and writes ``_checkpoint_meta.json`` with metadata.

    Args:
        skill_dir: Path to the current evolved skill directory.
        checkpoint_root: Root directory for all checkpoints.
        step: Current batch number.
        metrics: Per-batch metrics dict.
        changelog: Evolution changelog entries (optional).
        val_metrics: Most recent validation metrics (optional).

    Returns:
        Path to the saved checkpoint directory.
    """
    dest = checkpoint_root / f"step_{step:04d}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(skill_dir, dest)

    meta = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "changelog": changelog or [],
        "val_metrics": val_metrics,
    }
    (dest / "_checkpoint_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    log.info("Checkpoint saved: step_%04d", step)
    return dest


def _resume_from_checkpoint(
    checkpoint_path: str, target_skill_dir: Path,
) -> int:
    """Restore skill state from a checkpoint and return the step number.

    Copies the checkpoint's skill files into ``target_skill_dir``
    (overwriting existing files) and returns the step number so the
    pipeline can skip already-processed batches.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        target_skill_dir: Directory to restore the skill into.

    Returns:
        The step number from which to resume (already-completed batches).
    """
    src = Path(checkpoint_path)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    meta_file = src / "_checkpoint_meta.json"
    step = 0
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        step = meta.get("step", 0)

    # Copy skill files into target
    if target_skill_dir.exists():
        shutil.rmtree(target_skill_dir)
    shutil.copytree(src, target_skill_dir, ignore=shutil.ignore_patterns("_checkpoint_meta.json"))

    print(f"  Resumed from checkpoint: step_{step:04d}")
    return step


def _stage0_load_and_split(config: PipelineConfig):
    """Load dataset and split into train/val/test for batch training.

    Strategy:
    1. Try explicit test_split; if not found, hold out 20% of data as test.
    2. Try explicit val_split; if not found, hold out val_fraction from
       remaining data as val.
    3. The rest is train.

    Returns:
        (train_dialogues, val_dialogues, test_dialogues, split_counts)
    """
    from eval_tod.data_loader import load_dataset, list_available_splits

    data_path = str(config.resolved_data_path)
    split_counts = list_available_splits(data_path)
    avail = list(split_counts.keys())

    # Load the main data pool
    main_split = config.split or "train"
    main_split = main_split if main_split in split_counts else None

    all_dialogues = load_dataset(
        config.dataset_name, data_path, split=main_split,
    )
    all_dialogues = all_dialogues[config.start:(config.end or len(all_dialogues))]

    # ── Test set ──
    test_split_name = config.test_split or "test"
    if test_split_name in split_counts and test_split_name != (main_split or ""):
        # Load from explicit separate split
        test_dialogues = load_dataset(
            config.dataset_name, data_path, split=test_split_name,
        )
        test_dialogues = test_dialogues[config.start:(config.end or len(test_dialogues))]
        remaining = all_dialogues
    else:
        # Hold out a test fraction from the main pool
        test_dialogues, remaining = _split_train_val(
            all_dialogues, config.val_fraction, config.seed_split,
        )

    # ── Validation set ──
    val_split_name = config.val_split
    if val_split_name and val_split_name in split_counts and val_split_name != (main_split or ""):
        val_dialogues = load_dataset(
            config.dataset_name, data_path, split=val_split_name,
        )
        val_dialogues = val_dialogues[config.start:(config.end or len(val_dialogues))]
        train_raw = remaining
    else:
        # Hold out val_fraction from remaining data as val
        train_raw, val_dialogues = _split_train_val(
            remaining, config.val_fraction, config.seed_split,
        )

    return train_raw, val_dialogues, test_dialogues, split_counts
