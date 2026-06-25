"""Dataset splitting -- thin re-export from eval_tod.data.

All data loading and splitting logic lives in ``eval_tod.data``.
This module adds only pipeline-specific checkpoint utilities.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

# Re-export framework-agnostic split utilities
from eval_tod.data import (
    build_batches as _build_batches,
    load_and_split,
    split_train_val as _split_train_val,
)

from .config import PipelineConfig, _PROJECT_ROOT, _TRACE2SKILL

log = logging.getLogger(__name__)


def _stage0_load_and_split(config: PipelineConfig):
    """Load dataset and split into train/val/test for batch training.

    Delegates to ``eval_tod.data.load_and_split``.
    """
    return load_and_split(
        dataset_name=config.dataset_name,
        data_path=str(config.resolved_data_path),
        split=config.split or "train",
        val_split=config.val_split,
        test_split=config.test_split,
        val_fraction=config.val_fraction,
        seed=config.seed_split,
        start=config.start,
        end=config.end,
    )


# ══════════════════════════════════════════════════════════════════
# Pipeline-specific checkpoint utilities
# ══════════════════════════════════════════════════════════════════

def _save_batch_checkpoint(
    skill_dir: Path,
    checkpoint_root: Path,
    step: int,
    metrics: dict,
    changelog: list[str] | None = None,
    val_metrics: dict | None = None,
) -> Path:
    """Save a checkpoint of the current skill state."""
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
    """Restore skill state from a checkpoint and return the step number."""
    src = Path(checkpoint_path)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    meta_file = src / "_checkpoint_meta.json"
    step = 0
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        step = meta.get("step", 0)

    if target_skill_dir.exists():
        shutil.rmtree(target_skill_dir)
    shutil.copytree(src, target_skill_dir, ignore=shutil.ignore_patterns("_checkpoint_meta.json"))

    print(f"  Resumed from checkpoint: step_{step:04d}")
    return step
