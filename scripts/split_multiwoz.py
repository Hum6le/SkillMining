#!/usr/bin/env python3
"""Split MultiWOZ by scenario and downsample to 1/10 for fast iteration.

Usage:
    python scripts/split_multiwoz.py

Output:
    data/eval/multiwoz21/splits/
        scenario_{name}_train.json   (1/10 sampled)
        scenario_{name}_val.json
        scenario_{name}_test.json
        all_train.json               (union of all scenario trains, 1/10)
        all_val.json
        all_test.json
        split_summary.json
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_tod.data import load_multiwoz21, split_by_scenario

DATA_PATH = "data/eval/multiwoz21/data/data/dialogues.json"
OUT_DIR = Path("data/eval/multiwoz21/splits")
SAMPLE_FRAC = 0.1
SEED = 42


def sample_dialogues(dialogues: list, frac: float, seed: int) -> list:
    """Randomly sample a fraction of dialogues."""
    if not dialogues:
        return []
    rng = random.Random(seed)
    n = max(1, int(len(dialogues) * frac))
    if n > len(dialogues):
        n = len(dialogues)
    indices = rng.sample(range(len(dialogues)), n)
    return [dialogues[i] for i in sorted(indices)]


def dialogue_to_dict(d) -> dict:
    """Serialize Dialogue to a JSON-safe dict (keep only essential fields)."""
    return {
        "dialogue_id": d.dialogue_id,
        "original_id": d.original_id,
        "domains": d.domains,
        "goal": {
            "description": d.goal.description,
            "inform": d.goal.inform,
            "request": d.goal.request,
        },
        "turns": [
            {
                "speaker": t.speaker,
                "utterance": t.utterance,
                "dialogue_acts": t.dialogue_acts,
            }
            for t in d.turns
        ],
    }


def main():
    print(f"Loading: {DATA_PATH}")
    dialogues = load_multiwoz21(DATA_PATH)
    print(f"  {len(dialogues)} total dialogues")

    # ── Split by scenario ──────────────────────────────────────
    print(f"Splitting by scenario (80/10/10, seed={SEED})...")
    splits = split_by_scenario(dialogues, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Sample 1/10 per scenario, save per-scenario files ───────
    rng = random.Random(SEED)
    all_train, all_val, all_test = [], [], []
    seen_ids: set[str] = set()
    summary = {}

    for scenario, sd in sorted(splits.items()):
        name = "_".join(scenario) if scenario else "general"
        t_count = len(sd["train"])
        v_count = len(sd["val"])
        ts_count = len(sd["test"])

        # Sample 1/10
        train_s = sample_dialogues(sd["train"], SAMPLE_FRAC, SEED)
        val_s = sample_dialogues(sd["val"], SAMPLE_FRAC, SEED)
        test_s = sample_dialogues(sd["test"], SAMPLE_FRAC, SEED)

        # Save per-scenario files (use -- to separate scenario from split)
        for split_name, data in [("train", train_s), ("val", val_s), ("test", test_s)]:
            if not data:
                continue
            fname = f"scenario_{name}--{split_name}.json"
            with open(OUT_DIR / fname, "w", encoding="utf-8") as f:
                json.dump([dialogue_to_dict(d) for d in data], f, indent=2, ensure_ascii=False)

        # Collect into all_* sets (dedup by dialogue_id)
        for d in train_s:
            if d.dialogue_id not in seen_ids:
                all_train.append(d); seen_ids.add(d.dialogue_id)
        for d in val_s:
            if d.dialogue_id not in seen_ids:
                all_val.append(d); seen_ids.add(d.dialogue_id)
        for d in test_s:
            if d.dialogue_id not in seen_ids:
                all_test.append(d); seen_ids.add(d.dialogue_id)

        summary[name] = {
            "full": {"train": t_count, "val": v_count, "test": ts_count},
            "sampled_1_10": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
        }

    # ── Save all_*.json (deduped unions) ───────────────────────
    for split_name, data in [("train", all_train), ("val", all_val), ("test", all_test)]:
        with open(OUT_DIR / f"all_{split_name}.json", "w", encoding="utf-8") as f:
            json.dump([dialogue_to_dict(d) for d in data], f, indent=2, ensure_ascii=False)

    # ── Save summary ───────────────────────────────────────────
    summary["_totals"] = {
        "original": len(dialogues),
        "sampled_1_10": {"train": len(all_train), "val": len(all_val), "test": len(all_test)},
        "dedup_note": "all_*.json are deduplicated (each dialogue appears once); scenario_*.json may contain duplicates for multi-domain dialogues",
        "seed": SEED,
        "sample_frac": SAMPLE_FRAC,
    }
    with open(OUT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Report ─────────────────────────────────────────────────
    print(f"\nSaved to: {OUT_DIR}/")
    files = sorted(OUT_DIR.iterdir())
    for f in files:
        size = f.stat().st_size
        print(f"  {f.name} ({size/1024:.1f} KB)")
    print(f"\nOriginal: {len(dialogues)}")
    print(f"Sampled 1/10: train={len(all_train)}, val={len(all_val)}, test={len(all_test)}")
    print(f"Total sampled: {len(all_train) + len(all_val) + len(all_test)}")


if __name__ == "__main__":
    main()
