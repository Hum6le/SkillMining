#!/usr/bin/env python3
r"""ABCD 按 subflow 独立划分 train/test。

每个 subflow 的所有对话按比例分入 train/test，产出：
  data/eval/abcd/splits/{subflow}/
    ├── train.json    (该 subflow 的训练对话)
    └── test.json     (该 subflow 的测试对话)

用法：
  python scripts/split_abcd_by_intent.py --train-frac 0.8
  python scripts/split_abcd_by_intent.py --train-frac 0.8 --min-sessions 10
  python scripts/split_abcd_by_intent.py --output-dir outputs/my_splits
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from eval_tod.abcd.data import load_abcd_data

ABCD_DIR = "data/eval/abcd/data"
DEFAULT_OUT = Path("data/eval/abcd/splits")


def split_subflow_convs(
    convs: list[dict],
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split a single subflow's conversations into train/test."""
    rng = random.Random(seed)
    shuffled = list(convs)
    rng.shuffle(shuffled)
    n_train = max(1, min(int(len(shuffled) * train_frac), len(shuffled) - 1))
    return shuffled[:n_train], shuffled[n_train:]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Split ABCD by subflow — each subflow gets its own train/test")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--min-sessions", type=int, default=2,
                        help="Skip subflows with fewer total sessions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    print("Loading ABCD all splits...")
    all_convs = (load_abcd_data("train", ABCD_DIR) +
                 load_abcd_data("dev", ABCD_DIR) +
                 load_abcd_data("test", ABCD_DIR))
    print(f"  {len(all_convs)} total conversations")

    # Group by subflow
    by_subflow: dict[str, list[dict]] = defaultdict(list)
    for conv in all_convs:
        sf = str(conv.get("scenario", {}).get("subflow", "unknown"))
        by_subflow[sf].append(conv)

    # Split each subflow
    summary = {}
    total_train, total_test = 0, 0

    for sf, convs in sorted(by_subflow.items()):
        if len(convs) < args.min_sessions:
            continue

        train, test = split_subflow_convs(convs, args.train_frac, args.seed)

        sf_dir = out_dir / sf
        sf_dir.mkdir(parents=True, exist_ok=True)

        (sf_dir / "train.json").write_text(
            json.dumps(train, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        (sf_dir / "test.json").write_text(
            json.dumps(test, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")

        summary[sf] = {"train": len(train), "test": len(test), "total": len(convs)}
        total_train += len(train)
        total_test += len(test)

    # Save index
    (out_dir / "INDEX.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    n = len(summary)
    print(f"\nSplit complete: {n} subflows")
    print(f"  Total train: {total_train}, Total test: {total_test}")
    print(f"\n{'Subflow':35s} {'Train':>6s} {'Test':>6s} {'Total':>6s}")
    print("-" * 55)
    for sf, counts in sorted(summary.items(), key=lambda x: -x[1]["total"]):
        print(f"{sf:35s} {counts['train']:6d} {counts['test']:6d} {counts['total']:6d}")
    print(f"\nOutput: {out_dir}")
    print(f"  INDEX.json  — all subflows + counts")
    print(f"  {{subflow}}/train.json, test.json")


if __name__ == "__main__":
    main()
