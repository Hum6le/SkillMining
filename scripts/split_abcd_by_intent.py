#!/usr/bin/env python3
r"""ABCD 意图层划分：按 subflow 分层抽样，确保每个意图都有 train/test 样本。

输出两个 JSON 文件（只存 convo_id 和 subflow，轻量级）：
  - train_split.json: [{convo_id, subflow, flow}, ...]
  - test_split.json:  [{convo_id, subflow, flow}, ...]

划分策略：
  - 每个 subflow 的对话按 train_frac 分入 train/test
  - 确保每个 subflow 至少有 1 条在 test 中
  - 按 convo_id 排序输出，保证可复现

用法：
  # 从 ABCD train 分（训练时用）
  python scripts/split_abcd_by_intent.py --split train --train-frac 0.8

  # 从全量数据分
  python scripts/split_abcd_by_intent.py --split all --train-frac 0.8

  # 自定义输出路径
  python scripts/split_abcd_by_intent.py --split train --out-dir outputs/my_split
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


def split_by_subflow(
    conversations: list[dict],
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """按 subflow 分层划分，每个 subflow train_frac 进 train，其余进 test。"""
    rng = random.Random(seed)
    by_subflow: dict[str, list[dict]] = defaultdict(list)
    for conv in conversations:
        sf = str(conv.get("scenario", {}).get("subflow", "unknown"))
        by_subflow[sf].append(conv)

    train, test = [], []
    for sf, convs in sorted(by_subflow.items()):
        rng.shuffle(convs)
        n_train = max(1, min(int(len(convs) * train_frac), len(convs) - 1))
        # 确保 test 至少有 1 条
        if n_train == len(convs):
            n_train = len(convs) - 1
        train.extend(convs[:n_train])
        test.extend(convs[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def conv_to_record(conv: dict) -> dict:
    return {
        "convo_id": str(conv.get("convo_id", "?")),
        "subflow": str(conv.get("scenario", {}).get("subflow", "unknown")),
        "flow": str(conv.get("scenario", {}).get("flow", "unknown")),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split ABCD by subflow/intent")
    parser.add_argument("--split", default="train",
                        choices=["train", "dev", "test", "all"],
                        help="Which ABCD split to load (all = train+dev+test)")
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="Fraction per subflow for training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: data/eval/abcd/splits/)")
    args = parser.parse_args()

    # Load data
    if args.split == "all":
        convs = load_abcd_data("train", ABCD_DIR) + \
                load_abcd_data("dev", ABCD_DIR) + \
                load_abcd_data("test", ABCD_DIR)
    else:
        convs = load_abcd_data(args.split, ABCD_DIR)

    print(f"Loaded {len(convs)} conversations from '{args.split}'")

    # Split
    train, test = split_by_subflow(convs, args.train_frac, args.seed)

    # Count subflows
    train_sf = defaultdict(int)
    test_sf = defaultdict(int)
    for c in train:
        train_sf[str(c.get("scenario", {}).get("subflow", "unknown"))] += 1
    for c in test:
        test_sf[str(c.get("scenario", {}).get("subflow", "unknown"))] += 1

    all_sf = sorted(set(train_sf) | set(test_sf))
    print(f"\nSplit: {len(train)} train, {len(test)} test")
    print(f"Subflows: {len(all_sf)}")
    print(f"\n{'Subflow':35s} {'Train':>6s} {'Test':>6s} {'Total':>6s}")
    print("-" * 55)
    for sf in all_sf:
        tr, te = train_sf[sf], test_sf[sf]
        print(f"{sf:35s} {tr:6d} {te:6d} {tr+te:6d}")

    # Save
    out_dir = Path(args.out_dir) if args.out_dir else \
              Path("data/eval/abcd/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = [conv_to_record(c) for c in train]
    test_records = [conv_to_record(c) for c in test]

    train_path = out_dir / f"train_split_{args.split}.json"
    test_path = out_dir / f"test_split_{args.split}.json"

    train_path.write_text(json.dumps(train_records, indent=2, ensure_ascii=False), encoding="utf-8")
    test_path.write_text(json.dumps(test_records, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also save full conversations for the experiment
    train_full_path = out_dir / f"train_convs_{args.split}.json"
    test_full_path = out_dir / f"test_convs_{args.split}.json"
    train_full_path.write_text(
        json.dumps(train, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    test_full_path.write_text(
        json.dumps(test, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    print(f"\nSaved:")
    print(f"  Index:     {train_path}")
    print(f"  Index:     {test_path}")
    print(f"  Full conv: {train_full_path} ({len(train)} convs)")
    print(f"  Full conv: {test_full_path} ({len(test)} convs)")


if __name__ == "__main__":
    main()
