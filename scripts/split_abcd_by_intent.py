#!/usr/bin/env python3
r"""ABCD 按 subflow 整理数据 — 不改官方 train/dev/test 划分，只按意图统计。

用法：
  python scripts/split_abcd_by_intent.py --split train   # 整理 train
  python scripts/split_abcd_by_intent.py --split test    # 整理 test
  python scripts/split_abcd_by_intent.py --all            # 全部三个 split

输出 data/eval/abcd/splits/：
  {split}_convs.json     — 完整的 conversation 列表
  {split}_by_subflow.json — {subflow: [convo_ids]}
  split_summary.json      — 统计汇总
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from eval_tod.abcd.data import load_abcd_data

ABCD_DIR = "data/eval/abcd/data"
OUT_DIR = Path("data/eval/abcd/splits")


def organize_by_subflow(conversations: list[dict]) -> dict:
    """按 subflow 分组，返回 {subflow: [convo_id, ...]} 和统计。"""
    by_sf: dict[str, list[str]] = defaultdict(list)
    for conv in conversations:
        sf = str(conv.get("scenario", {}).get("subflow", "unknown"))
        cid = str(conv.get("convo_id", "?"))
        by_sf[sf].append(cid)
    return dict(by_sf)


def process_split(split_name: str):
    """Load one ABCD split, organize by subflow, save."""
    convs = load_abcd_data(split_name, ABCD_DIR)
    by_sf = organize_by_subflow(convs)

    # Save full conversations
    convs_path = OUT_DIR / f"{split_name}_convs.json"
    convs_path.write_text(
        json.dumps(convs, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    # Save subflow index
    index_path = OUT_DIR / f"{split_name}_by_subflow.json"
    index_path.write_text(json.dumps(by_sf, indent=2, ensure_ascii=False), encoding="utf-8")

    n_convs = len(convs)
    n_subflows = len(by_sf)
    print(f"  {split_name:6s}: {n_convs:5d} convs, {n_subflows:3d} subflows  "
          f"→ {convs_path.name}, {index_path.name}")

    return {"split": split_name, "n_convs": n_convs, "n_subflows": n_subflows,
            "subflows": {sf: len(ids) for sf, ids in sorted(by_sf.items())}}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Organize ABCD by subflow (no re-splitting)")
    parser.add_argument("--split", default=None,
                        choices=["train", "dev", "test"],
                        help="Single split to process")
    parser.add_argument("--all", action="store_true",
                        help="Process all three splits")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        splits = ["train", "dev", "test"]
    elif args.split:
        splits = [args.split]
    else:
        print("Use --split <name> or --all")
        sys.exit(1)

    summaries = {}
    print(f"Organizing ABCD by subflow...\n")
    for s in splits:
        summaries[s] = process_split(s)

    # Save summary
    summary_path = OUT_DIR / "split_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSummary → {summary_path}")
    for s, info in summaries.items():
        print(f"  {s}: {info['n_convs']} convs, {info['n_subflows']} subflows")


if __name__ == "__main__":
    main()
