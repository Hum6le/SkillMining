#!/usr/bin/env python3
"""Balance independent ABCD subflows across workflow API workers.

The cost proxy is the number of non-empty agent utterance prediction targets
in both train and test splits.  This is closer to LLM-call volume than the
number of conversations, while remaining deterministic and cheap to compute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _agent_turns(path: Path) -> int:
    conversations = json.loads(path.read_text(encoding="utf-8"))
    return sum(
        1
        for conversation in conversations
        for turn in conversation.get("delexed", [])
        if turn.get("speaker") == "agent" and str(turn.get("text", "")).strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan balanced ABCD workflow workers")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--subflows", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    splits_dir = Path(args.splits_dir)
    requested = set(args.subflows)
    candidates = [
        path for path in splits_dir.iterdir()
        if path.is_dir() and (path / "train.json").is_file() and (path / "test.json").is_file()
        and (not requested or path.name in requested)
    ]
    missing = requested - {path.name for path in candidates}
    if missing:
        parser.error(f"Unknown or incomplete subflow split(s): {', '.join(sorted(missing))}")

    jobs = sorted(
        ((path.name, _agent_turns(path / "train.json") + _agent_turns(path / "test.json")) for path in candidates),
        key=lambda row: (-row[1], row[0]),
    )
    buckets = [{"turns": 0, "subflows": []} for _ in range(args.workers)]
    # Longest-processing-time greedy scheduling is a simple, strong balance
    # heuristic for a small fixed number of workflow APIs.
    for subflow, turns in jobs:
        worker = min(range(args.workers), key=lambda index: (buckets[index]["turns"], index))
        buckets[worker]["turns"] += turns
        buckets[worker]["subflows"].append({"name": subflow, "turns": turns})

    payload = {
        "cost_unit": "train_plus_test_nonempty_agent_utterance_turns",
        "total_turns": sum(turns for _, turns in jobs),
        "workers": buckets,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for index, bucket in enumerate(buckets):
        names = ", ".join(item["name"] for item in bucket["subflows"])
        print(f"worker={index} turns={bucket['turns']} subflows={names}")


if __name__ == "__main__":
    main()
