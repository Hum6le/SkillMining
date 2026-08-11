#!/usr/bin/env python3
"""Create a fixed-split ASIoffline induction corpus for one ABCD subflow.

This is Step 1 of the reproduction.  It does not call an LLM and it does not
read ABCD dev/test examples.  It creates auditable per-trajectory artifacts
that the later program-induction step will consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from asi_offline import build_induction_corpus


def _read_subflow_train_split(path: Path, subflow: str) -> list[dict]:
    """Load and validate the shared, session-level induction split.

    ASI's original induction is performed over successful examples of the
    *same task template*.  ``subflow`` is ABCD's closest task-template unit.
    Requiring the shared per-subflow ``train.json`` makes that induction range
    explicit and keeps it identical to the other ABCD baselines.
    """
    if not path.is_file():
        raise FileNotFoundError(f"ASIoffline induction split does not exist: {path}")
    conversations = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(conversations, list):
        raise ValueError(f"ASIoffline induction split must be a JSON array: {path}")
    observed = {
        str(conversation.get("scenario", {}).get("subflow", ""))
        for conversation in conversations
    }
    if observed != {subflow}:
        raise ValueError(
            "ASIoffline induction split is not isolated to the requested "
            f"subflow {subflow!r}; observed {sorted(observed)!r}"
        )
    if not conversations:
        raise ValueError(f"ASIoffline induction split is empty: {path}")
    return conversations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a fixed train-only ABCD induction corpus for ASIoffline."
    )
    parser.add_argument(
        "--subflow",
        required=True,
        help="One ABCD subflow / ASI task-template group.",
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help=(
            "Fixed induction JSON array. Defaults to "
            "data/eval/abcd/splits/<subflow>/train.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to outputs/asi_offline_abcd_<subflow>/induction.",
    )
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument(
        "--min-actions",
        type=int,
        default=3,
        help="Minimum primitive actions per eligible trace (original ASI uses 3).",
    )
    args = parser.parse_args()

    if args.max_conversations is not None and args.max_conversations < 1:
        parser.error("--max-conversations must be positive when provided")

    subflow = args.subflow.strip()
    if not subflow:
        parser.error("--subflow must not be empty")
    train_path = (
        Path(args.train_file)
        if args.train_file
        else Path("data/eval/abcd/splits") / subflow / "train.json"
    )
    conversations = _read_subflow_train_split(train_path, subflow)
    if args.max_conversations is not None:
        conversations = conversations[: args.max_conversations]

    episodes = build_induction_corpus(
        conversations,
        source_split="train",
        min_actions=args.min_actions,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / f"asi_offline_abcd_{subflow}" / "induction"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes_path = output_dir / "induction_episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")

    manifest = {
        "method": "asioffline-abcd",
        "stage": "fixed_train_only_trace_preparation",
        "subflow": subflow,
        "source_split": "shared_subflow_train",
        "source_path": str(train_path.resolve()),
        "source_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "source_conversations": len(conversations),
        "eligible_episodes": len(episodes),
        "min_actions": args.min_actions,
        "episodes_path": str(episodes_path.resolve()),
        "policy": {
            "dev_test_used_for_induction": False,
            "task_template_group": "abcd_subflow",
            "cross_trace_consolidation": False,
            "test_time_skill_updates": False,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        "Prepared ASIoffline induction corpus: "
        f"{len(episodes)}/{len(conversations)} train conversations -> {episodes_path}"
    )


if __name__ == "__main__":
    main()
