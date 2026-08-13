#!/usr/bin/env python3
"""Run original-style ASI induction on a frozen ABCD train-only corpus.

This is the second reproduction step. It performs no replay, no environment
interaction, and no test-set read. Later stages consume its raw artifacts.
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

from asi_offline import (
    build_episode_induction_messages,
    episode_from_dict,
    induce_episode,
)
from llm import chat


def _load_episodes(path: Path) -> list:
    if not path.is_file():
        raise FileNotFoundError(f"induction episode file does not exist: {path}")
    episodes = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    episodes.append(episode_from_dict(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"invalid induction episode at line {line_number}") from exc
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Induce raw ASI functions from fixed ABCD train-only traces."
    )
    parser.add_argument("--episodes", required=True, help="Step-1 induction_episodes.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed episode artifacts and only run missing episodes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write exact prompts without calling an LLM.",
    )
    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive when provided")

    episodes_path = Path(args.episodes)
    episodes = [episode for episode in _load_episodes(episodes_path) if episode.eligible_for_induction]
    episodes = episodes[args.start_index :]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise ValueError("no eligible ASI episodes were selected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_path = output_dir / "raw_induction_artifacts.jsonl"
    existing_by_index: dict[int, dict] = {}
    if args.resume and artifacts_path.is_file():
        for line_number, line in enumerate(artifacts_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            artifact = json.loads(line)
            try:
                index = int(artifact["episode_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid existing raw artifact at line {line_number}: missing episode_index"
                ) from exc
            if index in existing_by_index:
                raise ValueError(f"duplicate episode_index in existing raw artifacts: {index}")
            existing_by_index[index] = artifact

    new_artifacts = 0
    total_selected = len(episodes)
    for index, episode in enumerate(episodes):
        episode_index = args.start_index + index
        existing = existing_by_index.get(episode_index)
        if existing and existing.get("status") == "completed" and existing.get("raw_response"):
            print(
                f"[ASI induction {index + 1}/{total_selected}] "
                f"episode={episode.conversation_id} status=already_completed",
                flush=True,
            )
            continue
        mode = "prompt_only" if args.dry_run else "calling_llm_chat"
        print(
            f"[ASI induction {index + 1}/{total_selected}] "
            f"episode={episode.conversation_id} actions={len(episode.primitive_actions)} "
            f"status={mode}",
            flush=True,
        )
        if args.dry_run:
            artifact = {
                "episode_id": episode.conversation_id,
                "action_count": len(episode.primitive_actions),
                "messages": build_episode_induction_messages(episode),
                "raw_response": "",
                "status": "prompt_only",
            }
        else:
            artifact = induce_episode(episode, chat, temperature=args.temperature).to_dict()
            artifact["status"] = "completed" if artifact["raw_response"] else "empty_response"
        artifact["episode_index"] = episode_index
        existing_by_index[episode_index] = artifact
        new_artifacts += 1
        print(
            f"[ASI induction {index + 1}/{total_selected}] "
            f"episode={episode.conversation_id} status={artifact['status']}",
            flush=True,
        )

    with artifacts_path.open("w", encoding="utf-8") as handle:
        for episode_index in sorted(existing_by_index):
            handle.write(json.dumps(existing_by_index[episode_index], ensure_ascii=False) + "\n")

    manifest = {
        "method": "asioffline-abcd",
        "stage": "per_trajectory_action_induction",
        "source_episodes": str(episodes_path.resolve()),
        "source_episodes_sha256": hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
        "selected_episodes": len(episodes),
        "new_artifacts": new_artifacts,
        "resumed": args.resume,
        "source_policy": "fixed_train_only",
        "validation_policy": "no_environment_replay_in_offline_stage",
        "temperature": args.temperature,
        "dry_run": args.dry_run,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Induction artifacts ready: {len(existing_by_index)} total, "
        f"{new_artifacts} newly generated -> {artifacts_path}"
    )


if __name__ == "__main__":
    main()
