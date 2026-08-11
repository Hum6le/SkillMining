#!/usr/bin/env python3
"""Statically validate raw ASI outputs against frozen ABCD expert traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from asi_offline import episode_from_dict, validate_asi_response


def _load_episode_map(path: Path) -> dict[str, object]:
    episodes = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            episode = episode_from_dict(json.loads(line))
            if episode.conversation_id in episodes:
                raise ValueError(f"duplicate conversation id in induction corpus: {episode.conversation_id}")
            episodes[episode.conversation_id] = episode
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline static validation for ASI ABCD artifacts.")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--raw-artifacts", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    episodes_path, raw_path = Path(args.episodes), Path(args.raw_artifacts)
    episode_by_id = _load_episode_map(episodes_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    accepted_functions = []
    for line_number, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        artifact = json.loads(line)
        episode_id = str(artifact.get("episode_id", ""))
        if episode_id not in episode_by_id:
            raise ValueError(f"raw artifact line {line_number} references an unknown episode: {episode_id!r}")
        result = validate_asi_response(str(artifact.get("raw_response", "")), episode_by_id[episode_id])
        payload = result.to_dict()
        payload["episode_index"] = artifact.get("episode_index")
        results.append(payload)
        if result.rewritten_trajectory_valid:
            accepted_functions.extend(candidate.to_dict() for candidate in result.accepted_functions)
    (output_dir / "validation_results.jsonl").write_text(
        "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results),
        encoding="utf-8",
    )
    (output_dir / "accepted_functions.json").write_text(
        json.dumps(accepted_functions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "method": "asioffline-abcd",
        "stage": "offline_static_action_and_rewrite_validation",
        "episodes_sha256": hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
        "raw_artifacts_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "validation_policy": {
            "environment_replay": False,
            "function_body_must_be_trace_grounded": True,
            "rewritten_trajectory_must_expand_to_expert_trace": True,
        },
        "validated_responses": len(results),
        "accepted_functions": len(accepted_functions),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validated {len(results)} raw ASI responses; accepted {len(accepted_functions)} functions -> {output_dir}")


if __name__ == "__main__":
    main()
