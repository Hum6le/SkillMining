#!/usr/bin/env python3
"""Generate an ABCD pseudocode skill library from an induction JSON split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm import chat
from skill_disco.pipeline import run_offline_pseudocode_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline SKILL-DISCO pseudocode generation for ABCD")
    parser.add_argument("--input", required=True, help="Induction conversation JSON array")
    parser.add_argument("--output", required=True, help="Full JSON artifact output")
    parser.add_argument("--library-output", required=True, help="Rendered SKILL.md output")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    conversations = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(conversations, list):
        raise ValueError("--input must be a JSON array of ABCD conversations")
    if args.limit is not None:
        conversations = conversations[:args.limit]
    artifact = run_offline_pseudocode_pipeline(
        conversations, chat, model=args.model, grouping_batch_size=args.batch_size, min_support=args.min_support
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    library = Path(args.library_output)
    library.parent.mkdir(parents=True, exist_ok=True)
    library.write_text(artifact["skill_library"], encoding="utf-8")
    print(f"Generated {len(artifact['contracts'])} pseudocode skills -> {library}")


if __name__ == "__main__":
    main()
