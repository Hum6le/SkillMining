#!/usr/bin/env python3
"""Create deterministic SKILL-DISCO trace-normalization artifacts for ABCD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skill_disco import normalize_abcd_conversation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize ABCD expert dialogues into SKILL-DISCO action IR"
    )
    parser.add_argument("--input", required=True, help="ABCD conversation JSON array")
    parser.add_argument("--output", required=True, help="Output normalized-traces JSON path")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    conversations = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(conversations, list):
        raise ValueError("--input must contain a JSON array of ABCD conversations")
    if args.limit is not None:
        conversations = conversations[: args.limit]

    traces = [normalize_abcd_conversation(conversation) for conversation in conversations]
    payload = {
        "method": "skill-disco-offline",
        "stage": "trace_normalization",
        "num_conversations": len(conversations),
        "num_nonempty_traces": sum(trace.action_count > 0 for trace in traces),
        "traces": [trace.to_dict() for trace in traces],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Normalized {payload['num_nonempty_traces']}/{payload['num_conversations']} "
        f"conversations to {output}"
    )


if __name__ == "__main__":
    main()
