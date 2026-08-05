#!/usr/bin/env python3
"""Run offline turn-level semantic abstraction over normalized ABCD traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm import chat
from skill_disco import (
    annotate_trace_semantics,
    build_semantic_abstraction_prompt,
    normalized_trace_from_dict,
)


def _write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Stage-1b semantic abstraction for SKILL-DISCO ABCD traces"
    )
    parser.add_argument("--input", required=True, help="Stage-1 normalized-traces JSON")
    parser.add_argument("--output", required=True, help="Semantic annotation JSON")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Keep completed conversations in --output")
    parser.add_argument("--dry-run", action="store_true", help="Write prompts without calling an LLM")
    args = parser.parse_args()

    normalized = json.loads(Path(args.input).read_text(encoding="utf-8"))
    trace_records = normalized.get("traces")
    if not isinstance(trace_records, list):
        raise ValueError("--input must be a normalized-traces artifact with a traces list")
    if args.limit is not None:
        trace_records = trace_records[: args.limit]

    output_path = Path(args.output)
    completed: dict[str, dict] = {}
    if args.resume and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        completed = {
            str(item.get("conversation_id")): item
            for item in previous.get("traces", [])
            if item.get("status") == "ok"
        }

    results: list[dict] = []
    for index, record in enumerate(trace_records, start=1):
        trace = normalized_trace_from_dict(record)
        if trace.conversation_id in completed:
            results.append(completed[trace.conversation_id])
            continue
        print(f"[{index}/{len(trace_records)}] conversation={trace.conversation_id}")
        prompt = build_semantic_abstraction_prompt(trace)
        artifact = {"conversation_id": trace.conversation_id, "prompt": prompt}
        if args.dry_run:
            artifact["status"] = "dry_run"
        else:
            try:
                annotations, raw_output = annotate_trace_semantics(
                    trace, chat, model=args.model
                )
                artifact.update(
                    status="ok",
                    raw_output=raw_output,
                    annotations=[annotation.to_dict() for annotation in annotations],
                )
            except Exception as exc:
                artifact.update(status="error", error=str(exc))
        results.append(artifact)
        payload = {
            "method": "skill-disco-offline",
            "stage": "semantic_abstraction",
            "normalized_input": str(Path(args.input)),
            "model": args.model,
            "traces": results,
        }
        _write_output(output_path, payload)

    ok = sum(item.get("status") == "ok" for item in results)
    print(f"Semantic abstraction complete: {ok}/{len(results)} parsed traces -> {output_path}")


if __name__ == "__main__":
    main()
