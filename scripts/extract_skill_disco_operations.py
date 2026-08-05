#!/usr/bin/env python3
"""Run offline Stage-2 operation extraction over semantically labeled ABCD traces."""

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
    SemanticEventAnnotation,
    build_operation_extraction_prompt,
    extract_trace_operations,
    normalized_trace_from_dict,
)


def _write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _annotations_from_artifact(record: dict) -> list[SemanticEventAnnotation]:
    return [
        SemanticEventAnnotation(
            turn_index=int(annotation["turn_index"]),
            dialogue_act=str(annotation["dialogue_act"]),
            intent=str(annotation["intent"]),
            state_updates=[str(value) for value in annotation.get("state_updates", [])],
            parameters=[str(value) for value in annotation.get("parameters", [])],
            control_signal=str(annotation["control_signal"]),
        )
        for annotation in record.get("annotations", [])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Stage-2 SKILL-DISCO subgoal operation extraction"
    )
    parser.add_argument("--normalized-input", required=True)
    parser.add_argument("--semantic-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    normalized = json.loads(Path(args.normalized_input).read_text(encoding="utf-8"))
    semantic = json.loads(Path(args.semantic_input).read_text(encoding="utf-8"))
    trace_records = normalized.get("traces", [])
    semantic_by_conversation = {
        str(record.get("conversation_id")): record
        for record in semantic.get("traces", [])
        if record.get("status") == "ok"
    }
    if args.limit is not None:
        trace_records = trace_records[: args.limit]

    output_path = Path(args.output)
    completed: dict[str, dict] = {}
    if args.resume and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        completed = {
            str(record.get("conversation_id")): record
            for record in previous.get("traces", [])
            if record.get("status") == "ok"
        }

    results: list[dict] = []
    for index, trace_record in enumerate(trace_records, start=1):
        trace = normalized_trace_from_dict(trace_record)
        if trace.conversation_id in completed:
            results.append(completed[trace.conversation_id])
            continue
        semantic_record = semantic_by_conversation.get(trace.conversation_id)
        if semantic_record is None:
            results.append({
                "conversation_id": trace.conversation_id,
                "status": "skipped_missing_semantics",
            })
            continue
        annotations = _annotations_from_artifact(semantic_record)
        print(f"[{index}/{len(trace_records)}] conversation={trace.conversation_id}")
        prompt = build_operation_extraction_prompt(trace, annotations)
        artifact = {"conversation_id": trace.conversation_id, "prompt": prompt}
        if args.dry_run:
            artifact["status"] = "dry_run"
        else:
            try:
                operations, rejected, raw_output = extract_trace_operations(
                    trace, annotations, chat, model=args.model
                )
                artifact.update(
                    status="ok",
                    raw_output=raw_output,
                    operations=[operation.to_dict() for operation in operations],
                    rejected_candidates=rejected,
                )
            except Exception as exc:
                artifact.update(status="error", error=str(exc))
        results.append(artifact)
        payload = {
            "method": "skill-disco-offline",
            "stage": "operation_extraction",
            "normalized_input": str(Path(args.normalized_input)),
            "semantic_input": str(Path(args.semantic_input)),
            "model": args.model,
            "traces": results,
        }
        _write_output(output_path, payload)

    ok = sum(record.get("status") == "ok" for record in results)
    print(f"Operation extraction complete: {ok}/{len(results)} parsed traces -> {output_path}")


if __name__ == "__main__":
    main()
