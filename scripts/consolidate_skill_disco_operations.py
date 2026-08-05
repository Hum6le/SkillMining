#!/usr/bin/env python3
"""Run Stage-3 two-pass SKILL-DISCO consolidation over extracted operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm import chat
from skill_disco.consolidation import (
    build_consolidation_prompt,
    build_grouping_prompt,
    consolidate_groups,
    group_operation_batch,
)
from skill_disco.operation_extraction import semantic_operation_from_dict


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Stage-3 SKILL-DISCO operation grouping and consolidation"
    )
    parser.add_argument("--operations-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    source = json.loads(Path(args.operations_input).read_text(encoding="utf-8"))
    operations = [
        semantic_operation_from_dict(operation)
        for trace in source.get("traces", [])
        if trace.get("status") == "ok"
        for operation in trace.get("operations", [])
    ]
    if not operations:
        raise ValueError("No accepted operations found in --operations-input")
    operation_by_id = {operation.operation_id: operation for operation in operations}
    if len(operation_by_id) != len(operations):
        raise ValueError("operation IDs must be unique before consolidation")
    total_conversations = len({operation.conversation_id for operation in operations})
    batches = [operations[index : index + args.batch_size] for index in range(0, len(operations), args.batch_size)]

    output_path = Path(args.output)
    batch_artifacts: list[dict] = []
    groups = []
    grouping_failed = False
    for batch_index, batch in enumerate(batches):
        print(f"Grouping batch {batch_index + 1}/{len(batches)} ({len(batch)} operations)")
        prompt = build_grouping_prompt(batch)
        artifact = {"batch_index": batch_index, "operation_ids": [operation.operation_id for operation in batch], "prompt": prompt}
        if args.dry_run:
            artifact["status"] = "dry_run"
        else:
            try:
                batch_groups, raw_output = group_operation_batch(
                    batch, chat, batch_index=batch_index, model=args.model
                )
                groups.extend(batch_groups)
                artifact.update(status="ok", raw_output=raw_output, groups=[group.to_dict() for group in batch_groups])
            except Exception as exc:
                grouping_failed = True
                artifact.update(status="error", error=str(exc))
        batch_artifacts.append(artifact)
        _write(output_path, {
            "method": "skill-disco-offline", "stage": "procedural_skill_consolidation",
            "operations_input": str(Path(args.operations_input)), "model": args.model,
            "grouping_batches": batch_artifacts,
            "groups": [group.to_dict() for group in groups], "clusters": [],
        })

    clusters = []
    consolidation_artifact: dict = {"status": "not_run"}
    if args.dry_run:
        consolidation_artifact = {"status": "dry_run", "reason": "grouping outputs require LLM execution"}
    elif not grouping_failed:
        prompt = build_consolidation_prompt(groups, operation_by_id)
        consolidation_artifact = {"prompt": prompt}
        try:
            clusters, raw_output = consolidate_groups(
                groups, operation_by_id, total_conversations, chat, model=args.model
            )
            consolidation_artifact.update(status="ok", raw_output=raw_output)
        except Exception as exc:
            consolidation_artifact.update(status="error", error=str(exc))
    else:
        consolidation_artifact = {"status": "skipped", "reason": "at least one grouping batch failed"}

    payload = {
        "method": "skill-disco-offline",
        "stage": "procedural_skill_consolidation",
        "operations_input": str(Path(args.operations_input)),
        "model": args.model,
        "num_operations": len(operations),
        "num_source_conversations": total_conversations,
        "grouping_batches": batch_artifacts,
        "groups": [group.to_dict() for group in groups],
        "consolidation": consolidation_artifact,
        "clusters": [cluster.to_dict() for cluster in clusters],
    }
    _write(output_path, payload)
    print(f"Consolidation complete: {len(clusters)} clusters -> {output_path}")


if __name__ == "__main__":
    main()
