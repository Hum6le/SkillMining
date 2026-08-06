#!/usr/bin/env python3
"""Generate offline Stage-4 SKILL-DISCO contracts from consolidated clusters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm import chat
from skill_disco.consolidation import skill_cluster_from_dict
from skill_disco.operation_extraction import semantic_operation_from_dict
from skill_disco.skill_specification import build_skill_specification_prompt, specify_skill_contract


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Stage-4 SKILL-DISCO skill specification")
    parser.add_argument("--clusters-input", required=True)
    parser.add_argument("--operations-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cluster_source = json.loads(Path(args.clusters_input).read_text(encoding="utf-8"))
    operation_source = json.loads(Path(args.operations_input).read_text(encoding="utf-8"))
    operation_by_id = {
        operation.operation_id: operation
        for trace in operation_source.get("traces", []) if trace.get("status") == "ok"
        for operation in [semantic_operation_from_dict(item) for item in trace.get("operations", [])]
    }
    results = []
    for raw_cluster in cluster_source.get("clusters", []):
        cluster = skill_cluster_from_dict(raw_cluster)
        operations = [operation_by_id[operation_id] for operation_id in cluster.operation_ids if operation_id in operation_by_id]
        artifact = {"cluster_id": cluster.cluster_id}
        if len(cluster.supporting_conversations) < args.min_support:
            artifact.update(status="skipped_low_support", support=len(cluster.supporting_conversations))
        elif len(operations) != len(cluster.operation_ids):
            artifact.update(status="error", error="cluster refers to unavailable operations")
        else:
            prompt = build_skill_specification_prompt(cluster, operations)
            artifact["prompt"] = prompt
            if args.dry_run:
                artifact["status"] = "dry_run"
            else:
                try:
                    contract, raw_output = specify_skill_contract(cluster, operations, chat, model=args.model)
                    artifact.update(status="ok", raw_output=raw_output, contract=contract.to_dict())
                except Exception as exc:
                    artifact.update(status="error", error=str(exc))
        results.append(artifact)
        _write(Path(args.output), {"method": "skill-disco-offline", "stage": "skill_specification", "contracts": results})
    print(f"Skill specification complete: {sum(item.get('status') == 'ok' for item in results)}/{len(results)} contracts -> {args.output}")


if __name__ == "__main__":
    main()
