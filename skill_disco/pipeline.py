"""Offline SKILL-DISCO pseudocode-generation pipeline, without Stage-5 verification."""

from __future__ import annotations

from typing import Any, Callable

from .abcd_trace import normalize_abcd_conversation
from .consolidation import consolidate_groups, group_operation_batch
from .operation_extraction import extract_trace_operations
from .pseudocode import render_skill_library
from .semantic_abstraction import annotate_trace_semantics
from .skill_specification import specify_skill_contract


def run_offline_pseudocode_pipeline(
    conversations: list[dict[str, Any]],
    chat_fn: Callable[..., str],
    *,
    model: str = "deepseek-chat",
    grouping_batch_size: int = 20,
    min_support: int = 2,
) -> dict[str, Any]:
    """Run Stages 1--4 and deterministic pseudocode rendering on induction data.

    The supplied ``chat_fn`` is normally ``llm.chat``. Tests may inject a
    scripted local function; this pipeline has no other model or network path.
    """
    if grouping_batch_size < 1:
        raise ValueError("grouping_batch_size must be positive")
    trace_artifacts: list[dict[str, Any]] = []
    all_operations = []
    for conversation in conversations:
        trace = normalize_abcd_conversation(conversation)
        annotations, semantic_raw = annotate_trace_semantics(trace, chat_fn, model=model)
        operations, rejected, operation_raw = extract_trace_operations(
            trace, annotations, chat_fn, model=model
        )
        all_operations.extend(operations)
        trace_artifacts.append({
            "conversation_id": trace.conversation_id,
            "normalized_trace": trace.to_dict(),
            "semantic_raw_output": semantic_raw,
            "semantic_annotations": [annotation.to_dict() for annotation in annotations],
            "operation_raw_output": operation_raw,
            "operations": [operation.to_dict() for operation in operations],
            "rejected_operation_candidates": rejected,
        })
    if not all_operations:
        raise ValueError("no multi-action operations were extracted")

    groups = []
    grouping_artifacts = []
    for batch_index, start in enumerate(range(0, len(all_operations), grouping_batch_size)):
        batch = all_operations[start : start + grouping_batch_size]
        batch_groups, raw_output = group_operation_batch(
            batch, chat_fn, batch_index=batch_index, model=model
        )
        groups.extend(batch_groups)
        grouping_artifacts.append({
            "batch_index": batch_index,
            "raw_output": raw_output,
            "groups": [group.to_dict() for group in batch_groups],
        })

    operation_by_id = {operation.operation_id: operation for operation in all_operations}
    total_conversations = len({operation.conversation_id for operation in all_operations})
    clusters, consolidation_raw = consolidate_groups(
        groups, operation_by_id, total_conversations, chat_fn, model=model
    )

    contracts = []
    contract_artifacts = []
    for cluster in clusters:
        operations = [operation_by_id[operation_id] for operation_id in cluster.operation_ids]
        if len(cluster.supporting_conversations) < min_support:
            contract_artifacts.append({
                "cluster_id": cluster.cluster_id,
                "status": "skipped_low_support",
                "support": len(cluster.supporting_conversations),
            })
            continue
        contract, raw_output = specify_skill_contract(cluster, operations, chat_fn, model=model)
        contracts.append(contract)
        contract_artifacts.append({
            "cluster_id": cluster.cluster_id,
            "status": "ok",
            "raw_output": raw_output,
            "contract": contract.to_dict(),
        })

    return {
        "method": "skill-disco-offline-pseudocode",
        "stages": ["trace_normalization", "semantic_abstraction", "operation_extraction", "consolidation", "skill_specification", "pseudocode_rendering"],
        "traces": trace_artifacts,
        "groups": [group.to_dict() for group in groups],
        "grouping": grouping_artifacts,
        "clusters": [cluster.to_dict() for cluster in clusters],
        "consolidation_raw_output": consolidation_raw,
        "contracts": [contract.to_dict() for contract in contracts],
        "contract_artifacts": contract_artifacts,
        "skill_library": render_skill_library(contracts),
    }
