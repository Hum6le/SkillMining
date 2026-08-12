"""Offline SKILL-DISCO pseudocode-generation pipeline, without Stage-5 verification."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable
import warnings

from .abcd_trace import normalize_abcd_conversation
from .consolidation import consolidate_groups, group_operation_batch
from .operation_extraction import extract_trace_operations
from .pseudocode import render_skill_library
from .semantic_abstraction import annotate_trace_semantics
from .skill_specification import specify_skill_contract


@dataclass(frozen=True)
class _LLMCallFailure(RuntimeError):
    """A model call exhausted its bounded retry budget."""

    attempts: int
    last_error: str

    def __str__(self) -> str:
        return f"LLM call failed after {self.attempts} attempt(s): {self.last_error}"


def _retrying_chat(chat_fn: Callable[..., str], *, max_attempts: int = 3) -> Callable[..., str]:
    """Wrap transient LLM transport failures without changing stage prompts."""
    def call(prompt: str, **kwargs: Any) -> str:
        last_error = "unknown error"
        for attempt in range(1, max_attempts + 1):
            try:
                response = chat_fn(prompt, **kwargs)
                if not isinstance(response, str):
                    raise RuntimeError(f"LLM returned {type(response).__name__}, expected str")
                return response
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                if attempt < max_attempts:
                    time.sleep(attempt)
        raise _LLMCallFailure(max_attempts, last_error)
    return call


def _failure_detail(error: Exception) -> dict[str, str]:
    return {"error_type": type(error).__name__, "error": str(error)}


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
    resilient_chat = _retrying_chat(chat_fn)
    trace_artifacts: list[dict[str, Any]] = []
    all_operations = []
    for conversation in conversations:
        trace = normalize_abcd_conversation(conversation)
        base_artifact = {
            "conversation_id": trace.conversation_id,
            "normalized_trace": trace.to_dict(),
        }
        try:
            annotations, semantic_raw = annotate_trace_semantics(
                trace, resilient_chat, model=model
            )
        except (ValueError, RuntimeError) as error:
            warnings.warn(
                f"Skipping trace {trace.conversation_id} after semantic abstraction failure: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            trace_artifacts.append({
                **base_artifact, "status": "skipped_semantic_abstraction", **_failure_detail(error),
            })
            continue
        try:
            operations, rejected, operation_raw = extract_trace_operations(
                trace, annotations, resilient_chat, model=model
            )
        except (ValueError, RuntimeError) as error:
            warnings.warn(
                f"Skipping trace {trace.conversation_id} after operation extraction failure: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            trace_artifacts.append({
                **base_artifact,
                "status": "skipped_operation_extraction",
                "semantic_raw_output": semantic_raw,
                "semantic_annotations": [annotation.to_dict() for annotation in annotations],
                **_failure_detail(error),
            })
            continue
        all_operations.extend(operations)
        trace_artifacts.append({
            **base_artifact,
            "status": "ok",
            "semantic_raw_output": semantic_raw,
            "semantic_annotations": [annotation.to_dict() for annotation in annotations],
            "operation_raw_output": operation_raw,
            "operations": [operation.to_dict() for operation in operations],
            "rejected_operation_candidates": rejected,
        })

    groups = []
    grouping_artifacts = []
    for batch_index, start in enumerate(range(0, len(all_operations), grouping_batch_size)):
        batch = all_operations[start : start + grouping_batch_size]
        try:
            batch_groups, raw_output = group_operation_batch(
                batch, resilient_chat, batch_index=batch_index, model=model
            )
        except (ValueError, RuntimeError) as error:
            warnings.warn(
                f"Skipping operation grouping batch {batch_index}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            grouping_artifacts.append({
                "batch_index": batch_index, "status": "skipped_grouping", **_failure_detail(error),
            })
            continue
        groups.extend(batch_groups)
        grouping_artifacts.append({
            "batch_index": batch_index,
            "status": "ok",
            "raw_output": raw_output,
            "groups": [group.to_dict() for group in batch_groups],
        })

    operation_by_id = {operation.operation_id: operation for operation in all_operations}
    total_conversations = len({operation.conversation_id for operation in all_operations})
    consolidation_raw = ""
    consolidation_error: dict[str, str] | None = None
    if groups:
        try:
            clusters, consolidation_raw = consolidate_groups(
                groups, operation_by_id, total_conversations, resilient_chat, model=model
            )
        except (ValueError, RuntimeError) as error:
            warnings.warn(
                f"Skipping group consolidation after LLM failure: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            clusters = []
            consolidation_error = _failure_detail(error)
    else:
        clusters = []
        consolidation_error = {"error_type": "NoOperations", "error": "no valid operations available for consolidation"}

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
        try:
            contract, raw_output = specify_skill_contract(
                cluster, operations, resilient_chat, model=model
            )
        except (ValueError, RuntimeError) as error:
            warnings.warn(
                f"Skipping skill contract for cluster {cluster.cluster_id}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            contract_artifacts.append({
                "cluster_id": cluster.cluster_id,
                "status": "skipped_contract_error",
                **_failure_detail(error),
            })
            continue
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
        "consolidation_error": consolidation_error,
        "contracts": [contract.to_dict() for contract in contracts],
        "contract_artifacts": contract_artifacts,
        "skill_library": render_skill_library(contracts),
    }
