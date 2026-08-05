"""Offline SKILL-DISCO components for ABCD trajectory distillation."""

from .abcd_trace import (
    NormalizedABCDTrace,
    NormalizedActionStep,
    normalize_abcd_conversation,
    normalized_trace_from_dict,
)
from .semantic_abstraction import (
    SemanticEventAnnotation,
    annotate_trace_semantics,
    build_semantic_abstraction_prompt,
)
from .operation_extraction import (
    SemanticOperation,
    extract_trace_operations,
    build_operation_extraction_prompt,
    semantic_operation_from_dict,
)

__all__ = [
    "NormalizedABCDTrace",
    "NormalizedActionStep",
    "normalize_abcd_conversation",
    "normalized_trace_from_dict",
    "SemanticEventAnnotation",
    "annotate_trace_semantics",
    "build_semantic_abstraction_prompt",
    "SemanticOperation",
    "extract_trace_operations",
    "build_operation_extraction_prompt",
    "semantic_operation_from_dict",
]
