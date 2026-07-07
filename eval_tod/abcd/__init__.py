"""ABCD evaluation module — AST + CDS metrics for the Action-Based Conversations Dataset.

Provides:
- ``ABCDTurnPrediction`` / ``ABCDPrediction`` — prediction dataclasses
- ``ABCDGroundTruth`` — ground-truth extraction from ABCD data
- ``compute_ast()`` — Action State Tracking (joint action accuracy)
- ``compute_cds()`` — Cascading Dialogue Success (sequence-level)
- ``evaluate_abcd()`` — convenience: run both at once
- ``load_abcd_data()`` / ``load_abcd_with_truth()`` — data loading
- ``get_utterance_text()`` — utterance pool lookup
"""

from .schemas import (
    ABCDGroundTruth,
    ABCDPrediction,
    ABCDTurnPrediction,
)
from .data import (
    get_utterance_text,
    get_utterance_pool_size,
    load_abcd_data,
    load_abcd_with_truth,
    extract_ground_truth,
)
from .metrics import (
    ASTAggregate,
    ASTResult,
    CDSAggregate,
    CDSResult,
    ABCDEvalResult,
    compute_ast,
    compute_ast_aggregate,
    compute_cds,
    compute_cds_aggregate,
    evaluate_abcd,
)
from .agent_skill import (
    SkillSelectingAgent,
    compute_selection_accuracy,
)

__all__ = [
    # Schemas
    "ABCDGroundTruth",
    "ABCDPrediction",
    "ABCDTurnPrediction",
    # Data
    "get_utterance_text",
    "get_utterance_pool_size",
    "load_abcd_data",
    "load_abcd_with_truth",
    "extract_ground_truth",
    # Metrics
    "ASTAggregate",
    "ASTResult",
    "CDSAggregate",
    "CDSResult",
    "ABCDEvalResult",
    "compute_ast",
    "compute_ast_aggregate",
    "compute_cds",
    "compute_cds_aggregate",
    "evaluate_abcd",
    # Agent
    "SkillSelectingAgent",
    "compute_selection_accuracy",
]
