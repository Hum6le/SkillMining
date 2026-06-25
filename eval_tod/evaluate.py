"""ToD evaluation -- generic, framework-agnostic.

Two-level API:

1. **Core** ``evaluate_predictions(dialogues, predictions)`` -- takes
   already-loaded data, returns metrics.  Works with any agent that
   produces ``Prediction`` objects.

2. **Convenience** ``evaluate(dataset_name, data_path, predictions_path)`` --
   loads data + predictions from disk, calls ``evaluate_predictions``.

Usage::

    # Direct (any framework)
    from eval_tod import evaluate_predictions
    from eval_tod.data import load_dataset

    dialogues = load_dataset("multiwoz21", data_path, split="test")
    predictions = my_agent.generate_predictions(dialogues)
    result = evaluate_predictions(dialogues, predictions)

    # File-based convenience
    from eval_tod import evaluate
    result = evaluate("multiwoz21", data_path, "outputs/preds.json")
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data import load_dataset, load_predictions
from .metrics import (
    compute_aggregate_metrics,
    compute_dialogue_metrics,
    llm_judge_evaluate,
)
from .schemas import (
    AggregateMetrics,
    Dialogue,
    DialogueMetrics,
    Prediction,
)


# ══════════════════════════════════════════════════════════════════
# Abstract agent interface
# ══════════════════════════════════════════════════════════════════

class AbstractTodAgent(ABC):
    """Interface that any ToD agent must implement.

    Different frameworks (Trace2Skill, AWM, ExpeL, etc.) implement
    ``generate_predictions()`` differently, but all produce the same
    ``list[Prediction]`` output.  This interface ensures that any
    agent can be plugged into the shared evaluation pipeline.
    """

    @abstractmethod
    def generate_predictions(self, dialogues: list[Dialogue]) -> list[Prediction]:
        """Run the agent on a list of dialogues and return predictions.

        Args:
            dialogues: List of ``Dialogue`` objects.

        Returns:
            List of ``Prediction`` objects (same length as ``dialogues``).
        """
        ...

    def predict_and_save(
        self, dialogues: list[Dialogue], output_path: str,
    ) -> list[Prediction]:
        """Run and persist predictions to a JSON file.

        Args:
            dialogues: List of ``Dialogue`` objects.
            output_path: Path to write predictions JSON.

        Returns:
            List of ``Prediction`` objects.
        """
        preds = self.generate_predictions(dialogues)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [_prediction_to_dict(p) for p in preds],
                f, indent=2, ensure_ascii=False,
            )
        print(f"Predictions saved to: {output_path} ({len(preds)} items)")
        return preds


# ══════════════════════════════════════════════════════════════════
# Core evaluation (framework-agnostic)
# ══════════════════════════════════════════════════════════════════

def evaluate_predictions(
    dialogues: list[Dialogue],
    predictions: list[Prediction],
    *,
    llm_judge: bool = False,
    llm_model: str = "deepseek-chat",
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_judge_sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate predictions against ground-truth dialogues.

    This is the core, framework-agnostic entry point.  Call it from any
    pipeline after your agent has produced predictions.

    Args:
        dialogues: Ground-truth dialogues with goals.
        predictions: Agent predictions (aligned by index with dialogues).
        llm_judge: If True, run multi-agent LLM judge evaluation.
        llm_model: LLM model for judge.
        llm_api_key: API key override.
        llm_base_url: API base URL override.
        llm_judge_sample_size: Sample N dialogues for LLM judge.

    Returns:
        Dict with ``aggregate``, ``per_dialogue``, ``llm_judge`` keys.
    """
    assert len(dialogues) == len(predictions), (
        f"Mismatch: {len(dialogues)} dialogues vs {len(predictions)} predictions"
    )

    # ── 1. Per-dialogue metrics ───────────────────────────────
    per_dialogue: list[DialogueMetrics] = []
    for dialogue, pred in zip(dialogues, predictions):
        per_dialogue.append(compute_dialogue_metrics(dialogue, pred))

    # ── 2. LLM Judge (optional) ───────────────────────────────
    llm_scores: Dict[str, float] = {}
    if llm_judge:
        print("Running LLM Judge (multi-agent: 5 specialists + 1 combiner)...")
        llm_scores = llm_judge_evaluate(
            dialogues=dialogues,
            predictions=predictions,
            model_name=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url,
            sample_size=llm_judge_sample_size,
        )

    # ── 3. Aggregate metrics ──────────────────────────────────
    aggregate = compute_aggregate_metrics(dialogues, predictions, llm_scores)

    return {
        "aggregate": _aggregate_to_dict(aggregate),
        "per_dialogue": [_dialogue_metrics_to_dict(m) for m in per_dialogue],
        "llm_judge": llm_scores,
    }


# ══════════════════════════════════════════════════════════════════
# File-based convenience wrapper
# ══════════════════════════════════════════════════════════════════

def evaluate(
    dataset_name: str,
    data_path: str,
    predictions_path: str,
    split: Optional[str] = None,
    llm_judge: bool = False,
    llm_judge_dimensions: Optional[List[str]] = None,
    llm_judge_sample_size: Optional[int] = None,
    llm_model: str = "deepseek-chat",
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load data + predictions from disk, align by dialogue_id, evaluate.

    Convenience wrapper around ``evaluate_predictions``.  Use this for
    quick file-based evaluation; use ``evaluate_predictions`` directly
    when you already have the data in memory.

    Args:
        dataset_name: e.g. ``"multiwoz21"``.
        data_path: Path to the dataset directory or JSON file.
        predictions_path: Path to predictions JSON file.
        split: Data split filter.
        llm_judge: If True, run LLM judge.
        llm_judge_dimensions: (unused, kept for backward compat).
        llm_judge_sample_size: Sample N dialogues for LLM judge.
        llm_model: LLM model name.
        llm_api_key: API key override.
        llm_base_url: API base URL override.
        output_path: Path to write results JSON.

    Returns:
        Dict with ``dataset``, ``split``, ``aggregate``, ``per_dialogue``, ``llm_judge``.
    """
    _ = llm_judge_dimensions  # unused, kept for backward compat

    # Load
    print(f"Loading dataset: {dataset_name} (split={split or 'all'})")
    dialogues = load_dataset(dataset_name, data_path, split)
    print(f"  Loaded {len(dialogues)} dialogues")

    pred_dicts = load_predictions(predictions_path)
    print(f"  Loaded {len(pred_dicts)} predictions")

    # Align predictions to dialogues by dialogue_id
    pred_lookup: Dict[str, Dict[str, Any]] = {
        p["dialogue_id"]: p for p in pred_dicts
    }

    predictions: list[Prediction] = []
    missing_count = 0
    for dialogue in dialogues:
        raw = pred_lookup.get(dialogue.dialogue_id)
        if raw is None:
            predictions.append(Prediction(dialogue_id=dialogue.dialogue_id))
            missing_count += 1
        else:
            predictions.append(Prediction(
                dialogue_id=dialogue.dialogue_id,
                inform_slots=raw.get("inform_slots", {}),
                request_slots=raw.get("request_slots", {}),
                booking=raw.get("booking", {}),
            ))

    if missing_count > 0:
        print(f"  Warning: {missing_count} dialogues have no prediction (scored as failed)")

    # Evaluate
    result = evaluate_predictions(
        dialogues, predictions,
        llm_judge=llm_judge,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_judge_sample_size=llm_judge_sample_size,
    )

    # Annotate with metadata
    result["dataset"] = dataset_name
    result["split"] = split or "all"

    # Write
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results written to: {output_path}")

    return result


# ══════════════════════════════════════════════════════════════════
# Serialization helpers
# ══════════════════════════════════════════════════════════════════

def _aggregate_to_dict(agg: AggregateMetrics) -> Dict[str, Any]:
    """Convert AggregateMetrics to a JSON-serializable dict."""
    return {
        "num_dialogues": agg.num_dialogues,
        "info_rate": agg.info_rate,
        "mean_info_rate": agg.mean_info_rate,
        "success_rate": agg.success_rate,
        "num_success": agg.num_success,
        "num_fail": agg.num_fail,
        "llm_judge_scores": agg.llm_judge_scores,
        "per_domain": {
            domain: _aggregate_to_dict(sub)
            for domain, sub in agg.per_domain_metrics.items()
        },
    }


def _dialogue_metrics_to_dict(dm: DialogueMetrics) -> Dict[str, Any]:
    """Convert DialogueMetrics to a JSON-serializable dict."""
    return {
        "dialogue_id": dm.dialogue_id,
        "info_rate": dm.info_rate,
        "success": dm.success,
        "inform_correct": dm.inform_correct,
        "inform_total": dm.inform_total,
        "request_correct": dm.request_correct,
        "request_total": dm.request_total,
        "booking_passed": dm.booking_passed,
        "domains_evaluated": dm.domains_evaluated,
    }


def _prediction_to_dict(pred: Prediction) -> Dict[str, Any]:
    """Convert Prediction to a JSON-serializable dict."""
    return {
        "dialogue_id": pred.dialogue_id,
        "inform_slots": pred.inform_slots,
        "request_slots": pred.request_slots,
        "booking": pred.booking,
    }


def print_summary(result: Dict[str, Any]) -> None:
    """Print a human-readable evaluation summary to stdout."""
    agg = result["aggregate"]

    print(f"\n{'=' * 60}")
    print(f"ToD EVALUATION RESULTS")
    print(f"  Dataset:  {result.get('dataset', 'N/A')}")
    print(f"  Split:    {result.get('split', 'N/A')}")
    print(f"{'=' * 60}")
    print(f"  Dialogues evaluated:  {agg['num_dialogues']}")
    print(f"  Information Rate:     {agg['info_rate']:.4f}  (slot-level)")
    print(f"  Mean Info Rate:       {agg['mean_info_rate']:.4f}  (per-dialogue avg)")
    print(f"  Success Rate:         {agg['success_rate']:.4f}")
    print(f"    Successful:         {agg['num_success']}")
    print(f"    Failed:             {agg['num_fail']}")

    if agg.get("llm_judge_scores"):
        print(f"  LLM Judge Scores:")
        for dim, score in agg["llm_judge_scores"].items():
            print(f"    {dim:24s}: {score:.2f}")

    if agg.get("per_domain"):
        print(f"\n  Per-Domain Breakdown:")
        for domain, dm in sorted(agg["per_domain"].items()):
            print(f"    [{domain}]")
            print(f"      Dialogues: {dm['num_dialogues']}")
            print(f"      Info Rate: {dm['info_rate']:.4f}")
            print(f"      Success:   {dm['success_rate']:.4f}")

    print(f"{'=' * 60}")
