"""Orchestrator: load data, align predictions, compute metrics, produce report."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_loader import load_dataset, load_predictions, list_available_splits
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
    """Run the full ToD evaluation pipeline.

    1. Load ground-truth dialogues from the dataset.
    2. Load agent predictions from a JSON file.
    3. Align predictions to dialogues by ``dialogue_id``.
    4. Compute per-dialogue metrics (Information Rate, Success Rate).
    5. Compute aggregate metrics (overall + per-domain).
    6. Optionally run LLM-as-a-Judge evaluation.
    7. Serialize results to JSON if ``output_path`` is given.

    Args:
        dataset_name: e.g. ``"multiwoz21"``.
        data_path: Path to the dataset directory or JSON file.
        predictions_path: Path to predictions JSON file.
        split: Data split filter (``"train"``, ``"validation"``, ``"test"``).
        llm_judge: If ``True``, run multi-agent LLM judge evaluation.
        llm_judge_dimensions: Dimensions for LLM judge (default: all 5).
        llm_judge_sample_size: Sample N dialogues for LLM judge.
        llm_model: LLM model name (default: ``"deepseek-chat"``).
        llm_api_key: API key override (falls back to env var).
        llm_base_url: API base URL override (falls back to env var).
        output_path: Path to write results JSON.  If ``None``, no file
            is written.

    Returns:
        Dict with keys: ``dataset``, ``split``, ``aggregate``,
        ``per_dialogue``, ``llm_judge``.
    """
    # ── 1. Load data ──────────────────────────────────────────
    print(f"Loading dataset: {dataset_name} (split={split or 'all'})")
    dialogues = load_dataset(dataset_name, data_path, split)
    print(f"  Loaded {len(dialogues)} dialogues")

    pred_dicts = load_predictions(predictions_path)
    print(f"  Loaded {len(pred_dicts)} predictions")

    # ── 2. Build prediction lookup ────────────────────────────
    pred_lookup: Dict[str, Dict[str, Any]] = {
        p["dialogue_id"]: p for p in pred_dicts
    }

    # ── 3. Align predictions to dialogues ─────────────────────
    predictions: List[Prediction] = []
    missing_count = 0

    for dialogue in dialogues:
        did = dialogue.dialogue_id
        raw = pred_lookup.get(did)
        if raw is None:
            # No prediction for this dialogue → treat as empty/failed
            predictions.append(Prediction(dialogue_id=did))
            missing_count += 1
        else:
            predictions.append(Prediction(
                dialogue_id=did,
                inform_slots=raw.get("inform_slots", {}),
                request_slots=raw.get("request_slots", {}),
                booking=raw.get("booking", {}),
            ))

    if missing_count > 0:
        print(f"  Warning: {missing_count} dialogues have no prediction (scored as failed)")

    # ── 4. Per-dialogue metrics ───────────────────────────────
    per_dialogue: List[DialogueMetrics] = []
    for dialogue, pred in zip(dialogues, predictions):
        per_dialogue.append(compute_dialogue_metrics(dialogue, pred))

    # ── 5. Aggregate metrics ──────────────────────────────────
    llm_scores: Dict[str, float] = {}
    if llm_judge:
        print(f"Running LLM Judge (multi-agent: 5 specialists + 1 combiner)...")
        llm_scores = llm_judge_evaluate(
            dialogues=dialogues,
            predictions=predictions,
            dimensions=llm_judge_dimensions,
            model_name=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url,
            sample_size=llm_judge_sample_size,
        )

    aggregate = compute_aggregate_metrics(dialogues, predictions, llm_scores)

    # ── 6. Build result ───────────────────────────────────────
    result = _build_result_dict(
        dataset_name=dataset_name,
        split=split,
        aggregate=aggregate,
        per_dialogue=per_dialogue,
        llm_scores=llm_scores,
    )

    # ── 7. Write output ───────────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results written to: {output_path}")

    return result


# ── Internal helpers ──────────────────────────────────────────────

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


def _build_result_dict(
    dataset_name: str,
    split: str | None,
    aggregate: AggregateMetrics,
    per_dialogue: List[DialogueMetrics],
    llm_scores: Dict[str, float],
) -> Dict[str, Any]:
    """Build the final result dictionary."""
    return {
        "dataset": dataset_name,
        "split": split or "all",
        "aggregate": _aggregate_to_dict(aggregate),
        "per_dialogue": [_dialogue_metrics_to_dict(m) for m in per_dialogue],
        "llm_judge": llm_scores,
    }


def print_summary(result: Dict[str, Any]) -> None:
    """Print a human-readable evaluation summary to stdout."""
    agg = result["aggregate"]

    print(f"\n{'=' * 60}")
    print(f"ToD EVALUATION RESULTS")
    print(f"  Dataset:  {result['dataset']}")
    print(f"  Split:    {result['split']}")
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
