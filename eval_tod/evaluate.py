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
# Unified evaluation — auto-detect dataset and run all metrics
# ══════════════════════════════════════════════════════════════════

def evaluate_all(
    dialogues,
    predictions,
    *,
    dataset_name: str = "multiwoz",
    llm_judge: bool = False,
    llm_model: str = "deepseek-chat",
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ALL applicable metrics on a set of dialogues + predictions.

    Auto-detects available data and runs every metric that applies:

    - **Slot metrics** (IR / Success Rate): run if dialogues have
      ``goal.inform`` / ``goal.request`` (MultiWOZ).
    - **Text metrics** (BERTScore / BLEU / ROUGE): run if any prediction
      has a non-empty ``response_text``.
    - **AST / CDS** (Action State Tracking / Cascading Dialogue Success):
      run if ``dataset_name == "abcd"`` and ground-truth action annotations
      are available.
    - **LLM Judge**: run if ``llm_judge=True``.

    Args:
        dialogues: Ground-truth dialogues (MultiWOZ ``Dialogue`` or ABCD
            conversation dicts).
        predictions: Agent predictions (``Prediction`` for MultiWOZ,
            ``ABCDPrediction`` for ABCD, or plain dicts with
            ``response_text``).
        dataset_name: ``"multiwoz"`` or ``"abcd"``.
        llm_judge: If True, run multi-agent LLM judge.
        llm_model / llm_api_key / llm_base_url: LLM config for judge.

    Returns:
        Dict with keys ``slot``, ``text``, ``ast_cds``, ``llm_judge``
        (only the ones that were actually run), plus a ``summary`` string.
    """
    result: Dict[str, Any] = {}

    # ── Slot metrics (MultiWOZ) ────────────────────────────────
    if dataset_name == "multiwoz" and hasattr(dialogues[0], "goal"):
        try:
            slot_result = evaluate_predictions(
                dialogues, predictions,
                llm_judge=llm_judge,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
            )
            result["slot"] = {
                "info_rate": slot_result["aggregate"]["info_rate"],
                "success_rate": slot_result["aggregate"]["success_rate"],
                "num_success": slot_result["aggregate"]["num_success"],
                "num_fail": slot_result["aggregate"]["num_fail"],
            }
            # Pass through per_dialogue for callers that need it (e.g. induction)
            result["per_dialogue"] = slot_result.get("per_dialogue", [])
            if llm_judge:
                result["llm_judge"] = slot_result.get("llm_judge", {})
        except Exception as e:
            result["slot"] = {"error": str(e)}

    # ── Text metrics (any dataset with response_text) ──────────
    resp_texts = _extract_response_texts(predictions)
    if any(resp_texts):
        refs = _extract_references(dialogues, dataset_name)
        if refs and len(refs) == len(resp_texts):
            from .text_eval import evaluate_responses
            try:
                text_result = evaluate_responses(resp_texts, refs)
                result["text"] = {
                    "bert_f1": text_result.bert_f1,
                    "bert_precision": text_result.bert_precision,
                    "bert_recall": text_result.bert_recall,
                    "bleu_1": text_result.bleu_1,
                    "bleu_4": text_result.bleu_4,
                    "rouge_1": text_result.rouge_1,
                    "rouge_2": text_result.rouge_2,
                    "rouge_l": text_result.rouge_l,
                    "num_samples": text_result.num_samples,
                    "per_sample": text_result.per_sample,
                }
            except Exception as e:
                result["text"] = {"error": str(e)}

    # ── AST / CDS (ABCD) — only if predictions are ABCDPrediction ──
    if dataset_name == "abcd":
        from .abcd.schemas import ABCDPrediction
        if predictions and isinstance(predictions[0], ABCDPrediction):
            try:
                from .abcd.data import extract_ground_truth
                from .abcd.metrics import evaluate_abcd

                all_gt = []
                for conv in dialogues:
                    all_gt.append(extract_ground_truth(conv))

                abcd_result = evaluate_abcd(all_gt, predictions)
                result["ast_cds"] = {
                    "ast_joint": abcd_result.ast.joint_accuracy,
                    "ast_action_name": abcd_result.ast.action_name_accuracy,
                    "ast_slot_value": abcd_result.ast.slot_value_accuracy,
                    "cds_overall": abcd_result.cds.overall_cds,
                    "num_action_turns": abcd_result.ast.total_action_turns,
                }
            except Exception as e:
                result["ast_cds"] = {"error": str(e)}

    # ── Summary line ───────────────────────────────────────────
    parts = [f"eval({dataset_name}, N={len(dialogues)})"]
    if "slot" in result and "error" not in result["slot"]:
        parts.append(
            f"IR={result['slot']['info_rate']:.4f} "
            f"SR={result['slot']['success_rate']:.4f}"
        )
    if "text" in result and "error" not in result["text"]:
        parts.append(
            f"BERT-F1={result['text']['bert_f1']:.4f} "
            f"BLEU-4={result['text']['bleu_4']:.1f} "
            f"ROUGE-L={result['text']['rouge_l']:.4f}"
        )
    if "ast_cds" in result and "error" not in result["ast_cds"]:
        parts.append(
            f"AST={result['ast_cds']['ast_joint']:.4f} "
            f"CDS={result['ast_cds']['cds_overall']:.4f}"
        )
    result["summary"] = "  ".join(parts)

    return result


def _extract_response_texts(predictions) -> list[str]:
    """Extract response_text from any prediction type."""
    texts = []
    for p in predictions:
        if hasattr(p, "response_text"):
            texts.append(p.response_text or "")
        elif isinstance(p, dict):
            texts.append(p.get("response_text", ""))
        else:
            texts.append("")
    return texts


def _extract_references(dialogues, dataset_name: str) -> list[str]:
    """Extract ground-truth reference utterances from dialogues."""
    if dataset_name == "multiwoz":
        refs = []
        for d in dialogues:
            if hasattr(d, "turns"):
                sys_utts = [t.utterance for t in d.turns if getattr(t, "speaker", "") == "system"]
                refs.append(sys_utts[-1] if sys_utts else "")
            else:
                refs.append("")
        return refs
    if dataset_name == "abcd":
        from .abcd.data import last_agent_response_texts

        return last_agent_response_texts(dialogues)
    return []


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
                response_text=raw.get("response_text", ""),
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
        "response_text": pred.response_text,
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
