"""ABCD evaluation metrics — AST and CDS.

AST (Action State Tracking)
----------------------------
Evaluates action-turn prediction accuracy.  An action turn is correct
only when both the *action name* and all *slot values* match ground truth
(joint exact match).

    AST = correct_action_turns / total_action_turns

Sub-metrics:
- **Action Name Accuracy**: fraction where ``predicted_action`` is correct.
- **Slot Value Accuracy**: fraction where all slot values match.
- **Joint AST**: fraction where both are correct simultaneously.

CDS (Cascading Dialogue Success)
---------------------------------
Measures sequence-level understanding.  For each starting turn *i* in a
dialogue, the model must predict the *remainder* of the conversation.
The score from position *i* is the number of consecutive correct
predictions *before the first error*, divided by the number of remaining
steps.  The per-dialogue CDS is the mean across all starting positions.

    For a dialogue of length L:
        score_i = steps_correct_before_first_error / (L - i)
        dialogue_cds = mean(score_i for i in 0..L-1)
        overall_cds = mean(dialogue_cds over all dialogues)

This is a strict metric — a single mistake breaks the cascade.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .schemas import (
    ABCDGroundTruth,
    ABCDPrediction,
    ABCDTurnPrediction,
)


# ══════════════════════════════════════════════════════════════════
# AST
# ══════════════════════════════════════════════════════════════════

@dataclass
class ASTResult:
    """Per-dialogue AST scores."""

    conversation_id: str
    num_action_turns: int = 0
    action_name_correct: int = 0
    slot_values_correct: int = 0
    action_correct_slot_values_correct: int = 0
    action_correct_turns: int = 0
    joint_correct: int = 0

    @property
    def action_name_accuracy(self) -> float:
        if self.num_action_turns == 0:
            return 1.0
        return self.action_name_correct / self.num_action_turns

    @property
    def slot_value_accuracy(self) -> float:
        if self.num_action_turns == 0:
            return 1.0
        return self.slot_values_correct / self.num_action_turns

    @property
    def joint_accuracy(self) -> float:
        """The AST metric: action name AND all slots correct."""
        if self.num_action_turns == 0:
            return 1.0
        return self.joint_correct / self.num_action_turns

    @property
    def slot_accuracy_given_action(self) -> float:
        """Slot exact-match accuracy conditioned on the correct action."""
        if self.action_correct_turns == 0:
            return 0.0
        return self.action_correct_slot_values_correct / self.action_correct_turns


@dataclass
class ASTAggregate:
    """Aggregated AST across all dialogues."""

    per_dialogue: list[ASTResult] = field(default_factory=list)

    @property
    def total_action_turns(self) -> int:
        return sum(r.num_action_turns for r in self.per_dialogue)

    @property
    def action_name_accuracy(self) -> float:
        total = self.total_action_turns
        if total == 0:
            return 0.0
        return sum(r.action_name_correct for r in self.per_dialogue) / total

    @property
    def slot_value_accuracy(self) -> float:
        total = self.total_action_turns
        if total == 0:
            return 0.0
        return sum(r.slot_values_correct for r in self.per_dialogue) / total

    @property
    def joint_accuracy(self) -> float:
        """Aggregated AST: joint correct / total action turns."""
        total = self.total_action_turns
        if total == 0:
            return 0.0
        return sum(r.joint_correct for r in self.per_dialogue) / total

    @property
    def action_correct_turns(self) -> int:
        return sum(r.action_correct_turns for r in self.per_dialogue)

    @property
    def slot_accuracy_given_action(self) -> float:
        """Slot exact-match accuracy among turns with the right action."""
        total = self.action_correct_turns
        if total == 0:
            return 0.0
        return sum(r.action_correct_slot_values_correct for r in self.per_dialogue) / total

    @property
    def mean_joint_accuracy(self) -> float:
        """Mean of per-dialogue joint accuracies."""
        n = len(self.per_dialogue)
        if n == 0:
            return 0.0
        return sum(r.joint_accuracy for r in self.per_dialogue) / n


def _casefold_slot(value: Any) -> str:
    """Normalize only Unicode, case, and repeated surrounding whitespace."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def _alnum_slot(value: Any) -> str:
    """Diagnostic-only comparison that also ignores punctuation/spacing."""
    return "".join(character for character in _casefold_slot(value) if character.isalnum())


def _multiset_equal(left: list[str], right: list[str]) -> bool:
    return Counter(left) == Counter(right)


def slot_match_profile(gold: list[Any] | None, predicted: list[Any] | None) -> dict[str, bool]:
    """Return nested slot-match views without changing the strict AST metric.

    ``strict_ordered`` remains the repository's historical headline metric.
    The other views isolate permutation, case/whitespace, and punctuation
    effects.  Multiset comparison is used so duplicate values are preserved.
    """
    gold_exact = [str(value) for value in (gold or [])]
    pred_exact = [str(value) for value in (predicted or [])]
    gold_casefold = [_casefold_slot(value) for value in gold_exact]
    pred_casefold = [_casefold_slot(value) for value in pred_exact]
    gold_alnum = [_alnum_slot(value) for value in gold_exact]
    pred_alnum = [_alnum_slot(value) for value in pred_exact]
    return {
        "strict_ordered": gold_exact == pred_exact,
        "exact_unordered": _multiset_equal(gold_exact, pred_exact),
        "casefold_ordered": gold_casefold == pred_casefold,
        "casefold_unordered": _multiset_equal(gold_casefold, pred_casefold),
        "punctuation_insensitive_ordered": gold_alnum == pred_alnum,
        "punctuation_insensitive_unordered": _multiset_equal(gold_alnum, pred_alnum),
    }


def slot_error_bucket(gold: list[Any] | None, predicted: list[Any] | None) -> str:
    """Assign an exclusive, progressively relaxed slot-error category."""
    profile = slot_match_profile(gold, predicted)
    if profile["strict_ordered"]:
        return "strict_correct"
    if profile["exact_unordered"]:
        return "permutation_only"
    if profile["casefold_ordered"]:
        return "case_or_whitespace_only"
    if profile["casefold_unordered"]:
        return "case_or_whitespace_plus_permutation"
    if profile["punctuation_insensitive_ordered"]:
        return "punctuation_or_formatting_only"
    if profile["punctuation_insensitive_unordered"]:
        return "punctuation_or_formatting_plus_permutation"
    if gold and not predicted:
        return "missing_all"
    if len(predicted or []) < len(gold or []):
        return "missing_values"
    if len(predicted or []) > len(gold or []):
        return "extra_values"
    return "value_mismatch"


def compute_ast_match_profiles(
    all_ground_truths: list[list[ABCDGroundTruth]],
    all_predictions: list[ABCDPrediction],
) -> dict[str, Any]:
    """Compute strict and diagnostic AST views over the same predictions."""
    profile_names = tuple(slot_match_profile([], []).keys())
    slot_correct = Counter({name: 0 for name in profile_names})
    joint_correct = Counter({name: 0 for name in profile_names})
    action_correct_slot_correct = Counter({name: 0 for name in profile_names})
    buckets: Counter[str] = Counter()
    total = 0
    action_correct = 0
    for dialogue_index, truths in enumerate(all_ground_truths):
        prediction = (
            all_predictions[dialogue_index]
            if dialogue_index < len(all_predictions)
            else ABCDPrediction(conversation_id=str(dialogue_index), turns=[])
        )
        pred_by_idx = {turn.turn_index: turn for turn in prediction.turns}
        for ground_truth in truths:
            if ground_truth.turn_type != "action":
                continue
            total += 1
            predicted = pred_by_idx.get(ground_truth.turn_index)
            predicted_action = predicted.predicted_action if predicted else None
            predicted_slots = predicted.predicted_slots if predicted else []
            action_ok = predicted_action == ground_truth.action_name
            action_correct += int(action_ok)
            matches = slot_match_profile(ground_truth.slot_values, predicted_slots)
            buckets[slot_error_bucket(ground_truth.slot_values, predicted_slots)] += 1
            for name, matched in matches.items():
                slot_correct[name] += int(matched)
                joint_correct[name] += int(action_ok and matched)
                action_correct_slot_correct[name] += int(action_ok and matched)
    profiles = {}
    for name in profile_names:
        profiles[name] = {
            "slot_correct": slot_correct[name],
            "slot_accuracy": slot_correct[name] / total if total else 0.0,
            "joint_correct": joint_correct[name],
            "joint_accuracy": joint_correct[name] / total if total else 0.0,
            "slot_correct_given_action": action_correct_slot_correct[name],
            "slot_accuracy_given_action": (
                action_correct_slot_correct[name] / action_correct if action_correct else 0.0
            ),
        }
    return {
        "num_action_turns": total,
        "num_action_correct_turns": action_correct,
        "profiles": profiles,
        "exclusive_slot_outcomes": dict(sorted(buckets.items())),
    }


def compute_ast(
    ground_truths: list[ABCDGroundTruth],
    prediction: ABCDPrediction,
    conversation_id: str = "",
) -> ASTResult:
    """Compute AST for one dialogue.

    Args:
        ground_truths: Extracted ground truth, one per turn.
        prediction: Model predictions for the same dialogue.
        conversation_id: Identifier for the result.

    Returns:
        ``ASTResult`` with per-turn accuracy breakdown.
    """
    result = ASTResult(conversation_id=conversation_id)

    # Build lookup by turn_index
    pred_by_idx: dict[int, ABCDTurnPrediction] = {
        p.turn_index: p for p in prediction.turns
    }

    for gt in ground_truths:
        if gt.turn_type != "action":
            continue
        result.num_action_turns += 1

        pred = pred_by_idx.get(gt.turn_index)
        if pred is None:
            continue  # no prediction for this turn → counted as wrong

        # Check action name
        if pred.predicted_action is not None and pred.predicted_action == gt.action_name:
            result.action_name_correct += 1

        # Check slot values (exact match, order-sensitive)
        gt_slots = gt.slot_values or []
        pred_slots = pred.predicted_slots or []
        slots_ok = gt_slots == pred_slots
        if slots_ok:
            result.slot_values_correct += 1

        # Joint: both correct
        action_ok = (
            pred.predicted_action is not None
            and pred.predicted_action == gt.action_name
        )
        if action_ok:
            result.action_correct_turns += 1
            if slots_ok:
                result.action_correct_slot_values_correct += 1
        if action_ok and slots_ok:
            result.joint_correct += 1

    return result


def compute_ast_aggregate(
    all_ground_truths: list[list[ABCDGroundTruth]],
    all_predictions: list[ABCDPrediction],
) -> ASTAggregate:
    """Compute aggregate AST across multiple dialogues.

    Args:
        all_ground_truths: Per-dialogue ground truth lists.
        all_predictions: Per-dialogue predictions (aligned by index).

    Returns:
        ``ASTAggregate`` with overall and per-dialogue scores.
    """
    agg = ASTAggregate()
    for idx, truths in enumerate(all_ground_truths):
        pred = (
            all_predictions[idx]
            if idx < len(all_predictions)
            else ABCDPrediction(conversation_id=str(idx), turns=[])
        )
        agg.per_dialogue.append(
            compute_ast(truths, pred, conversation_id=pred.conversation_id)
        )
    return agg


# ══════════════════════════════════════════════════════════════════
# CDS
# ══════════════════════════════════════════════════════════════════

@dataclass
class CDSResult:
    """Per-dialogue CDS scores."""

    conversation_id: str
    cascade_scores: list[float] = field(default_factory=list)
    # cascade_scores[i] = score starting from turn i

    @property
    def dialogue_cds(self) -> float:
        """Mean cascade score across all starting positions."""
        if not self.cascade_scores:
            return 0.0
        return sum(self.cascade_scores) / len(self.cascade_scores)


@dataclass
class CDSAggregate:
    """Aggregated CDS across all dialogues."""

    per_dialogue: list[CDSResult] = field(default_factory=list)

    @property
    def overall_cds(self) -> float:
        """Mean of per-dialogue CDS scores."""
        n = len(self.per_dialogue)
        if n == 0:
            return 0.0
        return sum(r.dialogue_cds for r in self.per_dialogue) / n


def _turn_match(
    gt: ABCDGroundTruth,
    pred: ABCDTurnPrediction | None,
) -> bool | None:
    """Check if a single turn prediction matches ground truth.

    Returns:
        ``True`` if correct, ``False`` if wrong, ``None`` if this turn
        should be excluded from scoring (e.g. customer turns).
    """
    if gt.turn_type == "customer":
        return None  # excluded — no prediction target, so don't count

    if pred is None:
        return False

    if gt.turn_type == "utterance":
        return pred.predicted_utterance_id == gt.utterance_id

    if gt.turn_type == "action":
        action_ok = pred.predicted_action == gt.action_name
        slots_ok = (pred.predicted_slots or []) == (gt.slot_values or [])
        return action_ok and slots_ok

    return False


def compute_cds(
    ground_truths: list[ABCDGroundTruth],
    prediction: ABCDPrediction,
    conversation_id: str = "",
) -> CDSResult:
    """Compute CDS for one dialogue — only scores actionable turns.

    Customer turns are excluded (skipped over) — they neither help nor hurt
    the cascade.  The cascade walks forward, counting consecutive correct
    predictions on *utterance* and *action* turns until the first error.

    For a dialogue with L total turns where K are actionable (non-customer):
        score_i = steps_correct / K_remaining
    """
    L = len(ground_truths)
    if L == 0:
        return CDSResult(conversation_id=conversation_id)

    pred_by_idx: dict[int, ABCDTurnPrediction] = {
        p.turn_index: p for p in prediction.turns
    }

    # Pre-compute: which positions are actionable (non-customer)
    actionable = [gt.turn_type != "customer" for gt in ground_truths]
    total_actionable = sum(actionable)
    if total_actionable == 0:
        return CDSResult(conversation_id=conversation_id, cascade_scores=[1.0] * L)

    scores: list[float] = []
    for start in range(L):
        # Count remaining actionable turns from this position
        remaining_actionable = sum(actionable[start:])
        if remaining_actionable == 0:
            scores.append(1.0)
            continue

        steps_correct = 0
        examined = 0
        for offset in range(L - start):
            t = start + offset
            gt = ground_truths[t]
            pred = pred_by_idx.get(t)
            match = _turn_match(gt, pred)
            if match is None:
                continue  # skip customer turns — don't break cascade
            examined += 1
            if match:
                steps_correct += 1
            else:
                break  # cascade stops at first error on an actionable turn
        scores.append(steps_correct / remaining_actionable)

    return CDSResult(
        conversation_id=conversation_id,
        cascade_scores=scores,
    )


def compute_cds_aggregate(
    all_ground_truths: list[list[ABCDGroundTruth]],
    all_predictions: list[ABCDPrediction],
) -> CDSAggregate:
    """Compute aggregate CDS across multiple dialogues.

    Args:
        all_ground_truths: Per-dialogue ground truth lists.
        all_predictions: Per-dialogue predictions (aligned by index).

    Returns:
        ``CDSAggregate`` with overall and per-dialogue scores.
    """
    agg = CDSAggregate()
    for idx, truths in enumerate(all_ground_truths):
        pred = (
            all_predictions[idx]
            if idx < len(all_predictions)
            else ABCDPrediction(conversation_id=str(idx), turns=[])
        )
        agg.per_dialogue.append(
            compute_cds(truths, pred, conversation_id=pred.conversation_id)
        )
    return agg


# ══════════════════════════════════════════════════════════════════
# Convenience: evaluate both at once
# ══════════════════════════════════════════════════════════════════

@dataclass
class ABCDEvalResult:
    """Combined AST + CDS evaluation result."""

    ast: ASTAggregate
    cds: CDSAggregate

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 55,
            "ABCD EVALUATION RESULTS",
            "=" * 55,
            f"  Dialogues evaluated:  {len(self.ast.per_dialogue)}",
            f"",
            f"  AST — Action State Tracking",
            f"    Total action turns:    {self.ast.total_action_turns}",
            f"    Action Name Accuracy:  {self.ast.action_name_accuracy:.4f}",
            f"    Slot Value Accuracy:   {self.ast.slot_value_accuracy:.4f}",
            f"    Joint AST:             {self.ast.joint_accuracy:.4f}",
            f"    Mean Joint AST:        {self.ast.mean_joint_accuracy:.4f}",
            f"",
            f"  CDS — Cascading Dialogue Success",
            f"    Overall CDS:           {self.cds.overall_cds:.4f}",
            "=" * 55,
        ]
        return "\n".join(lines)


def evaluate_abcd(
    all_ground_truths: list[list[ABCDGroundTruth]],
    all_predictions: list[ABCDPrediction],
) -> ABCDEvalResult:
    """Run both AST and CDS evaluation.

    Args:
        all_ground_truths: Per-dialogue ground truth lists.
        all_predictions: Per-dialogue predictions (aligned by index).

    Returns:
        ``ABCDEvalResult`` with both AST and CDS aggregates.
    """
    return ABCDEvalResult(
        ast=compute_ast_aggregate(all_ground_truths, all_predictions),
        cds=compute_cds_aggregate(all_ground_truths, all_predictions),
    )
