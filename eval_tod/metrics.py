"""Evaluation metrics for ToD agent outputs.

Provides three metric families:

1. **Information Rate** — slot-level precision: what fraction of goal
   slots (inform + request) were correctly handled by the agent.

2. **Success Rate** — binary per-dialogue pass/fail: ALL inform
   constraints met, ALL requests provided, booking reference present.

3. **LLM-as-a-Judge** — placeholder for multi-dimensional LLM-based
   evaluation (stub that returns zero scores).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .schemas import (
    AggregateMetrics,
    Dialogue,
    DialogueMetrics,
    Prediction,
)
from .utils import (
    extract_booking_domains,
    extract_inform_slots,
    extract_request_slots,
    format_dialogue_text,
    get_booking_reference,
    get_pred_inform_value,
    get_pred_request_slots,
    match_value,
    normalize_slot_value,
)


# ═══════════════════════════════════════════════════════════════════
# Information Rate
# ═══════════════════════════════════════════════════════════════════

def compute_information_rate(
    dialogue: Dialogue,
    prediction: Prediction,
) -> Tuple[float, int, int, int, int]:
    """Compute per-dialogue information rate.

    Information Rate = (correct_inform + correct_request)
                     / max(total_inform + total_request, 1)

    Returns:
        (info_rate, correct_inform, total_inform, correct_request, total_request)
    """
    goal = dialogue.goal

    # Collect all expected slots
    inform_slots = extract_inform_slots(goal.inform)
    request_slots = extract_request_slots(goal.request)

    # ── Inform evaluation ──
    correct_inform = 0
    for domain, slot_name, gt_value in inform_slots:
        pred_value = get_pred_inform_value(prediction.inform_slots, domain, slot_name)
        if pred_value is not None and match_value(pred_value, gt_value):
            correct_inform += 1

    total_inform = len(inform_slots)

    # ── Request evaluation ──
    correct_request = 0
    for domain, slot_name in request_slots:
        pred_slots = get_pred_request_slots(prediction.request_slots, domain)
        if normalize_slot_value(slot_name) in pred_slots:
            correct_request += 1

    total_request = len(request_slots)

    # ── Compute rate ──
    total = total_inform + total_request
    if total == 0:
        info_rate = 1.0
    else:
        info_rate = (correct_inform + correct_request) / total

    return info_rate, correct_inform, total_inform, correct_request, total_request


# ═══════════════════════════════════════════════════════════════════
# Success Rate (binary)
# ═══════════════════════════════════════════════════════════════════

def compute_success(
    dialogue: Dialogue,
    prediction: Prediction,
) -> Tuple[bool, Optional[bool]]:
    """Compute binary success for a single dialogue.

    A dialogue is successful if and only if:

    1. Every inform slot in the goal is matched by the prediction.
    2. Every request slot in the goal appears in the prediction.
    3. For any domain with booking sub-slots, a non-empty booking
       reference is present.

    Returns:
        (success, booking_passed) where ``booking_passed`` is ``None``
        if no booking was required.
    """
    goal = dialogue.goal

    # ── Check inform ──
    inform_slots = extract_inform_slots(goal.inform)
    for domain, slot_name, gt_value in inform_slots:
        pred_value = get_pred_inform_value(prediction.inform_slots, domain, slot_name)
        if pred_value is None or not match_value(pred_value, gt_value):
            return False, None

    # ── Check request ──
    request_slots = extract_request_slots(goal.request)
    for domain, slot_name in request_slots:
        pred_slots = get_pred_request_slots(prediction.request_slots, domain)
        if normalize_slot_value(slot_name) not in pred_slots:
            return False, None

    # ── Check booking ──
    booking_domains = extract_booking_domains(goal.inform)

    if not booking_domains:
        return True, None

    booking_passed = True
    for domain in booking_domains:
        ref = get_booking_reference(prediction.booking, domain)
        if not ref:
            return False, False

    return True, True


# ═══════════════════════════════════════════════════════════════════
# Per-dialogue metrics
# ═══════════════════════════════════════════════════════════════════

def compute_dialogue_metrics(
    dialogue: Dialogue,
    prediction: Prediction,
) -> DialogueMetrics:
    """Compute all metrics for a single dialogue.

    Args:
        dialogue: Ground-truth dialogue with goal.
        prediction: Agent prediction.

    Returns:
        ``DialogueMetrics`` with all per-dialogue scores.
    """
    info_rate, corr_inf, tot_inf, corr_req, tot_req = compute_information_rate(
        dialogue, prediction
    )
    success, booking_passed = compute_success(dialogue, prediction)

    return DialogueMetrics(
        dialogue_id=dialogue.dialogue_id,
        info_rate=info_rate,
        success=success,
        inform_correct=corr_inf,
        inform_total=tot_inf,
        request_correct=corr_req,
        request_total=tot_req,
        booking_passed=booking_passed,
        domains_evaluated=list(dialogue.goal.inform.keys() | dialogue.goal.request.keys()),
    )


# ═══════════════════════════════════════════════════════════════════
# Aggregate metrics
# ═══════════════════════════════════════════════════════════════════

def compute_aggregate_metrics(
    dialogues: List[Dialogue],
    predictions: List[Prediction],
    llm_judge_scores: Optional[Dict[str, float]] = None,
    compute_per_domain: bool = True,
) -> AggregateMetrics:
    """Compute aggregate metrics across all dialogues.

    Also computes per-domain breakdowns: for each domain that appears
    in any dialogue's goal, we compute a separate ``AggregateMetrics``
    considering only the slots belonging to that domain.

    Args:
        dialogues: Ground-truth dialogues.
        predictions: Agent predictions (aligned by index).
        llm_judge_scores: Optional LLM judge dimension scores.
        compute_per_domain: If True, also compute per-domain breakdowns.
            Set to False to avoid recursion when called from
            ``_compute_per_domain``.

    Returns:
        ``AggregateMetrics`` with overall and per-domain results.
    """
    assert len(dialogues) == len(predictions), (
        f"Mismatch: {len(dialogues)} dialogues vs {len(predictions)} predictions"
    )

    # ── Per-dialogue ──
    per_dialogue: List[DialogueMetrics] = []
    for dialogue, pred in zip(dialogues, predictions):
        per_dialogue.append(compute_dialogue_metrics(dialogue, pred))

    num = len(per_dialogue)
    num_success = sum(1 for m in per_dialogue if m.success)
    num_fail = num - num_success

    # ── Aggregated info rate: sum correct / sum total ──
    total_correct = sum(m.inform_correct + m.request_correct for m in per_dialogue)
    total_slots = sum(m.inform_total + m.request_total for m in per_dialogue)
    agg_info_rate = total_correct / max(total_slots, 1)

    # ── Mean info rate ──
    mean_info_rate = sum(m.info_rate for m in per_dialogue) / max(num, 1)

    # ── Success rate ──
    success_rate = num_success / max(num, 1)

    # ── Per-domain breakdown (only at top level to avoid recursion) ──
    per_domain: Dict[str, AggregateMetrics] = {}
    if compute_per_domain:
        per_domain = _compute_per_domain(dialogues, predictions)

    return AggregateMetrics(
        num_dialogues=num,
        info_rate=agg_info_rate,
        mean_info_rate=mean_info_rate,
        success_rate=success_rate,
        num_success=num_success,
        num_fail=num_fail,
        llm_judge_scores=llm_judge_scores or {},
        per_domain_metrics=per_domain,
    )


def _compute_per_domain(
    dialogues: List[Dialogue],
    predictions: List[Prediction],
) -> Dict[str, AggregateMetrics]:
    """Compute per-domain aggregate metrics.

    For each domain, we build a filtered view of the goal slots and
    recompute information rate and success rate considering only that
    domain's inform/request slots.
    """
    from copy import deepcopy

    # Collect all domains that appear across dialogues
    all_domains: set[str] = set()
    for d in dialogues:
        all_domains.update(d.goal.inform.keys())
        all_domains.update(d.goal.request.keys())

    per_domain: Dict[str, AggregateMetrics] = {}

    for domain in sorted(all_domains):
        domain_dialogues: List[Dialogue] = []
        domain_predictions: List[Prediction] = []

        for dialogue, pred in zip(dialogues, predictions):
            # Only include dialogue if this domain appears in its goal
            if domain not in dialogue.goal.inform and domain not in dialogue.goal.request:
                continue

            # Create a dialogue clone with only this domain's goal
            d_clone = deepcopy(dialogue)
            d_clone.goal.inform = {
                domain: dialogue.goal.inform[domain]
            } if domain in dialogue.goal.inform else {}
            d_clone.goal.request = {
                domain: dialogue.goal.request[domain]
            } if domain in dialogue.goal.request else {}

            domain_dialogues.append(d_clone)
            domain_predictions.append(pred)

        if domain_dialogues:
            per_domain[domain] = compute_aggregate_metrics(
                domain_dialogues, domain_predictions,
                compute_per_domain=False,  # avoid infinite recursion
            )

    return per_domain


# ═══════════════════════════════════════════════════════════════════
# LLM-as-a-Judge — Multi-agent evaluation system
# Adapted from D:\paper\LLMasaJudge
# ═══════════════════════════════════════════════════════════════════

# Re-export the judge config for convenience
DEFAULT_LLM_DIMENSIONS = [
    "task_completion",
    "slot_accuracy",
    "dialogue_fluency",
    "helpfulness",
    "efficiency",
]

# Legacy dimension descriptions (kept for backward compatibility)
DIMENSION_DESCRIPTIONS = {
    "task_completion": "Did the agent successfully complete the user's task/goal?",
    "slot_accuracy": "Were the slot values correct and accurate?",
    "dialogue_fluency": "Was the conversation natural and well-flowing?",
    "helpfulness": "Were the agent's responses actually helpful?",
    "efficiency": "Did the agent complete the task with minimal turns?",
    "fluency": "Is the agent's response natural, fluent, and well-formed?",
    "accuracy": "Are the provided values factually correct given the KB?",
    "completeness": "Did the agent cover all aspects of the user's goal?",
    "dialogue_quality": "Overall quality of the conversation flow.",
    "error_recovery": "How well did the agent handle errors or ambiguous requests?",
}


def _dialogue_to_judge_input(
    dialogue: Dialogue,
    prediction: Prediction,
) -> dict:
    """Convert a Dialogue + Prediction pair into the format expected by MultiAgentJudge.

    Returns:
        Dict with keys: ``dialogue_id``, ``goal_description``, ``turns_text``,
        ``inform_slots``, ``request_slots``, ``booking``.
    """
    # Remove HTML span tags from goal description for cleaner judge input
    import re
    clean_goal = re.sub(r"<span[^>]*>|</span>", "", dialogue.goal.description)

    # Format turns as text
    turns_lines = []
    for turn in dialogue.turns:
        speaker = "USER" if turn.speaker == "user" else "SYSTEM"
        turns_lines.append(f"[{speaker}] {turn.utterance}")

    return {
        "dialogue_id": dialogue.dialogue_id,
        "goal_description": clean_goal,
        "turns_text": "\n".join(turns_lines),
        "inform_slots": prediction.inform_slots,
        "request_slots": prediction.request_slots,
        "booking": prediction.booking,
    }


def llm_judge_evaluate(
    dialogues: List[Dialogue],
    predictions: List[Prediction],
    dimensions: Optional[List[str]] = None,
    model_name: str = "deepseek-chat",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate dialogues using the multi-agent LLM judge system.

    Uses the architecture from ``LLMasaJudge``: multiple specialist judges
    independently evaluate each dialogue from different perspectives, then a
    combiner synthesizes their scores into a final assessment.

    Each dialogue is scored on 5 dimensions (1-5 Likert scale):
    ``task_completion``, ``slot_accuracy``, ``dialogue_fluency``,
    ``helpfulness``, ``efficiency``.

    Args:
        dialogues: Ground truth dialogues.
        predictions: Agent predictions.
        dimensions: Unused — all 5 dimensions are always evaluated.
            (Accepted for backward compatibility.)
        model_name: LLM model to use. Default: ``"deepseek-chat"``.
        api_key: API key (falls back to ``OPENAI_API_KEY`` env var).
        base_url: API base URL (falls back to ``OPENAI_BASE_URL`` env var).
        sample_size: If set, randomly sample this many dialogues for LLM eval
            (useful for cost control on large datasets).

    Returns:
        dict mapping dimension name -> average score (1-5 scale).
    """
    from .judge import MultiAgentJudge

    _ = dimensions  # all dimensions always evaluated

    # Sampling (if requested)
    if sample_size is not None and sample_size < len(dialogues):
        indices = random.sample(range(len(dialogues)), sample_size)
        sampled_dialogues = [dialogues[i] for i in indices]
        sampled_predictions = [predictions[i] for i in indices]
    else:
        sampled_dialogues = list(dialogues)
        sampled_predictions = list(predictions)

    # Convert to judge input format
    judge_inputs = [
        _dialogue_to_judge_input(d, p)
        for d, p in zip(sampled_dialogues, sampled_predictions)
    ]

    # Initialize multi-agent judge
    judge = MultiAgentJudge(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )

    # Run batch evaluation
    results = judge.evaluate_batch(judge_inputs)

    # Aggregate per-dimension averages
    dim_names = list(DEFAULT_LLM_DIMENSIONS)
    dim_sums: Dict[str, float] = {d: 0.0 for d in dim_names}
    dim_counts: Dict[str, int] = {d: 0 for d in dim_names}

    for r in results:
        scores = r.get("scores", {})
        for dim in dim_names:
            val = scores.get(dim)
            if val is not None:
                dim_sums[dim] += float(val)
                dim_counts[dim] += 1

    # Compute averages, fall back to 0 if no scores
    return {
        dim: (dim_sums[dim] / dim_counts[dim]) if dim_counts[dim] > 0 else 0.0
        for dim in dim_names
    }
