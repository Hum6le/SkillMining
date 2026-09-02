"""Held-out evaluation and acceptance policy for online ASI updates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable

from eval_tod.abcd.agent import turn_results_to_abcd_predictions
from eval_tod.abcd.data import extract_ground_truth
from eval_tod.abcd.metrics import compute_ast
from eval_tod.cli import evaluate_abcd_bundle

from .runtime import create_asi_offline_abcd_agent, load_asi_library


@dataclass(frozen=True)
class ASIUpdateDecision:
    """Decision for promoting a candidate library version."""

    accepted: bool
    reason: str
    ast_before: float
    ast_after: float
    action_before: float
    action_after: float
    slot_before: float
    slot_after: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "ast_before": self.ast_before,
            "ast_after": self.ast_after,
            "action_before": self.action_before,
            "action_after": self.action_after,
            "slot_before": self.slot_before,
            "slot_after": self.slot_after,
        }


def _metric(result: dict[str, Any], *keys: str) -> float:
    value: Any = result
    for key in keys:
        value = value.get(key, {}) if isinstance(value, dict) else {}
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def evaluate_asi_library(
    library_path: str | Path,
    conversations: list[dict[str, Any]],
    *,
    model: str = "deepseek-chat",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a real ABCD rollout and evaluate one ASI library version."""
    output_path = Path(output_dir) if output_dir is not None else None
    response_logger = None
    if output_path is not None:
        from eval_tod.response_logger import ResponseLogger

        output_path.mkdir(parents=True, exist_ok=True)
        response_logger = ResponseLogger(str(output_path / "llm_responses"))
    agent = create_asi_offline_abcd_agent(
        load_asi_library(library_path),
        model=model,
        response_logger=response_logger,
    )
    turn_results = agent.generate_all_turn_predictions(
        conversations,
        predict_actions=True,
        verbose=False,
    )
    grouped = turn_results_to_abcd_predictions(turn_results, conversations)
    abcd_records = [
        {
            "conversation_id": prediction.conversation_id,
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "turn_type": turn.turn_type,
                    "predicted_utterance_id": turn.predicted_utterance_id,
                    "predicted_action": turn.predicted_action,
                    "predicted_slots": turn.predicted_slots,
                }
                for turn in prediction.turns
            ],
        }
        for prediction in grouped
    ]
    text_by_conversation: dict[str, str] = {}
    for row in turn_results:
        if row.get("target_type", "utterance") == "utterance":
            text_by_conversation[str(row.get("convo_id", ""))] = row.get("prediction", "")
    text_records = [
        {
            "dialogue_id": f"abcd-{conversation.get('convo_id', '?')}",
            "response_text": text_by_conversation.get(str(conversation.get("convo_id", "")), ""),
        }
        for conversation in conversations
    ]
    result = evaluate_abcd_bundle(
        conversations,
        text_records=text_records,
        abcd_records=abcd_records,
        text_prediction_key="response_text",
    )
    # The original ASI validates executable test trajectories as whole tasks.
    # ABCD has no executable browser environment, so expose the equivalent
    # conversation-level joint-AST pass rate for the promotion gate.
    per_conversation = []
    for conversation, prediction in zip(conversations, grouped):
        ast = compute_ast(
            extract_ground_truth(conversation),
            prediction,
            conversation_id=str(conversation.get("convo_id", "")),
        )
        per_conversation.append({
            "conversation_id": ast.conversation_id,
            "passed": ast.num_action_turns == 0 or ast.joint_correct == ast.num_action_turns,
            "joint_ast": ast.joint_accuracy,
        })
    passed = sum(1 for row in per_conversation if row["passed"])
    result["test_suite"] = {
        "num_conversations": len(per_conversation),
        "passed_conversations": passed,
        "pass_rate": passed / len(per_conversation) if per_conversation else 0.0,
        "per_conversation": per_conversation,
    }
    if output_path is not None:
        import json

        (output_path / "turn_predictions.json").write_text(
            json.dumps(turn_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_path / "abcd_predictions.json").write_text(
            json.dumps(abcd_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_path / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def _conversation_actions(conversation: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for turn in conversation.get("delexed", []):
        targets = turn.get("targets", [])
        if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
            actions.append(str(targets[2]))
    return actions


def build_action_centered_test_selection_prompt(
    candidates: Iterable[Any],
    conversations: Iterable[dict[str, Any]],
    *,
    max_size: int,
) -> str:
    """Build an LLM prompt for selecting real, action-relevant ABCD tests.

    The LLM selects IDs only. It never creates ABCD labels or slot values; the
    selected records remain the original database conversations.
    """
    action_names = sorted({
        str(action["action"])
        for candidate in candidates
        for action in candidate.primitive_actions
    })
    rows = []
    for conversation in conversations:
        rows.append({
            "conversation_id": str(conversation.get("convo_id", "")),
            "action_sequence": _conversation_actions(conversation),
        })
    return (
        "Construct a compact regression test suite for a newly induced ASI "
        "action. The test objective is whole-conversation joint AST: the "
        "canonical action names and all ordered slot values must be correct. "
        "Choose only existing conversation_id values from the candidate list; "
        "do not invent conversations, labels, actions, or slot values. Prefer "
        "diverse but action-relevant trajectories, including the candidate "
        "source trajectories when useful. Return JSON only in the form "
        '{"conversation_ids": ["id", ...]}.\n\n'
        f"Candidate primitive actions: {json.dumps(action_names)}\n"
        f"Maximum suite size: {max_size}\n"
        "Candidate conversations:\n"
        + json.dumps(rows, ensure_ascii=False, indent=2)
    )


def select_action_centered_test_suite(
    conversations: list[dict[str, Any]],
    candidates: Iterable[Any],
    *,
    max_size: int,
    selector: Callable[[str], str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve related real trajectories and optionally let an LLM select them."""
    if max_size < 1:
        raise ValueError("max_size must be positive")
    candidates = list(candidates)
    target_actions = {
        str(action["action"])
        for candidate in candidates
        for action in candidate.primitive_actions
    }
    scored: list[tuple[int, int, str, dict[str, Any]]] = []
    for conversation in conversations:
        actions = _conversation_actions(conversation)
        overlap = len(target_actions.intersection(actions))
        source = 1 if any(
            str(candidate.episode_id) == str(conversation.get("convo_id", ""))
            for candidate in candidates
        ) else 0
        scored.append((source, overlap, str(conversation.get("convo_id", "")), conversation))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    pool = [item[3] for item in scored if item[1] > 0 or item[0] > 0]
    if not pool:
        pool = [item[3] for item in scored]
    pool = pool[: max(max_size * 4, max_size)]
    selected_ids: list[str] = []
    if selector is not None and pool:
        raw = selector(build_action_centered_test_selection_prompt(
            candidates, pool, max_size=max_size,
        )) or ""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                selected_ids = [str(value) for value in payload.get("conversation_ids", [])]
            except (json.JSONDecodeError, AttributeError):
                selected_ids = []
    by_id = {str(row.get("convo_id", "")): row for row in pool}
    selected = [by_id[cid] for cid in selected_ids if cid in by_id][:max_size]
    selected_ids = [str(row.get("convo_id", "")) for row in selected]
    for row in pool:
        if len(selected) >= max_size:
            break
        cid = str(row.get("convo_id", ""))
        if cid not in selected_ids:
            selected.append(row)
            selected_ids.append(cid)
    return selected, {
        "candidate_actions": sorted(target_actions),
        "retrieval_pool_size": len(pool),
        "selected_ids": selected_ids,
        "selection_mode": "llm_then_retrieval_fallback" if selector else "retrieval",
    }


def decide_asi_update(
    baseline_result: dict[str, Any],
    candidate_result: dict[str, Any],
    *,
    min_ast_delta: float = 0.0,
    allow_action_regression: bool = False,
    allow_slot_regression: bool = False,
    min_test_pass_rate: float | None = None,
) -> ASIUpdateDecision:
    """Accept only improvements that do not violate configured regressions."""
    ast_before = _metric(baseline_result, "ast_cds", "ast_joint")
    ast_after = _metric(candidate_result, "ast_cds", "ast_joint")
    action_before = _metric(baseline_result, "ast_cds", "ast_action_name")
    action_after = _metric(candidate_result, "ast_cds", "ast_action_name")
    slot_before = _metric(baseline_result, "ast_cds", "ast_slot_value")
    slot_after = _metric(candidate_result, "ast_cds", "ast_slot_value")

    if min_test_pass_rate is not None:
        if not 0.0 <= min_test_pass_rate <= 1.0:
            raise ValueError("min_test_pass_rate must be in [0, 1]")
        test_pass_rate = _metric(candidate_result, "test_suite", "pass_rate")
        if test_pass_rate < min_test_pass_rate:
            return ASIUpdateDecision(
                False, "test_suite_pass_rate_below_threshold", ast_before, ast_after,
                action_before, action_after, slot_before, slot_after,
            )

    if not allow_action_regression and action_after < action_before:
        return ASIUpdateDecision(
            False, "action_accuracy_regressed", ast_before, ast_after,
            action_before, action_after, slot_before, slot_after,
        )
    if not allow_slot_regression and slot_after < slot_before:
        return ASIUpdateDecision(
            False, "slot_accuracy_regressed", ast_before, ast_after,
            action_before, action_after, slot_before, slot_after,
        )
    if ast_after < ast_before + min_ast_delta:
        return ASIUpdateDecision(
            False, "ast_joint_did_not_improve", ast_before, ast_after,
            action_before, action_after, slot_before, slot_after,
        )
    return ASIUpdateDecision(
        True, "ast_joint_improved_without_regression", ast_before, ast_after,
        action_before, action_after, slot_before, slot_after,
    )
