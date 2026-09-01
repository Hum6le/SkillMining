"""Held-out evaluation and acceptance policy for online ASI updates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_tod.abcd.agent import turn_results_to_abcd_predictions
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


def decide_asi_update(
    baseline_result: dict[str, Any],
    candidate_result: dict[str, Any],
    *,
    min_ast_delta: float = 0.0,
    allow_action_regression: bool = False,
    allow_slot_regression: bool = False,
) -> ASIUpdateDecision:
    """Accept only improvements that do not violate configured regressions."""
    ast_before = _metric(baseline_result, "ast_cds", "ast_joint")
    ast_after = _metric(candidate_result, "ast_cds", "ast_joint")
    action_before = _metric(baseline_result, "ast_cds", "ast_action_name")
    action_after = _metric(candidate_result, "ast_cds", "ast_action_name")
    slot_before = _metric(baseline_result, "ast_cds", "ast_slot_value")
    slot_after = _metric(candidate_result, "ast_cds", "ast_slot_value")

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
