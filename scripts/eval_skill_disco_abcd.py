#!/usr/bin/env python3
"""Evaluate a rendered SKILL-DISCO pseudocode library on a frozen ABCD test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_tod.abcd.agent import turn_results_to_abcd_predictions
from eval_tod.cli import evaluate_abcd_bundle
from eval_tod.response_logger import ResponseLogger
from skill_disco.runtime import create_skill_disco_abcd_agent, load_skill_library


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline SKILL-DISCO pseudocode on ABCD")
    parser.add_argument("--skill-library", required=True)
    parser.add_argument("--test-file", required=True, help="Frozen ABCD test JSON array")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    conversations = json.loads(Path(args.test_file).read_text(encoding="utf-8"))
    if args.max_test is not None:
        conversations = conversations[:args.max_test]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = create_skill_disco_abcd_agent(
        load_skill_library(args.skill_library), model=args.model,
        response_logger=ResponseLogger(str(output_dir / "llm_responses")),
    )
    turn_results = agent.generate_all_turn_predictions(conversations, predict_actions=True)
    grouped = turn_results_to_abcd_predictions(turn_results, conversations)
    abcd_records = [{
        "conversation_id": prediction.conversation_id,
        "turns": [{
            "turn_index": turn.turn_index, "turn_type": turn.turn_type,
            "predicted_utterance_id": turn.predicted_utterance_id,
            "predicted_action": turn.predicted_action, "predicted_slots": turn.predicted_slots,
        } for turn in prediction.turns],
    } for prediction in grouped]
    text_by_conversation = {
        str(row["convo_id"]): row.get("prediction", "") for row in turn_results
        if row.get("target_type", "utterance") == "utterance"
    }
    text_records = [{
        "dialogue_id": f"abcd-{conversation.get('convo_id', '?')}",
        "response_text": text_by_conversation.get(str(conversation.get("convo_id", "")), ""),
    } for conversation in conversations]
    result = evaluate_abcd_bundle(
        conversations, text_records=text_records, abcd_records=abcd_records, text_prediction_key="response_text"
    )
    (output_dir / "turn_predictions.json").write_text(json.dumps(turn_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "abcd_predictions.json").write_text(json.dumps(abcd_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["summary"])


if __name__ == "__main__":
    main()
