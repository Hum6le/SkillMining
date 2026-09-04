#!/usr/bin/env python3
"""Merge independent ABCD evaluation shards and compute global metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_tod.abcd import merge_turn_results
from eval_tod.abcd.agent import turn_results_to_abcd_predictions
from eval_tod.cli import evaluate_abcd_bundle
from scripts.llm_usage_utils import merge_usage_summaries, split_usage_summary


def _read_usage(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--subflow", required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversations = json.loads(args.test_file.read_text(encoding="utf-8"))
    shard_results = []
    for path in sorted(args.shard_root.glob("shard_*/turn_predictions.json")):
        shard_results.append(json.loads(path.read_text(encoding="utf-8")))
    turns = merge_turn_results(conversations, shard_results)
    grouped = turn_results_to_abcd_predictions(turns, conversations)
    abcd_records = [{
        "conversation_id": prediction.conversation_id,
        "turns": [{
            "turn_index": turn.turn_index,
            "turn_type": turn.turn_type,
            "predicted_utterance_id": turn.predicted_utterance_id,
            "predicted_action": turn.predicted_action,
            "predicted_slots": turn.predicted_slots,
        } for turn in prediction.turns],
    } for prediction in grouped]
    text_by_id = {
        str(row.get("convo_id", "")): row.get("prediction", "")
        for row in turns if row.get("target_type", "utterance") == "utterance"
    }
    text_records = [{
        "dialogue_id": f"abcd-{conversation.get('convo_id', '?')}",
        "response_text": text_by_id.get(str(conversation.get("convo_id", "")), ""),
    } for conversation in conversations]
    result = evaluate_abcd_bundle(
        conversations,
        text_records=text_records,
        abcd_records=abcd_records,
        text_prediction_key="response_text",
    )
    # The root resource directory may contain generation usage from a prior
    # --skip-final-test run; every shard is testing-only.
    existing_summary = _read_usage(args.output_dir / "summary.json")
    generation = existing_summary.get("llm_usage") if existing_summary else _read_usage(args.output_dir / "llm_usage.json")
    if isinstance(generation, dict) and isinstance(generation.get("generation"), dict):
        generation = generation["generation"]
    shard_testing = []
    for path in sorted(args.shard_root.glob("shard_*/llm_usage.json")):
        usage = _read_usage(path)
        if isinstance(usage.get("testing"), dict):
            usage = usage["testing"]
        if usage:
            shard_testing.append(usage)
    usage = split_usage_summary(generation, merge_usage_summaries(*shard_testing))
    result["llm_usage"] = usage
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "turn_predictions.json").write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "abcd_predictions.json").write_text(json.dumps(abcd_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "llm_usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps({
        "config": {"method": args.method, "subflow": args.subflow, "evaluation": "sharded"},
        "data": {"test_sessions": len(conversations)},
        "final_test": result,
        "llm_usage": usage,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.get("summary", result), ensure_ascii=False))


if __name__ == "__main__":
    main()
