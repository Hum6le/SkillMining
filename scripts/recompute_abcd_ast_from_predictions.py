#!/usr/bin/env python3
"""Recompute ABCD AST/CDS from saved predictions without LLM calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from eval_tod.abcd.agent import (
    _parse_action_response,
    normalize_action_name,
    build_action_vocab,
    turn_results_to_abcd_predictions,
)
from eval_tod.abcd.data import extract_ground_truth
from eval_tod.abcd.metrics import evaluate_abcd


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    data = _read(path)
    if not isinstance(data, list):
        raise ValueError(f"Prediction file must be a JSON list: {path}")
    if data and isinstance(data[0], dict) and "turns" in data[0]:
        rows: list[dict[str, Any]] = []
        for item in data:
            cid = str(item.get("conversation_id") or item.get("convo_id") or "")
            for turn in item.get("turns", []):
                row = dict(turn)
                row["convo_id"] = cid
                row.setdefault("target_type", "action")
                rows.append(row)
        return rows
    return [dict(row) for row in data if isinstance(row, dict)]


def _normalize_rows(rows: list[dict[str, Any]], vocab: set[str]) -> dict[str, Any]:
    parsed_from_raw = 0
    changed = 0
    nonempty = 0
    for row in rows:
        action = str(row.get("predicted_action") or "").strip()
        slots = row.get("predicted_slots")
        if not action and row.get("prediction"):
            action, parsed_slots, response = _parse_action_response(
                str(row["prediction"])
            )
            row["prediction"] = response
            if slots in (None, "", []):
                slots = parsed_slots
            parsed_from_raw += 1

        normalized = normalize_action_name(action, vocab)
        if action and normalized != action:
            changed += 1
        row["predicted_action"] = normalized
        row["predicted_slots"] = list(slots or [])
        if normalized:
            nonempty += 1

    return {
        "rows": len(rows),
        "parsed_from_raw_prediction": parsed_from_raw,
        "action_names_normalized": changed,
        "nonempty_actions": nonempty,
    }


def _serialize(predictions: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "conversation_id": pred.conversation_id,
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "turn_type": turn.turn_type,
                    "predicted_utterance_id": turn.predicted_utterance_id,
                    "predicted_action": turn.predicted_action,
                    "predicted_slots": turn.predicted_slots,
                }
                for turn in pred.turns
            ],
        }
        for pred in predictions
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute ABCD AST/CDS from saved turn predictions"
    )
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corrected-preds-output", default=None)
    args = parser.parse_args()

    conversations = _read(args.test_data)
    if not isinstance(conversations, list):
        raise ValueError("--test-data must contain a JSON list")

    rows = _load_rows(args.preds)
    vocab = build_action_vocab(conversations)
    normalization = _normalize_rows(rows, vocab)
    predictions = turn_results_to_abcd_predictions(rows, conversations)
    ground_truth = [extract_ground_truth(conv) for conv in conversations]
    result = evaluate_abcd(ground_truth, predictions)

    payload = {
        "test_data": str(Path(args.test_data).resolve()),
        "predictions": str(Path(args.preds).resolve()),
        "action_vocab_size": len(vocab),
        "normalization": normalization,
        "ast_cds": {
            "ast_joint": result.ast.joint_accuracy,
            "ast_action_name": result.ast.action_name_accuracy,
            "ast_slot_value": result.ast.slot_value_accuracy,
            "cds_overall": result.cds.overall_cds,
            "num_action_turns": result.ast.total_action_turns,
        },
        "summary": (
            f"AST={result.ast.joint_accuracy:.4f} "
            f"Action={result.ast.action_name_accuracy:.4f} "
            f"Slot={result.ast.slot_value_accuracy:.4f} "
            f"CDS={result.cds.overall_cds:.4f}"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.corrected_preds_output:
        corrected = Path(args.corrected_preds_output)
        corrected.parent.mkdir(parents=True, exist_ok=True)
        corrected.write_text(
            json.dumps(_serialize(predictions), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
