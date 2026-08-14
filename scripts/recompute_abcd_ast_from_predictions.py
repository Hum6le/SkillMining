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


def recompute_one(test_data_path: Path, preds_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conversations = _read(test_data_path)
    if not isinstance(conversations, list):
        raise ValueError(f"Test data must contain a JSON list: {test_data_path}")
    rows = _load_rows(preds_path)
    vocab = build_action_vocab(conversations)
    normalization = _normalize_rows(rows, vocab)
    predictions = turn_results_to_abcd_predictions(rows, conversations)
    ground_truth = [extract_ground_truth(conv) for conv in conversations]
    result = evaluate_abcd(ground_truth, predictions)
    payload = {
        "test_data": str(test_data_path.resolve()),
        "predictions": str(preds_path.resolve()),
        "action_vocab_size": len(vocab),
        "normalization": normalization,
        "ast_cds": {
            "ast_joint": result.ast.joint_accuracy,
            "ast_action_name": result.ast.action_name_accuracy,
            "ast_slot_value": result.ast.slot_value_accuracy,
            "cds_overall": result.cds.overall_cds,
            "num_action_turns": result.ast.total_action_turns,
        },
        "test_sessions": len(conversations),
        "summary": (
            f"AST={result.ast.joint_accuracy:.4f} "
            f"Action={result.ast.action_name_accuracy:.4f} "
            f"Slot={result.ast.slot_value_accuracy:.4f} "
            f"CDS={result.cds.overall_cds:.4f}"
        ),
    }
    return payload, _serialize(predictions)


def recompute_all(run_dir: Path, splits_dir: Path, pred_name: str) -> dict[str, Any]:
    """Recompute every subflow below a graph-mining output directory."""
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for preds_path in sorted(run_dir.glob(f"*/{pred_name}")):
        subflow = preds_path.parent.name
        test_data = splits_dir / subflow / "test.json"
        if not test_data.exists():
            missing.append(subflow)
            continue
        payload, _ = recompute_one(test_data, preds_path)
        records.append({
            "subflow": subflow,
            "predictions": str(preds_path.resolve()),
            "test_sessions": payload["test_sessions"],
            "action_turns": payload["ast_cds"]["num_action_turns"],
            "metrics": payload["ast_cds"],
            "normalization": payload["normalization"],
        })

    def weighted(metric: str, weight_key: str) -> float | None:
        values = [
            (float(row["metrics"][metric]), max(int(row[weight_key]), 1))
            for row in records if metric in row["metrics"]
        ]
        if not values:
            return None
        return sum(value * weight for value, weight in values) / sum(weight for _, weight in values)

    aggregate = {
        metric: weighted(metric, "action_turns")
        for metric in ("ast_joint", "ast_action_name", "ast_slot_value")
    }
    aggregate["cds_overall"] = weighted("cds_overall", "test_sessions")
    return {
        "mode": "all_subflows",
        "run_dir": str(run_dir.resolve()),
        "pred_name": pred_name,
        "num_subflows": len(records),
        "missing_subflows": missing,
        "records": records,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute ABCD AST/CDS from saved turn predictions"
    )
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--preds", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corrected-preds-output", default=None)
    parser.add_argument("--all", action="store_true", help="Recompute every subflow below --run-dir")
    parser.add_argument("--run-dir", default=None, help="Graph-mining output root containing one directory per subflow")
    parser.add_argument("--splits-dir", default="data/eval/abcd/splits")
    parser.add_argument("--pred-name", default="mined_predictions.json")
    args = parser.parse_args()

    if args.all:
        if not args.run_dir:
            parser.error("--all requires --run-dir")
        payload = recompute_all(Path(args.run_dir), Path(args.splits_dir), args.pred_name)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))
        return

    if not args.test_data or not args.preds:
        parser.error("single-subflow mode requires --test-data and --preds")

    payload, corrected_predictions = recompute_one(Path(args.test_data), Path(args.preds))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.corrected_preds_output:
        corrected = Path(args.corrected_preds_output)
        corrected.parent.mkdir(parents=True, exist_ok=True)
        corrected.write_text(
            json.dumps(corrected_predictions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
