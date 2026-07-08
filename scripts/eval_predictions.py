#!/usr/bin/env python3
"""Standalone evaluation from saved prediction JSON — no agent calls.

Usage:
  python scripts/eval_predictions.py \
    --predictions outputs/awm_abcd_xxx/test_final_preds.json \
    --split test

  python scripts/eval_predictions.py \
    --predictions outputs/abcd_hg_xxx/test_final_preds.json \
    --split test --dataset abcd
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from eval_tod.abcd.data import load_abcd_data
from eval_tod.schemas import Prediction
from eval_tod import evaluate_all

ABCD_DIR = "data/eval/abcd/data"


def load_predictions(path: str) -> list[Prediction]:
    """Load Prediction objects from a saved JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = []
    for item in data:
        preds.append(Prediction(
            dialogue_id=item.get("dialogue_id", ""),
            inform_slots=item.get("inform_slots", {}),
            request_slots=item.get("request_slots", {}),
            booking=item.get("booking", {}),
            response_text=item.get("response_text", ""),
        ))
    return preds


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate saved predictions (no agent calls)")
    parser.add_argument("--predictions", required=True,
                        help="Path to saved predictions JSON")
    parser.add_argument("--split", default="test",
                        choices=["train", "dev", "test"])
    parser.add_argument("--dataset", default="abcd",
                        choices=["abcd", "multiwoz"])
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    preds_path = Path(args.predictions)
    if not preds_path.exists():
        print(f"Error: {preds_path} not found")
        sys.exit(1)

    # Load data
    print(f"Loading ABCD {args.split} split...")
    dialogues = load_abcd_data(args.split, ABCD_DIR)
    if args.max_sessions:
        dialogues = dialogues[:args.max_sessions]

    # Load predictions
    print(f"Loading predictions from {preds_path}...")
    preds = load_predictions(str(preds_path))

    # Align: trim predictions to match dialogues
    if len(preds) > len(dialogues):
        print(f"  Trimming predictions: {len(preds)} → {len(dialogues)}")
        preds = preds[:len(dialogues)]
    elif len(preds) < len(dialogues):
        print(f"  Warning: {len(preds)} predictions < {len(dialogues)} dialogues")

    # Evaluate
    print(f"Evaluating {len(preds)} predictions on {len(dialogues)} dialogues...")
    result = evaluate_all(dialogues, preds, dataset_name=args.dataset)

    # Print results
    print(f"\n{'='*55}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*55}")

    text = result.get("text", {})
    if text and "error" not in text:
        print(f"\n  Text Metrics:")
        print(f"    BERT-F1:    {text.get('bert_f1', 'N/A'):.4f}")
        print(f"    BERT-P:     {text.get('bert_precision', 'N/A'):.4f}")
        print(f"    BERT-R:     {text.get('bert_recall', 'N/A'):.4f}")
        print(f"    BLEU-1:     {text.get('bleu_1', 'N/A'):.1f}")
        print(f"    BLEU-4:     {text.get('bleu_4', 'N/A'):.1f}")
        print(f"    ROUGE-1:    {text.get('rouge_1', 'N/A'):.4f}")
        print(f"    ROUGE-2:    {text.get('rouge_2', 'N/A'):.4f}")
        print(f"    ROUGE-L:    {text.get('rouge_l', 'N/A'):.4f}")

    ast_cds = result.get("ast_cds", {})
    if ast_cds and "error" not in ast_cds:
        print(f"\n  AST / CDS:")
        print(f"    AST Joint:       {ast_cds.get('ast_joint', 'N/A'):.4f}")
        print(f"    AST Action Name: {ast_cds.get('ast_action_name', 'N/A'):.4f}")
        print(f"    AST Slot Value:  {ast_cds.get('ast_slot_value', 'N/A'):.4f}")
        print(f"    CDS Overall:     {ast_cds.get('cds_overall', 'N/A'):.4f}")
        print(f"    Action Turns:    {ast_cds.get('num_action_turns', 'N/A')}")

    print(f"\n  Summary: {result.get('summary', 'N/A')}")
    print(f"{'='*55}")

    # Save
    out_path = preds_path.parent / f"eval_{preds_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
