#!/usr/bin/env python3
"""Standalone evaluation from saved prediction JSON - no agent calls.

This script now delegates to the unified eval_tod CLI helpers so the
alignment rules match the main evaluation entry points.
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
from eval_tod.cli import evaluate_abcd_bundle

ABCD_DIR = "data/eval/abcd/data"


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate saved predictions (no agent calls)")
    parser.add_argument("--predictions", required=True,
                        help="Path to saved text predictions JSON")
    parser.add_argument("--split", default="test",
                        choices=["train", "dev", "test"])
    parser.add_argument("--dataset", default="abcd",
                        choices=["abcd"])
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--abcd-predictions", default=None,
                        help="Optional separate ABCD action prediction JSON")
    parser.add_argument("--text-prediction-key", default="response_text")
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
    text_records = json.loads(preds_path.read_text(encoding="utf-8"))
    if not isinstance(text_records, list):
        print("Error: predictions file must contain a JSON array")
        sys.exit(1)

    abcd_records = None
    if args.abcd_predictions:
        abcd_path = Path(args.abcd_predictions)
        if not abcd_path.exists():
            print(f"Error: {abcd_path} not found")
            sys.exit(1)
        print(f"Loading ABCD action predictions from {abcd_path}...")
        abcd_records = json.loads(abcd_path.read_text(encoding="utf-8"))
        if not isinstance(abcd_records, list):
            print("Error: abcd_predictions file must contain a JSON array")
            sys.exit(1)

    # Evaluate
    print(f"Evaluating {len(dialogues)} dialogues...")
    result = evaluate_abcd_bundle(
        dialogues,
        text_records=text_records,
        abcd_records=abcd_records,
        text_prediction_key=args.text_prediction_key,
    )

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
        print(f"    METEOR:     {text.get('meteor', 'N/A'):.4f}")

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
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
