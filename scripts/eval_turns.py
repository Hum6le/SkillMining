#!/usr/bin/env python3
r"""ABCD Turn-Level Evaluation — intent-based split + per-turn prediction + full metrics.

Evaluates EVERY agent turn in every dialogue (not just the last one).

Usage:
  # Full pipeline: split by subflow → predict all turns → evaluate
  python scripts/eval_turns.py --split test --max-convs 50

  # Load pre-computed turn predictions
  python scripts/eval_turns.py --load-preds outputs/eval_turns_xxx/turn_predictions.json
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from eval_tod.abcd.data import load_abcd_data
from eval_tod.abcd.split import (
    TurnSample,
    split_by_subflow,
    extract_all_agent_turns,
    summarise_split,
)
from eval_tod.text_eval import evaluate_responses, TextEvalResult

ABCD_DIR = "data/eval/abcd/data"
MODEL = "deepseek-chat"

_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = Path(f"outputs/eval_turns_{_TIMESTAMP}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "eval.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_turns(
    turn_results: list[dict],
    label: str = "test",
) -> dict:
    """Evaluate turn-level predictions with text metrics + per-subflow breakdown.

    Args:
        turn_results: List of dicts with keys: prediction, reference, subflow, agent_turn_num.
        label: Label for logging.

    Returns:
        Dict with overall + per_subflow + per_position metrics.
    """
    predictions = [r["prediction"] for r in turn_results]
    references = [r["reference"] for r in turn_results]

    log.info(f"Evaluating {len(predictions)} turn predictions ({label})...")
    text_result = evaluate_responses(predictions, references)
    log.info(f"  BERT-F1={text_result.bert_f1:.4f}  BLEU-4={text_result.bleu_4:.1f}  "
             f"ROUGE-L={text_result.rouge_l:.4f}  METEOR={text_result.meteor:.4f}")

    # Per-subflow breakdown
    by_subflow: dict[str, dict[str, list]] = defaultdict(lambda: {"preds": [], "refs": []})
    for r in turn_results:
        sf = r.get("subflow", "unknown")
        by_subflow[sf]["preds"].append(r["prediction"])
        by_subflow[sf]["refs"].append(r["reference"])

    per_subflow: dict[str, dict] = {}
    for sf, data in sorted(by_subflow.items()):
        if len(data["preds"]) < 2:
            continue
        sf_result = evaluate_responses(data["preds"], data["refs"])
        per_subflow[sf] = {
            "n": len(data["preds"]),
            "bert_f1": round(sf_result.bert_f1, 4),
            "bleu_1": round(sf_result.bleu_1, 1),
            "bleu_4": round(sf_result.bleu_4, 1),
            "rouge_1": round(sf_result.rouge_1, 4),
            "rouge_2": round(sf_result.rouge_2, 4),
            "rouge_l": round(sf_result.rouge_l, 4),
            "meteor": round(sf_result.meteor, 4),
        }

    # Per-position breakdown (1st agent turn, 2nd, ...)
    by_position: dict[int, dict[str, list]] = defaultdict(lambda: {"preds": [], "refs": []})
    for r in turn_results:
        pos = r.get("agent_turn_num", 1)
        by_position[pos]["preds"].append(r["prediction"])
        by_position[pos]["refs"].append(r["reference"])

    per_position: dict[int, dict] = {}
    for pos, data in sorted(by_position.items()):
        if len(data["preds"]) < 2:
            continue
        pos_result = evaluate_responses(data["preds"], data["refs"])
        per_position[pos] = {
            "n": len(data["preds"]),
            "bert_f1": round(pos_result.bert_f1, 4),
            "bleu_1": round(pos_result.bleu_1, 1),
            "bleu_4": round(pos_result.bleu_4, 1),
            "rouge_1": round(pos_result.rouge_1, 4),
            "rouge_2": round(pos_result.rouge_2, 4),
            "rouge_l": round(pos_result.rouge_l, 4),
            "meteor": round(pos_result.meteor, 4),
        }

    return {
        "label": label,
        "n_samples": len(predictions),
        "overall": {
            "bert_f1": round(text_result.bert_f1, 4),
            "bert_precision": round(text_result.bert_precision, 4),
            "bert_recall": round(text_result.bert_recall, 4),
            "bleu_1": round(text_result.bleu_1, 1),
            "bleu_4": round(text_result.bleu_4, 1),
            "rouge_1": round(text_result.rouge_1, 4),
            "rouge_2": round(text_result.rouge_2, 4),
            "rouge_l": round(text_result.rouge_l, 4),
            "meteor": round(text_result.meteor, 4),
        },
        "per_subflow": per_subflow,
        "per_position": per_position,
    }


def print_results(results: dict):
    """Pretty-print evaluation results."""
    ov = results["overall"]
    print(f"\n{'='*55}")
    print(f"TURN-LEVEL EVALUATION: {results['label']}")
    print(f"{'='*55}")
    print(f"  Samples:    {results['n_samples']}")
    print(f"  BERT-F1:    {ov['bert_f1']:.4f}")
    print(f"  BLEU-1:     {ov['bleu_1']:.1f}")
    print(f"  BLEU-4:     {ov['bleu_4']:.1f}")
    print(f"  ROUGE-1:    {ov['rouge_1']:.4f}")
    print(f"  ROUGE-2:    {ov['rouge_2']:.4f}")
    print(f"  ROUGE-L:    {ov['rouge_l']:.4f}")
    print(f"  METEOR:     {ov['meteor']:.4f}")

    # Per-position
    print(f"\n  Per Agent Turn Position:")
    for pos in sorted(results.get("per_position", {}).keys()):
        p = results["per_position"][pos]
        print(f"    Turn #{pos}:  n={p['n']:5d}  "
              f"BERT={p['bert_f1']:.4f}  BLEU-4={p['bleu_4']:.1f}  "
              f"ROUGE-L={p['rouge_l']:.4f}  METEOR={p['meteor']:.4f}")

    # Top/bottom subflows
    ps = results.get("per_subflow", {})
    if ps:
        sorted_sf = sorted(ps.items(), key=lambda x: -x[1]["bert_f1"])
        print(f"\n  Top 5 Subflows (by BERT-F1):")
        for sf, m in sorted_sf[:5]:
            print(f"    {sf:35s}  n={m['n']:4d}  BERT={m['bert_f1']:.4f}")
        print(f"\n  Bottom 5 Subflows:")
        for sf, m in sorted_sf[-5:]:
            print(f"    {sf:35s}  n={m['n']:4d}  BERT={m['bert_f1']:.4f}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ABCD Turn-Level Evaluation")
    parser.add_argument("--split", default="test",
                        choices=["train", "dev", "test"])
    parser.add_argument("--max-convs", type=int, default=None,
                        help="Max conversations to load")
    parser.add_argument("--max-test-convs", type=int, default=None,
                        help="Max test conversations (after split)")
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="Train fraction per subflow")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-preds", type=str, default=None,
                        help="Load pre-computed turn predictions JSON (skip agent)")
    parser.add_argument("--no-agent", action="store_true",
                        help="Skip agent prediction (just show split stats)")
    args = parser.parse_args()

    # ── 1. Load data ──────────────────────────────────────────
    log.info(f"Loading ABCD {args.split} split...")
    all_convs = load_abcd_data(args.split, ABCD_DIR)
    if args.max_convs:
        all_convs = all_convs[:args.max_convs]
    log.info(f"  {len(all_convs)} conversations")

    # ── 2. Split by subflow ───────────────────────────────────
    log.info(f"Splitting by subflow (train_frac={args.train_frac})...")
    train_convs, test_convs = split_by_subflow(all_convs, args.train_frac, args.seed)
    if args.max_test_convs:
        test_convs = test_convs[:args.max_test_convs]
    log.info(f"  Train: {len(train_convs)} convs, Test: {len(test_convs)} convs")

    # Turn-level extraction
    train_turns = extract_all_agent_turns(train_convs)
    test_turns = extract_all_agent_turns(test_convs)
    log.info(f"  Train turns: {len(train_turns)}, Test turns: {len(test_turns)}")

    # Print split summary
    summary = summarise_split(train_turns, test_turns)
    print("\n" + summary)
    (OUT_DIR / "split_summary.txt").write_text(summary, encoding="utf-8")

    if args.no_agent:
        log.info("--no-agent: stopping after split stats.")
        return

    # ── 3. Get turn predictions ───────────────────────────────
    turn_results: list[dict] = []

    if args.load_preds:
        log.info(f"Loading turn predictions from {args.load_preds}...")
        turn_results = json.loads(Path(args.load_preds).read_text(encoding="utf-8"))
        log.info(f"  {len(turn_results)} turn predictions loaded")
    else:
        from eval_tod.abcd.agent import ABCDAgent
        from awm import MemoryStore, WorkflowStore

        agent = ABCDAgent(model=args.model, workflow=WorkflowStore(), memory=MemoryStore())

        log.info(f"Predicting ALL turns for {len(test_convs)} test conversations...")
        turn_results = agent.generate_all_turn_predictions(test_convs)

        # Save
        preds_path = OUT_DIR / "turn_predictions.json"
        preds_path.write_text(json.dumps(turn_results, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        log.info(f"Turn predictions saved → {preds_path}")

    # ── 4. Evaluate ───────────────────────────────────────────
    results = evaluate_turns(turn_results, label=f"test_{args.split}")

    # Save
    (OUT_DIR / "eval_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print_results(results)
    log.info(f"Done. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
