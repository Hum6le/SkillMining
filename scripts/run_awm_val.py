#!/usr/bin/env python3
"""AWM validation-only run — load trained workflow + memory, evaluate on a split.

Usage:
    # Validate on val split with a trained checkpoint
    python scripts/run_awm_val.py --workflow outputs/awm_run_.../awm_workflow.txt \
        --memory outputs/awm_run_.../awm_exemplars.json --split val

    # Test on test split
    python scripts/run_awm_val.py --workflow outputs/awm_run_.../awm_workflow.txt \
        --memory outputs/awm_run_.../awm_exemplars.json --split test

    # Without workflow/memory (cold start baseline)
    python scripts/run_awm_val.py --split test
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Defaults ────────────────────────────────────────────────────
DATA_DIR = "data/eval/multiwoz21/splits"
KB_DIR = "data/eval/multiwoz21/data/data"
MAX_TURNS = 6
MODEL = "deepseek-chat"

# ── Setup ─────────────────────────────────────────────────────
_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = Path(f"outputs/awm_val_{_TIMESTAMP}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "run.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="AWM validation-only run")
    parser.add_argument("--workflow", default=None, help="Path to workflow .txt file")
    parser.add_argument("--memory", default=None, help="Path to exemplars .json file")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="Which split to evaluate on (default: val)")
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--kb_dir", default=KB_DIR)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max_turns", type=int, default=MAX_TURNS)
    parser.add_argument("--eval_mode", default="both", choices=["slot", "text", "both"],
                        help="Evaluation mode: slot (IR/SR only), text (BERTScore/BLEU/ROUGE), both")
    parser.add_argument("--output_dir", default=None,
                        help="Custom output dir (default: auto-generated under outputs/)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Imports ────────────────────────────────────────────────
    from eval_tod.data import load_multiwoz21
    from eval_tod.kb import MultiWOZKB
    from eval_tod import evaluate_predictions, print_summary
    from eval_tod.response_logger import ResponseLogger
    from awm import AWMAgent, MemoryStore, WorkflowStore

    # ── Load data ─────────────────────────────────────────────
    split_file = {
        "train": "all_train.json",
        "val": "all_val.json",
        "test": "all_test.json",
    }[args.split]

    log.info(f"Loading {args.split} split: {args.data_dir}/{split_file}")
    dialogues = load_multiwoz21(f"{args.data_dir}/{split_file}")
    log.info(f"Loaded {len(dialogues)} dialogues")

    # ── Load workflow & memory ────────────────────────────────
    workflow = WorkflowStore()
    if args.workflow:
        workflow.load(args.workflow)
        log.info(f"Loaded workflow: {len(workflow)} lines from {args.workflow}")
    else:
        log.info("No workflow provided — cold start")

    memory = MemoryStore()
    if args.memory:
        memory.load(args.memory)
        log.info(f"Loaded memory: {len(memory)} exemplars from {args.memory}")
    else:
        log.info("No memory provided — cold start")

    # ── Init agent ────────────────────────────────────────────
    kb = MultiWOZKB(args.kb_dir)
    logger = ResponseLogger(str(out_dir / "llm_responses"))

    agent = AWMAgent(
        kb=kb,
        workflow=workflow,
        memory=memory,
        model=args.model,
        max_turns=args.max_turns,
        response_logger=logger,
        log_dir=str(out_dir / "trajectories"),
    )

    # ── Predict ───────────────────────────────────────────────
    log.info(f"{'='*50}")
    log.info(f"Running prediction on {len(dialogues)} dialogues...")
    preds = agent.predict_and_save(dialogues, str(out_dir / "predictions.json"))

    # ── Evaluate ──────────────────────────────────────────────
    eval_mode = args.eval_mode
    text_result = None

    if eval_mode in ("slot", "both"):
        result = evaluate_predictions(dialogues, preds)
        agg = result["aggregate"]
        print_summary(result)
    else:
        result = {}
        agg = {}

    if eval_mode in ("text", "both"):
        from eval_tod.text_eval import evaluate_responses
        refs: list[str] = []
        ptexts: list[str] = []
        for dialogue, pred in zip(dialogues, preds):
            sys_utts = [t.utterance for t in dialogue.turns if t.speaker == "system"]
            resp = pred.response_text if pred.response_text else ""
            if resp and sys_utts:
                refs.append(sys_utts[-1])
                ptexts.append(resp)
        if ptexts:
            text_result = evaluate_responses(ptexts, refs)
            log.info(f"Text eval: BERT-F1={text_result.bert_f1:.4f}  "
                     f"BLEU-1={text_result.bleu_1:.1f}  BLEU-4={text_result.bleu_4:.1f}  "
                     f"ROUGE-1={text_result.rouge_1:.4f}  ROUGE-L={text_result.rouge_l:.4f}")

    # ── Save summary ──────────────────────────────────────────
    summary = {
        "config": {
            "eval_mode": eval_mode,
            "split": args.split,
            "model": args.model,
            "max_turns": args.max_turns,
            "workflow": args.workflow,
            "memory": args.memory,
            "workflow_lines": len(workflow),
            "memory_exemplars": len(memory),
        },
        "data": {"num_dialogues": len(dialogues)},
        "aggregate": agg,
        "per_dialogue": result.get("per_dialogue", []),
        "llm_calls_logged": logger.count,
    }
    if text_result is not None:
        summary["text_eval"] = {
            "bert_f1": text_result.bert_f1,
            "bleu_1": text_result.bleu_1, "bleu_4": text_result.bleu_4,
            "rouge_1": text_result.rouge_1, "rouge_l": text_result.rouge_l,
            "num_samples": text_result.num_samples,
        }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"{'='*50}")
    log.info(f"DONE. Output: {out_dir}")
    log.info(f"Split: {args.split}  Dialogues: {len(dialogues)}")
    if agg:
        log.info(f"IR={agg.get('info_rate', 0):.4f}  SR={agg.get('success_rate', 0):.4f}")
    if text_result is not None:
        log.info(f"Text: BERT-F1={text_result.bert_f1:.4f}  BLEU-4={text_result.bleu_4:.1f}  ROUGE-L={text_result.rouge_l:.4f}")
    log.info(f"LLM calls logged: {logger.count}")
    return summary


if __name__ == "__main__":
    main()
