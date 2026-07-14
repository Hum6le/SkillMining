#!/usr/bin/env python3
r"""ABCD full experiment: intent split -> seed baseline -> training -> evaluation.

Flow:
  1. Split train/test by subflow.
  2. Run a seed baseline with an empty workflow.
  3. Train AWM in batches and induce workflows.
  4. Evaluate the trained workflow turn by turn.
  5. Compare seed vs trained.

Usage:
  python scripts/run_full_experiment.py
  python scripts/run_full_experiment.py --max-train-convs 200
  python scripts/run_full_experiment.py --skip-training
  python scripts/run_full_experiment.py --resume-from <checkpoint_dir>
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
from eval_tod.abcd.split import split_by_subflow, extract_all_agent_turns
from eval_tod.abcd.agent import compute_ast_from_turn_results
from eval_tod.cli import evaluate_text_records

ABCD_DIR = "data/eval/abcd/data"
BATCH_SIZE = 20
MAX_BATCHES = None
CHECKPOINT_EVERY = 10
MODEL = "deepseek-chat"
TRAIN_FRAC = 0.8
SEED = 42

_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = Path(f"outputs/full_experiment_{_TIMESTAMP}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "experiment.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def evaluate_turn_results(
    turn_results: list[dict],
    conversations: list[dict],
    label: str,
) -> dict:
    """Run text + AST/CDS metrics on turn-level predictions."""
    # 鈹€鈹€ Text metrics 鈹€鈹€
    preds = [r["prediction"] for r in turn_results]
    refs = [r["reference"] for r in turn_results]
    text_result = evaluate_text_records(preds, refs)

    # Per-subflow text
    by_sf: dict[str, dict[str, list]] = defaultdict(lambda: {"preds": [], "refs": []})
    for r in turn_results:
        sf = r.get("subflow", "unknown")
        by_sf[sf]["preds"].append(r["prediction"])
        by_sf[sf]["refs"].append(r["reference"])

    per_subflow = {}
    for sf, d in sorted(by_sf.items()):
        if len(d["preds"]) < 2:
            continue
        r = evaluate_text_records(d["preds"], d["refs"])
        per_subflow[sf] = {
            "n": len(d["preds"]),
            "bert_f1": round(r["bert_f1"], 4),
            "bleu_1": round(r["bleu_1"], 1),
            "bleu_4": round(r["bleu_4"], 1),
            "rouge_1": round(r["rouge_1"], 4),
            "rouge_2": round(r["rouge_2"], 4),
            "rouge_l": round(r["rouge_l"], 4),
            "meteor": round(r["meteor"], 4),
        }

    result = {
        "label": label,
        "n": len(preds),
        "text": {
            "bert_f1": round(text_result["bert_f1"], 4),
            "bleu_1": round(text_result["bleu_1"], 1),
            "bleu_4": round(text_result["bleu_4"], 1),
            "rouge_1": round(text_result["rouge_1"], 4),
            "rouge_2": round(text_result["rouge_2"], 4),
            "rouge_l": round(text_result["rouge_l"], 4),
            "meteor": round(text_result["meteor"], 4),
        },
        "per_subflow": per_subflow,
    }

    # 鈹€鈹€ AST / CDS (if action predictions available) 鈹€鈹€
    if any("predicted_action" in r for r in turn_results):
        try:
            from eval_tod.abcd.agent import turn_results_to_abcd_predictions
            from eval_tod.abcd.data import extract_ground_truth
            from eval_tod.abcd.metrics import evaluate_abcd

            abcd_preds = turn_results_to_abcd_predictions(turn_results, conversations)
            all_gt = [extract_ground_truth(c) for c in conversations]
            abcd_eval = evaluate_abcd(all_gt, abcd_preds)

            result["ast_cds"] = {
                "ast_joint": round(abcd_eval.ast.joint_accuracy, 4),
                "ast_action_name": round(abcd_eval.ast.action_name_accuracy, 4),
                "ast_slot_value": round(abcd_eval.ast.slot_value_accuracy, 4),
                "cds_overall": round(abcd_eval.cds.overall_cds, 4),
                "num_action_turns": abcd_eval.ast.total_action_turns,
            }
        except Exception as e:
            result["ast_cds"] = {"error": str(e)}

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ABCD Full Experiment")
    parser.add_argument("--max-train-convs", type=int, default=None)
    parser.add_argument("--max-test-convs", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "dev", "test"],
                        help="Single split: load + split by subflow (quick test)")
    parser.add_argument("--train-file", type=str, default=None,
                        help="Pre-split train convs JSON (from split_abcd_by_intent.py)")
    parser.add_argument("--test-file", type=str, default=None,
                        help="Pre-split test convs JSON (from split_abcd_by_intent.py)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=MAX_BATCHES)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-seed", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("FULL EXPERIMENT: Intent Split -> Seed -> Train -> Eval")
    log.info("=" * 55)

    # 1. Load + Split data
    if args.train_file and args.test_file:
        log.info("\n[1/4] Loading pre-split data...")
        train_convs = json.loads(Path(args.train_file).read_text(encoding="utf-8"))
        test_convs = json.loads(Path(args.test_file).read_text(encoding="utf-8"))
        log.info(f"  Train: {len(train_convs)} convs (from {args.train_file})")
        log.info(f"  Test:  {len(test_convs)} convs (from {args.test_file})")
    elif args.split:
        log.info("\n[1/4] Loading ABCD '%s' + splitting by subflow...", args.split)
        all_convs = load_abcd_data(args.split, ABCD_DIR)
        train_convs, test_convs = split_by_subflow(all_convs, args.train_frac, args.seed)
        log.info(f"  {len(train_convs)} train / {len(test_convs)} test  "
                 f"(from {len(all_convs)} '{args.split}' convs, train_frac={args.train_frac})")
    else:
        log.info("\n[1/4] Loading ABCD official train + test splits...")
        train_convs = load_abcd_data("train", ABCD_DIR)
        test_convs = load_abcd_data("test", ABCD_DIR)
        log.info(f"  Train: {len(train_convs)} convs, Test: {len(test_convs)} convs")

    if args.max_train_convs:
        train_convs = train_convs[:args.max_train_convs]
    if args.max_test_convs:
        test_convs = test_convs[:args.max_test_convs]
    log.info(f"  Train: {len(train_convs)} convs, Test: {len(test_convs)} convs")

    # Extract turns for evaluation
    test_turns = extract_all_agent_turns(test_convs)
    log.info(f"  Test turns: {len(test_turns)}")

    # Save split info
    split_info = {
        "train_convs": len(train_convs), "test_convs": len(test_convs),
        "train_turns": len(extract_all_agent_turns(train_convs)),
        "test_turns": len(test_turns),
        "test_subflows": list(set(t.subflow for t in test_turns)),
    }
    (OUT_DIR / "split_info.json").write_text(json.dumps(split_info, indent=2, ensure_ascii=False))

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?    # 2. Seed Baseline
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?    seed_results = None
    if not args.skip_seed:
        log.info("\n[2/4] Seed Baseline (no workflow, no memory)...")
        from eval_tod.abcd.agent import ABCDAgent
        from awm import MemoryStore, WorkflowStore

        seed_agent = ABCDAgent(
            model=args.model,
            workflow=WorkflowStore(), memory=MemoryStore(),
        )
        seed_turns = seed_agent.generate_all_turn_predictions(
            test_convs, predict_actions=True)
        seed_results = evaluate_turn_results(seed_turns, test_convs, "seed")

        _s = seed_results
        log.info(f"  Seed: BERT-F1={_s['text']['bert_f1']:.4f}  BLEU-4={_s['text']['bleu_4']:.1f}  "
                 f"ROUGE-L={_s['text']['rouge_l']:.4f}  METEOR={_s['text']['meteor']:.4f}"
                 + (f"  AST={_s['ast_cds']['ast_joint']:.4f}  CDS={_s['ast_cds']['cds_overall']:.4f}"
                    if 'ast_cds' in _s and 'error' not in _s['ast_cds'] else ""))

        # Save
        (OUT_DIR / "seed_predictions.json").write_text(
            json.dumps(seed_turns, indent=2, ensure_ascii=False), encoding="utf-8")
        (OUT_DIR / "seed_eval.json").write_text(
            json.dumps(seed_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?    # 3. Training
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?    trained_results = None
    if not args.skip_training:
        log.info("\n[3/4] AWM Batch Training...")
        from eval_tod.abcd.agent import ABCDAgent
        from eval_tod.response_logger import ResponseLogger
        from awm import MemoryStore, WorkflowStore

        logger = ResponseLogger(str(OUT_DIR / "llm_responses"))
        workflow = WorkflowStore()
        memory = MemoryStore()

        # Resume from checkpoint
        start_batch = 1
        if args.resume_from:
            import re
            ckpt_path = Path(args.resume_from)
            if not ckpt_path.exists():
                log.error(f"Checkpoint not found: {ckpt_path}")
                sys.exit(1)
            state_path = ckpt_path / "state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                start_batch = state.get("next_batch", 1)
            else:
                m = re.search(r"batch_(\d+)", ckpt_path.name)
                start_batch = int(m.group(1)) + 1 if m else 1
            wf_path = ckpt_path / "workflow.txt"
            if wf_path.exists():
                workflow.load(str(wf_path))
            mem_path = ckpt_path / "exemplars.json"
            if mem_path.exists():
                memory.load(str(mem_path))
            log.info(f"  Resumed from {ckpt_path}: starting batch {start_batch}")

        agent = ABCDAgent(
            model=args.model, workflow=workflow, memory=memory,
            response_logger=logger,
        )

        # Build batches
        batches = [train_convs[i:i + args.batch_size]
                   for i in range(0, len(train_convs), args.batch_size)]
        if args.max_batches:
            batches = batches[:args.max_batches]
        log.info(f"  Batches: {len(batches)} (batch_size={args.batch_size})")

        for batch_idx, batch in enumerate(batches, start=1):
            if batch_idx < start_batch:
                continue

            log.info(f"  Batch {batch_idx}/{len(batches)}: {len(batch)} dialogues")

            # Turn-level predictions with actions -> real AST scores
            turn_results = agent.generate_all_turn_predictions(
                batch, predict_actions=True, verbose=False)
            eval_dicts = compute_ast_from_turn_results(batch, turn_results)

            # Build last-turn Predictions from turn_results for induce/update_memory
            from eval_tod.schemas import Prediction
            last_turn_preds = []
            for conv in batch:
                cid = str(conv.get("convo_id", "?"))
                conv_turns = [r for r in turn_results if r["convo_id"] == cid]
                last = conv_turns[-1]["prediction"] if conv_turns else ""
                last_turn_preds.append(Prediction(
                    dialogue_id=f"abcd-{cid}",
                    inform_slots={}, request_slots={}, booking={},
                    response_text=last,
                ))

            agent.induce(batch, last_turn_preds, eval_dicts)
            agent.update_memory(batch, last_turn_preds, eval_dicts)

            # Checkpoint
            if batch_idx % CHECKPOINT_EVERY == 0:
                ckpt_dir = OUT_DIR / "checkpoints" / f"batch_{batch_idx:04d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                agent.save_workflow(str(ckpt_dir / "workflow.txt"))
                agent.save_memory(str(ckpt_dir / "exemplars.json"))
                (ckpt_dir / "state.json").write_text(json.dumps({
                    "next_batch": batch_idx + 1, "total_batches": len(batches),
                }))
                log.info(f"    Checkpoint: {ckpt_dir}")

        # Save trained state
        agent.save_workflow(str(OUT_DIR / "awm_workflow.txt"))
        agent.save_memory(str(OUT_DIR / "awm_exemplars.json"))

        # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?        # 4. Trained Evaluation
        # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?        log.info("\n[4/4] Trained Evaluation (with workflow + memory)...")
        log.info(f"  Workflow: {len(agent.workflow)} lines, Memory: {len(agent.memory)} exemplars")
        # Log a sample of the workflow to verify it was loaded
        wf_sample = agent.workflow.text[:200] if agent.workflow else "(empty)"
        log.info(f"  Workflow sample: {wf_sample}...")
        trained_turns = agent.generate_all_turn_predictions(
            test_convs, predict_actions=True)
        # Debug: check a few action predictions
        actions_found = sum(1 for r in trained_turns if r.get("predicted_action"))
        log.info(f"  Turns with predicted_action: {actions_found}/{len(trained_turns)}")
        if actions_found > 0:
            sample = next(r for r in trained_turns if r.get("predicted_action"))
            log.info(f"  Sample: turn={sample['turn_index']} "
                     f"action={sample.get('predicted_action','?')} "
                     f"resp={sample.get('prediction','')[:80]}")
        trained_results = evaluate_turn_results(trained_turns, test_convs, "trained")

        _t = trained_results
        log.info(f"  Trained: BERT-F1={_t['text']['bert_f1']:.4f}  BLEU-4={_t['text']['bleu_4']:.1f}  "
                 f"ROUGE-L={_t['text']['rouge_l']:.4f}  METEOR={_t['text']['meteor']:.4f}"
                 + (f"  AST={_t['ast_cds']['ast_joint']:.4f}  CDS={_t['ast_cds']['cds_overall']:.4f}"
                    if 'ast_cds' in _t and 'error' not in _t['ast_cds'] else ""))

        (OUT_DIR / "trained_predictions.json").write_text(
            json.dumps(trained_turns, indent=2, ensure_ascii=False), encoding="utf-8")
        (OUT_DIR / "trained_eval.json").write_text(
            json.dumps(trained_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5. Comparison
    print(f"\n{'='*55}")
    print(f"EXPERIMENT RESULTS")
    print(f"{'='*55}")

    if seed_results:
        st = seed_results["text"]
        print(f"\n  Seed Baseline (no training):")
        print(f"    BERT-F1:  {st['bert_f1']:.4f}")
        print(f"    BLEU-1:   {st['bleu_1']:.1f}")
        print(f"    BLEU-4:   {st['bleu_4']:.1f}")
        print(f"    ROUGE-1:  {st['rouge_1']:.4f}")
        print(f"    ROUGE-2:  {st['rouge_2']:.4f}")
        print(f"    ROUGE-L:  {st['rouge_l']:.4f}")
        print(f"    METEOR:   {st['meteor']:.4f}")
        if "ast_cds" in seed_results and "error" not in seed_results["ast_cds"]:
            ac = seed_results["ast_cds"]
            print(f"    AST:      {ac['ast_joint']:.4f}")
            print(f"    CDS:      {ac['cds_overall']:.4f}")

    if trained_results:
        tt = trained_results["text"]
        print(f"\n  After Training:")
        print(f"    BERT-F1:  {tt['bert_f1']:.4f}")
        print(f"    BLEU-1:   {tt['bleu_1']:.1f}")
        print(f"    BLEU-4:   {tt['bleu_4']:.1f}")
        print(f"    ROUGE-1:  {tt['rouge_1']:.4f}")
        print(f"    ROUGE-2:  {tt['rouge_2']:.4f}")
        print(f"    ROUGE-L:  {tt['rouge_l']:.4f}")
        print(f"    METEOR:   {tt['meteor']:.4f}")
        if "ast_cds" in trained_results and "error" not in trained_results["ast_cds"]:
            ac = trained_results["ast_cds"]
            print(f"    AST:      {ac['ast_joint']:.4f}")
            print(f"    CDS:      {ac['cds_overall']:.4f}")

    if seed_results and trained_results:
        delta_bert = trained_results["text"]['bert_f1'] - seed_results["text"]['bert_f1']
        delta_bleu = trained_results["text"]['bleu_4'] - seed_results["text"]['bleu_4']
        delta_rouge = trained_results["text"]['rouge_l'] - seed_results["text"]['rouge_l']
        delta_meteor = trained_results["text"]['meteor'] - seed_results["text"]['meteor']
        print(f"\n  Delta (Trained - Seed):")
        print(f"    BERT-F1:  {delta_bert:+.4f}")
        print(f"    BLEU-4:   {delta_bleu:+.1f}")
        print(f"    ROUGE-L:  {delta_rouge:+.4f}")
        print(f"    METEOR:   {delta_meteor:+.4f}")
        if "ast_cds" in seed_results and "ast_cds" in trained_results:
            sa = seed_results["ast_cds"]; ta = trained_results["ast_cds"]
            if "error" not in sa and "error" not in ta:
                print(f"    AST:      {ta['ast_joint'] - sa['ast_joint']:+.4f}")
                print(f"    CDS:      {ta['cds_overall'] - sa['cds_overall']:+.4f}")

    # Per-subflow comparison
    if seed_results and trained_results:
        all_sf = set(seed_results.get("per_subflow", {})) | set(trained_results.get("per_subflow", {}))
        print(f"\n  Per-Subflow Delta BERT-F1:")
        sf_deltas = []
        for sf in sorted(all_sf):
            seed_bert = seed_results.get("per_subflow", {}).get(sf, {}).get("bert_f1", 0)
            train_bert = trained_results.get("per_subflow", {}).get(sf, {}).get("bert_f1", 0)
            if seed_bert > 0 and train_bert > 0:
                sf_deltas.append((sf, seed_bert, train_bert, train_bert - seed_bert))
        sf_deltas.sort(key=lambda x: -x[3])
        for sf, seed_b, train_b, delta in sf_deltas[:10]:
            print(f"    {sf:35s}  {seed_b:.4f} -> {train_b:.4f}  ({delta:+.4f})")

    # Save summary
    summary = {
        "config": vars(args),
        "split": split_info,
        "seed": seed_results,
        "trained": trained_results,
    }
    (OUT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"DONE. Output: {OUT_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()

