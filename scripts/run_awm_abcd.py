#!/usr/bin/env python3
"""AWM full training run on ABCD dataset — generative end-to-end.

Usage:
    python scripts/run_awm_abcd.py

What it does:
    1. Load ABCD train/dev/test splits
    2. Batch-train ABCDAgent with iterative workflow induction
    3. After each batch: evaluate → induce workflows → update memory
    4. Periodic validation on held-out set
    5. Final evaluation on test set (all metrics)
    6. Save all outputs to outputs/awm_abcd_{timestamp}/
"""

import json
import logging
import random
import re
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from eval_tod.cli import evaluate_abcd_bundle

# ── Config ────────────────────────────────────────────────────
ABCD_DIR = "data/eval/abcd/data"
BATCH_SIZE = 20             # dialogues per batch
MAX_BATCHES = None          # None = all batches
CHECKPOINT_EVERY = 10       # save workflow/memory checkpoint every N batches
MODEL = "deepseek-chat"
SEED = 42

# ── Setup ─────────────────────────────────────────────────────
_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = Path(f"outputs/awm_abcd_{_TIMESTAMP}")
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


def _build_batches(items: list, size: int, max_batches: int | None = None) -> list[list]:
    """Split items into fixed-size batches."""
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    if max_batches:
        batches = batches[:max_batches]
    return batches


def main():
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="AWM training on ABCD")
    _parser.add_argument("--resume-from", type=str, default=None,
                         help="Resume from a checkpoint directory (e.g. outputs/awm_abcd_xxx/checkpoints/batch_0050)")
    _parser.add_argument(
        "--subflows",
        default="",
        help="Comma-separated subflows to mix; empty means all ABCD subflows",
    )
    _parser.add_argument(
        "--mixed-subflows",
        action="store_true",
        help="Mix selected subflows and hide flow/subflow labels from the agent",
    )
    _parser.add_argument(
        "--hide-subflow",
        action="store_true",
        help="Hide flow/subflow labels from the agent prompt",
    )
    _parser.add_argument("--max-train", type=int, default=None)
    _parser.add_argument("--max-dev", type=int, default=None)
    _parser.add_argument("--max-test", type=int, default=None)
    _args, _unknown = _parser.parse_known_args()

    from eval_tod.abcd.data import load_abcd_data
    from eval_tod.abcd.agent import ABCDAgent
    from eval_tod.response_logger import ResponseLogger
    from awm import MemoryStore, WorkflowStore

    # ── Load data ─────────────────────────────────────────────
    log.info("Loading ABCD dataset...")
    train_convs = load_abcd_data("train", ABCD_DIR)
    dev_convs = load_abcd_data("dev", ABCD_DIR)
    test_convs = load_abcd_data("test", ABCD_DIR)

    selected_subflows = {
        item.strip() for item in _args.subflows.split(",") if item.strip()
    }
    if selected_subflows:
        def keep(conv):
            return str(conv.get("scenario", {}).get("subflow", "")) in selected_subflows
        train_convs = [conv for conv in train_convs if keep(conv)]
        dev_convs = [conv for conv in dev_convs if keep(conv)]
        test_convs = [conv for conv in test_convs if keep(conv)]
    if _args.mixed_subflows:
        random.Random(SEED).shuffle(train_convs)
    if _args.max_train:
        train_convs = train_convs[:_args.max_train]
    if _args.max_dev:
        dev_convs = dev_convs[:_args.max_dev]
    if _args.max_test:
        test_convs = test_convs[:_args.max_test]
    log.info(f"Train: {len(train_convs)}, Dev: {len(dev_convs)}, Test: {len(test_convs)}")

    # ── Build batches ─────────────────────────────────────────
    batches = _build_batches(train_convs, BATCH_SIZE, MAX_BATCHES)
    log.info(f"Batches: {len(batches)} (batch_size={BATCH_SIZE})")
    run_config = {
        "subflows": sorted(selected_subflows),
        "mixed_subflows": bool(_args.mixed_subflows),
        "hide_subflow": bool(_args.hide_subflow),
        "max_train": _args.max_train,
        "max_dev": _args.max_dev,
        "max_test": _args.max_test,
        "batch_size": BATCH_SIZE,
        "max_batches": MAX_BATCHES,
    }

    # ── Init agent + memory ───────────────────────────────────
    logger = ResponseLogger(str(OUT_DIR / "llm_responses"))
    workflow = WorkflowStore()
    memory = MemoryStore()

    # ── Resume from checkpoint ────────────────────────────────
    start_batch = 1
    if _args.resume_from:
        ckpt_dir = Path(_args.resume_from)
        if not ckpt_dir.exists():
            log.error(f"Checkpoint not found: {ckpt_dir}")
            sys.exit(1)

        # Load state
        state_path = ckpt_dir / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            start_batch = state.get("next_batch", 1)
            previous_config = state.get("config", {})
            mismatches = [
                f"{key}: previous={previous_config[key]!r}, current={value!r}"
                for key, value in run_config.items()
                if key in previous_config and previous_config[key] != value
            ]
            if mismatches:
                raise ValueError(
                    "Checkpoint configuration does not match the current run:\n"
                    + "\n".join(f"- {item}" for item in mismatches)
                )
        else:
            # Infer from directory name: batch_0050 → 50
            m = re.search(r"batch_(\d+)", ckpt_dir.name)
            start_batch = int(m.group(1)) + 1 if m else 1

        # Load workflow + memory
        wf_path = ckpt_dir / "workflow.txt"
        if wf_path.exists():
            workflow.load(str(wf_path))
        mem_path = ckpt_dir / "exemplars.json"
        if mem_path.exists():
            memory.load(str(mem_path))

        log.info(f"Resumed from {ckpt_dir}: batch {start_batch}/{len(batches)}, "
                 f"workflow={len(workflow)} lines, memory={len(memory)} exemplars")

    agent = ABCDAgent(
        model=MODEL, workflow=workflow, memory=memory,
        expose_scenario_labels=not (_args.mixed_subflows or _args.hide_subflow),
        response_logger=logger,
    )

    # ── Batch training loop ───────────────────────────────────

    for batch_idx, batch in enumerate(batches, start=1):
        if batch_idx < start_batch:
            continue  # skip already processed batches

        log.info(f"{'─'*40}")
        log.info(f"Batch {batch_idx}/{len(batches)}: {len(batch)} dialogues")

        # 1. Run agent (turn-level) + compute real AST
        from eval_tod.abcd.agent import compute_ast_from_turn_results
        from eval_tod.schemas import Prediction
        turn_results = agent.generate_all_turn_predictions(
            batch, predict_actions=True, verbose=False)
        eval_dicts = compute_ast_from_turn_results(batch, turn_results)
        # Use last-turn predictions for induce
        preds = []
        for conv in batch:
            cid = str(conv.get("convo_id", "?"))
            conv_turns = [r for r in turn_results if r["convo_id"] == cid]
            last = conv_turns[-1]["prediction"] if conv_turns else ""
            preds.append(Prediction(
                dialogue_id=f"abcd-{cid}",
                inform_slots={}, request_slots={}, booking={},
                response_text=last,
            ))
        agent.induce(batch, preds, eval_dicts)
        agent.update_memory(batch, preds, eval_dicts)

        # 3. Checkpoint (with state for resume)
        if batch_idx % CHECKPOINT_EVERY == 0:
            ckpt_dir = OUT_DIR / "checkpoints" / f"batch_{batch_idx:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save_workflow(str(ckpt_dir / "workflow.txt"))
            agent.save_memory(str(ckpt_dir / "exemplars.json"))
            (ckpt_dir / "state.json").write_text(
                json.dumps({
                    "next_batch": batch_idx + 1,
                    "total_batches": len(batches),
                    "config": run_config,
                }),
                encoding="utf-8")
            log.info(f"  Checkpoint saved: {ckpt_dir}")

    # ── Final test evaluation ──────────────────────────────────
    log.info("=" * 50)
    log.info("Final test evaluation")
    test_agent = ABCDAgent(
        model=MODEL, workflow=workflow, memory=memory,
        expose_scenario_labels=not (_args.mixed_subflows or _args.hide_subflow),
        response_logger=logger,
    )

    # Save turn-level predictions with actions (for AST/CDS + error analysis)
    log.info("  Generating turn-level predictions with actions...")
    test_turns = test_agent.generate_all_turn_predictions(
        test_convs, predict_actions=True)
    (OUT_DIR / "test_turn_predictions.json").write_text(
        json.dumps(test_turns, indent=2, ensure_ascii=False),
        encoding="utf-8")
    log.info(f"  Saved {len(test_turns)} turn predictions")

    # Also save plain predictions for text metrics
    test_preds = test_agent.predict_and_save(test_convs, str(OUT_DIR / "test_final_preds.json"))
    text_records = [
        {
            "dialogue_id": pred.dialogue_id,
            "response_text": pred.response_text,
        }
        for pred in test_preds
    ]
    test_result = evaluate_abcd_bundle(
        test_convs,
        text_records=text_records,
        abcd_records=test_turns,
        text_prediction_key="response_text",
    )
    log.info(f"Final test: {test_result['summary']}")

    # ── Save everything ───────────────────────────────────────
    agent.save_workflow(str(OUT_DIR / "awm_workflow.txt"))
    agent.save_memory(str(OUT_DIR / "awm_exemplars.json"))

    summary = {
        "config": {
            "batch_size": BATCH_SIZE, "max_batches": MAX_BATCHES,
            "checkpoint_every": CHECKPOINT_EVERY,
            "model": MODEL, "seed": SEED,
            "dataset": "abcd", "eval_mode": "all",
            "subflows": sorted(selected_subflows),
            "mixed_subflows": bool(_args.mixed_subflows),
            "hide_subflow": bool(_args.hide_subflow),
            "max_train": _args.max_train, "max_dev": _args.max_dev,
            "max_test": _args.max_test,
        },
        "data": {
            "train": len(train_convs), "dev": len(dev_convs),
            "test": len(test_convs), "batches": len(batches),
        },
        "final_test": test_result,
        "workflow_lines": len(workflow),
        "memory_exemplars": len(memory),
        "llm_calls_logged": logger.count,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("=" * 50)
    log.info(f"DONE. Output: {OUT_DIR}")
    log.info(f"Final test: {test_result.get('summary', '')}")
    log.info(f"LLM calls logged: {logger.count}")
    return summary


if __name__ == "__main__":
    main()
