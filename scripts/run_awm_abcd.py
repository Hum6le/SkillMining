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
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Config ────────────────────────────────────────────────────
ABCD_DIR = "data/eval/abcd/data"
BATCH_SIZE = 20             # dialogues per batch
MAX_BATCHES = None          # None = all batches
VAL_EVERY = 5               # validate every N batches
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
    from eval_tod.abcd.data import load_abcd_data
    from eval_tod.abcd.agent import ABCDAgent
    from eval_tod.response_logger import ResponseLogger
    from eval_tod import evaluate_all
    from eval_tod.schemas import Prediction
    from awm import MemoryStore, WorkflowStore

    # ── Load data ─────────────────────────────────────────────
    log.info("Loading ABCD dataset...")
    train_convs = load_abcd_data("train", ABCD_DIR)
    dev_convs = load_abcd_data("dev", ABCD_DIR)
    test_convs = load_abcd_data("test", ABCD_DIR)
    log.info(f"Train: {len(train_convs)}, Dev: {len(dev_convs)}, Test: {len(test_convs)}")

    # ── Build batches ─────────────────────────────────────────
    batches = _build_batches(train_convs, BATCH_SIZE, MAX_BATCHES)
    log.info(f"Batches: {len(batches)} (batch_size={BATCH_SIZE})")

    # ── Init agent + memory ───────────────────────────────────
    logger = ResponseLogger(str(OUT_DIR / "llm_responses"))
    workflow = WorkflowStore()
    memory = MemoryStore()

    agent = ABCDAgent(
        model=MODEL, workflow=workflow, memory=memory,
        response_logger=logger,
    )

    # ── Seed baseline on dev (cold start) ─────────────────────
    log.info("=" * 50)
    log.info("Seed baseline on dev set (no workflow/memory)")
    seed_agent = ABCDAgent(
        model=MODEL,
        workflow=WorkflowStore(), memory=MemoryStore(),
        response_logger=logger,
    )
    seed_preds = seed_agent.generate_predictions(dev_convs)
    seed_val = evaluate_all(dev_convs, seed_preds, dataset_name="abcd")
    log.info(f"Seed dev: {seed_val['summary']}")

    # ── Batch training loop ───────────────────────────────────
    batch_metrics = []
    val_history = [{"label": "seed", **seed_val.get("text", {}), **seed_val.get("ast_cds", {})}]

    for batch_idx, batch in enumerate(batches, start=1):
        log.info(f"{'─'*40}")
        log.info(f"Batch {batch_idx}/{len(batches)}: {len(batch)} dialogues")

        # 1. Run agent
        preds = agent.generate_predictions(batch)

        # 2. Evaluate
        result = evaluate_all(batch, preds, dataset_name="abcd")
        batch_metrics.append({"batch": batch_idx, "summary": result.get("summary", "")})
        log.info(f"  {result['summary']}")

        # 3. Build per-dialogue metrics for induction + memory
        text_metrics = result.get("text", {})
        per_sample = text_metrics.get("per_sample", []) if isinstance(text_metrics, dict) else []
        eval_dicts = [
            {"bert_f1": s.get("bert_f1", 0)} if isinstance(s, dict) else {"bert_f1": 0}
            for s in per_sample
        ]
        # Pad if per_sample is shorter than batch
        while len(eval_dicts) < len(batch):
            eval_dicts.append({"bert_f1": 0})

        # 4. Induce workflow from this batch
        agent.induce(batch, preds, eval_dicts)

        # 5. Update memory with good responses
        agent.update_memory(batch, preds, eval_dicts)

        # 7. Checkpoint
        if batch_idx % CHECKPOINT_EVERY == 0:
            ckpt_dir = OUT_DIR / "checkpoints" / f"batch_{batch_idx:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save_workflow(str(ckpt_dir / "workflow.txt"))
            agent.save_memory(str(ckpt_dir / "exemplars.json"))
            log.info(f"  Checkpoint saved: {ckpt_dir}")

        # 5. Validation
        if batch_idx % VAL_EVERY == 0:
            val_agent = ABCDAgent(
                model=MODEL, workflow=workflow, memory=memory,
                response_logger=logger,
            )
            val_preds = val_agent.generate_predictions(dev_convs)
            val_result = evaluate_all(dev_convs, val_preds, dataset_name="abcd")
            val_history.append({
                "label": f"batch_{batch_idx}",
                **(val_result.get("text") or {}),
                **(val_result.get("ast_cds") or {}),
            })
            log.info(f"  Val: {val_result['summary']}")

    # ── Final test evaluation ──────────────────────────────────
    log.info("=" * 50)
    log.info("Final test evaluation")
    test_agent = ABCDAgent(
        model=MODEL, workflow=workflow, memory=memory,
        response_logger=logger,
    )
    test_preds = test_agent.predict_and_save(test_convs, str(OUT_DIR / "test_final_preds.json"))
    test_result = evaluate_all(test_convs, test_preds, dataset_name="abcd")
    log.info(f"Final test: {test_result['summary']}")

    # ── Save everything ───────────────────────────────────────
    agent.save_workflow(str(OUT_DIR / "awm_workflow.txt"))
    agent.save_memory(str(OUT_DIR / "awm_exemplars.json"))

    summary = {
        "config": {
            "batch_size": BATCH_SIZE, "max_batches": MAX_BATCHES,
            "val_every": VAL_EVERY, "checkpoint_every": CHECKPOINT_EVERY,
            "model": MODEL, "seed": SEED,
            "dataset": "abcd", "eval_mode": "all",
        },
        "data": {
            "train": len(train_convs), "dev": len(dev_convs),
            "test": len(test_convs), "batches": len(batches),
        },
        "seed_val": seed_val.get("text", {}),
        "final_test": test_result.get("text", {}),
        "batch_metrics": batch_metrics,
        "val_history": val_history,
        "workflow_lines": len(workflow),
        "memory_exemplars": len(memory),
        "llm_calls_logged": logger.count,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("=" * 50)
    log.info(f"DONE. Output: {OUT_DIR}")
    log.info(f"Seed dev:  {seed_val.get('summary', '')}")
    log.info(f"Final test: {test_result.get('summary', '')}")
    log.info(f"LLM calls logged: {logger.count}")
    return summary


if __name__ == "__main__":
    main()
