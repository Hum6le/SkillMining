#!/usr/bin/env python3
"""AWM full training run on 1/10 split MultiWOZ dataset.

Usage:
    python scripts/run_awm.py

What it does:
    1. Load 1/10 train/val/test splits
    2. Batch-train AWM agent with iterative workflow induction
    3. After each batch: evaluate → induce workflows → update memory
    4. Periodic validation on held-out set
    5. Final evaluation on test set
    6. Save all outputs to outputs/awm_run_{timestamp}/
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
DATA_DIR = "data/eval/multiwoz21/splits"
KB_DIR = "data/eval/multiwoz21/data/data"
BATCH_SIZE = 20          # dialogues per batch
MAX_BATCHES = None       # None = all batches
VAL_EVERY = 5            # validate every N batches
CHECKPOINT_EVERY = 10    # save workflow/memory checkpoint every N batches
MAX_TURNS = 6            # ReAct loop turns
MODEL = "deepseek-chat"
SEED = 42

# ── Setup ─────────────────────────────────────────────────────
_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = Path(f"outputs/awm_run_{_TIMESTAMP}")
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
    from eval_tod.data import load_multiwoz21, build_batches
    from eval_tod.kb import MultiWOZKB
    from eval_tod import evaluate_predictions
    from eval_tod.response_logger import ResponseLogger
    from awm import AWMAgent, MemoryStore, WorkflowStore

    # ── Load data ─────────────────────────────────────────────
    log.info("Loading 1/10 splits...")
    train_dialogues = load_multiwoz21(f"{DATA_DIR}/all_train.json")
    val_dialogues = load_multiwoz21(f"{DATA_DIR}/all_val.json")
    test_dialogues = load_multiwoz21(f"{DATA_DIR}/all_test.json")
    log.info(f"Train: {len(train_dialogues)}, Val: {len(val_dialogues)}, Test: {len(test_dialogues)}")

    # ── Build batches ─────────────────────────────────────────
    batches = build_batches(train_dialogues, BATCH_SIZE, MAX_BATCHES)
    log.info(f"Batches: {len(batches)} (batch_size={BATCH_SIZE})")

    # ── Init agent + memory ───────────────────────────────────
    kb = MultiWOZKB(KB_DIR)
    logger = ResponseLogger(str(OUT_DIR / "llm_responses"))
    workflow = WorkflowStore()
    memory = MemoryStore()

    agent = AWMAgent(
        kb=kb, workflow=workflow, memory=memory,
        model=MODEL, max_turns=MAX_TURNS,
        response_logger=logger,
        log_dir=str(OUT_DIR / "trajectories"),
    )

    # ── Seed baseline on val (cold start, no workflow/memory) ──
    log.info("=" * 50)
    log.info("Seed baseline on validation set (no workflow/memory)")
    seed_agent = AWMAgent(
        kb=kb, workflow=WorkflowStore(), memory=MemoryStore(),
        model=MODEL, max_turns=MAX_TURNS,
        response_logger=logger, log_dir=str(OUT_DIR / "trajectories_seed"),
    )
    seed_preds = seed_agent.generate_predictions(val_dialogues)
    seed_val = evaluate_predictions(val_dialogues, seed_preds)
    log.info(f"Seed val: IR={seed_val['aggregate']['info_rate']:.4f}  "
             f"SR={seed_val['aggregate']['success_rate']:.4f}")

    # ── Batch training loop ───────────────────────────────────
    batch_metrics = []
    val_history = [{"label": "seed", **seed_val["aggregate"]}]

    for batch_idx, batch in enumerate(batches, start=1):
        log.info(f"{'─'*40}")
        log.info(f"Batch {batch_idx}/{len(batches)}: {len(batch)} dialogues")

        # 1. Run agent
        preds = agent.generate_predictions(batch)

        # 2. Evaluate
        result = evaluate_predictions(batch, preds)
        agg = result["aggregate"]
        batch_metrics.append({"batch": batch_idx, **agg})
        log.info(f"  IR={agg['info_rate']:.4f}  SR={agg['success_rate']:.4f}  "
                 f"success={agg['num_success']}/{agg['num_success']+agg['num_fail']}")

        # 3. Induce workflow from this batch
        agent.induce(batch, preds, result["per_dialogue"],
                     trajectory_dir=str(OUT_DIR / "trajectories"))

        # 4. Update memory with successes
        agent.update_memory(batch, preds, result["per_dialogue"])

        # 5. Checkpoint
        if batch_idx % CHECKPOINT_EVERY == 0:
            ckpt_dir = OUT_DIR / "checkpoints" / f"batch_{batch_idx:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save_workflow(str(ckpt_dir / "workflow.txt"))
            agent.save_memory(str(ckpt_dir / "exemplars.json"))
            log.info(f"  Checkpoint saved: {ckpt_dir}")

        # 6. Validation (use same workflow + memory as training agent)
        if batch_idx % VAL_EVERY == 0:
            val_agent = AWMAgent(
                kb=kb, workflow=workflow, memory=memory,
                model=MODEL, max_turns=MAX_TURNS,
                response_logger=logger, log_dir=str(OUT_DIR / f"trajectories_val_{batch_idx:04d}"),
            )
            val_preds = val_agent.generate_predictions(val_dialogues)
            val_result = evaluate_predictions(val_dialogues, val_preds)
            val_agg = val_result["aggregate"]
            val_history.append({"label": f"batch_{batch_idx}", **val_agg})
            delta_ir = val_agg["info_rate"] - seed_val["aggregate"]["info_rate"]
            delta_sr = val_agg["success_rate"] - seed_val["aggregate"]["success_rate"]
            log.info(f"  Val: IR={val_agg['info_rate']:.4f} (Δ{delta_ir:+.4f})  "
                     f"SR={val_agg['success_rate']:.4f} (Δ{delta_sr:+.4f})")

    # ── Final test evaluation ──────────────────────────────────
    log.info("=" * 50)
    log.info("Final test evaluation")
    test_agent = AWMAgent(
        kb=kb, workflow=workflow, memory=memory,
        model=MODEL, max_turns=MAX_TURNS,
        response_logger=logger, log_dir=str(OUT_DIR / "trajectories_test_final"),
    )
    test_preds = test_agent.predict_and_save(test_dialogues, str(OUT_DIR / "test_final_preds.json"))
    test_result = evaluate_predictions(test_dialogues, test_preds)
    test_agg = test_result["aggregate"]
    log.info(f"Final test: IR={test_agg['info_rate']:.4f}  SR={test_agg['success_rate']:.4f}")

    # ── Save everything ───────────────────────────────────────
    agent.save_workflow(str(OUT_DIR / "awm_workflow.txt"))
    agent.save_memory(str(OUT_DIR / "awm_exemplars.json"))

    summary = {
        "config": {
            "batch_size": BATCH_SIZE, "max_batches": MAX_BATCHES,
            "val_every": VAL_EVERY, "checkpoint_every": CHECKPOINT_EVERY,
            "max_turns": MAX_TURNS, "model": MODEL, "seed": SEED,
        },
        "data": {
            "train": len(train_dialogues), "val": len(val_dialogues),
            "test": len(test_dialogues), "batches": len(batches),
        },
        "seed_val": seed_val["aggregate"],
        "final_test": test_agg,
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
    log.info(f"Seed val:   IR={seed_val['aggregate']['info_rate']:.4f}  "
             f"SR={seed_val['aggregate']['success_rate']:.4f}")
    log.info(f"Final test: IR={test_agg['info_rate']:.4f}  "
             f"SR={test_agg['success_rate']:.4f}")
    log.info(f"LLM calls logged: {logger.count}")
    return summary


if __name__ == "__main__":
    main()
