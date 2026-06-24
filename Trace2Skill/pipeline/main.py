#!/usr/bin/env python3
"""Pipeline orchestration -- ties together config, dataset_split, evaluate, train.

Two modes:
- **One-shot** (``batch_training=False``): the original 6-stage pipeline on a single split.
- **Batch training** (``batch_training=True``): iterates over training dialogues in batches,
  evolving the skill after each batch, with periodic checkpointing and validation.

Usage::

    from Trace2Skill.pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig(batch_training=True, split="train", ...)
    result = run_pipeline(config)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .config import (
    EvolutionConfig,
    PipelineConfig,
    PipelineResult,
    _TRACE2SKILL,
    _PROJECT_ROOT,
)
from .dataset_split import (
    _build_batches,
    _save_batch_checkpoint,
    _resume_from_checkpoint,
    _stage0_load_and_split,
)
from .evaluate import _run_validation
from .train import _run_oneshot_pipeline, _run_training_iteration

log = logging.getLogger(__name__)

# Ensure project root + Trace2Skill on sys.path
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_TRACE2SKILL) not in sys.path:
    sys.path.insert(0, str(_TRACE2SKILL))

from llm import resolve_config


# ══════════════════════════════════════════════════════════════════
# Main pipeline entry point
# ══════════════════════════════════════════════════════════════════


def run_pipeline(config: PipelineConfig | None = None, **kwargs) -> PipelineResult:
    """Run the full skill evolution pipeline.

    Two modes:
    - **One-shot** (default, ``batch_training=False``): the original
      6-stage pipeline on a single split.
    - **Batch training** (``batch_training=True``): iterates over
      training dialogues in batches, evolving the skill after each
      batch, with periodic checkpointing and validation.

    Args:
        config: PipelineConfig with all settings. If None, defaults are used.
        **kwargs: Override individual config fields (e.g. model="gpt-4o").

    Returns:
        PipelineResult with paths to all outputs and evaluation summaries.
    """
    if config is None:
        config = PipelineConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # ── Smoke-test overrides (applied first) ────────────────────
    if config.smoke_test:
        config.apply_smoke_test()

    # ── Resolve API config ──────────────────────────────────────
    cfg = resolve_config(
        api_key=config.api_key, base_url=config.base_url, model=config.model,
    )
    model, api_key, base_url = cfg["model"], cfg["api_key"], cfg["base_url"]

    # ── Prepare timestamped output directory ────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = config.resolved_output_dir / timestamp
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput dir: {out}")

    # ── Response logger ─────────────────────────────────────────
    from eval_tod.response_logger import ResponseLogger
    response_logger = ResponseLogger(str(out / "llm_responses"))

    # ══════════════════════════════════════════════════════════════
    # STAGE 0: Load dataset, report splits, select dialogues
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 0: Load dataset and split")
    print("=" * 60)

    from eval_tod.kb import MultiWOZKB
    from eval_tod.data_loader import load_dataset, list_available_splits

    kb = MultiWOZKB(str(config.resolved_db_dir))
    print(f"KB loaded: {kb.domains}")

    split_counts = list_available_splits(str(config.resolved_data_path))
    print(f"  Dataset:     {config.dataset_name}")
    print(f"  Data path:   {config.resolved_data_path}")
    print(f"  DB dir:      {config.resolved_db_dir}")
    print(f"  All splits:  {dict(split_counts)}")
    print(f"  Total:       {sum(split_counts.values())} dialogues")

    # ── Branch: batch training vs oneshot ───────────────────────
    if config.batch_training and not config.smoke_test:
        return _run_batch_training_pipeline(
            config, model, api_key, base_url, out, kb, split_counts, response_logger,
        )
    else:
        # Original one-shot flow
        if config.split and config.split not in split_counts:
            avail = list(split_counts.keys())
            raise ValueError(
                f"Requested split '{config.split}' not found in data. "
                f"Available: {avail}"
            )

        dialogues = load_dataset(
            config.dataset_name,
            str(config.resolved_data_path),
            split=config.split,
        )
        total_in_split = len(dialogues)

        start = config.start
        end = config.end if config.end is not None else total_in_split
        dialogues = dialogues[start:end]

        print(f"  Selected split: {config.split or 'all'}")
        print(f"    Full split:    {total_in_split} dialogues")
        print(f"    Slice:         [{start}:{end}] -> {len(dialogues)} dialogues")
        print(f"    Domains in slice: "
              f"{sorted(set(d for dg in dialogues for d in dg.domains))}")

        if config.smoke_test:
            print(f"  [SMOKE TEST] no LLM calls, minimal data")

        return _run_oneshot_pipeline(
            config, model, api_key, base_url, out, kb,
            dialogues, split_counts, start, end, total_in_split,
            response_logger,
        )


# ══════════════════════════════════════════════════════════════════
# Batch training pipeline
# ══════════════════════════════════════════════════════════════════


def _run_batch_training_pipeline(
    config: PipelineConfig,
    model: str,
    api_key: str,
    base_url: str,
    out: Path,
    kb,
    split_counts: dict,
    response_logger=None,
) -> PipelineResult:
    """Run the iterative batch-training pipeline.

    Stage 0: Load and split train/val/test
    Stage 1: Seed baseline on validation set
    Stage 2: Batch training loop with checkpointing
    Stage 3: Final test evaluation
    """
    from eval_tod.agent_skill import SkillPreloadedAgent

    # ── Stage 0 (batch): Load and split ─────────────────────────
    print("\n  [BATCH TRAINING MODE]")
    train_raw, val_dialogues, test_dialogues, _ = _stage0_load_and_split(config)

    print(f"  Train:            {len(train_raw)} dialogues")
    print(f"  Validation:       {len(val_dialogues)} dialogues")
    print(f"  Test:             {len(test_dialogues)} dialogues")

    batches = _build_batches(train_raw, config.batch_size, config.max_batches)
    print(f"  Batches:          {len(batches)} (batch_size={config.batch_size})")
    print(f"  Checkpoint every: {config.checkpoint_every or 'never'}")
    print(f"  Validate every:   {config.val_every or 'never'}")

    # ── Prepare evolved skill directory ─────────────────────────
    evolved_skills_dir = out / "evolved_skills"
    evolved_skill_dir = evolved_skills_dir / config.skill_subdir
    if evolved_skill_dir.exists():
        shutil.rmtree(evolved_skill_dir)
    shutil.copytree(
        config.resolved_skill_dir / config.skill_subdir,
        evolved_skill_dir,
    )

    # Resume from checkpoint if requested
    start_batch = 0
    if config.resume_from:
        start_batch = _resume_from_checkpoint(config.resume_from, evolved_skill_dir)

    # ── Stage 1 (batch): Seed baseline on validation ────────────
    print("\n" + "=" * 60)
    print("STAGE 1: Seed skill baseline on validation set")
    print("=" * 60)

    val_result = _run_validation(
        config, model, api_key, base_url, kb,
        str(config.resolved_skill_dir),  # seed skill
        val_dialogues, out, "seed_baseline",
        response_logger=response_logger,
    )
    seed_val_metrics = val_result["metrics"]
    print(f"  Seed val IR: {seed_val_metrics['info_rate']:.4f}, "
          f"Success: {seed_val_metrics['success_rate']:.4f}")

    # ── Also evaluate seed on test for comparison ───────────────
    test_result_seed = _run_validation(
        config, model, api_key, base_url, kb,
        str(config.resolved_skill_dir),  # seed skill
        test_dialogues, out, "test_seed_baseline",
        response_logger=response_logger,
    )
    seed_test_metrics = test_result_seed["metrics"]
    print(f"  Seed test IR: {seed_test_metrics['info_rate']:.4f}, "
          f"Success: {seed_test_metrics['success_rate']:.4f}")

    # ── Create reusable agent ───────────────────────────────────
    agent = SkillPreloadedAgent(
        kb=kb,
        skills_dir=str(evolved_skills_dir),
        model=model,
        max_turns=config.max_turns,
        log_dir=str(out / "trajectories"),
        api_key=api_key,
        base_url=base_url,
        response_logger=response_logger,
    )

    # ── Stage 2: Batch training loop ────────────────────────────
    print("\n" + "=" * 60)
    print(f"STAGE 2: Batch training ({len(batches)} batches, starting at batch {start_batch+1})")
    print("=" * 60)

    val_history: list[dict] = [seed_val_metrics]
    batch_metrics: list[dict] = []
    checkpoint_dir = out / "checkpoints"
    total_failed = 0
    total_llm_calls = 0

    for batch_idx in range(start_batch, len(batches)):
        batch = batches[batch_idx]
        epoch_num = batch_idx + 1

        print(f"\n{'─'*40}")
        print(f"  Batch {epoch_num}/{len(batches)}: {len(batch)} dialogues")
        print(f"{'─'*40}")

        iter_result = _run_training_iteration(
            config, model, api_key, base_url, kb,
            evolved_skills_dir, evolved_skill_dir,
            batch, epoch_num, out, agent=agent,
            response_logger=response_logger,
        )

        batch_metrics.append(iter_result["metrics"])
        total_failed += iter_result["metrics"].get("num_failures", 0)
        total_llm_calls += iter_result.get("llm_calls", 0)

        # ── Checkpoint ──────────────────────────────────────────
        if config.checkpoint_every and epoch_num % config.checkpoint_every == 0:
            _save_batch_checkpoint(
                evolved_skill_dir, checkpoint_dir, epoch_num,
                iter_result["metrics"],
                changelog=iter_result.get("changelog", []),
                val_metrics=val_history[-1] if val_history else None,
            )

        # ── Validation ──────────────────────────────────────────
        if config.val_every and epoch_num % config.val_every == 0:
            print(f"\n  Running validation at batch {epoch_num}...")
            val_result = _run_validation(
                config, model, api_key, base_url, kb,
                str(evolved_skills_dir),
                val_dialogues, out, f"val_step_{epoch_num:04d}",
                response_logger=response_logger,
            )
            vm = val_result["metrics"]
            val_history.append(vm)
            delta_ir = vm["info_rate"] - seed_val_metrics["info_rate"]
            delta_sr = vm["success_rate"] - seed_val_metrics["success_rate"]
            print(f"  Val IR: {vm['info_rate']:.4f} (delta={delta_ir:+.4f}), "
                  f"SR: {vm['success_rate']:.4f} (delta={delta_sr:+.4f})")
            if vm.get("llm_judge"):
                js = vm["llm_judge"]
                print(f"  Val Judge: {', '.join(f'{k}={v:.2f}' for k, v in js.items())}")

    # Final checkpoint
    if config.checkpoint_every:
        _save_batch_checkpoint(
            evolved_skill_dir, checkpoint_dir, len(batches),
            batch_metrics[-1] if batch_metrics else {},
            changelog=[],
            val_metrics=val_history[-1] if val_history else None,
        )

    # ── Stage 3: Final test evaluation ──────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 3: Final test evaluation with evolved skill")
    print("=" * 60)

    test_result_final = _run_validation(
        config, model, api_key, base_url, kb,
        str(evolved_skills_dir),
        test_dialogues, out, "test_final",
        response_logger=response_logger,
    )
    final_test_metrics = test_result_final["metrics"]

    delta_ir = final_test_metrics["info_rate"] - seed_test_metrics["info_rate"]
    delta_sr = final_test_metrics["success_rate"] - seed_test_metrics["success_rate"]
    print(f"\n  Seed    test IR: {seed_test_metrics['info_rate']:.4f}, "
          f"SR: {seed_test_metrics['success_rate']:.4f}")
    print(f"  Evolved test IR: {final_test_metrics['info_rate']:.4f}, "
          f"SR: {final_test_metrics['success_rate']:.4f}")
    print(f"  Delta:           IR={delta_ir:+.4f}, SR={delta_sr:+.4f}")

    # ── Save training trajectory ────────────────────────────────
    trajectory = {
        "seed_val_metrics": seed_val_metrics,
        "seed_test_metrics": seed_test_metrics,
        "final_test_metrics": final_test_metrics,
        "num_batches": len(batches),
        "total_failed": total_failed,
        "total_llm_calls": total_llm_calls,
        "batch_metrics": batch_metrics,
        "val_history": val_history,
    }
    (out / "training_trajectory.json").write_text(
        json.dumps(trajectory, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # ── Done ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"Output: {out}")
    print(f"Batches: {len(batches)}, Total failures: {total_failed}")
    print(f"Seed    test: IR={seed_test_metrics['info_rate']:.4f}  "
          f"SR={seed_test_metrics['success_rate']:.4f}")
    print(f"Evolved test: IR={final_test_metrics['info_rate']:.4f}  "
          f"SR={final_test_metrics['success_rate']:.4f}")
    print(f"Checkpoints:  {checkpoint_dir}")
    print(f"{'='*60}")

    return PipelineResult(
        output_dir=out,
        seed_predictions_path=Path(test_result_seed["predictions_path"]),
        seed_eval_path=Path(test_result_seed["eval_path"]),
        evolved_predictions_path=Path(test_result_final["predictions_path"]),
        evolved_eval_path=Path(test_result_final["eval_path"]),
        evolved_skill_dir=evolved_skill_dir,
        seed_eval=test_result_seed["eval_result"],
        evolved_eval=test_result_final["eval_result"],
        num_dialogues=len(train_raw),
        num_failed=total_failed,
        had_failures=total_failed > 0,
        num_batches=len(batches),
        batch_metrics=batch_metrics,
        val_history=val_history,
        checkpoint_dir=checkpoint_dir,
    )


# ══════════════════════════════════════════════════════════════════
# Script entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="ToD Skill Evolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (no LLM calls, minimal data):
  python -m Trace2Skill.pipeline.main --smoke-test

  # One-shot on test split:
  python -m Trace2Skill.pipeline.main --split test --end 50

  # Batch training with checkpointing:
  python -m Trace2Skill.pipeline.main --batch-training --batch-size 50 \\
      --checkpoint-every 10 --val-every 5 --max-batches 50
""".strip(),
    )
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run smoke test with dummy data, no LLM judge, no evolution")
    ap.add_argument("--split", default=None,
                    choices=["train", "validation", "test"],
                    help="Data split (default: test)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-sample", type=int, default=50)
    ap.add_argument("--output-dir", default="outputs/tod_pipeline")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--max-turns", type=int, default=6)
    # Batch
    ap.add_argument("--batch-training", action="store_true")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--val-split", default=None)
    ap.add_argument("--test-split", default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--resume-from", default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig()

    if args.smoke_test:
        config.apply_smoke_test()
    else:
        if args.split is not None:
            config.split = args.split
        config.start = args.start
        if args.end is not None:
            config.end = args.end
        config.model = args.model
        config.output_dir = args.output_dir
        config.max_turns = args.max_turns
        if args.data_path is not None:
            config.data_path = args.data_path
        if args.no_judge:
            config.llm_judge = False
        config.llm_judge_sample = args.judge_sample
        config.batch_training = args.batch_training
        config.batch_size = args.batch_size
        config.checkpoint_every = args.checkpoint_every
        config.val_every = args.val_every
        if args.val_split:
            config.val_split = args.val_split
        if args.test_split:
            config.test_split = args.test_split
        config.max_batches = args.max_batches
        config.resume_from = args.resume_from

    result = run_pipeline(config)
    if config.batch_training and not config.smoke_test:
        print(f"\nSeed    test: IR={result.seed_eval['aggregate']['info_rate']:.3f}  "
              f"SR={result.seed_eval['aggregate']['success_rate']:.3f}")
        print(f"Evolved test: IR={result.evolved_eval['aggregate']['info_rate']:.3f}  "
              f"SR={result.evolved_eval['aggregate']['success_rate']:.3f}")
    else:
        print(f"\nSeed:     IR={result.seed_eval['aggregate']['info_rate']:.3f}  "
              f"SR={result.seed_eval['aggregate']['success_rate']:.3f}")
        if result.had_failures:
            print(f"Evolved:  IR={result.evolved_eval['aggregate']['info_rate']:.3f}  "
                  f"SR={result.evolved_eval['aggregate']['success_rate']:.3f}")
