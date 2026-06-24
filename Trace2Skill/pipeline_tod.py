#!/usr/bin/env python3
"""ToD Skill Evolution Pipeline -- backward-compatible re-export.

This file is kept for backward compatibility.  All logic now lives in
the ``Trace2Skill.pipeline`` package.

Usage (both styles work)::

    # New style
    from Trace2Skill.pipeline import PipelineConfig, run_pipeline

    # Old style (still works)
    from Trace2Skill.pipeline_tod import PipelineConfig, run_pipeline

CLI::

    # New style
    python -m Trace2Skill.pipeline.main --smoke-test

    # Old style (still works)
    python -m Trace2Skill.pipeline_tod --smoke-test
"""

from Trace2Skill.pipeline.config import EvolutionConfig, PipelineConfig, PipelineResult
from Trace2Skill.pipeline.main import run_pipeline

__all__ = [
    "EvolutionConfig",
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline",
]

if __name__ == "__main__":
    # Delegate to the package's main CLI
    import argparse
    import logging

    ap = argparse.ArgumentParser(
        description="ToD Skill Evolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (no LLM calls, minimal data):
  python -m Trace2Skill.pipeline_tod --smoke-test

  # One-shot on test split:
  python -m Trace2Skill.pipeline_tod --split test --end 50

  # Batch training:
  python -m Trace2Skill.pipeline_tod --batch-training --batch-size 50 --val-every 5
""".strip(),
    )
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--split", default=None, choices=["train", "validation", "test"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-sample", type=int, default=50)
    ap.add_argument("--output-dir", default="outputs/tod_pipeline")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--max-turns", type=int, default=6)
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
