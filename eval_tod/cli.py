"""Command-line entry point for ToD evaluation.

Usage::

    python -m eval_tod.cli \\
        --dataset multiwoz21 \\
        --data_path data/eval/multiwoz21 \\
        --predictions outputs/predictions.json \\
        --split test \\
        --output results.json

    # With multi-agent LLM judge
    python -m eval_tod.cli \\
        --dataset multiwoz21 \\
        --data_path data/eval/multiwoz21 \\
        --predictions outputs/predictions.json \\
        --llm_judge \\
        --llm_model deepseek-chat \\
        --llm_sample_size 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evaluate import evaluate, print_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ToD agent predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m eval_tod.cli --dataset multiwoz21 --data_path data/eval/multiwoz21 --predictions preds.json
  python -m eval_tod.cli --dataset multiwoz21 --data_path data/eval/multiwoz21 --predictions preds.json --split test
  python -m eval_tod.cli --dataset multiwoz21 --data_path data/eval/multiwoz21 --predictions preds.json --llm_judge
        """.strip(),
    )

    parser.add_argument(
        "--dataset", required=True,
        choices=["multiwoz21"],
        help="Dataset name. Currently: multiwoz21",
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Path to dataset directory (e.g. data/eval/multiwoz21)",
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to predictions JSON file",
    )
    parser.add_argument(
        "--split", default=None,
        choices=["train", "validation", "test"],
        help="Data split to evaluate (default: all)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write results JSON (default: stdout summary only)",
    )
    parser.add_argument(
        "--llm_judge", action="store_true",
        help="Enable multi-agent LLM-as-a-Judge evaluation (5 specialist judges + 1 combiner)",
    )
    parser.add_argument(
        "--llm_dimensions", nargs="+", default=None,
        metavar="DIM",
        help="Dimensions for LLM judge (default: task_completion slot_accuracy dialogue_fluency helpfulness efficiency)",
    )
    parser.add_argument(
        "--llm_sample_size", type=int, default=None,
        help="Sample N dialogues for LLM judge (cost control)",
    )
    parser.add_argument(
        "--llm_model", default="deepseek-chat",
        help="LLM model for judge (default: deepseek-chat)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate data path exists
    if not Path(args.data_path).exists():
        print(f"Error: data_path does not exist: {args.data_path}", file=sys.stderr)
        return 1

    # Validate predictions file exists
    if not Path(args.predictions).exists():
        print(f"Error: predictions file does not exist: {args.predictions}", file=sys.stderr)
        return 1

    try:
        result = evaluate(
            dataset_name=args.dataset,
            data_path=args.data_path,
            predictions_path=args.predictions,
            split=args.split,
            llm_judge=args.llm_judge,
            llm_judge_dimensions=args.llm_dimensions,
            llm_judge_sample_size=args.llm_sample_size,
            llm_model=args.llm_model,
            output_path=args.output,
        )

        print_summary(result)
        return 0

    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
