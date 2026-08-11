#!/usr/bin/env python3
"""Run the complete frozen-library ASIoffline protocol for one ABCD subflow.

All model requests remain inside ``induce_asi_offline_abcd.py`` and
``eval_asi_offline_abcd.py``; both use the repository's ``llm.chat()`` entry
point. This runner only orchestrates local, auditable stage artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent


def _run(arguments: list[str]) -> None:
    print("+", " ".join(arguments))
    subprocess.run(arguments, cwd=_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete ASIoffline reproduction for one ABCD subflow.")
    parser.add_argument("--subflow", required=True)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-actions", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--resume-induction", action="store_true")
    parser.add_argument("--dry-run-induction", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--allow-empty-library",
        action="store_true",
        help="Permit test evaluation with no accepted functions (debug only).",
    )
    args = parser.parse_args()
    subflow = args.subflow.strip()
    if not subflow:
        parser.error("--subflow must not be empty")
    if args.min_actions < 3:
        parser.error("--min-actions must be at least 3 to match original ASI")

    split_dir = _ROOT / "data" / "eval" / "abcd" / "splits" / subflow
    train_file = Path(args.train_file) if args.train_file else split_dir / "train.json"
    test_file = Path(args.test_file) if args.test_file else split_dir / "test.json"
    output_dir = Path(args.output_dir) if args.output_dir else _ROOT / "outputs" / f"asi_offline_abcd_{subflow}"
    induction_dir = output_dir / "induction"
    raw_dir = output_dir / "raw_induction"
    validation_dir = output_dir / "static_validation"
    library_dir = output_dir / "frozen_library"

    prepare = [
        sys.executable, "scripts/prepare_asi_offline_abcd.py",
        "--subflow", subflow,
        "--train-file", str(train_file),
        "--output-dir", str(induction_dir),
        "--min-actions", str(args.min_actions),
    ]
    if args.max_train is not None:
        prepare.extend(["--max-conversations", str(args.max_train)])
    _run(prepare)

    induce = [
        sys.executable, "scripts/induce_asi_offline_abcd.py",
        "--episodes", str(induction_dir / "induction_episodes.jsonl"),
        "--output-dir", str(raw_dir),
        "--temperature", str(args.temperature),
    ]
    if args.resume_induction:
        induce.append("--resume")
    if args.dry_run_induction:
        induce.append("--dry-run")
    _run(induce)
    if args.dry_run_induction:
        print("Dry-run induction complete; validation, freezing, and evaluation were intentionally skipped.")
        return

    _run([
        sys.executable, "scripts/validate_asi_offline_abcd.py",
        "--episodes", str(induction_dir / "induction_episodes.jsonl"),
        "--raw-artifacts", str(raw_dir / "raw_induction_artifacts.jsonl"),
        "--output-dir", str(validation_dir),
    ])
    _run([
        sys.executable, "scripts/freeze_asi_offline_abcd_library.py",
        "--accepted-functions", str(validation_dir / "accepted_functions.json"),
        "--output-dir", str(library_dir),
    ])
    frozen = json.loads((library_dir / "manifest.json").read_text(encoding="utf-8"))
    if not frozen["frozen_unique_functions"] and not args.allow_empty_library:
        raise RuntimeError(
            "No ASI functions passed offline validation. Inspect raw_induction and "
            "static_validation before evaluating an empty-library baseline."
        )
    if args.skip_eval:
        print("Frozen ASI library complete; test evaluation skipped by request.")
        return

    evaluate = [
        sys.executable, "scripts/eval_asi_offline_abcd.py",
        "--skill-library", str(library_dir / "ASI_ACTIONS.md"),
        "--test-file", str(test_file),
        "--expected-subflow", subflow,
        "--output-dir", str(output_dir / "evaluation"),
    ]
    if args.max_test is not None:
        evaluate.extend(["--max-test", str(args.max_test)])
    _run(evaluate)


if __name__ == "__main__":
    main()
