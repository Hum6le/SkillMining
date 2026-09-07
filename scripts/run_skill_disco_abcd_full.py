#!/usr/bin/env python3
"""Unified per-subflow SKILL-DISCO runner with phase-split usage artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llm_usage_utils import split_usage_summary


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _phase_bucket(payload: dict, phase: str) -> dict:
    value = payload.get(phase)
    return value if isinstance(value, dict) else payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one complete SKILL-DISCO ABCD subflow")
    parser.add_argument("--subflow", required=True)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--skip-final-test", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "generation_artifact.json"
    library = output_dir / "SKILL.md"
    generation_cmd = [
        sys.executable, str(ROOT / "scripts" / "run_skill_disco_abcd.py"),
        "--input", str(args.train_file.resolve()), "--output", str(artifact),
        "--library-output", str(library), "--model", args.model,
        "--batch-size", str(args.batch_size), "--min-support", str(args.min_support),
        "--expected-subflow", args.subflow,
    ]
    subprocess.run(generation_cmd, cwd=ROOT, check=True)

    generation_usage = _phase_bucket(_read_json(output_dir / "llm_usage_generation.json"), "generation")
    final_test = None
    testing_usage: dict = {}
    if not args.skip_final_test:
        evaluation_dir = output_dir / "evaluation"
        evaluation_cmd = [
            sys.executable, str(ROOT / "scripts" / "eval_skill_disco_abcd.py"),
            "--skill-library", str(library), "--test-file", str(args.test_file.resolve()),
            "--output-dir", str(evaluation_dir), "--model", args.model,
            "--expected-subflow", args.subflow,
        ]
        subprocess.run(evaluation_cmd, cwd=ROOT, check=True)
        final_test = _read_json(evaluation_dir / "result.json")
        testing_usage = _phase_bucket(_read_json(evaluation_dir / "llm_usage.json"), "testing")

    usage = split_usage_summary(generation_usage, testing_usage if final_test is not None else None)
    summary = {
        "config": {
            "method": "skill_disco", "subflow": args.subflow, "model": args.model,
            "batch_size": args.batch_size, "min_support": args.min_support,
            "skip_final_test": args.skip_final_test,
        },
        "data": {
            "train_sessions": len(json.loads(args.train_file.read_text(encoding="utf-8"))),
            "test_sessions": len(json.loads(args.test_file.read_text(encoding="utf-8"))),
        },
        "artifacts": {
            "generation_artifact": str(artifact), "skill_library": str(library),
        },
        "final_test": final_test,
        "llm_usage": usage,
    }
    (output_dir / "llm_usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": final_test.get("summary") if final_test else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
