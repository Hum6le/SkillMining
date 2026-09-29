#!/usr/bin/env python3
"""Unified per-subflow SKILL-DISCO runner with phase-split usage artifacts."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--compile-and-verify", dest="compile_and_verify", action="store_true", default=True)
    parser.add_argument("--pseudocode-only", dest="compile_and_verify", action="store_false")
    parser.add_argument("--verification-fraction", type=float, default=0.2)
    parser.add_argument("--verification-cases", type=int, default=12)
    parser.add_argument("--max-synthesis-attempts", type=int, default=3)
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--resume-generation", action="store_true",
                        help="Reuse and repair an existing generation artifact before evaluation")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # The shared ABCD launcher assigns one workflow ID per worker. Both child
    # processes must keep that worker's environment for server-side llm.py.
    worker_env = os.environ.copy()
    artifact = output_dir / "generation_artifact.json"
    library = output_dir / "SKILL.md"
    generation_cmd = [
        sys.executable, str(ROOT / "scripts" / "run_skill_disco_abcd.py"),
        "--input", str(args.train_file.resolve()), "--output", str(artifact),
        "--library-output", str(library), "--model", args.model,
        "--batch-size", str(args.batch_size), "--min-support", str(args.min_support),
        "--expected-subflow", args.subflow,
    ]
    if args.compile_and_verify:
        generation_cmd.extend([
            "--compile-and-verify", "--verification-fraction", str(args.verification_fraction),
            "--verification-cases", str(args.verification_cases),
            "--max-synthesis-attempts", str(args.max_synthesis_attempts),
        ])
    if args.resume_generation and artifact.is_file():
        from skill_disco.name_resolution import make_verified_names_unique

        generated = _read_json(artifact)
        if not generated or not isinstance(generated.get("skill_library"), str):
            raise ValueError(f"cannot resume invalid generation artifact: {artifact}")
        if args.compile_and_verify and not isinstance(generated.get("compiled_skills"), list):
            raise ValueError(f"cannot resume non-compiled generation artifact: {artifact}")
        if make_verified_names_unique(generated):
            artifact.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
        library.write_text(generated["skill_library"], encoding="utf-8")
        print(f"Reused Skill-DisCo generation artifact: {artifact}")
    else:
        subprocess.run(generation_cmd, cwd=ROOT, env=worker_env, check=True)

    generated = _read_json(artifact)
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
        if args.compile_and_verify:
            evaluation_cmd.extend(["--generation-artifact", str(artifact)])
        subprocess.run(evaluation_cmd, cwd=ROOT, env=worker_env, check=True)
        final_test = _read_json(evaluation_dir / "result.json")
        testing_usage = _phase_bucket(_read_json(evaluation_dir / "llm_usage.json"), "testing")

    usage = split_usage_summary(generation_usage, testing_usage if final_test is not None else None)
    summary = {
        "config": {
            "method": "skill_disco", "subflow": args.subflow, "model": args.model,
            "batch_size": args.batch_size, "min_support": args.min_support,
            "compile_and_verify": args.compile_and_verify,
            "verification_fraction": args.verification_fraction if args.compile_and_verify else None,
            "verification_mode": "heldout_recorded_action_replay" if args.compile_and_verify else None,
            "test_runtime": "compiled_prefix_invocation" if args.compile_and_verify else "prompt_guidance",
            "skip_final_test": args.skip_final_test,
        },
        "data": {
            "train_sessions": len(json.loads(args.train_file.read_text(encoding="utf-8"))),
            "test_sessions": len(json.loads(args.test_file.read_text(encoding="utf-8"))),
        },
        "artifacts": {
            "generation_artifact": str(artifact), "skill_library": str(library),
        },
        "generation": {
            "candidate_contracts": len(generated.get("contracts", [])),
            "verified_skills": len(generated.get("verified_contracts", [])) if args.compile_and_verify else None,
            "induction_sessions": generated.get("split", {}).get("induction_conversations"),
            "verification_sessions": generated.get("split", {}).get("verification_conversations"),
        },
        "final_test": final_test,
        "llm_usage": usage,
    }
    (output_dir / "llm_usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": final_test.get("summary") if final_test else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
