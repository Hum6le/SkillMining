#!/usr/bin/env python3
r"""ABCD Trace2Skill-style pipeline driven by AST.

This is a separate pipeline from the original Trace2Skill MultiWOZ flow.
It keeps the same high-level loop:

1. Run a seed agent with a skill/workflow prompt
2. Evaluate on AST
3. Analyze failed conversations
4. Evolve the skill with Trace2Skill's ParallelSkillEvolver
5. Re-evaluate on AST

The key difference is that failure detection and optimization are driven by
ABCD Action State Tracking (AST) instead of MultiWOZ IR/Success.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

_TRACE2SKILL = _PROJECT_ROOT / "Trace2Skill"
if str(_TRACE2SKILL) in sys.path:
    sys.path.remove(str(_TRACE2SKILL))
sys.path.insert(0, str(_TRACE2SKILL))

from eval_tod.abcd.agent import ABCDAgent, compute_ast_from_turn_results
from eval_tod.abcd.data import load_abcd_data
from eval_tod.abcd.metrics import evaluate_abcd
from eval_tod.abcd.agent import turn_results_to_abcd_predictions
from eval_tod.abcd.data import extract_ground_truth
from eval_tod.cli import evaluate_text_records
from eval_tod.response_logger import ResponseLogger
from llm import _get_client, resolve_config

ABCD_DIR = "data/eval/abcd/data"
DEFAULT_SKILL_PATH = "eval_tod/skills/abcd_trace2skill/SKILL.md"
DEFAULT_MODEL = "deepseek-chat"


@dataclass
class PipelineOutputs:
    seed_eval: dict[str, Any]
    evolved_eval: dict[str, Any]
    output_dir: Path
    evolved_skill_path: Path


def _default_skill_text() -> str:
    return """---
name: abcd_trace2skill
description: Basic ABCD customer-service skill focused on choosing correct actions and slots
---

# ABCD Action-Slot Dialogue Skill

You are a customer service agent for retail support conversations.

## Primary objective

For each agent turn, first infer the correct backend action and required slot values,
then produce a short natural-language response that matches that action.

## Action discipline

- Predict the backend action before writing the response.
- Use slot values exactly when they are explicitly available in the dialogue context.
- Do not invent slot values that were not established by the customer or system state.
- If the action needs no slots, output no slots.
- If no backend action is needed, use `none`.

## Response discipline

- The response must be consistent with the chosen action.
- Keep responses concise and helpful.
- Avoid promising actions that are not reflected in the backend action choice.

## Common failure patterns to avoid

- Correct response text but wrong backend action
- Correct action name but missing or misordered slot values
- Taking action too early before verification
- Using stale customer information from earlier turns
"""


def _load_skill_text(skill_path: Path) -> str:
    if not skill_path.exists():
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(_default_skill_text(), encoding="utf-8")
    return skill_path.read_text(encoding="utf-8")


def _build_agent(model: str, workflow_text: str, response_logger: ResponseLogger) -> ABCDAgent:
    from awm import MemoryStore, WorkflowStore

    workflow = WorkflowStore()
    if workflow_text.strip():
        workflow.update(workflow_text)
    return ABCDAgent(
        model=model,
        workflow=workflow,
        memory=MemoryStore(),
        response_logger=response_logger,
    )


def _evaluate_turn_results(
    conversations: list[dict[str, Any]],
    turn_results: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    preds = [r["prediction"] for r in turn_results]
    refs = [r["reference"] for r in turn_results]
    text_eval = evaluate_text_records(preds, refs)

    abcd_preds = turn_results_to_abcd_predictions(turn_results, conversations)
    all_gt = [extract_ground_truth(conv) for conv in conversations]
    abcd_eval = evaluate_abcd(all_gt, abcd_preds)

    return {
        "label": label,
        "num_conversations": len(conversations),
        "num_turns": len(turn_results),
        "text": {
            "bert_f1": round(text_eval["bert_f1"], 4),
            "bleu_4": round(text_eval["bleu_4"], 1),
            "rouge_l": round(text_eval["rouge_l"], 4),
        },
        "ast_cds": {
            "ast_joint": round(abcd_eval.ast.joint_accuracy, 4),
            "ast_action_name": round(abcd_eval.ast.action_name_accuracy, 4),
            "ast_slot_value": round(abcd_eval.ast.slot_value_accuracy, 4),
            "cds_overall": round(abcd_eval.cds.overall_cds, 4),
            "num_action_turns": abcd_eval.ast.total_action_turns,
        },
        "summary": (
            f"AST={abcd_eval.ast.joint_accuracy:.4f} "
            f"Action={abcd_eval.ast.action_name_accuracy:.4f} "
            f"Slot={abcd_eval.ast.slot_value_accuracy:.4f} "
            f"CDS={abcd_eval.cds.overall_cds:.4f} "
            f"BERT-F1={text_eval['bert_f1']:.4f}"
        ),
    }


def _build_ast_failure_cases(
    conversations: list[dict[str, Any]],
    turn_results: list[dict[str, Any]],
    ast_scores: list[dict[str, Any]],
    *,
    log_dir: Path,
) -> list[dict[str, Any]]:
    by_convo: dict[str, list[dict[str, Any]]] = {}
    for row in turn_results:
        by_convo.setdefault(str(row["convo_id"]), []).append(row)

    failed_cases: list[dict[str, Any]] = []
    for conv, ast in zip(conversations, ast_scores):
        if ast.get("ast_score", 0.0) >= 1.0:
            continue

        convo_id = str(conv.get("convo_id", "?"))
        turns = sorted(by_convo.get(convo_id, []), key=lambda x: x["turn_index"])
        trajectory_lines = []
        for row in turns:
            trajectory_lines.append(f"[Context upto turn {row['turn_index']}]")
            trajectory_lines.append(row.get("context", ""))
            trajectory_lines.append(f"[Predicted action] {row.get('predicted_action', '')}")
            trajectory_lines.append(f"[Predicted slots] {row.get('predicted_slots', [])}")
            trajectory_lines.append(f"[Predicted response] {row.get('prediction', '')}")
            trajectory_lines.append(f"[Reference response] {row.get('reference', '')}")
            trajectory_lines.append("")

        safe_id = convo_id.replace("/", "_").replace("\\", "_")
        log_path = log_dir / f"{safe_id}.json"
        log_payload = {
            "convo_id": convo_id,
            "ast_score": ast.get("ast_score", 0.0),
            "turn_results": turns,
        }
        log_path.write_text(
            json.dumps(log_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        failed_cases.append({
            "dialogue_id": f"abcd-{convo_id}",
            "domains": [str(conv.get("scenario", {}).get("subflow", "unknown"))],
            "goal_description": json.dumps(conv.get("scenario", {}), ensure_ascii=False),
            "info_rate": ast.get("ast_score", 0.0),
            "success": False,
            "inform_correct": ast.get("action_correct", 0),
            "inform_total": ast.get("action_total", 0),
            "request_correct": ast.get("action_correct", 0),
            "request_total": ast.get("action_total", 0),
            "booking_passed": None,
            "inform_slots": {},
            "request_slots": {},
            "booking": {},
            "goal_inform": {},
            "goal_request": {},
            "has_booking": False,
            "trajectory": "\n".join(trajectory_lines),
        })

    return failed_cases


def _run_error_analysis(
    failed_cases: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    response_logger: ResponseLogger,
) -> Path:
    from eval_tod.error_analysis import ErrorAnalyzer

    analyzer = ErrorAnalyzer(
        model=model,
        workers=4,
        response_logger=response_logger,
    )
    analyzer.analyze_batch(failed_cases, output_dir=str(output_dir))

    parsed_path = output_dir.parent / f"{output_dir.name}_parsed.json"
    subprocess.run(
        [
            sys.executable,
            str(_TRACE2SKILL / "analysis" / "parse_error_analysis_outputs.py"),
            "--input_dir", str(output_dir),
            "--output", str(parsed_path),
        ],
        cwd=str(_TRACE2SKILL),
        check=True,
    )
    return parsed_path


def _run_skill_evolution(
    records_path: Path,
    skill_path: Path,
    output_dir: Path,
    model: str,
    response_logger: ResponseLogger,
) -> list[str]:
    from skill_evolver.parallel_evolving_agent import ParallelSkillEvolver

    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not records:
        return []

    evolver_client = _get_client(
        model=model,
        cache_tag="abcd_trace2skill",
        response_logger=response_logger,
    )

    evolver = ParallelSkillEvolver(
        client=evolver_client,
        skill_dir=str(skill_path.parent),
        batch_size=1,
        merge_batch_size=5,
        max_workers=4,
        max_merge_levels=5,
        temperature=0.3,
        max_tokens=None,
        verbose=True,
        dry_run=False,
        prompt_variant="generic",
        output_dir=output_dir,
        parse_failure_dir=output_dir.parent / "parse_failures",
        max_skill_lines=500,
        skip_translation=False,
        patch_pipeline="json",
    )
    result = evolver.run(records, input_mode="records")
    return result.get("changelog", [])


def run_pipeline(args) -> PipelineOutputs:
    model_cfg = resolve_config(model=args.model)
    model = model_cfg["model"]

    train_convs = load_abcd_data(args.train_split, args.data_path)
    test_convs = load_abcd_data(args.test_split, args.data_path)
    if args.max_train:
        train_convs = train_convs[:args.max_train]
    if args.max_test:
        test_convs = test_convs[:args.max_test]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir) / f"abcd_trace2skill_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(out_dir / "run.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("abcd_trace2skill")

    response_logger = ResponseLogger(out_dir / "llm_responses")

    seed_skill_path = Path(args.skill_path)
    seed_skill_text = _load_skill_text(seed_skill_path)
    evolved_skill_dir = out_dir / "evolved_skill"
    evolved_skill_dir.mkdir(parents=True, exist_ok=True)
    evolved_skill_path = evolved_skill_dir / "SKILL.md"
    evolved_skill_path.write_text(seed_skill_text, encoding="utf-8")

    log.info("Loaded %d train and %d test conversations", len(train_convs), len(test_convs))

    # Stage 1: seed run on training set to mine failures
    log.info("Stage 1: seed run on training set")
    seed_train_agent = _build_agent(model, seed_skill_text, response_logger)
    seed_train_turns = seed_train_agent.generate_all_turn_predictions(
        train_convs,
        predict_actions=True,
    )
    (out_dir / "seed_train_turns.json").write_text(
        json.dumps(seed_train_turns, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    train_ast_scores = compute_ast_from_turn_results(train_convs, seed_train_turns)
    train_eval = _evaluate_turn_results(train_convs, seed_train_turns, "seed_train")
    (out_dir / "seed_train_eval.json").write_text(
        json.dumps(train_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Seed train: %s", train_eval["summary"])

    failed_cases = _build_ast_failure_cases(
        train_convs,
        seed_train_turns,
        train_ast_scores,
        log_dir=out_dir / "failure_logs",
    )
    log.info("AST failures on train: %d / %d", len(failed_cases), len(train_convs))

    changelog: list[str] = []
    if failed_cases:
        # Stage 2: error analysis
        log.info("Stage 2: error analysis")
        error_dir = out_dir / "error_analysis" / "train_seed_failures"
        parsed_path = _run_error_analysis(
            failed_cases,
            error_dir,
            model,
            response_logger,
        )
        log.info("Parsed error analysis -> %s", parsed_path)

        # Stage 3: evolve skill
        log.info("Stage 3: skill evolution")
        changelog = _run_skill_evolution(
            parsed_path,
            evolved_skill_path,
            out_dir / "intermediates",
            model,
            response_logger,
        )
        log.info("Applied %d evolution changes", len(changelog))
    else:
        log.info("No train AST failures, skip evolution")

    # Stage 4: seed test evaluation
    log.info("Stage 4: seed evaluation on test")
    seed_test_agent = _build_agent(model, seed_skill_text, response_logger)
    seed_test_turns = seed_test_agent.generate_all_turn_predictions(
        test_convs,
        predict_actions=True,
    )
    seed_test_eval = _evaluate_turn_results(test_convs, seed_test_turns, "seed_test")
    (out_dir / "seed_test_turns.json").write_text(
        json.dumps(seed_test_turns, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "seed_test_eval.json").write_text(
        json.dumps(seed_test_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Seed test: %s", seed_test_eval["summary"])

    # Stage 5: evolved test evaluation
    log.info("Stage 5: evolved evaluation on test")
    evolved_skill_text = evolved_skill_path.read_text(encoding="utf-8")
    evolved_test_agent = _build_agent(model, evolved_skill_text, response_logger)
    evolved_test_turns = evolved_test_agent.generate_all_turn_predictions(
        test_convs,
        predict_actions=True,
    )
    evolved_test_eval = _evaluate_turn_results(test_convs, evolved_test_turns, "evolved_test")
    (out_dir / "evolved_test_turns.json").write_text(
        json.dumps(evolved_test_turns, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "evolved_test_eval.json").write_text(
        json.dumps(evolved_test_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Evolved test: %s", evolved_test_eval["summary"])

    summary = {
        "config": {
            "data_path": args.data_path,
            "train_split": args.train_split,
            "test_split": args.test_split,
            "max_train": args.max_train,
            "max_test": args.max_test,
            "model": model,
            "skill_path": str(seed_skill_path),
        },
        "seed_train": train_eval,
        "seed_test": seed_test_eval,
        "evolved_test": evolved_test_eval,
        "failed_train_cases": len(failed_cases),
        "changelog": changelog,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return PipelineOutputs(
        seed_eval=seed_test_eval,
        evolved_eval=evolved_test_eval,
        output_dir=out_dir,
        evolved_skill_path=evolved_skill_path,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ABCD Trace2Skill-style pipeline driven by AST",
    )
    parser.add_argument("--data-path", default=ABCD_DIR)
    parser.add_argument("--train-split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--test-split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--max-train", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=100)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--skill-path", default=DEFAULT_SKILL_PATH)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    result = run_pipeline(args)
    seed_ast = result.seed_eval["ast_cds"]["ast_joint"]
    evolved_ast = result.evolved_eval["ast_cds"]["ast_joint"]
    print("\n" + "=" * 60)
    print("ABCD TRACE2SKILL PIPELINE COMPLETE")
    print(f"Output:      {result.output_dir}")
    print(f"Skill:       {result.evolved_skill_path}")
    print(f"Seed AST:    {seed_ast:.4f}")
    print(f"Evolved AST: {evolved_ast:.4f}")
    print(f"Delta AST:   {evolved_ast - seed_ast:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
