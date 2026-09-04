#!/usr/bin/env python3
"""Unified evaluation wrapper for all ABCD methods.

The method-specific part is limited to loading its trained resource.  Shard
processes are read-only and select their workflow through
``SKILLMINING_WORKFLOW_ID``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_tod.abcd import merge_turn_results, parse_workflow_ids, shard_conversations
from eval_tod.abcd.agent import ABCDAgent, turn_results_to_abcd_predictions
from eval_tod.cli import evaluate_abcd_bundle
from eval_tod.response_logger import ResponseLogger
from scripts.llm_usage_utils import (
    get_usage, merge_usage_summaries, reset_usage, split_usage_summary, write_usage,
)


def _load_test(path: Path, subflow: str) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"test file must contain a JSON list: {path}")
    observed = {str(row.get("scenario", {}).get("subflow", "")) for row in rows}
    if observed != {subflow}:
        raise ValueError(f"test file is not isolated to {subflow!r}: {sorted(observed)}")
    return rows


def _build_agent(method: str, resource: Path, model: str, logger: ResponseLogger):
    if method == "awm":
        from awm import MemoryStore, WorkflowStore

        workflow = WorkflowStore()
        if (resource / "awm_workflow.txt").is_file():
            workflow.load(str(resource / "awm_workflow.txt"))
        memory = MemoryStore()
        if (resource / "awm_exemplars.json").is_file():
            memory.load(str(resource / "awm_exemplars.json"))
        reference = (resource / "awm_reference.md").read_text(encoding="utf-8") if (resource / "awm_reference.md").is_file() else ""
        return ABCDAgent(model=model, workflow=workflow, memory=memory,
                         reference_text=reference, expose_scenario_labels=False,
                         response_logger=logger)
    if method == "expel":
        from expel_adapter import ExpeLABCDAgent, ExpeLRuleStore

        return ExpeLABCDAgent(
            model=model,
            rule_store=ExpeLRuleStore.load(resource / "expel_rules.json"),
            expose_scenario_labels=False,
            response_logger=logger,
        )
    if method == "asi":
        from asi_offline import create_asi_offline_abcd_agent, load_asi_library

        library = resource / "asi_library" / "current" / "ASI_ACTIONS.md"
        return create_asi_offline_abcd_agent(
            load_asi_library(library), model=model, response_logger=logger
        )
    if method == "trace2skill":
        from awm import WorkflowStore
        from eval_tod.reference_lookup import load_trace2skill_references

        candidates = sorted(resource.glob("abcd_trace2skill_*/evolved_skill"))
        candidates += sorted(resource.glob("**/evolved_skill"))
        if not candidates:
            raise FileNotFoundError(f"no evolved_skill directory under {resource}")
        skill_dir = candidates[-1]
        skill_path = skill_dir / "SKILL.md"
        workflow = WorkflowStore()
        workflow.replace(skill_path.read_text(encoding="utf-8"))
        return ABCDAgent(
            model=model,
            workflow=workflow,
            reference_text=load_trace2skill_references(skill_path),
            expose_scenario_labels=False,
            response_logger=logger,
        )
    raise ValueError(f"unsupported method: {method}")


def _evaluate_rows(method: str, resource: Path, conversations: list[dict], model: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    reset_usage()
    logger = ResponseLogger(str(output / "llm_responses"))
    agent = _build_agent(method, resource, model, logger)
    turns = agent.generate_all_turn_predictions(conversations, predict_actions=True, verbose=False)
    grouped = turn_results_to_abcd_predictions(turns, conversations)
    abcd_records = [{
        "conversation_id": prediction.conversation_id,
        "turns": [{
            "turn_index": turn.turn_index,
            "turn_type": turn.turn_type,
            "predicted_utterance_id": turn.predicted_utterance_id,
            "predicted_action": turn.predicted_action,
            "predicted_slots": turn.predicted_slots,
        } for turn in prediction.turns],
    } for prediction in grouped]
    text_by_id = {
        str(row.get("convo_id", "")): row.get("prediction", "")
        for row in turns if row.get("target_type", "utterance") == "utterance"
    }
    text_records = [{
        "dialogue_id": f"abcd-{conversation.get('convo_id', '?')}",
        "response_text": text_by_id.get(str(conversation.get("convo_id", "")), ""),
    } for conversation in conversations]
    result = evaluate_abcd_bundle(
        conversations, text_records=text_records, abcd_records=abcd_records,
        text_prediction_key="response_text",
    )
    usage = get_usage()
    result["llm_usage"] = split_usage_summary(None, usage)
    (output / "turn_predictions.json").write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "abcd_predictions.json").write_text(json.dumps(abcd_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "llm_usage.json").write_text(
        json.dumps(result["llm_usage"], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _merge(method: str, subflow: str, test_file: Path, shard_root: Path, output: Path) -> None:
    conversations = _load_test(test_file, subflow)
    # The root usage is training usage only when the resource-producing run
    # still has its skip-final-test summary. Once a merged evaluation summary
    # exists, do not count that already-merged usage again on resume.
    training_usage = None
    root_summary_path = output / "summary.json"
    if root_summary_path.is_file():
        try:
            root_summary = json.loads(root_summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            root_summary = {}
        if isinstance(root_summary, dict):
            config = root_summary.get("config", {})
            if config.get("skip_final_test") is True:
                training_usage = root_summary.get("llm_usage")
                if training_usage is None:
                    usage_path = output / "llm_usage.json"
                    if usage_path.is_file():
                        try:
                            training_usage = json.loads(usage_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            training_usage = None
    if training_usage is None and method == "trace2skill":
        # Trace2Skill keeps the train/evolution run in a timestamped child
        # directory, while the unified evaluator writes its merged result at
        # the enclosing per-subflow directory.
        candidates = sorted(output.glob("abcd_trace2skill_*/summary.json"))
        for path in reversed(candidates):
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(candidate, dict):
                continue
            config = candidate.get("config", {})
            if config.get("skip_test_eval") is not True:
                continue
            training_usage = candidate.get("llm_usage")
            if training_usage is None:
                usage_path = path.parent / "llm_usage.json"
                if usage_path.is_file():
                    try:
                        training_usage = json.loads(usage_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        training_usage = None
            break
    shard_results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(shard_root.glob("shard_*/turn_predictions.json"))
    ]
    shard_usage = []
    for path in sorted(shard_root.glob("shard_*/llm_usage.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            shard_usage.append(payload)
    generation_usage = training_usage
    if isinstance(training_usage, dict) and isinstance(training_usage.get("generation"), dict):
        generation_usage = training_usage["generation"]
    # Each shard writes a testing-only phase-split snapshot. Accept legacy
    # flat tracker summaries as well.
    shard_testing_usage = [
        item["testing"] if isinstance(item.get("testing"), dict) else item
        for item in shard_usage
    ]
    testing_usage = merge_usage_summaries(*shard_testing_usage)
    usage = split_usage_summary(generation_usage, testing_usage)
    turns = merge_turn_results(conversations, shard_results)
    grouped = turn_results_to_abcd_predictions(turns, conversations)
    abcd_records = [{
        "conversation_id": prediction.conversation_id,
        "turns": [{
            "turn_index": turn.turn_index,
            "turn_type": turn.turn_type,
            "predicted_utterance_id": turn.predicted_utterance_id,
            "predicted_action": turn.predicted_action,
            "predicted_slots": turn.predicted_slots,
        } for turn in prediction.turns],
    } for prediction in grouped]
    text_by_id = {
        str(row.get("convo_id", "")): row.get("prediction", "")
        for row in turns if row.get("target_type", "utterance") == "utterance"
    }
    text_records = [{
        "dialogue_id": f"abcd-{conversation.get('convo_id', '?')}",
        "response_text": text_by_id.get(str(conversation.get("convo_id", "")), ""),
    } for conversation in conversations]
    result = evaluate_abcd_bundle(
        conversations, text_records=text_records, abcd_records=abcd_records,
        text_prediction_key="response_text",
    )
    result["llm_usage"] = usage
    output.mkdir(parents=True, exist_ok=True)
    (output / "turn_predictions.json").write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "abcd_predictions.json").write_text(json.dumps(abcd_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "llm_usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({
        "config": {"method": method, "subflow": subflow, "evaluation": "sharded"},
        "data": {"test_sessions": len(conversations)},
        "final_test": result,
        "llm_usage": usage,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.get("summary", result), ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ABCD method evaluator")
    parser.add_argument("--method", choices=("awm", "expel", "trace2skill", "asi"), required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--subflow", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--eval-workflow-ids", default=None)
    parser.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    conversations = _load_test(args.test_file, args.subflow)
    workflow_ids = parse_workflow_ids(args.eval_workflow_ids)
    if args.shard_index is not None:
        if args.shard_count is None or not workflow_ids:
            raise ValueError("shard mode requires --eval-workflow-ids and --shard-count")
        rows = shard_conversations(conversations, args.shard_index, args.shard_count)
        _evaluate_rows(args.method, args.resource_dir, rows, args.model, args.output_dir)
        return
    if not workflow_ids:
        _evaluate_rows(args.method, args.resource_dir, conversations, args.model, args.output_dir)
        return
    shard_root = args.output_dir / "eval_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen] = []
    for index, workflow_id in enumerate(workflow_ids):
        shard_output = shard_root / f"shard_{index}"
        shard_output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["SKILLMINING_WORKFLOW_ID"] = workflow_id
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--method", args.method, "--resource-dir", str(args.resource_dir),
            "--test-file", str(args.test_file), "--subflow", args.subflow,
            "--output-dir", str(shard_output), "--model", args.model,
            "--eval-workflow-ids", ",".join(workflow_ids),
            "--shard-index", str(index), "--shard-count", str(len(workflow_ids)),
        ]
        log = (shard_output / "run.log").open("w", encoding="utf-8")
        processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    exit_codes = [process.wait() for process in processes]
    failures = [code for code in exit_codes if code != 0]
    if failures:
        raise SystemExit(f"{len(failures)} evaluation shard(s) failed; inspect {shard_root}")
    _merge(args.method, args.subflow, args.test_file, shard_root, args.output_dir)


if __name__ == "__main__":
    main()
