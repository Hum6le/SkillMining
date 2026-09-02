#!/usr/bin/env python3
"""Evaluate one trained ABCD method resource on one test shard.

This process is intentionally read-only with respect to the trained method
resource.  The workflow ID is selected by the parent shell through
``SKILLMINING_WORKFLOW_ID``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_tod.abcd import merge_turn_results, shard_conversations
from eval_tod.abcd.agent import ABCDAgent, turn_results_to_abcd_predictions
from eval_tod.cli import evaluate_abcd_bundle
from eval_tod.response_logger import ResponseLogger


def _load_test(path: Path, subflow: str, index: int, count: int) -> list[dict]:
    conversations = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(conversations, list):
        raise ValueError(f"test file must contain a JSON list: {path}")
    observed = {str(row.get("scenario", {}).get("subflow", "")) for row in conversations}
    if observed != {subflow}:
        raise ValueError(f"test file is not isolated to {subflow!r}: {sorted(observed)}")
    return shard_conversations(conversations, index, count)


def _agent(method: str, resource: Path, model: str, logger: ResponseLogger):
    if method == "awm":
        from awm import MemoryStore, WorkflowStore
        from eval_tod.reference_lookup import load_trace2skill_references

        workflow = WorkflowStore()
        workflow_path = resource / "awm_workflow.txt"
        if workflow_path.is_file():
            workflow.load(str(workflow_path))
        memory = MemoryStore()
        memory_path = resource / "awm_exemplars.json"
        if memory_path.is_file():
            memory.load(str(memory_path))
        reference = ""
        reference_path = resource / "awm_reference.md"
        if reference_path.is_file():
            reference = reference_path.read_text(encoding="utf-8")
        return ABCDAgent(model=model, workflow=workflow, memory=memory,
                         reference_text=reference, expose_scenario_labels=False,
                         response_logger=logger)
    if method == "expel":
        from expel_adapter import ExpeLABCDAgent, ExpeLRuleStore
        rules = ExpeLRuleStore.load(resource / "expel_rules.json")
        return ExpeLABCDAgent(model=model, rule_store=rules,
                              expose_scenario_labels=False,
                              response_logger=logger)
    if method == "asi":
        from asi_offline import create_asi_offline_abcd_agent, load_asi_library
        library = resource / "asi_library" / "current" / "ASI_ACTIONS.md"
        return create_asi_offline_abcd_agent(load_asi_library(library), model=model,
                                             response_logger=logger)
    if method == "trace2skill":
        from eval_tod.reference_lookup import load_trace2skill_references
        skill_dirs = sorted(resource.glob("abcd_trace2skill_*/evolved_skill"))
        if not skill_dirs:
            skill_dirs = sorted(resource.glob("**/evolved_skill"))
        if not skill_dirs:
            raise FileNotFoundError(f"no evolved_skill directory under {resource}")
        skill_dir = skill_dirs[-1]
        skill_path = skill_dir / "SKILL.md"
        from awm import WorkflowStore

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("awm", "expel", "trace2skill", "asi"), required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--subflow", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()
    conversations = _load_test(args.test_file, args.subflow, args.shard_index, args.shard_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ResponseLogger(str(args.output_dir / "llm_responses"))
    agent = _agent(args.method, args.resource_dir, args.model, logger)
    turns = agent.generate_all_turn_predictions(conversations, predict_actions=True, verbose=False)
    grouped = turn_results_to_abcd_predictions(turns, conversations)
    records = [{
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
        conversations, text_records=text_records, abcd_records=records,
        text_prediction_key="response_text",
    )
    (args.output_dir / "turn_predictions.json").write_text(
        json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "abcd_predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"shard": args.shard_index, "conversations": len(conversations), "summary": result.get("summary", {})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
