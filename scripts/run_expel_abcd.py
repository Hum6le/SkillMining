#!/usr/bin/env python3
"""Run the ExpeL baseline on one ABCD subflow.

The script uses the shared full-turn ABCD runner and evaluator.  ExpeL rules
are the only learned resource; AWM workflow and exemplar memory are empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_tod.abcd.agent import compute_ast_from_turn_results, turn_results_to_abcd_predictions
from eval_tod.abcd.data import load_abcd_data
from eval_tod.cli import evaluate_abcd_bundle
from expel_adapter import ExpeLABCDAgent, ExpeLRuleStore


def _batches(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def _load_subflow(split, subflow, limit=None):
    path = ROOT / "data" / "eval" / "abcd" / "splits" / subflow / f"{split}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = [c for c in load_abcd_data(split, str(ROOT / "data" / "eval" / "abcd" / "data"))
                if str(c.get("scenario", {}).get("subflow", "")) == subflow]
    return data[:limit] if limit else data


def _serialize_predictions(turn_results, conversations):
    return [asdict(p) for p in turn_results_to_abcd_predictions(turn_results, conversations)]


def main():
    parser = argparse.ArgumentParser(description="ExpeL ABCD baseline")
    parser.add_argument("--subflow", required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--max-rules", type=int, default=20)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-from", type=str, default=None)
    args = parser.parse_args()
    if args.eval_only and not args.eval_from:
        parser.error("--eval-only requires --eval-from")

    out = ROOT / "outputs" / f"expel_abcd_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out.mkdir(parents=True, exist_ok=True)
    train = _load_subflow("train", args.subflow, args.max_train)
    test = _load_subflow("test", args.subflow, args.max_test)
    if not train or not test:
        raise ValueError(f"subflow {args.subflow!r} has no train/test data")

    rule_path = None
    if args.eval_only:
        source = Path(args.eval_from)
        rule_path = source / "expel_rules.json"
    elif args.resume_from:
        source = Path(args.resume_from)
        rule_path = source / "expel_rules.json"
    rules = ExpeLRuleStore.load(rule_path) if rule_path and rule_path.exists() else ExpeLRuleStore(args.max_rules)
    agent = ExpeLABCDAgent(model=args.model, rule_store=rules, expose_scenario_labels=True)
    batch_records = []

    if not args.eval_only:
        batches = _batches(train, args.batch_size)
        if args.max_batches:
            batches = batches[:args.max_batches]
        for batch_index, batch in enumerate(batches, 1):
            turns = agent.generate_all_turn_predictions(batch, predict_actions=True, verbose=False)
            metrics = compute_ast_from_turn_results(batch, turns)
            induction = agent.induce_rules(batch, turns, metrics)
            batch_records.append({"batch": batch_index, "metrics": metrics, "induction": induction})
            print(f"[ExpeL] batch={batch_index} rules={len(rules.rules)}")

    test_turns = agent.generate_all_turn_predictions(test, predict_actions=True, verbose=False)
    test_metrics = compute_ast_from_turn_results(test, test_turns)
    abcd_records = _serialize_predictions(test_turns, test)
    text_records = []
    for conv in test:
        rows = [r for r in test_turns if str(r.get("convo_id")) == str(conv.get("convo_id")) and r.get("target_type") == "utterance"]
        text_records.append({"dialogue_id": f"abcd-{conv.get('convo_id', '?')}", "response_text": rows[-1].get("prediction", "") if rows else ""})
    result = evaluate_abcd_bundle(test, text_records=text_records, abcd_records=abcd_records)

    (out / "expel_rules.json").write_text(json.dumps({"max_rules": rules.max_rules, "rules": [asdict(r) for r in rules.rules]}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "train_induction.json").write_text(json.dumps(batch_records, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "test_turn_predictions.json").write_text(json.dumps(test_turns, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "test_abcd_predictions.json").write_text(json.dumps(abcd_records, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(result["summary"])
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
