#!/usr/bin/env python3
"""Run a paired, turn-level MultiWOZ hotel pilot with AWM or standard Ours.

The dataset projection and sampling are identical for both methods. Run each
method in its own process with --workers 5 for ten concurrent case workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_tod.multiwoz_policy import (
    MultiWOZAWMAgent, MultiWOZStandardSkillAgent, load_hotel_pilot,
    ontology_from_train, predict_policy,
)
from eval_tod.sgd_adapter import gold_policy, score_policies
from eval_tod.sgd_errors import error_summary, record_error
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase


def run(args) -> dict:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {output}")
    if not args.dry_run:
        from llm import resolve_config
        resolve_config(model=args.model)
    data = load_hotel_pilot(args.source, seed=args.seed, fraction=args.fraction)
    train, evaluation = data["train"], data[args.split]
    ontology = ontology_from_train(train)
    manifest = {
        "protocol": "multiwoz21_official_split_first_domain_hotel_turn_ast_v1",
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "seed": args.seed, "fraction": args.fraction, "split": args.split,
        "model": args.model, "method": args.method, "workers": args.workers,
        "dialogues": {split: len(rows) for split, rows in data.items()},
        "policy_turns": {split: sum(len(d["target_indices"]["policy_turn_indices"]) for d in rows)
                         for split, rows in data.items()},
        "selected_ids": {split: [d["dialogue_id"] for d in rows] for split, rows in data.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    if args.dry_run:
        print(json.dumps({key: manifest[key] for key in ("dialogues", "policy_turns")}), flush=True)
        return manifest
    configure_usage(output)
    agent = None
    if args.method == "awm":
        agent = MultiWOZAWMAgent(family="hotel", model=args.model, ontology=ontology,
                                 predictor=predict_policy)
        agent.parallel_workers = args.workers
        for batch_index, start in enumerate(range(0, len(train), args.awm_batch_size), 1):
            batch = train[start:start + args.awm_batch_size]
            print(f"AWM train batch {batch_index}/{(len(train) + args.awm_batch_size - 1) // args.awm_batch_size}",
                  flush=True)
            result = agent.train_batch(batch, batch_index=batch_index,
                                       trace_path=output / "awm_training_turns.jsonl")
            with (output / "awm_training_batches.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            agent.save(output)
            if args.max_batches and batch_index >= args.max_batches:
                break
    else:
        agent = MultiWOZStandardSkillAgent(family="hotel", model=args.model, ontology=ontology,
                                           predictor=predict_policy, mining_method="trace_cover",
                                           seed=args.seed)
        agent.parallel_workers = args.workers
        agent.runtime_diagnostics_dir = output
        print(f"OURS offline mining: {len(train)} dialogues", flush=True)
        agent.mine(train, output / "offline", cases_per_node=3, compile_batch_size=8)
        print(f"OURS graph: {len(agent.graph['nodes'])} nodes; "
              f"{len(agent.graph['edges'])} observed edges", flush=True)
        print("OURS online refine", flush=True)
        agent.refine(train, output, batch_size=8, analysis_batch_size=8,
                     max_batches=args.max_batches)
        agent.save(output)
    set_phase("testing")
    cases = [(d, index) for d in evaluation
             for index in d["target_indices"]["policy_turn_indices"]]
    print(f"{args.method.upper()} {args.split}: {len(evaluation)} dialogues, {len(cases)} turns", flush=True)
    predictions = ordered_cases(lambda case: agent.predict(*case), cases, workers=args.workers,
                                directory=output, phase="testing")
    gold_rows, pred_rows = [], []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for outcome, (dialogue, turn_index) in zip(predictions, cases):
            gold = gold_policy(dialogue, turn_index)
            try:
                prediction, diagnostic = require_value(outcome)
                score_policies([gold], [prediction])
            except Exception as exc:
                failure = record_error(output, "testing", exc,
                                       dialogue_id=dialogue["dialogue_id"], turn_index=turn_index)
                prediction = {"call": None, "acts": [], "_execution_error": True}
                diagnostic = {"status": "error", "error": failure["error"]}
            gold_rows.append(gold)
            pred_rows.append(prediction)
            stream.write(json.dumps({"dialogue_id": dialogue["dialogue_id"], "turn_index": turn_index,
                                     "gold": gold, "prediction": prediction, "diagnostic": diagnostic},
                                    ensure_ascii=False) + "\n")
    errors = error_summary(output)
    usage = persist_usage()
    summary = {**manifest, **score_policies(gold_rows, pred_rows),
               "graph_nodes": len(agent.graph["nodes"]) if args.method == "ours" else None,
               "graph_edges": len(agent.graph["edges"]) if args.method == "ours" else None,
               "skill_stage": agent.stage if args.method == "ours" else None,
               "case_errors": errors, "llm_usage": usage,
               "execution_status": "completed_with_errors" if errors["total"] else "completed"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("method", "dialogues", "counts",
                                                     "turn_joint_ast", "case_errors", "execution_status")},
                     ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("awm", "ours"), required=True)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/eval/multiwoz21/data/data/dialogues.json")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--fraction", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--awm-batch-size", type=int, default=20)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.awm_batch_size < 1 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("Workers and batch limits must be positive")
    run(args)


if __name__ == "__main__":
    main()
