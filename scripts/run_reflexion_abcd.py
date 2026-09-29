#!/usr/bin/env python3
"""Train and evaluate one offline Reflexion adaptation on an ABCD subflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_tod.abcd.agent import compute_ast_from_turn_results
from eval_tod.response_logger import ResponseLogger
from reflexion_adapter.abcd import (
    ReflexionABCDAgent, ReflectionStore, build_reflection_feedback,
    customer_query, generate_reflection,
)
from scripts.llm_usage_utils import (
    get_usage, merge_usage_summaries, reset_usage, split_usage_summary,
)


def _read_conversations(path: Path, subflow: str) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty or invalid ABCD split: {path}")
    actual = {str(row.get("scenario", {}).get("subflow", "")) for row in rows}
    if actual != {subflow}:
        raise ValueError(f"split {path} contains subflows {sorted(actual)}, expected {subflow}")
    ids = [str(row.get("convo_id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"conversation IDs must be nonempty and unique: {path}")
    return rows


def _read_trials(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid trial checkpoint {path}:{line_number}") from exc
    return rows


def _append_trial(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _usage_from(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflexion on one ABCD subflow")
    parser.add_argument("--subflow", required=True)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-trials", type=int, default=2,
                        help="Maximum offline attempts per training conversation")
    parser.add_argument("--reflection-limit", type=int, default=3,
                        help="Maximum stored reflections injected into one prompt")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--skip-final-test", action="store_true")
    args = parser.parse_args()
    if args.max_trials < 1 or args.reflection_limit < 0:
        parser.error("--max-trials must be positive and --reflection-limit nonnegative")
    if args.max_train is not None and args.max_train < 1:
        parser.error("--max-train must be positive")
    if args.max_test is not None and args.max_test < 1:
        parser.error("--max-test must be positive")

    split_root = ROOT / "data" / "eval" / "abcd" / "splits" / args.subflow
    train = _read_conversations(args.train_file or split_root / "train.json", args.subflow)
    test = _read_conversations(args.test_file or split_root / "test.json", args.subflow)
    train = train[:args.max_train] if args.max_train else train
    test = test[:args.max_test] if args.max_test else test
    if {str(row["convo_id"]) for row in train} & {str(row["convo_id"]) for row in test}:
        raise ValueError("training and test conversation IDs overlap")

    out = args.output_dir or Path(os.environ.get(
        "ABCD_OUTPUT_DIR", ROOT / "outputs" / f"reflexion_abcd_{datetime.now():%Y-%m-%d_%H-%M-%S}",
    ))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.resume_from.resolve() if args.resume_from else out
    trial_path = out / "training_trials.jsonl"
    if not args.resume_from and trial_path.exists():
        raise ValueError(f"existing trial checkpoint requires --resume-from: {trial_path}")
    source_trials = _read_trials(source / "training_trials.jsonl")
    if source != out and source_trials:
        if trial_path.exists():
            raise ValueError("cannot resume into an output directory with existing trials")
        trial_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_trials), encoding="utf-8")
    trials = list(source_trials)
    allowed_ids = {str(row["convo_id"]) for row in train}
    if any(str(row.get("conversation_id", "")) not in allowed_ids for row in trials):
        raise ValueError("trial checkpoint contains conversations outside the training split")
    prior_usage = None
    if args.resume_from:
        prior_usage = (
            trials[-1].get("llm_usage_cumulative") if trials else None
        ) or _usage_from(source / "llm_usage_generation.json")
        if trials and prior_usage is None:
            raise ValueError("cannot resume trials without a generation usage snapshot")
    reset_usage()
    logger = ResponseLogger(str(out / "llm_responses"))
    reflection_path = out / "reflections.json"
    training_usage_path = out / "llm_usage_generation.json"

    for conversation in train:
        cid = str(conversation["convo_id"])
        existing = [row for row in trials if str(row["conversation_id"]) == cid]
        indices = [int(row["trial_index"]) for row in existing]
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError(f"noncontiguous Reflexion trials for conversation {cid}")
        if existing and existing[-1].get("succeeded"):
            continue
        if len(existing) >= args.max_trials:
            continue
        for trial_index in range(len(existing) + 1, args.max_trials + 1):
            store = ReflectionStore.from_trials(trials)
            actor = ReflexionABCDAgent(
                model=args.model, reflection_store=store, same_conversation=True,
                reflection_limit=args.reflection_limit, expose_scenario_labels=False,
                response_logger=logger,
            )
            turns = actor.generate_all_turn_predictions(
                [conversation], predict_actions=True, verbose=False,
            )
            metrics = compute_ast_from_turn_results([conversation], turns)[0]
            action_total = int(metrics.get("action_total", 0))
            succeeded = action_total == 0 or int(metrics.get("action_correct", 0)) == action_total
            feedback = build_reflection_feedback(conversation, turns, metrics) if not succeeded else None
            prior_for_task = store.select(
                "", conversation_id=cid, same_conversation=True,
                limit=args.reflection_limit,
            )
            reflection = generate_reflection(
                conversation, trial_index, feedback, prior_for_task,
                model=args.model, response_logger=logger,
            ) if feedback is not None else ""
            record = {
                "conversation_id": cid, "trial_index": trial_index,
                "query": customer_query(conversation),
                "metrics": metrics, "succeeded": succeeded,
                "feedback": feedback, "reflection": reflection,
            }
            cumulative_usage = merge_usage_summaries(prior_usage, get_usage()) if prior_usage else get_usage()
            record["llm_usage_cumulative"] = cumulative_usage
            _append_trial(trial_path, record)
            trials.append(record)
            ReflectionStore.from_trials(trials).save(reflection_path)
            training_usage_path.write_text(
                json.dumps(cumulative_usage, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            print(f"[Reflexion] convo={cid} trial={trial_index} "
                  f"AST={metrics.get('ast_score', 0):.3f} reflected={bool(reflection)}", flush=True)
            if succeeded:
                break

    ReflectionStore.from_trials(trials).save(reflection_path)
    generation_usage = merge_usage_summaries(prior_usage, get_usage()) if prior_usage else get_usage()
    training_usage_path.write_text(json.dumps(generation_usage, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "method": "reflexion", "subflow": args.subflow, "model": args.model,
        "max_trials": args.max_trials, "reflection_limit": args.reflection_limit,
        "skip_final_test": args.skip_final_test,
        "adaptation": "offline_train_feedback_frozen_test_memory",
    }
    # Let the shared evaluator read the exact prompt budget even when it runs
    # in this process; a crash here still leaves an incomplete, resumable run.
    (out / "summary.json").write_text(json.dumps({"config": {**config, "skip_final_test": False}, "final_test": None},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    if args.skip_final_test:
        final_test = None
        usage = split_usage_summary(generation_usage, None)
    else:
        from scripts.evaluate_abcd_method import _evaluate_rows

        evaluation_dir = out / "evaluation"
        _evaluate_rows("reflexion", out, test, args.model, evaluation_dir)
        final_test = json.loads((evaluation_dir / "result.json").read_text(encoding="utf-8"))
        testing_usage = final_test["llm_usage"]["testing"]
        usage = split_usage_summary(generation_usage, testing_usage)
        final_test["llm_usage"] = usage
    (out / "llm_usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "config": config,
        "data": {"train_sessions": len(train), "test_sessions": len(test)},
        "generation": {
            "trials": len(trials),
            "reflections": len(ReflectionStore.from_trials(trials).reflections),
        },
        "artifacts": {"training_trials": str(trial_path), "reflections": str(reflection_path)},
        "final_test": final_test,
        "llm_usage": usage,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "generation": summary["generation"],
                      "test_summary": final_test.get("summary") if final_test else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
