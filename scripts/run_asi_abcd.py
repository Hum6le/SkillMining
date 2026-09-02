#!/usr/bin/env python3
"""Run the ABCD online ASI protocol for one subflow.

The runner keeps induction and evaluation artifacts separate.  A candidate is
never made active before it passes source-rollout replay and train held-out
evaluation.  Test data is read only for the final report.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asi_offline import (
    ASIOnlineValidationResult,
    ASIOnlineLibraryManager,
    build_online_episode_batch,
    decide_asi_update,
    evaluate_asi_library,
    induce_online_episode,
    successful_online_episodes,
    validate_online_candidates,
)
from eval_tod.abcd.agent import ABCDAgent, compute_ast_from_turn_results
from eval_tod.cli import evaluate_abcd_bundle
from eval_tod.response_logger import ResponseLogger
from llm import chat


def _load(path: Path, subflow: str) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"split must be a JSON array: {path}")
    observed = {str(row.get("scenario", {}).get("subflow", "")) for row in rows}
    if observed != {subflow}:
        raise ValueError(f"{path} is not isolated to {subflow!r}: {sorted(observed)}")
    return rows


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_agent(library_path: Path, model: str, logger: ResponseLogger | None) -> ABCDAgent:
    from asi_offline import create_asi_offline_abcd_agent, load_asi_library

    return create_asi_offline_abcd_agent(
        load_asi_library(library_path), model=model, response_logger=logger
    )


def _call_induction(episode, model: str, response_logger: ResponseLogger):
    def call(messages, *, temperature=1.0):
        return chat(
            messages,
            model=model,
            temperature=temperature,
            response_logger=response_logger,
            call_tag="asi_online_induction",
        )

    return induce_online_episode(episode, call)


def _final_test(library_path: Path, conversations: list[dict[str, Any]], model: str, output_dir: Path) -> dict[str, Any]:
    return evaluate_asi_library(library_path, conversations, model=model, output_dir=output_dir)


def _format_duration(seconds: float | None) -> str:
    """Render an ETA without depending on an optional progress-bar package."""
    if seconds is None or seconds < 0:
        return "unknown"
    total = int(seconds + 0.5)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _log_progress(log: logging.Logger, completed: int, total: int, started_at: float) -> None:
    """Log total batch progress and an average-time ETA."""
    elapsed = max(0.0, time.monotonic() - started_at)
    average = elapsed / completed if completed else None
    remaining = (total - completed) * average if average is not None else None
    width = 24
    filled = int(width * completed / total) if total else width
    bar = "#" * filled + "." * (width - filled)
    log.info(
        "ASI progress [%s] %d/%d (%.1f%%), elapsed=%s, ETA=%s",
        bar,
        completed,
        total,
        100.0 * completed / total if total else 100.0,
        _format_duration(elapsed),
        _format_duration(remaining),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Online ASI ABCD runner for one subflow")
    parser.add_argument("--subflow", required=True)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--heldout-size", type=int, default=10)
    parser.add_argument("--min-actions", type=int, default=3)
    parser.add_argument("--max-induction-episodes", type=int, default=8)
    parser.add_argument("--min-ast-delta", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.heldout_size < 1:
        parser.error("--batch-size and --heldout-size must be positive")
    if args.max_induction_episodes < 1:
        parser.error("--max-induction-episodes must be positive")

    split_dir = ROOT / "data" / "eval" / "abcd" / "splits" / args.subflow
    train_file = args.train_file or split_dir / "train.json"
    test_file = args.test_file or split_dir / "test.json"
    output_dir = args.output_dir or ROOT / "outputs" / f"asi_abcd_{args.subflow}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("asi_abcd")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    file_handler = logging.FileHandler(output_dir / "run.log")
    logging.getLogger().addHandler(file_handler)

    train = _load(train_file, args.subflow)
    test = _load(test_file, args.subflow)
    batches = [train[i : i + args.batch_size] for i in range(0, len(train), args.batch_size)]
    if args.max_batches:
        batches = batches[: args.max_batches]
    total_batches = len(batches)
    progress_started_at = time.monotonic()
    manager = ASIOnlineLibraryManager(output_dir / "asi_library")
    response_logger = ResponseLogger(str(output_dir / "llm_responses"))
    history_path = output_dir / "batch_history.jsonl"
    completed: set[int] = set()
    if args.resume and history_path.is_file():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "completed":
                    completed.add(int(row["batch_index"]))

    for batch_index, batch in enumerate(batches, start=1):
        if batch_index in completed:
            log.info("Batch %d already completed; skipping", batch_index)
            _log_progress(log, len(completed), total_batches, progress_started_at)
            continue
        batch_dir = output_dir / "batches" / f"batch_{batch_index:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        library_path = manager.current_library_path()
        log.info("Batch %d/%d: rollout %d conversations", batch_index, len(batches), len(batch))
        agent = _build_agent(library_path, args.model, response_logger)
        turns = agent.generate_all_turn_predictions(batch, predict_actions=True, verbose=False)
        ast_results = compute_ast_from_turn_results(batch, turns)
        _write(batch_dir / "turn_predictions.json", turns)
        _write(batch_dir / "ast_results.json", ast_results)
        episodes = build_online_episode_batch(batch, turns, ast_results, source_split="train", min_actions=args.min_actions)
        eligible = successful_online_episodes(episodes)
        _write(batch_dir / "online_episodes.json", [episode.to_dict() for episode in episodes])
        log.info("Batch %d: eligible successful rollouts=%d/%d", batch_index, len(eligible), len(episodes))

        candidates = []
        induction_records = []
        for episode in eligible[: args.max_induction_episodes]:
            try:
                raw, episode_candidates, rejected = _call_induction(episode, args.model, response_logger)
                candidates.extend(episode_candidates)
                induction_records.append({
                    "episode_id": episode.conversation_id,
                    "raw_response": raw,
                    "candidates": [candidate.to_dict() for candidate in episode_candidates],
                    "rejected": rejected,
                })
            except Exception as exc:
                log.warning("Induction failed for %s: %s", episode.conversation_id, exc)
                induction_records.append({"episode_id": episode.conversation_id, "error": str(exc)})
        _write(batch_dir / "induction.json", induction_records)

        valid_candidates = []
        validation_records = []
        by_episode = {episode.conversation_id: episode for episode in eligible}
        for episode_id, episode_record in [(row["episode_id"], row) for row in induction_records if "candidates" in row]:
            episode = by_episode[episode_id]
            episode_candidates = [candidate for candidate in candidates if candidate.episode_id == episode_id]
            validation = validate_online_candidates(episode, episode_candidates)
            valid_candidates.extend(
                candidate for candidate in episode_candidates
                if candidate.skill_name in set(validation.accepted_candidates)
            )
            validation_records.append(validation.to_dict())
        _write(batch_dir / "validation.json", validation_records)

        batch_summary: dict[str, Any] = {
            "batch_index": batch_index,
            "status": "completed",
            "conversations": len(batch),
            "eligible_episodes": len(eligible),
            "induced_candidates": len(candidates),
            "valid_candidates": len(valid_candidates),
            "library_version": None,
        }
        if valid_candidates:
            heldout_start = batch_index * args.batch_size
            heldout = train[heldout_start : heldout_start + args.heldout_size]
            if not heldout:
                heldout = train[max(0, len(train) - args.heldout_size) :]
            baseline = evaluate_asi_library(library_path, heldout, model=args.model, output_dir=batch_dir / "heldout_before")
            validation = ASIOnlineValidationResult(
                episode_id=f"batch_{batch_index:04d}",
                accepted_candidates=[candidate.skill_name for candidate in valid_candidates],
                rejected_candidates=[],
                replay_valid=True,
                replay_errors=[],
                rewritten_trajectory=[],
            )
            update = manager.stage(valid_candidates, validation, ast_before=float(baseline.get("ast_cds", {}).get("ast_joint", 0.0)))
            candidate_result = evaluate_asi_library(Path(update.version_dir) / "ASI_ACTIONS.md", heldout, model=args.model, output_dir=batch_dir / "heldout_after")
            decision = decide_asi_update(baseline, candidate_result, min_ast_delta=args.min_ast_delta)
            batch_summary.update({"library_version": update.version, "decision": decision.to_dict()})
            if decision.accepted:
                manager.accept(update, ast_after=decision.ast_after)
                log.info("Batch %d: accepted library version %04d (%s)", batch_index, update.version, decision.reason)
            else:
                manager.rollback(update, ast_after=decision.ast_after, reason=decision.reason)
                log.info("Batch %d: rolled back library version %04d (%s)", batch_index, update.version, decision.reason)
        _write(batch_dir / "batch_summary.json", batch_summary)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(batch_summary, ensure_ascii=False) + "\n")
        completed.add(batch_index)
        _log_progress(log, len(completed), total_batches, progress_started_at)

    final = None
    if not args.skip_final_test:
        final = _final_test(manager.current_library_path(), test, args.model, output_dir / "final_test")
        _write(output_dir / "summary.json", {
            "config": {"method": "asi", "subflow": args.subflow, "batch_size": args.batch_size},
            "data": {"train_conversations": len(train), "test_conversations": len(test)},
            "final_test": final,
        })
        log.info("Final test: %s", final.get("summary", final))
    log.info("ASI run complete: %s", output_dir)


if __name__ == "__main__":
    main()
