#!/usr/bin/env python3
"""Resume frozen MultiWOZ validation with a larger local process pool."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_tod.multiwoz_policy import MultiWOZStandardSkillAgent, load_hotel_pilot, ontology_from_train, predict_policy
from eval_tod.sgd_adapter import gold_policy, score_policies
from eval_tod.sgd_errors import error_summary, record_error
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase
from scripts.llm_usage_utils import merge_usage_summaries
from scripts.replay_multiwoz_reflection import combine_usage


def read_cached(path, cases):
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # The previous writer may have been interrupted mid-line.
            raise
        if len(rows) >= len(cases):
            raise ValueError("Cached prediction count exceeds validation cases")
        dialogue, turn = cases[len(rows)]
        if (row["dialogue_id"], row["turn_index"], row["gold"]) != (
                dialogue["dialogue_id"], turn, gold_policy(dialogue, turn)):
            raise ValueError("Cached validation identity or gold target mismatch")
        if "prediction" not in row:
            raise ValueError("Cached row has no prediction")
        rows.append(row)
    return rows


def recover_usage(directory, interrupted):
    """Replace parent testing totals with all worker request logs, never add twice."""
    path = directory / "llm_usage.json"
    usage = json.loads(path.read_text(encoding="utf-8"))
    buckets = []
    for log in directory.glob("parallel_cases/testing_*/worker*/llm_requests.jsonl"):
        lines = log.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    continue
                raise
            if row["phase"] != "testing":
                raise ValueError("Unexpected phase in testing worker log")
            buckets.append({"total": row["total"], "by_call_tag": {row["call_tag"]: row["total"]}})
    testing = merge_usage_summaries({"total": {}}, *buckets)
    if testing["total"]["calls"] < usage.get("testing", {}).get("total", {}).get("calls", 0):
        raise ValueError("Worker request logs do not cover previously recorded testing usage")
    usage["by_phase"]["testing"] = testing
    usage["testing"] = testing
    usage["generation"] = merge_usage_summaries({"total": {}}, *[
        value for phase, value in usage["by_phase"].items() if phase != "testing"])
    usage["total"] = merge_usage_summaries(usage["generation"], testing)
    if interrupted:
        usage["incomplete_phases"] = sorted(set(usage.get("incomplete_phases", [])) | {"testing"})
        usage["accounting_complete"] = False
        usage["note"] = "Known completed calls recovered from worker logs; interrupted in-flight requests may have unreported usage."
    path.write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
    return usage


def stop_source(pid, directory):
    import psutil
    try:
        parent = psutil.Process(pid)
        command = parent.cmdline()
    except psutil.NoSuchProcess:
        return False
    if not any(Path(arg).name == "replay_multiwoz_reflection.py" for arg in command):
        raise ValueError("PID is not the expected reflection/evaluation runner")
    index = command.index("--output-dir") + 1
    if Path(command[index]).resolve() != directory:
        raise ValueError("PID output directory differs from requested evaluation")
    parent.suspend()  # Stop new case submissions before terminating its workers.
    processes = [*parent.children(recursive=True), parent]
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        process.kill()
    return True


def run(args):
    directory = args.run_dir.resolve()
    if (directory / "summary.json").exists():
        print("Validation already completed; no restart needed", flush=True)
        return
    manifest_path = directory / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(manifest["source"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Dataset source changed")
    data = load_hotel_pilot(source, seed=manifest["seed"], fraction=manifest["fraction"])
    for split in ("train", "validation"):
        if [d["dialogue_id"] for d in data[split]] != manifest["selected_ids"][split]:
            raise ValueError("Resume split selection differs from original run")
    cases = [(d, i) for d in data["validation"] for i in d["target_indices"]["policy_turn_indices"]]
    agent = MultiWOZStandardSkillAgent(family="hotel", model=manifest["model"],
        ontology=ontology_from_train(data["train"]), predictor=predict_policy, mining_method="trace_cover")
    agent.load(directory)
    trained = json.loads((directory / "ours_training.json").read_text(encoding="utf-8"))
    checkpoint_hash = agent.resource_hash()
    if checkpoint_hash != trained["final_sha256"] or agent.stage != "refined":
        raise ValueError("Resume requires the completed, successful reflection checkpoint")
    # Complete all read-only preflight checks before stopping the existing pool.
    read_cached(directory / "predictions.jsonl", cases)
    stopped = stop_source(args.stop_source_pid, directory)
    if (directory / "summary.json").exists():
        print("Source finished during preflight; existing summary retained", flush=True)
        return
    cached = read_cached(directory / "predictions.jsonl", cases)
    archive = directory / ("evaluation_before_10workers_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    archive.mkdir()
    for name in ("predictions.jsonl", "llm_usage.json", "data_manifest.json"):
        shutil.copy2(directory / name, archive / name)
    recover_usage(directory, stopped)
    resume = {"resumed_at": datetime.now().isoformat(), "previous_pid": args.stop_source_pid,
              "workers": args.workers, "cached_turns": len(cached), "remaining_turns": len(cases) - len(cached),
              "checkpoint_sha256": checkpoint_hash, "archive": str(archive),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest["workers"] = args.workers
    manifest["evaluation_resume"] = resume
    agent._write(manifest_path, manifest)
    agent._write(directory / "evaluation_resume.json", resume)
    configure_usage(directory)
    set_phase("testing")
    agent.parallel_workers = args.workers
    agent.runtime_diagnostics_dir = directory
    print(f"Reused {len(cached)}/{len(cases)} predictions; evaluating {len(cases)-len(cached)} remaining turns with {args.workers} workers", flush=True)
    gold_rows = [row["gold"] for row in cached]
    predicted = [row["prediction"] for row in cached]
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in cached:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        remaining = cases[len(cached):]
        outcomes = ordered_cases(lambda case: agent.predict(*case), remaining,
            workers=args.workers, directory=directory, phase="testing")
        for outcome, (dialogue, turn) in zip(outcomes, remaining):
            gold = gold_policy(dialogue, turn)
            try:
                prediction, diagnostic = require_value(outcome)
            except Exception as exc:
                record_error(directory, "testing", exc, dialogue_id=dialogue["dialogue_id"], turn_index=turn)
                prediction, diagnostic = {"call": None, "acts": [], "_execution_error": True}, {"status": "error"}
            gold_rows.append(gold)
            predicted.append(prediction)
            stream.write(json.dumps({"dialogue_id": dialogue["dialogue_id"], "turn_index": turn,
                "gold": gold, "prediction": prediction, "diagnostic": diagnostic}, ensure_ascii=False) + "\n")
            stream.flush()
    if agent.resource_hash() != checkpoint_hash:
        raise RuntimeError("Frozen skill changed during validation")
    usage = persist_usage()
    parent = Path(manifest["replay_from"])
    if (parent / "summary.json").exists():
        previous = json.loads((parent / "summary.json").read_text(encoding="utf-8"))["llm_usage"]
    else:
        previous = json.loads((parent / "llm_usage.json").read_text(encoding="utf-8"))
        previous["accounting_complete"] = False
    errors = error_summary(directory)
    summary = {**manifest, **score_policies(gold_rows, predicted), "graph_nodes": len(agent.graph["nodes"]),
        "graph_edges": len(agent.graph["edges"]), "skill_stage": agent.stage, "case_errors": errors,
        "llm_usage": combine_usage(previous, usage), "previous_attempt_usage": previous, "replay_usage": usage,
        "execution_status": "completed_with_errors" if errors["total"] else "completed"}
    agent._write(directory / "summary.json", summary)
    from scripts.audit_multiwoz_empty_values import audit
    audit(directory)
    print(json.dumps({"completed": len(predicted), "turn_joint_ast": summary["turn_joint_ast"], "errors": errors["total"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stop-source-pid", type=int, required=True)
    parser.add_argument("--workers", type=int, default=10)
    run(parser.parse_args())
