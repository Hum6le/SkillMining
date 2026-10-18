#!/usr/bin/env python3
"""Show matched MultiWOZ hotel pilot progress and online reflection failures."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _lines(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def snapshot(directory: Path, method: str) -> dict:
    manifest = _read(directory / "data_manifest.json")
    summary = _read(directory / "summary.json")
    launch = _read(directory.parent / "launch.json")
    pid = next((row.get("pid") for row in launch.get("processes", [])
                if row.get("method") == method), None)
    alive = None
    if pid is not None:
        try:
            import psutil
            alive = psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except (ImportError, OSError):
            pass
    errors = _lines(directory / "case_errors.jsonl")
    offline_errors = _lines(directory / "offline" / "case_errors.jsonl")
    counts: dict[str, int] = {}
    for row in errors:
        phase = row.get("phase", "unknown")
        counts[phase] = counts.get(phase, 0) + 1
    result = {"method": method, "output": str(directory),
              "status": summary.get("execution_status", "exited_without_summary" if alive is False
                                    else "running" if manifest else "not_started"),
              "pid": pid, "process_alive": alive,
              "train_turns": manifest.get("policy_turns", {}).get("train", manifest.get("train_turns")),
              "validation_turns": manifest.get("policy_turns", {}).get("validation", manifest.get("validation_turns")),
              "errors_by_phase": counts,
              "offline_errors": len(offline_errors),
              "offline_errors_by_phase": {
                  phase: sum(row.get("phase") == phase for row in offline_errors)
                  for phase in sorted({row.get("phase", "unknown") for row in offline_errors})
              },
              "reflection_errors": [
                  {"group_index": row.get("group_index"), "exception": row.get("exception"),
                   "error": str(row.get("error", ""))[:180], "time": row.get("time")}
                  for row in errors if row.get("phase") == "online_reflection"
              ]}
    result["worker_cases_completed"] = {
        phase: len(list(directory.glob(f"**/parallel_cases/{phase}_*/worker*/case_*_usage.json")))
        for phase in ("offline_compile", "train_rollout", "online_diagnosis", "testing")
    }
    if method.startswith("ours"):
        schedule = _read(directory / "online_schedule.json")
        result["scheduled_train_turns"] = schedule.get("num_selected_samples")
        result["rollout_turns_written"] = len(_lines(directory / "ours_training_turns.jsonl"))
        result["diagnosis_reports"] = len(list(directory.glob("diagnosis_*.json")))
        result["reflection_reports"] = sum(bool(re.fullmatch(r"reflection_\d{4}(?:_error)?\.json", p.name))
                                           for p in directory.glob("reflection_*.json"))
        offline_state = _read(directory / "offline" / "sgd_standard_state.json")
        state = offline_state.get("state", {})
        result["graph_nodes"] = summary.get("graph_nodes", len(state.get("nodes", [])) or None)
    elif method == "trace2skill":
        result["training_batches_written"] = len(list(directory.glob("batches/batch_*/summary.json")))
        result["training_turns_written"] = len(_lines(directory / "training_turns.jsonl"))
        result["evolution_errors"] = counts.get("trace2skill_evolution", 0)
        result["analysis_errors"] = counts.get("trace2skill_analysis", 0)
    else:
        result["training_batches_written"] = len(_lines(directory / "awm_training_batches.jsonl"))
        result["training_turns_written"] = len(_lines(directory / "awm_training_turns.jsonl"))
    result["validation_predictions_written"] = len(_lines(directory / "predictions.jsonl"))
    if summary:
        result["turn_joint_ast"] = summary.get("turn_joint_ast")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=15.)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least 1 second")
    while True:
        methods = ("awm", "ours", "trace2skill") if (args.root / "trace2skill").exists() else ("awm", "ours")
        launch = _read(args.root / "launch.json")
        methods = tuple(dict.fromkeys([*methods, *[entry["method"]
            for entry in launch.get("processes", []) if entry.get("method", "").startswith("ours")]]))
        directories = {method: Path(next((entry["output_dir"]
                        for entry in launch.get("processes", [])
                        if entry.get("method") == method and entry.get("output_dir")),
                        str(args.root / method))) for method in methods}
        rows = [snapshot(directories[method], method) for method in methods]
        print(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                          "methods": rows}, ensure_ascii=False), flush=True)
        if all(row["status"].startswith("completed") for row in rows):
            comparison = {"protocol": "paired_multiwoz21_hotel_first_domain_10pct",
                          "generated_at": datetime.now().isoformat(timespec="seconds"),
                          "methods": {}}
            for method in methods:
                summary = _read(directories[method] / "summary.json")
                usage = summary.get("llm_usage", {}).get("total", {}).get("total", {})
                comparison["methods"][method] = {
                    key: summary.get(key) for key in
                    ("action_set_exact", "action_micro_f1", "slot_exact_given_action",
                     "turn_joint_ast", "counts", "case_errors", "graph_nodes", "graph_edges")
                }
                comparison["methods"][method]["llm_calls"] = usage.get("calls")
                comparison["methods"][method]["llm_tokens"] = usage.get("total_tokens")
                comparison["methods"][method]["offline_errors"] = len(
                    _lines(directories[method] / "offline" / "case_errors.jsonl"))
            (args.root / "comparison.json").write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not args.watch or all(row["status"].startswith("completed") or
                                 row["status"] == "exited_without_summary" for row in rows):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
