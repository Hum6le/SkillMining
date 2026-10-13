#!/usr/bin/env python3
"""Run and aggregate independent SGD train-seen domain experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from eval_tod.sgd_adapter import metrics_from_counts
from scripts.run_sgd_domain_eval import add_ours_arguments, validate_ours_arguments
from scripts.llm_usage_utils import merge_usage_summaries


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("all", "oracle", "standard", "frequency_graph", "awm", "ours"), default="all")
    parser.add_argument("--domain", help="Run only one train-seen domain family")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/eval/sgd/splits_by_domain")
    parser.add_argument("--archive", type=Path, default=ROOT / "data/eval/sgd/data.zip")
    parser.add_argument("--output-dir", type=Path, help="New run directory")
    parser.add_argument("--resume-run", type=Path, help="Reuse a run directory and skip completed tasks")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-dialogues", type=int)
    parser.add_argument("--awm-resource-root", type=Path,
                        help="Frozen AWM resources, with one directory per domain")
    parser.add_argument("--awm-batch-size", type=int, default=20)
    parser.add_argument("--awm-max-train", type=int)
    parser.add_argument("--awm-max-batches", type=int)
    parser.add_argument("--awm-workflow-max-chars", type=int, default=8000)
    parser.add_argument("--awm-exemplar-max-chars", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--workflow-ids", default="", help="Comma-separated API workflow IDs, one per worker")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write a load plan without model calls")
    parser.add_argument("--aggregate-only", action="store_true", help="Rebuild metric/usage totals from an existing --resume-run, without calls")
    add_ours_arguments(parser, resource_option="--ours-resource-root")
    args = parser.parse_args()
    validate_ours_arguments(parser, args)
    if args.resume_run and args.output_dir:
        parser.error("--resume-run and --output-dir are mutually exclusive")
    if args.aggregate_only and not args.resume_run:
        parser.error("--aggregate-only requires --resume-run")
    if args.workers < 1 or (args.max_dialogues is not None and args.max_dialogues < 1):
        parser.error("--workers and --max-dialogues must be positive")
    if any(value is not None and value < 1 for value in (
        args.awm_batch_size, args.awm_max_train, args.awm_max_batches,
        args.awm_workflow_max_chars, args.awm_exemplar_max_chars,
    )):
        parser.error("AWM limits and budgets must be positive")
    return args


def _aggregate(run_root: Path, tasks: list[dict]) -> dict:
    grouped: dict[str, Counter] = {}
    usage_groups = {}
    missing_usage = []
    incomplete_usage = []
    completed = []
    case_errors = []
    failed = []
    for task in tasks:
        path = run_root / task["method"] / task["domain"] / "summary.json"
        if not path.is_file():
            failed.append(task)
            partial = path.parent / "llm_usage.json"
            if partial.is_file():
                incomplete_usage.append({"task": task, "llm_usage": json.loads(partial.read_text(encoding="utf-8"))})
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        inherited_errors = summary.get("resource_training_errors") or {}
        if summary.get("case_errors", {}).get("total", 0) or inherited_errors.get("total", 0):
            case_errors.append({**task, **summary.get("case_errors", {}),
                                "resource_training_errors": inherited_errors})
        counts = grouped.setdefault(task["method"], Counter())
        counts.update(summary["counts"])
        usage = summary.get("llm_usage")
        if usage is None:
            usage_path = path.parent / "llm_usage.json"
            if usage_path.is_file():
                usage = json.loads(usage_path.read_text(encoding="utf-8"))
        if usage is None or "generation" not in usage:
            # Historical summaries stored test-only usage and a separate
            # mining bucket. Keep these recoverable figures, flag missingness.
            if "usage" in summary:
                generation = {"total": summary.get("mining_usage", {})}
                testing = {"total": summary["usage"]}
                usage = {"generation": generation, "testing": testing,
                         "total": merge_usage_summaries(generation, testing), "by_phase": {}}
            else:
                usage = None
            missing_usage.append({"method": task["method"], "domain": task["domain"],
                                  "reason": "legacy_or_missing_phase_accounting"})
        if usage is not None:
            usage_groups.setdefault(task["method"], []).append(usage)
        completed.append({"method": task["method"], "domain": task["domain"],
                          "turns": summary["counts"].get("turns", 0)})
    methods = {name: metrics_from_counts(counts) for name, counts in sorted(grouped.items())}
    for name, result in methods.items():
        snapshots = usage_groups.get(name, [])
        def merged_phase(key):
            return merge_usage_summaries({"total": {}}, *(item.get(key, {}) for item in snapshots))
        result["llm_usage"] = {key: merged_phase(key) for key in ("generation", "testing", "total")}
        phases = {phase for item in snapshots for phase in item.get("by_phase", {})}
        result["llm_usage"]["by_phase"] = {
            phase: merge_usage_summaries({"total": {}}, *(item.get("by_phase", {}).get(phase, {}) for item in snapshots))
            for phase in sorted(phases)
        }
        result["mining_usage"] = result["llm_usage"]["generation"]["total"]
        result["usage"] = result["llm_usage"]["testing"]["total"]
        result["total_usage"] = result["llm_usage"]["total"]["total"]
    aggregate = {
        "protocol": "family_event_weighted; multi-domain dialogues can contribute to several families",
        "completed": completed,
        "missing": failed,
        "case_errors": case_errors,
        "execution_status": "incomplete" if failed else "completed_with_errors" if case_errors else "completed",
        "methods": methods,
        "usage_missing_or_legacy": missing_usage,
        "incomplete_task_usage": incomplete_usage,
        "run_total_usage": merge_usage_summaries(
            {"total": {}}, *(result["llm_usage"]["total"] for result in methods.values()),
            *(item["llm_usage"].get("total", {}) for item in incomplete_usage)),
    }
    (run_root / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_root / "llm_usage.json").write_text(json.dumps({
        "total": aggregate["run_total_usage"],
        "by_method": {name: result["llm_usage"] for name, result in methods.items()},
        "incomplete_task_usage": incomplete_usage,
        "usage_missing_or_legacy": missing_usage,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return aggregate


def _domain_command(args, task: dict, output: Path) -> list[str]:
    """Invoke the standalone runner, forwarding every method option verbatim."""
    command = [sys.executable, str(ROOT / "scripts/run_sgd_domain_eval.py"),
               "--domain", task["domain"], "--method", task["method"],
               "--split", args.split, "--splits-dir", str(args.splits_dir.resolve()),
               "--archive", str(args.archive.resolve()), "--output-dir", str(output), "--model", args.model]
    if args.max_dialogues is not None:
        command += ["--max-dialogues", str(args.max_dialogues)]
    if task["method"] == "awm":
        for key in ("awm_batch_size", "awm_workflow_max_chars", "awm_exemplar_max_chars", "awm_max_train", "awm_max_batches"):
            value = getattr(args, key)
            if value is not None:
                command += ["--" + key.replace("_", "-"), str(value)]
        if args.awm_resource_root:
            command += ["--awm-resource-dir", str(args.awm_resource_root.resolve() / task["domain"])]
    if task["method"] == "ours":
        for key, value in vars(args).items():
            if key == "seed" or (key.startswith("ours_") and key != "ours_resource_root"):
                if isinstance(value, bool):
                    if value:
                        command.append("--" + key.replace("_", "-"))
                elif value is not None:
                    command += ["--" + key.replace("_", "-"), str(value)]
        if args.ours_resource_root:
            command += ["--ours-resource-dir", str(args.ours_resource_root.resolve() / task["domain"])]
    return command


def main() -> None:
    args = _arguments()
    if args.aggregate_only:
        root = args.resume_run.resolve()
        saved = json.loads((root / "workflow_load_plan.json").read_text(encoding="utf-8"))
        result = _aggregate(root, saved["tasks"])
        print(f"Reaggregated without LLM calls: {root / 'aggregate.json'}", flush=True)
        if result["missing"]:
            raise SystemExit(f"Incomplete run: {len(result['missing'])} missing summaries")
        return
    index_path = args.splits_dir / "INDEX.json"
    if not index_path.is_file():
        raise SystemExit(f"Missing SGD split index: {index_path}; run scripts/split_sgd_by_domain.py")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    eligible = index["train_families"]
    if args.domain and args.domain not in eligible:
        raise SystemExit(f"Domain {args.domain!r} was not present in SGD train")
    domains = [args.domain] if args.domain else [
        family for family in eligible
        if index["families"][family][args.split].get("policy_turns", 0) > 0
    ]
    methods = ("standard", "frequency_graph") if args.method == "all" else (args.method,)
    tasks = [
        {"method": method, "domain": domain,
         "estimated_policy_turns": index["families"][domain][args.split].get("policy_turns", 0)}
        for method in methods for domain in domains
    ]
    for task in tasks:
        task["input_sha256"] = {
            split: hashlib.sha256((args.splits_dir / task["domain"] / f"{split}.jsonl.gz").read_bytes()).hexdigest()
            for split in sorted({"train", args.split})
        }
    ids = [part.strip() for part in args.workflow_ids.split(",") if part.strip()]
    if ids and args.workers != 1 and args.workers != len(ids):
        raise SystemExit("--workers must equal the number of --workflow-ids")
    worker_count = len(ids) if ids else args.workers
    run_root = (args.resume_run or args.output_dir or
                ROOT / "outputs" / f"full_sgd_{datetime.now():%Y-%m-%d_%H-%M-%S}").resolve()
    if not args.resume_run and run_root.exists() and any(run_root.iterdir()):
        raise SystemExit("Output directory is nonempty; use a new directory or --resume-run")
    run_root.mkdir(parents=True, exist_ok=True)
    assignments: list[list[dict]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for task in sorted(tasks, key=lambda item: item["estimated_policy_turns"], reverse=True):
        worker = min(range(worker_count), key=lambda index: loads[index])
        assignments[worker].append(task)
        loads[worker] += task["estimated_policy_turns"]
    plan = {
        "run_root": str(run_root), "split": args.split, "model": args.model,
        "max_dialogues": args.max_dialogues, "tasks": tasks,
        "awm": {
            "resource_root": str(args.awm_resource_root.resolve()) if args.awm_resource_root else None,
            "batch_size": args.awm_batch_size, "max_train": args.awm_max_train,
            "max_batches": args.awm_max_batches,
            "workflow_max_chars": args.awm_workflow_max_chars,
            "exemplar_max_chars": args.awm_exemplar_max_chars,
        },
        "ours": {key: str(value.resolve()) if isinstance(value, Path) else value
                 for key, value in vars(args).items() if key.startswith("ours_") or key == "seed"},
        "archive": str(args.archive.resolve()), "splits_dir": str(args.splits_dir.resolve()),
        "workers": [
            {"worker": index, "workflow_id": ids[index] if ids else None,
             "estimated_policy_turns": loads[index], "tasks": assigned}
            for index, assigned in enumerate(assignments)
        ],
    }
    for worker in plan["workers"]:
        for task in worker["tasks"]:
            task["command"] = _domain_command(args, task, run_root / task["method"] / task["domain"])
    plan_path = run_root / "workflow_load_plan.json"
    if args.resume_run and plan_path.is_file():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        for key in ("split", "model", "max_dialogues", "tasks", "workers", "awm", "ours", "archive", "splits_dir"):
            if previous.get(key) != plan[key]:
                raise SystemExit(f"Resume configuration differs for {key}; use the original settings or a new output directory")
    else:
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"SGD run: {run_root} | split={args.split} | domains={len(domains)} | tasks={len(tasks)} | workers={worker_count}", flush=True)
    if args.dry_run:
        print(f"Dry run: {run_root / 'workflow_load_plan.json'}", flush=True)
        return

    def work(worker_index: int) -> list[dict]:
        failures = []
        for task in assignments[worker_index]:
            output = run_root / task["method"] / task["domain"]
            summary = output / "summary.json"
            if args.resume_run and summary.is_file():
                print(f"SKIP {task['method']}/{task['domain']} (summary exists)", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            command = _domain_command(args, task, output)
            env = os.environ.copy()
            if ids:
                env["SKILLMINING_WORKFLOW_ID"] = ids[worker_index]
            env["PYTHONUNBUFFERED"] = "1"
            (output / "invocation.json").write_text(json.dumps({"command": command,
                "workflow_id": env.get("SKILLMINING_WORKFLOW_ID"),
                "isolation": "one standalone process per domain"}, indent=2) + "\n", encoding="utf-8")
            print(f"START {task['method']}/{task['domain']} worker={worker_index}", flush=True)
            with (output / "run.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                failures.append({**task, "returncode": result.returncode})
                print(f"FAILED {task['method']}/{task['domain']} (see {output / 'run.log'})", flush=True)
                if args.stop_on_error:
                    break
            else:
                print(f"DONE {task['method']}/{task['domain']}", flush=True)
        return failures

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        results = list(pool.map(work, range(worker_count)))
    failures = [item for group in results for item in group]
    aggregate = _aggregate(run_root, tasks)
    print(f"Aggregate: {run_root / 'aggregate.json'}", flush=True)
    if failures or aggregate["missing"]:
        raise SystemExit(f"SGD run incomplete: {len(aggregate['missing'])} task(s) lack summary")
    if aggregate["case_errors"]:
        raise SystemExit(f"SGD finished all tasks with case errors in {len(aggregate['case_errors'])} task(s); see aggregate.json/case_errors.jsonl")


if __name__ == "__main__":
    main()
