#!/usr/bin/env python3
"""Run and aggregate independent SGD train-seen domain experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_tod.sgd_adapter import metrics_from_counts


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("all", "oracle", "standard", "frequency_graph", "awm"), default="all")
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
    args = parser.parse_args()
    if args.resume_run and args.output_dir:
        parser.error("--resume-run and --output-dir are mutually exclusive")
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
    completed = []
    failed = []
    for task in tasks:
        path = run_root / task["method"] / task["domain"] / "summary.json"
        if not path.is_file():
            failed.append(task)
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        counts = grouped.setdefault(task["method"], Counter())
        counts.update(summary["counts"])
        completed.append({"method": task["method"], "domain": task["domain"],
                          "turns": summary["counts"].get("turns", 0)})
    aggregate = {
        "protocol": "family_event_weighted; multi-domain dialogues can contribute to several families",
        "completed": completed,
        "missing": failed,
        "methods": {name: metrics_from_counts(counts) for name, counts in sorted(grouped.items())},
    }
    (run_root / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return aggregate


def main() -> None:
    args = _arguments()
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
    ids = [part.strip() for part in args.workflow_ids.split(",") if part.strip()]
    if ids and args.workers != 1 and args.workers != len(ids):
        raise SystemExit("--workers must equal the number of --workflow-ids")
    worker_count = len(ids) if ids else args.workers
    run_root = (args.resume_run or args.output_dir or
                ROOT / "outputs" / f"full_sgd_{datetime.now():%Y-%m-%d_%H-%M-%S}").resolve()
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
        "workers": [
            {"worker": index, "workflow_id": ids[index] if ids else None,
             "estimated_policy_turns": loads[index], "tasks": assigned}
            for index, assigned in enumerate(assignments)
        ],
    }
    plan_path = run_root / "workflow_load_plan.json"
    if args.resume_run and plan_path.is_file():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        for key in ("split", "model", "max_dialogues", "tasks", "workers", "awm"):
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
            command = [
                sys.executable, str(ROOT / "scripts/run_sgd_domain_eval.py"),
                "--domain", task["domain"], "--method", task["method"],
                "--split", args.split, "--splits-dir", str(args.splits_dir),
                "--archive", str(args.archive), "--output-dir", str(output),
                "--model", args.model,
            ]
            if args.max_dialogues is not None:
                command += ["--max-dialogues", str(args.max_dialogues)]
            if task["method"] == "awm":
                command += [
                    "--awm-batch-size", str(args.awm_batch_size),
                    "--awm-workflow-max-chars", str(args.awm_workflow_max_chars),
                    "--awm-exemplar-max-chars", str(args.awm_exemplar_max_chars),
                ]
                if args.awm_max_train is not None:
                    command += ["--awm-max-train", str(args.awm_max_train)]
                if args.awm_max_batches is not None:
                    command += ["--awm-max-batches", str(args.awm_max_batches)]
                if args.awm_resource_root:
                    command += ["--awm-resource-dir", str(args.awm_resource_root / task["domain"])]
            env = os.environ.copy()
            if ids:
                env["SKILLMINING_WORKFLOW_ID"] = ids[worker_index]
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


if __name__ == "__main__":
    main()
