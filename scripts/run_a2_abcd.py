#!/usr/bin/env python3
"""A2 two-stage ABCD ablation: schema, offline, online, and full.

Each coarse subflow is trained and evaluated independently. In a multi-flow
run, each workflow ID owns one serial worker; workers run concurrently.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPLITS = ROOT / "data/eval/abcd/splits"
ARMS = ("schema_only", "offline_only", "online_only", "full")
SCHEMA_SKILL = (
    "# ABCD action skill\n\n"
    "Follow the action and ordered-slot contract supplied by the runtime. "
    "No workflow transitions or procedural rules are initialized.\n"
)


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(value: str) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("--workflow-ids contains duplicate IDs")
    return ids


def _execution_arms(args: argparse.Namespace) -> tuple[str, ...]:
    """Keep the experiment arms fixed; skipping is a per-invocation choice."""
    skipped = set(getattr(args, "skip_arms", ()))
    return tuple(arm for arm in args.arms if arm not in skipped)


def _subflows(args: argparse.Namespace) -> list[str]:
    index = _read(SPLITS / "INDEX.json")
    selected = (sorted(index) if args.all else
                [name.strip() for name in args.subflows.split(",") if name.strip()]
                if args.subflows else [args.subflow])
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Selected subflows must be nonempty and unique")
    for name in selected:
        if name not in index or not all((SPLITS / name / f"{split}.json").is_file()
                                         for split in ("train", "test")):
            raise ValueError(f"Unknown or incomplete ABCD subflow: {name}")
    return selected


def _online_command(args: argparse.Namespace, subflow: str, arm: str,
                    output: Path, offline: Path, workflow_ids: list[str]) -> list[str]:
    cmd = [sys.executable, str(ROOT / "scripts/run_backbone_online_refine.py"),
           "--subflow", subflow, "--output-dir", str(output),
           "--model", args.model, "--batch-size", str(args.batch_size),
           "--refinement-mode", args.refinement_mode]
    cmd += ["--schema-cold-start"] if arm == "online_only" else ["--offline-dir", str(offline)]
    if workflow_ids:
        cmd += ["--refine-workflow-ids", ",".join(workflow_ids)]
        # The existing test sharder needs fork. On Windows, use the first ID
        # for serial test evaluation; rollout waves still use all IDs.
        if "fork" in mp.get_all_start_methods() and len(workflow_ids) > 1:
            cmd += ["--eval-workflow-ids", ",".join(workflow_ids)]
    for key in ("max_train", "max_test", "max_batches"):
        value = getattr(args, key)
        if value is not None:
            cmd += ["--" + key.replace("_", "-"), str(value)]
    if args.skip_utterance_eval:
        cmd += ["--skip-utterance-eval"]
    if args.resume and (output / "skill_dag_state.json").is_file():
        cmd = [item for item in cmd if item not in ("--offline-dir", str(offline))]
        cmd += ["--resume"]
    return cmd


def _run_online(args: argparse.Namespace, subflow: str, arm: str,
                flow_dir: Path, workflow_ids: list[str]) -> None:
    output = flow_dir / arm
    output.mkdir(parents=True, exist_ok=True)
    command = _online_command(args, subflow, arm, output, flow_dir / "offline_artifact", workflow_ids)
    env = os.environ.copy()
    env["ABCD_OUTPUT_DIR"] = str(flow_dir / "eval_aux")
    env["PYTHONUNBUFFERED"] = "1"
    if workflow_ids:
        env["SKILLMINING_WORKFLOW_ID"] = workflow_ids[0]
    with (output / "launcher.log").open("a", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{arm}/{subflow} failed (exit {result.returncode}); see {output / 'launcher.log'}")


def _run_direct(args: argparse.Namespace, subflow: str, flow_dir: Path,
                workflow_ids: list[str]) -> None:
    execution_arms = _execution_arms(args)
    if not any(arm in execution_arms for arm in ("schema_only", "offline_only", "full")):
        return
    # Keep this import inside the worker, so --dry-run and --help need no ML deps.
    from awm import MemoryStore, WorkflowStore
    from eval_tod.abcd.agent import ABCDAgent
    from scripts.llm_usage_utils import get_usage, reset_usage, split_usage_summary
    from scripts.run_subflow_eval import evaluate_agent_on_subflow, load_subflow_data, mine_subflow_skill_backbone

    train, test = load_subflow_data(subflow)
    if args.max_train is not None:
        train = train[:args.max_train]
    if args.max_test is not None:
        test = test[:args.max_test]
    offline = flow_dir / "offline_artifact"
    offline_usage_path = offline / "mining_usage.json"
    need_offline = any(arm in execution_arms for arm in ("offline_only", "full"))
    if need_offline and not (offline / "complete.json").is_file():
        offline.mkdir(parents=True, exist_ok=True)
        reset_usage()
        mined = mine_subflow_skill_backbone(subflow, train, artifact_dir=offline)
        for key, name in (("skill_md", "skill.md"), ("reference_md", "reference.md"),
                          ("action_rules_md", "action_rules.md"), ("slot_policies_md", "slot_policies.md")):
            (offline / name).write_text(mined.get(key, ""), encoding="utf-8")
        _json(offline / "subgraph.json", mined["subgraph"])
        _json(offline_usage_path, get_usage())
        _json(offline / "complete.json", {"subflow": subflow, "max_train": args.max_train,
                                           "train_sessions": len(train), "method": "backbone"})
    if need_offline:
        meta = _read(offline / "complete.json")
        if meta.get("subflow") != subflow or meta.get("max_train") != args.max_train:
            raise RuntimeError(f"Cached offline artifact has a different training scope: {offline}")

    for arm in execution_arms:
        if arm not in ("schema_only", "offline_only"):
            continue
        out = flow_dir / arm
        result_path = out / "result.json"
        if args.resume and result_path.is_file():
            continue
        out.mkdir(parents=True, exist_ok=True)
        reset_usage()
        workflow = WorkflowStore()
        reference = rules = slots = ""
        if arm == "schema_only":
            workflow.update(SCHEMA_SKILL)
        else:
            workflow.update((offline / "skill.md").read_text(encoding="utf-8"))
            reference = (offline / "reference.md").read_text(encoding="utf-8")
            rules = (offline / "action_rules.md").read_text(encoding="utf-8")
            slots = (offline / "slot_policies.md").read_text(encoding="utf-8")
        agent = ABCDAgent(model=args.model, workflow=workflow, workflow_max_chars=None,
                          memory=MemoryStore(),
                          reference_text=reference, action_rules_text=rules,
                          slot_policies_text=slots, expose_scenario_labels=False,
                          workflow_id=workflow_ids[0] if workflow_ids else None)
        result = evaluate_agent_on_subflow(
            agent, test, arm, subflow, save_dir=out,
            skip_utterance_eval=args.skip_utterance_eval,
            eval_workflow_ids=workflow_ids if len(workflow_ids) > 1
            and "fork" in mp.get_all_start_methods() else None,
        )
        worker_usage = result.pop("_evaluation_worker_usage", [])
        from scripts.llm_usage_utils import merge_usage_summaries
        testing = merge_usage_summaries(get_usage(), *worker_usage)
        generation = _read(offline_usage_path) if arm == "offline_only" else None
        result["llm_usage"] = split_usage_summary(generation, testing)
        result["subflow"] = subflow
        result["arm"] = arm
        result["test_sessions"] = len(test)
        _json(result_path, result)


def _run_flow(args: argparse.Namespace, subflow: str, workflow_ids: list[str]) -> None:
    flow_dir = Path(args.output_dir) / subflow
    flow_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ABCD_OUTPUT_DIR"] = str(flow_dir / "eval_aux")
    if workflow_ids:
        os.environ["SKILLMINING_WORKFLOW_ID"] = workflow_ids[0]
    _run_direct(args, subflow, flow_dir, workflow_ids)
    for arm in _execution_arms(args):
        if arm not in ("online_only", "full"):
            continue
        result_path = flow_dir / arm / "online_refine_result.json"
        if args.resume and result_path.is_file():
            continue
        _run_online(args, subflow, arm, flow_dir, workflow_ids)
    if "full" in _execution_arms(args):
        # The online runner accounts for its own calls; add the shared offline
        # compilation budget once to the full arm, without altering its raw log.
        from scripts.llm_usage_utils import merge_usage_summaries, split_usage_summary
        full_path = flow_dir / "full" / "online_refine_result.json"
        if full_path.is_file():
            full = _read(full_path)
            online_usage = full.get("llm_usage", {})
            offline_usage = _read(flow_dir / "offline_artifact" / "mining_usage.json")
            full["a2_total_usage"] = split_usage_summary(
                merge_usage_summaries(offline_usage, online_usage.get("generation")),
                online_usage.get("testing"),
            )
            _json(flow_dir / "full" / "a2_result.json", full)


def _aggregate(args: argparse.Namespace, subflows: list[str]) -> None:
    root = Path(args.output_dir)
    rows = []
    for arm in args.arms:
        for subflow in subflows:
            path = root / subflow / arm / ("result.json" if arm in ("schema_only", "offline_only")
                                           else "a2_result.json" if arm == "full"
                                           else "online_refine_result.json")
            if path.is_file():
                data = _read(path)
                rows.append({"subflow": subflow, "arm": arm, "joint_ast": data["ast_cds"]["ast_joint"],
                             "action": data["ast_cds"]["ast_action_name"],
                             "slot": data["ast_cds"]["ast_slot_value"],
                             "action_turns": data["ast_cds"]["num_action_turns"],
                             "usage": data.get("a2_total_usage") if arm == "full"
                             else data.get("llm_usage"), "result": str(path)})
    aggregate = {}
    for arm in args.arms:
        group = [row for row in rows if row["arm"] == arm]
        denom = sum(row["action_turns"] for row in group)
        aggregate[arm] = {
            "subflows": len(group), "action_turns": denom,
            **{key: round(sum(row[key] * row["action_turns"] for row in group) / denom, 6)
               if denom else None for key in ("joint_ast", "action", "slot")},
        }
    _json(root / "a2_aggregate.json", {"protocol": "ABCD independent coarse subflows",
                                         "arms": list(args.arms),
                                         "execution_arms": list(_execution_arms(args)),
                                         "skipped_arms": list(getattr(args, "skip_arms", ())),
                                         "rows": rows, "aggregate": aggregate})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subflow", help="One ABCD coarse subflow")
    group.add_argument("--subflows", help="Comma-separated ABCD coarse subflows for a pilot")
    group.add_argument("--all", action="store_true", help="All ten ABCD coarse subflows")
    parser.add_argument("--workflow-ids", default=os.environ.get("SKILLMINING_WORKFLOW_ID", ""),
                        help="Comma-separated workflow IDs; parallel across subflows or online batches")
    parser.add_argument("--output-dir", default=f"outputs/a2_abcd_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--arms", default=",".join(ARMS), help="Comma-separated subset of A2 arms")
    parser.add_argument("--skip-arms", default="",
                        help="Comma-separated configured arms to skip for this invocation only; "
                             "compatible with --resume without changing the experiment arms")
    parser.add_argument("--model", default="deepseek-chat", help="Base model (default: deepseek-chat)")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--refinement-mode", choices=("standard", "constrained-repair", "trace2skill-hybrid"),
                        default="standard")
    parser.add_argument("--skip-utterance-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show workflow assignment without API calls")
    parser.add_argument("--worker-subflow", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = str(Path(args.output_dir).resolve())
    args.arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if not args.arms or len(args.arms) != len(set(args.arms)) or any(arm not in ARMS for arm in args.arms):
        parser.error("--arms must be a unique comma-separated subset of " + ",".join(ARMS))
    args.skip_arms = tuple(item.strip() for item in args.skip_arms.split(",") if item.strip())
    if len(args.skip_arms) != len(set(args.skip_arms)) or any(arm not in args.arms for arm in args.skip_arms):
        parser.error("--skip-arms must be a unique comma-separated subset of --arms")
    if not _execution_arms(args):
        parser.error("--skip-arms cannot skip every configured arm")
    for key in ("max_train", "max_test", "max_batches", "batch_size"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    workflow_ids = _ids(args.workflow_ids)
    subflows = _subflows(args)
    if args.worker_subflow:
        _run_flow(args, args.worker_subflow, workflow_ids)
        return 0
    # The same workflow must never serve two concurrent subflow workers.
    buckets = {wid: [] for wid in workflow_ids} if len(subflows) > 1 and workflow_ids else {}
    if buckets:
        sizes = _read(SPLITS / "INDEX.json")
        load = {wid: 0 for wid in workflow_ids}
        for subflow in sorted(subflows, key=lambda sf: sizes[sf]["train"] + sizes[sf]["test"], reverse=True):
            wid = min(workflow_ids, key=lambda item: load[item])
            buckets[wid].append(subflow)
            load[wid] += sizes[subflow]["train"] + sizes[subflow]["test"]
    else:
        buckets = {",".join(workflow_ids): subflows}
    print(json.dumps({"output_dir": args.output_dir, "arms": args.arms,
                      "execution_arms": _execution_arms(args), "skipped_arms": args.skip_arms,
                      "assignments": buckets}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()) and not args.resume:
        parser.error("Output directory is nonempty; use --resume or a fresh --output-dir")
    config_path = root / "a2_config.json"
    if args.resume and config_path.is_file():
        prior = _read(config_path).get("arguments", {})
        for key in ("arms", "model", "max_train", "max_test", "max_batches",
                    "batch_size", "refinement_mode", "skip_utterance_eval"):
            current = list(args.arms) if key == "arms" else getattr(args, key)
            if prior.get(key) != current:
                parser.error(f"Resume configuration differs for {key}: {prior.get(key)!r} vs {current!r}")
    root.mkdir(parents=True, exist_ok=True)
    if not config_path.is_file():
        _json(config_path, {"arguments": {**vars(args), "arms": list(args.arms)},
                            "assignments": buckets})

    def worker(ids: str, names: list[str]) -> list[tuple[str, str]]:
        failures = []
        for name in names:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker-subflow", name,
                   "--subflow", name, "--workflow-ids", ids, "--output-dir", str(root),
                   "--arms", ",".join(args.arms), "--batch-size", str(args.batch_size),
                   "--refinement-mode", args.refinement_mode]
            if args.skip_arms:
                cmd += ["--skip-arms", ",".join(args.skip_arms)]
            if args.model:
                cmd += ["--model", args.model]
            for key in ("max_train", "max_test", "max_batches"):
                value = getattr(args, key)
                if value is not None:
                    cmd += ["--" + key.replace("_", "-"), str(value)]
            if args.skip_utterance_eval:
                cmd += ["--skip-utterance-eval"]
            if args.resume:
                cmd += ["--resume"]
            flow_dir = root / name
            flow_dir.mkdir(parents=True, exist_ok=True)
            with (flow_dir / "a2_worker.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                failures.append((name, str(flow_dir / "a2_worker.log")))
            else:
                print(f"Completed {name}", flush=True)
        return failures

    failures = []
    with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
        futures = [pool.submit(worker, ids, names) for ids, names in buckets.items()]
        for future in as_completed(futures):
            failures.extend(future.result())
    _aggregate(args, subflows)
    if failures:
        for name, log in failures:
            print(f"FAILED {name}: {log}", file=sys.stderr)
        return 1
    print(f"A2 aggregate: {root / 'a2_aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
