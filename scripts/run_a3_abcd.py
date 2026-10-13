#!/usr/bin/env python3
"""A3: frozen ABCD structure ablation from one shared offline artifact."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_a2_abcd import _ids, _json, _read, _subflows

ARMS = ("flat", "backbone_only", "backbone_residual")
VERSION = "a3-shared-cards-v1"


def _hash(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pairs(edges) -> dict[tuple[str, str], dict]:
    return {(str(edge["source"]), str(edge["target"])): dict(edge) for edge in edges}


def build_variants(subflow: str, graph: dict, action_rules: str, slot_policies: str) -> dict:
    """Project identical induced semantics, without additional LLM compilation.

    The flat control reuses the old unordered compiler's stable hash ordering.
    Every variant has the SAME per-action/slot resources. Only explicit graph
    layout and transition-card availability/organization change. No raw dialogue
    reference is exposed, since it would reintroduce removed residual routes.
    """
    from skill_mining.skill_writer import _render_backbone_tree, _stable_unordered

    nodes = {str(node["id"]): node for node in graph.get("nodes", [])}
    root = str(graph.get("backbone", {}).get("root", "<START>"))
    tree = _pairs(graph.get("backbone", {}).get("edges", []))
    selected = dict(tree)
    selected.update(_pairs(graph.get("residual_edges", [])))
    for edges in graph.get("local_transitions", {}).values():
        selected.update(_pairs(edges))
    # Artificial root edges organize the tree but are not dialogue transitions.
    selected = {pair: edge for pair, edge in selected.items() if pair[0] in nodes and pair[1] in nodes}
    tree_pairs = set(tree) & set(selected)
    rules = {(str(source), str(rule["target"])): rule
             for source, rows in graph.get("transition_induction", {}).get("rules_by_source", {}).items()
             for rule in rows if rule.get("target")}
    missing = set(selected) - set(rules)
    if missing:
        raise ValueError(f"Shared artifact lacks induced transition cards: {sorted(missing)}")
    if not action_rules.strip() or not slot_policies.strip():
        raise ValueError("Shared action and slot resources must be nonempty")
    order = {str(node): index for index, node in enumerate(graph.get("backbone", {}).get("compilation_order", []))}
    labels = lambda pair: (str(nodes[pair[0]]["label"]), str(nodes[pair[1]]["label"]))
    shared = {
        "action_rules.md": action_rules, "slot_policies.md": slot_policies,
    }
    result = {}
    for arm in ARMS:
        active = selected if arm != "backbone_only" else {pair: selected[pair] for pair in tree_pairs}
        pairs = list(active)
        if arm == "flat":
            pairs = _stable_unordered(pairs, subflow, lambda pair: f"{pair[0]}->{pair[1]}")
            node_order = _stable_unordered(list(nodes), subflow, lambda node: node)
            layout = "## Unordered Actions\n" + "\n".join(f"- `{nodes[node]['label']}`" for node in node_order)
            instruction = "Transition cards have no preferred order or main-path designation."
        else:
            pairs.sort(key=lambda pair: (order.get(pair[0], 10**9), pair not in tree_pairs, pair))
            layout = "## Backbone\n```text\n" + _render_backbone_tree(graph) + "\n```"
            instruction = "The backbone organizes observed behavior; it is not a mandatory execution constraint."
        skill = (f"# Skill: {subflow}\n\n{layout}\n\n{instruction}\n"
                 "Retrieve transition evidence when deciding the next action. Retrieve the shared "
                 "action rule and slot policy for procedural details. Use only values grounded in the current dialogue.\n")
        reference = [f"# Transition cards: {subflow}", ""]
        for pair in pairs:
            source, target = labels(pair)
            rule = rules[pair]
            reference.extend([f"## {source} -> {target}", ""])
            if arm != "flat":
                reference.append("- Placement: " + ("backbone" if pair in tree_pairs else "residual"))
            reference.extend([f"- Induction status: {rule.get('status', 'underspecified')}",
                              "- Condition: " + (str(rule.get("condition") or "").strip()
                                                or "Insufficient observable evidence to identify a unique trigger."), ""])
        resources = {**shared, "skill.md": skill, "reference.md": "\n".join(reference).rstrip() + "\n"}
        result[arm] = {"resources": resources, "edges": list(active.values()),
                       "tree_pairs": tree_pairs, "nodes": list(nodes), "root": root,
                       "resource_hashes": {name: _hash(text) for name, text in resources.items()}}
    return result


def graph_metrics(subflow: str, variant: dict, conversations: list[dict]) -> dict:
    """Coverage uses held-out action transitions, not LLM rollout predictions."""
    from scripts.eval_offline_graph_10flow import action_sequence
    pairs = set(_pairs(variant["edges"]))
    nodes = set(variant["nodes"])
    degree = Counter(source for source, _ in pairs)
    events = hits = actions = covered_actions = routes = complete = 0
    for conversation in conversations:
        sequence = action_sequence(subflow, conversation)
        transitions = list(zip(sequence, sequence[1:]))
        actions += len(sequence)
        covered_actions += sum(node in nodes for node in sequence)
        events += len(transitions)
        hits += sum(pair in pairs for pair in transitions)
        if transitions:
            routes += 1
            complete += int(all(pair in pairs for pair in transitions))
    return {"nodes": len(nodes), "edges": len(pairs),
            "backbone_edges": len(pairs & variant["tree_pairs"]),
            "residual_edges": len(pairs - variant["tree_pairs"]),
            "branch_sources": sum(count > 1 for count in degree.values()),
            "max_out_degree": max(degree.values(), default=0),
            "routing_complexity": sum(math.log2(count) for count in degree.values()),
            "transition_events": events, "covered_transitions": hits,
            "transition_coverage": hits / events if events else None,
            "action_events": actions, "covered_actions": covered_actions,
            "action_coverage": covered_actions / actions if actions else None,
            "route_dialogues": routes, "complete_routes": complete,
            "complete_route_rate": complete / routes if routes else None}


def _prepare(args, subflow, train, flow_dir):
    from scripts.llm_usage_utils import get_usage, reset_usage
    from scripts.run_subflow_eval import mine_subflow_skill_backbone
    source = Path(args.offline_root) / subflow / "offline_artifact" if args.offline_root else flow_dir / "offline_artifact"
    required = ("subgraph.json", "action_rules.md", "slot_policies.md", "mining_usage.json", "complete.json")
    if not args.offline_root and not (source / "complete.json").is_file():
        source.mkdir(parents=True, exist_ok=True)
        reset_usage()
        mined = mine_subflow_skill_backbone(subflow, train, artifact_dir=source)
        for key, name in (("skill_md", "skill.md"), ("reference_md", "reference.md"),
                          ("action_rules_md", "action_rules.md"), ("slot_policies_md", "slot_policies.md")):
            (source / name).write_text(mined.get(key, ""), encoding="utf-8")
        _json(source / "subgraph.json", mined["subgraph"])
        _json(source / "mining_usage.json", get_usage())
        _json(source / "complete.json", {"subflow": subflow, "max_train": args.max_train,
                                         "train_hash": _hash(train), "method": "backbone"})
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Incomplete shared offline artifact: {source / name}")
    meta = _read(source / "complete.json")
    if meta.get("subflow") != subflow or meta.get("max_train") != args.max_train:
        raise ValueError("Shared artifact's subflow/max_train differs from this experiment")
    if meta.get("train_hash") and meta["train_hash"] != _hash(train):
        raise ValueError("Shared artifact training data hash differs")
    graph = _read(source / "subgraph.json")
    if "transition_induction" not in graph:
        raise ValueError("Shared graph must contain transition_induction; graph-only outputs are insufficient")
    variants = build_variants(subflow, graph, (source / "action_rules.md").read_text(encoding="utf-8"),
                              (source / "slot_policies.md").read_text(encoding="utf-8"))
    return source, variants, _read(source / "mining_usage.json")


def _run_flow(args, subflow, workflow_ids):
    flow_dir = Path(args.output_dir) / subflow
    flow_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ABCD_OUTPUT_DIR"] = str(flow_dir / "eval_aux")
    if workflow_ids:
        os.environ["SKILLMINING_WORKFLOW_ID"] = workflow_ids[0]
    from awm import MemoryStore, WorkflowStore
    from eval_tod.abcd.agent import ABCDAgent
    from scripts.llm_usage_utils import get_usage, reset_usage, merge_usage_summaries, split_usage_summary
    from scripts.run_subflow_eval import load_subflow_data, evaluate_agent_on_subflow

    train, test = load_subflow_data(subflow)
    if args.max_train is not None:
        train = train[:args.max_train]
    if args.max_test is not None:
        test = test[:args.max_test]
    source, variants, mining_usage = _prepare(args, subflow, train, flow_dir)
    for arm in args.arms:
        if arm in args.skip_arms:
            continue
        variant = variants[arm]
        output = flow_dir / arm
        manifest = {"version": VERSION, "subflow": subflow, "arm": arm,
                    "source": str(source), "train_hash": _hash(train), "test_hash": _hash(test),
                    "resource_hashes": variant["resource_hashes"]}
        if args.resume and (output / "result.json").is_file():
            if _read(output / "manifest.json") != manifest:
                raise ValueError(f"Completed arm has changed resources/data: {output}")
            continue
        output.mkdir(parents=True, exist_ok=True)
        for name, text in variant["resources"].items():
            (output / name).write_text(text, encoding="utf-8")
        _json(output / "manifest.json", manifest)
        metrics = graph_metrics(subflow, variant, test)
        _json(output / "graph_metrics.json", metrics)
        _json(output / "available_edges.json", variant["edges"])
        workflow = WorkflowStore()
        workflow.update(variant["resources"]["skill.md"])
        agent = ABCDAgent(model=args.model, workflow=workflow, memory=MemoryStore(),
                          workflow_max_chars=args.workflow_max_chars,
                          reference_text=variant["resources"]["reference.md"],
                          reference_top_k=args.reference_top_k, reference_max_chars=args.reference_max_chars,
                          action_rules_text=variant["resources"]["action_rules.md"],
                          slot_policies_text=variant["resources"]["slot_policies.md"],
                          action_selection_candidate_limit=args.action_selection_candidate_limit,
                          expose_scenario_labels=False, workflow_id=workflow_ids[0] if workflow_ids else None)
        reset_usage()
        result = evaluate_agent_on_subflow(agent, test, arm, subflow, save_dir=output,
                  skip_utterance_eval=args.skip_utterance_eval,
                  eval_workflow_ids=workflow_ids if len(workflow_ids) > 1 and "fork" in mp.get_all_start_methods() else None)
        workers = result.pop("_evaluation_worker_usage", [])
        testing = merge_usage_summaries(get_usage(), *workers)
        result.update(subflow=subflow, arm=arm, graph_metrics=metrics,
                      llm_usage=split_usage_summary(mining_usage, testing),
                      resource_chars={name: len(text) for name, text in variant["resources"].items()},
                      protocol=VERSION, online_refinement=False)
        _json(output / "result.json", result)


def _aggregate(args, subflows):
    rows, summary = [], {}
    for arm in args.arms:
        group = []
        for subflow in subflows:
            path = Path(args.output_dir) / subflow / arm / "result.json"
            if path.is_file():
                row = _read(path)
                group.append(row)
                rows.append(row)
        denom = sum(row["ast_cds"]["num_action_turns"] for row in group)
        total_edges = sum(row["graph_metrics"]["edges"] for row in group)
        entry = {"completed_subflows": len(group), "expected_subflows": len(subflows),
                 "complete": len(group) == len(subflows), "action_turns": denom,
                 "total_edges": total_edges, "mean_edges": total_edges / len(group) if group else None,
                 "total_nodes": sum(row["graph_metrics"]["nodes"] for row in group),
                 "total_branch_sources": sum(row["graph_metrics"].get("branch_sources", 0) for row in group),
                 "total_routing_complexity": sum(row["graph_metrics"].get("routing_complexity", 0) for row in group)}
        for name, key in (("joint_ast", "ast_joint"), ("action", "ast_action_name"), ("slot", "ast_slot_value")):
            entry[name] = sum(row["ast_cds"][key] * row["ast_cds"]["num_action_turns"] for row in group) / denom if denom else None
        for metric, num, den in (("transition_coverage", "covered_transitions", "transition_events"),
                                 ("complete_route_rate", "complete_routes", "route_dialogues"),
                                 ("action_coverage", "covered_actions", "action_events")):
            n = sum(row["graph_metrics"][num] for row in group)
            d = sum(row["graph_metrics"][den] for row in group)
            entry[metric] = n / d if d else None
            entry[den] = d
        # Missing token accounting is never interpreted as zero inference cost.
        buckets = [row.get("llm_usage", {}).get("testing", {}).get("total", {}) for row in group]
        available = bool(group) and all(bucket.get("usage_available") for bucket in buckets)
        entry["runtime_tokens_available"] = available
        entry["runtime_tokens"] = sum(bucket["total_tokens"] for bucket in buckets) if available else None
        entry["runtime_tokens_per_action_turn"] = entry["runtime_tokens"] / denom if available and denom else None
        entry["runtime_calls"] = sum(bucket["calls"] for bucket in buckets) if group and all("calls" in bucket for bucket in buckets) else None
        entry["runtime_usage_sources"] = sorted({bucket.get("usage_source", "unavailable") for bucket in buckets})
        summary[arm] = entry
    _json(Path(args.output_dir) / "a3_aggregate.json", {"protocol": VERSION, "arms": list(args.arms),
          "skipped_arms": list(args.skip_arms), "rows": rows, "aggregate": summary})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--subflow")
    group.add_argument("--subflows", default="")
    parser.add_argument("--output-dir", default=f"outputs/a3_abcd_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--offline-root", help="Reuse A2 root/<subflow>/offline_artifact; read only")
    parser.add_argument("--workflow-ids", default=os.environ.get("SKILLMINING_WORKFLOW_ID", ""))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--skip-arms", default="")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--workflow-max-chars", type=int, default=8000)
    parser.add_argument("--reference-top-k", type=int, default=3)
    parser.add_argument("--reference-max-chars", type=int, default=1800)
    parser.add_argument("--action-selection-candidate-limit", type=int, default=3)
    parser.add_argument("--skip-utterance-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-subflow", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = str(Path(args.output_dir).resolve())
    if args.offline_root:
        args.offline_root = str(Path(args.offline_root).resolve())
    for key in ("arms", "skip_arms"):
        values = tuple(item.strip() for item in getattr(args, key).split(",") if item.strip())
        if len(values) != len(set(values)) or any(item not in ARMS for item in values):
            parser.error(f"--{key.replace('_', '-')} must contain unique valid arms")
        setattr(args, key, values)
    if not args.arms or not set(args.skip_arms) <= set(args.arms) or not set(args.arms) - set(args.skip_arms):
        parser.error("At least one configured arm must execute; skipped arms must belong to --arms")
    for key, minimum in (("max_train", 1), ("max_test", 1), ("workflow_max_chars", 1000),
                         ("reference_top_k", 0), ("reference_max_chars", 200), ("action_selection_candidate_limit", 1)):
        value = getattr(args, key)
        if value is not None and value < minimum:
            parser.error(f"--{key.replace('_', '-')} must be >= {minimum}")
    ids, subflows = _ids(args.workflow_ids), _subflows(args)
    if args.worker_subflow:
        _run_flow(args, args.worker_subflow, ids)
        return 0
    buckets = {wid: subflows[index::len(ids)] for index, wid in enumerate(ids)} if len(ids) > 1 and len(subflows) > 1 else {",".join(ids): subflows}
    buckets = {wid: names for wid, names in buckets.items() if names}
    print(json.dumps({"output_dir": args.output_dir, "arms": args.arms, "skip_arms": args.skip_arms,
                      "assignments": buckets, "online_refinement": False}, indent=2), flush=True)
    if args.dry_run:
        return 0
    root = Path(args.output_dir)
    config = root / "a3_config.json"
    settings = {key: value for key, value in vars(args).items()
                if key not in {"resume", "dry_run", "skip_arms", "workflow_ids", "worker_subflow"}}
    settings.update(arms=list(args.arms), version=VERSION, selected_subflows=subflows)
    if root.exists() and any(root.iterdir()) and not args.resume:
        parser.error("Output directory is nonempty; use --resume or a fresh directory")
    if args.resume and root.exists() and any(root.iterdir()) and not config.is_file():
        parser.error("Cannot safely resume without a3_config.json")
    if config.is_file() and _read(config) != settings:
        parser.error("Resume experimental settings differ from a3_config.json")
    root.mkdir(parents=True, exist_ok=True)
    if not config.is_file():
        _json(config, settings)

    def worker(wids, names):
        failures = []
        for name in names:
            command = [sys.executable, str(Path(__file__).resolve()), "--subflow", name,
                       "--worker-subflow", name, "--output-dir", str(root), "--workflow-ids", wids,
                       "--arms", ",".join(args.arms)]
            for key in ("model", "offline_root", "max_train", "max_test", "workflow_max_chars", "reference_top_k",
                        "reference_max_chars", "action_selection_candidate_limit"):
                value = getattr(args, key)
                if value is not None:
                    command += ["--" + key.replace("_", "-"), str(value)]
            if args.skip_arms:
                command += ["--skip-arms", ",".join(args.skip_arms)]
            for key in ("resume", "skip_utterance_eval"):
                if getattr(args, key):
                    command += ["--" + key.replace("_", "-")]
            directory = root / name
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "a3_worker.log").open("a", encoding="utf-8") as log:
                log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
                log.flush()
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                failures.append(name)
        return failures

    failures = []
    with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
        for future in as_completed([pool.submit(worker, wid, names) for wid, names in buckets.items()]):
            failures.extend(future.result())
    _aggregate(args, subflows)
    if failures:
        print("FAILED subflows: " + ", ".join(failures), file=sys.stderr)
        return 1
    print(f"A3 aggregate: {root / 'a3_aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
