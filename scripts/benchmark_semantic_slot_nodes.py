#!/usr/bin/env python3
"""Compare canonical vs selective action+semantic-slot graph identities, no rollout."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_sop_graphs import mine, read, dump, DEFAULT_METHODS
from skill_mining.semantic_slot_nodes import (DEFAULT_ACTIONS, make_policy, mine_with_node_identity,
                                             evaluate_common_action_targets, common_rates)
from skill_mining.sop_reference_benchmark import build_reference, score_reference, aggregate_reference, digest


def evaluate_flow(flow, args, policy, reference):
    train, test = [read(args.splits_dir / flow / f"{split}.json") for split in ("train", "test")]
    if args.max_train is not None:
        train = train[:args.max_train]
    if args.max_test is not None:
        test = test[:args.max_test]
    rows = []
    train_turns = sum(1 for conv in train for turn in conv.get("delexed", [])
                      if len(turn.get("targets", [])) >= 3 and turn["targets"][1] == "take_action")
    for method in args.methods:
        for mode in args.node_identity:
            cache = args.baseline_run / "graphs" / method / f"{flow}.json" if args.baseline_run else None
            cached = False
            if mode == "action" and cache is not None and cache.is_file():
                provenance = read(args.baseline_run / "per_flow" / method / f"{flow}.json")
                if provenance.get("train_sha256") != digest(train):
                    raise ValueError(f"Cached baseline train scope mismatch: {cache}")
                graph, cached = read(cache), True
            else:
                graph = mine_with_node_identity(lambda f, data: mine(method, f, data), flow, train,
                                               node_identity=mode, policy=policy)
            result = evaluate_common_action_targets(flow, graph, train, test, policy)
            sop = score_reference(graph, reference["flows"][flow])
            retained_edges = sum(len(edges) for edges in graph.get("local_transitions", {}).values())
            row = {"flow": flow, "method": method, "node_identity": mode,
                   "train_action_turns": train_turns, "train_sha256": digest(train), "test_sha256": digest(test),
                   "baseline_cached": cached, "baseline_source": str(cache) if cached else None,
                   "nodes": len(graph["nodes"]), "edges": retained_edges,
                   "extra_nodes": graph.get("node_identity", {}).get("extra_nodes", 0),
                   "typed_events": graph.get("node_identity", {}).get("typed_action_events", 0),
                   "common_action_prediction": result, "sop_projected": sop}
            dump(args.output_dir / "graphs" / method / mode / f"{flow}.json", graph)
            dump(args.output_dir / "per_flow" / method / mode / f"{flow}.json", row)
            show = lambda value: f"{value:.4f}" if value is not None else "N/A"
            print(f"{flow:22s} {method:20s} {mode:13s} nodes={row['nodes']} edges={retained_edges} "
                  f"actionMRR={show(result['next_action_mrr'])} recall={show(result['next_action_recall'])}", flush=True)
            rows.append(row)
    return rows


def aggregate(rows):
    result = {}
    for method, mode in sorted({(row["method"], row["node_identity"]) for row in rows}):
        group = [row for row in rows if (row["method"], row["node_identity"]) == (method, mode)]
        keys = ("transitions", "hits", "top1_hits", "reciprocal_rank_sum", "candidate_sum",
                "branch_transitions", "branch_hits", "branch_reciprocal_rank_sum", "branch_candidate_sum",
                "source_backoffs", "training_transitions", "training_entropy_sum")
        counts = {key: sum(row["common_action_prediction"].get(key, 0) for row in group) for key in keys}
        result[f"{method}/{mode}"] = {"flows": len(group),
            "nodes": sum(row["nodes"] for row in group), "edges": sum(row["edges"] for row in group),
            "extra_nodes": sum(row["extra_nodes"] for row in group),
            "typed_events": sum(row["typed_events"] for row in group),
            "common_action_prediction": {**counts, **common_rates(counts),
                "training_conditional_entropy": counts["training_entropy_sum"] / max(counts["training_transitions"], 1)},
            "sop_projected": aggregate_reference([{**row["sop_projected"], "train_action_turns": row["train_action_turns"]}
                                                  for row in group])}
    return result


def report(summary):
    lines = ["# Selective action + semantic-slot nodes", "",
        "All targets are canonical NEXT ACTIONS in both arms. No next-target slots select the source state.",
        "Rank outgoing canonical actions by retained training support; merge target-role weights.",
        "Replay scores use common transition/branch denominators. Sizes are sums over domains.",
        "SOP-D projects identities to API actions and cannot measure role separation itself. Train entropy is only a diagnostic.", "",
        "| Method/identity | Nodes | Edges | Action recall | Action MRR | Branch MRR | Branch candidates | Train H(next action|node) | Projected SOP-D |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary.items():
        p = row["common_action_prediction"]
        fmt = lambda value: f"{value:.4f}" if value is not None else "N/A"
        lines.append(f"| {name} | {row['nodes']} | {row['edges']} | {fmt(p['next_action_recall'])} | "
                     f"{fmt(p['next_action_mrr'])} | {fmt(p['branch_mrr'])} | {fmt(p['branch_average_candidates'])} | "
                     f"{fmt(p['training_conditional_entropy'])} | {fmt(row['sop_projected']['sop_discrimination'])} |")
    lines += ["", "This is a separate graph-only branch. API actions/slot targets and the existing runtime are not changed.",
              "Enum policies are explicit domain priors; frequent IDs, names, emails and amounts never become node categories.",
              "The old test.json has been used for development; these are NOT untouched final-test results."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/eval/abcd/splits")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/eval/abcd/data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / f"outputs/semantic_slot_nodes_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--methods", default="discriminative,trace_cover,joint_trace_cover")
    parser.add_argument("--node-identity", default="action,semantic-slot", help="action, semantic-slot, or both comma-separated")
    parser.add_argument("--semantic-slot-actions", default=",".join(DEFAULT_ACTIONS), help="Conservative enum-typed action whitelist")
    parser.add_argument("--baseline-run", type=Path, help="Reuse existing baseline graphs after verifying training hashes")
    parser.add_argument("--flows", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    args = parser.parse_args()
    args.methods = tuple(filter(None, args.methods.split(",")))
    args.node_identity = tuple(filter(None, args.node_identity.split(",")))
    if not args.methods or set(args.methods) - set(DEFAULT_METHODS) or len(set(args.methods)) != len(args.methods):
        parser.error("Invalid methods")
    if not args.node_identity or set(args.node_identity) - {"action", "semantic-slot"} or len(set(args.node_identity)) != len(args.node_identity):
        parser.error("Invalid --node-identity")
    if args.workers < 1 or any(x is not None and x < 1 for x in (args.max_train, args.max_test)):
        parser.error("Workers/caps must be positive")
    policy = make_policy(tuple(filter(None, args.semantic_slot_actions.split(","))))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a fresh output directory")
    flows = args.flows.split(",") if args.flows else sorted(read(args.splits_dir / "INDEX.json"))
    if len(set(flows)) != len(flows) or (not args.flows and len(flows) != 10):
        parser.error("Expected ten unique flows")
    reference = build_reference(*(read(args.data_dir / name) for name in ("kb.json", "ontology.json", "guidelines.json")))
    dump(args.output_dir / "reference.json", reference)
    dump(args.output_dir / "config.json", {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
         "node_policy": policy, "policy_sha256": digest(policy), "graph_only": True,
         "source_sha256": {name: digest((ROOT / name).read_text(encoding="utf-8")) for name in (
             "skill_mining/semantic_slot_nodes.py", "scripts/benchmark_semantic_slot_nodes.py",
             "skill_mining/backbone_workflow_mining.py", "skill_mining/joint_trace_cover.py")},
         "llm_calls": 0, "reference_used_by_miners": False})
    rows, failures = [], []
    with patch("llm.chat", side_effect=AssertionError("No LLM calls permitted")) as blocked:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(evaluate_flow, flow, args, policy, reference): flow for flow in flows}
            for future in as_completed(futures):
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    failures.append({"flow": futures[future], "error": repr(exc)})
                    print(f"FAILED {futures[future]}: {exc}", flush=True)
    if blocked.call_count:
        failures.append({"error": f"{blocked.call_count} attempted LLM calls"})
    rows.sort(key=lambda row: (row["method"], row["node_identity"], row["flow"]))
    summary = aggregate(rows)
    dump(args.output_dir / "results.json", {"protocol": "semantic_slot_graph_comparison_v1", "rows": rows,
         "aggregate": summary, "complete": not failures, "failures": failures, "llm_calls": 0})
    text = report(summary)
    (args.output_dir / "report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"Saved: {args.output_dir}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
