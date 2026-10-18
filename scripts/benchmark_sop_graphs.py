#!/usr/bin/env python3
"""Graph-only ABCD SOP-reference comparison. CPU only; all LLM calls forbidden."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skill_mining.sop_reference_benchmark import (build_reference, score_reference,
                                                 aggregate_reference, digest)
from scripts import eval_offline_graph_10flow as replay

DEFAULT_METHODS = ("support_lift", "discriminative", "heuristics", "trace_cover", "joint_trace_cover",
                   "joint_trace_cover_previous", "observable_trace_cover", "structural_trace_cover", "motif_trace_cover")
CONTROLS = ("empty", "complete", "reference_only", "full_dfg")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mine(method, flow, train):
    # Match the existing graph benchmark defaults; no parameters fitted here.
    local = dict(max_outgoing_edges=3, min_branch_support=2)
    cover = dict(min_branch_support=2, trace_coverage_target=0.8,
                 corpus_fitness_target=0.95, routing_complexity_weight=2.0)
    if method == "support_lift":
        return replay._mine_backbone_workflow_support_lift(flow, train, **local)["subgraph"]
    if method == "discriminative":
        return replay.mine_backbone_workflow(flow, train, **local)["subgraph"]
    if method == "heuristics":
        return replay.mine_backbone_workflow_heuristics(flow, train, **local)["subgraph"]
    if method == "joint_trace_cover_previous":
        cover.update(trace_coverage_target=0.85, corpus_fitness_target=0.94)
    functions = {"trace_cover": replay.mine_backbone_workflow_trace_cover,
                 "joint_trace_cover": replay.mine_backbone_workflow_joint_trace_cover,
                 "joint_trace_cover_previous": replay.mine_backbone_workflow_joint_trace_cover,
                 "observable_trace_cover": replay.mine_backbone_workflow_observable_trace_cover,
                 "structural_trace_cover": replay.mine_backbone_workflow_structural_trace_cover,
                 "motif_trace_cover": replay.mine_backbone_workflow_motif_trace_cover}
    return functions[method](flow, train, **cover)["subgraph"]


def control_graph(flow, pairs, observed):
    nodes = set(a for pair in pairs | observed for a in pair)
    local = {}
    def edge(a, b):
        return {"source": f"{flow}:{a}", "target": f"{flow}:{b}", "priority": 1, "support": 1, "score": 1.0}
    for a, b in sorted(pairs):
        local.setdefault(f"{flow}:{a}", []).append(edge(a, b))
    return {"nodes": [{"id": f"{flow}:{a}"} for a in sorted(nodes)],
            "edges": [edge(a, b) for a, b in sorted(pairs | observed)],
            "backbone": {"edges": []}, "local_transitions": local, "residual_edges": []}


def flow_run(flow, args, reference):
    train_path, test_path = [args.splits_dir / flow / (split + ".json") for split in ("train", "test")]
    train, test = read(train_path), read(test_path)
    if args.max_train is not None:
        train = train[:args.max_train]
    if args.max_test is not None:
        test = test[:args.max_test]
    sequences = [replay.action_sequence(flow, c) for c in train]
    observed_counts = Counter((a.split(":", 1)[-1], b.split(":", 1)[-1])
                              for seq in sequences for a, b in zip(seq, seq[1:]))
    observed = set(observed_counts)
    train_turns = sum(map(len, sequences))
    ref = reference["flows"][flow]
    positives = {(e["source"], e["target"]) for e in ref["positive"]}
    ref_nodes = set(a for seq in ref["routes"].values() for a in seq)
    reverse = {(e["source"], e["target"]) for e in ref["reverse_order"]}
    # The complete sanity control covers the reference vocabulary, not arbitrary unknown actions.
    controls = {"empty": set(), "reference_only": positives,
                "complete": {(a, b) for a in ref_nodes for b in ref_nodes}, "full_dfg": observed}
    rows = []
    for method in args.methods + tuple(args.cached_graphs) + CONTROLS:
        started = time.monotonic()
        if method in args.cached_graphs:
            graph = read(args.cached_graphs[method] / f"{flow}.json")
        else:
            graph = control_graph(flow, controls[method], observed) if method in CONTROLS else mine(method, flow, train)
        row = {"flow": flow, "method": method, "train_action_turns": train_turns,
               "train_sessions": len(train), "train_sha256": digest(train), "test_sha256": digest(test),
               "reference_sha256": digest(ref), "evaluation_role": "development-only",
               "sop": score_reference(graph, ref), "replay": replay.evaluate_graph(flow, graph, test),
               "elapsed_seconds": time.monotonic() - started,
               "reference_edge_candidate_ceiling": len(positives & observed) / len(positives)}
        row["reference_audit"] = {
            "reverse_probe_pairs_seen_in_train": len(reverse & observed),
            "train_reverse_transition_mass": sum(observed_counts[pair] for pair in reverse) / max(sum(observed_counts.values()), 1),
            "note": "Train-observed reverse probes may be valid retries/composition; this is not an illegality oracle."}
        row["replay"]["conditioned_branch_mrr"] = row["replay"][
            "structural_next_action_mrr" if graph.get("structural_refinement") or graph.get("motif_refinement")
            else "guard_next_action_mrr" if graph.get("observable_router") else "branch_next_action_mrr"]
        if method in args.cached_graphs:
            prior_row_path = args.cached_graphs[method].parent.parent / "per_flow" / args.cached_graphs[method].name / f"{flow}.json"
            prior_row = read(prior_row_path) if prior_row_path.is_file() else {}
            row["cached_graph"] = {"path": str(args.cached_graphs[method] / f"{flow}.json"),
                                   "sha256": digest(graph), "original_train_scope_verified": bool(prior_row)
                                   and prior_row.get("train_sha256") == digest(train)}
        # Save every graph separately, keeping all existing algorithms unchanged.
        dump(args.output_dir / "graphs" / method / f"{flow}.json", graph)
        dump(args.output_dir / "per_flow" / method / f"{flow}.json", row)
        print(f"{flow:22s} {method:26s} SOP={row['sop']['sop_discrimination']:.4f} "
              f"recall={row['sop']['sop_recall']:.4f} reverse_reject={row['sop']['reverse_order_rejection']} "
              f"edges={row['replay']['retained_edges']}", flush=True)
        rows.append(row)
    return rows


def summarize(rows):
    summaries = {}
    for method in dict.fromkeys(row["method"] for row in rows):
        group = [row for row in rows if row["method"] == method]
        sop = aggregate_reference([{**row["sop"], "train_action_turns": row["train_action_turns"]} for row in group])
        replay_summary = replay.aggregate([row["replay"] for row in group])
        branch_count = sum(row["replay"]["branch_transitions"] for row in group)
        replay_summary["conditioned_branch_mrr"] = sum(row["replay"]["conditioned_branch_mrr"]
            * row["replay"]["branch_transitions"] for row in group) / branch_count if branch_count else 0.0
        summaries[method] = {"sop": sop, "replay": replay_summary}
    return summaries


def markdown(summary):
    fmt = lambda value: f"{value:.4f}" if value is not None else "N/A"
    lines = ["# SOP reference graph benchmark (development)", "",
        "SOP-D = harmonic mean(reference adjacency recall, reverse nominal-order rejection).",
        "Explicitly unordered troubleshoot_site has NO negative class: report recall alone there; disclose recall-only weight.",
        "This is a structural proxy, NOT certified business compliance. Unknown edges are not negatives.",
        "SOP metrics: train-action-turn weighted per-flow scores; replay retains its original transition denominators.",
        "Edges are SUMS over flows. Refined/motif states are projected to canonical actions for SOP-D; their benefit is assessed separately by replay.", "",
        "| Method | SOP-D ↑ | SOP recall ↑ | Reverse rejection ↑ | Trace recall ↑ | Branch MRR ↑ | Conditioned MRR ↑ | Edges ↓ | Extra states ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, result in summary.items():
        s, r = result["sop"], result["replay"]
        lines.append(f"| {method} | {s['sop_discrimination']:.4f} | {s['sop_recall']:.4f} | "
                     f"{fmt(s['reverse_order_rejection'])} | {r['retained_transition_recall']:.4f} | "
                     f"{r['branch_next_action_mrr']:.4f} | {r['conditioned_branch_mrr']:.4f} | {r['retained_edges']} | {r['extra_structural_state_nodes']} |")
    lines += ["", "Empty scores 0, reference_only scores 1. Complete scores 0 on ordered flows and 1 on the explicitly unordered flow; it is NOT falsely labelled illegal there.",
              "Reference-only is privileged, not a deployable miner. Reverse rejection averages only flows with an order-negative class.",
              "No parameter sweep/algorithm changes were performed; all LLM calls were blocked.",
              "Previously used test.json replay is DEVELOPMENT evidence, not a new untouched final test."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=replay.DEFAULT_SPLITS)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/eval/abcd/data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / f"outputs/sop_graph_benchmark_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--flows", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-train", type=int, help="Sanity only; full train if omitted")
    parser.add_argument("--max-test", type=int, help="Sanity only; full development replay if omitted")
    parser.add_argument("--nominal-only", action="store_true", help="Sensitivity: omit the four sourced shortcut annotations")
    parser.add_argument("--cached-graphs", action="append", default=[], metavar="NAME=GRAPH_DIR",
                        help="Re-score saved per-flow graphs without re-mining; repeatable, source hashes recorded")
    args = parser.parse_args()
    args.methods = tuple(filter(None, args.methods.split(",")))
    cached = {}
    for item in args.cached_graphs:
        name, separator, directory = item.partition("=")
        if (not separator or not name or not all(c.isalnum() or c in "_-" for c in name)
                or name in cached or name in args.methods + CONTROLS):
            parser.error("--cached-graphs requires a unique safe NAME=GRAPH_DIR")
        cached[name] = Path(directory).resolve()
    args.cached_graphs = cached
    if ((not args.methods and not cached) or len(set(args.methods)) != len(args.methods)
            or set(args.methods) - set(DEFAULT_METHODS)):
        parser.error("Invalid/duplicate methods")
    if args.workers < 1 or any(x is not None and x < 1 for x in (args.max_train, args.max_test)):
        parser.error("Workers and sample caps must be positive")
    flows = args.flows.split(",") if args.flows else sorted(read(args.splits_dir / "INDEX.json"))
    if len(set(flows)) != len(flows) or (not args.flows and len(flows) != 10):
        parser.error("Expected ten unique coarse flows")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a fresh --output-dir; existing runs are never overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = build_reference(*(read(args.data_dir / name) for name in ("kb.json", "ontology.json", "guidelines.json")),
                                include_shortcuts=not args.nominal_only)
    dump(args.output_dir / "reference.json", reference)  # Frozen before any miner is run.
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        revision = "unknown"
    dump(args.output_dir / "config.json", {"arguments": {key: ({name: str(path) for name, path in value.items()}
         if key == "cached_graphs" else str(value) if isinstance(value, Path) else value)
         for key, value in vars(args).items()}, "flows": flows, "git_head": revision,
         "source_hashes": {name: digest((ROOT / name).read_text(encoding="utf-8")) for name in (
             "skill_mining/sop_reference_benchmark.py", "scripts/benchmark_sop_graphs.py",
             "skill_mining/backbone_workflow_mining.py", "skill_mining/joint_trace_cover.py")},
         "miner_defaults": {"max_outgoing_edges": 3, "min_branch_support": 2, "rho": 0.8, "eta": 0.95,
                            "routing_complexity_weight": 2.0, "previous_joint_rho_eta": [0.85, 0.94]},
         "llm_calls": 0, "reference_input_to_miners": False})
    rows, failures = [], []
    with patch("llm.chat", side_effect=AssertionError("LLM calls forbidden in SOP graph benchmark")) as blocked_calls:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(flow_run, flow, args, reference): flow for flow in flows}
            for future in as_completed(futures):
                flow = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    failures.append({"flow": flow, "error": repr(exc)})
                    print(f"FAILED {flow}: {exc}", flush=True)
    if blocked_calls.call_count:
        raise RuntimeError(f"Unexpected LLM call attempts: {blocked_calls.call_count}; not a graph-only run")
    rows.sort(key=lambda row: (row["method"], row["flow"]))
    summary = summarize(rows)
    for method in args.methods + tuple(args.cached_graphs) + CONTROLS:
        if method in summary and summary[method]["sop"]["num_flows"] != len(flows):
            failures.append({"method": method, "error": "incomplete flow coverage"})
    dump(args.output_dir / "results.json", {"reference_protocol": reference["protocol"],
        "llm_calls": 0, "complete": not failures, "failures": failures, "rows": rows, "aggregate": summary})
    report = markdown(summary)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"Saved: {args.output_dir}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
