#!/usr/bin/env python3
"""Compare old tree TraceCover with graph log/support objectives, CPU only."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
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

from scripts import benchmark_sop_graphs as bench
from scripts import eval_offline_graph_10flow as replay
from skill_mining.sop_reference_benchmark import build_reference, digest, score_reference
from skill_mining.treefree_trace_cover import mine_treefree_trace_cover


def common_ranking(graph):
    ranked = copy.deepcopy(graph)
    for outgoing in ranked["local_transitions"].values():
        outgoing.sort(key=lambda row: (-row.get("num_sessions", 0), -row.get("support", 0),
                                       -row.get("score", 0), row["target"]))
        for index, row in enumerate(outgoing, 1):
            row["priority"] = index
    return ranked


def graph_metrics(graph, flow, train, rho, gamma):
    sequences = [replay.action_sequence(flow, row) for row in train]
    sessions = [Counter(zip(seq, seq[1:])) for seq in sequences if len(seq) > 1]
    selected = {(edge["source"], edge["target"])
                for outgoing in graph["local_transitions"].values() for edge in outgoing}
    occurrence = Counter()
    for counts in sessions:
        occurrence.update(counts)
    source = Counter()
    for (u, _), count in occurrence.items():
        source[u] += count
    degree = Counter(u for u, _ in selected)
    total = sum(occurrence.values())
    cost = len(selected) + gamma * sum(source[u] / total * d * (d-1) / 2 for u, d in degree.items())
    roots = {seq[0] for seq in sequences if seq}
    seen = set(roots)
    while True:
        more = {v for u, v in selected if u in seen} - seen
        if not more:
            break
        seen.update(more)
    return {"coverage": sum(min(1, sum(count for edge, count in counts.items() if edge in selected)
                                      / (rho * sum(counts.values()))) for counts in sessions) / len(sessions),
            "cost": cost, "reachable": {row["id"] for row in graph["nodes"]} <= seen,
            "rare_selected_edges": sum(occurrence[edge] < 2 for edge in selected),
            "self_loops": sum(u == v for u, v in selected)}


def compare_flow(flow, args, reference):
    train, test = [bench.read(args.splits_dir / flow / f"{split}.json") for split in ("train", "test")]
    start = time.perf_counter()
    graphs = {}
    old = replay.mine_backbone_workflow_joint_trace_cover(
        flow, train, min_branch_support=2, trace_coverage_target=args.rho,
        corpus_fitness_target=args.eta, routing_complexity_weight=args.gamma,
        tree_evidence_weight=args.beta, max_moves=args.quick_moves, search_width=args.old_shortlist)
    graphs["old_tree"] = old["subgraph"]
    elapsed = {"old_tree": time.perf_counter()-start}
    for evidence in ("log", "support", "support_mean_matched", "none"):
        start = time.perf_counter()
        result = mine_treefree_trace_cover(flow, train, evidence=evidence,
            rho=args.rho, eta=args.eta, gamma=args.gamma, beta=args.beta,
            max_moves=args.max_moves, shortlist_size=args.shortlist,
            snapshot_moves=(args.quick_moves,), progress=lambda message: print(message, flush=True))
        key = f"graph_{evidence}"
        graphs[key] = result["subgraph"]
        graphs[key + f"_budget{args.quick_moves}"] = result["snapshots"][args.quick_moves]
        elapsed[key] = time.perf_counter()-start
        elapsed[key + f"_budget{args.quick_moves}"] = result["snapshots"][args.quick_moves]["treefree_trace_cover"]["elapsed_seconds"]
    rows = []
    for method, graph in graphs.items():
        metrics = graph_metrics(graph, flow, train, args.rho, args.gamma)
        assert metrics["coverage"] >= args.eta - 1e-8, (flow, method, metrics)
        assert metrics["reachable"], (flow, method)
        if "treefree_trace_cover" in graph:
            meta = graph["treefree_trace_cover"]
            assert abs(metrics["coverage"]-meta["coverage"]) < 1e-9
            assert abs(metrics["cost"]-meta["cost"]) < 1e-8
        row = {"flow": flow, "method": method, "train_action_turns": sum(len(replay.action_sequence(flow, c)) for c in train),
               "train_dialogues": len(train), "train_sha256": digest(train), "test_sha256": digest(test),
               "reference_sha256": digest(reference["flows"][flow]), "evaluation_role": "development-only",
               "sop": score_reference(graph, reference["flows"][flow]),
               "replay": replay.evaluate_graph(flow, graph, test),
               "common_rank_replay": replay.evaluate_graph(flow, common_ranking(graph), test),
               "structure": metrics, "mining_seconds": elapsed[method],
               "search": graph.get("treefree_trace_cover", graph.get("joint_trace_cover"))}
        bench.dump(args.output_dir / "graphs" / method / f"{flow}.json", graph)
        bench.dump(args.output_dir / "per_flow" / method / f"{flow}.json", row)
        rows.append(row)
        print(f"DONE {flow:22s} {method:32s} SOP={row['sop']['sop_discrimination']:.4f} "
              f"trace={row['replay']['retained_transition_recall']:.4f} edges={row['replay']['retained_edges']} "
              f"moves={len(row['search']['accepted_moves'])}", flush=True)
    return rows


def summarize(rows):
    compatible = [{**row, "replay": {**row["replay"],
                   "conditioned_branch_mrr": row["replay"]["branch_next_action_mrr"]}} for row in rows]
    summary = bench.summarize(compatible)
    for method, result in summary.items():
        group = [row for row in rows if row["method"] == method]
        common = replay.aggregate([row["common_rank_replay"] for row in group])
        result["common_rank_branch_mrr"] = common["branch_next_action_mrr"]
        result["mining_seconds"] = sum(row["mining_seconds"] for row in group)
        result["cost_sum"] = sum(row["structure"]["cost"] for row in group)
        result["moves"] = sum(len(row["search"]["accepted_moves"]) for row in group)
        result["stop_reasons"] = dict(Counter(row["search"]["stop_reason"] for row in group))
        result["all_reachable"] = all(row["structure"]["reachable"] for row in group)
        result["minimum_training_coverage"] = min(row["structure"]["coverage"] for row in group)
    return summary


def report(summary, rows, args):
    lines = ["# Tree-free TraceCover weighting pilot", "",
        f"ABCD {len(set(r['flow'] for r in rows))} coarse flows, full training splits; existing development replay and frozen SOP v2 probes.",
        f"rho={args.rho}, eta={args.eta}, gamma={args.gamma}, beta={args.beta}; old max_moves={args.quick_moves}; graph max_moves={args.max_moves}; old shortlist={args.old_shortlist}; graph shortlist={args.shortlist}.",
        "No LLM calls, skill compilation, agent rollouts, or SR measurement. Reference is not an input to miners.", "",
        "log: existing log(1+distinct dialogues) + 0.5 log(lift).",
        "support: distinct supporting dialogues / ALL training dialogues.",
        "support_mean_matched: support multiplied by mean(log weights)/mean(support weights), computed from training data per flow.",
        "none: beta*W is zero; graph-complexity-only control.",
        "All graph variants start from the full observed DFG. The old miner uses its original initializer and support-filtered pool.",
        "budget snapshots separate the new initialization cost from convergence. No global-optimality claim.",
        "Graph residual_edges means unselected observations; old residual_edges means selected non-tree edges, so those counts are not compared.",
        "Common-rank MRR orders ALL methods by distinct-dialogue support, then occurrence support, then old log score; native MRR remains in raw results.", "",
        "| Method | SOP-D | SOP recall | Reverse rejection | Dev trace recall | Edges | Common-rank branch MRR | Moves | Mining seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, result in summary.items():
        s,r=result["sop"],result["replay"]
        lines.append(f"| {method} | {s['sop_discrimination']:.4f} | {s['sop_recall']:.4f} | "
                     f"{s['reverse_order_rejection']:.4f} | {r['retained_transition_recall']:.4f} | "
                     f"{r['retained_edges']} | {result['common_rank_branch_mrr']:.4f} | {result['moves']} | {result['mining_seconds']:.2f} |")
    lines += ["", "## Per-flow final comparison", "",
              "| Flow | Method | SOP-D | Dev trace recall | Edges | F_train | C | Stop |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        if "_budget" not in row["method"]:
            lines.append(f"| {row['flow']} | {row['method']} | {row['sop']['sop_discrimination']:.4f} | "
                         f"{row['replay']['retained_transition_recall']:.4f} | {row['replay']['retained_edges']} | "
                         f"{row['structure']['coverage']:.4f} | {row['structure']['cost']:.2f} | {row['search']['stop_reason']} |")
    lines += ["", "SOP-D is the existing structural development proxy, not complete SOP gold or business compliance.",
              "Scores compare model variants and weighting scales, not an isolated proof of benefit from tree removal.",
              "Mining times are CPU measurements with concurrent flows and should not be read as a calibrated runtime benchmark."]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir",type=Path,default=replay.DEFAULT_SPLITS)
    parser.add_argument("--output-dir",type=Path,default=ROOT/f"outputs/treefree_weighting_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--flows",default="")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--rho",type=float,default=.8)
    parser.add_argument("--eta",type=float,default=.95)
    parser.add_argument("--gamma",type=float,default=2.)
    parser.add_argument("--beta",type=float,default=1.)
    parser.add_argument("--max-moves",type=int,default=512)
    parser.add_argument("--quick-moves",type=int,default=12)
    parser.add_argument("--shortlist",type=int,default=16)
    parser.add_argument("--old-shortlist",type=int,default=16)
    args=parser.parse_args()
    if args.max_moves<args.quick_moves or args.workers<1:
        parser.error("max-moves must cover quick-moves; workers must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a fresh output directory")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    flows=args.flows.split(",") if args.flows else sorted(bench.read(args.splits_dir/"INDEX.json"))
    resource=ROOT/"data/eval/abcd/data"
    reference=build_reference(*(bench.read(resource/name) for name in ("kb.json","ontology.json","guidelines.json")))
    bench.dump(args.output_dir/"reference.json",reference)
    config={"arguments":{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
            "flows":flows,"reference_sha256":digest(reference),"reference_input_to_miners":False,
            "llm_calls":0,"git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "source_hashes":{name:digest((ROOT/name).read_text(encoding="utf-8")) for name in (
                "scripts/benchmark_treefree_tracecover.py","skill_mining/treefree_trace_cover.py",
                "skill_mining/joint_trace_cover.py","skill_mining/backbone_workflow_mining.py",
                "scripts/eval_offline_graph_10flow.py","skill_mining/sop_reference_benchmark.py")}}
    bench.dump(args.output_dir/"config.json",config)
    rows,failures=[],[]
    with patch("llm.chat",side_effect=AssertionError("LLM calls forbidden")) as calls:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs={pool.submit(compare_flow,flow,args,reference):flow for flow in flows}
            for job in as_completed(jobs):
                try:
                    rows.extend(job.result())
                except Exception as exc:
                    failures.append({"flow":jobs[job],"error":repr(exc)})
                    print(f"FAILED {jobs[job]}: {exc!r}",flush=True)
        if calls.call_count:
            raise RuntimeError("A miner attempted an LLM call")
    rows.sort(key=lambda row:(row["method"],row["flow"]))
    summary=summarize(rows)
    bench.dump(args.output_dir/"results.json",{"complete":not failures,"failures":failures,"llm_calls":0,
                                              "rows":rows,"aggregate":summary})
    text=report(summary,rows,args)
    (args.output_dir/"report.md").write_text(text,encoding="utf-8")
    print(text.split("## Per-flow")[0],flush=True)
    if failures:
        raise SystemExit(1)


if __name__=="__main__":
    main()
