#!/usr/bin/env python3
"""Collect all graph-only joint TraceCover development trials, including failures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = [
    "train_graph_nodes", "retained_edges", "retained_transition_recall",
    "complete_route_rate", "branch_next_action_top1", "branch_next_action_mrr",
    "avg_candidate_size", "routing_pair_complexity", "observable_pair_distinguishability",
]


def collect(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/offline_graph_benchmark.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        config = report["config"]
        searches = [r["joint_search"] for r in report["per_flow"] if "joint_search" in r]
        roots = unsupported = 0
        for graph_path in (path.parent / "graphs").glob("*.json"):
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            root_rows = [e for e in graph["backbone"]["edges"] if e["source"] == "<START>"]
            roots += len(root_rows)
            unsupported += sum(not e["support"] for e in root_rows)
        rows.append({
            "run": path.parent.name, "method": report["method"],
            "rho": config["trace_coverage_target"], "eta": config["corpus_fitness_target"],
            "gamma": config["routing_complexity_weight"],
            "beta": config.get("tree_evidence_weight"),
            "search_mode": config.get("joint_search_mode") if searches else "baseline",
            "ranking": config.get("joint_ranking") if searches else "backbone",
            "root_policy": config.get("joint_root_policy", "all_legacy") if searches else "original",
            **{metric: report["weighted_overall"][metric] for metric in METRICS},
            "root_edges": roots, "unsupported_root_edges": unsupported,
            "accepted_moves": sum(len(s["accepted_moves"]) for s in searches),
            "changed_backbone_edges": sum(s["backbone_changed_edges"] for s in searches),
            "removed_initial_backbone_edges": sum(s["removed_initial_backbone_edges"] for s in searches),
            "train_target_feasible": all(s["target_feasible"] for s in searches),
            "objective_monotone": all(m["objective_after"] < m["objective_before"]
                                      for s in searches for m in s["accepted_moves"]),
            "objective_reduction_sum": sum(s["initial_objective"] - s["final_objective"] for s in searches),
            "report": str(path),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--version", default="20261010")
    args = parser.parse_args()
    rows = collect(args.run_root)
    if not rows:
        raise SystemExit("No completed reports")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    baseline = next(r for r in rows if r["run"] == "baseline")
    for row in rows:
        row["edge_delta_pct"] = 100 * (row["retained_edges"] / baseline["retained_edges"] - 1)
        row["recall_delta_pp"] = 100 * (row["retained_transition_recall"] - baseline["retained_transition_recall"])
        row["route_delta_pp"] = 100 * (row["complete_route_rate"] - baseline["complete_route_rate"])
        row["branch_mrr_delta"] = row["branch_next_action_mrr"] - baseline["branch_next_action_mrr"]
    stem = f"RESULTS_{args.version}"
    payload = {
        "date": "2026-10-10", "protocol": "10-flow graph-only development; no untouched-test claims",
        "baseline": "baseline", "runs": rows,
    }
    (args.report_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.report_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Joint TraceCover: complete development trial results", "", "Date: 2026-10-10.", "",
        "All graphs are mined separately on the same 10 coarse-flow train splits, then replayed on the previously used held-out DEVELOPMENT split. No LLM calls, skill compilation, or rollout. Sizes are sums across flows; recall/candidates/pairs are transition-weighted, branch metrics branch-transition-weighted, routes route-dialogue-weighted. Pair distinguishability is the existing training-derived observable diagnostic, not a learned routing policy.", "",
        "| Run | rho / eta | Root policy | Edges | Recall | Complete routes | Branch Top1 | Branch MRR | Candidates | Pairs | Distinction | Backbone changes |", 
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['rho']} / {row['eta']} | {row['root_policy']} | "
            f"{row['retained_edges']} | {100*row['retained_transition_recall']:.2f}% | "
            f"{100*row['complete_route_rate']:.2f}% | {100*row['branch_next_action_top1']:.2f}% | "
            f"{row['branch_next_action_mrr']:.4f} | {row['avg_candidate_size']:.3f} | "
            f"{row['routing_pair_complexity']:.3f} | {row['observable_pair_distinguishability']:.4f} | "
            f"{row['changed_backbone_edges']} |"
        )
    lines += ["", "## Audit and interpretation boundaries", "",
              "- Early r1/r2 runs allowed every action to be a virtual root child. They are retained as diagnostic trials, not valid evidence of coherent process compression. Observed-entry constraints were introduced in r3.",
              "- Graph-only train-frequency candidate ordering must be ablated separately: it changes MRR without changing graph topology. The no-search support-ranking baseline isolates this effect.",
              "- rho/eta sensitivity changes the coverage requirement and is not a matched-objective ablation. Improved coverage with more edges is a tradeoff, not Pareto domination.",
              "- Every accepted move is checked for decreasing objective and training fitness feasibility. The shortlisted, budgeted neighborhood is not a certified global or local optimum.",
              "- This benchmark has already influenced method development. Select a paper configuration on inner-train data and evaluate frozen settings on a genuinely held-out set before making generalization claims.", "",
              "## Acceptance audit", "",
              "| Run | Accepted moves | Removed initial tree edges | Unsupported root edges | Fitness feasible | Objective monotone |", 
              "|---|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append(f"| {row['run']} | {row['accepted_moves']} | {row['removed_initial_backbone_edges']} | "
                     f"{row['unsupported_root_edges']} | {row['train_target_feasible']} | {row['objective_monotone']} |")
    text = "\n".join(lines) + "\n"
    (args.report_dir / f"{stem}.md").write_text(text, encoding="utf-8")
    # Timestamped first; fixed-name copy contains exactly the same report.
    (args.report_dir / "EXPERIMENT_RESULTS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
