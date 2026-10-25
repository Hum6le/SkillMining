"""ABCD graph-only ranking replay with a common TRAIN ranking and branch set.

Reuses audited, frozen native graphs; never calls LLMs or compiles skills.
Process-miner graphs are action adjacency projections, not Petri execution.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_offline_graph_10flow import action_sequence
from skill_mining.treefree_trace_cover import observed_graph
from skill_mining.spine_tradeoff import ranked_select, rankings
from skill_mining.sop_reference_benchmark import digest

SOP = ROOT / "outputs/sop_graph_benchmark_20261018_v2"
TREE = ROOT / "outputs/treefree_weighting_20261021_fullcheck_v2"
PM = ROOT / "outputs/abcd_process_mining_20261022_full_v1"
CACHED = {
    "support_lift": (SOP, "support_lift"),
    "discriminative": (SOP, "discriminative"),
    "heuristics_dependency": (SOP, "heuristics"),
    "tracecover_original": (SOP, "trace_cover"),
    "joint_original": (TREE, "old_tree"),
    "tracecover_log": (TREE, "graph_log"),
    "tracecover_support_matched": (TREE, "graph_support_mean_matched"),
    "pm_heuristics_net": (PM, "heuristics_net"),
    "pm_inductive": (PM, "inductive"),
    "pm_inductive_noise02": (PM, "inductive_f20"),
}
MATCHED = {"frequency_k385": "tracecover_original", "frequency_k438": "tracecover_support_matched",
           "frequency_k752": "tracecover_log"}
ORDER = ["support_lift", "discriminative", "heuristics_dependency", "tracecover_original",
         "joint_original", "tracecover_support_matched", "tracecover_log", "pm_heuristics_net",
         "pm_inductive", "pm_inductive_noise02", "full_dfg", *MATCHED]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_pairs(graph):
    return {(e["source"], e["target"]) for outgoing in graph["local_transitions"].values() for e in outgoing}


def rank_pairs(pairs, evidence):
    outgoing = defaultdict(list)
    for source, target in pairs:
        outgoing[source].append(target)
    for source, targets in outgoing.items():
        targets.sort(key=lambda target: (-evidence.get((source, target), {}).get("num_sessions", 0),
            -evidence.get((source, target), {}).get("support", 0),
            -evidence.get((source, target), {}).get("score", 0.), target))
    return dict(sorted(outgoing.items()))


def evaluate_sequences(sequences, ranked, branching):
    """Columns: reciprocal rank, top1, hit, candidate count, shared branch mask."""
    rows = []
    for seq in sequences:
        for source, target in zip(seq, seq[1:]):
            candidates = ranked.get(source, [])
            rank = candidates.index(target) + 1 if target in candidates else 0
            rows.append([1 / rank if rank else 0., int(rank == 1), int(rank > 0), len(candidates), int(source in branching)])
    return np.asarray(rows, dtype=float).reshape(-1, 5)


def summarize(values):
    branch = values[values[:, 4] == 1]
    def stats(rows, prefix):
        means = rows[:, :4].mean(axis=0) if len(rows) else [0.] * 4
        return {prefix + "transitions": len(rows), prefix + "mrr": float(means[0]),
            prefix + "top1": float(means[1]), prefix + "recall": float(means[2]), prefix + "candidates": float(means[3])}
    return {**stats(values, ""), **stats(branch, "branch_")}


def direct_audit(sequences, ranked, branching, values):
    """Recount ranks using a pair lookup, independently of list.index scoring."""
    lookup = {(source, target): i for source, targets in ranked.items() for i, target in enumerate(targets, 1)}
    events = [(source, target) for seq in sequences for source, target in zip(seq, seq[1:])]
    ranks = [lookup.get(pair, 0) for pair in events]
    assert len(events) == len(values)
    expected = np.array([[1 / r if r else 0., int(r == 1), int(r > 0), len(ranked.get(source, [])), int(source in branching)]
                         for (source, _), r in zip(events, ranks)], dtype=float).reshape(-1, 5)
    np.testing.assert_array_equal(values, expected)


def paired_intervals(values, event_ids, repetitions):
    ids = sorted(set(event_ids))
    lookup = {sid: i for i, sid in enumerate(ids)}
    assignment = np.array([lookup[sid] for sid in event_ids])
    rng = np.random.default_rng(20261025)
    rows = []
    for base, native in MATCHED.items():
        for metric, col, branch_only in (("mrr", 0, False), ("branch_mrr", 0, True)):
            mask = values[native][:, 4] == 1 if branch_only else np.ones(len(event_ids), dtype=bool)
            difference = values[native][mask, col] - values[base][mask, col]
            sums = np.bincount(assignment[mask], weights=difference, minlength=len(ids))
            counts = np.bincount(assignment[mask], minlength=len(ids))
            draws = []
            for _ in range(repetitions):
                sample = rng.integers(len(ids), size=len(ids))
                denominator = counts[sample].sum()
                if denominator:
                    draws.append(sums[sample].sum() / denominator)
            low, high = np.quantile(draws, [.025, .975])
            rows.append({"native": native, "baseline": base, "metric": metric,
                "difference": float(difference.mean()), "ci95_low": float(low), "ci95_high": float(high)})
    return rows


def run(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "source_snapshot").mkdir()
    sources = ["scripts/benchmark_graph_ranking.py", "scripts/eval_offline_graph_10flow.py",
               "skill_mining/treefree_trace_cover.py", "skill_mining/spine_tradeoff.py"]
    protocol = {"dataset": "ABCD ten coarse flows", "llm_calls": 0,
        "ranking": "all methods: distinct TRAIN dialogue support desc, TRAIN occurrence support desc, TRAIN log/lift score desc, target lexicographic",
        "unobserved_pm_pairs": "retain and count all projected relations; zero TRAIN support/score; lexical ties; after observed candidates",
        "branch_set": "sources with >=2 distinct successors in the common full TRAIN DFG; shared across every method",
        "evaluation": "teacher-forced previous gold action; missing successor=0; all heldout transitions included",
        "split_role": "historical development replay, not untouched test",
        "candidate_counts": "averaged over heldout transition occurrences, including empty candidate sets",
        "graph_sizes": "native edge budgets, plus per-flow exact-size frequency + observed-entry reachability repair controls",
        "production_caveat": "original miners have different eligibility rules; matched frequency controls use all observed edges",
        "process_miners": "cached official PM4Py 2.7.23 native Heuristics-net dependencies and Inductive process-tree footprints; projection rankings added by this bench, not native PM predictions",
        "reference_input": "no guideline probes read or used for selection or ranking",
        "cache_note": "historical per-flow cache has eval fields; only TRAIN provenance and completion/projection status consumed",
        "bootstrap": f"{args.bootstrap} paired dialogue cluster draws, fixed graphs, transition-micro estimand",
        "sources_sha256": {s: sha(ROOT / s) for s in sources}}
    dump(out / "protocol.json", protocol)
    for source in sources:
        (out / "source_snapshot" / Path(source).name).write_bytes((ROOT / source).read_bytes())
    flows = sorted(p.stem for p in (TREE / "graphs/graph_log").glob("*.json"))
    plans, provenance, state = {}, {}, {}
    for flow in flows:
        train = read(args.splits_dir / flow / "train.json")
        data = observed_graph(flow, train)
        evidence = {(e["source"], e["target"]): e for e in data["edges"]}
        full = set(evidence)
        branching = {u for u in {p[0] for p in full} if len({v for s, v in full if s == u}) >= 2}
        graphs, provenance[flow] = {}, {}
        for method, (root, directory) in CACHED.items():
            path = root / "graphs" / directory / (flow + ".json")
            prior = read(root / "per_flow" / directory / (flow + ".json"))
            assert prior["train_sha256"] == digest(train), (flow, method, "TRAIN hash mismatch")
            graph = read(path)
            if root == PM:
                assert prior["status"] == "completed" and prior["projection"]["complete"]
                assert digest(graph) == prior["graph_sha256"]
            graphs[method] = selected_pairs(graph)
            assert {n for p in graphs[method] for n in p} <= {n["id"] for n in data["nodes"]}
            provenance[flow][method] = {"graph": str(path), "file_sha256": sha(path), "train_sha256": prior["train_sha256"]}
        graphs["full_dfg"] = full
        frequency = rankings(data["edges"], "log", 1., data["train_dialogues"], None, flow)[0]["frequency"]
        pairs = [(e["source"], e["target"]) for e in data["edges"]]
        for method, native in MATCHED.items():
            selected, repair = ranked_select(pairs, frequency, len(graphs[native]), {n["id"] for n in data["nodes"]}, set(data["entries"]))
            graphs[method] = {pairs[i] for i in selected}
            provenance[flow][method] = {"budget_matched_to": native, "budget": len(selected), "repair": repair}
        plans[flow] = {"ranked": {m: rank_pairs(graphs[m], evidence) for m in ORDER},
                       "branch_sources": sorted(branching), "full_train_pairs": sorted(full),
                       "train_sha256": digest(train)}
        state[flow] = {"graphs": graphs, "branching": branching}
    dump(out / "provenance.json", provenance)
    dump(out / "frozen_rankings.json", plans)
    freeze = sha(out / "frozen_rankings.json")
    dump(out / "selection_freeze.json", {"sha256": freeze, "completed_before_heldout_dialogue_read": True})
    values, per_flow, event_ids, cases = defaultdict(list), [], [], []
    heldout_hashes = {}
    for flow in flows:
        heldout = read(args.splits_dir / flow / "test.json")
        seqs = [action_sequence(flow, d) for d in heldout]
        heldout_hashes[flow] = digest(heldout)
        for dialogue, seq in zip(heldout, seqs):
            for index, (u, v) in enumerate(zip(seq, seq[1:])):
                event_ids.append(str(dialogue["convo_id"]))
                cases.append({"flow": flow, "dialogue_id": str(dialogue["convo_id"]), "transition": index, "source": u, "target": v})
        for method in ORDER:
            ranked = plans[flow]["ranked"][method]
            result = evaluate_sequences(seqs, ranked, state[flow]["branching"])
            direct_audit(seqs, ranked, state[flow]["branching"], result)
            values[method].append(result)
            per_flow.append({"flow": flow, "method": method, "edges": len(state[flow]["graphs"][method]), **summarize(result)})
        print("SCORED " + flow, flush=True)
    values = {m: np.concatenate(v) for m, v in values.items()}
    assert len({len(v) for v in values.values()}) == 1
    branch_mask = values["full_dfg"][:, 4]
    assert all(np.array_equal(v[:, 4], branch_mask) for v in values.values())
    summary = [{"method": method, "edges": sum(len(state[f]["graphs"][method]) for f in flows), **summarize(values[method])} for method in ORDER]
    intervals = paired_intervals(values, event_ids, args.bootstrap)
    assert sha(out / "frozen_rankings.json") == freeze
    dump(out / "results.json", {"protocol": protocol, "summary": summary, "per_flow": per_flow, "paired_intervals": intervals})
    dump(out / "cases.json", cases)
    np.savez_compressed(out / "per_transition_metrics.npz", **values)
    dump(out / "audit.json", {"all_rank_metrics_independently_recounted": True,
        "same_branch_masks": True, "rankings_sha256_unchanged": True, "heldout_hashes": heldout_hashes,
        "transitions": len(cases), "branch_transitions": int(branch_mask.sum()),
        "dialogues_with_transitions": len(set(event_ids)), "flow_method_evaluations": len(per_flow), "llm_calls": 0})
    for filename, rows in (("summary.csv", summary), ("per_flow.csv", per_flow), ("paired_intervals.csv", intervals)):
        with (out / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = ["# ABCD graph ranking: common TRAIN ranking", "", "CPU only; LLM calls = 0. Historical development replay.", "",
        "| Method | Edges | Recall % | Top1 % | MRR | Branch MRR | Candidates | Branch candidates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        report.append(f"| {row['method']} | {row['edges']} | {100*row['recall']:.2f} | {100*row['top1']:.2f} | {row['mrr']:.4f} | {row['branch_mrr']:.4f} | {row['candidates']:.3f} | {row['branch_candidates']:.3f} |")
    report += ["", "Paired differences: native minus exact-size frequency baseline. Fixed graphs; dialogue bootstrap.", "",
        "| Native | Baseline | Metric | Difference | 95% interval |", "|---|---|---|---:|---:|"]
    for row in intervals:
        report.append(f"| {row['native']} | {row['baseline']} | {row['metric']} | {row['difference']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |")
    report += ["", "## Protocol", "", *["- " + k + ": " + str(v) for k, v in protocol.items() if k != "sources_sha256"], "",
        "MRR ranks NEXT canonical action given the true preceding action. This does not measure slot filling or LLM skill use.",
        "Process miner edge counts are projected action relation counts, not native Petri arc counts or model storage. Generalized unobserved relations are retained; these are not token-replay conformance results.",
        "Native graphs have different sizes and eligibility rules. Only frequency controls are size-matched; do not present the entire table as equal-budget comparison.", ""]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/eval/abcd/splits")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    with patch("llm.chat", side_effect=AssertionError("No LLM in graph-only replay")):
        run(args)
