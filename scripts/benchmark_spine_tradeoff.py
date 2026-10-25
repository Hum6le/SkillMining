#!/usr/bin/env python3
"""Matched-size ABCD spine curves from frozen native TraceCover trajectories.

Selections are all frozen before held-out dialogues or guideline probes are
loaded. All input/output artifacts are separate from production experiments.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skill_mining.treefree_trace_cover import observed_graph
from skill_mining.spine_tradeoff import make_plans, coverage_matrix, coverage_vector, reachable
from skill_mining.sop_reference_benchmark import build_reference, digest
from scripts.eval_offline_graph_10flow import action_sequence

VARIANTS = {"paper_log_lift": "graph_log", "support_mean_matched": "graph_support_mean_matched"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_key(variant, point, method, seed, split):
    return f"{variant}__{point}__{method}__{seed}__{split}"


def aggregate(per_flow, vectors, variants, points, seeds):
    grouped = defaultdict(list)
    for row in per_flow:
        grouped[row["variant"], row["point"], row["method"], row["seed"]].append(row)
    runs = []
    for (variant, point, method, seed), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r["flow"])
        r = {"variant": variant, "point": point, "alpha": rows[0]["alpha"], "method": method, "seed": seed,
             "edges": sum(x["budget"] for x in rows), "full_edges": sum(x["observed_edges"] for x in rows),
             "repair_added": sum(x["repair_added"] for x in rows), "repair_pruned": sum(x["repair_pruned"] for x in rows)}
        r["edge_ratio"] = r["edges"] / r["full_edges"]
        for split in ("train", "heldout"):
            values = np.concatenate([vectors.pop((variant, point, method, seed, x["flow"], split)) for x in rows])
            vectors[vector_key(variant, point, method, seed, split)] = values
            r[split + "_coverage"] = float(values.mean())
            r[split + "_sessions"] = len(values)
            r[split + "_saturated_F"] = float(np.minimum(values / .8, 1).mean())
        r["guideline_hits"] = sum(x["guideline_hits"] for x in rows)
        r["guideline_total"] = sum(x["guideline_total"] for x in rows)
        r["guideline_micro_recall"] = r["guideline_hits"] / r["guideline_total"]
        r["guideline_weighted_recall"] = sum(x["guideline_hits"] / x["guideline_total"] * x["train_action_turns"] for x in rows) / sum(x["train_action_turns"] for x in rows)
        runs.append(r)
    curves = []
    for variant in variants:
        for point in range(points):
            subset = [r for r in runs if r["variant"] == variant and r["point"] == point]
            assert len({r["edges"] for r in subset}) == 1
            for method in ("tracecover", "frequency", "weight", "random"):
                group = [r for r in subset if r["method"] == method]
                assert len(group) == (len(seeds) if method == "random" else 1)
                row = {k: group[0][k] for k in ("variant", "point", "alpha", "method", "edges", "full_edges", "edge_ratio")}
                row["replicates"] = len(group)
                for metric in ("train_coverage", "heldout_coverage", "guideline_micro_recall", "guideline_weighted_recall"):
                    vals = [r[metric] for r in group]
                    row[metric] = float(np.mean(vals))
                    row[metric + "_seed_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.
                curves.append(row)
    return runs, curves


def endpoint_bootstrap(vectors, variants, seeds, replicates):
    rows = []
    rng = np.random.default_rng(20261025)
    for variant in variants:
        for split in ("train", "heldout"):
            native = vectors[vector_key(variant, 0, "tracecover", None, split)]
            for method in ("frequency", "weight", "random"):
                baseline = (np.mean([vectors[vector_key(variant, 0, method, s, split)] for s in seeds], axis=0)
                            if method == "random" else vectors[vector_key(variant, 0, method, None, split)])
                difference = native - baseline
                values = np.array([difference[rng.integers(len(difference), size=len(difference))].mean() for _ in range(replicates)])
                low, high = np.quantile(values, [.025, .975])
                rows.append({"variant": variant, "split": split, "baseline": method,
                    "native_minus_baseline_pp": 100 * float(difference.mean()),
                    "ci95_low_pp": 100 * float(low), "ci95_high_pp": 100 * float(high),
                    "sessions": len(difference), "bootstrap_replicates": replicates,
                    "scope": "fixed fitted graphs; descriptive historical development analysis; random baseline averaged over seeds"})
    return rows


def run(args):
    if args.output_dir.exists():
        raise FileExistsError("Use a new experiment output directory")
    args.output_dir.mkdir(parents=True)
    alphas = np.linspace(0, 1, args.points).tolist()
    seeds = list(range(42, 42 + args.random_seeds))
    flows = sorted(args.flows.split(",")) if args.flows else sorted(p.stem for p in (args.cache_dir / "graphs/graph_log").glob("*.json"))
    protocol = {"version": "native_tracecover_path_equal_size_reachability_v1", "flows": flows,
        "variants": VARIANTS, "alphas": alphas, "random_priority_seeds": seeds, "bootstrap_replicates": args.bootstrap,
        "budget_rule": "k_flow = native_final_k + ceil(alpha * (full_edges-native_final_k)); x=sum k_flow/sum full_edges",
        "native_choice": "last TRAIN search state at each size; no heldout envelope or budget reoptimization",
        "baselines": "top-k priority, ranked reachable-crossing additions, lowest-priority reachability-safe removals to exactly k",
        "random": "uniform random priority permutation + deterministic repair; not uniform over feasible graphs",
        "coverage": "mean per-session retained occurrence fraction; no saturation; unseen heldout pairs remain in denominator",
        "entry_policy": "all and only observed TRAIN first actions; virtual links fixed and not budgeted",
        "evaluation_role": "historical heldout DEVELOPMENT replay, already used in prior method development",
        "llm_calls": 0, "source_hashes": {name: sha(ROOT / name) for name in
            ("skill_mining/spine_tradeoff.py", "scripts/benchmark_spine_tradeoff.py", "skill_mining/treefree_trace_cover.py")}}
    dump(args.output_dir / "protocol.json", protocol)
    snapshot = args.output_dir / "source_snapshot"
    snapshot.mkdir()
    for name in protocol["source_hashes"]:
        (snapshot / Path(name).name).write_bytes((ROOT / name).read_bytes())
    data_by_flow, plans, catalogs = {}, [], {}
    train_ids = set()
    # Phase 1: no held-out/guideline files are read here or by the selectors.
    for flow in flows:
        train = read(args.splits_dir / flow / "train.json")
        data = observed_graph(flow, train)
        sequences = [action_sequence(flow, d) for d in train]
        ids = [str(d["convo_id"]) for d in train]
        if len(ids) != len(set(ids)) or train_ids.intersection(ids):
            raise ValueError("TRAIN dialogue IDs overlap across flows")
        train_ids.update(ids)
        pairs = [(e["source"], e["target"]) for e in data["edges"]]
        matrix, lengths, audit = coverage_matrix(sequences, pairs)
        data_by_flow[flow] = {"data": data, "train_matrix": matrix, "train_lengths": lengths, "train_audit": audit}
        catalogs[flow] = {"edges": data["edges"], "nodes": data["nodes"], "entries": data["entries"],
            "train_sha256": digest(train), "train_action_turns": sum(map(len, sequences)), "train_coverage_audit": audit}
        for variant, cache_method in VARIANTS.items():
            path = args.cache_dir / "graphs" / cache_method / (flow + ".json")
            graph = read(path)
            original = read(args.cache_dir / "per_flow" / cache_method / (flow + ".json"))
            if original["train_sha256"] != digest(train):
                raise ValueError("Native cached graph was fitted on different TRAIN data")
            original_pool = {(e["source"], e["target"]): e for e in graph["edges"]}
            if set(original_pool) != set(pairs):
                raise ValueError("Different observed candidate pools")
            for edge in data["edges"]:
                old = original_pool[edge["source"], edge["target"]]
                if any(edge[k] != old[k] for k in ("support", "num_sessions", "score")):
                    raise ValueError("Native training edge evidence mismatch")
            selected, weights = make_plans(flow, data, graph, alphas, seeds, variant)
            plans.extend(selected)
            catalogs[flow][variant] = {"native_graph_path": str(path), "native_graph_sha256": sha(path),
                "parameters": {k: graph["treefree_trace_cover"][k] for k in ("evidence", "rho", "eta", "gamma", "beta", "support_scale")},
                "weights": weights}
        print(f"FROZEN {flow}: {len(pairs)} observed edge types", flush=True)
    dump(args.output_dir / "catalogs.json", catalogs)
    selected_path = args.output_dir / "selections.jsonl"
    selected_path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in plans), encoding="utf-8")
    frozen_sha = sha(selected_path)
    dump(args.output_dir / "selection_freeze.json", {"selections_sha256": frozen_sha, "plans": len(plans),
        "selection_completed_before_heldout_read": True, "catalogs_sha256": sha(args.output_dir / "catalogs.json")})
    # Phase 2: graphs are immutable; now load held-out sequences and reference.
    resource = ROOT / "data/eval/abcd/data"
    reference = build_reference(*(read(resource / name) for name in ("kb.json", "ontology.json", "guidelines.json")))
    dump(args.output_dir / "reference.json", reference)
    heldout_ids, vectors, scored, ids_by_split = set(), {}, [], {"train": [], "heldout": []}
    for flow in flows:
        training = read(args.splits_dir / flow / "train.json")
        heldout = read(args.splits_dir / flow / "test.json")
        ids = [str(d["convo_id"]) for d in heldout]
        if len(set(ids)) != len(ids) or heldout_ids.intersection(ids) or train_ids.intersection(ids):
            raise ValueError("Dialogue ID overlap in TRAIN/heldout projections")
        heldout_ids.update(ids)
        data = data_by_flow[flow]
        pairs = [(e["source"], e["target"]) for e in data["data"]["edges"]]
        seqs = [action_sequence(flow, d) for d in heldout]
        matrix, lengths, audit = coverage_matrix(seqs, pairs)
        data.update(heldout_matrix=matrix, heldout_lengths=lengths, heldout_audit=audit)
        catalogs[flow].update(heldout_sha256=digest(heldout), heldout_coverage_audit=audit)
        for split, dialogues in (("train", training), ("heldout", heldout)):
            ids_by_split[split].extend(str(d["convo_id"]) for d in dialogues if len(action_sequence(flow, d)) > 1)
        positives = {(flow + ":" + e["source"], flow + ":" + e["target"]) for e in reference["flows"][flow]["positive"]}
        for plan in (p for p in plans if p["flow"] == flow):
            selected = set(plan["indices"])
            if len(selected) != plan["budget"] or not reachable(
                    {n["id"] for n in data["data"]["nodes"]}, set(data["data"]["entries"]), [pairs[i] for i in selected])[0]:
                raise ValueError("Frozen plan violates exact size or entry reachability")
            row = {k: plan[k] for k in ("flow", "variant", "point", "alpha", "method", "seed", "budget", "observed_edges", "repair_added", "repair_pruned")}
            for split in ("train", "heldout"):
                values = coverage_vector(data[split + "_matrix"], selected)
                row[split + "_coverage"] = float(values.mean())
                row[split + "_sessions"] = len(values)
                row[split + "_transition_recall"] = float(np.average(values, weights=data[split + "_lengths"]))
                vectors[plan["variant"], plan["point"], plan["method"], plan["seed"], flow, split] = values
            row["train_action_turns"] = catalogs[flow]["train_action_turns"]
            row["guideline_hits"] = len(positives.intersection(pairs[i] for i in selected))
            row["guideline_total"] = len(positives)
            scored.append(row)
        print(f"SCORED {flow}: heldout sessions={audit['scored_sessions']}, unseen occurrences={audit['unseen_transition_occurrences']}", flush=True)
    runs, curves = aggregate(scored, vectors, VARIANTS, len(alphas), seeds)
    bootstrap = endpoint_bootstrap(vectors, VARIANTS, seeds, args.bootstrap)
    assert sha(selected_path) == frozen_sha
    np.savez_compressed(args.output_dir / "session_coverage.npz", **vectors)
    dump(args.output_dir / "session_ids.json", ids_by_split)
    dump(args.output_dir / "evaluation_audit.json", {"selection_sha256_unchanged": True,
        "all_plans_exact_size_and_reachable": True, "all_flow_budgets_matched": True,
        "selection_plans": len(plans), "zero_llm_calls": True,
        "scored_sessions": {s: len(ids_by_split[s]) for s in ids_by_split},
        "data": {f: {s: data_by_flow[f][s + "_audit"] for s in ("train", "heldout")} for f in flows}})
    dump(args.output_dir / "evaluated_catalogs.json", catalogs)
    dump(args.output_dir / "results.json", {"protocol": protocol, "runs": runs, "curves": curves, "endpoint_bootstrap": bootstrap})
    for filename, rows in (("curves.csv", curves), ("runs.csv", runs), ("per_flow.csv", scored), ("endpoint_bootstrap.csv", bootstrap)):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("DONE: " + str(args.output_dir / "results.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/eval/abcd/splits")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/treefree_weighting_20261021_fullcheck_v2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flows", default="")
    parser.add_argument("--points", type=int, default=11)
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.points < 2 or args.random_seeds < 2 or args.bootstrap < 1:
        parser.error("Use >=2 points/random seeds and >=1 bootstrap replicate")
    with patch("llm.chat", side_effect=AssertionError("No LLM calls in spine tradeoff experiment")):
        run(args)
