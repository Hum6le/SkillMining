"""CPU-only equal-size spine comparisons on the existing SGD seen-schema split.

TRAIN-only selection is frozen before validation dialogues are read. Existing
production TraceCover is a separate endpoint, not relabelled as tree-free.
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
from eval_tod.sgd_adapter import gold_policy, iter_family_dialogues
from eval_tod.sgd_skill_agent import policy_heads, policy_token
from skill_mining.backbone_workflow_mining import mine_backbone_workflow_trace_cover
from skill_mining.treefree_trace_cover import observed_graph, mine_treefree_trace_cover
from skill_mining.spine_tradeoff import make_plans, rankings, ranked_select, coverage_matrix, coverage_vector

VARIANTS = {"paper_log_lift": "log", "support_mean_matched": "support_mean_matched",
            "production_tracecover": "production"}
METHODS = ("tracecover", "frequency", "weight", "random")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(family, dialogues):
    projected, sequences, ids, heads = [], [], [], {}
    for dialogue in dialogues:
        tokens = []
        indices = dialogue["target_indices"]["policy_turn_indices"]
        if indices != sorted(set(indices)):
            raise ValueError("Policy target indices must be unique and chronological")
        for index in indices:
            policy = gold_policy(dialogue, index)
            token = policy_token(policy)
            heads[token] = policy_heads(policy)
            tokens.append(token)
        sid = str(dialogue["dialogue_id"])
        ids.append(sid)
        sequences.append([family + ":" + token for token in tokens])
        projected.append({"convo_id": sid, "delexed": [
            {"speaker": "action", "text": "", "targets": [None, "take_action", t, []]} for t in tokens]})
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dialogue within a family")
    return projected, sequences, ids, heads


def endpoint_plans(family, data, graph, seeds):
    pairs = [(e["source"], e["target"]) for e in data["edges"]]
    lookup = {pair: i for i, pair in enumerate(pairs)}
    chosen = {lookup[e["source"], e["target"]] for es in graph["local_transitions"].values() for e in es}
    common = {"flow": family, "variant": "production_tracecover", "point": 0, "alpha": 0.,
              "budget": len(chosen), "observed_edges": len(pairs)}
    plans = [{**common, "method": "tracecover", "seed": None, "indices": sorted(chosen),
              "repair_added": 0, "repair_pruned": 0}]
    nodes, entries = {n["id"] for n in data["nodes"]}, set(data["entries"])
    for method in METHODS[1:]:
        for seed in seeds if method == "random" else [None]:
            orders, _ = rankings(data["edges"], "log", 1., data["train_dialogues"], seed, family)
            selected, repair = ranked_select(pairs, orders[method], len(chosen), nodes, entries)
            plans.append({**common, "method": method, "seed": seed, "indices": sorted(selected), **repair})
    return plans


def key(variant, point, method, seed, split):
    return f"{variant}__{point}__{method}__{seed}__{split}"


def cluster_bootstrap(difference, dialogue_ids, replicates, rng):
    """Resample original dialogues, retaining all of their family projections.

    Each draw estimates the same projection-weighted mean as the point estimate.
    This intentionally does not give each unique original dialogue equal weight.
    """
    ids = sorted(set(dialogue_ids))
    lookup = {sid: i for i, sid in enumerate(ids)}
    assignment = np.array([lookup[sid] for sid in dialogue_ids])
    sums = np.bincount(assignment, weights=difference, minlength=len(ids))
    counts = np.bincount(assignment, minlength=len(ids))
    values = []
    for _ in range(replicates):
        draw = rng.integers(len(ids), size=len(ids))
        values.append(float(sums[draw].sum() / counts[draw].sum()))
    return np.quantile(values, [.025, .975]).tolist()


def independent_check(sequences, pairs, selected, nodes, entries, values):
    """Direct recount and independent closure, separate from matrix/BFS helper."""
    edges = {pairs[i] for i in selected}
    seen = set(entries)
    while True:
        extended = seen | {b for a, b in edges if a in seen}
        if extended == seen:
            break
        seen = extended
    assert set(nodes) <= seen
    direct = []
    for seq in sequences:
        if len(seq) > 1:
            counts = Counter(zip(seq, seq[1:]))
            direct.append(sum(n for edge, n in counts.items() if edge in edges) / sum(counts.values()))
    np.testing.assert_allclose(values, direct, atol=1e-12, rtol=0)


def run(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "graphs").mkdir()
    (out / "source_snapshot").mkdir()
    index = json.loads((args.splits_dir / "INDEX.json").read_text(encoding="utf-8"))
    families = sorted(f for f in index["train_families"]
                      if index["families"][f]["validation"].get("policy_turns", 0))
    if args.families:
        requested = args.families.split(",")
        if not set(requested) <= set(families):
            raise ValueError("Family must have seen-schema validation targets")
        families = sorted(requested)
    seeds = list(range(42, 42 + args.random_seeds))
    alphas = np.linspace(0, 1, args.points).tolist()
    source_files = ["scripts/benchmark_sgd_spine_tradeoff.py", "skill_mining/spine_tradeoff.py",
        "skill_mining/treefree_trace_cover.py", "skill_mining/backbone_workflow_mining.py",
        "eval_tod/sgd_adapter.py", "eval_tod/sgd_skill_agent.py"]
    protocol = {"dataset": "SGD 20% seen-schema", "families": families, "variants": VARIANTS,
        "split_root": str(args.splits_dir), "selection_split": "official train", "heldout_split": "official validation, training-seen exact schemas",
        "random_priority_seeds": seeds, "alphas": alphas, "llm_calls": 0,
        "node_identity": "unordered bundle of (action kind, service family); no slots/API method; all policy targets including non-call",
        "session_unit": "family dialogue projection, not deduplicated original dialogue",
        "coverage": "unsaturated per-projection retained transition occurrence fraction; unseen validation edges remain in denominator",
        "budget": "per-family native final k + ceil(alpha*(full-k)); production variant endpoint only",
        "entry_policy": "observed TRAIN first nodes, all TRAIN nodes must be reachable; fixed virtual entry edges not budgeted",
        "baseline_selection": "top-k priority + ranked crossing-edge repair + reverse-priority safe pruning to exact k",
        "random_sampling": "uniform priority permutation then repair; not uniform feasible subgraphs",
        "bootstrap": "2000 paired original-dialogue cluster draws; fixed fitted graphs and fixed random-seed ensemble",
        "guideline_reference": None,
        "production_note": "recomputed original SGD default; not company saved graph; weight baseline uses shared log/lift edge score",
        "production_candidate_caveat": "native min_branch_support=2 restricts residual-edge eligibility, while comparison baselines rank all observed edges; auxiliary whole-method comparison, not an isolated same-eligibility selection-objective comparison",
        "treefree_note": "new TRAIN-only graph experiments, not production SGD trained agent artifacts; native search checkpoints, no fixed-budget reoptimization",
        "evaluation_role": "existing validation used in previous development, not a new blind test",
        "source_sha256": {name: sha(ROOT / name) for name in source_files}}
    protocol["bootstrap"] = protocol["bootstrap"].replace("2000", str(args.bootstrap))
    dump(out / "protocol.json", protocol)
    for name in source_files:
        (out / "source_snapshot" / Path(name).name).write_bytes((ROOT / name).read_bytes())
    plans, storage, catalogs, training_ids = [], {}, {}, set()
    for family in families:
        train = list(iter_family_dialogues(args.splits_dir, family, "train"))
        projected, seqs, ids, heads = project(family, train)
        training_ids.update(ids)
        data = observed_graph(family, projected)
        pairs = [(e["source"], e["target"]) for e in data["edges"]]
        matrix, lengths, audit = coverage_matrix(seqs, pairs)
        storage[family] = {"data": data, "train_sequences": seqs, "train_ids": ids,
                          "train_matrix": matrix, "train_lengths": lengths}
        catalogs[family] = {"edges": data["edges"], "nodes": data["nodes"], "entries": data["entries"],
            "policy_heads": heads, "train_audit": audit, "train_policy_turns": sum(map(len, seqs)),
            "train_sha256": sha(args.splits_dir / family / "train.jsonl.gz")}
        for variant, evidence in VARIANTS.items():
            if evidence == "production":
                graph = mine_backbone_workflow_trace_cover(family, projected, min_branch_support=2,
                    trace_coverage_target=.8, corpus_fitness_target=.95, routing_complexity_weight=2.)["subgraph"]
                chosen = endpoint_plans(family, data, graph, seeds)
            else:
                graph = mine_treefree_trace_cover(family, projected, evidence=evidence,
                    rho=.8, eta=.95, gamma=2., beta=1., max_moves=512, shortlist_size=100000)["subgraph"]
                chosen, _ = make_plans(family, data, graph, alphas, seeds, variant)
            dump(out / "graphs" / (family + "__" + variant + ".json"), graph)
            plans.extend(chosen)
            print(f"FROZEN {family}/{variant}: {chosen[0]['budget']}/{len(pairs)} edges", flush=True)
    dump(out / "catalogs.json", catalogs)
    selection_path = out / "selections.jsonl"
    selection_path.write_text("".join(json.dumps(p) + "\n" for p in plans), encoding="utf-8")
    freeze = sha(selection_path)
    dump(out / "selection_freeze.json", {"sha256": freeze, "catalogs_sha256": sha(out / "catalogs.json"),
        "completed_before_validation_dialogue_read": True, "plans": len(plans)})
    # Only now load the held-out dialogue files.
    vectors, scored, session_ids, audit = {}, [], {"train": [], "heldout": []}, {}
    for family in families:
        state, catalog = storage[family], catalogs[family]
        validation = list(iter_family_dialogues(args.splits_dir, family, "validation"))
        _, seqs, ids, _ = project(family, validation)
        assert not training_ids.intersection(ids), "TRAIN/validation original dialogue overlap"
        pairs = [(e["source"], e["target"]) for e in state["data"]["edges"]]
        matrix, lengths, held_audit = coverage_matrix(seqs, pairs)
        state.update(heldout_sequences=seqs, heldout_ids=ids, heldout_matrix=matrix, heldout_lengths=lengths)
        audit[family] = {"train": catalog["train_audit"], "heldout": held_audit,
                         "heldout_policy_turns": sum(map(len, seqs)),
                         "heldout_sha256": sha(args.splits_dir / family / "validation.jsonl.gz")}
        for split in session_ids:
            session_ids[split].extend({"family": family, "dialogue_id": sid} for sid, seq in
                zip(state[split + "_ids"], state[split + "_sequences"]) if len(seq) > 1)
        for plan in (p for p in plans if p["flow"] == family):
            selected = set(plan["indices"])
            assert len(selected) == plan["budget"]
            row = {k: plan[k] for k in ("flow", "variant", "point", "alpha", "method", "seed", "budget", "observed_edges")}
            for split in session_ids:
                values = coverage_vector(state[split + "_matrix"], selected)
                independent_check(state[split + "_sequences"], pairs, selected,
                                  [n["id"] for n in catalog["nodes"]], catalog["entries"], values)
                row[split + "_coverage"] = float(values.mean())
                vectors.setdefault(key(plan["variant"], plan["point"], plan["method"], plan["seed"], split), []).append(values)
            scored.append(row)
        print(f"AUDITED {family}: {held_audit['scored_sessions']} validation projections", flush=True)
    vectors = {k: np.concatenate(v) for k, v in vectors.items()}
    grouped = defaultdict(list)
    for p in plans:
        grouped[p["variant"], p["point"], p["method"], p["seed"]].append(p)
    runs = []
    for (variant, point, method, seed), group in grouped.items():
        row = {"variant": variant, "point": point, "method": method, "seed": seed,
            "edges": sum(p["budget"] for p in group), "full_edges": sum(p["observed_edges"] for p in group),
            "repair_added": sum(p["repair_added"] for p in group), "repair_pruned": sum(p["repair_pruned"] for p in group)}
        row["edge_ratio"] = row["edges"] / row["full_edges"]
        for split in session_ids:
            values = vectors[key(variant, point, method, seed, split)]
            row[split + "_coverage"] = float(values.mean())
            row[split + "_saturated_F"] = float(np.minimum(values / .8, 1).mean())
        runs.append(row)
    curves = []
    for variant in VARIANTS:
        for point in range(1 if variant == "production_tracecover" else args.points):
            subset = [r for r in runs if r["variant"] == variant and r["point"] == point]
            assert len({r["edges"] for r in subset}) == 1
            for method in METHODS:
                group = [r for r in subset if r["method"] == method]
                row = {k: group[0][k] for k in ("variant", "point", "method", "edges", "full_edges", "edge_ratio")}
                for split in session_ids:
                    vals = [r[split + "_coverage"] for r in group]
                    row[split + "_coverage"] = float(np.mean(vals))
                    row[split + "_coverage_seed_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.
                curves.append(row)
    intervals, rng = [], np.random.default_rng(20261025)
    for variant in VARIANTS:
        for split, ids in session_ids.items():
            native = vectors[key(variant, 0, "tracecover", None, split)]
            for method in METHODS[1:]:
                baseline = np.mean([vectors[key(variant, 0, method, seed, split)] for seed in seeds], axis=0) if method == "random" else vectors[key(variant, 0, method, None, split)]
                difference = native - baseline
                lo, hi = cluster_bootstrap(difference, [x["dialogue_id"] for x in ids], args.bootstrap, rng)
                intervals.append({"variant": variant, "split": split, "baseline": method,
                    "native_minus_baseline_pp": float(difference.mean()) * 100,
                    "ci95_low_pp": lo * 100, "ci95_high_pp": hi * 100})
    assert sha(selection_path) == freeze
    dump(out / "session_ids.json", session_ids)
    dump(out / "evaluation_audit.json", {"selection_sha256_unchanged": True,
        "all_plans_directly_recounted_exact_size_and_independently_reachable": True,
        "plans": len(plans), "data": audit,
        "session_counts": {s: {"projections": len(ids), "unique_original_dialogues": len({i["dialogue_id"] for i in ids})} for s, ids in session_ids.items()}})
    dump(out / "results.json", {"protocol": protocol, "runs": runs, "curves": curves, "endpoint_bootstrap": intervals})
    np.savez_compressed(out / "session_coverage.npz", **vectors)
    for filename, rows in (("per_family.csv", scored), ("curves.csv", curves), ("runs.csv", runs), ("endpoint_bootstrap.csv", intervals)):
        with (out / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("DONE " + str(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/eval/sgd/protocol_20pct_seed42_seen_schema_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families")
    parser.add_argument("--points", type=int, default=11)
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.points < 2 or args.random_seeds < 2 or args.bootstrap < 2:
        parser.error("At least two points, random seeds and bootstrap draws required")
    with patch("llm.chat", side_effect=AssertionError("LLM forbidden in graph experiment")):
        run(args)
