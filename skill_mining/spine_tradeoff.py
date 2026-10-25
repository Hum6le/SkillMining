"""CPU-only, exact-size graph comparisons; no production miner modifications."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import numpy as np


def reachable(nodes, entries, pairs):
    outgoing = defaultdict(list)
    for source, target in pairs:
        outgoing[source].append(target)
    seen = set(entries)
    queue = list(sorted(seen))
    for source in queue:
        for target in outgoing[source]:
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return set(nodes) <= seen, seen


def ranked_select(pairs, order, k, nodes, entries):
    """Top-k, ranked crossing-edge repair, then ranked reachability-safe pruning.

    Every real edge added for repair counts toward k. This is a transparent
    heuristic, not an optimal fixed-budget modular-score solver.
    """
    pairs = list(pairs)
    order = list(order)
    if sorted(order) != list(range(len(pairs))):
        raise ValueError("Ranking must be a permutation of the observed edges")
    if not len(set(nodes) - set(entries)) <= k <= len(pairs):
        raise ValueError("Infeasible edge budget")
    if not reachable(nodes, entries, pairs)[0]:
        raise ValueError("Observed graph cannot reach all nodes from the observed entries")
    selected = set(order[:k])
    added, pruned = [], []
    while True:
        ok, seen = reachable(nodes, entries, [pairs[i] for i in selected])
        if ok:
            break
        candidates = [i for i in order if i not in selected and pairs[i][0] in seen and pairs[i][1] not in seen]
        if not candidates:
            raise RuntimeError("No observed crossing edge for reachability repair")
        selected.add(candidates[0])
        added.append(candidates[0])
    for i in reversed(order):
        if len(selected) == k:
            break
        if i in selected and reachable(nodes, entries, [pairs[j] for j in selected if j != i])[0]:
            selected.remove(i)
            pruned.append(i)
    if len(selected) != k or not reachable(nodes, entries, [pairs[i] for i in selected])[0]:
        raise RuntimeError("Exact-budget reachability repair failed")
    return selected, {"repair_added_indices": added, "repair_pruned_indices": pruned,
                      "repair_added": len(added), "repair_pruned": len(pruned)}


def native_states(graph, pairs):
    """Last native training-search state at each cardinality, including exchanges."""
    pair_index = {pair: i for i, pair in enumerate(pairs)}
    selected = set(range(len(pairs)))
    states = {len(selected): set(selected)}
    for move in graph["treefree_trace_cover"]["accepted_moves"]:
        removed = pair_index[tuple(move["removed"])]
        added = pair_index[tuple(move["added"])] if move["added"] is not None else None
        if removed not in selected or (added is not None and added in selected):
            raise ValueError("Invalid native move replay")
        selected.remove(removed)
        if added is not None:
            selected.add(added)
        if len(selected) != move["selected_edges"]:
            raise ValueError("Native move cardinality mismatch")
        states[len(selected)] = set(selected)
    endpoint = {(e["source"], e["target"]) for edges in graph["local_transitions"].values() for e in edges}
    if {pairs[i] for i in selected} != endpoint:
        raise ValueError("Native replay differs from saved final graph")
    if set(states) != set(range(len(selected), len(pairs) + 1)):
        raise ValueError("Native search path skips edge cardinalities")
    return states


def rankings(rows, evidence, support_scale, train_dialogues, seed, flow):
    frequency = sorted(range(len(rows)), key=lambda i: (-rows[i]["support"], rows[i]["source"], rows[i]["target"]))
    if evidence == "log":
        values = [r["score"] for r in rows]
    elif evidence == "support_mean_matched":
        values = [r["num_sessions"] / train_dialogues * support_scale for r in rows]
    else:
        raise ValueError("This experiment only evaluates the two explicitly named native variants")
    weight = sorted(range(len(rows)), key=lambda i: (-values[i], rows[i]["source"], rows[i]["target"]))
    flow_seed = int.from_bytes(hashlib.sha256(f"{flow}/{seed}".encode()).digest()[:8], "little")
    random = np.random.default_rng(flow_seed).permutation(len(rows)).tolist()
    return {"frequency": frequency, "weight": weight, "random": random}, values


def make_plans(flow, data, graph, alphas, seeds, variant):
    rows = data["edges"]
    pairs = [(r["source"], r["target"]) for r in rows]
    nodes = {n["id"] for n in data["nodes"]}
    entries = set(data["entries"])
    meta = graph["treefree_trace_cover"]
    states = native_states(graph, pairs)
    minimum, full = min(states), len(pairs)
    orders, values = rankings(rows, meta["evidence"], meta["support_scale"], data["train_dialogues"], seeds[0], flow)
    plans = []
    for point, alpha in enumerate(alphas):
        k = minimum + math.ceil(alpha * (full - minimum) - 1e-12)
        common = {"flow": flow, "variant": variant, "point": point, "alpha": alpha,
                  "budget": k, "observed_edges": full}
        selected = states[k]
        if not reachable(nodes, entries, [pairs[i] for i in selected])[0]:
            raise ValueError("Native checkpoint violates entry reachability")
        plans.append({**common, "method": "tracecover", "seed": None, "indices": sorted(selected),
                      "repair_added": 0, "repair_pruned": 0})
        for method in ("frequency", "weight"):
            selected, repair = ranked_select(pairs, orders[method], k, nodes, entries)
            plans.append({**common, "method": method, "seed": None, "indices": sorted(selected), **repair})
        for seed in seeds:
            order = rankings(rows, meta["evidence"], meta["support_scale"], data["train_dialogues"], seed, flow)[0]["random"]
            selected, repair = ranked_select(pairs, order, k, nodes, entries)
            plans.append({**common, "method": "random", "seed": seed, "indices": sorted(selected), **repair})
    return plans, values


def coverage_matrix(sequences, pairs):
    """Retained occurrences / ORIGINAL length; unseen pairs stay in denominator."""
    lookup = {pair: i for i, pair in enumerate(pairs)}
    counts = [Counter(zip(s, s[1:])) for s in sequences if len(s) > 1]
    lengths = np.array([sum(c.values()) for c in counts], dtype=np.int64)
    matrix = np.zeros((len(counts), len(pairs)), dtype=np.float64)
    unseen = 0
    for i, c in enumerate(counts):
        for pair, n in c.items():
            if pair in lookup:
                matrix[i, lookup[pair]] = n / lengths[i]
            else:
                unseen += n
    return matrix, lengths, {"all_sessions": len(sequences), "scored_sessions": len(counts),
        "excluded_zero_transition_sessions": len(sequences) - len(counts),
        "transition_occurrences": int(lengths.sum()), "unseen_transition_occurrences": unseen}


def coverage_vector(matrix, selected):
    if not selected:
        return np.zeros(len(matrix), dtype=float)
    return matrix[:, sorted(selected)].sum(axis=1)
