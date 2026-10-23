"""Experimental, graph-only TraceCover; production miners are unchanged.

Start from the full observed DFG. Minimize C(S) - beta * sum(w[e] for e in S)
by strictly improving deletions and one-edge exchanges, subject to saturated
trace coverage and reachability from observed entries. No tree is computed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import math
import time
from typing import Any, Callable

import numpy as np

from eval_tod.abcd.action_schema import canonical_action_name, load_action_schema


def observed_graph(flow: str, conversations: list[dict[str, Any]]) -> dict[str, Any]:
    schema = load_action_schema()
    nodes: Counter[str] = Counter()
    starts: Counter[str] = Counter()
    occurrences: Counter[tuple[str, str]] = Counter()
    supporting: dict[tuple[str, str], set[str]] = defaultdict(set)
    sessions = []
    for index, conversation in enumerate(conversations):
        sequence = []
        for turn in conversation.get("delexed") or []:
            target = turn.get("targets") or []
            if len(target) < 3 or target[1] != "take_action" or not target[2]:
                continue
            action, _ = canonical_action_name(target[2], schema.get("actions"))
            if action:
                sequence.append(f"{flow}:{action}")
        nodes.update(sequence)
        if sequence:
            starts[sequence[0]] += 1
        counts = Counter(zip(sequence, sequence[1:]))
        if counts:
            sessions.append(counts)
        occurrences.update(counts)
        sid = str(conversation.get("convo_id") or f"row-{index}")
        for edge in counts:
            supporting[edge].add(sid)
    total_actions = sum(nodes.values())
    edges = []
    for (source, target), count in sorted(occurrences.items()):
        probability = count / nodes[source]
        prior = nodes[target] / total_actions
        lift = probability / max(prior, 1e-9)
        n = len(supporting[source, target])
        # Match the existing DFG's serialized support/lift scores exactly.
        score = round(math.log1p(n) + .5 * math.log(max(lift, 1e-9)), 4)
        edges.append({"source": source, "target": target, "support": count,
                      "num_sessions": n, "score": score, "lift": round(lift, 4),
                      "probability": round(probability, 4),
                      "evidence_session_ids": sorted(supporting[source, target])[:3]})
    return {"nodes": [{"id": node, "action": node.split(":", 1)[-1], "frequency": nodes[node]}
                      for node in sorted(nodes)], "edges": edges, "sessions": sessions,
            "entries": dict(sorted(starts.items())), "train_dialogues": len(conversations)}


def mine_treefree_trace_cover(
    flow: str, conversations: list[dict[str, Any]], *,
    evidence: str = "log", rho: float = .8, eta: float = .95,
    gamma: float = 2.0, beta: float = 1.0, max_moves: int = 512,
    shortlist_size: int = 16, snapshot_moves: tuple[int, ...] = (12,),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if evidence not in {"log", "support", "support_mean_matched", "none"}:
        raise ValueError("Unknown evidence weighting")
    if not 0 < rho <= 1 or not 0 < eta <= 1 or gamma < 0 or beta < 0:
        raise ValueError("Invalid objective or coverage parameters")
    if max_moves < 0 or shortlist_size < 1:
        raise ValueError("Invalid search budget")
    data = observed_graph(flow, conversations)
    rows, sessions = data["edges"], data["sessions"]
    if not sessions:
        raise ValueError("At least one training dialogue must contain a transition")
    pairs = [(row["source"], row["target"]) for row in rows]
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    sources = Counter()
    incidence: dict[int, dict[int, int]] = defaultdict(dict)
    saturation = np.array([rho * sum(counts.values()) for counts in sessions])
    for session, counts in enumerate(sessions):
        for pair, count in counts.items():
            incidence[pair_index[pair]][session] = count
            sources[pair[0]] += count
    total = sum(sources.values())
    source_p = {node: count / total for node, count in sources.items()}
    log_weights = np.array([row["score"] for row in rows], dtype=float)
    support_weights = np.array([row["num_sessions"] / data["train_dialogues"] for row in rows])
    scale = 1.0
    if evidence == "log":
        weights = log_weights
    elif evidence == "none":
        weights = np.zeros(len(rows))
    else:
        if evidence == "support_mean_matched":
            scale = float(log_weights.mean() / support_weights.mean())
            if scale <= 0:
                raise ValueError("Mean matching requires a positive mean log score")
        weights = support_weights * scale

    selected = set(range(len(pairs)))
    covered = np.array([sum(counts.values()) for counts in sessions], dtype=float)
    nodes = {row["id"] for row in data["nodes"]}
    entries = set(data["entries"])
    history, snapshots = [], {}
    evaluated = coverage_checks = reachability_checks = 0
    delta_cache = {}
    started = time.perf_counter()

    def scores(indices):
        degrees = Counter(pairs[index][0] for index in indices)
        cost = len(indices) + gamma * sum(source_p.get(u, 0) * d * (d - 1) / 2
                                         for u, d in degrees.items())
        weight = sum(float(weights[index]) for index in sorted(indices))
        return cost, weight, cost - beta * weight

    def reachable(removed=-1, added=-1):
        adjacency = defaultdict(list)
        for index in selected:
            if index != removed:
                source, target = pairs[index]
                adjacency[source].append(target)
        if added >= 0:
            source, target = pairs[added]
            adjacency[source].append(target)
        seen, queue = set(entries), list(entries)
        for source in queue:
            for target in adjacency[source]:
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return nodes <= seen

    def changes(removed, added):
        key = removed, added
        if key not in delta_cache:
            delta = {i: -count for i, count in incidence[removed].items()}
            if added >= 0:
                for i, count in incidence[added].items():
                    delta[i] = delta.get(i, 0) + count
            indices = np.array(sorted(delta), dtype=int)
            delta_cache[key] = indices, np.array([delta[i] for i in indices], dtype=float)
        return delta_cache[key]

    def materialize(stop_reason):
        cost, weight, objective = scores(selected)
        local = {node: [] for node in sorted(nodes)}
        for index in selected:
            row = {**rows[index], "kind": "selected", "objective_weight": float(weights[index])}
            local[row["source"]].append(row)
        for outgoing in local.values():
            outgoing.sort(key=lambda row: (-row["objective_weight"], -row["num_sessions"],
                                           -row["support"], row["target"]))
            for priority, row in enumerate(outgoing, 1):
                row["priority"] = priority
        return {"nodes": copy.deepcopy(data["nodes"]), "edges": copy.deepcopy(rows),
                "local_transitions": local,
                "residual_edges": [{**rows[index], "kind": "residual"}
                                   for index in sorted(set(range(len(rows))) - selected)],
                # Compatibility field only: no tree is constructed or selected.
                "backbone": {"root": "<START>", "edges": []},
                "entry_nodes": data["entries"], "mining_method": "treefree_trace_cover",
                "treefree_trace_cover": {
                    "evidence": evidence, "support_scale": scale, "rho": rho, "eta": eta,
                    "gamma": gamma, "beta": beta, "max_moves": max_moves,
                    "shortlist_size": shortlist_size, "iterations": len(history),
                    "stop_reason": stop_reason, "coverage": float(np.minimum(covered / saturation, 1).mean()),
                    "cost": cost, "evidence_reward": weight, "objective": objective,
                    "reachable": reachable(), "candidate_pool": "all_observed_transitions",
                    "observed_edges": len(rows), "selected_edges": len(selected),
                    "proposals_evaluated": evaluated, "coverage_checks": coverage_checks,
                    "reachability_checks": reachability_checks,
                    "elapsed_seconds": time.perf_counter() - started,
                    "accepted_moves": copy.deepcopy(history)}}

    assert reachable()
    _, _, objective = scores(selected)
    stop_reason = "move_budget"
    for step in range(max_moves):
        degrees = Counter(pairs[index][0] for index in selected)
        unused = sorted(set(range(len(pairs))) - selected)
        proposals = []
        for removed in sorted(selected):
            source = pairs[removed][0]
            dc = -1 - gamma * source_p.get(source, 0) * (degrees[source] - 1)
            delta_j = dc + beta * float(weights[removed])
            evaluated += 1
            if delta_j < -1e-10:
                proposals.append((delta_j, removed, -1))
            for added in unused:
                new_source = pairs[added][0]
                add_degree = degrees[new_source] - int(new_source == source)
                delta_j = (dc + 1 + gamma * source_p.get(new_source, 0) * add_degree
                           + beta * float(weights[removed] - weights[added]))
                evaluated += 1
                if delta_j < -1e-10:
                    proposals.append((delta_j, removed, added))
        proposals.sort()
        utility = np.minimum(covered / saturation, 1)
        fitness = float(utility.mean())
        coverage_feasible = 0
        best = None
        # Lazy coverage/reachability checks preserve the same first-q rule.
        # Once a reachable candidate is found, all later exact scores are worse
        # or tied, so building the rest of the shortlist cannot change the winner.
        for delta_j, removed, added in proposals:
            indices, delta = changes(removed, added)
            next_fitness = fitness + float((np.minimum((covered[indices] + delta) / saturation[indices], 1)
                                           - utility[indices]).sum()) / len(sessions)
            coverage_checks += 1
            if next_fitness + 1e-12 >= eta:
                coverage_feasible += 1
                reachability_checks += 1
                if reachable(removed, added):
                    best = (delta_j, removed, added, next_fitness, indices, delta)
                    break
                if coverage_feasible >= shortlist_size:
                    break
        if best is None:
            stop_reason = "shortlist_no_improvement" if coverage_feasible else "neighborhood_no_improvement"
            break
        delta_j, removed, added, next_fitness, indices, delta = best
        previous = objective
        selected.remove(removed)
        if added >= 0:
            selected.add(added)
        covered[indices] += delta
        cost, weight, objective = scores(selected)
        assert objective < previous - 1e-10
        assert abs(objective - (previous + delta_j)) < 1e-7
        assert float(np.minimum(covered / saturation, 1).mean()) + 1e-12 >= eta
        history.append({"step": step + 1, "removed": list(pairs[removed]),
                        "added": list(pairs[added]) if added >= 0 else None,
                        "objective_before": previous, "objective_after": objective,
                        "cost": cost, "evidence_reward": weight, "fitness": next_fitness,
                        "selected_edges": len(selected)})
        if step + 1 in snapshot_moves:
            snapshots[step + 1] = materialize("snapshot_budget")
        if progress and (step + 1) % 25 == 0:
            progress(f"{flow}/{evidence}: moves={step+1} edges={len(selected)} F={next_fitness:.4f}")
    graph = materialize(stop_reason)
    for budget in snapshot_moves:
        if budget not in snapshots and budget >= len(history):
            snapshots[budget] = copy.deepcopy(graph)
    return {"subgraph": graph, "snapshots": snapshots}
