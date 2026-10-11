"""Backbone-aware feasible local search, as a separate TraceCover branch.

Minimize C(G) - beta max_{B subset G} W(B) subject to saturated trace fitness.
The old TraceCover graph is an initializer, not an immutable backbone. Search
uses deletions and one-edge exchanges; a configurable shortlist bounds the
expensive arborescence evaluations. No global/local optimality is claimed.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import networkx as nx

from skill_mining.backbone_workflow_mining import (
    ROOT, _edge_observation_index, _has_path, _observable_distinction,
    _session_edge_occurrences, mine_backbone_workflow_trace_cover,
    observable_transition_events,
)


Pair = tuple[str, str]


def mine_backbone_workflow_joint_trace_cover(
    subflow: str, conversations: list[dict[str, Any]], *,
    min_branch_support: int = 2, trace_coverage_target: float = 0.8,
    corpus_fitness_target: float = 0.95, routing_complexity_weight: float = 2.0,
    tree_evidence_weight: float = 1.0, max_moves: int = 12,
    search_width: int = 16, search_mode: str = "joint",
    ranking: str = "backbone", allow_exchanges: bool = True,
    root_policy: str = "observed",
) -> dict[str, Any]:
    if not math.isfinite(tree_evidence_weight) or tree_evidence_weight < 0:
        raise ValueError("tree_evidence_weight must be finite and nonnegative")
    if max_moves < 0 or search_width < 1:
        raise ValueError("max_moves must be nonnegative and search_width positive")
    if search_mode not in {"joint", "fixed"} or ranking not in {"backbone", "support"}:
        raise ValueError("Unknown search mode or ranking")
    if root_policy not in {"observed", "all"}:
        raise ValueError("Unknown root policy")
    base = mine_backbone_workflow_trace_cover(
        subflow, conversations, min_branch_support=min_branch_support,
        trace_coverage_target=trace_coverage_target,
        corpus_fitness_target=corpus_fitness_target,
        routing_complexity_weight=routing_complexity_weight,
    )
    graph = base["subgraph"]
    edge_rows = {(e["source"], e["target"]): e for e in graph["edges"]}
    selected = {
        (e["source"], e["target"])
        for edges in graph["local_transitions"].values() for e in edges
    }
    initial_selected = set(selected)
    tree_rows = list(graph["backbone"]["edges"])
    fixed_pairs = {(e["source"], e["target"]) for e in tree_rows if e["source"] != ROOT}
    initial_tree_pairs = set(fixed_pairs)
    # Preserve the original support policy: initial backbone exemptions plus
    # support-qualified residuals. The pool is frozen before local search.
    eligible = fixed_pairs | {
        pair for pair, row in edge_rows.items() if int(row["support"]) >= min_branch_support
    }
    starts = Counter()
    for result in base["operator_results"]:
        operations = result.get("ordered_operations") or []
        if operations:
            starts[f"{operations[0][0]}:{operations[0][1]}"] += 1
    nodes = [row["id"] for row in graph["nodes"]]
    roots = [{
        "source": ROOT, "target": node, "support": starts[node],
        "num_sessions": starts[node], "score": math.log1p(starts[node]) - 0.25,
        "probability": 0.0, "lift": 0.0,
        "condition": {"kind": "session_entry"}, "evidence_session_ids": [],
        "kind": "backbone",
    } for node in nodes if root_policy == "all" or starts[node] > 0]
    sessions = _session_edge_occurrences(subflow, conversations)
    n = max(len(sessions), 1)
    saturation = [trace_coverage_target * sum(row.values()) for row in sessions]
    covered = [sum(c for e, c in row.items() if e in selected) for row in sessions]
    incidence: dict[Pair, list[tuple[int, int]]] = defaultdict(list)
    source_counts = Counter()
    for i, row in enumerate(sessions):
        for pair, count in row.items():
            incidence[pair].append((i, count))
            source_counts[pair[0]] += count
    total = max(sum(source_counts.values()), 1)
    source_p = {u: c / total for u, c in source_counts.items()}

    def fitness(values):
        return sum(min(c / s, 1.0) for c, s in zip(values, saturation)) / n

    def complexity(pairs):
        degrees = Counter(u for u, _ in pairs)
        routing = sum(source_p.get(u, 0) * d * (d - 1) / 2 for u, d in degrees.items())
        return len(pairs) + routing_complexity_weight * routing, routing

    tree_cache = {}
    tree_calls = 0

    def tree_for(pairs):
        nonlocal tree_calls
        key = frozenset(pairs)
        if key in tree_cache:
            return tree_cache[key]
        directed = nx.DiGraph()
        directed.add_nodes_from([ROOT, *nodes])
        for row in roots:
            directed.add_edge(ROOT, row["target"], weight=row["score"], payload=row)
        for pair in sorted(pairs):
            if pair[0] != pair[1]:
                row = edge_rows[pair]
                directed.add_edge(*pair, weight=float(row["score"]), payload=row)
        if len(nx.descendants(directed, ROOT)) != len(nodes):
            tree_cache[key] = (-math.inf, [])
            return tree_cache[key]
        tree = nx.maximum_spanning_arborescence(directed, attr="weight", preserve_attrs=True)
        rows = [{**data["payload"], "kind": "backbone"}
                for _, _, data in tree.edges(data=True)]
        result = (sum(float(row["score"]) for row in rows), rows)
        tree_cache[key] = result
        tree_calls += 1
        return result

    # The initializer retained a maximum tree of the full observed graph.
    # Computing on the eligible pool explicitly gives a safe global upper
    # bound on tree weight for pruning candidate evaluations.
    max_tree_weight, _ = tree_for(eligible)
    tree_weight, tree_rows = tree_for(selected)
    if not tree_rows and nodes:
        raise ValueError("Initializer is not reachable from observed entries; use root_policy='all' for diagnosis only")
    current_fitness = fitness(covered)
    initial_fitness = current_fitness
    cost, routing = complexity(selected)
    objective = cost - tree_evidence_weight * tree_weight
    initial_objective = objective
    history = []
    proposals_total = 0
    stop_reason = "move_budget"
    if current_fitness + 1e-12 < corpus_fitness_target:
        stop_reason = "initializer_target_unattainable"
    else:
        for step in range(max_moves):
            current_tree = {(e["source"], e["target"]) for e in tree_rows if e["source"] != ROOT}
            degrees = Counter(u for u, _ in selected)
            # Sum of independent best-parent weights upper-bounds a maximum
            # arborescence (it relaxes acyclicity). This tighter bound screens
            # costly backbone removals before spending shortlist evaluations.
            incoming = defaultdict(list)
            for row in roots:
                incoming[row["target"]].append((float(row["score"]), (ROOT, row["target"])))
            for pair in sorted(selected):
                if pair[0] != pair[1]:
                    incoming[pair[1]].append((float(edge_rows[pair]["score"]), pair))
            best_parent_weight = {u: max(value for value, _ in incoming[u]) for u in nodes}
            relaxed_weight = sum(best_parent_weight.values())
            remaining = sorted(eligible - selected)
            proposals = []
            for removed in sorted(selected):
                if search_mode == "fixed" and removed in fixed_pairs:
                    continue
                after_delete = dict(incidence[removed])
                remaining_parent_weight = max(
                    (value for value, pair in incoming[removed[1]] if pair != removed), default=-math.inf,
                )
                # Deleting an edge decreases |E| and its sibling-pair cost.
                delete_cost = cost - 1 - routing_complexity_weight * source_p.get(removed[0], 0) * (degrees[removed[0]] - 1)
                additions = [None]
                if allow_exchanges:
                    additions += remaining
                for added in additions:
                    candidate_cost = delete_cost
                    if added is not None:
                        after_degree = degrees[added[0]] - int(added[0] == removed[0])
                        candidate_cost += 1 + routing_complexity_weight * source_p.get(added[0], 0) * after_degree
                    replacement_parent_weight = remaining_parent_weight
                    if added is not None and added[0] != added[1] and added[1] == removed[1]:
                        replacement_parent_weight = max(replacement_parent_weight, float(edge_rows[added]["score"]))
                    if not math.isfinite(replacement_parent_weight):
                        continue
                    upper_weight = relaxed_weight - best_parent_weight[removed[1]] + replacement_parent_weight
                    if added is not None and added[0] != added[1] and added[1] != removed[1]:
                        upper_weight += max(float(edge_rows[added]["score"]) - best_parent_weight[added[1]], 0.0)
                    upper_weight = (tree_weight if search_mode == "fixed"
                                    else min(max_tree_weight, upper_weight))
                    lower_objective = candidate_cost - tree_evidence_weight * upper_weight
                    if lower_objective >= objective - 1e-10:
                        continue
                    changes = {i: -c for i, c in after_delete.items()}
                    if added is not None:
                        for i, count in incidence[added]:
                            changes[i] = changes.get(i, 0) + count
                    candidate_fitness = current_fitness + sum(
                        min((covered[i] + change) / saturation[i], 1.0)
                        - min(covered[i] / saturation[i], 1.0)
                        for i, change in changes.items()
                    ) / n
                    if candidate_fitness + 1e-12 < corpus_fitness_target:
                        continue
                    proposals.append((lower_objective, -candidate_fitness, removed,
                                      added or ("", ""), candidate_cost, changes))
            proposals.sort(key=lambda item: item[:4])
            proposals_total += len(proposals)
            best = None
            for proposal in proposals[:search_width]:
                _, negative_fitness, removed, added, candidate_cost, changes = proposal
                pairs = selected - {removed}
                if added != ("", ""):
                    pairs.add(added)
                if search_mode == "fixed":
                    candidate_weight, candidate_tree = tree_weight, tree_rows
                elif removed not in current_tree and added == ("", ""):
                    candidate_weight, candidate_tree = tree_weight, tree_rows
                else:
                    candidate_weight, candidate_tree = tree_for(pairs)
                if not math.isfinite(candidate_weight):
                    continue
                candidate_objective = candidate_cost - tree_evidence_weight * candidate_weight
                if candidate_objective >= objective - 1e-10:
                    continue
                key = (candidate_objective, negative_fitness, removed, added)
                if best is None or key < best[0]:
                    best = (key, pairs, candidate_weight, candidate_tree, changes, candidate_cost)
            if best is None:
                stop_reason = "shortlist_no_improvement" if proposals else "neighborhood_no_improvement"
                break
            key, selected, tree_weight, tree_rows, changes, cost = best
            old_objective = objective
            objective = key[0]
            for i, change in changes.items():
                covered[i] += change
            current_fitness = fitness(covered)
            _, routing = complexity(selected)
            new_tree = {(e["source"], e["target"]) for e in tree_rows if e["source"] != ROOT}
            history.append({
                "step": step + 1, "removed": list(key[2]),
                "added": list(key[3]) if key[3] != ("", "") else None,
                "removed_backbone_edge": key[2] in current_tree,
                "backbone_changed_edges": len(current_tree.symmetric_difference(new_tree)),
                "objective_before": old_objective, "objective_after": objective,
                "fitness": current_fitness, "complexity": cost,
                "tree_weight": tree_weight, "retained_edges": len(selected),
                "feasible_proposals": len(proposals),
                "evaluated_proposals": min(len(proposals), search_width),
            })

    # Rebuild all consumer-facing graph metadata; no stale backbone or routing
    # diagnostics survive selection. Keep full observed edges for recall ceilings.
    tree_pairs = {(e["source"], e["target"]) for e in tree_rows if e["source"] != ROOT}
    parent = {e["target"]: e["source"] for e in tree_rows}
    children = defaultdict(list)
    frequencies = {row["id"]: int(row["frequency"]) for row in graph["nodes"]}
    for row in tree_rows:
        children[row["source"]].append(row["target"])
    for siblings in children.values():
        siblings.sort(key=lambda node: (-frequencies[node], node))
    order, queue = [], list(children[ROOT])
    while queue:
        node = queue.pop(0)
        order.append(node)
        queue.extend(children[node])
    main_path, current = [], ROOT
    while children[current]:
        current = max(children[current], key=lambda node: (frequencies[node], node))
        main_path.append(current)
    graph["backbone"] = {"root": ROOT, "edges": sorted(tree_rows, key=lambda e: (e["source"], e["target"])),
                         "compilation_order": order, "main_path": main_path}
    old_priorities = {(e["source"], e["target"]): e["priority"]
                      for edges in graph["local_transitions"].values() for e in edges}
    local, residual = {}, []
    for node in nodes:
        pairs = [pair for pair in selected if pair[0] == node]
        if ranking == "support":
            pairs.sort(key=lambda pair: (-edge_rows[pair]["support"], -float(edge_rows[pair]["score"]), pair))
        else:
            pairs.sort(key=lambda pair: (pair not in tree_pairs, old_priorities.get(pair, 10**9),
                                         -float(edge_rows[pair]["score"]), pair))
        rows = []
        for priority, pair in enumerate(pairs, 1):
            kind = "backbone" if pair in tree_pairs else (
                "retry" if _has_path(parent, node, pair[1]) else "branch")
            row = {**edge_rows[pair], "kind": kind, "priority": priority}
            rows.append(row)
            if kind != "backbone":
                residual.append(row)
        local[node] = rows
    graph["local_transitions"], graph["residual_edges"] = local, residual
    occurrences = sum(covered)
    graph["coverage_pct"] = round(100 * occurrences / total, 1)
    observations = _edge_observation_index(observable_transition_events(subflow, conversations))
    distinctions = []
    for node in nodes:
        pairs = sorted(pair for pair in selected if pair[0] == node)
        for i, left in enumerate(pairs):
            for right in pairs[i + 1:]:
                distinctions.append({"source": node, "left_target": left[1], "right_target": right[1],
                                     **_observable_distinction(observations.get(left, []), observations.get(right, []))})
    graph["trace_cover"].update({
        "final_fitness": round(current_fitness, 8), "selected_residual_edges": len(residual),
        "routing_pair_complexity": round(routing, 8),
        "mean_selected_pair_distinguishability": sum(row["gap"] for row in distinctions) / max(len(distinctions), 1),
        "mean_selected_pair_ambiguity": sum(row["ambiguity"] for row in distinctions) / max(len(distinctions), 1),
        "observable_pair_distinctions": distinctions,
    })
    graph["joint_trace_cover"] = {
        "objective": "edge_count + gamma * routing_pairs - beta * maximum_tree_weight",
        "tree_evidence_weight": tree_evidence_weight, "search_mode": search_mode,
        "ranking": ranking, "max_moves": max_moves, "search_width": search_width,
        "root_policy": root_policy,
        "root_edges": sum(e["source"] == ROOT for e in tree_rows),
        "unsupported_root_edges": sum(e["source"] == ROOT and not e["support"] for e in tree_rows),
        "allow_exchanges": allow_exchanges, "stop_reason": stop_reason,
        "initial_objective": initial_objective, "final_objective": objective,
        "initial_fitness": initial_fitness, "final_fitness": current_fitness,
        "target_feasible": current_fitness + 1e-12 >= corpus_fitness_target,
        "initial_retained_edges": len(initial_selected), "final_retained_edges": len(selected),
        "initial_backbone_edges": len(initial_tree_pairs),
        "backbone_changed_edges": len(initial_tree_pairs.symmetric_difference(tree_pairs)),
        "removed_initial_backbone_edges": len(initial_tree_pairs - tree_pairs),
        "arborescence_evaluations": tree_calls, "feasible_proposals": proposals_total,
        "accepted_moves": history,
        "optimality_claim": "none; bounded shortlist deletion/exchange search",
    }
    graph["mining_method"] = "joint_trace_cover" if search_mode == "joint" else "fixed_trace_cover_pruning"
    base["skill_info"].update(mining_method=graph["mining_method"], coverage_pct=graph["coverage_pct"])
    return base
