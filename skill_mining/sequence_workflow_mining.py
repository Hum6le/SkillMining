#!/usr/bin/env python3
"""Sequence-aware workflow mining for ABCD subflows.

This module keeps the legacy hypergraph miner intact and provides a newer
alternative that treats workflows as canonical action sequences.  Raw slot
values are preserved as examples, but they are not baked into graph node IDs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


def canonical_operator_id(subflow: str, action_name: str) -> str:
    """Return a stable operator node ID without instance-specific slot values."""
    return f"{subflow}:{str(action_name).strip()}"


def short_label(node_id: str) -> str:
    parts = node_id.split(":", 1)
    return parts[1] if len(parts) == 2 else node_id


@dataclass
class CanonicalStep:
    node_id: str
    action_name: str
    slots: list[str] = field(default_factory=list)
    turn_index: int | None = None


def abcd_to_canonical_sequences(
    conversations: list[dict],
    max_sessions: int | None = None,
) -> list[dict[str, Any]]:
    """Convert ABCD conversations into canonical action sequences.

    Each output session keeps the ordered canonical node IDs and also stores
    slot examples so the skill writer can still describe slot discipline.
    """
    sessions: list[dict[str, Any]] = []
    for conv in conversations:
        if max_sessions and len(sessions) >= max_sessions:
            break

        scenario = conv.get("scenario", {})
        subflow = str(scenario.get("subflow", "unknown"))
        convo_id = str(conv.get("convo_id", "?"))
        steps: list[CanonicalStep] = []

        for turn_idx, turn in enumerate(conv.get("delexed", [])):
            targets = turn.get("targets", [])
            if len(targets) < 3 or targets[1] != "take_action":
                continue

            action_name = str(targets[2]).strip()
            if not action_name:
                continue

            raw_slots = targets[3] if len(targets) > 3 else []
            slots = [str(v) for v in raw_slots] if isinstance(raw_slots, list) else []
            steps.append(
                CanonicalStep(
                    node_id=canonical_operator_id(subflow, action_name),
                    action_name=action_name,
                    slots=slots,
                    turn_index=turn_idx,
                )
            )

        if not steps:
            continue

        sessions.append(
            {
                "session_id": convo_id,
                "subflow": subflow,
                "steps": steps,
                "sequence": [s.node_id for s in steps],
            }
        )

    return sessions


def collapse_consecutive_duplicates(sequence: list[str]) -> list[str]:
    """Remove repeated adjacent actions while preserving revisits later."""
    if not sequence:
        return []
    collapsed = [sequence[0]]
    for node_id in sequence[1:]:
        if node_id != collapsed[-1]:
            collapsed.append(node_id)
    return collapsed


def build_sequence_subgraph(
    sessions: list[dict[str, Any]],
    min_edge_support: int = 2,
    min_edge_ratio: float = 0.1,
    max_nodes: int = 30,
    max_edges: int = 60,
    max_pathways: int = 3,
) -> dict[str, Any]:
    """Build a weighted workflow subgraph from canonical action sequences."""
    node_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    edge_sessions: dict[tuple[str, str], set[str]] = defaultdict(set)
    slot_examples: dict[str, list[list[str]]] = defaultdict(list)

    for session in sessions:
        sid = str(session["session_id"])
        collapsed = collapse_consecutive_duplicates(list(session["sequence"]))

        for step in session["steps"]:
            node_counts[step.node_id] += 1
            if step.slots and len(slot_examples[step.node_id]) < 20:
                slot_examples[step.node_id].append(step.slots)

        for src, tgt in zip(collapsed, collapsed[1:]):
            if src == tgt:
                continue
            edge_counts[(src, tgt)] += 1
            edge_sessions[(src, tgt)].add(sid)

    n_sessions = len(sessions)
    support_threshold = max(min_edge_support, int(n_sessions * min_edge_ratio + 0.999))
    kept_edges = [
        (src, tgt, weight)
        for (src, tgt), weight in edge_counts.items()
        if weight >= support_threshold
    ]

    if not kept_edges and edge_counts:
        # Keep the strongest transitions for very small subflows.
        strongest = max(edge_counts.values())
        kept_edges = [
            (src, tgt, weight)
            for (src, tgt), weight in edge_counts.items()
            if weight == strongest
        ]

    edge_nodes = {src for src, _, _ in kept_edges} | {tgt for _, tgt, _ in kept_edges}
    selected_nodes = set(edge_nodes)

    for node_id, _ in node_counts.most_common(max_nodes):
        if len(selected_nodes) >= max_nodes:
            break
        selected_nodes.add(node_id)

    nodes = [
        {
            "id": node_id,
            "label": short_label(node_id),
            "frequency": int(node_counts[node_id]),
            "slot_examples": slot_examples.get(node_id, [])[:5],
        }
        for node_id in sorted(selected_nodes, key=lambda n: (-node_counts[n], n))
    ]

    edges = [
        {
            "source": src,
            "target": tgt,
            "weight": int(weight),
            "num_sessions": len(edge_sessions[(src, tgt)]),
            "probability": round(weight / max(node_counts[src], 1), 4),
        }
        for src, tgt, weight in sorted(kept_edges, key=lambda x: (-x[2], x[0], x[1]))
        if src in selected_nodes and tgt in selected_nodes
    ][:max_edges]

    subgraph = {
        "nodes": nodes,
        "edges": edges,
        "n_selected_nodes": len(nodes),
        "n_selected_edges": len(edges),
        "n_sessions": n_sessions,
        "coverage_pct": _transition_coverage_pct(sessions, edges),
        "min_edge_support": support_threshold,
    }
    subgraph["pathways"] = find_sequence_pathways(subgraph, top_k=max_pathways)
    subgraph["branch_points"] = find_sequence_branch_points(subgraph)
    return subgraph


def _transition_coverage_pct(sessions: list[dict[str, Any]], edges: list[dict]) -> float:
    kept = {(e["source"], e["target"]) for e in edges}
    total = 0
    covered = 0
    for session in sessions:
        collapsed = collapse_consecutive_duplicates(list(session["sequence"]))
        transitions = list(zip(collapsed, collapsed[1:]))
        total += len(transitions)
        covered += sum(1 for edge in transitions if edge in kept)
    return round(100 * covered / max(total, 1), 1)


def find_sequence_pathways(subgraph: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
    """Extract high-support greedy paths from the sequence graph."""
    nodes = {n["id"]: n for n in subgraph.get("nodes", [])}
    outgoing: dict[str, list[dict]] = defaultdict(list)
    incoming: dict[str, list[dict]] = defaultdict(list)
    for edge in subgraph.get("edges", []):
        outgoing[edge["source"]].append(edge)
        incoming[edge["target"]].append(edge)

    for node_id in outgoing:
        outgoing[node_id].sort(
            key=lambda e: (-e["weight"], -e.get("probability", 0), e["target"])
        )

    entries = [node_id for node_id in nodes if node_id not in incoming]
    if not entries:
        entries = [n["id"] for n in sorted(nodes.values(), key=lambda n: -n["frequency"])]

    pathways: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, ...]] = set()
    for entry in entries:
        path = _walk_highest_support_path(entry, nodes, outgoing)
        path_key = tuple(step["node"] for step in path)
        if len(path) <= 1 or path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        total_weight = sum(step.get("edge_weight", 0) for step in path)
        pathways.append(
            {
                "entry": entry,
                "steps": path,
                "length": len(path),
                "total_weight": total_weight,
            }
        )
        if len(pathways) >= top_k:
            break

    pathways.sort(key=lambda p: (-p["total_weight"], -p["length"]))
    return pathways


def _walk_highest_support_path(
    entry: str,
    nodes: dict[str, dict],
    outgoing: dict[str, list[dict]],
    max_steps: int = 20,
) -> list[dict[str, Any]]:
    path: list[dict[str, Any]] = []
    current = entry
    visited: set[str] = set()

    for _ in range(max_steps):
        if current in visited or current not in nodes:
            break
        visited.add(current)
        path.append({"node": current, "label": nodes[current]["label"]})

        candidates = [edge for edge in outgoing.get(current, []) if edge["target"] not in visited]
        if not candidates:
            break

        best = candidates[0]
        path[-1]["next"] = best["target"]
        path[-1]["edge_weight"] = best["weight"]
        path[-1]["edge_probability"] = best.get("probability", 0)
        current = best["target"]

    return path


def find_sequence_branch_points(
    subgraph: dict[str, Any],
    min_probability: float = 0.15,
) -> list[dict[str, Any]]:
    """Find meaningful branch points from retained outgoing transitions."""
    node_labels = {n["id"]: n["label"] for n in subgraph.get("nodes", [])}
    outgoing: dict[str, list[dict]] = defaultdict(list)
    for edge in subgraph.get("edges", []):
        if edge.get("probability", 0) >= min_probability:
            outgoing[edge["source"]].append(edge)

    branches: list[dict[str, Any]] = []
    for node_id, edges in outgoing.items():
        if len(edges) < 2:
            continue
        ordered = sorted(edges, key=lambda e: (-e["weight"], e["target"]))
        branches.append(
            {
                "node": node_id,
                "label": node_labels.get(node_id, short_label(node_id)),
                "num_branches": len(ordered),
                "branches": [
                    {
                        "target": edge["target"],
                        "weight": edge["weight"],
                        "probability": edge.get("probability", 0),
                        "label": node_labels.get(edge["target"], short_label(edge["target"])),
                    }
                    for edge in ordered
                ],
            }
        )

    return sorted(branches, key=lambda b: (-b["num_branches"], b["node"]))


def canonical_sequences_to_operator_results(
    sessions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose canonical sequences in the legacy operator_results shape."""
    results: list[dict[str, Any]] = []
    for idx, session in enumerate(sessions):
        ordered_operations = []
        for node_id in session["sequence"]:
            subflow, action = node_id.split(":", 1)
            ordered_operations.append([subflow, action])
        results.append(
            {
                "session_id": session["session_id"],
                "index": idx,
                "ordered_operations": ordered_operations,
            }
        )
    return results


def mine_sequence_workflow(
    subflow: str,
    conversations: list[dict],
    min_edge_support: int = 2,
    min_edge_ratio: float = 0.1,
    max_nodes: int = 30,
) -> dict[str, Any]:
    """Mine a canonical, sequence-aware workflow for one subflow."""
    sessions = abcd_to_canonical_sequences(conversations)
    subgraph = build_sequence_subgraph(
        sessions,
        min_edge_support=min_edge_support,
        min_edge_ratio=min_edge_ratio,
        max_nodes=max_nodes,
    )
    selected_vertices = [node["id"] for node in subgraph["nodes"]]
    return {
        "skill_info": {
            "selected_vertices": selected_vertices,
            "num_selected": len(selected_vertices),
            "coverage_pct": subgraph["coverage_pct"],
            "num_sessions": len(conversations),
            "mining_method": "sequence",
        },
        "subgraph": subgraph,
        "operator_results": canonical_sequences_to_operator_results(sessions),
        "canonical_sessions": sessions,
    }
