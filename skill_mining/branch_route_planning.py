"""Deterministically organize retained backbone edges into routing clusters."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any


def edge_key(source: str, target: str) -> str:
    return f"{source}=>{target}"


def _first_main_ancestor(
    node_id: str,
    parent: dict[str, str],
    main_nodes: set[str],
) -> str | None:
    current = node_id
    seen: set[str] = set()
    while current not in seen:
        if current in main_nodes:
            return current
        seen.add(current)
        current = parent.get(current, "")
        if not current:
            return None
    return None


def _is_backbone_ancestor(
    candidate: str,
    node_id: str,
    parent: dict[str, str],
) -> bool:
    current = node_id
    seen: set[str] = set()
    while current not in seen:
        if current == candidate:
            return True
        seen.add(current)
        current = parent.get(current, "")
        if not current:
            return False
    return False


def _find_rejoin_path(
    start: str,
    outgoing: dict[str, list[dict[str, Any]]],
    main_nodes: set[str],
    max_hops: int = 6,
) -> tuple[str | None, list[str]]:
    """Find a compact retained-edge route from a branch node back to the spine."""
    if start in main_nodes:
        return start, [start]
    queue: deque[tuple[str, list[str]]] = deque([(start, [start])])
    visited = {start}
    while queue:
        node_id, path = queue.popleft()
        if len(path) - 1 >= max_hops:
            continue
        candidates = sorted(
            outgoing.get(node_id, []),
            key=lambda edge: (-float(edge.get("probability", 0.0)), edge["target"]),
        )
        for edge in candidates:
            target = edge["target"]
            next_path = path + [target]
            if target in main_nodes:
                return target, next_path
            if target not in visited:
                visited.add(target)
                queue.append((target, next_path))
    return None, [start]


def build_branch_route_plan(subgraph: dict[str, Any]) -> dict[str, Any]:
    """Build a backbone-aligned routing plan without an LLM.

    Each retained non-main edge occurs exactly once in an anchor cluster.  The
    plan adds structural metadata (attachment anchor, likely rejoin point,
    retry/return type) so the LLM can write routes rather than a flat edge list.
    """
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    backbone = subgraph.get("backbone", {})
    main_path = [node_id for node_id in backbone.get("main_path", []) if node_id in nodes]
    main_nodes = set(main_path)
    main_pairs = set(zip(main_path, main_path[1:]))
    parent = {
        edge["target"]: edge["source"]
        for edge in backbone.get("edges", [])
        if edge.get("source") and edge.get("target")
    }
    outgoing = {
        source: list(edges)
        for source, edges in (subgraph.get("local_transitions") or {}).items()
    }
    induction = (subgraph.get("transition_induction") or {}).get("rules_by_source", {})

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source, edges in outgoing.items():
        for edge in edges:
            target = edge.get("target")
            if not target or source not in nodes or target not in nodes:
                continue
            if (source, target) in main_pairs:
                continue
            anchor = _first_main_ancestor(source, parent, main_nodes) or source
            rejoin, route_nodes = _find_rejoin_path(target, outgoing, main_nodes)
            kind = str(edge.get("kind") or "branch")
            if target == source or _is_backbone_ancestor(target, source, parent):
                route_type = "retry_or_loop"
            elif rejoin:
                route_type = "rejoin_main_path"
            else:
                route_type = "alternative_or_terminal"
            induced = next(
                (rule for rule in induction.get(source, []) if rule.get("target") == target),
                None,
            )
            clusters[anchor].append({
                "edge_id": edge_key(source, target),
                "source": source,
                "source_label": nodes[source]["label"],
                "target": target,
                "target_label": nodes[target]["label"],
                "kind": kind,
                "route_type": route_type,
                "priority": int((induced or edge).get("priority", edge.get("priority", 1))),
                "relation": str((induced or {}).get("relation") or "unspecified"),
                "condition": str((induced or {}).get("condition") or "").strip(),
                "observed_condition": edge.get("condition", {}),
                "support": int(edge.get("support", 0)),
                "probability": float(edge.get("probability", 0.0)),
                "likely_rejoin": rejoin,
                "likely_rejoin_label": nodes[rejoin]["label"] if rejoin in nodes else "",
                "suggested_route_nodes": route_nodes,
                "suggested_route_labels": [nodes[item]["label"] for item in route_nodes if item in nodes],
            })

    ordered_clusters = []
    for anchor, routes in sorted(
        clusters.items(),
        key=lambda item: (main_path.index(item[0]) if item[0] in main_path else len(main_path), item[0]),
    ):
        routes.sort(key=lambda route: (route["priority"], route["source"], route["target"]))
        normal_next = ""
        if anchor in main_path:
            index = main_path.index(anchor)
            if index + 1 < len(main_path):
                normal_next = main_path[index + 1]
        ordered_clusters.append({
            "anchor": anchor,
            "anchor_label": nodes.get(anchor, {}).get("label", anchor),
            "normal_next": normal_next,
            "normal_next_label": nodes.get(normal_next, {}).get("label", ""),
            "routes": routes,
        })

    return {
        "format": "backbone_branch_route_plan_v1",
        "main_path": main_path,
        "main_path_labels": [nodes[node_id]["label"] for node_id in main_path],
        "clusters": ordered_clusters,
        "selected_edge_ids": sorted(
            route["edge_id"]
            for cluster in ordered_clusters
            for route in cluster["routes"]
        ),
    }
