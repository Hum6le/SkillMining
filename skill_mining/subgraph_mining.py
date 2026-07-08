#!/usr/bin/env python3
r"""Per-Subflow Subgraph Mining: 从 operator 序列构建加权有向图，提取 subgraph。

与 vertex cover（无序节点集合）不同，这里同时选择节点和边，保留顺序信息：

  operator 序列: A→B→C, A→B→D, A→C→E
       │
  加权 DAG:  A ──2──→ B ──1──→ C
             │                  │
             └──1──→ C ──1──→ E
             └─────→ D (via B)

  提取 subgraph:
    - 选中节点 = vertex cover 选出的算子
    - 选中边 = 两个选中节点之间的高频边
    - 主干道 = 最高权路径

用法：
  python skill_mining/subgraph_mining.py \
    --skills skill_mining/output/abcd_session_hg/per_subflow_vertex_subsets.json \
    --operator-results skill_mining/output/abcd_session_hg/operator_results.json \
    --split train --max-sessions 500
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_SKILL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SKILL_DIR.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SKILL_DIR) in sys.path:
    sys.path.remove(str(_SKILL_DIR))
sys.path.insert(0, str(_SKILL_DIR))

from eval_tod.abcd.data import load_abcd_data
from abcd_session_hg import (
    abcd_to_operator_results,
    SessionHypergraph,
    greedy_vertex_cover,
    group_by_subflow,
)

_OUTPUT_DIR = _SKILL_DIR / "output" / "abcd_subgraph"


# ═══════════════════════════════════════════════════════════════
# DAG construction
# ═══════════════════════════════════════════════════════════════

def build_dag_from_operator_results(
    op_results: list[dict],
) -> dict[str, Any]:
    """Build a weighted directed graph from operator sequences.

    Returns:
        {nodes: {id, label, frequency}, edges: {source, target, weight, sessions}}
    """
    node_freq: dict[str, int] = defaultdict(int)
    edge_freq: dict[tuple[str, str], int] = defaultdict(int)
    edge_sessions: dict[tuple[str, str], set[str]] = defaultdict(set)

    for session in op_results:
        sid = str(session.get("session_id", ""))
        ops = session.get("ordered_operations", [])

        # Count nodes
        seen_ops: set[str] = set()
        for pair in ops:
            if len(pair) < 2:
                continue
            node = f"{pair[0]}:{pair[1]}"
            node_freq[node] += 1
            seen_ops.add(node)

        # Count edges (sequential pairs within this session)
        for i in range(len(ops) - 1):
            if len(ops[i]) < 2 or len(ops[i+1]) < 2:
                continue
            src = f"{ops[i][0]}:{ops[i][1]}"
            tgt = f"{ops[i+1][0]}:{ops[i+1][1]}"
            if src != tgt:  # skip self-loops
                edge_freq[(src, tgt)] += 1
                edge_sessions[(src, tgt)].add(sid)

    nodes = [
        {"id": node, "label": _short_label(node), "frequency": freq}
        for node, freq in sorted(node_freq.items(), key=lambda x: -x[1])
    ]

    edges = [
        {
            "source": src, "target": tgt,
            "weight": weight,
            "num_sessions": len(edge_sessions[(src, tgt)]),
        }
        for (src, tgt), weight in sorted(edge_freq.items(), key=lambda x: -x[1])
    ]

    return {"nodes": nodes, "edges": edges}


def _short_label(node_id: str) -> str:
    """Extract a shorter display label from a full operator name."""
    # "subflow:action_name:slot1,slot2" → "action_name"
    parts = node_id.split(":", 1)
    if len(parts) >= 2:
        return parts[1].split(":")[0] if ":" in parts[1] else parts[1]
    return node_id


# ═══════════════════════════════════════════════════════════════
# Subgraph extraction
# ═══════════════════════════════════════════════════════════════

def extract_subgraph(
    dag: dict,
    selected_vertices: set[str],
    min_edge_weight: int = 1,
) -> dict:
    """Extract subgraph containing selected vertices and edges between them.

    Returns:
        {nodes, edges, n_selected_nodes, n_selected_edges}
    """
    sel_set = set(selected_vertices)

    sub_nodes = [n for n in dag["nodes"] if n["id"] in sel_set]
    sub_edges = [
        e for e in dag["edges"]
        if e["source"] in sel_set and e["target"] in sel_set
        and e["weight"] >= min_edge_weight
    ]

    return {
        "nodes": sub_nodes,
        "edges": sub_edges,
        "n_selected_nodes": len(sub_nodes),
        "n_selected_edges": len(sub_edges),
    }


# ═══════════════════════════════════════════════════════════════
# Pathway analysis
# ═══════════════════════════════════════════════════════════════

def find_main_pathway(
    subgraph: dict,
    top_k: int = 3,
) -> list[dict]:
    """Find the highest-weight paths through the subgraph.

    Uses a greedy approach: start from nodes with no incoming edges,
    follow the highest-weight outgoing edge until a node with no outgoing edges.
    """
    nodes = {n["id"]: n for n in subgraph["nodes"]}
    edges = subgraph["edges"]

    # Build adjacency
    outgoing: dict[str, list[dict]] = defaultdict(list)
    incoming: dict[str, list[dict]] = defaultdict(list)
    for e in edges:
        outgoing[e["source"]].append(e)
        incoming[e["target"]].append(e)

    # Sort outgoing edges by weight desc
    for src in outgoing:
        outgoing[src].sort(key=lambda x: -x["weight"])

    # Find entry nodes (no incoming edges in subgraph, or highest frequency)
    entry_nodes = [n["id"] for n in subgraph["nodes"]
                   if n["id"] not in incoming or not incoming[n["id"]]]
    if not entry_nodes:
        entry_nodes = [n["id"] for n in sorted(subgraph["nodes"],
                       key=lambda x: -x["frequency"])[:3]]

    pathways = []
    for entry in entry_nodes[:top_k]:
        path = _greedy_walk(entry, outgoing, nodes, max_steps=20)
        if len(path) > 1:
            total_weight = sum(s["weight"] for s in path if "weight" in s)
            pathways.append({
                "entry": entry,
                "steps": path,
                "length": len(path),
                "total_weight": total_weight,
            })

    # Sort by total weight desc
    pathways.sort(key=lambda x: -x["total_weight"])
    return pathways


def _greedy_walk(
    start: str,
    outgoing: dict[str, list[dict]],
    nodes: dict,
    max_steps: int = 20,
) -> list[dict]:
    """Greedy walk: always take the highest-weight outgoing edge."""
    path: list[dict] = []
    current = start
    visited: set[str] = set()

    for _ in range(max_steps):
        if current in visited:
            break
        visited.add(current)

        if current in nodes:
            path.append({"node": current, "label": nodes[current]["label"]})

        next_edges = [e for e in outgoing.get(current, [])
                      if e["target"] not in visited]
        if not next_edges:
            break

        best = next_edges[0]  # already sorted by weight desc
        path[-1]["next"] = best["target"]
        path[-1]["edge_weight"] = best["weight"]
        current = best["target"]

    return path


def find_branch_points(subgraph: dict) -> list[dict]:
    """Find nodes with multiple outgoing edges (decision points)."""
    outgoing: dict[str, list[dict]] = defaultdict(list)
    for e in subgraph["edges"]:
        outgoing[e["source"]].append(e)

    branches = []
    for node_id, out_edges in outgoing.items():
        if len(out_edges) >= 2:
            # Get node label
            node_label = ""
            for n in subgraph["nodes"]:
                if n["id"] == node_id:
                    node_label = n["label"]
                    break

            branches.append({
                "node": node_id,
                "label": node_label,
                "num_branches": len(out_edges),
                "branches": [
                    {"target": e["target"], "weight": e["weight"],
                     "label": _short_label(e["target"])}
                    for e in sorted(out_edges, key=lambda x: -x["weight"])
                ],
            })

    return sorted(branches, key=lambda x: -x["num_branches"])


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Per-Subflow Subgraph Mining")
    parser.add_argument("--skills", default=None,
                        help="per_subflow_vertex_subsets.json (if pre-computed)")
    parser.add_argument("--split", default="train",
                        choices=["train", "dev", "test"])
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--max-vertices", type=int, default=30)
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--min-edge-weight", type=int, default=1)
    parser.add_argument("--max-intents", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading ABCD {args.split}...")
    convs = load_abcd_data(args.split)
    if args.max_sessions:
        convs = convs[:args.max_sessions]
    print(f"  {len(convs)} conversations")

    # Group by subflow
    groups = group_by_subflow(convs)
    print(f"  {len(groups)} subflows")

    # Get vertex sets (mine or load)
    if args.skills:
        print(f"Loading pre-computed vertex sets from {args.skills}...")
        skills_data = json.loads(Path(args.skills).read_text(encoding="utf-8"))
        vertex_sets = skills_data.get("per_subflow", skills_data.get("intent_skills", skills_data))
    else:
        print(f"Mining per-subflow vertex cover...")
        vertex_sets = {}
        for sf, sf_convs in sorted(groups.items()):
            if len(sf_convs) < args.min_sessions:
                continue
            op_results = abcd_to_operator_results(sf_convs)
            if not op_results:
                continue
            hg = SessionHypergraph.from_operator_results(op_results)
            selected, _ = greedy_vertex_cover(hg, rho=args.rho, max_vertices=args.max_vertices)
            coverage = sum(1 for e in hg.hyperedges
                           if len(selected & e.vertices) >= math.ceil(args.rho * e.size))
            vertex_sets[sf] = {
                "selected_vertices": sorted(selected),
                "coverage_pct": round(100 * coverage / max(len(hg.hyperedges), 1), 1),
            }
            print(f"  {sf}: {len(selected)} vertices, {vertex_sets[sf]['coverage_pct']}% coverage")

    # Build DAG + subgraph per subflow
    all_subgraphs: dict[str, dict] = {}
    intents = list(vertex_sets.items())
    if args.max_intents:
        intents = intents[:args.max_intents]

    print(f"\nBuilding subgraphs for {len(intents)} subflows...")
    for sf, vset in intents:
        if len(groups.get(sf, [])) < args.min_sessions:
            continue

        # Convert to operators + build DAG
        op_results = abcd_to_operator_results(groups[sf])
        dag = build_dag_from_operator_results(op_results)

        # Extract subgraph
        selected = set(vset.get("selected_vertices", []))
        sub = extract_subgraph(dag, selected, min_edge_weight=args.min_edge_weight)

        # Pathway + branches
        pathways = find_main_pathway(sub)
        branches = find_branch_points(sub)

        all_subgraphs[sf] = {
            **sub,
            "pathways": pathways,
            "branch_points": branches,
            "n_sessions": len(groups[sf]),
            "coverage_pct": vset.get("coverage_pct", 0),
        }

        print(f"  {sf}: {sub['n_selected_nodes']} nodes, {sub['n_selected_edges']} edges, "
              f"{len(pathways)} pathways, {len(branches)} branch points")

    # Save
    out_path = out_dir / "per_subflow_subgraphs.json"
    out_path.write_text(json.dumps(all_subgraphs, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    total_nodes = sum(s["n_selected_nodes"] for s in all_subgraphs.values())
    total_edges = sum(s["n_selected_edges"] for s in all_subgraphs.values())
    print(f"\nDone. {len(all_subgraphs)} subgraphs → {out_path}")
    print(f"  Total nodes: {total_nodes}, Total edges: {total_edges}")

    # Print one example
    if all_subgraphs:
        first = list(all_subgraphs.keys())[0]
        sg = all_subgraphs[first]
        print(f"\nExample: {first}")
        print(f"  Nodes ({sg['n_selected_nodes']}):")
        for n in sg["nodes"][:8]:
            print(f"    {n['id']} (freq={n['frequency']})")
        print(f"  Edges ({sg['n_selected_edges']}):")
        for e in sg["edges"][:8]:
            print(f"    {e['source']} → {e['target']} (w={e['weight']})")
        if sg["pathways"]:
            p = sg["pathways"][0]
            print(f"  Main pathway ({p['length']} steps, weight={p['total_weight']}):")
            for step in p["steps"]:
                nxt = f" → {step.get('next', 'END')}" if "next" in step else ""
                print(f"    {step['node']}{nxt}")
        if sg["branch_points"]:
            print(f"  Branch points ({len(sg['branch_points'])}):")
            for bp in sg["branch_points"][:3]:
                targets = ", ".join(f"{b['target']}(w={b['weight']})" for b in bp["branches"])
                print(f"    {bp['node']}: {targets}")


if __name__ == "__main__":
    main()
