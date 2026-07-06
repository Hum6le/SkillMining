#!/usr/bin/env python3
r"""ABCD → Session Hypergraph → Mined Subgraphs.

Converts ABCD action annotations directly to operator_results, builds a
session hypergraph, and solves the high-coverage vertex-subset MILP.

Usage::

    python skill_mining/abcd_session_hg.py --split test --top 3
    python skill_mining/abcd_session_hg.py --split train --max-sessions 500 --top 5

Output::

    skill_mining/output/abcd_session_hg/
        operator_results.json    # converted ABCD operator sequences
        session_hg_stats.json    # hypergraph statistics
        mined_subgraphs/         # PuLP MILP top-K solutions (JSON + PNG)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

# ── Path setup ────────────────────────────────────────────────
_SKILL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SKILL_DIR.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SKILL_DIR) in sys.path:
    sys.path.remove(str(_SKILL_DIR))
sys.path.insert(0, str(_SKILL_DIR))

_OUTPUT_DIR = _SKILL_DIR / "output" / "abcd_session_hg"

# ── ABCD → operator_results ───────────────────────────────────

def abcd_to_operator_results(
    conversations: list[dict],
    max_sessions: int | None = None,
) -> list[dict]:
    """Convert ABCD action turns to operator_results format.

    Each ABCD ``action`` turn yields one operator pair:
        role = subflow name
        operation = next_action name (e.g. ``pull-up-account``)

    Only action turns (``speaker == "action"``) are included; agent
    utterance turns and customer turns are skipped.
    """
    results: list[dict] = []
    for conv in conversations:
        if max_sessions and len(results) >= max_sessions:
            break

        scenario = conv.get("scenario", {})
        subflow = scenario.get("subflow", "unknown")
        convo_id = str(conv.get("convo_id", "?"))

        ordered_ops: list[list[str]] = []
        for turn in conv.get("delexed", []):
            targets = turn.get("targets", [])
            if len(targets) < 3:
                continue
            action_type = targets[1]
            if action_type != "take_action":
                continue
            next_action = targets[2]
            if not next_action:
                continue
            slot_values = targets[3] if len(targets) > 3 else []
            # Build a human-readable operation name
            op_name = str(next_action).strip()
            if slot_values and isinstance(slot_values, list):
                op_name += ":" + ",".join(str(v)[:20] for v in slot_values)
            ordered_ops.append([subflow, op_name])

        if not ordered_ops:
            continue

        results.append({
            "session_id": convo_id,
            "index": len(results),
            "ordered_operations": ordered_ops,
            "dialogue": " ".join(
                t.get("text", "") for t in conv.get("delexed", [])
            )[:500],
        })

    return results


# ── Session Hypergraph (from session2hg_v2) ────────────────────

def node_name(role: str, operation: str) -> str:
    return f"{role}:{operation}"


@dataclass(frozen=True)
class Hyperedge:
    record_index: int
    session_id: str
    vertices: FrozenSet[str]

    @property
    def size(self) -> int:
        return len(self.vertices)


class SessionHypergraph:
    def __init__(self, hyperedges: Sequence[Hyperedge], vertex_incidence: Dict[str, List[int]]):
        self.hyperedges = list(hyperedges)
        self.vertex_incidence = vertex_incidence
        self.vertices = frozenset(vertex_incidence.keys())

    @classmethod
    def from_operator_results(cls, results: list[dict]) -> "SessionHypergraph":
        hyperedges: list[Hyperedge] = []
        vertex_incidence: Dict[str, List[int]] = {}

        for idx, result in enumerate(results):
            ordered = result.get("ordered_operations") or []
            bag: set[str] = set()
            for pair in ordered:
                if not pair or len(pair) < 2:
                    continue
                role, operation = str(pair[0]), str(pair[1])
                n = node_name(role, operation)
                if n:
                    bag.add(n)
            if not bag:
                continue
            sid = str(result.get("session_id", "") or f"index_{idx}")
            e = Hyperedge(record_index=idx, session_id=sid, vertices=frozenset(bag))
            he_idx = len(hyperedges)
            hyperedges.append(e)
            for v in e.vertices:
                vertex_incidence.setdefault(v, []).append(he_idx)

        return cls(hyperedges=hyperedges, vertex_incidence=vertex_incidence)

    def stats(self) -> dict:
        sizes = [e.size for e in self.hyperedges]
        freq = defaultdict(int)
        for v, inc in self.vertex_incidence.items():
            freq[len(inc)] += 1
        return {
            "num_hyperedges": len(self.hyperedges),
            "num_vertices": len(self.vertices),
            "avg_hyperedge_size": sum(sizes) / max(len(sizes), 1),
            "max_hyperedge_size": max(sizes) if sizes else 0,
            "vertex_degree_distribution": dict(sorted(freq.items())),
            "top_vertices_by_degree": sorted(
                [(v, len(inc)) for v, inc in self.vertex_incidence.items()],
                key=lambda x: -x[1],
            )[:30],
        }


# ── Greedy high-coverage vertex set (fast alternative to MILP) ──

def greedy_vertex_cover(
    hg: SessionHypergraph,
    rho: float = 0.8,
    max_vertices: int | None = None,
) -> Tuple[Set[str], List[dict]]:
    """Greedy vertex selection maximizing hyperedge coverage.

    At each step, picks the vertex that covers the most *currently
    uncovered* hyperedges.  A hyperedge *e* is covered when
    |S ∩ e| ≥ ⌈ρ·|e|⌉.
    """
    uncovered = list(range(len(hg.hyperedges)))
    selected: Set[str] = set()
    history: list[dict] = []

    while uncovered:
        # Count how many uncovered edges each vertex covers
        best_v = None
        best_score = -1
        for v, inc in hg.vertex_incidence.items():
            if v in selected:
                continue
            uncovered_inc = [i for i in inc if i in uncovered]
            # Count edges where adding v would reach the coverage threshold
            score = 0
            for i in uncovered_inc:
                e = hg.hyperedges[i]
                current = len(selected & e.vertices)
                needed = math.ceil(rho * e.size)
                if current < needed and current + 1 >= needed:
                    score += 1  # this vertex "completes" coverage for e
                elif current < needed:
                    score += 0.5  # this vertex makes progress
            if score > best_score:
                best_score = score
                best_v = v

        if best_v is None or best_score <= 0:
            break

        selected.add(best_v)
        # Update uncovered
        new_uncovered = []
        for i in uncovered:
            e = hg.hyperedges[i]
            if len(selected & e.vertices) < math.ceil(rho * e.size):
                new_uncovered.append(i)
        uncovered = new_uncovered

        history.append({
            "vertex": best_v,
            "score": best_score,
            "covered_edges": len(hg.hyperedges) - len(uncovered),
            "total_edges": len(hg.hyperedges),
        })

        if max_vertices and len(selected) >= max_vertices:
            break

    return selected, history


# ── Per-Subflow / Built-in Intent ─────────────────────────────

def group_by_subflow(conversations: list[dict]) -> dict[str, list[dict]]:
    """用 ABCD 自带的 scenario.subflow 标签分组。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for conv in conversations:
        subflow = str(conv.get("scenario", {}).get("subflow", "unknown"))
        groups[subflow].append(conv)
    return dict(groups)


def mine_per_subflow(
    conversations: list[dict],
    rho: float = 0.8,
    max_vertices: int = 30,
    min_sessions: int = 2,
) -> dict:
    """Per-subflow hypergraph + vertex cover。

    Returns:
        {subflow_name: {selected_vertices, coverage_pct, num_sessions, ...}}
    """
    groups = group_by_subflow(conversations)
    results: dict[str, dict] = {}

    for subflow, convs in sorted(groups.items()):
        if len(convs) < min_sessions:
            continue

        op_results = abcd_to_operator_results(convs)
        if not op_results:
            continue

        hg = SessionHypergraph.from_operator_results(op_results)
        selected, _ = greedy_vertex_cover(hg, rho=rho, max_vertices=max_vertices)

        coverage = sum(
            1 for e in hg.hyperedges
            if len(selected & e.vertices) >= math.ceil(rho * e.size)
        )
        total_e = len(hg.hyperedges)

        results[subflow] = {
            "selected_vertices": sorted(selected),
            "num_selected": len(selected),
            "num_sessions": len(convs),
            "coverage": coverage,
            "total_hyperedges": total_e,
            "coverage_pct": round(100 * coverage / max(total_e, 1), 1),
        }

    return results


def compare_global_vs_per_subflow(
    conversations: list[dict],
    per_subflow_results: dict,
    rho: float = 0.8,
    max_vertices: int = 30,
) -> dict:
    """对比全局 vertex cover 和 per-subflow vertex cover。"""
    global_results = abcd_to_operator_results(conversations)
    global_hg = SessionHypergraph.from_operator_results(global_results)
    global_selected, _ = greedy_vertex_cover(global_hg, rho=rho, max_vertices=max_vertices)
    global_coverage = sum(
        1 for e in global_hg.hyperedges
        if len(global_selected & e.vertices) >= math.ceil(rho * e.size)
    )

    # Per-subflow 合并
    all_intent_vertices: set[str] = set()
    total_intent_coverage = 0
    total_intent_edges = 0
    for result in per_subflow_results.values():
        all_intent_vertices.update(result["selected_vertices"])
        total_intent_coverage += result["coverage"]
        total_intent_edges += result["total_hyperedges"]

    global_only = global_selected - all_intent_vertices
    intent_only = all_intent_vertices - global_selected
    shared = global_selected & all_intent_vertices

    # 桥接算子（出现在 ≥2 个 subflow 的 vertex set 中）
    op_subflows: dict[str, list[str]] = defaultdict(list)
    for sf, result in per_subflow_results.items():
        for op in result["selected_vertices"]:
            op_subflows[op].append(sf)
    bridges = sorted(
        [{"operator": op, "subflows": sfs, "num_subflows": len(sfs)}
         for op, sfs in op_subflows.items() if len(sfs) >= 2],
        key=lambda x: -x["num_subflows"],
    )

    return {
        "global": {
            "num_vertices": len(global_selected),
            "selected_vertices": sorted(global_selected),
            "coverage": global_coverage,
            "total_hyperedges": len(global_hg.hyperedges),
            "coverage_pct": round(100 * global_coverage / max(len(global_hg.hyperedges), 1), 1),
        },
        "per_subflow_merged": {
            "num_unique_vertices": len(all_intent_vertices),
            "total_coverage": total_intent_coverage,
            "total_hyperedges": total_intent_edges,
            "coverage_pct": round(100 * total_intent_coverage / max(total_intent_edges, 1), 1),
        },
        "set_analysis": {
            "global_only": sorted(global_only),
            "intent_only": sorted(intent_only),
            "shared": sorted(shared),
            "num_global_only": len(global_only),
            "num_intent_only": len(intent_only),
            "num_shared": len(shared),
        },
        "bridge_operators": bridges,
    }


# ── Main pipeline ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ABCD → Session Hypergraph → Mined Subgraphs"
    )
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Limit number of ABCD conversations")
    parser.add_argument("--rho", type=float, default=0.8,
                        help="Coverage threshold: a hyperedge e is covered when |S∩e| >= ceil(rho*|e|)")
    parser.add_argument("--max-vertices", type=int, default=30,
                        help="Max vertices to select (per subflow when --use-builtin-intent)")
    parser.add_argument("--min-sessions", type=int, default=2,
                        help="Min sessions per subflow (only applies with --use-builtin-intent)")
    parser.add_argument("--use-builtin-intent", action="store_true",
                        help="Use ABCD built-in subflow labels to group sessions by intent, "
                             "build per-subflow hypergraphs, and compare with global")
    parser.add_argument("--output-dir", default=None,
                        help="Custom output dir")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ABCD ──────────────────────────────────────────
    from eval_tod.abcd.data import load_abcd_data
    print(f"Loading ABCD {args.split} split...")
    convs = load_abcd_data(args.split)
    if args.max_sessions:
        convs = convs[:args.max_sessions]
    print(f"  {len(convs)} conversations")

    # ── 2. Convert to operator_results ─────────────────────────
    print("Converting to operator_results...")
    results = abcd_to_operator_results(convs)
    print(f"  {len(results)} sessions with operators")

    with open(out_dir / "operator_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 3. Global hypergraph ───────────────────────────────────
    print("Building global session hypergraph...")
    hg = SessionHypergraph.from_operator_results(results)
    stats = hg.stats()
    print(f"  Hyperedges: {stats['num_hyperedges']}")
    print(f"  Vertices:   {stats['num_vertices']}")
    print(f"  Avg edge size: {stats['avg_hyperedge_size']:.1f}")

    with open(out_dir / "session_hg_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ── 4. Global greedy vertex cover ──────────────────────────
    print(f"\nGlobal greedy vertex selection (rho={args.rho}, max_vertices={args.max_vertices})...")
    selected, history = greedy_vertex_cover(hg, rho=args.rho, max_vertices=args.max_vertices)
    coverage = sum(
        1 for e in hg.hyperedges
        if len(selected & e.vertices) >= math.ceil(args.rho * e.size)
    )
    print(f"  Selected {len(selected)} vertices")
    print(f"  Covers {coverage}/{len(hg.hyperedges)} hyperedges "
          f"({100 * coverage / max(len(hg.hyperedges), 1):.1f}%)")

    result = {
        "selected_vertices": sorted(selected),
        "num_selected": len(selected),
        "coverage": coverage,
        "total_hyperedges": len(hg.hyperedges),
        "coverage_pct": 100 * coverage / max(len(hg.hyperedges), 1),
        "selection_history": history,
    }
    with open(out_dir / "vertex_subset.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ── 5. Per-subflow (built-in intent) ───────────────────────
    if args.use_builtin_intent:
        print(f"\n{'=' * 50}")
        print(f"Per-subflow hypergraph + vertex cover")
        print(f"(rho={args.rho}, max_vertices={args.max_vertices}, "
              f"min_sessions={args.min_sessions})")
        print(f"{'=' * 50}")

        per_subflow = mine_per_subflow(
            convs, rho=args.rho, max_vertices=args.max_vertices,
            min_sessions=args.min_sessions,
        )
        print(f"\n  Valid subflows: {len(per_subflow)}")
        for sf, info in sorted(per_subflow.items(), key=lambda x: -x[1]["num_sessions"]):
            print(f"    {sf:35s}  {info['num_sessions']:3d} sessions  "
                  f"{info['num_selected']:3d} vertices  {info['coverage_pct']:5.1f}% coverage")

        with open(out_dir / "per_subflow_vertex_subsets.json", "w", encoding="utf-8") as f:
            json.dump(per_subflow, f, indent=2, ensure_ascii=False)

        # Comparison
        print(f"\nGlobal vs Per-Subflow comparison...")
        comparison = compare_global_vs_per_subflow(
            convs, per_subflow, rho=args.rho, max_vertices=args.max_vertices,
        )
        g = comparison["global"]
        ps = comparison["per_subflow_merged"]
        sa = comparison["set_analysis"]
        print(f"  Global:         {g['num_vertices']} vertices → {g['coverage_pct']:.1f}% coverage")
        print(f"  Per-subflow:    {ps['num_unique_vertices']} unique vertices → {ps['coverage_pct']:.1f}% coverage")
        print(f"  Set diff:       global-only={sa['num_global_only']}, "
              f"intent-only={sa['num_intent_only']}, shared={sa['num_shared']}")
        print(f"  Bridge ops:     {len(comparison['bridge_operators'])} (appear in ≥2 subflows)")

        with open(out_dir / "per_subflow_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

    # ── 6. Summary ─────────────────────────────────────────────
    print(f"\nDone. Output: {out_dir}")
    print(f"  operator_results.json          — {len(results)} sessions")
    print(f"  session_hg_stats.json          — global hypergraph statistics")
    print(f"  vertex_subset.json             — global vertex set + coverage")
    if args.use_builtin_intent:
        print(f"  per_subflow_vertex_subsets.json — per-subflow vertex sets")
        print(f"  per_subflow_comparison.json    — global vs per-subflow comparison")


if __name__ == "__main__":
    main()
