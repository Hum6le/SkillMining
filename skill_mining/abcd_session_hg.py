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
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SKILL_DIR) not in sys.path:
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


# ── Main pipeline ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ABCD → Session Hypergraph → Mined Subgraphs"
    )
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Limit number of ABCD conversations")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top-K vertex subsets (MILP or greedy)")
    parser.add_argument("--rho", type=float, default=0.8,
                        help="Coverage threshold: a hyperedge e is covered when |S∩e| >= ceil(rho*|e|)")
    parser.add_argument("--max-vertices", type=int, default=30,
                        help="Max vertices to select")
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

    # ── 3. Build session hypergraph ────────────────────────────
    print("Building session hypergraph...")
    hg = SessionHypergraph.from_operator_results(results)
    stats = hg.stats()
    print(f"  Hyperedges: {stats['num_hyperedges']}")
    print(f"  Vertices:   {stats['num_vertices']}")
    print(f"  Avg edge size: {stats['avg_hyperedge_size']:.1f}")
    print(f"  Top vertices by degree:")
    for v, deg in stats["top_vertices_by_degree"][:10]:
        print(f"    {v}: {deg}")

    with open(out_dir / "session_hg_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ── 4. Greedy vertex subset ────────────────────────────────
    print(f"\nGreedy vertex selection (rho={args.rho}, max_vertices={args.max_vertices})...")
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

    # ── 5. Summary ─────────────────────────────────────────────
    print(f"\nDone. Output: {out_dir}")
    print(f"  operator_results.json  — {len(results)} sessions")
    print(f"  session_hg_stats.json  — hypergraph statistics")
    print(f"  vertex_subset.json     — selected vertex set + coverage")


if __name__ == "__main__":
    main()
