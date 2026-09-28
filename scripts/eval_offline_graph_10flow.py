#!/usr/bin/env python3
"""Fast graph-only benchmark for the current 10-flow ABCD split.

The benchmark mines each flow from train.json and replays held-out test action
sequences against the resulting graph.  It never compiles a skill, calls an
LLM, or runs an agent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval_tod.abcd.action_schema import canonical_action_name, load_action_schema
from skill_mining.backbone_workflow_mining import (
    ROOT,
    _mine_backbone_workflow_support_lift,
    mine_backbone_workflow,
    mine_backbone_workflow_heuristics,
    mine_backbone_workflow_motif_trace_cover,
    mine_backbone_workflow_observable_trace_cover,
    mine_backbone_workflow_structural_trace_cover,
    mine_backbone_workflow_trace_cover,
    observable_transition_events,
    route_observable_tree,
    route_motif_state,
    route_structural_tree,
)


DEFAULT_SPLITS = ROOT_DIR / "data" / "eval" / "abcd" / "splits"


def action_sequence(flow: str, conversation: dict[str, Any]) -> list[str]:
    schema = load_action_schema()
    result: list[str] = []
    for turn in conversation.get("delexed") or []:
        targets = turn.get("targets") or []
        if len(targets) < 3 or targets[1] != "take_action" or not targets[2]:
            continue
        action, _ = canonical_action_name(targets[2], schema.get("actions"))
        if action:
            result.append(f"{flow}:{action}")
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_graph(
    flow: str,
    subgraph: dict[str, Any],
    test_conversations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one mined graph by teacher-forced held-out trace replay."""
    full_pairs = {
        (str(edge["source"]), str(edge["target"]))
        for edge in subgraph.get("edges", [])
    }
    backbone_pairs = {
        (str(edge["source"]), str(edge["target"]))
        for edge in subgraph.get("backbone", {}).get("edges", [])
        if str(edge.get("source")) != ROOT
    }
    ranked_targets: dict[str, list[str]] = {}
    for source, edges in subgraph.get("local_transitions", {}).items():
        ordered = sorted(
            edges,
            key=lambda edge: (
                int(edge.get("priority", 10**9)),
                -float(edge.get("score", 0.0)),
                -int(edge.get("support", 0)),
                str(edge.get("target", "")),
            ),
        )
        ranked_targets[str(source)] = list(dict.fromkeys(str(edge["target"]) for edge in ordered))
    retained_pairs = {
        (source, target)
        for source, targets in ranked_targets.items()
        for target in targets
    }
    retained_pairs_by_source: dict[str, set[str]] = defaultdict(set)
    for source, target in retained_pairs:
        retained_pairs_by_source[source].add(target)
    full_targets: dict[str, set[str]] = defaultdict(set)
    for source, target in full_pairs:
        full_targets[source].add(target)
    branching_sources = {source for source, targets in full_targets.items() if len(targets) >= 2}
    observable_router = subgraph.get("observable_router") or {}
    router_sources = observable_router.get("sources") or {}
    motif_graph = subgraph.get("motif_refinement") or {}
    structural_graph = subgraph.get("structural_refinement") or motif_graph
    is_motif_refinement = bool(motif_graph)
    structural_routers = structural_graph.get("routers") or {}
    structural_modes = {
        str(mode.get("mode_id")): mode for mode in structural_graph.get("nodes") or []
    }
    refined_pairs = {
        (str(edge.get("source")), str(edge.get("target")))
        for edge in structural_graph.get("edges") or []
    }
    history_order = int(structural_graph.get("history_order", 3))

    counts: dict[str, float] = defaultdict(float)
    route_coverages: list[float] = []
    full_route_coverages: list[float] = []
    refined_route_coverages: list[float] = []
    test_pair_types: set[tuple[str, str]] = set()

    for conversation in test_conversations:
        sequence = action_sequence(flow, conversation)
        pairs = list(zip(sequence, sequence[1:]))
        observed_events = observable_transition_events(flow, [conversation]) if router_sources else []
        if not pairs:
            continue
        counts["route_dialogues"] += 1
        route_hits = 0
        full_route_hits = 0
        refined_route_hits = 0
        for pair_index, (source, target) in enumerate(pairs):
            pair = (source, target)
            test_pair_types.add(pair)
            counts["transitions"] += 1
            is_full = pair in full_pairs
            is_backbone = pair in backbone_pairs
            is_retained = pair in retained_pairs
            counts["full_hits"] += float(is_full)
            counts["backbone_hits"] += float(is_backbone)
            counts["retained_hits"] += float(is_retained)
            full_route_hits += int(is_full)
            route_hits += int(is_retained)

            candidates = ranked_targets.get(source, [])
            counts["candidate_size_sum"] += len(candidates)
            counts["routing_pair_complexity_sum"] += len(candidates) * max(len(candidates) - 1, 0) / 2
            rank = candidates.index(target) + 1 if target in candidates else 0
            counts["mrr_sum"] += 1.0 / rank if rank else 0.0
            counts["top1_hits"] += float(rank == 1)

            structural_candidates = list(candidates)
            source_state = ""
            target_state = ""
            source_structural = structural_routers.get(source)
            target_structural = structural_routers.get(target)
            if source_structural:
                source_leaf = (
                    route_motif_state(
                        source_structural, sequence[:pair_index + 1], history_order,
                    )
                    if is_motif_refinement else
                    route_structural_tree(
                        source_structural.get("tree") or {}, sequence[:pair_index + 1], history_order,
                    )
                )
                source_state = str(source_leaf.get("mode_id") or "")
                source_mode = structural_modes.get(source_state, {})
                proposed = [
                    str(value) for value in source_mode.get("candidate_actions") or []
                    if str(value) in retained_pairs_by_source.get(source, set())
                ]
                if proposed:
                    structural_candidates = proposed
                if is_motif_refinement and len(source_structural.get("modes") or []) > 1:
                    counts["motif_split_assignments"] += 1
            if target_structural:
                target_leaf = (
                    route_motif_state(
                        target_structural, sequence[:pair_index + 2], history_order,
                    )
                    if is_motif_refinement else
                    route_structural_tree(
                        target_structural.get("tree") or {}, sequence[:pair_index + 2], history_order,
                    )
                )
                target_state = str(target_leaf.get("mode_id") or "")
            refined_hit = bool(source_state and target_state and (source_state, target_state) in refined_pairs)
            counts["refined_state_edge_hits"] += float(refined_hit)
            refined_route_hits += int(refined_hit)

            if source in branching_sources:
                counts["branch_transitions"] += 1
                counts["branch_candidate_size_sum"] += len(candidates)
                counts["branch_hits"] += float(is_retained)
                counts["branch_mrr_sum"] += 1.0 / rank if rank else 0.0
                counts["branch_top1_hits"] += float(rank == 1)

                guarded_candidates = list(candidates)
                source_router = router_sources.get(source)
                if source_router and pair_index < len(observed_events):
                    event_features = set(observed_events[pair_index].get("features") or [])
                    leaf = route_observable_tree(source_router.get("tree") or {}, event_features)
                    mode_by_id = {
                        str(mode.get("mode_id")): mode
                        for mode in source_router.get("modes") or []
                    }
                    mode = mode_by_id.get(str(leaf.get("mode_id")), {})
                    proposed = [
                        str(value) for value in mode.get("candidate_targets") or []
                        if str(value) in retained_pairs_by_source.get(source, set())
                    ]
                    if proposed:
                        guarded_candidates = proposed
                        counts["routing_mode_assignments"] += 1
                guard_rank = (
                    guarded_candidates.index(target) + 1
                    if target in guarded_candidates else 0
                )
                counts["guard_candidate_size_sum"] += len(guarded_candidates)
                counts["guard_hits"] += float(guard_rank > 0)
                counts["guard_top1_hits"] += float(guard_rank == 1)
                counts["guard_mrr_sum"] += 1.0 / guard_rank if guard_rank else 0.0

                structural_rank = (
                    structural_candidates.index(target) + 1
                    if target in structural_candidates else 0
                )
                counts["structural_candidate_size_sum"] += len(structural_candidates)
                counts["structural_hits"] += float(structural_rank > 0)
                counts["structural_top1_hits"] += float(structural_rank == 1)
                counts["structural_mrr_sum"] += 1.0 / structural_rank if structural_rank else 0.0
                if source_state:
                    counts["structural_state_assignments"] += 1
                    mode = structural_modes.get(source_state, {})
                    if is_motif_refinement:
                        counts["seen_history_assignments"] += float(source_leaf.get("exact"))
                    else:
                        signature = "|".join(
                            value.split(":", 1)[-1]
                            for value in sequence[max(0, pair_index + 1 - history_order):pair_index + 1]
                        )
                        counts["seen_history_assignments"] += float(
                            signature in set(mode.get("observed_history_signatures") or [])
                        )

        route_coverages.append(route_hits / len(pairs))
        full_route_coverages.append(full_route_hits / len(pairs))
        refined_route_coverages.append(refined_route_hits / len(pairs))

    transitions = counts["transitions"]
    branch_transitions = counts["branch_transitions"]
    route_dialogues = counts["route_dialogues"]
    retained_type_hits = len(test_pair_types & retained_pairs)
    full_type_hits = len(test_pair_types & full_pairs)
    active_sources = sum(bool(targets) for targets in ranked_targets.values())
    metrics = {
        "test_dialogues": len(test_conversations),
        "route_dialogues": int(route_dialogues),
        "heldout_transitions": int(transitions),
        "heldout_transition_types": len(test_pair_types),
        "train_graph_nodes": len(subgraph.get("nodes", [])),
        "train_full_edges": len(full_pairs),
        "backbone_edges": len(backbone_pairs),
        "retained_edges": len(retained_pairs),
        "residual_edges": len(subgraph.get("residual_edges", [])),
        "retained_edge_ratio": _safe_ratio(len(retained_pairs), len(full_pairs)),
        "edge_compression": 1.0 - _safe_ratio(len(retained_pairs), len(full_pairs)),
        "retained_edges_per_node": _safe_ratio(len(retained_pairs), len(subgraph.get("nodes", []))),
        "avg_retained_out_degree": _safe_ratio(len(retained_pairs), active_sources),
        "full_graph_transition_recall": _safe_ratio(counts["full_hits"], transitions),
        "backbone_transition_recall": _safe_ratio(counts["backbone_hits"], transitions),
        "retained_transition_recall": _safe_ratio(counts["retained_hits"], transitions),
        "full_graph_unique_transition_recall": _safe_ratio(full_type_hits, len(test_pair_types)),
        "retained_unique_transition_recall": _safe_ratio(retained_type_hits, len(test_pair_types)),
        "next_action_top1": _safe_ratio(counts["top1_hits"], transitions),
        "next_action_mrr": _safe_ratio(counts["mrr_sum"], transitions),
        "avg_candidate_size": _safe_ratio(counts["candidate_size_sum"], transitions),
        "routing_pair_complexity": _safe_ratio(counts["routing_pair_complexity_sum"], transitions),
        "branch_transitions": int(branch_transitions),
        "branch_candidate_recall": _safe_ratio(counts["branch_hits"], branch_transitions),
        "branch_next_action_top1": _safe_ratio(counts["branch_top1_hits"], branch_transitions),
        "branch_next_action_mrr": _safe_ratio(counts["branch_mrr_sum"], branch_transitions),
        "branch_avg_candidate_size": _safe_ratio(counts["branch_candidate_size_sum"], branch_transitions),
        "routing_mode_coverage": _safe_ratio(counts["routing_mode_assignments"], branch_transitions),
        "guard_candidate_recall": _safe_ratio(counts["guard_hits"], branch_transitions),
        "guard_next_action_top1": _safe_ratio(counts["guard_top1_hits"], branch_transitions),
        "guard_next_action_mrr": _safe_ratio(counts["guard_mrr_sum"], branch_transitions),
        "guard_avg_candidate_size": _safe_ratio(counts["guard_candidate_size_sum"], branch_transitions),
        "observable_pair_distinguishability": float(
            (subgraph.get("trace_cover") or {}).get("mean_selected_pair_distinguishability") or 0.0
        ),
        "routing_mode_nodes": int(observable_router.get("num_modes", 0)),
        "extra_routing_mode_nodes": int(observable_router.get("extra_mode_nodes", 0)),
        "structural_state_coverage": _safe_ratio(counts["structural_state_assignments"], branch_transitions),
        "structural_seen_history_coverage": _safe_ratio(counts["seen_history_assignments"], branch_transitions),
        "structural_candidate_recall": _safe_ratio(counts["structural_hits"], branch_transitions),
        "structural_next_action_top1": _safe_ratio(counts["structural_top1_hits"], branch_transitions),
        "structural_next_action_mrr": _safe_ratio(counts["structural_mrr_sum"], branch_transitions),
        "structural_avg_candidate_size": _safe_ratio(counts["structural_candidate_size_sum"], branch_transitions),
        "refined_state_edge_recall": _safe_ratio(counts["refined_state_edge_hits"], transitions),
        "coarse_routing_entropy": float(structural_graph.get("coarse_routing_entropy", 0.0)),
        "refined_routing_entropy": float(structural_graph.get("refined_routing_entropy", 0.0)),
        "structural_ambiguity_reduction": float(structural_graph.get("structural_ambiguity_reduction", 0.0)),
        "refined_graph_nodes": int(structural_graph.get("refined_nodes", 0)),
        "refined_graph_edges": int(structural_graph.get("refined_edges", 0)),
        "extra_structural_state_nodes": int(structural_graph.get("extra_state_nodes", 0)),
        "refined_structural_units": int(structural_graph.get("refined_structural_units", 0)),
        "num_structural_decisions": int(structural_graph.get("num_structural_decisions", 0)),
        "base_data_code_bits": float(structural_graph.get("base_data_code_bits", 0.0)),
        "refined_data_code_bits": float(structural_graph.get("refined_data_code_bits", 0.0)),
        "incremental_model_code_bits": float(structural_graph.get("incremental_model_code_bits", 0.0)),
        "mdl_net_gain_bits": float(structural_graph.get("mdl_net_gain_bits", 0.0)),
        "motif_state_nodes": int(motif_graph.get("refined_nodes", 0)),
        "extra_motif_state_nodes": int(motif_graph.get("extra_state_nodes", 0)),
        "motif_graph_edges": int(motif_graph.get("refined_edges", 0)),
        "motif_state_coverage": _safe_ratio(counts["structural_state_assignments"], branch_transitions)
        if is_motif_refinement else 0.0,
        "motif_seen_coverage": _safe_ratio(counts["seen_history_assignments"], branch_transitions)
        if is_motif_refinement else 0.0,
        "motif_candidate_recall": _safe_ratio(counts["structural_hits"], branch_transitions)
        if is_motif_refinement else 0.0,
        "motif_next_action_top1": _safe_ratio(counts["structural_top1_hits"], branch_transitions)
        if is_motif_refinement else 0.0,
        "motif_next_action_mrr": _safe_ratio(counts["structural_mrr_sum"], branch_transitions)
        if is_motif_refinement else 0.0,
        "motif_avg_candidate_size": _safe_ratio(counts["structural_candidate_size_sum"], branch_transitions)
        if is_motif_refinement else 0.0,
        "motif_split_coverage": _safe_ratio(counts["motif_split_assignments"], branch_transitions)
        if is_motif_refinement else 0.0,
        "motif_split_actions": int(motif_graph.get("num_split_actions", 0)),
        "motif_role_mdl_gain_bits": float(motif_graph.get("local_mdl_gain_bits", 0.0)),
        "mean_route_coverage": _safe_ratio(sum(route_coverages), route_dialogues),
        "route_coverage_at_80": _safe_ratio(sum(value >= 0.8 for value in route_coverages), route_dialogues),
        "complete_route_rate": _safe_ratio(sum(value == 1.0 for value in route_coverages), route_dialogues),
        "full_graph_complete_route_rate": _safe_ratio(
            sum(value == 1.0 for value in full_route_coverages), route_dialogues
        ),
        "refined_complete_route_rate": _safe_ratio(
            sum(value == 1.0 for value in refined_route_coverages), route_dialogues
        ),
    }
    return metrics


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate with explicit transition/dialogue/type denominators."""
    transition_weighted = [
        "full_graph_transition_recall", "backbone_transition_recall",
        "retained_transition_recall", "next_action_top1", "next_action_mrr",
        "avg_candidate_size",
        "routing_pair_complexity",
        "refined_state_edge_recall",
    ]
    branch_weighted = [
        "branch_candidate_recall", "branch_next_action_top1",
        "branch_next_action_mrr", "branch_avg_candidate_size",
        "routing_mode_coverage", "guard_candidate_recall",
        "guard_next_action_top1", "guard_next_action_mrr",
        "guard_avg_candidate_size", "observable_pair_distinguishability",
        "structural_state_coverage", "structural_seen_history_coverage",
        "structural_candidate_recall", "structural_next_action_top1",
        "structural_next_action_mrr", "structural_avg_candidate_size",
        "coarse_routing_entropy", "refined_routing_entropy",
        "structural_ambiguity_reduction",
        "motif_state_coverage", "motif_seen_coverage",
        "motif_candidate_recall", "motif_next_action_top1",
        "motif_next_action_mrr", "motif_avg_candidate_size",
        "motif_split_coverage",
    ]
    dialogue_weighted = [
        "mean_route_coverage", "route_coverage_at_80", "complete_route_rate",
        "full_graph_complete_route_rate",
        "refined_complete_route_rate",
    ]
    type_weighted = [
        "full_graph_unique_transition_recall", "retained_unique_transition_recall",
    ]
    result: dict[str, Any] = {
        "num_flows": len(rows),
        "test_dialogues": sum(int(row["test_dialogues"]) for row in rows),
        "route_dialogues": sum(int(row["route_dialogues"]) for row in rows),
        "heldout_transitions": sum(int(row["heldout_transitions"]) for row in rows),
        "branch_transitions": sum(int(row["branch_transitions"]) for row in rows),
        "heldout_transition_types_sum": sum(int(row["heldout_transition_types"]) for row in rows),
        "train_graph_nodes": sum(int(row["train_graph_nodes"]) for row in rows),
        "train_full_edges": sum(int(row["train_full_edges"]) for row in rows),
        "backbone_edges": sum(int(row["backbone_edges"]) for row in rows),
        "retained_edges": sum(int(row["retained_edges"]) for row in rows),
        "residual_edges": sum(int(row["residual_edges"]) for row in rows),
        "routing_mode_nodes": sum(int(row.get("routing_mode_nodes", 0)) for row in rows),
        "extra_routing_mode_nodes": sum(int(row.get("extra_routing_mode_nodes", 0)) for row in rows),
        "refined_graph_nodes": sum(int(row.get("refined_graph_nodes", 0)) for row in rows),
        "refined_graph_edges": sum(int(row.get("refined_graph_edges", 0)) for row in rows),
        "extra_structural_state_nodes": sum(int(row.get("extra_structural_state_nodes", 0)) for row in rows),
        "refined_structural_units": sum(int(row.get("refined_structural_units", 0)) for row in rows),
        "num_structural_decisions": sum(int(row.get("num_structural_decisions", 0)) for row in rows),
        "base_data_code_bits": sum(float(row.get("base_data_code_bits", 0.0)) for row in rows),
        "refined_data_code_bits": sum(float(row.get("refined_data_code_bits", 0.0)) for row in rows),
        "incremental_model_code_bits": sum(float(row.get("incremental_model_code_bits", 0.0)) for row in rows),
        "mdl_net_gain_bits": sum(float(row.get("mdl_net_gain_bits", 0.0)) for row in rows),
        "motif_state_nodes": sum(int(row.get("motif_state_nodes", 0)) for row in rows),
        "extra_motif_state_nodes": sum(int(row.get("extra_motif_state_nodes", 0)) for row in rows),
        "motif_graph_edges": sum(int(row.get("motif_graph_edges", 0)) for row in rows),
        "motif_split_actions": sum(int(row.get("motif_split_actions", 0)) for row in rows),
        "motif_role_mdl_gain_bits": sum(float(row.get("motif_role_mdl_gain_bits", 0.0)) for row in rows),
    }

    def weighted(metric: str, denominator: str) -> float:
        total = sum(float(row[denominator]) for row in rows)
        return _safe_ratio(
            sum(float(row.get(metric, 0.0)) * float(row[denominator]) for row in rows), total
        )

    for metric in transition_weighted:
        result[metric] = weighted(metric, "heldout_transitions")
    for metric in branch_weighted:
        result[metric] = weighted(metric, "branch_transitions")
    for metric in dialogue_weighted:
        result[metric] = weighted(metric, "route_dialogues")
    for metric in type_weighted:
        result[metric] = weighted(metric, "heldout_transition_types")
    if result["num_structural_decisions"]:
        result["coarse_routing_entropy"] = result["base_data_code_bits"] / result["num_structural_decisions"]
        result["refined_routing_entropy"] = result["refined_data_code_bits"] / result["num_structural_decisions"]
    result["structural_ambiguity_reduction"] = (
        1.0 - result["refined_routing_entropy"] / result["coarse_routing_entropy"]
        if result.get("coarse_routing_entropy", 0.0) > 0 else 0.0
    )
    result["retained_edge_ratio"] = _safe_ratio(result["retained_edges"], result["train_full_edges"])
    result["edge_compression"] = 1.0 - result["retained_edge_ratio"]
    result["retained_edges_per_node"] = _safe_ratio(result["retained_edges"], result["train_graph_nodes"])
    result["effective_structural_units"] = result["retained_edges"] + result["extra_routing_mode_nodes"]
    result["fitness_simplicity_hmean"] = _safe_ratio(
        2.0 * result["retained_transition_recall"] * result["edge_compression"],
        result["retained_transition_recall"] + result["edge_compression"],
    )
    result["avg_retained_out_degree"] = _safe_ratio(
        sum(float(row["avg_retained_out_degree"]) * max(int(row["train_graph_nodes"]), 1) for row in rows),
        sum(max(int(row["train_graph_nodes"]), 1) for row in rows),
    )
    return result


def _pct(value: Any) -> str:
    return f"{100 * float(value):.2f}%"


def render_markdown(method: str, rows: list[dict[str, Any]], overall: dict[str, Any]) -> str:
    lines = [
        "# Offline Graph Benchmark: ABCD 10-Flow",
        "",
        f"Method: `{method}`",
        "",
        "No skill compilation, LLM calls, or agent rollout is used. Graphs are mined on each flow's train split and replayed on its held-out test split.",
        "",
        "| Flow | Test transitions | Retained recall | MRR | Branch recall | Branch MRR | Mean route cov. | Complete routes | Retained/full edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['flow']} | {row['heldout_transitions']} | "
            f"{_pct(row['retained_transition_recall'])} | {row['next_action_mrr']:.4f} | "
            f"{_pct(row['branch_candidate_recall'])} | {row['branch_next_action_mrr']:.4f} | "
            f"{_pct(row['mean_route_coverage'])} | {_pct(row['complete_route_rate'])} | "
            f"{row['retained_edges']}/{row['train_full_edges']} |"
        )
    lines.extend([
        f"| **Weighted overall** | **{overall['heldout_transitions']}** | "
        f"**{_pct(overall['retained_transition_recall'])}** | **{overall['next_action_mrr']:.4f}** | "
        f"**{_pct(overall['branch_candidate_recall'])}** | **{overall['branch_next_action_mrr']:.4f}** | "
        f"**{_pct(overall['mean_route_coverage'])}** | **{_pct(overall['complete_route_rate'])}** | "
        f"**{overall['retained_edges']}/{overall['train_full_edges']}** |",
        "",
        "## Additional weighted metrics",
        "",
        f"- Full train-graph transition recall: {_pct(overall['full_graph_transition_recall'])}",
        f"- Backbone-only transition recall: {_pct(overall['backbone_transition_recall'])}",
        f"- Retained unique-transition recall: {_pct(overall['retained_unique_transition_recall'])}",
        f"- Next-action top-1 from graph ranking: {_pct(overall['next_action_top1'])}",
        f"- Average candidate size: {overall['avg_candidate_size']:.3f}",
        f"- Expected sibling-pair comparisons: {overall['routing_pair_complexity']:.3f}",
        f"- Edge compression (smaller retained graph is better): {_pct(overall['edge_compression'])}",
        f"- Retained edges per node: {overall['retained_edges_per_node']:.3f}",
        f"- Fitness-simplicity harmonic mean: {overall['fitness_simplicity_hmean']:.4f}",
        f"- Branch-only next-action top-1: {_pct(overall['branch_next_action_top1'])}",
        f"- Branch-only average candidate size: {overall['branch_avg_candidate_size']:.3f}",
        f"- Observable pair distinguishability: {overall['observable_pair_distinguishability']:.4f}",
        f"- Routing-mode coverage: {_pct(overall['routing_mode_coverage'])}",
        f"- Guard-conditioned candidate recall: {_pct(overall['guard_candidate_recall'])}",
        f"- Guard-conditioned next-action top-1: {_pct(overall['guard_next_action_top1'])}",
        f"- Guard-conditioned next-action MRR: {overall['guard_next_action_mrr']:.4f}",
        f"- Guard-conditioned average candidate size: {overall['guard_avg_candidate_size']:.3f}",
        f"- Learned routing modes / extra mode nodes: {overall['routing_mode_nodes']} / {overall['extra_routing_mode_nodes']}",
        f"- Effective structural units (edges + extra modes): {overall['effective_structural_units']}",
        f"- Structural state-assignment coverage: {_pct(overall['structural_state_coverage'])}",
        f"- Seen-history state coverage: {_pct(overall['structural_seen_history_coverage'])}",
        f"- Structural next-action recall: {_pct(overall['structural_candidate_recall'])}",
        f"- Structural next-action top-1: {_pct(overall['structural_next_action_top1'])}",
        f"- Structural next-action MRR: {overall['structural_next_action_mrr']:.4f}",
        f"- Structural average candidate size: {overall['structural_avg_candidate_size']:.3f}",
        f"- Coarse/refined routing entropy: {overall['coarse_routing_entropy']:.4f} / {overall['refined_routing_entropy']:.4f}",
        f"- Structural ambiguity reduction: {_pct(overall['structural_ambiguity_reduction'])}",
        f"- Refined graph nodes / edges / units: {overall['refined_graph_nodes']} / {overall['refined_graph_edges']} / {overall['refined_structural_units']}",
        f"- MDL data-code reduction: {overall['base_data_code_bits'] - overall['refined_data_code_bits']:.1f} bits",
        f"- Incremental model code: {overall['incremental_model_code_bits']:.1f} bits",
        f"- MDL net gain: {overall['mdl_net_gain_bits']:.1f} bits",
        f"- Refined state-edge recall: {_pct(overall['refined_state_edge_recall'])}",
        f"- Refined complete-route rate: {_pct(overall['refined_complete_route_rate'])}",
        f"- Motif state nodes / extra nodes / edges: {overall['motif_state_nodes']} / {overall['extra_motif_state_nodes']} / {overall['motif_graph_edges']}",
        f"- Seen causal-motif coverage: {_pct(overall['motif_seen_coverage'])}",
        f"- Motif-conditioned candidate recall: {_pct(overall['motif_candidate_recall'])}",
        f"- Motif-conditioned next-action top-1 / MRR: {_pct(overall['motif_next_action_top1'])} / {overall['motif_next_action_mrr']:.4f}",
        f"- Motif-conditioned average candidate size: {overall['motif_avg_candidate_size']:.3f}",
        f"- Split action nodes / split-transition coverage: {overall['motif_split_actions']} / {_pct(overall['motif_split_coverage'])}",
        f"- Local role-abstraction MDL gain: {overall['motif_role_mdl_gain_bits']:.1f} bits",
        f"- Route coverage >= 80%: {_pct(overall['route_coverage_at_80'])}",
        f"- Full train-graph complete-route upper bound: {_pct(overall['full_graph_complete_route_rate'])}",
        "",
        "Transition metrics are weighted by held-out transitions; branch metrics by held-out transitions whose train-graph source has at least two targets; route metrics by held-out dialogues containing at least one action transition.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--method",
        choices=["discriminative", "support_lift", "heuristics", "trace_cover", "observable_trace_cover", "structural_trace_cover", "motif_trace_cover"],
        default="discriminative",
    )
    parser.add_argument("--max-outgoing-edges", type=int, default=3)
    parser.add_argument("--min-branch-support", type=int, default=2)
    parser.add_argument("--discriminative-lambda", type=float, default=1.0)
    parser.add_argument("--discriminative-clip", type=float, default=3.0)
    parser.add_argument("--dependency-weight", type=float, default=1.0)
    parser.add_argument("--trace-coverage-target", type=float, default=0.8)
    parser.add_argument("--corpus-fitness-target", type=float, default=0.95)
    parser.add_argument("--routing-complexity-weight", type=float, default=2.0)
    parser.add_argument("--routing-mode-max-depth", type=int, default=2)
    parser.add_argument("--routing-mode-min-leaf-support", type=int, default=12)
    parser.add_argument("--routing-mode-min-information-gain", type=float, default=0.10)
    parser.add_argument("--structural-history-order", type=int, default=3)
    parser.add_argument("--structural-state-max-depth", type=int, default=3)
    parser.add_argument("--structural-state-min-leaf-support", type=int, default=8)
    parser.add_argument("--structural-state-node-penalty", type=float, default=1.0)
    parser.add_argument("--structural-state-edge-penalty", type=float, default=2.0)
    parser.add_argument("--motif-history-order", type=int, default=1)
    parser.add_argument("--motif-min-support", type=int, default=8)
    parser.add_argument("--motif-state-node-penalty", type=float, default=1.0)
    parser.add_argument("--motif-state-edge-penalty", type=float, default=2.0)
    parser.add_argument("--flows", default="", help="Optional comma-separated subset; default is INDEX.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = args.splits_dir / "INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    flows = [item.strip() for item in args.flows.split(",") if item.strip()] or sorted(index)
    if not args.flows and len(flows) != 10:
        raise RuntimeError(f"Expected the current 10-flow INDEX.json, found {len(flows)} flows")
    output_dir = args.output_dir or (
        ROOT_DIR / "outputs" / f"offline_graph_benchmark_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for flow in flows:
        flow_dir = args.splits_dir / flow
        train = json.loads((flow_dir / "train.json").read_text(encoding="utf-8"))
        test = json.loads((flow_dir / "test.json").read_text(encoding="utf-8"))
        if args.method == "support_lift":
            mined = _mine_backbone_workflow_support_lift(
                flow, train,
                max_outgoing_edges=args.max_outgoing_edges,
                min_branch_support=args.min_branch_support,
            )
        elif args.method == "heuristics":
            mined = mine_backbone_workflow_heuristics(
                flow, train,
                max_outgoing_edges=args.max_outgoing_edges,
                min_branch_support=args.min_branch_support,
                dependency_weight=args.dependency_weight,
            )
        elif args.method == "motif_trace_cover":
            mined = mine_backbone_workflow_motif_trace_cover(
                flow, train,
                min_branch_support=args.min_branch_support,
                trace_coverage_target=args.trace_coverage_target,
                corpus_fitness_target=args.corpus_fitness_target,
                routing_complexity_weight=args.routing_complexity_weight,
                motif_history_order=args.motif_history_order,
                motif_min_support=args.motif_min_support,
                motif_state_node_penalty=args.motif_state_node_penalty,
                motif_state_edge_penalty=args.motif_state_edge_penalty,
            )
        elif args.method == "structural_trace_cover":
            mined = mine_backbone_workflow_structural_trace_cover(
                flow, train,
                min_branch_support=args.min_branch_support,
                trace_coverage_target=args.trace_coverage_target,
                corpus_fitness_target=args.corpus_fitness_target,
                routing_complexity_weight=args.routing_complexity_weight,
                history_order=args.structural_history_order,
                state_max_depth=args.structural_state_max_depth,
                state_min_leaf_support=args.structural_state_min_leaf_support,
                state_node_penalty=args.structural_state_node_penalty,
                state_edge_penalty=args.structural_state_edge_penalty,
            )
        elif args.method == "observable_trace_cover":
            mined = mine_backbone_workflow_observable_trace_cover(
                flow, train,
                min_branch_support=args.min_branch_support,
                trace_coverage_target=args.trace_coverage_target,
                corpus_fitness_target=args.corpus_fitness_target,
                routing_complexity_weight=args.routing_complexity_weight,
                routing_mode_max_depth=args.routing_mode_max_depth,
                routing_mode_min_leaf_support=args.routing_mode_min_leaf_support,
                routing_mode_min_information_gain=args.routing_mode_min_information_gain,
            )
        elif args.method == "trace_cover":
            mined = mine_backbone_workflow_trace_cover(
                flow, train,
                min_branch_support=args.min_branch_support,
                trace_coverage_target=args.trace_coverage_target,
                corpus_fitness_target=args.corpus_fitness_target,
                routing_complexity_weight=args.routing_complexity_weight,
            )
        else:
            mined = mine_backbone_workflow(
                flow, train,
                max_outgoing_edges=args.max_outgoing_edges,
                min_branch_support=args.min_branch_support,
                discriminative_lambda=args.discriminative_lambda,
                discriminative_clip=args.discriminative_clip,
            )
        row = {"flow": flow, **evaluate_graph(flow, mined["subgraph"], test)}
        rows.append(row)
        print(
            f"{flow:24s} transitions={row['heldout_transitions']:4d} "
            f"retained={row['retained_transition_recall']:.4f} "
            f"branch_mrr={row['branch_next_action_mrr']:.4f} "
            f"guard_mrr={row['guard_next_action_mrr']:.4f}"
        )

    overall = aggregate(rows)
    payload = {
        "protocol": "abcd_10flow_offline_graph_replay_v1",
        "method": args.method,
        "config": {
            "max_outgoing_edges": args.max_outgoing_edges,
            "min_branch_support": args.min_branch_support,
            "discriminative_lambda": args.discriminative_lambda,
            "discriminative_clip": args.discriminative_clip,
            "dependency_weight": args.dependency_weight,
            "trace_coverage_target": args.trace_coverage_target,
            "corpus_fitness_target": args.corpus_fitness_target,
            "routing_complexity_weight": args.routing_complexity_weight,
            "routing_mode_max_depth": args.routing_mode_max_depth,
            "routing_mode_min_leaf_support": args.routing_mode_min_leaf_support,
            "routing_mode_min_information_gain": args.routing_mode_min_information_gain,
            "structural_history_order": args.structural_history_order,
            "structural_state_max_depth": args.structural_state_max_depth,
            "structural_state_min_leaf_support": args.structural_state_min_leaf_support,
            "structural_state_node_penalty": args.structural_state_node_penalty,
            "structural_state_edge_penalty": args.structural_state_edge_penalty,
            "motif_history_order": args.motif_history_order,
            "motif_min_support": args.motif_min_support,
            "motif_state_node_penalty": args.motif_state_node_penalty,
            "motif_state_edge_penalty": args.motif_state_edge_penalty,
        },
        "per_flow": rows,
        "weighted_overall": overall,
    }
    json_path = output_dir / "offline_graph_benchmark.json"
    md_path = output_dir / "offline_graph_benchmark.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(args.method, rows, overall), encoding="utf-8")
    print(f"\nJSON: {json_path}")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
