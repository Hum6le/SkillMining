#!/usr/bin/env python3
"""Hierarchical residual hypergraph mining for skill induction.

This module implements the structural part of the proposed pipeline:

    ordered traces -> behavior basis -> backbone -> residual branches
                    -> low-support reference motifs

The miner is deliberately deterministic. An agent-generated candidate graph can
be supplied as evidence, but the selection of the backbone and branches is
performed from trace support and ordered transitions. The output includes the
representative traces needed by a later semantic-reasoning stage.

Example:
    python skill_mining/hierarchical_skill_mining.py \
        --operator-results skill_mining/output/abcd_session_hg/operator_results.json \
        --output skill_mining/output/abcd_hierarchical_skill.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Session:
    session_id: str
    sequence: tuple[str, ...]
    raw_sequence: tuple[str, ...]
    record: dict[str, Any]


def canonical_action(value: str) -> str:
    """Remove instance arguments while retaining the operation identity."""
    value = re.sub(r"\s+", " ", str(value or "").strip())
    if ":" in value:
        value = value.split(":", 1)[0]
    return value


def canonical_node(pair: Sequence[Any]) -> str:
    if len(pair) < 2:
        return ""
    role = re.sub(r"\s+", " ", str(pair[0]).strip())
    action = canonical_action(str(pair[1]))
    return f"{role}:{action}" if role and action else ""


def load_sessions(path: Path) -> list[Session]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sessions: list[Session] = []
    for index, record in enumerate(data):
        ordered = record.get("ordered_operations") or []
        raw = tuple(canonical_node(pair) for pair in ordered)
        raw = tuple(node for node in raw if node)
        if not raw:
            continue
        collapsed = [raw[0]]
        for node in raw[1:]:
            if node != collapsed[-1]:
                collapsed.append(node)
        sessions.append(Session(
            session_id=str(record.get("session_id") or f"session_{index}"),
            sequence=tuple(collapsed),
            raw_sequence=raw,
            record=record,
        ))
    return sessions


def transition_counts(sessions: Iterable[Session]) -> tuple[Counter, Counter]:
    edge_weight: Counter[tuple[str, str]] = Counter()
    edge_sessions: Counter[tuple[str, str]] = Counter()
    for session in sessions:
        seen: set[tuple[str, str]] = set()
        for source, target in zip(session.sequence, session.sequence[1:]):
            edge_weight[(source, target)] += 1
            seen.add((source, target))
        for edge in seen:
            edge_sessions[edge] += 1
    return edge_weight, edge_sessions


def coverage(session: Session, selected: set[str], rho: float) -> float:
    if not session.sequence:
        return 0.0
    covered = len(set(session.sequence) & selected)
    return covered / len(set(session.sequence))


def greedy_behavior_basis(
    sessions: list[Session],
    rho: float,
    max_nodes: int | None,
) -> tuple[list[str], dict[str, Any]]:
    """Select a compact high-coverage vertex basis from session hyperedges."""
    candidates = sorted({node for s in sessions for node in s.sequence})
    selected: set[str] = set()
    uncovered = set(range(len(sessions)))
    target = {i for i, s in enumerate(sessions) if coverage(s, selected, rho) < rho}
    history: list[dict[str, Any]] = []

    while target and candidates and (max_nodes is None or len(selected) < max_nodes):
        best = None
        best_score = (-1.0, -1.0, "")
        for node in candidates:
            if node in selected:
                continue
            gain = 0.0
            session_gain = 0
            for index in target:
                session = sessions[index]
                before = coverage(session, selected, rho)
                after = coverage(session, selected | {node}, rho)
                if after > before:
                    gain += after - before
                    session_gain += 1
            # Prefer nodes that cover many partially explained sessions, then
            # nodes with broader cross-session support.
            support = sum(node in s.sequence for s in sessions)
            score = (gain, session_gain + support / max(len(sessions), 1), node)
            if score > best_score:
                best_score = score
                best = node
        if best is None or best_score[0] <= 0:
            break
        selected.add(best)
        newly_covered = {
            i for i in target if coverage(sessions[i], selected, rho) >= rho
        }
        uncovered -= newly_covered
        target -= newly_covered
        history.append({
            "node": best,
            "marginal_gain": round(best_score[0], 6),
            "newly_covered_sessions": sorted(newly_covered),
            "remaining_sessions": sorted(target),
        })

    covered = [i for i, s in enumerate(sessions) if coverage(s, selected, rho) >= rho]
    return sorted(selected), {
        "coverage_ratio": round(len(covered) / max(len(sessions), 1), 6),
        "covered_session_indices": covered,
        "selection_history": history,
    }


def project_sequence(sequence: Sequence[str], selected: set[str]) -> list[str]:
    projected: list[str] = []
    for node in sequence:
        if node in selected and (not projected or projected[-1] != node):
            projected.append(node)
    return projected


def choose_backbone_path(selected: set[str], sessions: list[Session]) -> list[str]:
    """Find a high-support, cycle-free path through the selected behavior basis."""
    edge_weight: Counter[tuple[str, str]] = Counter()
    for session in sessions:
        projected = project_sequence(session.sequence, selected)
        for edge in zip(projected, projected[1:]):
            edge_weight[edge] += 1

    outgoing: dict[str, list[tuple[str, int]]] = defaultdict(list)
    incoming: Counter[str] = Counter()
    for (source, target), weight in edge_weight.items():
        outgoing[source].append((target, weight))
        incoming[target] += weight
    for source in outgoing:
        outgoing[source].sort(key=lambda item: (-item[1], item[0]))

    starts = sorted(selected, key=lambda n: (-sum(
        1 for s in sessions if s.sequence and s.sequence[0] == n
    ), incoming[n], n))
    if not starts:
        return []
    best_path: list[str] = []
    best_score = -1
    for start in starts[: max(1, min(10, len(starts)))]:
        path = [start]
        score = 0
        current = start
        while outgoing.get(current):
            choices = [(node, weight) for node, weight in outgoing[current] if node not in path]
            if not choices:
                break
            target, weight = choices[0]
            path.append(target)
            score += weight
            current = target
        if (score, len(path), tuple(path)) > (best_score, len(best_path), tuple(best_path)):
            best_path = path
            best_score = score
    return best_path


def residual_segments(session: Session, backbone: Sequence[str]) -> list[dict[str, Any]]:
    """Extract off-backbone segments and their attachment points."""
    if not backbone:
        return []
    positions = {node: index for index, node in enumerate(backbone)}
    occurrences = [
        (index, positions[node], node)
        for index, node in enumerate(session.sequence)
        if node in positions
    ]
    segments: list[dict[str, Any]] = []
    for left, right in zip(occurrences, occurrences[1:]):
        left_index, left_order, left_node = left
        right_index, right_order, right_node = right
        off_path = list(session.sequence[left_index + 1:right_index])
        if off_path:
            segments.append({
                "attachment": [left_node, right_node],
                "path": off_path,
                "session_id": session.session_id,
                "record": session.record,
                "start_step": left_index + 1,
                "end_step": right_index - 1,
                "backbone_order": [left_order, right_order],
            })

    # Capture prefix/suffix deviations when a trace starts or ends outside the
    # selected backbone.
    if occurrences and occurrences[0][0] > 0:
        first = occurrences[0]
        segments.insert(0, {
            "attachment": ["START", first[2]],
            "path": list(session.sequence[:first[0]]),
            "session_id": session.session_id,
            "record": session.record,
            "start_step": 0,
            "end_step": first[0] - 1,
            "backbone_order": [-1, first[1]],
        })
    if occurrences and occurrences[-1][0] < len(session.sequence) - 1:
        last = occurrences[-1]
        segments.append({
            "attachment": [last[2], "END"],
            "path": list(session.sequence[last[0] + 1:]),
            "session_id": session.session_id,
            "record": session.record,
            "start_step": last[0] + 1,
            "end_step": len(session.sequence) - 1,
            "backbone_order": [last[1], len(backbone)],
        })
    return segments


def mine_residual_layers(
    sessions: list[Session],
    backbone: list[str],
    min_branch_support: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        for segment in residual_segments(session, backbone):
            grouped[tuple(segment["attachment"])].append(segment)

    branches: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for attachment, segments in sorted(grouped.items()):
        path_counts = Counter(tuple(segment["path"]) for segment in segments)
        for path, count in sorted(path_counts.items(), key=lambda item: (-item[1], item[0])):
            item = {
                "attachment": list(attachment),
                "path": list(path),
                "support": count,
                "support_sessions": [
                    segment["session_id"] for segment in segments
                    if tuple(segment["path"]) == path
                ],
                "representative_traces": [
                    {
                        "session_id": segment["session_id"],
                        "dialogue": segment["record"].get("dialogue", ""),
                        "local_path": segment["path"],
                        "start_step": segment["start_step"],
                        "end_step": segment["end_step"],
                    }
                    for segment in segments
                    if tuple(segment["path"]) == path
                ][:3],
                "semantic_status": "evidence_pending",
            }
            if count >= min_branch_support:
                item["layer"] = "branch"
                branches.append(item)
            else:
                item["layer"] = "reference"
                references.append(item)
    return branches, references


def build_agent_evidence(
    sessions: list[Session],
    backbone: list[str],
    branches: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Package motifs and traces for a later semantic reasoning agent."""
    return {
        "instruction": (
            "Infer semantic names, preconditions, effects and branch guards "
            "only from the structural motif and its representative traces. "
            "Return an abstention when evidence is insufficient."
        ),
        "backbone_motif": {
            "path": backbone,
            "supporting_sessions": [
                s.session_id for s in sessions if len(project_sequence(s.sequence, set(backbone))) >= 2
            ][:20],
        },
        "branch_motifs": branches,
        "reference_motifs": references,
    }


def render_reasoning_prompt(evidence: dict[str, Any]) -> str:
    """Render a grounded prompt for a semantic-reasoning agent."""
    return """# Semantic Motif Interpretation\n\nYou are given a mined workflow motif. Infer only semantic properties supported by the structural evidence and representative traces. Do not invent business rules. If a property is not identifiable, return `unknown`.\n\nReturn JSON with this schema:\n```json\n{\n  \"semantic_name\": \"...\",\n  \"goal\": \"...\",\n  \"parameters\": [],\n  \"preconditions\": [],\n  \"postconditions\": [],\n  \"branches\": [\n    {\"guard\": \"...\", \"meaning\": \"...\", \"evidence_session_ids\": []}\n  ],\n  \"confidence\": 0.0,\n  \"unknowns\": []\n}\n```\n\n## Mined evidence\n\n```json\n""" + json.dumps(evidence, indent=2, ensure_ascii=False) + "\n```\n"


def mine_sessions(
    sessions: list[Session],
    input_path: str,
    rho: float,
    max_basis_nodes: int | None,
    min_branch_support: int,
) -> dict[str, Any]:
    basis, basis_stats = greedy_behavior_basis(sessions, rho, max_basis_nodes)
    backbone = choose_backbone_path(set(basis), sessions)
    branches, references = mine_residual_layers(sessions, backbone, min_branch_support)
    edge_weight, edge_sessions = transition_counts(sessions)

    result = {
        "algorithm": "hierarchical_residual_hypergraph_cover",
        "input": input_path,
        "config": {
            "coverage_ratio": rho,
            "max_basis_nodes": max_basis_nodes,
            "min_branch_support": min_branch_support,
        },
        "statistics": {
            "num_sessions": len(sessions),
            "num_unique_actions": len({node for s in sessions for node in s.sequence}),
            "num_observed_transitions": len(edge_weight),
            "num_backbone_nodes": len(backbone),
            "num_branches": len(branches),
            "num_references": len(references),
        },
        "behavior_basis": {
            "nodes": basis,
            **basis_stats,
        },
        "backbone": {
            "layer": "main",
            "path": backbone,
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "weight": edge_weight[(source, target)],
                    "session_support": edge_sessions[(source, target)],
                }
                for source, target in zip(backbone, backbone[1:])
            ],
        },
        "branches": branches,
        "references": references,
    }
    result["agent_evidence"] = build_agent_evidence(
        sessions, backbone, branches, references
    )
    return result


def mine(
    input_path: Path,
    output_path: Path,
    rho: float,
    max_basis_nodes: int | None,
    min_branch_support: int,
) -> dict[str, Any]:
    result = mine_sessions(
        load_sessions(input_path),
        str(input_path),
        rho,
        max_basis_nodes,
        min_branch_support,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def write_reasoning_prompt(output_path: Path, result: dict[str, Any]) -> Path:
    prompt_path = output_path.with_suffix(".reasoning_prompt.md")
    prompt_path.write_text(
        render_reasoning_prompt(result["agent_evidence"]),
        encoding="utf-8",
    )
    return prompt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rho", type=float, default=0.8,
                        help="Fraction of unique session nodes covered by the basis")
    parser.add_argument("--max-basis-nodes", type=int, default=None)
    parser.add_argument("--min-branch-support", type=int, default=2)
    parser.add_argument("--reasoning-prompt", action="store_true",
                        help="Also write a grounded semantic-reasoning prompt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = mine(
        input_path=args.operator_results,
        output_path=args.output,
        rho=args.rho,
        max_basis_nodes=args.max_basis_nodes,
        min_branch_support=args.min_branch_support,
    )
    prompt_path = None
    if args.reasoning_prompt:
        prompt_path = write_reasoning_prompt(args.output, result)
    print(json.dumps(result["statistics"], indent=2))
    print(f"Wrote {args.output}")
    if prompt_path:
        print(f"Wrote {prompt_path}")


if __name__ == "__main__":
    main()
