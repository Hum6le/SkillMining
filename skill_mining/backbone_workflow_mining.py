#!/usr/bin/env python3
"""Offline state-aware action-backbone mining for ABCD subflows.

Unlike vertex cover, this miner keeps every observed canonical action.  It
learns a rooted directed backbone for compilation order, then retains a small
set of evidence-backed local outgoing transitions for each action.  The
backbone is deliberately a skeleton: branch and retry edges are represented
separately instead of being discarded merely because they are not in the tree.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from eval_tod.abcd.action_schema import canonical_action_name, load_action_schema

try:
    import networkx as nx
except ImportError:  # pragma: no cover - compatibility fallback for old environments
    nx = None


ROOT = "<START>"
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?:\+?\d[\d ()-]{6,}\d)")
_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_FAILURE_RE = re.compile(r"\b(?:fail(?:ed|ure)?|invalid|incorrect|not found|unable|cannot|can't|error|retry)\b", re.I)


def _node_id(subflow: str, action: str) -> str:
    return f"{subflow}:{str(action).strip()}"


def _label(node_id: str) -> str:
    return node_id.split(":", 1)[-1]


def _original_text(conversation: dict[str, Any], turn_index: int, fallback: dict[str, Any]) -> str:
    original = conversation.get("original") or []
    if 0 <= turn_index < len(original):
        row = original[turn_index]
        if isinstance(row, dict):
            return str(row.get("text") or "")
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            return str(row[1] or "")
    return str(fallback.get("text") or "")


def _entity_types(text: str) -> set[str]:
    """Conservative, value-only state features available before an action."""
    result: set[str] = set()
    if _EMAIL_RE.search(text):
        result.add("email")
    if _PHONE_RE.search(text):
        result.add("phone")
    if _ZIP_RE.search(text):
        result.add("zip")
    return result


def _slot_type(value: str) -> str:
    """Infer a conservative value format for an ordered ABCD slot position."""
    text = str(value).strip()
    if _EMAIL_RE.fullmatch(text):
        return "email"
    if _PHONE_RE.fullmatch(text):
        return "phone"
    if _ZIP_RE.fullmatch(text):
        return "zip"
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return "number"
    if re.fullmatch(r"[A-Za-z0-9_-]{5,}", text):
        return "identifier"
    return "text"


def _state_before(conversation: dict[str, Any], action_turn_index: int) -> dict[str, Any]:
    """Build a compact runtime-observable state snapshot before one action."""
    actions: list[str] = []
    entity_types: set[str] = set()
    failure_signal = False
    for index, turn in enumerate(conversation.get("delexed") or []):
        if index >= action_turn_index:
            break
        targets = turn.get("targets") or []
        if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
            actions.append(str(targets[2]))
        if str(turn.get("speaker") or "") == "customer":
            text = _original_text(conversation, index, turn)
            entity_types |= _entity_types(text)
            failure_signal = failure_signal or bool(_FAILURE_RE.search(text))
    return {
        "previous_action": actions[-1] if actions else "",
        "account_selected": "pull-up-account" in actions,
        "credential_types": sorted(entity_types),
        "credential_count": len(entity_types),
        "failure_signal": failure_signal,
    }


def _condition_summary(states: list[dict[str, Any]]) -> dict[str, Any]:
    """Return only stable, observable state facts shared by edge evidence."""
    if not states:
        return {"kind": "transition_observed"}
    n = len(states)
    account_rate = sum(bool(state.get("account_selected")) for state in states) / n
    credential_counts = [int(state.get("credential_count", 0)) for state in states]
    type_counts: Counter[str] = Counter(
        entity for state in states for entity in state.get("credential_types", [])
    )
    common_types = sorted(
        entity for entity, count in type_counts.items() if count / n >= 0.7
    )
    result: dict[str, Any] = {
        "kind": "observed_state_pattern",
        "min_credential_count": min(credential_counts),
        "common_credential_types": common_types,
        "account_selected_rate": round(account_rate, 3),
    }
    if account_rate >= 0.8:
        result["account_selected"] = True
    if sum(bool(state.get("failure_signal")) for state in states) / n >= 0.6:
        result["failure_signal"] = True
    return result


def _has_path(parent: dict[str, str], source: str, target: str) -> bool:
    """Whether following backbone parents from source reaches target."""
    current = source
    seen: set[str] = set()
    while current != ROOT and current not in seen:
        if current == target:
            return True
        seen.add(current)
        current = parent.get(current, ROOT)
    return current == target


def _break_cycles(parent: dict[str, str], candidates: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    """Turn independent best-parent choices into a root-reachable arborescence."""
    while True:
        cycle: list[str] | None = None
        for start in sorted(parent):
            seen: dict[str, int] = {}
            chain: list[str] = []
            current = start
            while current != ROOT and current not in seen:
                seen[current] = len(chain)
                chain.append(current)
                current = parent.get(current, ROOT)
            if current in seen:
                cycle = chain[seen[current]:]
                break
        if not cycle:
            return parent

        cycle_set = set(cycle)
        alternatives: list[tuple[float, str, str]] = []
        for child in cycle:
            current_parent = parent[child]
            current_score = next(
                (edge["score"] for edge in candidates[child] if edge["source"] == current_parent),
                0.0,
            )
            for edge in candidates[child]:
                if edge["source"] in cycle_set:
                    continue
                alternatives.append((current_score - edge["score"], child, edge["source"]))
                break
        if not alternatives:
            parent[min(cycle)] = ROOT
        else:
            _, child, source = min(alternatives, key=lambda row: (row[0], row[1], row[2]))
            parent[child] = source


def mine_backbone_workflow(
    subflow: str,
    conversations: list[dict[str, Any]],
    max_outgoing_edges: int = 3,
    min_branch_support: int = 2,
) -> dict[str, Any]:
    """Mine all-action backbone plus compact per-action transition evidence."""
    node_counts: Counter[str] = Counter()
    start_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    edge_sessions: dict[tuple[str, str], set[str]] = defaultdict(set)
    edge_states: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    slot_examples: dict[str, list[list[str]]] = defaultdict(list)
    action_slot_counts: dict[str, Counter[int]] = defaultdict(Counter)
    slot_position_types: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    action_schema = load_action_schema()
    operator_results: list[dict[str, Any]] = []

    for conversation in conversations:
        sid = str(conversation.get("convo_id") or "?")
        steps: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(conversation.get("delexed") or []):
            targets = turn.get("targets") or []
            if len(targets) < 3 or targets[1] != "take_action" or not targets[2]:
                continue
            canonical_action, suffix_slots = canonical_action_name(
                targets[2], action_schema.get("actions")
            )
            if not canonical_action:
                continue
            node = _node_id(subflow, canonical_action)
            slots = targets[3] if len(targets) > 3 and isinstance(targets[3], list) else []
            slots = [str(value) for value in suffix_slots] + [str(value) for value in slots]
            steps.append({"node": node, "turn_index": turn_index, "slots": slots})
            node_counts[node] += 1
            action_slot_counts[node][len(slots)] += 1
            for position, value in enumerate(slots):
                slot_position_types[node][position][_slot_type(value)] += 1
            if slots and len(slot_examples[node]) < 8:
                slot_examples[node].append(slots)
        if not steps:
            continue

        collapsed: list[dict[str, Any]] = []
        for step in steps:
            if not collapsed or collapsed[-1]["node"] != step["node"]:
                collapsed.append(step)
        start_counts[collapsed[0]["node"]] += 1
        for source, target in zip(collapsed, collapsed[1:]):
            key = (source["node"], target["node"])
            edge_counts[key] += 1
            edge_sessions[key].add(sid)
            if len(edge_states[key]) < 12:
                edge_states[key].append(_state_before(conversation, target["turn_index"]))
        operator_results.append({
            "session_id": sid,
            "index": len(operator_results),
            "ordered_operations": [[subflow, _label(step["node"])] for step in collapsed],
        })

    nodes = sorted(node_counts)
    total_transitions = max(sum(edge_counts.values()), 1)
    edge_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for (source, target), count in edge_counts.items():
        probability = count / max(node_counts[source], 1)
        target_prior = node_counts[target] / max(sum(node_counts.values()), 1)
        lift = probability / max(target_prior, 1e-9)
        score = math.log1p(len(edge_sessions[(source, target)])) + 0.5 * math.log(max(lift, 1e-9))
        edge_rows[(source, target)] = {
            "source": source,
            "target": target,
            "support": int(count),
            "num_sessions": len(edge_sessions[(source, target)]),
            "probability": round(probability, 4),
            "lift": round(lift, 4),
            "score": round(score, 4),
            "condition": _condition_summary(edge_states[(source, target)]),
            "evidence_session_ids": sorted(edge_sessions[(source, target)])[:3],
        }

    # Each node chooses its strongest observed predecessor or the virtual root.
    candidates: dict[str, list[dict[str, Any]]] = {node: [] for node in nodes}
    for edge in edge_rows.values():
        candidates[edge["target"]].append({**edge, "score": float(edge["score"])})
    for node in nodes:
        root_score = math.log1p(start_counts[node]) - 0.25
        candidates[node].append({
            "source": ROOT, "target": node, "support": int(start_counts[node]),
            "num_sessions": int(start_counts[node]), "probability": 0.0,
            "lift": 0.0, "score": root_score,
            "condition": {"kind": "session_entry"}, "evidence_session_ids": [],
        })
        candidates[node].sort(key=lambda edge: (-edge["score"], -edge["support"], edge["source"]))

    if nx is not None:
        graph = nx.DiGraph()
        graph.add_node(ROOT)
        for target, edges in candidates.items():
            for edge in edges:
                graph.add_edge(
                    edge["source"], target,
                    weight=float(edge["score"]),
                    payload=edge,
                )
        tree = nx.maximum_spanning_arborescence(
            graph, attr="weight", preserve_attrs=True,
        )
        backbone_edges = [
            {**data["payload"], "kind": "backbone"}
            for _, _, data in tree.edges(data=True)
        ]
        parent = {edge["target"]: edge["source"] for edge in backbone_edges}
    else:
        # The project depends on networkx, but retain a deterministic fallback
        # for environments that only need to inspect existing artifacts.
        parent = {node: candidates[node][0]["source"] for node in nodes}
        parent = _break_cycles(parent, candidates)
        backbone_edges = []
        for target in sorted(nodes):
            source = parent[target]
            selected = next(edge for edge in candidates[target] if edge["source"] == source)
            backbone_edges.append({**selected, "kind": "backbone"})

    backbone_edges.sort(key=lambda edge: (edge["source"], edge["target"]))

    children: dict[str, list[str]] = defaultdict(list)
    for edge in backbone_edges:
        children[edge["source"]].append(edge["target"])
    for source in children:
        children[source].sort(key=lambda node: (-node_counts[node], node))
    order: list[str] = []
    queue = list(children[ROOT])
    while queue:
        node = queue.pop(0)
        order.append(node)
        queue.extend(children.get(node, []))

    # Keep a compact local view of outgoing edges. The backbone child is always
    # retained; non-backbone edges require support unless they are the sole edge.
    backbone_pairs = {(edge["source"], edge["target"]) for edge in backbone_edges}
    local_transitions: dict[str, list[dict[str, Any]]] = {}
    residual_edges: list[dict[str, Any]] = []
    for source in nodes:
        outgoing = [edge for edge in edge_rows.values() if edge["source"] == source]
        outgoing.sort(key=lambda edge: (
            (source, edge["target"]) not in backbone_pairs,
            -edge["score"], -edge["support"], edge["target"],
        ))
        selected: list[dict[str, Any]] = []
        for edge in outgoing:
            is_backbone = (source, edge["target"]) in backbone_pairs
            if not is_backbone and edge["support"] < min_branch_support:
                continue
            if len(selected) >= max_outgoing_edges and not is_backbone:
                continue
            kind = "backbone" if is_backbone else (
                "retry" if _has_path(parent, source, edge["target"]) else "branch"
            )
            item = {**edge, "kind": kind}
            selected.append(item)
            if kind != "backbone":
                residual_edges.append(item)
        for priority, item in enumerate(selected, start=1):
            item["priority"] = priority
        local_transitions[source] = selected

    def best_main_path() -> list[str]:
        path: list[str] = []
        current = ROOT
        seen: set[str] = set()
        while current in children and children[current]:
            options = children[current]
            current = max(options, key=lambda node: (node_counts[node], node))
            if current in seen:
                break
            path.append(current)
            seen.add(current)
        return path

    graph_nodes = [
        {
            "id": node,
            "label": _label(node),
            "frequency": int(node_counts[node]),
            "slot_examples": slot_examples[node][:5],
            "observed_slot_counts": sorted(action_slot_counts[node]),
            "slot_contract": {
                "min_slots": min(action_slot_counts[node]) if action_slot_counts[node] else 0,
                "max_slots": max(action_slot_counts[node]) if action_slot_counts[node] else 0,
                "positions": [
                    {
                        "position": position + 1,
                        "required_rate": round(
                            sum(count for length, count in action_slot_counts[node].items() if length > position)
                            / max(sum(action_slot_counts[node].values()), 1),
                            3,
                        ),
                        "value_types": [
                            kind for kind, _ in slot_position_types[node][position].most_common()
                        ],
                    }
                    for position in sorted(slot_position_types[node])
                ],
            },
        }
        for node in sorted(nodes, key=lambda node: (order.index(node) if node in order else len(order), node))
    ]
    all_edges = sorted(edge_rows.values(), key=lambda edge: (-edge["score"], edge["source"], edge["target"]))
    retained_pairs = {
        (edge["source"], edge["target"])
        for transitions in local_transitions.values()
        for edge in transitions
    }
    coverage = sum(
        count for pair, count in edge_counts.items() if pair in retained_pairs
    ) / total_transitions
    subgraph = {
        "mining_method": "backbone",
        "nodes": graph_nodes,
        "edges": all_edges,
        "n_selected_nodes": len(graph_nodes),
        "n_selected_edges": len(all_edges),
        "n_sessions": len(operator_results),
        "coverage_pct": round(100 * coverage, 1),
        "backbone": {
            "root": ROOT,
            "edges": backbone_edges,
            "compilation_order": order,
            "main_path": best_main_path(),
        },
        "local_transitions": local_transitions,
        "residual_edges": residual_edges,
        "max_outgoing_edges": max_outgoing_edges,
        "min_branch_support": min_branch_support,
    }
    return {
        "skill_info": {
            "selected_vertices": nodes,
            "num_selected": len(nodes),
            "coverage_pct": subgraph["coverage_pct"],
            "num_sessions": len(conversations),
            "mining_method": "backbone",
        },
        "subgraph": subgraph,
        "operator_results": operator_results,
    }


def sample_transition_cases(
    subflow: str,
    conversations: list[dict[str, Any]],
    max_cases_per_edge: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Collect compact, jointly comparable cases for every observed edge."""
    schema = load_action_schema()
    cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for conversation in conversations:
        steps: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(conversation.get("delexed") or []):
            targets = turn.get("targets") or []
            if len(targets) < 3 or targets[1] != "take_action" or not targets[2]:
                continue
            action, _ = canonical_action_name(targets[2], schema.get("actions"))
            if action:
                steps.append({"node": _node_id(subflow, action), "turn_index": turn_index})
        collapsed: list[dict[str, Any]] = []
        for step in steps:
            if not collapsed or collapsed[-1]["node"] != step["node"]:
                collapsed.append(step)
        for source, target in zip(collapsed, collapsed[1:]):
            key = f"{source['node']} -> {target['node']}"
            if len(cases[key]) >= max_cases_per_edge:
                continue
            target_index = target["turn_index"]
            context_lines: list[str] = []
            for index in range(max(0, target_index - 3), target_index):
                turn = (conversation.get("delexed") or [])[index]
                speaker, text = _get_speaker_text(conversation, index, turn)
                if text:
                    context_lines.append(f"{speaker}: {text}")
            cases[key].append({
                "conversation_id": str(conversation.get("convo_id") or "?"),
                "state": _state_before(conversation, target_index),
                "context": "\n".join(context_lines),
            })
    return dict(cases)


def _get_speaker_text(conversation: dict[str, Any], index: int, turn: dict[str, Any]) -> tuple[str, str]:
    text = _original_text(conversation, index, turn).strip()
    speaker = str(turn.get("speaker") or "unknown").title()
    return speaker, text
