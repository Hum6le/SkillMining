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


def _scenario_values(value: Any) -> list[str]:
    """Flatten scalar scenario facts for conservative slot-source matching."""
    if isinstance(value, dict):
        return [item for child in value.values() for item in _scenario_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _scenario_values(child)]
    text = str(value or "").strip()
    return [text] if text else []


def _slot_source_before(
    conversation: dict[str, Any], action_turn_index: int, slot_value: str,
) -> str:
    """Classify where an observed value was available before an action."""
    needle = str(slot_value).strip().casefold()
    if not needle:
        return "unresolved"
    prior_rows = []
    for index, turn in enumerate(conversation.get("delexed") or []):
        if index >= action_turn_index:
            break
        prior_rows.append((str(turn.get("speaker") or ""), _original_text(conversation, index, turn)))
    latest_customer_index = max(
        (index for index, (speaker, _) in enumerate(prior_rows) if speaker.casefold() == "customer"),
        default=-1,
    )
    for index in range(len(prior_rows) - 1, -1, -1):
        speaker, text = prior_rows[index]
        if needle in str(text).casefold():
            return (
                "current_customer"
                if speaker.casefold() == "customer" and index == latest_customer_index
                else "prior_dialogue"
            )
    if any(needle == candidate.casefold() for candidate in _scenario_values(conversation.get("scenario") or {})):
        return "scenario"
    return "unresolved"


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


def _mine_backbone_workflow_support_lift(
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
    slot_position_sources: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
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
                slot_position_sources[node][position][
                    _slot_source_before(conversation, turn_index, value)
                ] += 1
            if slots and len(slot_examples[node]) < 8:
                slot_examples[node].append(slots)
        if not steps:
            continue

        collapsed: list[dict[str, Any]] = []
        for step in steps:
            if not collapsed or collapsed[-1]["node"] != step["node"]:
                collapsed.append(step)
        start_counts[collapsed[0]["node"]] += 1
        # Collapse consecutive repetitions for the session's canonical
        # operator sequence, but retain each repetition as an explicit
        # self-edge in the transition graph (e.g. A -> A -> B).
        for source, target in zip(steps, steps[1:]):
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
        # A self-edge is useful evidence for retry/repetition induction, but
        # cannot be a parent edge in a directed spanning arborescence.
        if edge["source"] == edge["target"]:
            continue
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
                        "source_types": [
                            source for source, _ in slot_position_sources[node][position].most_common()
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


def _session_edge_sets(subflow: str, conversations: list[dict[str, Any]]) -> dict[str, set[tuple[str, str]]]:
    """Return the complete observed edge set of each session."""
    schema = load_action_schema()
    result: dict[str, set[tuple[str, str]]] = {}
    for conversation in conversations:
        actions = []
        for turn in conversation.get("delexed") or []:
            targets = turn.get("targets") or []
            if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
                action, _ = canonical_action_name(targets[2], schema.get("actions"))
                if action:
                    actions.append(_node_id(subflow, action))
        result[str(conversation.get("convo_id") or "?")] = set(zip(actions, actions[1:]))
    return result


def _rebuild_backbone_from_scores(
    graph: dict[str, Any], conversations: list[dict[str, Any]], subflow: str,
    max_outgoing_edges: int, min_branch_support: int,
) -> None:
    """Recompute arborescence and retained residuals after edge reweighting."""
    nodes = [str(node["id"]) for node in graph.get("nodes", [])]
    frequencies = {str(node["id"]): int(node.get("frequency", 0)) for node in graph.get("nodes", [])}
    schema = load_action_schema()
    starts: Counter[str] = Counter()
    for conversation in conversations:
        for turn in conversation.get("delexed") or []:
            targets = turn.get("targets") or []
            if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
                action, _ = canonical_action_name(targets[2], schema.get("actions"))
                if action:
                    starts[_node_id(subflow, action)] += 1
                    break

    candidates: dict[str, list[dict[str, Any]]] = {node: [] for node in nodes}
    for edge in graph.get("edges", []):
        source, target = str(edge["source"]), str(edge["target"])
        if source != target and target in candidates:
            candidates[target].append({**edge, "score": float(edge["score"])})
    for node in nodes:
        candidates[node].append({
            "source": ROOT, "target": node, "support": int(starts[node]),
            "num_sessions": int(starts[node]), "probability": 0.0, "lift": 0.0,
            "base_weight": round(math.log1p(starts[node]) - 0.25, 4),
            "score": math.log1p(starts[node]) - 0.25,
            "final_backbone_weight": round(math.log1p(starts[node]) - 0.25, 4),
            "condition": {"kind": "session_entry"}, "evidence_session_ids": [],
        })
        candidates[node].sort(key=lambda edge: (-float(edge["score"]), -int(edge.get("support", 0)), str(edge["source"])))

    if nx is not None:
        tree_graph = nx.DiGraph()
        tree_graph.add_node(ROOT)
        for target, edges in candidates.items():
            for edge in edges:
                tree_graph.add_edge(edge["source"], target, weight=float(edge["score"]), payload=edge)
        tree = nx.maximum_spanning_arborescence(tree_graph, attr="weight", preserve_attrs=True)
        backbone_edges = [{**data["payload"], "kind": "backbone"} for _, _, data in tree.edges(data=True)]
        parent = {str(edge["target"]): str(edge["source"]) for edge in backbone_edges}
    else:  # pragma: no cover - normal environments include networkx
        parent = {node: str(candidates[node][0]["source"]) for node in nodes}
        parent = _break_cycles(parent, candidates)
        backbone_edges = [
            {**next(edge for edge in candidates[node] if edge["source"] == parent[node]), "kind": "backbone"}
            for node in sorted(nodes)
        ]

    children: dict[str, list[str]] = defaultdict(list)
    for edge in backbone_edges:
        children[str(edge["source"])].append(str(edge["target"]))
    for source in children:
        children[source].sort(key=lambda node: (-frequencies.get(node, 0), node))
    order: list[str] = []
    queue = list(children[ROOT])
    while queue:
        node = queue.pop(0)
        order.append(node)
        queue.extend(children.get(node, []))

    backbone_pairs = {(str(edge["source"]), str(edge["target"])) for edge in backbone_edges}
    local, residual = {}, []
    for source in nodes:
        outgoing = [edge for edge in graph.get("edges", []) if str(edge["source"]) == source]
        outgoing.sort(key=lambda edge: (
            (str(edge["source"]), str(edge["target"])) not in backbone_pairs,
            -float(edge["score"]), -int(edge.get("support", 0)), str(edge["target"]),
        ))
        selected = []
        for edge in outgoing:
            pair = (str(edge["source"]), str(edge["target"]))
            is_backbone = pair in backbone_pairs
            if not is_backbone and int(edge.get("support", 0)) < min_branch_support:
                continue
            if len(selected) >= max_outgoing_edges and not is_backbone:
                continue
            kind = "backbone" if is_backbone else (
                "retry" if _has_path(parent, source, str(edge["target"])) else "branch"
            )
            selected.append({**edge, "kind": kind})
            if kind != "backbone":
                residual.append(selected[-1])
        for priority, edge in enumerate(selected, 1):
            edge["priority"] = priority
        local[source] = selected

    graph["backbone"] = {
        "root": ROOT,
        "edges": sorted(backbone_edges, key=lambda edge: (edge["source"], edge["target"])),
        "compilation_order": order,
        "main_path": _best_backbone_path(children, graph.get("nodes", [])),
    }
    graph["local_transitions"] = local
    graph["residual_edges"] = residual
    retained = {(str(edge["source"]), str(edge["target"])) for edges in local.values() for edge in edges}
    session_edges = _session_edge_sets(subflow, conversations)
    graph["coverage_pct"] = round(
        100 * sum(len(edges & retained) for edges in session_edges.values())
        / max(sum(len(edges) for edges in session_edges.values()), 1), 1,
    )


def mine_backbone_workflow_discriminative(
    subflow: str, conversations: list[dict[str, Any]], max_outgoing_edges: int = 3,
    min_branch_support: int = 2, discriminative_lambda: float = 1.0,
    discriminative_clip: float = 3.0, cohort_max_skills: int = 8,
    cohort_min_sessions: int = 20,
) -> dict[str, Any]:
    """Mine one backbone using temporary session cohorts to reweight edges.

    Cohorts are a training-only contrast set. They never create separate
    runtime skills or route a test dialogue; they only reward transitions that
    are stable inside one recurring trajectory pattern but uncommon outside it.
    """
    base = _mine_backbone_workflow_support_lift(
        subflow, conversations, max_outgoing_edges=max_outgoing_edges,
        min_branch_support=min_branch_support,
    )
    graph = base["subgraph"]
    try:
        from skill_mining.semantic_subflow import discover_motif_prototypes
        min_sessions = min(max(2, cohort_min_sessions), max(len(conversations) // 2, 2))
        cohort_result = discover_motif_prototypes(
            subflow, conversations, max_skills=cohort_max_skills,
            min_sessions=min_sessions,
        )
    except Exception as exc:  # a backbone must remain available for every split
        cohort_result = {
            "protocol": "weighted_motif_prototypes_v1", "skills": [],
            "session_assignments": {}, "selected_k": 0,
            "error": repr(exc),
        }

    assignment = cohort_result.get("session_assignments", {})
    members: dict[str, set[str]] = defaultdict(set)
    for sid, row in assignment.items():
        skill_id = str(row.get("skill_id", "")) if isinstance(row, dict) else ""
        if skill_id:
            members[skill_id].add(str(sid))
    session_edges = _session_edge_sets(subflow, conversations)
    all_sessions = set(session_edges)
    summaries = []
    for skill_id, ids in sorted(members.items()):
        if len(ids) < 2:
            continue
        summaries.append({"cohort_id": skill_id, "num_sessions": len(ids)})

    for edge in graph.get("edges", []):
        pair = (str(edge["source"]), str(edge["target"]))
        base_weight = float(edge["score"])
        best_score, best_id, best_inside, best_outside = 0.0, "", 0, 0
        for skill_id, ids in members.items():
            if len(ids) < 2 or len(all_sessions - ids) < 1:
                continue
            inside = sum(pair in session_edges.get(sid, set()) for sid in ids)
            outside_ids = all_sessions - ids
            outside = sum(pair in session_edges.get(sid, set()) for sid in outside_ids)
            epsilon = 1.0
            inside_rate = (inside + epsilon) / (len(ids) + 2 * epsilon)
            outside_rate = (outside + epsilon) / (len(outside_ids) + 2 * epsilon)
            value = math.log(inside_rate / outside_rate)
            if value > best_score:
                best_score, best_id, best_inside, best_outside = value, skill_id, inside, outside
        bonus = discriminative_lambda * min(max(best_score, 0.0), discriminative_clip)
        edge["base_weight"] = round(base_weight, 4)
        edge["best_cohort_id"] = best_id or None
        edge["support_in_cohort"] = int(best_inside)
        edge["support_outside_cohort"] = int(best_outside)
        edge["discriminative_log_odds"] = round(max(best_score, 0.0), 4)
        edge["score"] = round(base_weight + bonus, 4)
        edge["final_backbone_weight"] = edge["score"]

    _rebuild_backbone_from_scores(
        graph, conversations, subflow, max_outgoing_edges, min_branch_support,
    )
    graph["mining_method"] = "discriminative_backbone"
    graph["cohort_reweighting"] = {
        "protocol": cohort_result.get("protocol", "weighted_motif_prototypes_v1"),
        "selected_cohorts": summaries,
        "selected_k": int(cohort_result.get("selected_k", 0) or 0),
        "lambda": discriminative_lambda,
        "clip": discriminative_clip,
        "cohort_min_sessions": min_sessions,
    }
    base["skill_info"]["mining_method"] = "discriminative_backbone"
    base["skill_info"]["coverage_pct"] = graph["coverage_pct"]
    return base


def mine_backbone_workflow(
    subflow: str, conversations: list[dict[str, Any]], max_outgoing_edges: int = 3,
    min_branch_support: int = 2, discriminative_lambda: float = 1.0,
    discriminative_clip: float = 3.0,
) -> dict[str, Any]:
    """Default backbone miner: discriminative session-aware arborescence."""
    return mine_backbone_workflow_discriminative(
        subflow, conversations, max_outgoing_edges=max_outgoing_edges,
        min_branch_support=min_branch_support,
        discriminative_lambda=discriminative_lambda,
        discriminative_clip=discriminative_clip,
    )


def mine_backbone_workflow_session_coverage(
    subflow: str,
    conversations: list[dict[str, Any]],
    max_outgoing_edges: int = 3,
    min_branch_support: int = 2,
    coverage_lambda: float = 0.2,
    max_swap_rounds: int = 3,
    discriminative_lambda: float = 1.0,
    discriminative_clip: float = 3.0,
) -> dict[str, Any]:
    """Compatibility alias for the discriminative backbone.

    ``backbone_coverage`` remains accepted by historical commands, but no
    longer performs a separate edge-swap optimization. This prevents a silent
    divergence between the two names after discriminative reweighting became
    the canonical backbone objective.
    """
    return mine_backbone_workflow_discriminative(
        subflow, conversations, max_outgoing_edges=max_outgoing_edges,
        min_branch_support=min_branch_support,
        discriminative_lambda=discriminative_lambda,
        discriminative_clip=discriminative_clip,
    )

    # Historical implementation retained below for artifact compatibility;
    # unreachable by design after the method unification above.
    base = mine_backbone_workflow(
        subflow,
        conversations,
        max_outgoing_edges=max_outgoing_edges,
        min_branch_support=min_branch_support,
    )
    graph = base["subgraph"]
    nodes = [node["id"] for node in graph["nodes"]]
    edge_rows = {(edge["source"], edge["target"]): edge for edge in graph["edges"]}
    parent = {
        edge["target"]: edge["source"]
        for edge in graph["backbone"]["edges"]
    }
    session_edges: list[set[tuple[str, str]]] = []
    schema = load_action_schema()
    for conversation in conversations:
        actions: list[str] = []
        for turn in conversation.get("delexed") or []:
            targets = turn.get("targets") or []
            if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
                action, _ = canonical_action_name(targets[2], schema.get("actions"))
                if action:
                    actions.append(action)
        session_edges.append({
            (_node_id(subflow, source), _node_id(subflow, target))
            for source, target in zip(actions, actions[1:])
        })

    def coverage(pairs: set[tuple[str, str]]) -> float:
        if not session_edges:
            return 0.0
        return sum(
            len(edges & pairs) / max(len(edges), 1)
            for edges in session_edges
        ) / len(session_edges)

    def edge_score(pairs: set[tuple[str, str]]) -> float:
        return sum(float(edge_rows[pair]["score"]) for pair in pairs if pair in edge_rows)

    def objective(pairs: set[tuple[str, str]]) -> float:
        return edge_score(pairs) + coverage_lambda * coverage(pairs)

    current_pairs = {
        (edge["source"], edge["target"])
        for edge in graph["backbone"]["edges"]
        if edge["source"] != ROOT
    }
    for _ in range(max_swap_rounds):
        current_objective = objective(current_pairs)
        best_delta = 0.0
        best_change: tuple[str, str, str] | None = None
        for target in nodes:
            old_source = parent[target]
            old_pair = (old_source, target)
            for (source, candidate_target), _edge in edge_rows.items():
                if candidate_target != target or source == old_source:
                    continue
                current = source
                seen: set[str] = set()
                creates_cycle = False
                while current != ROOT and current not in seen:
                    if current == target:
                        creates_cycle = True
                        break
                    seen.add(current)
                    current = parent.get(current, ROOT)
                if creates_cycle:
                    continue
                trial = set(current_pairs)
                trial.discard(old_pair)
                trial.add((source, target))
                delta = objective(trial) - current_objective
                if delta > best_delta + 1e-9:
                    best_delta = delta
                    best_change = (target, old_source, source)
        if best_change is None:
            break
        target, old_source, new_source = best_change
        parent[target] = new_source
        current_pairs.discard((old_source, target))
        current_pairs.add((new_source, target))

    root_edges = {
        edge["target"]: edge
        for edge in graph["backbone"]["edges"]
        if edge["source"] == ROOT
    }
    backbone_edges = []
    for target in sorted(nodes):
        source = parent[target]
        if source == ROOT:
            edge = root_edges[target]
        else:
            edge = edge_rows[(source, target)]
        backbone_edges.append({**edge, "kind": "backbone"})

    children: dict[str, list[str]] = defaultdict(list)
    for edge in backbone_edges:
        children[edge["source"]].append(edge["target"])
    for source in children:
        children[source].sort()
    order: list[str] = []
    queue = list(children[ROOT])
    while queue:
        node = queue.pop(0)
        order.append(node)
        queue.extend(children.get(node, []))

    backbone_pairs = {(edge["source"], edge["target"]) for edge in backbone_edges}
    local: dict[str, list[dict[str, Any]]] = {}
    residual: list[dict[str, Any]] = []
    for source in nodes:
        outgoing = [edge for edge in graph["edges"] if edge["source"] == source]
        outgoing.sort(key=lambda edge: (
            (edge["source"], edge["target"]) not in backbone_pairs,
            -edge["score"], edge["target"],
        ))
        selected = []
        for edge in outgoing:
            is_backbone = (edge["source"], edge["target"]) in backbone_pairs
            if not is_backbone and edge["support"] < min_branch_support:
                continue
            if len(selected) >= max_outgoing_edges and not is_backbone:
                continue
            kind = "backbone" if is_backbone else (
                "retry" if _has_path(parent, source, edge["target"]) else "branch"
            )
            selected.append({**edge, "kind": kind})
            if kind != "backbone":
                residual.append(selected[-1])
        for priority, edge in enumerate(selected, 1):
            edge["priority"] = priority
        local[source] = selected

    retained_pairs = {
        (edge["source"], edge["target"])
        for edges in local.values() for edge in edges
    }
    graph["mining_method"] = "backbone_coverage"
    graph["backbone"] = {
        "root": ROOT,
        "edges": sorted(backbone_edges, key=lambda edge: (edge["source"], edge["target"])),
        "compilation_order": order,
        "main_path": _best_backbone_path(children, graph["nodes"]),
    }
    graph["local_transitions"] = local
    graph["residual_edges"] = residual
    turn_score = edge_score(current_pairs)
    mean_coverage = coverage(current_pairs)
    route_coverage = (
        sum(
            len(edges & current_pairs) / max(len(edges), 1) >= 0.8
            for edges in session_edges
        ) / max(len(session_edges), 1)
    )
    graph["coverage_objective"] = {
        "turn_edge_score": round(turn_score, 4),
        "session_mean_coverage": round(mean_coverage, 4),
        "session_route_coverage_at_80pct": round(route_coverage, 4),
        "lambda": coverage_lambda,
        "combined_objective": round(
            turn_score + coverage_lambda * mean_coverage, 4
        ),
        "swap_rounds": max_swap_rounds,
    }
    graph["coverage_pct"] = round(
        100 * sum(len(edges & retained_pairs) for edges in session_edges)
        / max(sum(len(edges) for edges in session_edges), 1),
        1,
    )
    base["skill_info"]["mining_method"] = "backbone_coverage"
    base["skill_info"]["coverage_pct"] = graph["coverage_pct"]
    return base


def _best_backbone_path(children: dict[str, list[str]], nodes: list[dict[str, Any]]) -> list[str]:
    frequencies = {node["id"]: node.get("frequency", 0) for node in nodes}
    path: list[str] = []
    current = ROOT
    seen: set[str] = set()
    while children.get(current):
        current = max(children[current], key=lambda node: (frequencies.get(node, 0), node))
        if current in seen:
            break
        seen.add(current)
        path.append(current)
    return path


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
            action, suffix_slots = canonical_action_name(targets[2], schema.get("actions"))
            if action:
                raw_slots = targets[3] if len(targets) > 3 and isinstance(targets[3], list) else []
                steps.append({
                    "node": _node_id(subflow, action),
                    "turn_index": turn_index,
                    "slots": [str(value) for value in suffix_slots] + [str(value) for value in raw_slots],
                })
        # Keep repeated actions as self-edge evidence. The same action node is
        # still used; repetition is represented by source == target.
        for source, target in zip(steps, steps[1:]):
            key = f"{source['node']} -> {target['node']}"
            if len(cases[key]) >= max_cases_per_edge:
                continue
            target_index = target["turn_index"]
            context_lines: list[str] = []
            # Keep the full prefix so continuation-mode induction can inspect
            # earlier verification, request, and failure turns. The compiler
            # prompt applies its own per-case character budget.
            for index in range(0, target_index):
                turn = (conversation.get("delexed") or [])[index]
                speaker, text = _get_speaker_text(conversation, index, turn)
                if text:
                    context_lines.append(f"{speaker}: {text}")
            inter_action_lines: list[str] = []
            # This is the interaction which actually mediates an action edge.
            # It may contain an agent proposal followed by a user acceptance,
            # neither of which is represented by the action graph alone.
            for index in range(source["turn_index"] + 1, target_index):
                turn = (conversation.get("delexed") or [])[index]
                speaker, text = _get_speaker_text(conversation, index, turn)
                if text:
                    inter_action_lines.append(f"{speaker}: {text}")
            cases[key].append({
                "conversation_id": str(conversation.get("convo_id") or "?"),
                "state": _state_before(conversation, target_index),
                "source_slots": source.get("slots", []),
                "target_slots": target.get("slots", []),
                "context": "\n".join(context_lines),
                "inter_action_dialogue": "\n".join(inter_action_lines),
            })
    return dict(cases)


def _get_speaker_text(conversation: dict[str, Any], index: int, turn: dict[str, Any]) -> tuple[str, str]:
    text = _original_text(conversation, index, turn).strip()
    speaker = str(turn.get("speaker") or "unknown").title()
    return speaker, text
