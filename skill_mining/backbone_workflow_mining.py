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
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{2,}", re.I)
_ROUTING_STOPWORDS = {
    "the", "and", "that", "this", "with", "for", "you", "your", "are", "was",
    "have", "has", "had", "but", "not", "can", "could", "would", "should",
    "from", "they", "them", "their", "just", "please", "want", "need", "help",
    "hello", "thanks", "thank", "okay", "account", "order",
}


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


def observable_features_from_context(
    context: str,
    account_selected: bool = False,
) -> set[str]:
    """Extract runtime-observable binary routing features from dialogue text.

    Lexical features are restricted to the latest customer utterance.  Global
    entity/failure flags may use the full visible context.  The function is
    shared by offline mining and runtime routing so no train-only feature can
    leak into a learned guard.
    """
    text = str(context or "")
    customer_lines = [
        line.split("]", 1)[-1].strip()
        for line in text.splitlines()
        if line.strip().casefold().startswith("[customer]")
    ]
    latest_customer = customer_lines[-1] if customer_lines else text
    tokens = [
        token.casefold() for token in _TOKEN_RE.findall(latest_customer)
        if token.casefold() not in _ROUTING_STOPWORDS
    ]
    features = {f"customer_token:{token}" for token in set(tokens)}
    features.update(
        f"customer_bigram:{left}_{right}"
        for left, right in zip(tokens, tokens[1:])
        if left != right
    )
    entities = _entity_types(text)
    features.update(f"has_{entity}" for entity in entities)
    for threshold in (1, 2, 3):
        if len(entities) >= threshold:
            features.add(f"credential_count_ge_{threshold}")
    if account_selected:
        features.add("account_selected")
    if _FAILURE_RE.search(text):
        features.add("failure_signal")
    return features


def _observable_features_before(
    conversation: dict[str, Any], action_turn_index: int,
) -> set[str]:
    """Build exactly the text/state view available immediately before an action."""
    lines: list[str] = []
    for index, turn in enumerate(conversation.get("delexed") or []):
        if index >= action_turn_index:
            break
        speaker = str(turn.get("speaker") or "")
        label = {"agent": "Agent", "customer": "Customer", "action": "System"}.get(
            speaker.casefold(), speaker.title() or "Turn",
        )
        text = _original_text(conversation, index, turn)
        if text:
            lines.append(f"[{label}] {text}")
    state = _state_before(conversation, action_turn_index)
    return observable_features_from_context(
        "\n".join(lines), account_selected=bool(state.get("account_selected")),
    )


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


def mine_backbone_workflow_heuristics(
    subflow: str,
    conversations: list[dict[str, Any]],
    max_outgoing_edges: int = 3,
    min_branch_support: int = 2,
    dependency_weight: float = 1.0,
) -> dict[str, Any]:
    """Process-mining baseline using the Heuristics Miner dependency signal.

    The classic dependency measure discounts bidirectional/noisy directly-
    follows relations.  We use it only to reweight the existing DFG before
    applying the same arborescence and local-edge budget as the main miner.
    """
    base = _mine_backbone_workflow_support_lift(
        subflow,
        conversations,
        max_outgoing_edges=max_outgoing_edges,
        min_branch_support=min_branch_support,
    )
    graph = base["subgraph"]
    support = {
        (str(edge["source"]), str(edge["target"])): int(edge.get("support", 0))
        for edge in graph.get("edges", [])
    }
    for edge in graph.get("edges", []):
        source, target = str(edge["source"]), str(edge["target"])
        forward = support.get((source, target), 0)
        if source == target:
            dependency = forward / (forward + 1.0)
        else:
            backward = support.get((target, source), 0)
            dependency = (forward - backward) / (forward + backward + 1.0)
        base_weight = float(edge.get("score", 0.0))
        edge["base_weight"] = round(base_weight, 4)
        edge["dependency"] = round(dependency, 6)
        edge["score"] = round(base_weight + dependency_weight * dependency, 4)
        edge["final_backbone_weight"] = edge["score"]
    _rebuild_backbone_from_scores(
        graph, conversations, subflow, max_outgoing_edges, min_branch_support,
    )
    graph["mining_method"] = "heuristics_dependency_backbone"
    graph["heuristics_miner"] = {"dependency_weight": dependency_weight}
    base["skill_info"]["mining_method"] = graph["mining_method"]
    base["skill_info"]["coverage_pct"] = graph["coverage_pct"]
    return base


def _session_edge_occurrences(
    subflow: str,
    conversations: list[dict[str, Any]],
) -> list[Counter[tuple[str, str]]]:
    """Return directly-follows occurrence counts for every session."""
    schema = load_action_schema()
    sessions: list[Counter[tuple[str, str]]] = []
    for conversation in conversations:
        actions: list[str] = []
        for turn in conversation.get("delexed") or []:
            targets = turn.get("targets") or []
            if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
                action, _ = canonical_action_name(targets[2], schema.get("actions"))
                if action:
                    actions.append(_node_id(subflow, action))
        counts: Counter[tuple[str, str]] = Counter(zip(actions, actions[1:]))
        if counts:
            sessions.append(counts)
    return sessions


def observable_transition_events(
    subflow: str,
    conversations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return directly-follows events with only pre-target observable features."""
    schema = load_action_schema()
    events: list[dict[str, Any]] = []
    for conversation in conversations:
        steps: list[tuple[str, int]] = []
        for turn_index, turn in enumerate(conversation.get("delexed") or []):
            targets = turn.get("targets") or []
            if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
                action, _ = canonical_action_name(targets[2], schema.get("actions"))
                if action:
                    steps.append((_node_id(subflow, action), turn_index))
        for (source, _), (target, target_turn) in zip(steps, steps[1:]):
            events.append({
                "source": source,
                "target": target,
                "features": sorted(_observable_features_before(conversation, target_turn)),
                "session_id": str(conversation.get("convo_id") or "?"),
            })
    return events


def _edge_observation_index(
    events: list[dict[str, Any]],
) -> dict[tuple[str, str], list[set[str]]]:
    result: dict[tuple[str, str], list[set[str]]] = defaultdict(list)
    for event in events:
        result[(str(event["source"]), str(event["target"]))].append(
            set(map(str, event.get("features") or []))
        )
    return result


def _observable_distinction(
    left: list[set[str]],
    right: list[set[str]],
    smoothing: float = 0.5,
) -> dict[str, Any]:
    """Find the best single observable separator for two routing branches."""
    if not left or not right:
        return {"feature": "", "gap": 0.0, "ambiguity": 1.0, "left_rate": 0.0, "right_rate": 0.0}
    features = set().union(*left, *right)
    best = (0.0, "", 0.0, 0.0)
    for feature in features:
        left_rate = (sum(feature in row for row in left) + smoothing) / (len(left) + 2 * smoothing)
        right_rate = (sum(feature in row for row in right) + smoothing) / (len(right) + 2 * smoothing)
        gap = abs(left_rate - right_rate)
        candidate = (gap, feature, left_rate, right_rate)
        if candidate > best:
            best = candidate
    gap, feature, left_rate, right_rate = best
    return {
        "feature": feature,
        "gap": round(gap, 6),
        "ambiguity": round(1.0 - gap, 6),
        "left_rate": round(left_rate, 6),
        "right_rate": round(right_rate, 6),
        "preferred_for": "left" if left_rate >= right_rate else "right",
    }


def _label_entropy(events: list[dict[str, Any]]) -> float:
    counts = Counter(str(event["target"]) for event in events)
    total = max(sum(counts.values()), 1)
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count)


def _build_routing_tree(
    source: str,
    events: list[dict[str, Any]],
    max_depth: int,
    min_leaf_support: int,
    min_information_gain: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Learn a shallow interpretable partition of one action's decision contexts."""
    mode_rows: list[dict[str, Any]] = []
    next_mode = [0]

    def build(rows: list[dict[str, Any]], depth: int, guard: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(str(row["target"]) for row in rows)
        ranked = [target for target, _ in counts.most_common()]
        parent_entropy = _label_entropy(rows)
        best: tuple[float, str, list[dict[str, Any]], list[dict[str, Any]]] | None = None
        if depth < max_depth and len(counts) > 1 and len(rows) >= 2 * min_leaf_support:
            features = sorted(set().union(*(set(row.get("features") or []) for row in rows)))
            for feature in features:
                present = [row for row in rows if feature in set(row.get("features") or [])]
                absent = [row for row in rows if feature not in set(row.get("features") or [])]
                if len(present) < min_leaf_support or len(absent) < min_leaf_support:
                    continue
                weighted = (
                    len(present) * _label_entropy(present)
                    + len(absent) * _label_entropy(absent)
                ) / len(rows)
                gain = parent_entropy - weighted
                candidate = (gain, feature, present, absent)
                if best is None or (gain, feature) > (best[0], best[1]):
                    best = candidate
        if best is not None and best[0] >= min_information_gain:
            gain, feature, present, absent = best
            return {
                "kind": "split",
                "feature": feature,
                "information_gain": round(gain, 6),
                "present": build(present, depth + 1, guard + [{"feature": feature, "present": True}]),
                "absent": build(absent, depth + 1, guard + [{"feature": feature, "present": False}]),
            }
        mode_id = f"{source}#mode{next_mode[0]}"
        next_mode[0] += 1
        total = max(len(rows), 1)
        mode = {
            "mode_id": mode_id,
            "source": source,
            "guard": guard,
            "support": len(rows),
            "candidate_targets": ranked,
            "target_probabilities": {
                target: round(count / total, 6) for target, count in counts.most_common()
            },
            "entropy": round(parent_entropy, 6),
        }
        mode_rows.append(mode)
        return {"kind": "leaf", "mode_id": mode_id, "candidate_targets": ranked}

    return build(events, 0, []), mode_rows


def route_observable_tree(tree: dict[str, Any], features: set[str]) -> dict[str, Any]:
    """Route an observable feature set to one learned decision-mode leaf."""
    node = tree
    while node.get("kind") == "split":
        node = node["present"] if str(node.get("feature")) in features else node["absent"]
    return node


def _build_observable_router(
    events: list[dict[str, Any]],
    retained_pairs: set[tuple[str, str]],
    max_depth: int,
    min_leaf_support: int,
    min_information_gain: float,
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        pair = (str(event["source"]), str(event["target"]))
        if pair in retained_pairs:
            by_source[pair[0]].append(event)
    sources: dict[str, Any] = {}
    total_extra_modes = 0
    for source, rows in sorted(by_source.items()):
        if len({str(row["target"]) for row in rows}) < 2:
            continue
        tree, modes = _build_routing_tree(
            source, rows, max_depth=max_depth,
            min_leaf_support=min_leaf_support,
            min_information_gain=min_information_gain,
        )
        sources[source] = {"tree": tree, "modes": modes}
        total_extra_modes += max(len(modes) - 1, 0)
    return {
        "format": "observable_routing_modes_v1",
        "max_depth": max_depth,
        "min_leaf_support": min_leaf_support,
        "min_information_gain": min_information_gain,
        "sources": sources,
        "num_mode_sources": len(sources),
        "num_modes": sum(len(row["modes"]) for row in sources.values()),
        "extra_mode_nodes": total_extra_modes,
    }


def structural_history_features(history: list[str], history_order: int = 3) -> set[str]:
    """Encode only the already executed action path; no dialogue text is used."""
    labels = [_label(str(action)) for action in history if str(action)]
    if not labels:
        return set()
    features: set[str] = {f"start:{labels[0]}"}
    for lag in range(1, history_order + 1):
        if len(labels) > lag:
            features.add(f"lag{lag}:{labels[-1 - lag]}")
    for action in set(labels[:-1]):
        features.add(f"seen:{action}")
    current_count = labels.count(labels[-1])
    for threshold in (2, 3):
        if current_count >= threshold:
            features.add(f"current_count_ge_{threshold}")
    for threshold in (3, 5, 8):
        if len(labels) >= threshold:
            features.add(f"position_ge_{threshold}")
    return features


def structural_action_occurrences(
    subflow: str,
    conversations: list[dict[str, Any]],
    history_order: int = 3,
) -> list[dict[str, Any]]:
    """Return action occurrences labelled only by action-prefix structure."""
    schema = load_action_schema()
    rows: list[dict[str, Any]] = []
    for conversation in conversations:
        sequence: list[str] = []
        for turn in conversation.get("delexed") or []:
            targets = turn.get("targets") or []
            if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
                action, _ = canonical_action_name(targets[2], schema.get("actions"))
                if action:
                    sequence.append(_node_id(subflow, action))
        for index, source in enumerate(sequence):
            history = sequence[:index + 1]
            rows.append({
                "source": source,
                "target": sequence[index + 1] if index + 1 < len(sequence) else "",
                "history": history,
                "history_signature": tuple(_label(value) for value in history[-history_order:]),
                "features": sorted(structural_history_features(history, history_order)),
                "session_id": str(conversation.get("convo_id") or "?"),
            })
    return rows


def _build_structural_tree(
    source: str,
    rows: list[dict[str, Any]],
    max_depth: int,
    min_leaf_support: int,
    node_penalty: float,
    edge_penalty: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """MDL-style state splitting over action-history predicates only."""
    modes: list[dict[str, Any]] = []
    next_mode = [0]

    def build(group: list[dict[str, Any]], depth: int, guard: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(str(row["target"]) for row in group if row.get("target"))
        total_targets = max(sum(counts.values()), 1)
        entropy = _label_entropy([row for row in group if row.get("target")])
        best: tuple[float, float, str, list[dict[str, Any]], list[dict[str, Any]], int] | None = None
        if depth < max_depth and len(counts) > 1 and len(group) >= 2 * min_leaf_support:
            features = sorted(set().union(*(set(row.get("features") or []) for row in group)))
            for feature in features:
                present = [row for row in group if feature in set(row.get("features") or [])]
                absent = [row for row in group if feature not in set(row.get("features") or [])]
                if len(present) < min_leaf_support or len(absent) < min_leaf_support:
                    continue
                weighted_entropy = (
                    len(present) * _label_entropy([row for row in present if row.get("target")])
                    + len(absent) * _label_entropy([row for row in absent if row.get("target")])
                ) / len(group)
                information_gain = max(entropy - weighted_entropy, 0.0)
                gain_bits = len(group) * information_gain
                parent_degree = len(counts)
                child_degree = len({row["target"] for row in present if row.get("target")}) + len({
                    row["target"] for row in absent if row.get("target")
                })
                extra_edges = max(child_degree - parent_degree, 0)
                code_length = math.log2(len(group) + 1)
                mdl_penalty = (node_penalty + edge_penalty * extra_edges) * code_length
                net_gain = gain_bits - mdl_penalty
                candidate = (net_gain, information_gain, feature, present, absent, extra_edges)
                if best is None or (net_gain, information_gain, feature) > (best[0], best[1], best[2]):
                    best = candidate
        if best is not None and best[0] > 0.0:
            net_gain, information_gain, feature, present, absent, extra_edges = best
            return {
                "kind": "split",
                "feature": feature,
                "information_gain": round(information_gain, 8),
                "mdl_net_gain_bits": round(net_gain, 8),
                "extra_edge_penalty_units": extra_edges,
                "present": build(present, depth + 1, guard + [{"feature": feature, "present": True}]),
                "absent": build(absent, depth + 1, guard + [{"feature": feature, "present": False}]),
            }
        mode_id = f"{source}#state{next_mode[0]}"
        next_mode[0] += 1
        mode = {
            "mode_id": mode_id,
            "base_action": source,
            "guard": guard,
            "support": len(group),
            "candidate_actions": [target for target, _ in counts.most_common()],
            "target_probabilities": {
                target: round(count / total_targets, 8) for target, count in counts.most_common()
            },
            "entropy": round(entropy, 8),
            "observed_history_signatures": sorted({
                "|".join(map(str, row.get("history_signature") or ())) for row in group
            }),
        }
        modes.append(mode)
        return {"kind": "leaf", "mode_id": mode_id}

    return build(rows, 0, []), modes


def route_structural_tree(tree: dict[str, Any], history: list[str], history_order: int = 3) -> dict[str, Any]:
    """Assign an executed action prefix to one refined process state."""
    features = structural_history_features(history, history_order)
    node = tree
    while node.get("kind") == "split":
        node = node["present"] if str(node.get("feature")) in features else node["absent"]
    return node


def motif_history_signature(history: list[str], history_order: int = 3) -> str:
    """Return a causal, anchored process motif for the current action occurrence.

    The motif contains only the already executed path.  Besides the bounded
    predecessor chain, it records whether the anchor action is a revisit and
    its nearest loop-back distance.  Consequently it is available both while
    mining traces and while executing the resulting graph.
    """
    labels = [_label(str(action)) for action in history if str(action)]
    if not labels:
        return ""
    anchor = labels[-1]
    predecessors = labels[max(0, len(labels) - history_order - 1):-1]
    padded = ["<START>"] * max(history_order - len(predecessors), 0) + predecessors
    previous_positions = [index for index, value in enumerate(labels[:-1]) if value == anchor]
    loop_distance = len(labels) - 1 - previous_positions[-1] if previous_positions else 0
    return "|".join([
        "path=" + ">".join(padded),
        f"revisit={int(bool(previous_positions))}",
        f"loop={min(loop_distance, history_order + 1)}",
    ])


def motif_action_occurrences(
    subflow: str,
    conversations: list[dict[str, Any]],
    history_order: int = 3,
) -> list[dict[str, Any]]:
    """Lift action traces into occurrence rows with causal process motifs."""
    rows = structural_action_occurrences(subflow, conversations, history_order)
    for row in rows:
        row["motif_signature"] = motif_history_signature(
            list(row.get("history") or []), history_order,
        )
    return rows


def _motif_state_cost(
    rows: list[dict[str, Any]],
    code_length: float,
    node_penalty: float,
    edge_penalty: float,
) -> tuple[float, float, int]:
    target_rows = [row for row in rows if row.get("target")]
    entropy = _label_entropy(target_rows)
    out_degree = len({str(row["target"]) for row in target_rows})
    data_bits = len(target_rows) * entropy
    model_bits = (node_penalty + edge_penalty * out_degree) * code_length
    return data_bits + model_bits, data_bits, out_degree


def _cluster_action_motifs(
    source: str,
    rows: list[dict[str, Any]],
    min_motif_support: int,
    node_penalty: float,
    edge_penalty: float,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, float]]:
    """Agglomeratively compress occurrence motifs under a local MDL objective.

    Exact causal motifs form micro-states.  Rare motifs share an initial
    backoff state, after which pairs are merged whenever doing so shortens the
    joint next-action and graph code.  A source is left unsplit unless the
    final partition beats the one-state model, preventing gratuitous states.
    """
    by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_signature[str(row.get("motif_signature") or "")].append(row)
    frequent = {
        signature: group for signature, group in by_signature.items()
        if len(group) >= min_motif_support
    }
    rare_signatures = sorted(set(by_signature) - set(frequent))
    clusters: list[dict[str, Any]] = [
        {"signatures": {signature}, "rows": list(group)}
        for signature, group in sorted(frequent.items())
    ]
    if rare_signatures:
        clusters.append({
            "signatures": set(rare_signatures),
            "rows": [row for signature in rare_signatures for row in by_signature[signature]],
        })
    if not clusters:
        clusters = [{"signatures": set(by_signature), "rows": list(rows)}]

    code_length = math.log2(len(rows) + 1)
    while len(clusters) > 1:
        best: tuple[float, int, int, dict[str, Any]] | None = None
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                left, right = clusters[left_index], clusters[right_index]
                merged = {
                    "signatures": set(left["signatures"]) | set(right["signatures"]),
                    "rows": list(left["rows"]) + list(right["rows"]),
                }
                left_cost = _motif_state_cost(
                    left["rows"], code_length, node_penalty, edge_penalty,
                )[0]
                right_cost = _motif_state_cost(
                    right["rows"], code_length, node_penalty, edge_penalty,
                )[0]
                merged_cost = _motif_state_cost(
                    merged["rows"], code_length, node_penalty, edge_penalty,
                )[0]
                saving = left_cost + right_cost - merged_cost
                candidate = (saving, left_index, right_index, merged)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        if best is None or best[0] <= 0.0:
            break
        _, left_index, right_index, merged = best
        clusters = [
            cluster for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ] + [merged]

    base_cost, base_data_bits, _ = _motif_state_cost(
        rows, code_length, node_penalty, edge_penalty,
    )
    refined_cost = sum(
        _motif_state_cost(cluster["rows"], code_length, node_penalty, edge_penalty)[0]
        for cluster in clusters
    )
    if refined_cost >= base_cost:
        clusters = [{"signatures": set(by_signature), "rows": list(rows)}]
        refined_cost = base_cost

    clusters.sort(key=lambda cluster: (-len(cluster["rows"]), sorted(cluster["signatures"])))
    modes: list[dict[str, Any]] = []
    signature_to_mode: dict[str, str] = {}
    refined_data_bits = 0.0
    for index, cluster in enumerate(clusters):
        mode_id = f"{source}#motif{index}"
        counts = Counter(str(row["target"]) for row in cluster["rows"] if row.get("target"))
        support = sum(counts.values())
        entropy = _label_entropy([row for row in cluster["rows"] if row.get("target")])
        refined_data_bits += support * entropy
        signatures = sorted(cluster["signatures"])
        mode = {
            "mode_id": mode_id,
            "base_action": source,
            "motif_signatures": signatures,
            "observed_history_signatures": signatures,
            "support": len(cluster["rows"]),
            "candidate_actions": [target for target, _ in counts.most_common()],
            "target_probabilities": {
                target: round(count / max(support, 1), 8)
                for target, count in counts.most_common()
            },
            "entropy": round(entropy, 8),
        }
        modes.append(mode)
        for signature in signatures:
            signature_to_mode[signature] = mode_id
    return modes, signature_to_mode, {
        "base_data_bits": base_data_bits,
        "refined_data_bits": refined_data_bits,
        "local_mdl_gain_bits": max(base_cost - refined_cost, 0.0),
    }


def route_motif_state(
    router: dict[str, Any], history: list[str], history_order: int = 3,
) -> dict[str, Any]:
    """Route a prefix to a learned motif role, backing off on unseen motifs."""
    signature = motif_history_signature(history, history_order)
    modes = {str(mode.get("mode_id")): mode for mode in router.get("modes") or []}
    mode_id = str((router.get("signature_to_mode") or {}).get(signature) or "")
    if mode_id in modes:
        return {"kind": "motif", "mode_id": mode_id, "exact": True, "signature": signature}
    default_mode = max(
        modes.values(), key=lambda mode: (int(mode.get("support", 0)), str(mode.get("mode_id"))),
        default={},
    )
    return {
        "kind": "motif", "mode_id": str(default_mode.get("mode_id") or ""),
        "exact": False, "signature": signature,
    }


def _build_structural_refined_graph(
    subflow: str,
    conversations: list[dict[str, Any]],
    base_graph: dict[str, Any],
    history_order: int,
    max_depth: int,
    min_leaf_support: int,
    node_penalty: float,
    edge_penalty: float,
) -> dict[str, Any]:
    """Expand canonical actions into a single history-refined process graph."""
    retained_pairs = {
        (str(edge["source"]), str(edge["target"]))
        for edges in (base_graph.get("local_transitions") or {}).values()
        for edge in edges
    }
    occurrences = structural_action_occurrences(subflow, conversations, history_order)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences:
        pair = (str(row["source"]), str(row.get("target") or ""))
        if row.get("target") and pair in retained_pairs:
            by_source[str(row["source"])].append(row)

    routers: dict[str, Any] = {}
    modes_by_id: dict[str, dict[str, Any]] = {}
    for node in base_graph.get("nodes", []):
        source = str(node["id"])
        rows = by_source.get(source) or [{
            "source": source, "target": "", "history": [source],
            "history_signature": (_label(source),), "features": [],
        }]
        tree, modes = _build_structural_tree(
            source, rows, max_depth=max_depth, min_leaf_support=min_leaf_support,
            node_penalty=node_penalty, edge_penalty=edge_penalty,
        )
        routers[source] = {"tree": tree, "modes": modes}
        modes_by_id.update({str(mode["mode_id"]): mode for mode in modes})

    def assign(source: str, history: list[str]) -> str:
        router = routers[source]
        return str(route_structural_tree(router["tree"], history, history_order).get("mode_id"))

    edge_counts: Counter[tuple[str, str]] = Counter()
    edge_action: dict[tuple[str, str], str] = {}
    for conversation in conversations:
        sequence_rows = [
            row for row in structural_action_occurrences(subflow, [conversation], history_order)
        ]
        for left, right in zip(sequence_rows, sequence_rows[1:]):
            pair = (str(left["source"]), str(right["source"]))
            if pair not in retained_pairs:
                continue
            source_state = assign(str(left["source"]), list(left["history"]))
            target_state = assign(str(right["source"]), list(right["history"]))
            edge_counts[(source_state, target_state)] += 1
            edge_action[(source_state, target_state)] = str(right["source"])

    outgoing_totals: Counter[str] = Counter()
    for (source_state, _), count in edge_counts.items():
        outgoing_totals[source_state] += count
    refined_edges = [
        {
            "source": source_state,
            "target": target_state,
            "action": edge_action[(source_state, target_state)],
            "support": count,
            "probability": round(count / max(outgoing_totals[source_state], 1), 8),
        }
        for (source_state, target_state), count in sorted(edge_counts.items())
    ]
    refined_edges.sort(key=lambda edge: (edge["source"], -edge["probability"], edge["target"]))

    transition_rows = [row for rows in by_source.values() for row in rows if row.get("target")]
    coarse_entropy_sum = sum(
        len(rows) * _label_entropy([row for row in rows if row.get("target")])
        for rows in by_source.values() if any(row.get("target") for row in rows)
    )
    refined_entropy_sum = sum(
        int(mode["support"]) * float(mode["entropy"])
        for mode in modes_by_id.values() if mode.get("candidate_actions")
    )
    denominator = max(len(transition_rows), 1)
    coarse_entropy = coarse_entropy_sum / denominator
    refined_entropy = refined_entropy_sum / denominator
    extra_nodes = sum(max(len(router["modes"]) - 1, 0) for router in routers.values())
    base_nodes = len(base_graph.get("nodes", []))
    base_edges = len(retained_pairs)
    refined_node_count = len(modes_by_id)
    refined_edge_count = len(refined_edges)
    corpus_code_length = math.log2(len(transition_rows) + 1)
    base_data_bits = len(transition_rows) * coarse_entropy
    refined_data_bits = len(transition_rows) * refined_entropy
    incremental_model_bits = (
        node_penalty * extra_nodes
        + edge_penalty * max(refined_edge_count - base_edges, 0)
    ) * corpus_code_length
    return {
        "format": "structural_state_refined_graph_v1",
        "history_order": history_order,
        "max_depth": max_depth,
        "min_leaf_support": min_leaf_support,
        "node_penalty": node_penalty,
        "edge_penalty": edge_penalty,
        "routers": routers,
        "nodes": sorted(modes_by_id.values(), key=lambda row: str(row["mode_id"])),
        "edges": refined_edges,
        "base_nodes": base_nodes,
        "base_edges": base_edges,
        "refined_nodes": refined_node_count,
        "refined_edges": refined_edge_count,
        "extra_state_nodes": extra_nodes,
        "base_structural_units": base_nodes + base_edges,
        "refined_structural_units": refined_node_count + refined_edge_count,
        "num_structural_decisions": len(transition_rows),
        "base_data_code_bits": round(base_data_bits, 4),
        "refined_data_code_bits": round(refined_data_bits, 4),
        "incremental_model_code_bits": round(incremental_model_bits, 4),
        "mdl_net_gain_bits": round(base_data_bits - refined_data_bits - incremental_model_bits, 4),
        "coarse_routing_entropy": round(coarse_entropy, 8),
        "refined_routing_entropy": round(refined_entropy, 8),
        "structural_ambiguity_reduction": round(
            1.0 - refined_entropy / coarse_entropy if coarse_entropy > 0 else 0.0,
            8,
        ),
    }


def _build_motif_refined_graph(
    subflow: str,
    conversations: list[dict[str, Any]],
    base_graph: dict[str, Any],
    history_order: int,
    min_motif_support: int,
    node_penalty: float,
    edge_penalty: float,
) -> dict[str, Any]:
    """Split merged action nodes into MDL-compressed causal motif roles."""
    retained_pairs = {
        (str(edge["source"]), str(edge["target"]))
        for edges in (base_graph.get("local_transitions") or {}).values()
        for edge in edges
    }
    occurrences = motif_action_occurrences(subflow, conversations, history_order)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences:
        pair = (str(row["source"]), str(row.get("target") or ""))
        if row.get("target") and pair in retained_pairs:
            by_source[str(row["source"])].append(row)

    routers: dict[str, Any] = {}
    modes_by_id: dict[str, dict[str, Any]] = {}
    base_data_bits = 0.0
    refined_data_bits = 0.0
    local_mdl_gain_bits = 0.0
    for node in base_graph.get("nodes", []):
        source = str(node["id"])
        rows = by_source.get(source) or [{
            "source": source, "target": "", "history": [source],
            "motif_signature": motif_history_signature([source], history_order),
        }]
        modes, signature_to_mode, statistics = _cluster_action_motifs(
            source, rows, min_motif_support=min_motif_support,
            node_penalty=node_penalty, edge_penalty=edge_penalty,
        )
        if len(modes) == 1:
            base_order = [
                str(edge["target"])
                for edge in (base_graph.get("local_transitions") or {}).get(source, [])
            ]
            if base_order:
                modes[0]["candidate_actions"] = base_order
        routers[source] = {
            "modes": modes,
            "signature_to_mode": signature_to_mode,
        }
        modes_by_id.update({str(mode["mode_id"]): mode for mode in modes})
        base_data_bits += statistics["base_data_bits"]
        refined_data_bits += statistics["refined_data_bits"]
        local_mdl_gain_bits += statistics["local_mdl_gain_bits"]

    def assign(source: str, history: list[str]) -> str:
        return str(route_motif_state(
            routers[source], history, history_order,
        ).get("mode_id") or "")

    edge_counts: Counter[tuple[str, str]] = Counter()
    edge_action: dict[tuple[str, str], str] = {}
    for conversation in conversations:
        sequence_rows = motif_action_occurrences(subflow, [conversation], history_order)
        for left, right in zip(sequence_rows, sequence_rows[1:]):
            pair = (str(left["source"]), str(right["source"]))
            if pair not in retained_pairs:
                continue
            source_state = assign(str(left["source"]), list(left["history"]))
            target_state = assign(str(right["source"]), list(right["history"]))
            if not source_state or not target_state:
                continue
            edge_counts[(source_state, target_state)] += 1
            edge_action[(source_state, target_state)] = str(right["source"])

    outgoing_totals: Counter[str] = Counter()
    for (source_state, _), count in edge_counts.items():
        outgoing_totals[source_state] += count
    refined_edges = [
        {
            "source": source_state,
            "target": target_state,
            "action": edge_action[(source_state, target_state)],
            "support": count,
            "probability": round(count / max(outgoing_totals[source_state], 1), 8),
        }
        for (source_state, target_state), count in sorted(edge_counts.items())
    ]
    refined_edges.sort(key=lambda edge: (edge["source"], -edge["probability"], edge["target"]))

    transition_rows = [row for rows in by_source.values() for row in rows if row.get("target")]
    denominator = max(len(transition_rows), 1)
    coarse_entropy = base_data_bits / denominator
    refined_entropy = refined_data_bits / denominator
    base_nodes = len(base_graph.get("nodes", []))
    base_edges = len(retained_pairs)
    refined_node_count = len(modes_by_id)
    refined_edge_count = len(refined_edges)
    extra_nodes = max(refined_node_count - base_nodes, 0)
    corpus_code_length = math.log2(len(transition_rows) + 1)
    incremental_model_bits = (
        node_penalty * extra_nodes
        + edge_penalty * max(refined_edge_count - base_edges, 0)
    ) * corpus_code_length
    return {
        "format": "causal_motif_refined_graph_v1",
        "history_order": history_order,
        "min_motif_support": min_motif_support,
        "node_penalty": node_penalty,
        "edge_penalty": edge_penalty,
        "routers": routers,
        "nodes": sorted(modes_by_id.values(), key=lambda row: str(row["mode_id"])),
        "edges": refined_edges,
        "base_nodes": base_nodes,
        "base_edges": base_edges,
        "refined_nodes": refined_node_count,
        "refined_edges": refined_edge_count,
        "extra_state_nodes": extra_nodes,
        "base_structural_units": base_nodes + base_edges,
        "refined_structural_units": refined_node_count + refined_edge_count,
        "num_structural_decisions": len(transition_rows),
        "base_data_code_bits": round(base_data_bits, 4),
        "refined_data_code_bits": round(refined_data_bits, 4),
        "incremental_model_code_bits": round(incremental_model_bits, 4),
        "mdl_net_gain_bits": round(
            base_data_bits - refined_data_bits - incremental_model_bits, 4,
        ),
        "local_mdl_gain_bits": round(local_mdl_gain_bits, 4),
        "num_split_actions": sum(
            len(router.get("modes") or []) > 1 for router in routers.values()
        ),
        "coarse_routing_entropy": round(coarse_entropy, 8),
        "refined_routing_entropy": round(refined_entropy, 8),
        "structural_ambiguity_reduction": round(
            1.0 - refined_entropy / coarse_entropy if coarse_entropy > 0 else 0.0, 8,
        ),
    }


def mine_backbone_workflow_trace_cover(
    subflow: str,
    conversations: list[dict[str, Any]],
    min_branch_support: int = 2,
    trace_coverage_target: float = 0.8,
    corpus_fitness_target: float = 0.95,
    routing_complexity_weight: float = 2.0,
    observable_ambiguity: bool = False,
    routing_mode_max_depth: int = 2,
    routing_mode_min_leaf_support: int = 12,
    routing_mode_min_information_gain: float = 0.10,
    measure_observable_distinction: bool = True,
) -> dict[str, Any]:
    """Mine a compact graph by saturated trace-cover maximization.

    This follows the process-discovery principle that a useful model should
    balance replay fitness and simplicity.  Starting from the support/lift
    arborescence, each residual edge is charged one unit of complexity and is
    selected by its marginal gain in mean saturated per-trace coverage.  The
    procedure stops at the requested corpus fitness, producing the greedy
    set-cover approximation to the smallest adequate residual graph.
    """
    if not 0.0 < trace_coverage_target <= 1.0:
        raise ValueError("trace_coverage_target must be in (0, 1]")
    if not 0.0 < corpus_fitness_target <= 1.0:
        raise ValueError("corpus_fitness_target must be in (0, 1]")
    if routing_complexity_weight < 0.0:
        raise ValueError("routing_complexity_weight must be non-negative")
    # Build the complete DFG and a stable organizational skeleton.  Residual
    # selection below replaces the fixed per-source top-k rule.
    base = _mine_backbone_workflow_support_lift(
        subflow,
        conversations,
        max_outgoing_edges=10**9,
        min_branch_support=min_branch_support,
    )
    graph = base["subgraph"]
    edge_rows = {
        (str(edge["source"]), str(edge["target"])): edge
        for edge in graph.get("edges", [])
    }
    backbone_pairs = {
        (str(edge["source"]), str(edge["target"]))
        for edge in graph.get("backbone", {}).get("edges", [])
        if str(edge.get("source")) != ROOT
    }
    sessions = _session_edge_occurrences(subflow, conversations)
    observable_events = (
        observable_transition_events(subflow, conversations)
        if observable_ambiguity or measure_observable_distinction else []
    )
    edge_observations = _edge_observation_index(observable_events)
    totals = [sum(counts.values()) for counts in sessions]
    saturation = [trace_coverage_target * total for total in totals]
    covered = [
        sum(count for edge, count in counts.items() if edge in backbone_pairs)
        for counts in sessions
    ]

    def utility(index: int, value: float | None = None) -> float:
        amount = covered[index] if value is None else value
        return min(amount / max(saturation[index], 1e-12), 1.0)

    def mean_fitness() -> float:
        return sum(utility(index) for index in range(len(sessions))) / max(len(sessions), 1)

    candidate_sessions: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for index, counts in enumerate(sessions):
        for edge, count in counts.items():
            row = edge_rows.get(edge)
            if edge not in backbone_pairs and row is not None and int(row.get("support", 0)) >= min_branch_support:
                candidate_sessions[edge].append((index, count))

    selected = set(backbone_pairs)
    source_occurrences: Counter[str] = Counter()
    for counts in sessions:
        for (source, _target), count in counts.items():
            source_occurrences[source] += count
    total_source_occurrences = max(sum(source_occurrences.values()), 1)
    selected_out_degree: Counter[str] = Counter(source for source, _target in selected)
    selection_history: list[dict[str, Any]] = []
    remaining = set(candidate_sessions)
    while remaining and mean_fitness() + 1e-12 < corpus_fitness_target:
        best_edge: tuple[str, str] | None = None
        best_key = (-1.0, -1.0, -1, -math.inf, "", "")
        for edge in remaining:
            gain = 0.0
            for index, count in candidate_sessions[edge]:
                gain += utility(index, covered[index] + count) - utility(index)
            gain /= max(len(sessions), 1)
            row = edge_rows[edge]
            source_probability = source_occurrences[edge[0]] / total_source_occurrences
            # The original TraceCover assumes every new sibling pair is fully
            # ambiguous. Observable TraceCover replaces each unit pair cost by
            # empirical pre-action state overlap: 1 - best feature-rate gap.
            selected_siblings = [pair for pair in selected if pair[0] == edge[0]]
            if observable_ambiguity:
                ambiguity_sum = sum(
                    float(_observable_distinction(
                        edge_observations.get(edge, []),
                        edge_observations.get(sibling, []),
                    )["ambiguity"])
                    for sibling in selected_siblings
                )
            else:
                ambiguity_sum = float(selected_out_degree[edge[0]])
            sibling_pair_cost = source_probability * ambiguity_sum
            marginal_complexity = 1.0 + routing_complexity_weight * sibling_pair_cost
            key = (
                gain / marginal_complexity,
                gain,
                int(row.get("num_sessions", 0)),
                float(row.get("score", 0.0)),
                edge[0],
                edge[1],
            )
            if key > best_key:
                best_key, best_edge = key, edge
        if best_edge is None or best_key[0] <= 0.0:
            break
        before = mean_fitness()
        selected.add(best_edge)
        remaining.remove(best_edge)
        selected_out_degree[best_edge[0]] += 1
        for index, count in candidate_sessions[best_edge]:
            covered[index] += count
        selection_history.append({
            "edge_id": f"{best_edge[0]}=>{best_edge[1]}",
            "gain_per_complexity": round(best_key[0], 8),
            "raw_fitness_gain": round(best_key[1], 8),
            "fitness_before": round(before, 8),
            "fitness_after": round(mean_fitness(), 8),
        })

    parent = {
        str(edge["target"]): str(edge["source"])
        for edge in graph.get("backbone", {}).get("edges", [])
    }
    selected_order = {
        tuple(item["edge_id"].split("=>", 1)): index
        for index, item in enumerate(selection_history)
    }
    local: dict[str, list[dict[str, Any]]] = {}
    residual: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        source = str(node["id"])
        outgoing = [
            edge_rows[pair]
            for pair in selected
            if pair[0] == source and pair in edge_rows
        ]
        outgoing.sort(key=lambda edge: (
            (str(edge["source"]), str(edge["target"])) not in backbone_pairs,
            selected_order.get((str(edge["source"]), str(edge["target"])), 10**9),
            -float(edge.get("score", 0.0)),
            str(edge["target"]),
        ))
        rows = []
        for edge in outgoing:
            pair = (str(edge["source"]), str(edge["target"]))
            is_backbone = pair in backbone_pairs
            kind = "backbone" if is_backbone else (
                "retry" if _has_path(parent, source, str(edge["target"])) else "branch"
            )
            item = {**edge, "kind": kind}
            rows.append(item)
            if not is_backbone:
                residual.append(item)
        for priority, edge in enumerate(rows, 1):
            edge["priority"] = priority
        local[source] = rows

    retained_occurrences = sum(
        sum(count for edge, count in counts.items() if edge in selected)
        for counts in sessions
    )
    total_occurrences = sum(totals)
    graph["local_transitions"] = local
    graph["residual_edges"] = residual
    graph["coverage_pct"] = round(100 * retained_occurrences / max(total_occurrences, 1), 1)
    graph["mining_method"] = (
        "observable_trace_cover" if observable_ambiguity else "saturated_trace_cover"
    )
    observable_pair_rows: list[dict[str, Any]] = []
    if observable_events:
        selected_by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for pair in selected:
            selected_by_source[pair[0]].append(pair)
        for source, pairs in selected_by_source.items():
            for left_index, left in enumerate(sorted(pairs)):
                for right in sorted(pairs)[left_index + 1:]:
                    distinction = _observable_distinction(
                        edge_observations.get(left, []), edge_observations.get(right, []),
                    )
                    observable_pair_rows.append({
                        "source": source,
                        "left_target": left[1],
                        "right_target": right[1],
                        **distinction,
                    })
    if observable_ambiguity:
        router = _build_observable_router(
            observable_events, selected,
            max_depth=routing_mode_max_depth,
            min_leaf_support=routing_mode_min_leaf_support,
            min_information_gain=routing_mode_min_information_gain,
        )
        graph["observable_router"] = router
        by_edge: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in observable_pair_rows:
            left = (row["source"], row["left_target"])
            right = (row["source"], row["right_target"])
            by_edge[left].append({**row, "sibling_target": row["right_target"]})
            by_edge[right].append({
                **row,
                "sibling_target": row["left_target"],
                "left_rate": row["right_rate"],
                "right_rate": row["left_rate"],
                "preferred_for": "left" if row["preferred_for"] == "right" else "right",
            })
        for edges in local.values():
            for edge in edges:
                edge["observable_distinctions"] = by_edge.get(
                    (str(edge["source"]), str(edge["target"])), [],
                )
    graph["trace_cover"] = {
        "trace_coverage_target": trace_coverage_target,
        "corpus_fitness_target": corpus_fitness_target,
        "routing_complexity_weight": routing_complexity_weight,
        "initial_backbone_fitness": round(
            sum(
                min(
                    sum(count for edge, count in counts.items() if edge in backbone_pairs)
                    / max(trace_coverage_target * total, 1e-12),
                    1.0,
                )
                for counts, total in zip(sessions, totals)
            ) / max(len(sessions), 1),
            8,
        ),
        "final_fitness": round(mean_fitness(), 8),
        "selected_residual_edges": len(residual),
        "candidate_residual_edges": len(candidate_sessions),
        "routing_pair_complexity": round(sum(
            source_occurrences[source] / total_source_occurrences
            * degree * max(degree - 1, 0) / 2
            for source, degree in selected_out_degree.items()
        ), 8),
        "observable_ambiguity_enabled": observable_ambiguity,
        "mean_selected_pair_ambiguity": round(
            sum(float(row["ambiguity"]) for row in observable_pair_rows)
            / max(len(observable_pair_rows), 1),
            8,
        ),
        "mean_selected_pair_distinguishability": round(
            sum(float(row["gap"]) for row in observable_pair_rows)
            / max(len(observable_pair_rows), 1),
            8,
        ),
        "observable_pair_distinctions": observable_pair_rows,
        "selection_history": selection_history,
    }
    base["skill_info"]["mining_method"] = graph["mining_method"]
    base["skill_info"]["coverage_pct"] = graph["coverage_pct"]
    return base


def mine_backbone_workflow_observable_trace_cover(
    subflow: str,
    conversations: list[dict[str, Any]],
    min_branch_support: int = 2,
    trace_coverage_target: float = 0.8,
    corpus_fitness_target: float = 0.95,
    routing_complexity_weight: float = 2.0,
    routing_mode_max_depth: int = 2,
    routing_mode_min_leaf_support: int = 12,
    routing_mode_min_information_gain: float = 0.10,
) -> dict[str, Any]:
    """TraceCover with observable sibling ambiguity and learned routing modes."""
    return mine_backbone_workflow_trace_cover(
        subflow,
        conversations,
        min_branch_support=min_branch_support,
        trace_coverage_target=trace_coverage_target,
        corpus_fitness_target=corpus_fitness_target,
        routing_complexity_weight=routing_complexity_weight,
        observable_ambiguity=True,
        routing_mode_max_depth=routing_mode_max_depth,
        routing_mode_min_leaf_support=routing_mode_min_leaf_support,
        routing_mode_min_information_gain=routing_mode_min_information_gain,
    )


def mine_backbone_workflow_structural_trace_cover(
    subflow: str,
    conversations: list[dict[str, Any]],
    min_branch_support: int = 2,
    trace_coverage_target: float = 0.8,
    corpus_fitness_target: float = 0.95,
    routing_complexity_weight: float = 2.0,
    history_order: int = 3,
    state_max_depth: int = 3,
    state_min_leaf_support: int = 8,
    state_node_penalty: float = 1.0,
    state_edge_penalty: float = 2.0,
) -> dict[str, Any]:
    """TraceCover followed by pure action-history MDL state refinement."""
    base = mine_backbone_workflow_trace_cover(
        subflow,
        conversations,
        min_branch_support=min_branch_support,
        trace_coverage_target=trace_coverage_target,
        corpus_fitness_target=corpus_fitness_target,
        routing_complexity_weight=routing_complexity_weight,
        measure_observable_distinction=False,
    )
    graph = base["subgraph"]
    graph["structural_refinement"] = _build_structural_refined_graph(
        subflow,
        conversations,
        graph,
        history_order=history_order,
        max_depth=state_max_depth,
        min_leaf_support=state_min_leaf_support,
        node_penalty=state_node_penalty,
        edge_penalty=state_edge_penalty,
    )
    graph["mining_method"] = "structural_trace_cover"
    base["skill_info"]["mining_method"] = graph["mining_method"]
    return base


def mine_backbone_workflow_motif_trace_cover(
    subflow: str,
    conversations: list[dict[str, Any]],
    min_branch_support: int = 2,
    trace_coverage_target: float = 0.8,
    corpus_fitness_target: float = 0.95,
    routing_complexity_weight: float = 2.0,
    motif_history_order: int = 1,
    motif_min_support: int = 8,
    motif_state_node_penalty: float = 1.0,
    motif_state_edge_penalty: float = 2.0,
) -> dict[str, Any]:
    """TraceCover with occurrence-level causal motif role discovery.

    This is intentionally a separate branch: the original action graph and
    both existing refinements remain unchanged.  The miner first retains the
    TraceCover action graph, then reconstructs occurrence-level prefix motifs
    and MDL-compresses them into multiple roles for the same canonical action.
    """
    base = mine_backbone_workflow_trace_cover(
        subflow,
        conversations,
        min_branch_support=min_branch_support,
        trace_coverage_target=trace_coverage_target,
        corpus_fitness_target=corpus_fitness_target,
        routing_complexity_weight=routing_complexity_weight,
        measure_observable_distinction=False,
    )
    graph = base["subgraph"]
    graph["motif_refinement"] = _build_motif_refined_graph(
        subflow,
        conversations,
        graph,
        history_order=motif_history_order,
        min_motif_support=motif_min_support,
        node_penalty=motif_state_node_penalty,
        edge_penalty=motif_state_edge_penalty,
    )
    graph["mining_method"] = "motif_trace_cover"
    base["skill_info"]["mining_method"] = graph["mining_method"]
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
