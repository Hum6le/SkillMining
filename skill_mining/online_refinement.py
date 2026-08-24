"""Evidence-calibrated online refinement for graph-compiled ABCD skills.

This module deliberately separates deterministic graph maintenance from LLM
language induction.  It maintains a rooted DAG-like control skeleton with
explicit retry/revisit edges, schedules contrastive rollout batches, and
localizes gold-supervised rollout feedback to graph regions.  A later LLM
stage can turn only the selected edge records into natural-language guards.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_tod.abcd.action_schema import canonical_action_name, load_action_schema


ROOT = "<START>"


_GUARD_INDUCTION_PROMPT = """You are refining one local decision in a mined
customer-service skill graph. The graph topology has already been selected by
offline and online evidence. Do not invent actions, edges, state variables, or
customer-specific slot values.

<current_skill>
{skill_context}
</current_skill>

<source_local_evidence>
{edge_context}
</source_local_evidence>

Write one concise natural-language routing guard for the target edge. Compare
every sibling edge from the same source. Use observable dialogue cues, such as
what the customer asks for, what has already been confirmed, or which earlier
step succeeded. A cue must not select multiple siblings. If the evidence does
not support a unique distinction, say so explicitly and mark it unresolved.

Return only JSON:
{{
  "guard": "...",
  "status": "resolved" | "uncertain",
  "rationale": "..."
}}
"""


def _edge_id(source: str, target: str) -> str:
    return f"{source}=>{target}"


def _actions(conversation: dict[str, Any]) -> list[str]:
    schema = load_action_schema()
    actions = []
    for turn in conversation.get("delexed") or []:
        targets = turn.get("targets") or []
        if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
            action, _ = canonical_action_name(targets[2], schema.get("actions"))
            if action:
                actions.append(action)
    return actions


def session_signature(conversation: dict[str, Any]) -> frozenset[str]:
    """Complete node + directed-transition signature used by the scheduler."""
    actions = _actions(conversation)
    return frozenset(
        {f"node:{action}" for action in actions}
        | {_edge_id(source, target) for source, target in zip(actions, actions[1:])}
    )


def _weighted_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    # Edge features represent an action decision, so give them more weight
    # than action-presence context when selecting representative sessions.
    union = left | right
    if not union:
        return 0.0
    weight = lambda feature: 1.5 if "=>" in feature else 1.0
    return sum(weight(feature) for feature in left & right) / sum(weight(feature) for feature in union)


def initialize_skill_dag(subgraph: dict[str, Any], subflow: str) -> dict[str, Any]:
    """Create the online state from an offline backbone mining artifact.

    The offline tree remains immutable in the first online-refinement version.
    Any non-backbone edge begins as reference-only until sufficient online
    evidence and a resolved guard justify promotion.
    """
    nodes = {str(node["id"]): str(node.get("label") or node["id"]) for node in subgraph.get("nodes", [])}
    order = list(subgraph.get("backbone", {}).get("compilation_order", []))
    order_index = {node: index for index, node in enumerate(order)}
    backbone_pairs = {
        (str(edge["source"]), str(edge["target"]))
        for edge in subgraph.get("backbone", {}).get("edges", [])
    }
    state_edges = {}
    for raw_edge in subgraph.get("edges", []):
        source, target = str(raw_edge["source"]), str(raw_edge["target"])
        is_backbone = (source, target) in backbone_pairs
        existing_kind = str(raw_edge.get("kind") or "")
        if is_backbone:
            kind, visibility = "backbone", "skill"
        elif source == target or existing_kind == "retry":
            kind, visibility = "retry", "reference"
        elif order_index.get(target, math.inf) < order_index.get(source, -1):
            kind, visibility = "revisit", "reference"
        else:
            kind, visibility = "candidate_branch", "reference"
        state_edges[_edge_id(source, target)] = {
            "source": source,
            "target": target,
            "source_action": nodes.get(source, source),
            "target_action": nodes.get(target, target),
            "kind": kind,
            "visibility": visibility,
            "offline_support": int(raw_edge.get("support", 0) or 0),
            "offline_sessions": int(raw_edge.get("num_sessions", 0) or 0),
            "gold_support": 0,
            "rollout_success": 0,
            "rollout_failure": 0,
            "slot_failures": 0,
            "competing_targets": Counter(),
            "guard": "",
            "guard_status": "resolved" if is_backbone else "pending",
            "evidence": [],
        }
    return {
        "schema_version": 1,
        "subflow": subflow,
        "root": ROOT,
        "nodes": [{"id": node, "label": label, "topological_order": order_index.get(node)} for node, label in nodes.items()],
        "backbone_order": order,
        "edges": state_edges,
        "batches_processed": 0,
        "patches": [],
    }


def save_skill_dag(state: dict[str, Any], path: Path) -> None:
    """Persist JSON-safe online state, including Counter-backed fields."""
    serializable = deepcopy(state)
    for edge in serializable.get("edges", {}).values():
        edge["competing_targets"] = dict(edge.get("competing_targets", {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


def load_skill_dag(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    for edge in state.get("edges", {}).values():
        edge["competing_targets"] = Counter(edge.get("competing_targets", {}))
    return state


def _source_target_session_index(conversations: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for conversation in conversations:
        actions = _actions(conversation)
        for source, target in set(zip(actions, actions[1:])):
            index[(source, target)].append(conversation)
    return index


def _representatives(conversations: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep a medoid plus diverse sessions; discard redundant duplicates."""
    signatures = [(conversation, session_signature(conversation)) for conversation in conversations]
    unique: dict[frozenset[str], dict[str, Any]] = {}
    for conversation, signature in signatures:
        key = signature
        old = unique.get(key)
        if old is None or str(conversation.get("convo_id", "")) < str(old.get("convo_id", "")):
            unique[key] = conversation
    candidates = [(conversation, signature) for signature, conversation in unique.items()]
    if len(candidates) <= limit:
        return [conversation for conversation, _ in candidates]
    selected = [max(candidates, key=lambda item: sum(_weighted_jaccard(item[1], other[1]) for other in candidates))]
    while len(selected) < limit:
        remaining = [item for item in candidates if item not in selected]
        selected.append(max(remaining, key=lambda item: min(1.0 - _weighted_jaccard(item[1], chosen[1]) for chosen in selected)))
    return [conversation for conversation, _ in selected]


def schedule_contrastive_batches(
    conversations: list[dict[str, Any]], state: dict[str, Any], batch_size: int = 20,
    per_transition_cap: int = 3, max_batches: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Create graph-structured rollout batches without using class labels.

    A batch preferentially pairs distinct targets leaving the same source
    action. Within any one transition motif it keeps only a few structurally
    non-redundant representative sessions. Remaining capacity is filled by
    diverse sessions, so batches contain both contrast and coverage.
    """
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    edge_index = _source_target_session_index(conversations)
    action_to_node = {node["label"]: node["id"] for node in state.get("nodes", [])}
    by_source: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for (source_action, target_action), members in edge_index.items():
        source_node = action_to_node.get(source_action, source_action)
        target_node = action_to_node.get(target_action, target_action)
        edge = state.get("edges", {}).get(_edge_id(source_node, target_node))
        confidence = edge_confidence(edge) if edge else 0.5
        # Multiple targets from a source are inherently contrastive. Low
        # confidence edges are scheduled earlier because they are candidates
        # for online guard refinement.
        by_source[source_action].append((target_action, members, confidence))

    contrast_sets: list[list[dict[str, Any]]] = []
    for source, options in sorted(by_source.items()):
        if len(options) < 2:
            continue
        selected = []
        for _, members, _ in sorted(options, key=lambda item: (item[2], -len(item[1]), item[0])):
            selected.extend(_representatives(members, per_transition_cap))
        if len(selected) >= 2:
            contrast_sets.append(selected)

    batches: list[list[dict[str, Any]]] = []
    used: set[str] = set()
    for group in contrast_sets:
        batch = []
        for conversation in group:
            sid = str(conversation.get("convo_id", "?"))
            if sid not in used and len(batch) < batch_size:
                batch.append(conversation)
                used.add(sid)
        if batch:
            batches.append(batch)

    # Ensure every remaining session is eventually observed. Sessions are
    # ordered by novelty against the active batch, reducing duplicate traces.
    remaining = [conversation for conversation in conversations if str(conversation.get("convo_id", "?")) not in used]
    for batch in batches:
        while len(batch) < batch_size and remaining:
            batch_signatures = [session_signature(item) for item in batch]
            choice = max(
                remaining,
                key=lambda item: min(1.0 - _weighted_jaccard(session_signature(item), other) for other in batch_signatures),
            )
            batch.append(choice)
            remaining.remove(choice)
    while remaining:
        batch = remaining[:batch_size]
        remaining = remaining[batch_size:]
        batches.append(batch)
    return batches[:max_batches] if max_batches else batches


def edge_confidence(edge: dict[str, Any] | None, alpha: float = 1.0, beta: float = 1.0) -> float:
    """Beta-smoothed rollout reliability for a known transition."""
    if edge is None:
        return alpha / (alpha + beta)
    success = int(edge.get("rollout_success", 0) or 0)
    failure = int(edge.get("rollout_failure", 0) or 0)
    return (success + alpha) / (success + failure + alpha + beta)


def _gold_action_rows(conversation: dict[str, Any], turn_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Align direct rollout predictions to gold action turns in one session."""
    schema = load_action_schema()
    rows_by_turn = {int(row["turn_index"]): row for row in turn_results if "turn_index" in row}
    result = []
    for turn_index, turn in enumerate(conversation.get("delexed") or []):
        targets = turn.get("targets") or []
        if len(targets) < 3 or targets[1] != "take_action" or not targets[2]:
            continue
        gold, _ = canonical_action_name(targets[2], schema.get("actions"))
        raw = rows_by_turn.get(turn_index, {})
        predicted, _ = canonical_action_name(raw.get("predicted_action", ""), schema.get("actions"))
        gold_slots = [str(value) for value in (targets[3] if len(targets) > 3 and isinstance(targets[3], list) else [])]
        predicted_slots = [str(value) for value in (raw.get("predicted_slots") or [])]
        result.append({
            "turn_index": turn_index,
            "gold_action": gold,
            "predicted_action": predicted,
            "action_correct": gold == predicted,
            "slot_correct": gold_slots == predicted_slots,
            "context": str(raw.get("context", ""))[-1600:],
            "react_trace": raw.get("react_trace", []),
        })
    return result


def localize_rollout_batch(
    conversations: list[dict[str, Any]], turn_results: list[dict[str, Any]], state: dict[str, Any],
    max_evidence_per_edge: int = 6,
) -> dict[str, Any]:
    """Attribute AST feedback to gold graph edges and competing predictions."""
    by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in turn_results:
        by_conversation[str(row.get("convo_id", "?"))].append(row)
    action_to_node = {node["label"]: node["id"] for node in state.get("nodes", [])}
    events = []
    for conversation in conversations:
        sid = str(conversation.get("convo_id", "?"))
        actions = _gold_action_rows(conversation, by_conversation.get(sid, []))
        for previous, current in zip(actions, actions[1:]):
            source = action_to_node.get(previous["gold_action"], previous["gold_action"])
            target = action_to_node.get(current["gold_action"], current["gold_action"])
            key = _edge_id(source, target)
            edge = state.setdefault("edges", {}).setdefault(key, {
                "source": source, "target": target,
                "source_action": previous["gold_action"], "target_action": current["gold_action"],
                "kind": "candidate_branch", "visibility": "reference", "offline_support": 0,
                "offline_sessions": 0, "gold_support": 0, "rollout_success": 0,
                "rollout_failure": 0, "slot_failures": 0, "competing_targets": Counter(),
                "guard": "", "guard_status": "pending", "evidence": [],
            })
            edge["gold_support"] += 1
            action_ok = bool(previous["action_correct"] and current["action_correct"])
            slot_ok = bool(previous["slot_correct"] and current["slot_correct"])
            if action_ok:
                edge["rollout_success"] += 1
            else:
                edge["rollout_failure"] += 1
            if not slot_ok:
                edge["slot_failures"] += 1
            predicted_target = str(current.get("predicted_action") or "")
            if predicted_target and predicted_target != current["gold_action"]:
                predicted_node = action_to_node.get(predicted_target, predicted_target)
                edge["competing_targets"][predicted_node] += 1
            evidence = {
                "conversation_id": sid,
                "source_turn": previous["turn_index"],
                "target_turn": current["turn_index"],
                "action_success": action_ok,
                "slot_success": slot_ok,
                "predicted_target": predicted_target,
                "context": current["context"],
                "react_trace": current["react_trace"],
            }
            if len(edge["evidence"]) < max_evidence_per_edge:
                edge["evidence"].append(evidence)
            events.append({"edge_id": key, **evidence})
    state["batches_processed"] = int(state.get("batches_processed", 0)) + 1
    return {"events": events, "num_events": len(events)}


@dataclass(frozen=True)
class RefinementPolicy:
    min_gold_support: int = 3
    min_confidence: float = 0.60
    min_conflict_count: int = 2
    max_skill_branches_per_source: int = 3


def propose_refinement_patches(state: dict[str, Any], policy: RefinementPolicy = RefinementPolicy()) -> list[dict[str, Any]]:
    """Select topology/visibility changes; never edits natural-language guards."""
    order = {node: index for index, node in enumerate(state.get("backbone_order", []))}
    active_by_source: Counter[str] = Counter(
        edge["source"] for edge in state.get("edges", {}).values()
        if edge.get("visibility") == "skill" and edge.get("kind") != "backbone"
    )
    patches = []
    for edge_id, edge in sorted(state.get("edges", {}).items()):
        kind, visibility = edge.get("kind"), edge.get("visibility")
        confidence = edge_confidence(edge)
        conflict_count = sum(int(value) for value in edge.get("competing_targets", {}).values())
        forward = order.get(edge["target"], math.inf) >= order.get(edge["source"], -1)
        eligible = (
            kind in {"candidate_branch", "promoted_branch"}
            and forward
            and int(edge.get("gold_support", 0)) >= policy.min_gold_support
            and confidence >= policy.min_confidence
        )
        if eligible and visibility != "skill" and active_by_source[edge["source"]] < policy.max_skill_branches_per_source:
            patches.append({
                "operation": "promote_to_skill",
                "edge_id": edge_id,
                "reason": "supported forward branch with calibrated rollout reliability",
                "confidence": round(confidence, 6),
                "gold_support": edge["gold_support"],
                "requires_guard_induction": edge.get("guard_status") != "resolved",
            })
            if edge.get("guard_status") != "resolved":
                # Promotion eligibility is evidence for usefulness, not proof
                # that the sibling decision is intelligible. Request the local
                # guard in the same batch; the first promotion stays deferred
                # until this guard is resolved.
                patches.append({
                    "operation": "induce_guard",
                    "edge_id": edge_id,
                    "reason": "supported branch needs a sibling-distinguishing guard before promotion",
                    "confidence": round(confidence, 6),
                    "conflict_count": conflict_count,
                    "evidence_ids": [item["conversation_id"] for item in edge.get("evidence", [])],
                })
            active_by_source[edge["source"]] += 1
        elif visibility == "skill" and kind != "backbone" and (
            confidence < policy.min_confidence or conflict_count >= policy.min_conflict_count
        ):
            patches.append({
                "operation": "sink_to_reference",
                "edge_id": edge_id,
                "reason": "low-confidence or conflict-prone branch should not mislead the main skill",
                "confidence": round(confidence, 6),
                "conflict_count": conflict_count,
                "requires_guard_induction": False,
            })
        elif conflict_count >= policy.min_conflict_count or (
            int(edge.get("gold_support", 0)) >= policy.min_gold_support and confidence < policy.min_confidence
        ):
            patches.append({
                "operation": "induce_guard",
                "edge_id": edge_id,
                "reason": "low-confidence or high-conflict edge requires joint sibling-edge explanation",
                "confidence": round(confidence, 6),
                "conflict_count": conflict_count,
                "evidence_ids": [item["conversation_id"] for item in edge.get("evidence", [])],
            })
    return patches


def build_guard_induction_context(state: dict[str, Any], edge_id: str, max_cases: int = 3) -> dict[str, Any]:
    """Prepare a source-local, contrastive context for a later LLM guard call."""
    edge = state.get("edges", {}).get(edge_id)
    if edge is None:
        raise KeyError(f"Unknown edge: {edge_id}")
    siblings = [
        other for other in state.get("edges", {}).values()
        if other.get("source") == edge.get("source") and other.get("target") != edge.get("target")
    ]

    def compact(item: dict[str, Any]) -> dict[str, Any]:
        positives = [case for case in item.get("evidence", []) if case.get("action_success")][:max_cases]
        negatives = [case for case in item.get("evidence", []) if not case.get("action_success")][:max_cases]
        return {
            "edge_id": _edge_id(item["source"], item["target"]),
            "source_action": item.get("source_action"),
            "target_action": item.get("target_action"),
            "kind": item.get("kind"),
            "visibility": item.get("visibility"),
            "guard": item.get("guard"),
            "guard_status": item.get("guard_status"),
            "gold_support": item.get("gold_support", 0),
            "confidence": round(edge_confidence(item), 6),
            "competing_targets": dict(item.get("competing_targets", {})),
            "positive_cases": positives,
            "negative_cases": negatives,
        }

    return {
        "subflow": state.get("subflow"),
        "backbone_order": state.get("backbone_order", []),
        "target_edge": compact(edge),
        "sibling_edges": [compact(item) for item in sorted(siblings, key=lambda item: item["target"])],
        "instruction": (
            "Compare the target edge with every sibling edge from the same source. "
            "Infer only observable, natural-language routing cues from positive and negative cases. "
            "If the cases do not distinguish the targets, retain uncertainty rather than inventing a guard."
        ),
    }


def _parse_guard_response(raw: str) -> dict[str, str]:
    """Extract the deliberately small guard schema from an LLM response."""
    payload = raw.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        payload = "\n".join(lines[1:-1]) if len(lines) >= 3 else ""
    start, end = payload.find("{"), payload.rfind("}")
    if start >= 0 and end > start:
        payload = payload[start:end + 1]
    try:
        parsed = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return {"guard": "", "status": "uncertain", "rationale": "LLM response was not valid JSON."}
    status = str(parsed.get("status", "uncertain")).strip().lower()
    return {
        "guard": str(parsed.get("guard", "")).strip(),
        "status": "resolved" if status == "resolved" and str(parsed.get("guard", "")).strip() else "uncertain",
        "rationale": str(parsed.get("rationale", "")).strip(),
    }


def induce_guard_patches(
    state: dict[str, Any], patches: list[dict[str, Any]], skill_context: str,
    model: str, max_cases: int = 3, response_logger: Any = None,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    """Ask the LLM only for guards selected by deterministic diagnostics.

    The function intentionally does not alter topology or visibility. Those
    decisions remain auditable outputs of :func:`propose_refinement_patches`.
    """
    from llm import chat, resolve_config

    cfg = resolve_config(model=model)
    results = []
    for patch in patches:
        if patch.get("operation") != "induce_guard":
            continue
        edge_id = str(patch["edge_id"])
        context = build_guard_induction_context(state, edge_id, max_cases=max_cases)
        prompt = _GUARD_INDUCTION_PROMPT.format(
            skill_context=skill_context[-12000:] or "[No existing skill text]",
            edge_context=json.dumps(context, ensure_ascii=False, indent=2),
        )
        raw = ""
        parsed = {"guard": "", "status": "uncertain", "rationale": "No guard response."}
        last_error = ""
        for attempt in range(1, max(1, max_retries) + 1):
            try:
                raw = chat(
                    [{"role": "user", "content": prompt}], model=cfg["model"],
                    api_key=cfg["api_key"], base_url=cfg["base_url"], temperature=0.0,
                    response_logger=response_logger,
                )
                parsed = _parse_guard_response(raw)
                if parsed["status"] == "resolved":
                    break
                last_error = parsed["rationale"] or "response did not provide a resolved guard"
            except Exception as exc:  # retain the edge as reference-only
                last_error = repr(exc)
            if attempt < max(1, max_retries):
                time.sleep(float(attempt))
        if parsed["status"] != "resolved" and last_error:
            parsed["rationale"] = last_error
        edge = state["edges"][edge_id]
        edge["guard"] = parsed["guard"]
        edge["guard_status"] = parsed["status"]
        results.append({
            "edge_id": edge_id,
            "prompt": prompt,
            "raw_response": raw,
            **parsed,
        })
    return results


def render_online_resources(state: dict[str, Any]) -> tuple[str, str]:
    """Render promoted guards and deferred branches as separate resources."""
    skill_lines = ["## Online-refined transition guards", ""]
    reference_lines = ["# Online transition evidence", ""]
    for edge_id, edge in sorted(state.get("edges", {}).items()):
        source = edge.get("source_action", edge.get("source"))
        target = edge.get("target_action", edge.get("target"))
        guard = str(edge.get("guard", "")).strip()
        confidence = edge_confidence(edge)
        if edge.get("visibility") == "skill" and edge.get("kind") != "backbone" and guard:
            skill_lines.extend([
                f"- From `{source}`, transition to `{target}` when: {guard}",
            ])
        elif edge.get("visibility") == "reference" and (
            edge.get("gold_support", 0) or edge.get("offline_support", 0)
        ):
            status = "resolved guard" if guard else "deferred: no reliable guard"
            reference_lines.extend([
                f"## {source} -> {target}",
                f"- Status: {status}",
                f"- Online reliability: {confidence:.3f}; gold support: {edge.get('gold_support', 0)}.",
                *([f"- Guard candidate: {guard}"] if guard else []),
                "",
            ])
    if len(skill_lines) == 2:
        skill_lines.append("- No non-backbone transition has met the online promotion criteria yet.")
    return "\n".join(skill_lines).rstrip() + "\n", "\n".join(reference_lines).rstrip() + "\n"


def apply_refinement_patches(state: dict[str, Any], patches: list[dict[str, Any]]) -> None:
    """Apply only deterministic visibility changes; guard patches remain pending."""
    for patch in patches:
        edge = state.get("edges", {}).get(patch.get("edge_id"))
        if edge is None:
            continue
        if patch["operation"] == "promote_to_skill":
            edge["kind"] = "promoted_branch"
            edge["visibility"] = "skill" if edge.get("guard_status") == "resolved" else "reference"
        elif patch["operation"] == "sink_to_reference" and edge.get("kind") != "backbone":
            edge["visibility"] = "reference"
        elif patch["operation"] == "induce_guard":
            edge["guard_status"] = "pending"
    state.setdefault("patches", []).append({"batch": state.get("batches_processed", 0), "patches": patches})
