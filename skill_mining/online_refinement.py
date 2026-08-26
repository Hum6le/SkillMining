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
import re
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_tod.abcd.action_schema import canonical_action_name, load_action_schema


ROOT = "<START>"


def _tokenize_for_lookup(text: str) -> set[str]:
    return {
        token.lower() for token in re.findall(r"[a-zA-Z0-9_'-]+", text)
        if len(token) >= 3
    }


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

_SLOT_POLICY_INDUCTION_PROMPT = """You are refining the slot policy of a
customer-service skill. The action names and gold trajectories are fixed.

For each requested action, infer a concise GENERAL policy for ordered slot
values from the supplied successful and failed rollouts. Explain value source,
availability timing, reuse constraints, and missing-value handling. Do not
copy literal customer values, names, emails, ids, or invent slot names/hidden
state. Do not alter the workflow topology or action choice.

<existing_slot_resource>
{existing_slot_resource}
</existing_slot_resource>

<slot_error_evidence>
{evidence}
</slot_error_evidence>

Return valid JSON only:
{{"policies":[{{"action":"canonical action name","policy":"concise natural-language policy"}}]}}
Return one item for every requested action. If the evidence cannot support a
safe generalization, use an empty policy string for that action.
"""

_JOINT_REFINEMENT_PROMPT = """You are updating one mined customer-service
skill from a batch of rollout feedback. Treat action choice and ordered slot
construction as one joint policy: first decide the correct transition using
observable dialogue evidence, then specify how the selected action obtains
its ordered values.

The supplied `gold_action` and `gold_slots` fields are supervision: compare
them directly against `predicted_action` and `predicted_slots` to diagnose the
failure. Learn a reusable rule from the contrast, not a transcript-specific
answer.

Do not invent actions, edges, hidden state, slot names, or literal customer
values. A routing guard must distinguish its target from every sibling target.
A slot policy must state general value source, availability timing, reuse, and
missing-value behavior. Return an empty string when evidence is insufficient.

<current_skill>
{skill_context}
</current_skill>
<existing_slot_resource>
{slot_resource}
</existing_slot_resource>
<transition_evidence>
{transition_evidence}
</transition_evidence>
<slot_evidence>
{slot_evidence}
</slot_evidence>

Return valid JSON only:
{{
  "guards": [{{"edge_id":"source=>target", "guard":"...", "status":"resolved|uncertain", "rationale":"..."}}],
  "slot_policies": [{{"action":"canonical action", "policy":"...", "status":"resolved|uncertain"}}]
}}
"""

_AUTONOMOUS_RESOURCE_REFLECTION_PROMPT = """You maintain a graph-compiled
customer-service skill and its progressive-disclosure resources. Inspect the
complete rollout-versus-ground-truth trajectories below, then decide yourself
whether any resource needs an update. Do not assume that every batch requires
a change.

Use `transition_guard` only for a listed graph edge and only when it separates
sibling destinations. Use `action_rule` for an action-level procedure error,
`slot_policy` for ordered-value source/timing/reuse errors after the action is
correct, and `reference` for useful but uncertain or exception-only evidence.
Never invent action names, edges, hidden state, slot names, or literal private
values. Learn general rules from the gold/prediction contrast.

The runtime provides an MCP-style local resource lookup. First choose which
small parts of the auxiliary resources are relevant to this batch; retrieved
observations are supplied below. The complete current skill is always visible
because it is the executable control contract. Do not assume an un-retrieved
auxiliary resource says anything in particular.

<current_skill>{skill}</current_skill>
<retrieved_resources>{retrieved_resources}</retrieved_resources>
<graph_edges>{graph_edges}</graph_edges>
<rollout_supervision>
Each record contains the model's final prediction and, when retrieval was
used, only its generated `reference_query`; the dialogue context and gold
action/slots are supplied separately in that same record.
{rollout_supervision}
</rollout_supervision>

Return valid JSON only:
{{"updates":[{{"resource":"transition_guard|action_rule|slot_policy|reference",
"edge_id":"required only for transition_guard", "action":"required only for action_rule/slot_policy",
"content":"concise natural-language replacement or addition", "status":"resolved|uncertain",
"rationale":"grounded in specific rollout-vs-gold evidence"}}]}}
"""

_RESOURCE_LOOKUP_PLANNER_PROMPT = """You are planning local MCP-style
lookups before refining a customer-service skill from rollout-vs-ground-truth
evidence. You may query only these resources:
- `reference`: deferred transition evidence and exception cases.
- `action_rules`: action-level procedure rules.
- `slot_policies`: ordered value-source, timing, and reuse policies.

Choose only the resources needed to diagnose the supplied errors. Queries must
be concise action names, edge names, or dialogue-goal terms. At most 4 lookups
total and at most 2 per resource. The complete current skill is already shown
below, so never request it. Return valid JSON only:
{{"lookups":[{{"resource":"reference|action_rules|slot_policies","query":"concise query","top_k":1}}]}}

<current_skill>{skill}</current_skill>
<rollout_supervision>{rollout_supervision}</rollout_supervision>
<graph_edges>{graph_edges}</graph_edges>
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
            "slot_total": 0,
            "slot_success": 0,
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
        "slot_policies": {},
        "action_rules": {},
        "reference_notes": [],
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
    conversations: list[dict[str, Any]], state: dict[str, Any], batch_size: int = 8,
    per_transition_cap: int = 3, target_selection_rate: float = 0.30,
    max_batches: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Create graph-structured rollout batches without using class labels.

    A batch preferentially pairs distinct targets leaving the same source
    action. Within any one transition motif it keeps only a few structurally
    non-redundant representative sessions. This is intentionally a budgeted
    sampler: unselected sessions are not appended merely for full coverage.
    Full training-set rollout would erase the efficiency benefit of selecting
    contrastive representatives in the first place.
    """
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if not 0.0 < target_selection_rate <= 1.0:
        raise ValueError("target_selection_rate must be in (0, 1]")
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

    # The fixed cap is a floor for sparse transitions, not the complete
    # rollout budget. Expand it when necessary to approach the requested
    # whole-flow sampling rate.
    target_sessions = math.ceil(len(conversations) * target_selection_rate)
    eligible_transitions = sum(len(options) for options in by_source.values() if len(options) >= 2)
    representative_cap = max(per_transition_cap, math.ceil(target_sessions / max(eligible_transitions, 1)))
    rounds_by_source: list[list[list[dict[str, Any]]]] = []
    for source, options in sorted(by_source.items()):
        if len(options) < 2:
            continue
        representatives_by_target = []
        for _, members, _ in sorted(options, key=lambda item: (item[2], -len(item[1]), item[0])):
            representatives_by_target.append(_representatives(members, representative_cap))
        if sum(bool(items) for items in representatives_by_target) >= 2:
            source_rounds = []
            for round_index in range(max(len(items) for items in representatives_by_target)):
                source_round = [items[round_index] for items in representatives_by_target if round_index < len(items)]
                if len(source_round) >= 2:
                    source_rounds.append(source_round)
            if source_rounds:
                rounds_by_source.append(source_rounds)

    batches: list[list[dict[str, Any]]] = []
    used: set[str] = set()
    # Interleave source-local rounds so the budget covers several ambiguity
    # sites rather than exhausting one source before visiting another.
    for round_index in range(max((len(rounds) for rounds in rounds_by_source), default=0)):
        for source_rounds in rounds_by_source:
            if len(used) >= target_sessions:
                break
            if round_index >= len(source_rounds):
                continue
            contrast_batch = [
                conversation for conversation in source_rounds[round_index]
                if str(conversation.get("convo_id", "?")) not in used
            ]
            # A batch without at least two alternatives cannot teach a local
            # sibling-edge distinction; omit it rather than padding it with
            # unrelated sessions.
            if len(contrast_batch) >= 2:
                # Keep the global sampling budget hard up to one unavoidable
                # two-way comparison. A source can have many outgoing edges;
                # taking its complete sibling set could otherwise consume far
                # more than the requested 30% in a single round.
                remaining = target_sessions - len(used)
                # Preserve a small complete sibling set (for example, three
                # alternatives against a remaining budget of two), but trim a
                # high-degree source that would materially overshoot it.
                if len(contrast_batch) > remaining + 1:
                    contrast_batch = contrast_batch[:max(2, remaining)]
                used.update(str(conversation.get("convo_id", "?")) for conversation in contrast_batch)
                for index in range(0, len(contrast_batch), batch_size):
                    chunk = contrast_batch[index:index + batch_size]
                    if len(chunk) >= 2:
                        batches.append(chunk)
        if len(used) >= target_sessions:
            break

    # Contrastive coverage is the high-value part of the schedule, but it may
    # cover only a small fraction of a flow whose graph has many singleton
    # transitions. Fill the remaining budget with diverse unused sessions so
    # target_selection_rate is an actual whole-flow budget rather than merely
    # an upper bound for eligible sibling groups. These supplemental sessions
    # are placed in separate bounded batches and are never used to fabricate a
    # sibling comparison.
    remaining = target_sessions - len(used)
    if remaining > 0:
        unused = [
            conversation for conversation in conversations
            if str(conversation.get("convo_id", "?")) not in used
        ]
        supplemental = _representatives(unused, remaining)
        for index in range(0, len(supplemental), batch_size):
            chunk = supplemental[index:index + batch_size]
            if chunk:
                batches.append(chunk)
                used.update(str(conversation.get("convo_id", "?")) for conversation in chunk)

    # A flow with no source-local alternative still needs a bounded probe set.
    # Select diverse representatives, not every session.
    if not batches and conversations:
        probe = _representatives(conversations, min(batch_size, target_sessions))
        if probe:
            batches = [probe]
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
            "gold_slot_count": len(gold_slots),
            "predicted_slot_count": len(predicted_slots),
            "gold_slots": gold_slots,
            "predicted_slots": predicted_slots,
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
            # This record evaluates the decision ``previous.gold_action ->
            # current.gold_action``. The source action is provided by the
            # gold trajectory during teacher-forced rollout context, so
            # requiring it to be predicted correctly again turns edge
            # reliability into the product of two independent action scores.
            # Attribute success to the target decision only.
            action_ok = bool(current["action_correct"])
            slot_ok = bool(current["slot_correct"])
            if action_ok:
                edge["rollout_success"] += 1
            else:
                edge["rollout_failure"] += 1
            # Slot quality is meaningful only when the selected action is
            # correct. Otherwise the values belong to another schema and must
            # not be blamed on this action's slot policy.
            slot_evaluable = action_ok
            if slot_evaluable:
                edge["slot_total"] += 1
                if slot_ok:
                    edge["slot_success"] += 1
            if slot_evaluable and not slot_ok:
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
                "slot_evaluable": slot_evaluable,
                "gold_action": current["gold_action"],
                "predicted_action": current["predicted_action"],
                "gold_slots": current["gold_slots"],
                "predicted_slots": current["predicted_slots"],
                "gold_slot_count": current["gold_slot_count"],
                "predicted_slot_count": current["predicted_slot_count"],
                "predicted_target": predicted_target,
                "context": current["context"],
                "react_trace": current["react_trace"],
            }
            if len(edge["evidence"]) < max_evidence_per_edge:
                edge["evidence"].append(evidence)
            events.append({"edge_id": key, **evidence})
            if slot_evaluable:
                policy = state.setdefault("slot_policies", {}).setdefault(current["gold_action"], {
                    "action": current["gold_action"], "slot_total": 0, "slot_success": 0,
                    "slot_failures": 0, "evidence": [], "policy": "", "status": "pending",
                })
                policy["slot_total"] += 1
                if slot_ok:
                    policy["slot_success"] += 1
                else:
                    policy["slot_failures"] += 1
                if len(policy["evidence"]) < max_evidence_per_edge:
                    policy["evidence"].append(evidence)
    state["batches_processed"] = int(state.get("batches_processed", 0)) + 1
    return {"events": events, "num_events": len(events)}


@dataclass(frozen=True)
class RefinementPolicy:
    min_gold_support: int = 3
    min_confidence: float = 0.60
    min_conflict_count: int = 2
    max_skill_branches_per_source: int = 3
    min_slot_support: int = 3
    min_slot_confidence: float = 0.70


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
    for action, slot_policy in sorted(state.get("slot_policies", {}).items()):
        total = int(slot_policy.get("slot_total", 0) or 0)
        failures = int(slot_policy.get("slot_failures", 0) or 0)
        confidence = (int(slot_policy.get("slot_success", 0) or 0) + 1) / (total + 2)
        if total >= policy.min_slot_support and failures and (
            confidence < policy.min_slot_confidence or not str(slot_policy.get("policy", "")).strip()
        ):
            patches.append({
                "operation": "induce_slot_policy",
                "action": action,
                "reason": "action is correct but ordered slots are unreliable; refine value-source and reuse policy",
                "slot_confidence": round(confidence, 6),
                "slot_total": total,
                "slot_failures": failures,
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


def induce_slot_policy_patches(
    state: dict[str, Any], patches: list[dict[str, Any]], existing_slot_resource: str,
    model: str, max_retries: int = 3,
) -> list[dict[str, Any]]:
    """Induce all newly diagnosed action-level slot policies in one LLM call.

    Unlike edge guards, slot policies are action-centric: a single policy can
    repair the same action after several incoming transitions. Batching them
    mirrors AWM's trajectory reflection while keeping online call cost bounded.
    """
    requested = sorted({str(patch["action"]) for patch in patches if patch.get("operation") == "induce_slot_policy"})
    if not requested:
        return []
    evidence = []
    for action in requested:
        record = state.get("slot_policies", {}).get(action, {})
        evidence.append({
            "action": action,
            "slot_total": record.get("slot_total", 0),
            "slot_success": record.get("slot_success", 0),
            "slot_failures": record.get("slot_failures", 0),
            "cases": record.get("evidence", [])[-6:],
        })
    prompt = _SLOT_POLICY_INDUCTION_PROMPT.format(
        existing_slot_resource=existing_slot_resource[-10000:] or "[No existing slot resource]",
        evidence=json.dumps(evidence, ensure_ascii=False, indent=2),
    )
    from llm import chat, resolve_config

    cfg = resolve_config(model=model)
    raw = ""
    parsed: dict[str, Any] = {}
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            raw = chat(
                [{"role": "user", "content": prompt}], model=cfg["model"],
                api_key=cfg["api_key"], base_url=cfg["base_url"], temperature=0.0,
            ).strip()
            payload = raw.strip()
            if payload.startswith("```"):
                payload = "\n".join(payload.splitlines()[1:-1])
            start, end = payload.find("{"), payload.rfind("}")
            parsed = json.loads(payload[start:end + 1]) if start >= 0 and end > start else {}
            if isinstance(parsed.get("policies"), list):
                break
        except (ValueError, TypeError, json.JSONDecodeError):
            parsed = {}
        if attempt < max(1, max_retries):
            time.sleep(float(attempt))

    returned = {
        str(item.get("action", "")).strip(): str(item.get("policy", "")).strip()
        for item in parsed.get("policies", []) if isinstance(item, dict)
    }
    results = []
    for action in requested:
        policy_text = returned.get(action, "")
        record = state["slot_policies"][action]
        if policy_text:
            record["policy"] = policy_text
            record["status"] = "resolved"
        else:
            record["status"] = "uncertain"
        results.append({"action": action, "policy": policy_text, "status": record["status"], "prompt": prompt, "raw_response": raw})
    return results


def induce_joint_refinement_patches(
    state: dict[str, Any], patches: list[dict[str, Any]], skill_context: str,
    existing_slot_resource: str, model: str, max_retries: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Jointly refine transition guards and slot policies in one batch call."""
    edge_ids = sorted({str(patch["edge_id"]) for patch in patches if patch.get("operation") == "induce_guard"})
    actions = sorted({str(patch["action"]) for patch in patches if patch.get("operation") == "induce_slot_policy"})
    if not edge_ids and not actions:
        return {"guards": [], "slot_policies": []}
    transition_evidence = [build_guard_induction_context(state, edge_id, max_cases=3) for edge_id in edge_ids]
    slot_evidence = [{
        "action": action,
        "slot_total": state["slot_policies"][action].get("slot_total", 0),
        "slot_success": state["slot_policies"][action].get("slot_success", 0),
        "slot_failures": state["slot_policies"][action].get("slot_failures", 0),
        "cases": state["slot_policies"][action].get("evidence", [])[-6:],
    } for action in actions]
    prompt = _JOINT_REFINEMENT_PROMPT.format(
        skill_context=skill_context[-12000:] or "[No existing skill text]",
        slot_resource=existing_slot_resource[-10000:] or "[No existing slot resource]",
        transition_evidence=json.dumps(transition_evidence, ensure_ascii=False, indent=2),
        slot_evidence=json.dumps(slot_evidence, ensure_ascii=False, indent=2),
    )
    from llm import chat, resolve_config
    cfg = resolve_config(model=model)
    raw, payload, last_error = "", {}, ""
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            raw = chat([{"role": "user", "content": prompt}], model=cfg["model"], api_key=cfg["api_key"],
                       base_url=cfg["base_url"], temperature=0.0).strip()
            text = raw
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1])
            start, end = text.find("{"), text.rfind("}")
            payload = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
            if isinstance(payload.get("guards"), list) and isinstance(payload.get("slot_policies"), list):
                break
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        if attempt < max(1, max_retries):
            time.sleep(float(attempt))

    guards, slot_results = [], []
    guard_by_id = {str(item.get("edge_id", "")): item for item in payload.get("guards", []) if isinstance(item, dict)}
    for edge_id in edge_ids:
        item = guard_by_id.get(edge_id, {})
        parsed = _parse_guard_response(json.dumps(item)) if item else {"guard": "", "status": "uncertain", "rationale": "No joint guard returned."}
        edge = state["edges"][edge_id]
        edge["guard"], edge["guard_status"] = parsed["guard"], parsed["status"]
        guards.append({"edge_id": edge_id, **parsed, "prompt": prompt, "raw_response": raw})
    policy_by_action = {str(item.get("action", "")): item for item in payload.get("slot_policies", []) if isinstance(item, dict)}
    for action in actions:
        item = policy_by_action.get(action, {})
        policy_text = str(item.get("policy", "")).strip()
        status = "resolved" if policy_text and str(item.get("status", "")).lower() == "resolved" else "uncertain"
        record = state["slot_policies"][action]
        if status == "resolved":
            record["policy"] = policy_text
        record["status"] = status
        slot_results.append({"action": action, "policy": policy_text, "status": status, "prompt": prompt, "raw_response": raw})
    return {"guards": guards, "slot_policies": slot_results}


def _resource_lookup_sections(resource: str, text: str, query: str, top_k: int = 1) -> list[dict[str, str]]:
    """Small local MCP lookup over Markdown sections for optimizer reflection."""
    tokens = _tokenize_for_lookup(query)
    matches = list(re.finditer(r"(?m)^(?:#{1,4})\s+.*$", text))
    sections = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.start():end].strip()
        if body:
            score = len(tokens & _tokenize_for_lookup(body))
            sections.append((score, match.group(0).lstrip("# ").strip(), body[:1800]))
    if not sections and text.strip():
        sections = [(0, resource, text[:1800])]
    return [
        {"resource": resource, "title": title, "content": body}
        for _, title, body in sorted(sections, key=lambda item: (-item[0], item[1]))[:max(1, top_k)]
    ]


def _plan_resource_lookups(
    compact_supervision: list[dict[str, Any]], graph_edges: list[dict[str, Any]],
    skill: str, model: str, max_retries: int,
) -> tuple[list[dict[str, Any]], str, str]:
    """Ask the optimizer which resources it wants before exposing contents."""
    prompt = _RESOURCE_LOOKUP_PLANNER_PROMPT.format(
        skill=skill or "[empty]",
        rollout_supervision=json.dumps(compact_supervision, ensure_ascii=False, indent=2),
        graph_edges=json.dumps(graph_edges, ensure_ascii=False),
    )
    from llm import chat, resolve_config
    cfg = resolve_config(model=model)
    raw, payload, last_error = "", {}, ""
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            raw = chat([{"role": "user", "content": prompt}], model=cfg["model"], api_key=cfg["api_key"],
                       base_url=cfg["base_url"], temperature=0.0).strip()
            start, end = raw.find("{"), raw.rfind("}")
            payload = json.loads(raw[start:end + 1]) if start >= 0 and end > start else {}
            if isinstance(payload.get("lookups"), list):
                break
        except Exception as exc:
            last_error = repr(exc)
        if attempt < max(1, max_retries):
            time.sleep(float(2 ** (attempt - 1)))
    allowed, planned = {"reference", "action_rules", "slot_policies"}, []
    per_resource: Counter[str] = Counter()
    for item in payload.get("lookups", []):
        if not isinstance(item, dict):
            continue
        resource, query = str(item.get("resource", "")), str(item.get("query", "")).strip()
        if resource in allowed and query and per_resource[resource] < 2 and len(planned) < 4:
            planned.append({"resource": resource, "query": query[:300], "top_k": max(1, min(2, int(item.get("top_k", 1) or 1)))})
            per_resource[resource] += 1
    return planned, prompt, last_error


def autonomous_resource_reflection(
    state: dict[str, Any], rollout_supervision: list[dict[str, Any]], skill: str,
    reference: str, action_rules: str, slot_policies: str, model: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Let the LLM select and apply bounded resource updates for one batch."""
    def reference_query_from_trace(trace: Any) -> str:
        """Keep only the model-produced retrieval query, never ReAct inputs."""
        for step in trace if isinstance(trace, list) else []:
            if not isinstance(step, dict):
                continue
            if step.get("action") != "retrieve_reference":
                continue
            action_input = step.get("action_input", {})
            if isinstance(action_input, dict):
                return str(action_input.get("query", ""))[:300]
            return str(action_input)[:300]
        return ""

    supervised_rows = [row for row in rollout_supervision if row.get("gold") is not None]
    compact_supervision = [{
        "conversation_id": row.get("conversation_id"), "turn_index": row.get("turn_index"),
        "target_type": row.get("target_type"), "context": str(row.get("context", ""))[-700:],
        "prediction": str(row.get("prediction", ""))[:500],
        "predicted_action": row.get("predicted_action", ""),
        "predicted_slots": row.get("predicted_slots", []), "gold": row.get("gold"),
        "reference_query": reference_query_from_trace(row.get("react_trace")),
    } for row in supervised_rows[-32:]]
    graph_edges = [{
        "edge_id": edge_id, "source_action": edge["source_action"],
        "target_action": edge["target_action"], "kind": edge["kind"],
    } for edge_id, edge in state.get("edges", {}).items()]
    resources = {"reference": reference, "action_rules": action_rules, "slot_policies": slot_policies}
    lookups, planner_prompt, planner_error = _plan_resource_lookups(
        compact_supervision, graph_edges, skill, model, max_retries,
    )
    retrieved = []
    for lookup in lookups:
        retrieved.extend(_resource_lookup_sections(
            lookup["resource"], resources[lookup["resource"]], lookup["query"], lookup["top_k"],
        ))
    prompt = _AUTONOMOUS_RESOURCE_REFLECTION_PROMPT.format(
        skill=skill or "[empty]",
        retrieved_resources=json.dumps(retrieved, ensure_ascii=False, indent=2) or "[]",
        graph_edges=json.dumps(graph_edges, ensure_ascii=False),
        rollout_supervision=json.dumps(compact_supervision, ensure_ascii=False, indent=2),
    )
    from llm import chat, resolve_config
    cfg = resolve_config(model=model)
    raw, payload, last_error = "", {}, ""
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            raw = chat([{"role": "user", "content": prompt}], model=cfg["model"], api_key=cfg["api_key"],
                       base_url=cfg["base_url"], temperature=0.0).strip()
            text = "\n".join(raw.splitlines()[1:-1]) if raw.startswith("```") else raw
            start, end = text.find("{"), text.rfind("}")
            payload = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
            if isinstance(payload.get("updates"), list):
                break
        except Exception as exc:
            payload = {}
            last_error = repr(exc)
        if attempt < max(1, max_retries):
            time.sleep(float(2 ** (attempt - 1)))

    valid_actions = {str(node["label"]) for node in state.get("nodes", [])}
    accepted, rejected = [], []
    for update in payload.get("updates", []):
        if not isinstance(update, dict):
            continue
        resource, content = str(update.get("resource", "")), str(update.get("content", "")).strip()
        status = str(update.get("status", "uncertain")).lower()
        if resource not in {"transition_guard", "action_rule", "slot_policy", "reference"} or not content:
            rejected.append({"update": update, "reason": "unsupported_resource_or_empty_content"})
            continue
        if resource == "transition_guard":
            edge_id = str(update.get("edge_id", ""))
            if edge_id not in state.get("edges", {}):
                rejected.append({"update": update, "reason": "unknown_edge"})
                continue
            edge = state["edges"][edge_id]
            edge["guard"] = content
            edge["guard_status"] = "resolved" if status == "resolved" else "uncertain"
        elif resource in {"action_rule", "slot_policy"}:
            action = str(update.get("action", ""))
            if action not in valid_actions:
                rejected.append({"update": update, "reason": "unknown_action"})
                continue
            bucket = "action_rules" if resource == "action_rule" else "slot_policies"
            record = state.setdefault(bucket, {}).setdefault(action, {"action": action})
            record["policy" if resource == "slot_policy" else "rule"] = content
            record["status"] = "resolved" if status == "resolved" else "uncertain"
        else:
            state.setdefault("reference_notes", []).append({"content": content, "status": status, "rationale": update.get("rationale", "")})
        accepted.append(update)
    return {
        "planner_prompt": planner_prompt, "planner_prompt_chars": len(planner_prompt),
        "lookups": lookups, "retrieved_resources": retrieved, "planner_error": planner_error,
        "prompt": prompt, "prompt_chars": len(prompt), "raw_response": raw,
        "accepted": accepted, "rejected": rejected,
        "error": last_error if not payload else "",
    }


def render_online_action_rules(state: dict[str, Any]) -> str:
    lines = ["# Online Action Rule Refinements", ""]
    for action, record in sorted(state.get("action_rules", {}).items()):
        rule = str(record.get("rule", "")).strip()
        if rule and record.get("status") == "resolved":
            lines.extend([f"#### `{action}`", rule, ""])
    return "\n".join(lines).rstrip() + "\n"


def render_online_slot_policies(state: dict[str, Any]) -> str:
    """Render resolved online refinements in the agent's policy-resource format."""
    lines = ["# Online Slot Policy Refinements", ""]
    for action, record in sorted(state.get("slot_policies", {}).items()):
        policy = str(record.get("policy", "")).strip()
        if policy and record.get("status") == "resolved":
            lines.extend([f"#### `{action}`", policy, ""])
    if len(lines) == 2:
        lines.append("No online slot policy refinements have been validated yet.")
    return "\n".join(lines).rstrip() + "\n"


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
    for note in state.get("reference_notes", []):
        if str(note.get("content", "")).strip():
            reference_lines.extend(["## Online-maintained note", f"- {note['content']}", ""])
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


def summarize_refinement_state(
    state: dict[str, Any], policy: RefinementPolicy,
) -> dict[str, Any]:
    """Explain promotion eligibility and blocking reasons for every branch."""
    order = {node: index for index, node in enumerate(state.get("backbone_order", []))}
    rows = []
    counts: Counter[str] = Counter()
    for edge_id, edge in sorted(state.get("edges", {}).items()):
        if edge.get("kind") not in {"candidate_branch", "promoted_branch"}:
            continue
        confidence = edge_confidence(edge)
        support = int(edge.get("gold_support", 0) or 0)
        conflict = sum(int(value) for value in edge.get("competing_targets", {}).values())
        forward = order.get(edge.get("target"), math.inf) >= order.get(edge.get("source"), -1)
        blockers = []
        if not forward:
            blockers.append("revisit_or_unknown_backbone_order")
        if support < policy.min_gold_support:
            blockers.append("insufficient_gold_support")
        if confidence < policy.min_confidence:
            blockers.append("low_target_action_reliability")
        if edge.get("guard_status") != "resolved":
            blockers.append("guard_unresolved")
        if edge.get("visibility") != "skill" and blockers:
            for blocker in blockers:
                counts[blocker] += 1
        rows.append({
            "edge_id": edge_id,
            "source_action": edge.get("source_action"),
            "target_action": edge.get("target_action"),
            "visibility": edge.get("visibility"),
            "kind": edge.get("kind"),
            "gold_support": support,
            "rollout_success": int(edge.get("rollout_success", 0) or 0),
            "rollout_failure": int(edge.get("rollout_failure", 0) or 0),
            "slot_failures": int(edge.get("slot_failures", 0) or 0),
            "confidence": round(confidence, 6),
            "conflict_count": conflict,
            "guard_status": edge.get("guard_status"),
            "blockers": blockers,
        })
    return {
        "batches_processed": int(state.get("batches_processed", 0)),
        "policy": {
            "min_gold_support": policy.min_gold_support,
            "min_confidence": policy.min_confidence,
            "min_conflict_count": policy.min_conflict_count,
            "max_skill_branches_per_source": policy.max_skill_branches_per_source,
        },
        "num_candidate_branches": len(rows),
        "blocker_counts": dict(sorted(counts.items())),
        "branches": rows,
    }
