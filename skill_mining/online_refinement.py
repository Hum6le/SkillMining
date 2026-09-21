"""Evidence-calibrated online refinement for graph-compiled ABCD skills.

This module deliberately separates deterministic graph maintenance from LLM
language induction.  It maintains a rooted DAG-like control skeleton with
explicit retry/revisit edges, schedules contrastive rollout batches, and
localizes gold-supervised rollout feedback to graph regions.  A later LLM
stage can turn only the selected edge records into natural-language guards.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_tod.abcd.action_schema import canonical_action_name, load_action_schema
from eval_tod.abcd.metrics import slot_error_bucket, slot_match_profile


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

Interpret evidence action-centrically. When the action prediction is wrong,
its emitted values belong to the wrong action and are not slot-policy evidence.
When action and slots are both correct, treat that case as positive evidence
for the observed value-source, order, reuse, and missing-value behavior. Use
successful cases to complete or sharpen an underspecified existing policy,
not only to preserve it. A success alone must not trigger a destructive
rewrite. Revise a slot policy for action-correct slot failures by contrasting
them with successful cases for the same action, and preserve compatible
patterns that are repeatedly supported by those successes.

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

For a transition guard, the gold/predicted labels identify what was correct or
incorrect; they are NOT an explanation of why the route should be chosen.
Infer that explanation from the dialogue context, especially the agent's
immediately preceding request/offer and the customer's response, clarification,
acceptance, rejection, or unresolved concern. Compare sibling targets jointly
and state the conversational difference that makes one route appropriate.
Never turn this into a state table, slot-schema test, frequency rule, or
action-name paraphrase.

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
customer-service skill and its progressive-disclosure resources. The current
skill is a compact backbone extracted from a complete transition graph. Many
valid but less central transitions are intentionally not written in skill.md;
they are retained in reference.md for retrieval.

Inspect the rollout-versus-ground-truth trajectories, complete current skill,
retrieved resource snippets, and graph edges. Decide yourself which changes,
if any, would make the skill more correct, compact, and well-organized.

The current skill is the established executable baseline. Preserve its existing
workflow, action placement, transition logic, slot discipline, and retrieval
guidance by default. Online evidence from one batch is usually partial: do not
rewrite or remove a working rule merely because the current batch did not show
it. Prefer a minimal compatible addition, clarification, exception note, or
retrieval instruction that leaves the existing behavior intact.

Only delete existing skill/resource logic when the supplied evidence shows a
clear contradiction, systematic error, misleading instruction, or duplicate
rule that cannot be resolved by a compatible update. When evidence is
ambiguous, retain the old logic and revise its local wording to include the
new condition, uncertainty, or retrieval guidance instead. Never trade away a
previously supported route to fix one isolated failure.

You may promote a useful transition and its target action branch from
reference into skill, revise an existing action/transition explanation, move a
confusing or mostly harmful skill rule back to reference, refine an action rule
or slot policy, add a retrieval instruction where a deferred transition should
be consulted, or make no change. A promotion is not complete if it only adds a
sentence to a routing-policy resource: the executable skill must also make the
target action reachable from the source and preserve or add the target action's
role/placement description. Use the `action_node` update for that case and
provide a compatible skill edit when the existing action card needs to be
clarified. Do not assume every observed error requires a skill edit. When
information belongs in reference, ensure the relevant skill rule explains when
to retrieve it instead of flattening all exceptional logic into skill.md.

When a promotion is accepted, the compiler materializes the promoted
transition as a skill-visible backbone/DAG edge and refreshes the Backbone
Tree, Backbone Edges, Routing Policies, and any missing target Action Rule
together. Reference-only transitions remain retrievable and must not be
presented as backbone edges.

Use only listed action names and graph edges. Learn general rules from the
gold/prediction contrast: never hard-code literal customer values or invent
hidden state, slot names, actions, or edges.

For transition edits, gold and predicted actions identify the error only.
Derive the routing explanation from the dialogue context itself: compare what
the agent and customer said before the decision with sibling routes, and write
the natural-language circumstance that distinguishes them. Never use graph
kind, edge frequency, a synthetic state field, slot schema, or an action name
as the claimed cause of a transition. For utterance edits, use the dialogue
and gold response to infer a reusable response policy, not a copied reply.

The runtime provides an MCP-style local resource lookup. First choose which
small parts of the auxiliary resources are relevant to this batch; retrieved
observations are supplied below. The complete current skill is always visible
because it is the executable control contract. Do not assume an un-retrieved
auxiliary resource says anything in particular.

Treat the supplied batch as joint positive and negative evidence. Explicitly
summarize the successful patterns that should be preserved and the failed
patterns that should be repaired or avoided. Use repeated successful cases to
complete or sharpen underspecified action rules and slot-binding policies,
including value source, order, reuse, and missing-value behavior. Make this
comparison in the same reflection call: a failure alone is not sufficient
evidence to delete a previously working rule, and a success is evidence both
for retaining valid behavior and, when the existing description is vague,
for making that behavior more precise.

The primary online objective is to improve joint AST, not an isolated action
accuracy or isolated slot accuracy. For an action turn, AST is successful only
when both the predicted action and its ordered slot values match the gold
decision. Diagnose action and slot errors together and prefer an update that
improves the complete decision without regressing the other part. A rule that
raises action accuracy while causing slot-value failures is not an improvement.
Agent-utterance quality may be considered as secondary evidence, but it must
never replace the joint action-and-slot AST objective.

Apply this AST triage exactly for every action turn. (1) If the predicted
action is wrong, diagnose routing/action selection; do not attribute its slot
list to the gold action's Action Card. (2) If action and ordered slots are both
correct, treat the case as positive evidence for the observed Action Card
pattern. Repeated successes may sharpen an underspecified card, but success
alone must not justify deleting or weakening valid logic. (3) If the action is
correct but ordered slots are wrong, compare the failure with successful cases
for the same action and make a compatible slot-binding update when the
contrast supports one. Routing itself must remain unchanged when it is
already correct.

<current_skill>{skill}</current_skill>
<retrieved_resources>{retrieved_resources}</retrieved_resources>
<graph_edges>{graph_edges}</graph_edges>
<evidence_packets>
The transition packet contains sibling-route evidence. The action-card packet
contains same-action AST successes, action-correct slot failures, and action
selection failures. Use the packet labels as diagnostic organization, not as
semantic explanations.
{evidence_packets}
</evidence_packets>
<rollout_supervision>
Each record contains the model's final prediction and, when retrieval was
used, only the generated retrieval query and returned retrieval content from
the ReAct trace. The raw trace, tool arguments, and duplicated messages are
never supplied. The dialogue context and gold action/slots or gold agent
response are supplied separately in that same record.
{rollout_supervision}
</rollout_supervision>

Return valid JSON only:
{{"decision":"update|no_update","no_update_reason":"required only for no_update",
"updates":[{{"resource":"transition_guard|action_node|action_rule|slot_policy|reference",
"edge_id":"required for transition_guard; optional but recommended for edge-scoped reference",
"action":"required for action_node/action_rule/slot_policy",
"op":"upsert|delete",
"content":"concise natural-language replacement or addition; empty only for delete", "status":"resolved|uncertain",
"rationale":"grounded in specific rollout-vs-gold evidence"}}],
"skill_operations":[{{"op":"upsert|delete",
"resource":"optional transition_guard|action_rule; identifies the semantic update being materialized",
"edge_id":"optional graph edge ID; required when applying a transition_guard",
"action":"optional action name; recommended when materializing an action rule",
"match_text":"an exact, unique excerpt copied from current_skill",
"new_text":"complete, compatibility-preserving replacement for match_text; empty only for delete",
"occurrence":"optional 1-based occurrence number, only when match_text is repeated",
"rationale":"why this local edit improves the executable skill"}}]}}

<filesystem_mcp>
You may directly edit the complete current skill without relying on any fixed
heading, HTML marker, transition block, or line number. Choose the most
appropriate existing prose to revise or remove. Each `match_text` must be
copied exactly from `current_skill` and must identify one location:
- `upsert`: replace that exact local excerpt with its complete updated version.
  Include the previous valid logic in `new_text` and integrate the new evidence
  into it naturally, rather than appending a detached patch elsewhere.
- `delete`: remove that exact excerpt; set `new_text` to an empty string.
If an otherwise appropriate excerpt occurs more than once, use a longer local
excerpt first. Only when that is impractical, provide `occurrence` as a
1-based position for the intended match; never apply the same edit to every
occurrence.
The executor applies operations in order and requires each match to occur
exactly once in the then-current file. Prefer a sufficiently specific local
paragraph or bullet over an entire document. Do not emit a skill operation
when a resource-only update is enough.

Preservation policy for `skill_operations`:
- Default to `upsert` of a small, relevant paragraph or bullet. Preserve its
  valid meaning and revise it into one coherent rule that includes the new
  compatible condition or retrieval guidance.
- Use `delete` only when the matched text is demonstrably harmful or duplicated
  and cannot be made compatible by clarification. Do not delete a route merely
  because it is rare, absent from this batch, or uncertain.
- Never upsert an entire skill, a whole workflow section, or a
  large unrelated paragraph when a smaller excerpt can express the update.
- A patch rationale must name the preserved behavior and the concrete evidence
  supporting the change.

If an update changes a transition guard that should affect runtime routing,
also emit the matching `skill_operation` that integrates, rewrites, or removes
that routing prose in `skill.md`. A transition update without such an edit is
reference-only and must not be presented as a change to the executable skill.
</filesystem_mcp>
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

_BATCH_ROOT_CAUSE_PROMPT = """You are diagnosing a compact batch of similar
action-turn rollouts from one frozen skill snapshot. Write a substantial,
specific root-cause analysis. Do not merely restate gold and predicted labels.

The rollout records include deterministic slot-comparison diagnostics. Treat
those diagnostics as authoritative. Do not collapse `permutation_only`,
`case_or_whitespace_only`, `case_or_whitespace_plus_permutation`, or
`punctuation_or_formatting_only` into a generic wrong-value explanation.
For every action-correct slot failure, explicitly reason in this order:
value count -> semantic value multiset -> ordered position -> case/whitespace
-> punctuation/formatting. Distinguish a genuinely wrong value from a value
that is correct but serialized in the wrong position or surface form.

In addition to diagnosing failures, mine reusable behavior from successful
trajectories. Explicitly summarize: (1) action order and transition strategy;
(2) ordered slot usage, including any supported variants rather than inventing
one global order; (3) normalization/surface-form behavior by value type; and
(4) which successful behaviors should be preserved. A successful trajectory
is positive evidence even when another trajectory in the batch fails.

Explicitly determine whether the failure came from one or more of:
- missing, ambiguous, contradictory, or overly verbose skill text;
- a needed reference section not being queried or not being retrieved;
- retrieved reference evidence being irrelevant or misleading;
- the correct reference being available but ignored during action selection;
- an incorrect graph route or insufficient sibling-edge distinction;
- an action card missing value-source/order/reuse guidance;
- correct action selection followed by incorrect slot grounding;
- insufficient evidence, in which case say what remains unresolved.

Compare successful cases, failures, and counterexamples in the batch. Explain
the causal chain from visible prompt/retrieval evidence to the final error.
Candidate changes are proposals only; a later reflection combines several
batch reports before anything is written.

<current_skill>{skill}</current_skill>
<local_graph>{local_graph}</local_graph>
<available_local_resources>{available_resources}</available_local_resources>
<batch_rollouts>{rollouts}</batch_rollouts>

Return JSON only:
{{"summary":"a detailed multi-sentence root-cause analysis",
"root_causes":[{{"category":"skill_missing|skill_ambiguous|reference_not_queried|reference_not_retrieved|reference_misleading|retrieval_ignored|graph_routing|action_order|action_card|slot_order|normalization|slot_grounding|preserved_success|insufficient_evidence|other",
"analysis":"detailed causal explanation","evidence_ids":["sample ids"],
"confidence":0.0}}],
"graph_footprint":{{"nodes":["actions"],"edges":["source=>target"]}},
"candidate_updates":[{{"resource":"transition_guard|action_node|action_rule|slot_policy|reference",
"edge_id":"optional","action":"optional","op":"upsert|delete",
"content":"proposed text","status":"resolved|uncertain","rationale":"..."}}],
"candidate_skill_operations":[{{"operation_id":"stable descriptive id",
"op":"replace|insert_before|insert_after|delete","match_text":"exact excerpt from current_skill",
"new_text":"text selected by the model; empty only for delete","rationale":"..."}}],
"unresolved_questions":["..."]}}
"""

_GROUP_REFLECTION_PROMPT = """You are the final reflection stage for a local
neighborhood of a graph-compiled skill. The reports below were produced from
similar rollout batches. Synthesize their root-cause analyses, resolve
conflicting proposals, and return one coherent result. A group contains at
most 16 batch reports.

Preserve the useful behaviors identified in successful trajectories. In the
final synthesis explicitly consider four dimensions: action order and
transition strategy; ordered slot usage and supported order variants;
normalization/surface-form behavior; and successful behaviors that must remain
unchanged. Treat deterministic slot-match profiles in the reports as ground
truth for the error type. Do not turn a case-plus-permutation or formatting
error into a generic value-grounding rule, and do not promote one observed
local order to a global invariant without supporting evidence.

You can read the compact complete skill and a locally sampled graph. You may
revise skill text and you decide the exact insertion/replacement location by
copying an exact match_text and choosing replace, insert_before, insert_after,
or delete. Prefer coherent revisions over repeatedly appending case-specific
rules. Do not copy conversation IDs into executable text.

The synthesis summary should be reasonably detailed. It must state whether
the underlying deficiency is in skill text, graph routing, reference query or
retrieval, use of retrieved evidence, action selection, or slot grounding.

<current_skill>{skill}</current_skill>
<local_graph>{local_graph}</local_graph>
<batch_reports>{batch_reports}</batch_reports>

Return JSON only:
{{"decision":"update|no_update","summary":"detailed synthesis and causal chain",
"merged_root_causes":[{{"analysis":"...","batch_ids":["..."],"confidence":0.0}}],
"updates":[{{"resource":"transition_guard|action_node|action_rule|slot_policy|reference",
"edge_id":"optional","action":"optional","op":"upsert|delete","content":"...",
"status":"resolved|uncertain","rationale":"..."}}],
"skill_operations":[{{"operation_id":"unique stable id",
"op":"replace|insert_before|insert_after|delete","match_text":"exact excerpt",
"new_text":"complete text chosen by the model","rationale":"..."}}],
"rejected_candidates":[{{"reason":"..."}}],"unresolved_questions":["..."]}}
"""

_TRACE2SKILL_LOCAL_MAP_PROMPT = """You are the MAP stage of a Trace2Skill-style
skill evolution pass, adapted to one local neighborhood of a graph-compiled
customer-service skill. The supplied records are already diagnosed rollout
batches containing both strict successes and failures.

Extract a small, evidence-grounded candidate patch. Preserve recurring success
behavior and contrast it with failures. Explicitly cover action order, sibling
route distinctions, ordered slot usage, normalization, and reference retrieval
when supported. Do not turn one local observation into a global rule, invent
actions or slot values, or rewrite the whole skill. Candidate changes are not
applied at this stage; a REDUCE stage will reconcile all MAP outputs.

<current_skill>{skill}</current_skill>
<local_graph>{local_graph}</local_graph>
<success_distillation>{success_distillation}</success_distillation>
<failure_distillation>{failure_distillation}</failure_distillation>
<diagnosed_records>{batch_reports}</diagnosed_records>

Return JSON only:
{{"summary":"reusable success and failure patterns",
"root_causes":[{{"analysis":"...","batch_ids":["..."],"confidence":0.0}}],
"graph_footprint":{{"nodes":["..."],"edges":["..."]}},
"candidate_updates":[{{"resource":"transition_guard|action_node|action_rule|slot_policy|reference",
"edge_id":"optional","action":"optional","op":"upsert|delete","content":"...",
"status":"resolved|uncertain","rationale":"..."}}],
"candidate_skill_operations":[{{"operation_id":"unique stable id",
"op":"replace|insert_before|insert_after|delete","match_text":"exact excerpt",
"new_text":"complete local revision","rationale":"..."}}],
"preserved_successes":[{{"behavior":"...","evidence_ids":["..."]}}],
"unresolved_questions":["..."]}}
"""

_TRACE2SKILL_SUCCESS_DISTILL_PROMPT = """Distill reusable behavior from strict
successful action-turn trajectories in one graph neighborhood. Focus on action
order, sibling-route cues, ordered slot values, normalization, and retrieval
behavior. State applicability conditions and behavior that later edits must
preserve. Do not infer rules not supported by the examples.

<local_graph>{local_graph}</local_graph>
<successful_trajectories>{trajectories}</successful_trajectories>

Return JSON only: {{"summary":"...","patterns":[{{"behavior":"...",
"evidence_ids":["..."],"scope":{{"actions":["..."],"edges":["..."]}}}}]}}
"""

_TRACE2SKILL_FAILURE_DISTILL_PROMPT = """Distill reusable corrections from
failed action-turn trajectories in one graph neighborhood. Separate routing,
retrieval, ordered-slot, and normalization causes using the deterministic
profiles. Contrast failures with visible context; do not invent hidden state,
actions, slot names, or customer-specific rules.

<local_graph>{local_graph}</local_graph>
<failed_trajectories>{trajectories}</failed_trajectories>

Return JSON only: {{"summary":"...","corrections":[{{"cause":"...",
"correction":"...","evidence_ids":["..."],
"scope":{{"actions":["..."],"edges":["..."]}}}}]}}
"""

_TRACE2SKILL_LOCAL_REDUCE_PROMPT = """You are the REDUCE/APPLY stage of a
Trace2Skill-style evolution pass for one local neighborhood of a graph-compiled
customer-service skill. Reconcile the MAP candidates below into one minimal,
coherent patch that can be written back to graph resources and skill text.

Resolve contradictions using evidence coverage and confidence. A proposed fix
must preserve the listed successful action order, route distinctions, ordered
slot behavior, and normalization behavior. Prefer action- or edge-scoped
updates. Reject unsupported global generalizations and duplicate rules. The
complete graph remains in JSON; do not flatten it into skill text. You decide
the exact text insertion/replacement location by copying an exact excerpt from
current_skill. Do not include conversation IDs or literal customer values in
executable text.

<current_skill>{skill}</current_skill>
<local_graph>{local_graph}</local_graph>
<map_candidates>{map_candidates}</map_candidates>

Return JSON only:
{{"decision":"update|no_update","summary":"detailed reduced causal analysis",
"merged_root_causes":[{{"analysis":"...","batch_ids":["..."],"confidence":0.0}}],
"updates":[{{"resource":"transition_guard|action_node|action_rule|slot_policy|reference",
"edge_id":"optional","action":"optional","op":"upsert|delete","content":"...",
"status":"resolved|uncertain","rationale":"..."}}],
"skill_operations":[{{"operation_id":"unique stable id",
"op":"replace|insert_before|insert_after|delete","match_text":"exact excerpt",
"new_text":"complete local revision","rationale":"..."}}],
"preserved_successes":[{{"behavior":"...","evidence_ids":["..."]}}],
"rejected_candidates":[{{"reason":"..."}}],"unresolved_questions":["..."]}}
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
    """Complete structural node + directed-transition signature.

    User utterances are intentionally excluded from this key. They remain in
    the rollout evidence shown to the refinement model, while this signature
    answers only the graph-structural question: which actions and transitions
    occur in the session?
    """
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


def _signature_distance(left: frozenset[str], right: frozenset[str]) -> int:
    """Structural edit distance used to find near-neighbor trajectories."""
    return len(left ^ right)


def initialize_skill_dag(subgraph: dict[str, Any], subflow: str) -> dict[str, Any]:
    """Create the online state from an offline backbone mining artifact.

    The offline tree remains immutable. Any non-backbone edge begins as
    reference-only; the online optimizer may later promote a supported branch
    and its target action placement into the working skill.
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
        "schema_version": 2,
        "subflow": subflow,
        "root": ROOT,
        "nodes": [{"id": node, "label": label, "topological_order": order_index.get(node)} for node, label in nodes.items()],
        "backbone_order": order,
        "edges": state_edges,
        "slot_policies": {},
        "action_rules": {},
        "reference_notes": [],
        "evidence_pool": {
            "transition": {},
            "action_card": {},
            "unresolved": [],
        },
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
        for key, default in {
            "gold_support": 0, "rollout_success": 0, "rollout_failure": 0,
            "slot_total": 0, "slot_success": 0, "slot_failures": 0,
            "evidence": [], "guard": "", "guard_status": "pending",
        }.items():
            edge.setdefault(key, default)
    for action, record in state.setdefault("slot_policies", {}).items():
        record.setdefault("action", action)
        for key, default in {
            "slot_total": 0, "slot_success": 0, "slot_failures": 0,
            "evidence": [], "policy": "", "status": "pending",
        }.items():
            record.setdefault(key, default)
    state.setdefault("action_rules", {})
    state.setdefault("reference_notes", [])
    pool = state.setdefault("evidence_pool", {})
    pool.setdefault("transition", {})
    pool.setdefault("action_card", {})
    pool.setdefault("unresolved", [])
    return state


def _source_target_session_index(conversations: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for conversation in conversations:
        actions = _actions(conversation)
        for source, target in set(zip(actions, actions[1:])):
            index[(source, target)].append(conversation)
    return index


def _representative_groups(
    conversations: list[dict[str, Any]], limit: int,
) -> list[list[dict[str, Any]]]:
    """Select structural groups without discarding same-signature sessions.

    A previous version collapsed every exact signature to one conversation.
    That made the online optimizer blind to wording, slot values, and
    success/failure variation inside the same graph structure.  We now retain
    all members of a selected signature group contiguously. ``limit`` remains
    the sampling budget, so a very large group may be split at the budget
    boundary, but it is never silently reduced to one representative.
    """
    if limit <= 0 or not conversations:
        return []
    grouped: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
    for conversation in conversations:
        grouped[session_signature(conversation)].append(conversation)
    candidates = list(grouped.items())
    if sum(len(members) for _, members in candidates) <= limit:
        return [members for _, members in candidates]

    signatures = [signature for signature, _ in candidates]
    first_index = max(
        range(len(candidates)),
        key=lambda index: sum(_weighted_jaccard(signatures[index], other) for other in signatures),
    )
    selected_indices = [first_index]
    while len(selected_indices) < len(candidates):
        remaining = [index for index in range(len(candidates)) if index not in selected_indices]
        if not remaining:
            break
        next_index = max(
            remaining,
            key=lambda index: min(
                _signature_distance(signatures[index], signatures[chosen])
                for chosen in selected_indices
            ),
        )
        selected_indices.append(next_index)

    result: list[list[dict[str, Any]]] = []
    remaining_budget = limit
    for index in selected_indices:
        if remaining_budget <= 0:
            break
        members = candidates[index][1]
        selected_members = members[:remaining_budget]
        if selected_members:
            result.append(selected_members)
            remaining_budget -= len(selected_members)
    return result


def _representatives(conversations: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Flatten grouped representatives while preserving group contiguity."""
    return [conversation for group in _representative_groups(conversations, limit) for conversation in group]


def schedule_contrastive_batches(
    conversations: list[dict[str, Any]], state: dict[str, Any], batch_size: int = 8,
    per_transition_cap: int = 3, target_selection_rate: float = 0.30,
    max_batches: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Create graph-structured rollout batches without using class labels.

    A batch preferentially pairs distinct targets leaving the same source
    action. Exact structural signatures are grouped and retained together;
    nearby signatures are ordered together so the LLM sees a minimal graph
    contrast. This is intentionally a budgeted sampler: unselected sessions
    are not appended merely for full coverage.
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
        representative_groups_by_target = []
        for _, members, _ in sorted(options, key=lambda item: (item[2], -len(item[1]), item[0])):
            representative_groups_by_target.append(
                _representative_groups(members, representative_cap)
            )
        if sum(bool(items) for items in representative_groups_by_target) >= 2:
            # Align each alternative with the closest structural neighbor of
            # the first target. This makes the context explain the smallest
            # graph difference available, instead of comparing arbitrary
            # trajectories that merely share the source action.
            anchor_groups = next(
                (groups for groups in representative_groups_by_target if groups), []
            )
            anchor_signature = session_signature(anchor_groups[0][0]) if anchor_groups else frozenset()
            for groups in representative_groups_by_target[1:]:
                groups.sort(
                    key=lambda group: _signature_distance(
                        session_signature(group[0]), anchor_signature
                    )
                )
            source_rounds = []
            for round_index in range(max(len(items) for items in representative_groups_by_target)):
                source_round = [
                    conversation
                    for groups in representative_groups_by_target
                    if round_index < len(groups)
                    for conversation in groups[round_index]
                ]
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
                for index in range(0, len(contrast_batch), batch_size):
                    chunk = contrast_batch[index:index + batch_size]
                    if len(chunk) >= 2:
                        batches.append(chunk)
                        used.update(str(conversation.get("convo_id", "?")) for conversation in chunk)
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


def build_action_turn_samples(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand conversations into independent action-turn + prefix samples."""
    samples = []
    for conversation in conversations:
        convo_id = str(conversation.get("convo_id", "?"))
        turns = conversation.get("delexed") or []
        for turn_index, turn in enumerate(turns):
            targets = turn.get("targets") or []
            if len(targets) < 3 or targets[1] != "take_action":
                continue
            source = "ROOT"
            source_turn = None
            prefix_action_sequence = []
            for prefix_turn in turns[:turn_index]:
                prefix_targets = prefix_turn.get("targets") or []
                if len(prefix_targets) >= 3 and prefix_targets[1] == "take_action":
                    prefix_action_sequence.append(str(prefix_targets[2]))
            for previous_index in range(turn_index - 1, -1, -1):
                prev = turns[previous_index]
                ptargets = prev.get("targets") or []
                if len(ptargets) >= 3 and ptargets[1] == "take_action":
                    source = str(ptargets[2])
                    source_turn = previous_index
                    break
            samples.append({"sample_id": f"{convo_id}:{turn_index}", "conversation_id": convo_id, "convo_id": convo_id,
                            "turn_index": turn_index, "source_action": source,
                            "source_turn": source_turn,
                            "prefix_action_sequence": prefix_action_sequence,
                            "target_action": str(targets[2]),
                            "gold_slots": targets[3] if len(targets) > 3 and isinstance(targets[3], list) else [],
                            "conversation": conversation})
    return samples


def schedule_action_turn_batches(conversations: list[dict[str, Any]], batch_size: int = 8,
                                 max_batches: int | None = None) -> list[list[dict[str, Any]]]:
    """Schedule independent action turns; one session may occur in many batches."""
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for sample in build_action_turn_samples(conversations):
        groups[sample["source_action"]][sample["target_action"]].append(sample)
    batches = []
    for source in sorted(groups):
        targets = groups[source]
        # Round-robin target queues so sibling outcomes share a batch.
        queues = [targets[target] for target in sorted(targets)]
        if len(queues) == 1:
            queue = queues[0]
            for i in range(0, len(queue), batch_size):
                batches.append(queue[i:i + batch_size])
            continue
        while any(queues):
            batch = []
            for queue in queues:
                if queue and len(batch) < batch_size:
                    batch.append(queue.pop(0))
            if batch:
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
        # A turn absent from rows_by_turn was intentionally not rolled out
        # (e.g. utterance turns or another action-turn sample in the same
        # session). Never convert that absence into a synthetic failure.
        if turn_index not in rows_by_turn:
            continue
        raw = rows_by_turn[turn_index]
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
    slot_events = []
    action_events = []
    for batch_item in conversations:
        sample = batch_item if isinstance(batch_item, dict) and "conversation" in batch_item else None
        conversation = sample["conversation"] if sample is not None else batch_item
        sid = str(conversation.get("convo_id", "?"))
        actions = _gold_action_rows(conversation, by_conversation.get(sid, []))
        if sample is not None:
            target_turn = int(sample.get("turn_index", -1))
            actions = [row for row in actions if row["turn_index"] == target_turn]
        # Slot binding is an action-level property, not a property of the
        # incoming transition. Record every gold action turn, including the
        # first action and single-action sessions. Only action-correct turns
        # can be valid evidence for that action's ordered slot policy.
        for current in actions:
            action_ok = bool(current["action_correct"])
            slot_ok = bool(current["slot_correct"])
            if not action_ok:
                continue
            evidence = {
                "conversation_id": sid,
                "source_turn": None,
                "target_turn": current["turn_index"],
                "action_success": True,
                "slot_success": slot_ok,
                "slot_evaluable": True,
                "gold_action": current["gold_action"],
                "predicted_action": current["predicted_action"],
                "gold_slots": current["gold_slots"],
                "predicted_slots": current["predicted_slots"],
                "gold_slot_count": current["gold_slot_count"],
                "predicted_slot_count": current["predicted_slot_count"],
                "context": current["context"],
                "react_trace": current["react_trace"],
            }
            policy = state.setdefault("slot_policies", {}).setdefault(current["gold_action"], {
                "action": current["gold_action"], "slot_total": 0, "slot_success": 0,
                "slot_failures": 0, "evidence": [], "policy": "", "status": "pending",
            })
            for field, default in {
                "slot_total": 0, "slot_success": 0, "slot_failures": 0,
                "evidence": [], "policy": "", "status": "pending",
            }.items():
                policy.setdefault(field, default)
            policy["slot_total"] += 1
            if slot_ok:
                policy["slot_success"] += 1
            else:
                policy["slot_failures"] += 1
            if len(policy["evidence"]) < max_evidence_per_edge:
                policy["evidence"].append(evidence)
            slot_events.append(evidence)
        edge_pairs = list(zip(actions, actions[1:]))
        if sample is not None and actions:
            current = next(
                (row for row in actions if row["turn_index"] == int(sample.get("turn_index", -1))),
                None,
            )
            if current is not None:
                edge_pairs = [({
                    "turn_index": sample.get("source_turn"),
                    "gold_action": str(sample.get("source_action", "ROOT")),
                }, current)]
        for previous, current in edge_pairs:
            source = action_to_node.get(previous["gold_action"], previous["gold_action"])
            target = action_to_node.get(current["gold_action"], current["gold_action"])
            key = _edge_id(source, target)
            edge = state.setdefault("edges", {}).setdefault(key, {
                "source": source, "target": target,
                "source_action": previous["gold_action"], "target_action": current["gold_action"],
                "kind": "candidate_branch", "visibility": "reference", "offline_support": 0,
                "offline_sessions": 0, "gold_support": 0, "rollout_success": 0,
                "rollout_failure": 0, "slot_total": 0, "slot_success": 0,
                "slot_failures": 0, "competing_targets": Counter(),
                "guard": "", "guard_status": "pending", "evidence": [],
            })
            edge.setdefault("slot_total", 0)
            edge.setdefault("slot_success", 0)
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
                "slot_evaluable": action_ok,
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
        # Keep action-selection failures in a separate packet. They are useful
        # for diagnosing routing, but must never be counted as slot-policy
        # evidence for the gold action.
        for current in actions:
            if current["action_correct"]:
                continue
            action_events.append({
                "conversation_id": sid,
                "source_turn": None,
                "target_turn": current["turn_index"],
                "action_success": False,
                "slot_success": False,
                "slot_evaluable": False,
                "gold_action": current["gold_action"],
                "predicted_action": current["predicted_action"],
                "gold_slots": current["gold_slots"],
                "predicted_slots": current["predicted_slots"],
                "gold_slot_count": current["gold_slot_count"],
                "predicted_slot_count": current["predicted_slot_count"],
                "context": current["context"],
                "react_trace": current["react_trace"],
            })
    state["batches_processed"] = int(state.get("batches_processed", 0)) + 1
    return {
        "events": events,
        "num_events": len(events),
        "slot_events": slot_events,
        "num_slot_events": len(slot_events),
        "action_events": action_events,
        "num_action_events": len(action_events),
    }


def _classify_online_evidence(item: dict[str, Any]) -> list[str]:
    """Attach explicit AST error dimensions for downstream LLM diagnosis."""
    action_ok = bool(item.get("action_success"))
    slot_ok = bool(item.get("slot_success"))
    gold_slots = [str(value) for value in item.get("gold_slots", [])]
    predicted_slots = [str(value) for value in item.get("predicted_slots", [])]
    item["slot_match_profile"] = slot_match_profile(gold_slots, predicted_slots)
    item["slot_error_bucket"] = slot_error_bucket(gold_slots, predicted_slots)
    errors: list[str] = []
    if not action_ok:
        errors.append("wrong_action")
    elif not slot_ok:
        gold_count = int(item.get("gold_slot_count", len(gold_slots)) or 0)
        predicted_count = int(item.get("predicted_slot_count", len(predicted_slots)) or 0)
        profile = item["slot_match_profile"]
        if gold_count == 0 and predicted_count > 0:
            errors.append("forbidden_slots_for_zero_slot_action")
        elif predicted_count < gold_count:
            errors.append("missing_slot")
        elif predicted_count > gold_count:
            errors.append("extra_slot")
        if not profile["strict_ordered"]:
            if profile["exact_unordered"]:
                errors.append("wrong_slot_order")
            elif profile["casefold_ordered"]:
                errors.append("case_or_whitespace_only")
            elif profile["casefold_unordered"]:
                errors.append("case_plus_slot_order")
            elif profile["punctuation_insensitive_ordered"]:
                errors.append("formatting_only")
            elif profile["punctuation_insensitive_unordered"]:
                errors.append("formatting_plus_slot_order")
            else:
                errors.append("wrong_slot_value")
    return errors or ["ast_success"]


def build_online_evidence_packets(
    localized: dict[str, Any], max_examples_per_bucket: int = 2,
    max_transition_examples: int = 48, max_action_examples: int = 72,
) -> dict[str, Any]:
    """Organize one rollout batch into transition and action-card evidence.

    Transition packets compare sibling route decisions. Action-card packets
    compare successes with action-correct slot failures for the same action.
    This is a deterministic repackaging step; it does not add labels beyond
    the already computed AST comparison.
    """
    events = (
        list(localized.get("events", []))
        + list(localized.get("slot_events", []))
        + list(localized.get("action_events", []))
    )
    by_transition: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"positive": [], "negative": []}
    )
    by_action: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"success": [], "slot_failure": [], "action_failure": []}
    )
    seen: set[tuple[str, int, str, str]] = set()
    action_seen: set[tuple[str, int, str]] = set()
    for raw in events:
        # Build a prompt-safe projection. The complete localized artifact is
        # still written to disk, but packets must never carry serialized
        # ReAct inputs/traces into the optimizer prompt.
        item = {
            key: raw.get(key)
            for key in (
                "edge_id", "conversation_id", "source_turn", "target_turn",
                "gold_action", "predicted_action", "predicted_target",
                "action_success", "slot_success", "slot_evaluable",
                "gold_slots", "predicted_slots", "gold_slot_count",
                "predicted_slot_count", "context", "slot_match_profile",
                "slot_error_bucket",
            )
            if key in raw
        }
        item["context"] = str(item.get("context", ""))[-1200:]
        action = str(item.get("gold_action", ""))
        if not action:
            continue
        item["error_types"] = _classify_online_evidence(item)
        item["action_correct"] = bool(item.get("action_success"))
        item["slot_correct"] = bool(item.get("slot_success")) if item.get("slot_evaluable", True) else None
        evidence_key = (
            str(item.get("conversation_id", "?")),
            int(item.get("target_turn", -1) or -1),
            str(item.get("edge_id", "")),
            ",".join(item["error_types"]),
        )
        if evidence_key in seen:
            continue
        seen.add(evidence_key)
        edge_id = str(item.get("edge_id", ""))
        if edge_id:
            bucket = "positive" if item["action_correct"] else "negative"
            by_transition[edge_id][bucket].append(item)
        action_identity = (
            str(item.get("conversation_id", "?")),
            int(item.get("target_turn", -1) or -1),
            ",".join(item["error_types"]),
        )
        if action_identity in action_seen:
            continue
        action_seen.add(action_identity)
        if item["action_correct"]:
            bucket = "success" if item.get("slot_correct") else "slot_failure"
        else:
            bucket = "action_failure"
        by_action[action][bucket].append(item)

    def trim(buckets: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
        # Keep both outcomes whenever possible; the on-disk batch artifact
        # remains complete, so this limit only controls prompt size.
        return {key: values[:max_examples_per_bucket] for key, values in buckets.items()}

    def bounded_group(
        group: dict[str, dict[str, list[dict[str, Any]]]], limit: int,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Apply a global prompt budget while retaining every outcome type."""
        trimmed = {key: trim(value) for key, value in sorted(group.items())}
        result: dict[str, dict[str, list[dict[str, Any]]]] = {}
        keys = list(trimmed)
        bucket_names = sorted({bucket for value in trimmed.values() for bucket in value})
        selected = 0
        index = 0
        while keys and selected < limit:
            key = keys[index % len(keys)]
            buckets = trimmed[key]
            bucket = bucket_names[index % len(bucket_names)]
            values = buckets.get(bucket, [])
            if values:
                result.setdefault(key, {name: [] for name in bucket_names})[bucket].append(values.pop(0))
                selected += 1
            index += 1
            if index > max(1, len(keys) * max(1, len(bucket_names)) * max_examples_per_bucket * 2):
                break
        return result

    transition_packet = bounded_group(by_transition, max_transition_examples)
    action_packet = bounded_group(by_action, max_action_examples)
    actual_examples = sum(
        len(items)
        for group in (transition_packet, action_packet)
        for buckets in group.values()
        for items in buckets.values()
    )
    return {
        "transition": transition_packet,
        "action_card": action_packet,
        "counts": {
            "transition_edges": len(transition_packet),
            "actions": len(action_packet),
            "raw_examples": len(events),
            "examples": actual_examples,
            "serialized_chars": len(json.dumps({
                "transition": transition_packet, "action_card": action_packet,
            }, ensure_ascii=False)),
        },
    }


def accumulate_online_evidence(
    state: dict[str, Any], packets: dict[str, Any], batch_index: int,
    max_examples_per_bucket: int = 12,
) -> None:
    """Accumulate bounded semantic evidence while preserving full batch files."""
    pool = state.setdefault("evidence_pool", {})
    transition_pool = pool.setdefault("transition", {})
    action_pool = pool.setdefault("action_card", {})
    unresolved = pool.setdefault("unresolved", [])

    def add(target: dict[str, Any], key: str, buckets: dict[str, list[dict[str, Any]]]) -> None:
        record = target.setdefault(key, {bucket: [] for bucket in buckets})
        for bucket, values in buckets.items():
            existing = record.setdefault(bucket, [])
            known = {
                (str(item.get("conversation_id", "?")), int(item.get("target_turn", -1) or -1))
                for item in existing
            }
            for value in values:
                item = dict(value)
                item["batch_index"] = batch_index
                identity = (str(item.get("conversation_id", "?")), int(item.get("target_turn", -1) or -1))
                if identity not in known:
                    existing.append(item)
                    known.add(identity)
            del existing[max_examples_per_bucket:]

    for key, buckets in packets.get("transition", {}).items():
        add(transition_pool, key, buckets)
    for key, buckets in packets.get("action_card", {}).items():
        add(action_pool, key, buckets)
    for action, buckets in packets.get("action_card", {}).items():
        for item in buckets.get("action_failure", []):
            unresolved.append({
                "batch_index": batch_index,
                "action": action,
                "conversation_id": item.get("conversation_id"),
                "target_turn": item.get("target_turn"),
                "reason": "action failure is not valid slot-policy evidence",
            })
    del unresolved[max_examples_per_bucket * max(1, len(action_pool)):]


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

    def compact_case(case: dict[str, Any]) -> dict[str, Any]:
        """Expose dialogue supervision, never serialized ReAct/state payloads."""
        return {
            "conversation_id": case.get("conversation_id"),
            "gold_action": case.get("gold_action"),
            "predicted_action": case.get("predicted_action"),
            "action_success": bool(case.get("action_success")),
            "context": str(case.get("context", ""))[-1200:],
        }

    def compact(item: dict[str, Any]) -> dict[str, Any]:
        positives = [
            compact_case(case) for case in item.get("evidence", [])
            if case.get("action_success")
        ][:max_cases]
        negatives = [
            compact_case(case) for case in item.get("evidence", [])
            if not case.get("action_success")
        ][:max_cases]
        return {
            "edge_id": _edge_id(item["source"], item["target"]),
            "source_action": item.get("source_action"),
            "target_action": item.get("target_action"),
            "guard": item.get("guard"),
            "positive_cases": positives,
            "negative_cases": negatives,
        }

    return {
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
                if not str(raw or "").strip() and os.getenv("SKILLMINING_STOP_ON_ERROR") == "1":
                    raise RuntimeError(
                        f"guard_induction returned an empty response (edge_id={edge_id})"
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


def _react_model_output_projection(trace: Any, max_chars: int = 1200) -> dict[str, Any]:
    """Project a ReAct trace to model outputs useful for refinement.

    The raw trace is retained in rollout artifacts for debugging, but it is
    deliberately not part of any optimizer prompt.  Tool inputs, duplicated
    conversation messages, and framework metadata add noise and can dominate
    the context window.  Keep only the generated retrieval query/result and
    the model's generated prediction when those fields are available.
    """
    projection: dict[str, Any] = {"reference_query": "", "retrieval_content": []}
    if not isinstance(trace, list):
        return projection
    for step in trace:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action", ""))
        action_input = step.get("action_input")
        if action in {"retrieve_reference", "retrieve_action_card", "retrieve_slot_policy"}:
            if not projection["reference_query"]:
                if isinstance(action_input, dict):
                    projection["reference_query"] = str(action_input.get("query", ""))[:300]
                elif action_input:
                    projection["reference_query"] = str(action_input)[:300]
            # Different agent/runtime versions call this field result,
            # observation, output, or tool_output.  Accept all of them but
            # serialize only the returned content, never the tool arguments.
            result = next((step.get(key) for key in
                           ("result", "observation", "output", "tool_output", "action_output")
                           if step.get(key) is not None), None)
            if result:
                projection["retrieval_content"].append({
                    "action": action,
                    "content": str(result)[:max_chars],
                })
    return projection


def _plan_resource_lookups(
    compact_supervision: list[dict[str, Any]], graph_edges: list[dict[str, Any]],
    skill: str, model: str, max_retries: int, response_logger: Any = None,
    workflow_id: str | None = None,
) -> tuple[list[dict[str, Any]], str, str, str]:
    """Ask the optimizer which resources it wants before exposing contents."""
    # The planner only needs the model's decision surface.  In particular,
    # never serialize the raw ReAct trajectory here: it contains tool inputs,
    # duplicated dialogue, and implementation details that are irrelevant to
    # deciding which resource section to retrieve.
    planner_supervision = [{
        key: row.get(key)
        for key in (
            "conversation_id", "turn_index", "prediction", "predicted_action",
            "predicted_slots", "gold", "action_correct", "slot_correct",
            "ast_outcome", "reference_query", "retrieval_content",
        )
        if key in row
    } for row in compact_supervision[:160]]
    prompt = _RESOURCE_LOOKUP_PLANNER_PROMPT.format(
        skill=skill or "[empty]",
        rollout_supervision=json.dumps(planner_supervision, ensure_ascii=False, indent=2),
        graph_edges=json.dumps(graph_edges[:240], ensure_ascii=False),
    )
    from llm import chat, resolve_config
    cfg = resolve_config(model=model)
    raw, payload, last_error = "", {}, ""
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            raw = _online_refinement_chat(
                [{"role": "user", "content": prompt}], model=cfg["model"],
                api_key=cfg["api_key"], base_url=cfg["base_url"], temperature=0.0,
                response_logger=response_logger, call_tag="online_resource_planner",
                workflow_id=workflow_id,
            ).strip()
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
    return planned, prompt, raw, last_error


def _online_refinement_chat(messages: list[dict[str, str]], *, model: str,
                             api_key: str | None = None, base_url: str | None = None,
                             temperature: float = 0.0, response_logger: Any = None,
                             call_tag: str = "online_resource_reflection",
                             workflow_id: str | None = None) -> str:
    """Call one explicitly selected workflow without mutating process env."""
    if workflow_id:
        import copy
        import config as project_config
        import llm
        workflow_config = copy.deepcopy(project_config.LLM_CONFIG)
        workflow_config["provider"] = "workflow"
        workflow_config["workflow_id"] = workflow_id
        raw_response = llm.chat(
            messages, model=model, temperature=temperature, config=workflow_config,
            response_logger=response_logger, call_tag=call_tag,
        )
        if not str(raw_response or "").strip() and os.getenv("SKILLMINING_STOP_ON_ERROR") == "1":
            raise RuntimeError(
                "Workflow optimizer returned an empty response"
                f" (workflow_id={workflow_id}, call_tag={call_tag})"
            )
        return raw_response or ""
    from llm import chat
    raw_response = chat(messages, model=model, api_key=api_key, base_url=base_url,
                        temperature=temperature, response_logger=response_logger,
                        call_tag=call_tag)
    if not str(raw_response or "").strip() and os.getenv("SKILLMINING_STOP_ON_ERROR") == "1":
        raise RuntimeError(
            f"LLM returned an empty response (workflow_id=<config.py>, call_tag={call_tag})"
        )
    return raw_response or ""


def autonomous_resource_reflection(
    state: dict[str, Any], rollout_supervision: list[dict[str, Any]], skill: str,
    reference: str, action_rules: str, slot_policies: str, model: str,
    max_retries: int = 3, response_logger: Any = None,
    evidence_packets: dict[str, Any] | None = None,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    """Let the LLM select and apply bounded resource updates for one batch."""
    def _token_estimate(text: str) -> int:
        """Conservative local estimate when a workflow omits token usage."""
        text = str(text or "")
        cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
        return cjk_chars + max(0, len(text) - cjk_chars) // 4

    def _supervision_outcome(row: dict[str, Any]) -> str:
        """Compute the per-turn joint AST outcome from action and slots."""
        gold = row.get("gold")
        if not isinstance(gold, dict) or not gold.get("gold_action"):
            # Text metrics are aggregate-level in this runner, so exact string
            # equality would be a misleading per-turn success signal.
            return "unscored"
        action_ok = str(row.get("predicted_action", "")) == str(gold["gold_action"])
        predicted_slots = row.get("predicted_slots", [])
        gold_slots = gold.get("gold_slots", [])
        slots_ok = list(predicted_slots) == list(gold_slots)
        return "success" if action_ok and slots_ok else "failure"

    supervised_rows = [
        row for row in rollout_supervision
        if row.get("gold") is not None
    ]
    compact_supervision = [{
        "conversation_id": row.get("conversation_id"), "turn_index": row.get("turn_index"),
        "target_type": row.get("target_type", "action"), "context": str(row.get("context", ""))[-2000:],
        "prediction": str(row.get("prediction", ""))[:500],
        "predicted_action": row.get("predicted_action", ""),
        "predicted_slots": row.get("predicted_slots", []), "gold": row.get("gold"),
        "action_correct": _supervision_outcome(row) == "success" or (
            isinstance(row.get("gold"), dict)
            and str(row.get("predicted_action", "")) == str(row["gold"].get("gold_action", ""))
        ),
        "slot_correct": (
            list(row.get("predicted_slots", [])) == list(row["gold"].get("gold_slots", []))
            if isinstance(row.get("gold"), dict) else None
        ),
        "gold_response": str(row.get("gold_response", ""))[:500],
        "ast_outcome": _supervision_outcome(row),
        "evidence_outcome": _supervision_outcome(row),
        **_react_model_output_projection(row.get("react_trace")),
    } for row in supervised_rows]
    packets = evidence_packets or {"transition": {}, "action_card": {}, "counts": {}}

    def prompt_packet_view(source: dict[str, Any], max_examples: int = 60, max_chars: int = 90000) -> dict[str, Any]:
        """Create a hard-bounded, trace-free packet view for the LLM prompt."""
        result: dict[str, Any] = {"transition": {}, "action_card": {}}
        selected = 0
        for group_name in ("transition", "action_card"):
            group = source.get(group_name, {}) if isinstance(source, dict) else {}
            for key, buckets in group.items() if isinstance(group, dict) else []:
                for bucket, values in buckets.items() if isinstance(buckets, dict) else []:
                    for raw in values if isinstance(values, list) else []:
                        if selected >= max_examples:
                            break
                        item = {
                            field: raw.get(field)
                            for field in (
                                "edge_id", "conversation_id", "source_turn", "target_turn",
                                "gold_action", "predicted_action", "predicted_target",
                                "action_success", "slot_success", "slot_evaluable",
                                "gold_slots", "predicted_slots", "gold_slot_count",
                                "predicted_slot_count",
                            )
                            if field in raw
                        }
                        item["context"] = str(raw.get("context", ""))[-600:]
                        item["error_types"] = raw.get("error_types", [])
                        result.setdefault(group_name, {}).setdefault(key, {}).setdefault(bucket, []).append(item)
                        selected += 1
                    if selected >= max_examples:
                        break
                if selected >= max_examples:
                    break
            if selected >= max_examples:
                break
        encoded = json.dumps(result, ensure_ascii=False, indent=2)
        # Context excerpts are already bounded; this final guard protects
        # against unexpectedly large slot/prediction fields in legacy data.
        if len(encoded) > max_chars:
            encoded = encoded[:max_chars]
            result = {"truncated_packet_json": encoded}
        result["counts"] = {
            "selected_examples": selected,
            "serialized_chars": len(json.dumps(result, ensure_ascii=False)),
        }
        return result

    prompt_packets = prompt_packet_view(packets)
    # The packets carry the contextual examples. Keep only a compact index in
    # the main reflection prompt; otherwise every turn is serialized twice.
    prompt_supervision = compact_supervision
    if evidence_packets:
        prompt_supervision = [{
            key: row.get(key)
            for key in (
                "conversation_id", "turn_index", "predicted_action", "predicted_slots",
                "gold", "action_correct", "slot_correct", "ast_outcome", "reference_query",
                "retrieval_content",
            )
        } for row in compact_supervision[:160]]
    graph_edges = [{
        "edge_id": edge_id, "source_action": edge["source_action"],
        "target_action": edge["target_action"], "kind": edge["kind"],
        "visibility": edge.get("visibility", "reference"),
        "offline_support": int(edge.get("offline_support", 0) or 0),
        "gold_support": int(edge.get("gold_support", 0) or 0),
        "rollout_success": int(edge.get("rollout_success", 0) or 0),
        "rollout_failure": int(edge.get("rollout_failure", 0) or 0),
        "guard_status": edge.get("guard_status", "pending"),
    } for edge_id, edge in state.get("edges", {}).items()]
    resources = {"reference": reference, "action_rules": action_rules, "slot_policies": slot_policies}
    lookups, planner_prompt, planner_raw, planner_error = _plan_resource_lookups(
        compact_supervision, graph_edges, skill, model, max_retries, response_logger,
        workflow_id=workflow_id,
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
        rollout_supervision=json.dumps(prompt_supervision, ensure_ascii=False, indent=2),
        evidence_packets=json.dumps(prompt_packets, ensure_ascii=False, indent=2),
    )
    if workflow_id:
        cfg = {"model": model, "api_key": None, "base_url": None}
    else:
        from llm import resolve_config
        cfg = resolve_config(model=model)
    raw, payload, last_error = "", {}, ""
    reflection_attempts: list[dict[str, Any]] = []
    for attempt in range(1, max(1, max_retries) + 1):
        reflection_prompt = prompt
        try:
            if attempt > 1:
                reflection_prompt += (
                    "\n\nRETRY REQUIREMENT: Your previous response was not valid JSON. "
                    "Return exactly one complete JSON object, with double-quoted keys "
                    "and strings, no markdown fences, no comments, and no text before "
                    "or after the object."
                )
            raw = _online_refinement_chat(
                [{"role": "user", "content": reflection_prompt}], model=cfg["model"],
                api_key=cfg["api_key"], base_url=cfg["base_url"], temperature=0.0,
                response_logger=response_logger, call_tag="online_resource_reflection",
                workflow_id=workflow_id,
            ).strip()
            text = "\n".join(raw.splitlines()[1:-1]) if raw.startswith("```") else raw
            start = text.find("{")
            if start < 0:
                raise json.JSONDecodeError("No JSON object found", text, 0)
            payload, _ = json.JSONDecoder().raw_decode(text[start:])
            if not isinstance(payload, dict):
                raise json.JSONDecodeError("Reflection JSON must be an object", text, start)
            if isinstance(payload.get("updates"), list):
                reflection_attempts.append({
                    "attempt": attempt, "status": "parsed",
                    "input_chars": len(reflection_prompt),
                    "input_token_estimate": _token_estimate(reflection_prompt),
                    "output_chars": len(raw),
                    "output_token_estimate": _token_estimate(raw),
                    "json_start": start,
                    "output_tail_chars": len(raw) - raw.rfind("\n"),
                })
                break
            raise json.JSONDecodeError("Reflection JSON has no updates list", text, start)
        except Exception as exc:
            # A non-empty response with malformed JSON is a recoverable model
            # formatting error. Retry it even in strict diagnostic mode; only
            # an empty response or a final exhausted retry should terminate.
            if not isinstance(exc, json.JSONDecodeError) and os.getenv("SKILLMINING_STOP_ON_ERROR") == "1":
                raise RuntimeError(
                    f"online_resource_reflection failed (workflow_id={workflow_id or '<config.py>'}, "
                    f"attempt={attempt}): {exc}"
                ) from exc
            payload = {}
            last_error = repr(exc)
            json_error_position = getattr(exc, "pos", None)
            reflection_attempts.append({
                "attempt": attempt, "status": "failed",
                "input_chars": len(reflection_prompt),
                "input_token_estimate": _token_estimate(reflection_prompt),
                "output_chars": len(raw),
                "output_token_estimate": _token_estimate(raw),
                "json_error_position": json_error_position,
                "error": last_error,
                # Do not duplicate raw output in the diagnostic record; raw_response
                # remains the single inspectable source of full model output.
                "output_prefix": raw[:200],
                "output_suffix": raw[-400:],
            })
        if attempt < max(1, max_retries):
            time.sleep(float(2 ** (attempt - 1)))

    if not payload and last_error and os.getenv("SKILLMINING_STOP_ON_ERROR") == "1":
        raise RuntimeError(
            f"online_resource_reflection returned invalid JSON after {max(1, max_retries)} attempts "
            f"(workflow_id={workflow_id or '<config.py>'}): {last_error}"
        )

    proposed_skill_operations, rejected_skill_operations = [], []
    for item in payload.get("skill_operations", []):
        if not isinstance(item, dict):
            rejected_skill_operations.append({"operation": item, "reason": "not_an_object"})
            continue
        op = str(item.get("op", "")).strip().lower()
        match_text = str(item.get("match_text", ""))
        new_text = str(item.get("new_text", ""))
        rationale = str(item.get("rationale", "")).strip()
        occurrence = item.get("occurrence")
        if occurrence not in (None, ""):
            try:
                occurrence = int(occurrence)
            except (TypeError, ValueError):
                occurrence = None
        if op == "upsert": op = "replace"
        if op not in {"replace", "insert_before", "insert_after", "delete"} or not match_text:
            rejected_skill_operations.append({"operation": item, "reason": "invalid_dynamic_skill_operation"})
            continue
        if (op == "delete" and new_text) or (op != "delete" and not new_text):
            rejected_skill_operations.append({"operation": item, "reason": "invalid_dynamic_skill_content"})
            continue
        if op == "delete" and not rationale:
            rejected_skill_operations.append({
                "operation": item,
                "reason": "delete_requires_conflict_rationale",
            })
            continue
        proposed_skill_operations.append({
            "operation_id": str(item.get("operation_id", "")).strip(),
            "op": op, "resource": str(item.get("resource", "")).strip(),
            "edge_id": str(item.get("edge_id", "")).strip(),
            "action": str(item.get("action", "")).strip(),
            "match_text": match_text, "new_text": new_text,
            "occurrence": occurrence, "rationale": rationale,
        })

    valid_actions = {str(node["label"]) for node in state.get("nodes", [])}
    accepted, rejected = [], []
    for update in payload.get("updates", []):
        if not isinstance(update, dict):
            continue
        resource, content = str(update.get("resource", "")), str(update.get("content", "")).strip()
        operation = str(update.get("op", "upsert")).strip().lower()
        status = str(update.get("status", "uncertain")).lower()
        if resource not in {"transition_guard", "action_node", "action_rule", "slot_policy", "reference"} or operation not in {"upsert", "delete"}:
            rejected.append({"update": update, "reason": "unsupported_resource_or_empty_content"})
            continue
        if operation == "delete" and resource not in {"transition_guard", "action_node", "action_rule"}:
            rejected.append({"update": update, "reason": "delete_not_supported_for_resource"})
            continue
        if operation == "upsert" and not content:
            rejected.append({"update": update, "reason": "empty_upsert_content"})
            continue
        if resource == "transition_guard":
            edge_id = str(update.get("edge_id", ""))
            if edge_id not in state.get("edges", {}):
                rejected.append({"update": update, "reason": "unknown_edge"})
                continue
            edge = state["edges"][edge_id]
            edge["guard"] = content if operation == "upsert" else ""
            edge["guard_status"] = "resolved" if operation == "upsert" and status == "resolved" else "uncertain"
            # The optimizer, not a confidence threshold or topology heuristic,
            # decides whether this documented transition belongs in skill.
            if operation == "upsert" and edge["guard_status"] == "resolved":
                if edge.get("kind") == "candidate_branch":
                    edge["kind"] = "promoted_branch"
                edge["visibility"] = "skill"
            else:
                edge["visibility"] = "reference"
        elif resource == "action_node":
            action = str(update.get("action", "")).strip()
            edge_id = str(update.get("edge_id", "")).strip()
            if action not in valid_actions:
                rejected.append({"update": update, "reason": "unknown_action"})
                continue
            if edge_id and edge_id not in state.get("edges", {}):
                rejected.append({"update": update, "reason": "unknown_edge"})
                continue
            node = next((item for item in state.get("nodes", []) if str(item.get("label")) == action), None)
            if node is None:
                rejected.append({"update": update, "reason": "unknown_action_node"})
                continue
            promotions = node.setdefault("online_promotions", [])
            if operation == "upsert":
                record = {
                    "action": action, "edge_id": edge_id, "content": content,
                    "status": "resolved" if status == "resolved" else "uncertain",
                    "rationale": update.get("rationale", ""),
                }
                promotions[:] = [item for item in promotions if item.get("edge_id") != edge_id]
                promotions.append(record)
                state["node_promotions"] = [
                    item for item in state.get("node_promotions", [])
                    if not (item.get("action") == action and item.get("edge_id") == edge_id)
                ]
                state["node_promotions"].append(record)
                if edge_id and status == "resolved":
                    edge = state["edges"][edge_id]
                    if edge.get("kind") == "candidate_branch":
                        edge["kind"] = "promoted_branch"
                    edge["visibility"] = "skill"
            else:
                promotions[:] = [item for item in promotions if item.get("edge_id") != edge_id]
                state["node_promotions"] = [
                    item for item in state.get("node_promotions", [])
                    if not (item.get("action") == action and item.get("edge_id") == edge_id)
                ]
                if edge_id and edge_id in state.get("edges", {}):
                    edge = state["edges"][edge_id]
                    if edge.get("kind") != "backbone":
                        edge["visibility"] = "reference"
        elif resource in {"action_rule", "slot_policy"}:
            action = str(update.get("action", ""))
            if action not in valid_actions:
                rejected.append({"update": update, "reason": "unknown_action"})
                continue
            bucket = "action_rules" if resource == "action_rule" else "slot_policies"
            record = state.setdefault(bucket, {}).setdefault(action, {"action": action})
            record["policy" if resource == "slot_policy" else "rule"] = content if operation == "upsert" else ""
            record["status"] = "resolved" if status == "resolved" else "uncertain"
        else:
            edge_id = str(update.get("edge_id", ""))
            if edge_id and edge_id not in state.get("edges", {}):
                rejected.append({"update": update, "reason": "unknown_reference_edge"})
                continue
            state.setdefault("reference_notes", []).append({
                "edge_id": edge_id, "content": content, "status": status,
                "rationale": update.get("rationale", ""),
            })
        accepted.append(update)
    return {
        "planner_prompt": planner_prompt, "planner_prompt_chars": len(planner_prompt),
        "planner_raw_response": planner_raw, "lookups": lookups,
        "retrieved_resources": retrieved, "planner_error": planner_error,
        "prompt": prompt, "prompt_chars": len(prompt), "raw_response": raw,
        "reflection_attempts": reflection_attempts,
        "reflection_io": {
            "input_chars": len(prompt),
            "input_token_estimate": _token_estimate(prompt),
            "last_output_chars": len(raw),
            "last_output_token_estimate": _token_estimate(raw),
            "retry_count": len(reflection_attempts),
        },
        "accepted": accepted, "rejected": rejected,
        "proposed_skill_operations": proposed_skill_operations,
        "rejected_skill_operations": rejected_skill_operations,
        "model_decision": str(payload.get("decision", "")),
        "model_no_update_reason": str(payload.get("no_update_reason", "")),
        "evidence_packet_counts": packets.get("counts", {}),
        "prompt_component_chars": {
            "skill": len(skill or ""),
            "retrieved_resources": len(json.dumps(retrieved, ensure_ascii=False)),
            "graph_edges": len(json.dumps(graph_edges, ensure_ascii=False)),
            "rollout_supervision": len(json.dumps(prompt_supervision, ensure_ascii=False)),
            "evidence_packets": len(json.dumps(prompt_packets, ensure_ascii=False)),
        },
        "error": last_error if not payload else "",
    }


def _rollout_record_features(record: dict[str, Any]) -> set[str]:
    sample = record.get("sample", {})
    row = record.get("result", {})
    sequence = sample.get("prefix_action_sequence") or []
    features = {f"seq:{index}:{action}" for index, action in enumerate(sequence[-5:])}
    for name in ("source_action", "target_action"):
        value = str(sample.get(name, ""))
        if value: features.add(f"{name}:{value}")
    predicted = str(row.get("predicted_action", ""))
    if predicted: features.add(f"predicted:{predicted}")
    features.update(f"ctx:{token}" for token in _tokenize_for_lookup(str(row.get("context", ""))))
    return features


def build_post_rollout_batches(
    records: list[dict[str, Any]], max_batch_size: int = 12,
) -> list[list[dict[str, Any]]]:
    """Cluster completed rollouts by trajectory, semantics, graph locality and outcome.

    There are no fixed success/failure quotas. A greedy similarity graph keeps
    closely related traces together while allowing cross-source batches when
    their prefixes, contexts or predicted alternatives are genuinely similar.
    """
    pending = list(records)
    features = {id(item): _rollout_record_features(item) for item in pending}

    def similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
        lf, rf = features[id(left)], features[id(right)]
        lexical = len(lf & rf) / max(len(lf | rf), 1)
        ls, rs = left.get("sample", {}), right.get("sample", {})
        graph = 0.0
        if ls.get("source_action") == rs.get("source_action"): graph += 0.35
        if ls.get("target_action") == rs.get("target_action"): graph += 0.20
        lp, rp = left.get("result", {}), right.get("result", {})
        if lp.get("predicted_action") == rp.get("predicted_action"): graph += 0.10
        left_ok = lp.get("predicted_action") == ls.get("target_action")
        right_ok = rp.get("predicted_action") == rs.get("target_action")
        outcome = 0.10 if left_ok == right_ok else 0.16  # contrast is also informative
        return lexical + graph + outcome

    batches: list[list[dict[str, Any]]] = []
    while pending:
        seed = pending.pop(0)
        ranked = sorted(((similarity(seed, item), item) for item in pending),
                        key=lambda pair: pair[0], reverse=True)
        best = ranked[0][0] if ranked else 0.0
        adaptive_floor = max(0.08, best * 0.45)
        selected = [item for value, item in ranked if value >= adaptive_floor][
            :max(0, max_batch_size - 1)]
        chosen = {id(item) for item in selected}
        pending = [item for item in pending if id(item) not in chosen]
        batches.append([seed, *selected])
    return batches


def lookup_graph_neighborhood(
    state: dict[str, Any], nodes: list[str], radius: int = 1,
) -> dict[str, Any]:
    """Return a bounded JSON graph view around action labels or node ids."""
    label_to_id = {str(node.get("label")): str(node.get("id")) for node in state.get("nodes", [])}
    selected = {label_to_id.get(str(node), str(node)) for node in nodes if node}
    edges = state.get("edges", {})
    for _ in range(max(0, radius)):
        selected |= {
            endpoint
            for edge in edges.values()
            if str(edge.get("source")) in selected or str(edge.get("target")) in selected
            for endpoint in (str(edge.get("source")), str(edge.get("target")))
        }
    node_rows = [node for node in state.get("nodes", []) if str(node.get("id")) in selected]
    edge_rows = [{"edge_id": edge_id, **{
        key: edge.get(key) for key in (
            "source", "target", "source_action", "target_action", "kind", "visibility",
            "gold_support", "rollout_success", "rollout_failure", "guard", "guard_status",
        )
    }} for edge_id, edge in edges.items()
        if str(edge.get("source")) in selected and str(edge.get("target")) in selected]
    return {"nodes": node_rows, "edges": edge_rows}


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    start = text.find("{")
    if start < 0: return {}
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _chat_for_json_with_retries(
    messages: list[dict[str, str]], *, model: str, response_logger: Any,
    call_tag: str, workflow_id: str | None, max_retries: int = 3,
    retry_base_delay: float = 2.0,
) -> tuple[dict[str, Any], str, str, int]:
    """Return parsed JSON without letting a transient workflow error escape."""
    raw = ""
    last_error = ""
    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        try:
            raw = _online_refinement_chat(
                messages, model=model, api_key=None, base_url=None,
                temperature=0.0, response_logger=response_logger,
                call_tag=call_tag, workflow_id=workflow_id,
            )
            payload = _parse_json_object(raw)
            if payload:
                return payload, raw, "", attempt
            last_error = "empty_or_invalid_json_response"
        except Exception as exc:
            last_error = repr(exc)
        if attempt < attempts and retry_base_delay > 0:
            time.sleep(float(retry_base_delay) * (2 ** (attempt - 1)))
    return {}, raw, last_error, attempts


def diagnose_rollout_batch(
    batch_id: str, records: list[dict[str, Any]], state: dict[str, Any], skill: str,
    model: str, response_logger: Any = None, workflow_id: str | None = None,
    reference: str = "", action_rules: str = "", slot_policies: str = "",
    max_retries: int = 3, retry_base_delay: float = 2.0,
) -> dict[str, Any]:
    nodes = sorted({str(record.get("sample", {}).get(key, ""))
                    for record in records for key in ("source_action", "target_action") if record.get("sample", {}).get(key)})
    local_graph = lookup_graph_neighborhood(state, nodes, radius=1)
    resource_query = " ".join(nodes)
    available_resources = []
    for resource_name, resource_text in (
        ("reference", reference), ("action_rules", action_rules),
        ("slot_policies", slot_policies),
    ):
        if resource_text:
            available_resources.extend(_resource_lookup_sections(
                resource_name, resource_text, resource_query, 2,
            ))
    prompt_records = []
    for record in records:
        sample = record.get("sample", {})
        result = record.get("result", {})
        gold_slots = [str(value) for value in sample.get("gold_slots", [])]
        predicted_slots = [str(value) for value in result.get("predicted_slots", [])]
        slot_profile = slot_match_profile(gold_slots, predicted_slots)
        action_ok = str(result.get("predicted_action", "")) == str(sample.get("target_action", ""))
        prompt_records.append({
            "sample_id": sample.get("sample_id"),
            "prefix_action_sequence": sample.get("prefix_action_sequence", []),
            "source_action": sample.get("source_action"),
            "gold_target": sample.get("target_action"),
            "gold_slots": gold_slots,
            "predicted_action": result.get("predicted_action"),
            "predicted_slots": predicted_slots,
            "action_correct": action_ok,
            "slot_exact_correct": bool(slot_profile["strict_ordered"]),
            "slot_error_bucket": (
                "strict_correct" if slot_profile["strict_ordered"]
                else slot_error_bucket(gold_slots, predicted_slots)
            ),
            "slot_match_profile": slot_profile,
            "context": str(result.get("context", ""))[-1800:],
            **_react_model_output_projection(result.get("react_trace")),
        })
    prompt = _BATCH_ROOT_CAUSE_PROMPT.format(
        skill=skill, local_graph=json.dumps(local_graph, ensure_ascii=False),
        available_resources=json.dumps(available_resources, ensure_ascii=False, indent=2),
        rollouts=json.dumps(prompt_records, ensure_ascii=False, indent=2),
    )
    report, raw, error, attempts = _chat_for_json_with_retries(
        [{"role": "user", "content": prompt}], model=model,
        response_logger=response_logger, call_tag="online_batch_root_cause",
        workflow_id=workflow_id, max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )
    report.update({"batch_id": batch_id, "sample_ids": [item["sample"]["sample_id"] for item in records],
                   "local_graph": local_graph, "raw_response": raw,
                   "status": "error" if error else "ok", "error": error,
                   "attempts": attempts,
                   "trajectory_exemplars": (
                       [item for item in prompt_records if item["action_correct"] and item["slot_exact_correct"]][:2]
                       + [item for item in prompt_records if not (
                           item["action_correct"] and item["slot_exact_correct"]
                       )][:4]
                   )})
    return report


def group_batch_reports(reports: list[dict[str, Any]], max_reports: int = 16) -> list[list[dict[str, Any]]]:
    """Group reports by overlapping/neighboring graph footprint, capped at 16."""
    def footprint(report: dict[str, Any]) -> set[str]:
        values = set((report.get("graph_footprint") or {}).get("nodes", []))
        for node in (report.get("local_graph") or {}).get("nodes", []):
            values.add(str(node.get("id", ""))); values.add(str(node.get("label", "")))
        return {value for value in values if value}

    def score(left: dict[str, Any], right: dict[str, Any]) -> float:
        ln, rn = footprint(left), footprint(right)
        graph = len(ln & rn) / max(len(ln | rn), 1)
        lt = _tokenize_for_lookup(str(left.get("summary", "")))
        rt = _tokenize_for_lookup(str(right.get("summary", "")))
        semantic = len(lt & rt) / max(len(lt | rt), 1)
        return 2.0 * graph + semantic

    pending = list(reports); groups = []
    while pending:
        seed = pending.pop(0)
        ranked = sorted(((score(seed, report), report) for report in pending),
                        key=lambda item: item[0], reverse=True)
        selected = [report for value, report in ranked
                    if value > 0.0][:max(0, max_reports - 1)]
        chosen = {id(item) for item in selected}
        pending = [item for item in pending if id(item) not in chosen]
        groups.append([seed, *selected])
    return groups


def reflect_batch_report_group(
    reports: list[dict[str, Any]], state: dict[str, Any], skill: str, model: str,
    response_logger: Any = None, workflow_id: str | None = None,
    max_retries: int = 3, retry_base_delay: float = 2.0,
) -> dict[str, Any]:
    reports = reports[:16]
    nodes = sorted({node for report in reports for node in
                    (report.get("graph_footprint") or {}).get("nodes", [])})
    nodes = sorted(set(nodes) | {
        str(node.get("label") or node.get("id"))
        for report in reports for node in (report.get("local_graph") or {}).get("nodes", [])
        if node.get("label") or node.get("id")
    })
    local_graph = lookup_graph_neighborhood(state, nodes, radius=1)
    compact = [{key: report.get(key) for key in (
        "batch_id", "summary", "root_causes", "graph_footprint", "candidate_updates",
        "candidate_skill_operations", "unresolved_questions")}
        for report in reports]
    prompt = _GROUP_REFLECTION_PROMPT.format(skill=skill,
        local_graph=json.dumps(local_graph, ensure_ascii=False),
        batch_reports=json.dumps(compact, ensure_ascii=False, indent=2))
    result, raw, error, attempts = _chat_for_json_with_retries(
        [{"role": "user", "content": prompt}], model=model,
        response_logger=response_logger, call_tag="online_group_reflection",
        workflow_id=workflow_id, max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )
    result.update({"batch_ids": [report.get("batch_id") for report in reports],
                   "local_graph": local_graph, "raw_response": raw,
                   "prompt_chars": len(prompt), "status": "error" if error else "ok",
                   "error": error, "attempts": attempts})
    return result


def trace2skill_hybrid_reflect_report_group(
    reports: list[dict[str, Any]], state: dict[str, Any], skill: str, model: str,
    response_logger: Any = None, workflow_id: str | None = None,
    map_batch_size: int = 4,
    max_retries: int = 3, retry_base_delay: float = 2.0,
) -> dict[str, Any]:
    """Run local success/failure MAP-REDUCE, then emit native graph updates."""
    reports = reports[:16]
    if map_batch_size <= 0:
        raise ValueError("map_batch_size must be positive")
    nodes = sorted({
        str(node.get("label") or node.get("id"))
        for report in reports
        for node in (report.get("local_graph") or {}).get("nodes", [])
        if node.get("label") or node.get("id")
    } | {
        str(node)
        for report in reports
        for node in (report.get("graph_footprint") or {}).get("nodes", [])
        if node
    })
    local_graph = lookup_graph_neighborhood(state, nodes, radius=1)
    exemplars = [
        exemplar for report in reports
        for exemplar in report.get("trajectory_exemplars", [])
        if isinstance(exemplar, dict)
    ]
    successes = [item for item in exemplars
                 if item.get("action_correct") and item.get("slot_exact_correct")]
    failures = [item for item in exemplars
                if not (item.get("action_correct") and item.get("slot_exact_correct"))]

    def distill(prompt_template: str, trajectories: list[dict[str, Any]], tag: str) -> dict[str, Any]:
        if not trajectories:
            return {"status": "skipped", "summary": "no matching trajectories"}
        prompt = prompt_template.format(
            local_graph=json.dumps(local_graph, ensure_ascii=False),
            trajectories=json.dumps(trajectories[:24], ensure_ascii=False, indent=2),
        )
        payload, raw, error, attempts = _chat_for_json_with_retries(
            [{"role": "user", "content": prompt}], model=model,
            response_logger=response_logger, call_tag=tag, workflow_id=workflow_id,
            max_retries=max_retries, retry_base_delay=retry_base_delay,
        )
        payload.update({"status": "error" if error else "ok", "error": error,
                        "attempts": attempts, "raw_response": raw})
        return payload

    success_distillation = distill(
        _TRACE2SKILL_SUCCESS_DISTILL_PROMPT, successes,
        "online_trace2skill_success_distill",
    )
    failure_distillation = distill(
        _TRACE2SKILL_FAILURE_DISTILL_PROMPT, failures,
        "online_trace2skill_failure_distill",
    )
    map_outputs: list[dict[str, Any]] = []
    for start in range(0, len(reports), map_batch_size):
        report_batch = reports[start:start + map_batch_size]
        compact = [{key: report.get(key) for key in (
            "batch_id", "summary", "root_causes", "graph_footprint",
            "candidate_updates", "candidate_skill_operations",
            "unresolved_questions", "trajectory_exemplars",
        )} for report in report_batch]
        prompt = _TRACE2SKILL_LOCAL_MAP_PROMPT.format(
            skill=skill,
            local_graph=json.dumps(local_graph, ensure_ascii=False),
            success_distillation=json.dumps(success_distillation, ensure_ascii=False, indent=2),
            failure_distillation=json.dumps(failure_distillation, ensure_ascii=False, indent=2),
            batch_reports=json.dumps(compact, ensure_ascii=False, indent=2),
        )
        mapped, raw, error, attempts = _chat_for_json_with_retries(
            [{"role": "user", "content": prompt}], model=model,
            response_logger=response_logger,
            call_tag="online_trace2skill_local_map", workflow_id=workflow_id,
            max_retries=max_retries, retry_base_delay=retry_base_delay,
        )
        mapped.update({
            "map_index": len(map_outputs) + 1,
            "batch_ids": [report.get("batch_id") for report in report_batch],
            "raw_response": raw,
            "prompt_chars": len(prompt),
            "status": "error" if error else "ok",
            "error": error,
            "attempts": attempts,
        })
        map_outputs.append(mapped)

    successful_maps = [mapped for mapped in map_outputs if mapped.get("status") == "ok"]
    reduce_payload = [{key: mapped.get(key) for key in (
        "map_index", "batch_ids", "summary", "root_causes", "graph_footprint",
        "candidate_updates", "candidate_skill_operations", "preserved_successes",
        "unresolved_questions",
    )} for mapped in successful_maps]
    if not reduce_payload:
        return {
            "mode": "trace2skill_hybrid", "status": "error",
            "error": "all_local_map_calls_failed",
            "batch_ids": [report.get("batch_id") for report in reports],
            "local_graph": local_graph, "map_outputs": map_outputs,
            "success_distillation": success_distillation,
            "failure_distillation": failure_distillation,
            "updates": [], "skill_operations": [],
        }
    prompt = _TRACE2SKILL_LOCAL_REDUCE_PROMPT.format(
        skill=skill,
        local_graph=json.dumps(local_graph, ensure_ascii=False),
        map_candidates=json.dumps(reduce_payload, ensure_ascii=False, indent=2),
    )
    result, raw, error, attempts = _chat_for_json_with_retries(
        [{"role": "user", "content": prompt}], model=model,
        response_logger=response_logger,
        call_tag="online_trace2skill_local_reduce", workflow_id=workflow_id,
        max_retries=max_retries, retry_base_delay=retry_base_delay,
    )
    result.update({
        "mode": "trace2skill_hybrid",
        "batch_ids": [report.get("batch_id") for report in reports],
        "local_graph": local_graph,
        "map_outputs": map_outputs,
        "success_distillation": success_distillation,
        "failure_distillation": failure_distillation,
        "raw_response": raw,
        "prompt_chars": len(prompt),
        "status": "error" if error else "ok",
        "error": error,
        "attempts": attempts,
    })
    return result


def apply_reflection_updates_to_state(
    state: dict[str, Any], updates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply final group-level semantic decisions to graph/resource state."""
    valid_actions = {str(node.get("label")) for node in state.get("nodes", [])}
    accepted, rejected = [], []
    for update in updates:
        resource = str(update.get("resource", "")); op = str(update.get("op", "upsert"))
        content = str(update.get("content", "")).strip()
        if resource == "transition_guard":
            edge_id = str(update.get("edge_id", "")); edge = state.get("edges", {}).get(edge_id)
            if edge is None:
                rejected.append({"update": update, "reason": "unknown_edge"}); continue
            edge["guard"] = content if op == "upsert" else ""
            edge["guard_status"] = str(update.get("status", "uncertain"))
            if op == "upsert" and edge["guard_status"] == "resolved":
                if edge.get("kind") == "candidate_branch": edge["kind"] = "promoted_branch"
                edge["visibility"] = "skill"
            elif edge.get("kind") != "backbone": edge["visibility"] = "reference"
        elif resource in {"action_rule", "slot_policy", "action_node"}:
            action = str(update.get("action", ""))
            if action not in valid_actions:
                rejected.append({"update": update, "reason": "unknown_action"}); continue
            bucket = "slot_policies" if resource == "slot_policy" else "action_rules"
            record = state.setdefault(bucket, {}).setdefault(action, {"action": action})
            record["policy" if bucket == "slot_policies" else "rule"] = content if op == "upsert" else ""
            record["status"] = str(update.get("status", "uncertain"))
        elif resource == "reference":
            state.setdefault("reference_notes", []).append({
                "edge_id": str(update.get("edge_id", "")), "content": content,
                "status": str(update.get("status", "uncertain")),
                "rationale": str(update.get("rationale", "")),
            })
        else:
            rejected.append({"update": update, "reason": "unsupported_resource"}); continue
        accepted.append(update)
    return accepted, rejected


def render_online_action_rules(state: dict[str, Any]) -> str:
    lines = ["# Online Action Rule Refinements", ""]
    for action, record in sorted(state.get("action_rules", {}).items()):
        rule = str(record.get("rule", "")).strip()
        if rule and record.get("status") == "resolved":
            lines.extend([f"#### `{action}`", rule, ""])
    return "\n".join(lines).rstrip() + "\n"


def apply_dynamic_skill_operations(
    skill: str, operations: list[dict[str, Any]], applied_operation_ids: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply model-authored, content-addressed edits to the current skill.

    Unlike the legacy anchor editor, this treats the complete skill as the
    editable artifact. Exact unique excerpts provide an optimistic-concurrency
    check without coupling online refinement to a particular compiler layout.
    """
    current = skill
    applied_operation_ids = applied_operation_ids if applied_operation_ids is not None else set()
    applied: list[dict[str, Any]] = []
    for operation in operations:
        op = str(operation.get("op", "")).strip().lower()
        if op == "upsert": op = "replace"
        operation_id = str(operation.get("operation_id", "")).strip() or hashlib.sha256(
            json.dumps(operation, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        match_text = str(operation.get("match_text", ""))
        new_text = str(operation.get("new_text", ""))
        occurrence = operation.get("occurrence")
        record = {
            "op": op, "operation_id": operation_id,
            "resource": str(operation.get("resource", "")),
            "edge_id": str(operation.get("edge_id", "")),
            "action": str(operation.get("action", "")),
            "match_text": match_text,
            "new_text": new_text,
            "occurrence": occurrence,
            "rationale": str(operation.get("rationale", "")),
        }
        try:
            if operation_id in applied_operation_ids:
                record["skipped"] = "operation_already_applied"
                applied.append(record)
                continue
            if op not in {"replace", "insert_before", "insert_after", "delete"}:
                raise ValueError("unsupported dynamic skill operation")
            if not match_text:
                raise ValueError("dynamic skill operation requires match_text")
            if op == "delete" and new_text:
                raise ValueError("delete operation must use an empty new_text")
            if op != "delete" and not new_text:
                raise ValueError("text operation requires new_text")
            occurrences = current.count(match_text)
            if occurrences == 0:
                raise ValueError("match_text was not found in current skill")
            if occurrences > 1:
                if not isinstance(occurrence, int) or not 1 <= occurrence <= occurrences:
                    raise ValueError(
                        "match_text occurs multiple times; provide a valid 1-based occurrence"
                    )
                positions = [
                    index for index in range(len(current))
                    if current.startswith(match_text, index)
                ]
                position = positions[occurrence - 1]
            else:
                position = current.index(match_text)
            if occurrences != 1 and position < 0:
                raise ValueError(
                    f"could not locate occurrence={occurrence} for match_text (found={occurrences})"
                )
            if op == "replace":
                current = current[:position] + new_text + current[position + len(match_text):]
            elif op == "insert_before":
                current = current[:position] + new_text + current[position:]
            elif op == "insert_after":
                end = position + len(match_text)
                current = current[:end] + new_text + current[end:]
            else:
                current = current[:position] + current[position + len(match_text):]
            record["applied"] = True
            applied_operation_ids.add(operation_id)
        except Exception as exc:
            record["error"] = repr(exc)
        applied.append(record)
    return current, applied


def apply_working_skill_operations(
    skill: str, state: dict[str, Any], updates: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Apply optimizer-approved edits only inside compiler-owned MCP anchors."""
    from skill_mining.skill_writer import (
        _delete_routing_transition_rule,
        _ensure_action_rules_region,
        _upsert_action_rule,
        _upsert_routing_transition_rule,
        _upsert_transition_rule,
    )

    applied = []
    current = skill
    for update in updates:
        resource = str(update.get("resource", ""))
        operation = str(update.get("op", "upsert")).lower()
        content = str(update.get("content", "")).strip()
        try:
            if resource == "action_rule":
                action = str(update.get("action", ""))
                current = _ensure_action_rules_region(current)
                if operation == "upsert":
                    current = _upsert_action_rule(current, action, f"#### `{action}`\n{content}")
                elif operation == "delete":
                    heading = re.escape(f"#### `{action}`")
                    pattern = rf"(?ms)^{heading}\s*$.*?(?=^####\s+`|^<!-- ACTION_RULES_END -->\s*$)"
                    current, count = re.subn(pattern, "", current, count=1)
                    if count != 1:
                        raise ValueError("action rule heading was not found inside skill")
                else:
                    continue
            elif resource == "transition_guard":
                edge = state["edges"][str(update.get("edge_id", ""))]
                source, target = str(edge["source"]), str(edge["target"])
                source_label, target_label = str(edge["source_action"]), str(edge["target_action"])
                if operation == "upsert":
                    if "<!-- ROUTING_SECTION_START -->" in current:
                        current = _upsert_routing_transition_rule(
                            current, source, target, source_label, target_label, content,
                        )
                    else:
                        block = f"##### `{source_label}` -> `{target_label}`\n- Transition when: {content}"
                        current = _upsert_transition_rule(current, source, target, block)
                elif operation == "delete":
                    if "<!-- ROUTING_SECTION_START -->" in current:
                        current = _delete_routing_transition_rule(current, source, target)
                    else:
                        key = re.escape(f"<!-- EDGE_RULE:{source}=>{target} -->")
                        pattern = rf"(?ms)^{key}\s*$.*?(?=^<!-- EDGE_RULE:|^<!-- TRANSITION_RULES_END -->\s*$)"
                        current, count = re.subn(pattern, "", current, count=1)
                        if count != 1:
                            raise ValueError("transition rule key was not found inside skill")
                else:
                    continue
            else:
                continue
            applied.append({"resource": resource, "op": operation, "action": update.get("action"), "edge_id": update.get("edge_id")})
        except Exception as exc:
            applied.append({"resource": resource, "op": operation, "error": repr(exc), "action": update.get("action"), "edge_id": update.get("edge_id")})
    return current, applied


def _materialize_promoted_graph_structure(skill: str, state: dict[str, Any]) -> str:
    """Synchronize promoted graph edges with the executable skill structure.

    A promoted transition is a structural update, not merely extra routing
    prose.  Keep the state graph authoritative, then refresh the compact tree
    and edge table and add an action rule for any newly exposed target node.
    The operation is deliberately deterministic; the LLM only decides which
    evidence-backed transition should be promoted and supplies its prose.
    """
    from skill_mining.skill_writer import (
        _action_rule_labels,
        _ensure_action_rules_region,
        _render_backbone_edge_table,
        _render_backbone_tree,
        _upsert_action_rule,
    )

    nodes = {str(item.get("id")): item for item in state.get("nodes", [])}
    promoted = [
        edge for edge in state.get("edges", {}).values()
        if edge.get("visibility") == "skill"
        and edge.get("kind") in {"promoted_branch", "backbone"}
    ]
    if not promoted:
        return skill

    # The online skill is a DAG-compatible backbone: preserve every original
    # backbone edge and add accepted promoted edges.  Do not silently add an
    # unknown edge or node that was not present in the offline full graph.
    backbone_edges = []
    seen_edges: set[tuple[str, str]] = set()
    for edge in state.get("edges", {}).values():
        if edge.get("kind") not in {"backbone", "promoted_branch"} or edge.get("visibility") != "skill":
            continue
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        if source not in nodes or target not in nodes or (source, target) in seen_edges:
            continue
        seen_edges.add((source, target))
        backbone_edges.append({
            "source": source,
            "target": target,
            "support": edge.get("offline_support", 0),
            "score": edge.get("score", edge.get("offline_support", 0)),
        })

    order = list(state.get("backbone_order", []))
    for edge in promoted:
        for node_id in (str(edge.get("source", "")), str(edge.get("target", ""))):
            if node_id in nodes and node_id not in order:
                order.append(node_id)
    state["backbone_order"] = order
    for item in state.get("nodes", []):
        item["topological_order"] = order.index(item["id"]) if item.get("id") in order else None

    subgraph = {
        "nodes": list(nodes.values()),
        "backbone": {"root": state.get("root", ROOT), "compilation_order": order, "edges": backbone_edges},
    }
    current = skill

    # Keep the structural sections synchronized when the organized compiler
    # format is present. Legacy skills simply retain their existing layout.
    tree = _render_backbone_tree(subgraph)
    edge_table = _render_backbone_edge_table(subgraph)
    current, tree_count = re.subn(
        r"(?ms)(^### Backbone Tree\s*\n).*?(?=^### Backbone Edges\s*$)",
        lambda match: match.group(1) + tree + "\n\n",
        current,
        count=1,
    )
    current, edge_count = re.subn(
        r"(?ms)(^### Backbone Edges\s*\n).*?(?=^### Routing Policies\s*$)",
        lambda match: match.group(1) + edge_table + "\n\n",
        current,
        count=1,
    )

    existing_actions = _action_rule_labels(current)
    for edge in promoted:
        target = str(edge.get("target", ""))
        target_label = str(edge.get("target_action", nodes.get(target, {}).get("label", target)))
        if target_label in existing_actions:
            continue
        action_record = state.get("action_rules", {}).get(target_label, {})
        content = str(action_record.get("rule", "")).strip()
        if not content:
            node = nodes.get(target, {})
            promotions = node.get("online_promotions", []) if isinstance(node, dict) else []
            content = next(
                (str(item.get("content", "")).strip() for item in promotions if str(item.get("content", "")).strip()),
                "",
            )
        if not content:
            content = (
                f"- Role: Execute `{target_label}` when the promoted transition reaches this action.\n"
                "- Use only values explicitly grounded in the current dialogue; "
                "retrieve deferred transition details from the reference when needed."
            )
        current = _ensure_action_rules_region(current)
        current = _upsert_action_rule(current, target_label, f"#### `{target_label}`\n{content}")
        existing_actions.add(target_label)

    return current


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
        if edge.get("visibility") == "skill" and guard:
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
            # The edge title alone is not enough for lexical retrieval: the
            # runtime query is usually phrased in dialogue language. Retain a
            # small bounded set of rollout contexts as evidence, never as
            # executable policy.
            evidence = edge.get("evidence", []) or []
            for item in evidence[:3]:
                snippet = str(item.get("context", "")).strip().replace("\n", " ")
                if snippet:
                    reference_lines.append(f"- Dialogue evidence: {snippet[:900]}")
            if evidence:
                reference_lines.append("")
    for note in state.get("reference_notes", []):
        if str(note.get("content", "")).strip():
            edge = state.get("edges", {}).get(str(note.get("edge_id", "")), {})
            title = (
                f"{edge.get('source_action')} -> {edge.get('target_action')}"
                if edge else "Online-maintained note"
            )
            reference_lines.extend([f"## {title}", f"- {note['content']}", ""])
    if len(skill_lines) == 2:
        skill_lines.append("- No non-backbone transition has met the online promotion criteria yet.")
    return "\n".join(skill_lines).rstrip() + "\n", "\n".join(reference_lines).rstrip() + "\n"


def _replace_generated_markdown_section(skill: str, heading: str, body: str) -> str:
    """Replace one generated level-2 section without touching user-authored text."""
    section = f"{heading}\n\n{body.strip()}\n"
    pattern = rf"(?ms)^{re.escape(heading)}\s*$.*?(?=^##\s+|\Z)"
    if re.search(pattern, skill):
        return re.sub(pattern, section + "\n", skill, count=1)
    return skill.rstrip() + "\n\n" + section


def render_online_skill_additions(state: dict[str, Any]) -> str:
    """Render executable online branches, including their target node role."""
    rows = []
    for edge_id, edge in sorted(state.get("edges", {}).items()):
        if edge.get("kind") == "backbone" or edge.get("visibility") != "skill":
            continue
        source = edge.get("source_action", edge.get("source"))
        target = edge.get("target_action", edge.get("target"))
        guard = str(edge.get("guard", "")).strip() or "when the current dialogue matches the supported evidence"
        rows.append(f"- From `{source}`, the executable workflow may branch to `{target}` when {guard}.")
        node = next((item for item in state.get("nodes", []) if item.get("label") == target), None)
        promotions = (node or {}).get("online_promotions", [])
        promotion = next((item for item in promotions if item.get("edge_id") == edge_id), None)
        if promotion and str(promotion.get("content", "")).strip():
            rows.append(f"  - Target action role and placement: {str(promotion['content']).strip()}")
        else:
            rows.append(f"  - Target action `{target}` is an executable branch attached after `{source}`; follow its existing action rule and slot discipline.")
    if not rows:
        return ""
    return "\n".join(rows)


def merge_online_skill_additions(skill: str, state: dict[str, Any]) -> str:
    """Make state-level branch promotions visible in the executable skill.

    Promoted edges are first materialized in the compiler-owned routing region
    when that region exists. The small prose section is deliberately retained
    as a readable node-placement index; it is not a second routing contract.
    """
    body = render_online_skill_additions(state)
    if not body:
        return skill
    current = _materialize_promoted_graph_structure(skill, state)
    try:
        from skill_mining.skill_writer import _upsert_routing_transition_rule
        for edge_id, edge in sorted(state.get("edges", {}).items()):
            if edge.get("kind") == "backbone" or edge.get("visibility") != "skill":
                continue
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if not source or not target:
                continue
            guard = str(edge.get("guard", "")).strip() or "the current dialogue matches the supported branch evidence"
            current = _upsert_routing_transition_rule(
                current, source, target,
                str(edge.get("source_action", source)),
                str(edge.get("target_action", target)), guard,
            )
    except (ImportError, ValueError):
        # Legacy/non-compiled skills may not have routing anchors. The
        # generated branch section below remains a usable fallback.
        pass
    return _replace_generated_markdown_section(current, "## Online-promoted graph branches", body)


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
