#!/usr/bin/env python3
r"""Per-Intent Skill.md + Reference.md 生成器。

对每个 subflow 的 vertex set：
  1. skill.md — LLM 总结的 skill 描述（intent, triggers, actions, strategy）
  2. reference.md — 每个 operator 对应的原始对话片段

对话片段提取利用 ABCD 的结构化 turn（targets 字段）直接定位，
无需额外 LLM 调用。Skill.md 需要 LLM 生成描述性内容。

用法：
  python skill_mining/skill_writer.py \
    --skills skill_mining/output/abcd_session_hg/per_subflow_vertex_subsets.json \
    --split train --max-sessions 200 \
    --output-dir outputs/skills
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_SKILL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SKILL_DIR.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SKILL_DIR) in sys.path:
    sys.path.remove(str(_SKILL_DIR))
sys.path.insert(0, str(_SKILL_DIR))

from eval_tod.abcd.data import load_abcd_data

_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "skills"

_SKILL_LLM_MAX_RETRIES = 3
_SKILL_LLM_RETRY_BASE_DELAY = 2.0


def _short_label(node_id: str) -> str:
    """Extract short display label from full operator name."""
    parts = node_id.split(":", 1)
    if len(parts) >= 2:
        return parts[1].split(":")[0] if ":" in parts[1] else parts[1]
    return node_id


# ═══════════════════════════════════════════════════════════════
# Dialogue snippet extraction (no LLM needed for ABCD)
# ═══════════════════════════════════════════════════════════════

def _find_operator_snippets(
    conversations: list[dict],
    subflow: str,
    operators: list[str],
    max_snippets_per_op: int = 3,
) -> Dict[str, list[dict]]:
    """Find dialogue snippets for each operator from ABCD conversations.

    Each operator looks like ``subflow:action_name`` or ``subflow:action_name:slots``.
    We match action turns where ``targets[1] == "take_action"`` and
    ``targets[2]`` matches the action name.

    Returns:
        {operator_name: [{convo_id, turn_index, snippet_text, action_name, slots}]}
    """
    # Parse operators to get action names
    op_action_names: dict[str, str] = {}  # operator → action_name
    for op in operators:
        parts = op.split(":", 1)
        if len(parts) >= 2:
            action_name = parts[1].split(":")[0] if ":" in parts[1] else parts[1]
        else:
            action_name = parts[0]
        op_action_names[op] = action_name

    # Collect matches
    op_snippets: dict[str, list[dict]] = defaultdict(list)

    for conv in conversations:
        conv_subflow = str(conv.get("scenario", {}).get("subflow", ""))
        if conv_subflow != subflow:
            continue

        convo_id = str(conv.get("convo_id", "?"))
        delexed = conv.get("delexed", [])

        for turn_idx, turn in enumerate(delexed):
            targets = turn.get("targets", [])
            if len(targets) < 3 or targets[1] != "take_action":
                continue
            action_name = str(targets[2])

            # Find which operator this matches
            for op, expected_action in op_action_names.items():
                if action_name != expected_action:
                    continue
                if len(op_snippets[op]) >= max_snippets_per_op:
                    continue

                # Extract surrounding context
                context_before = _get_context(delexed, turn_idx, before=2, speaker_filter=None)
                context_after = _get_context(delexed, turn_idx, before=0, after=2, speaker_filter=None)

                snippet_text = context_before + "\n" + _format_turn(turn) + "\n" + context_after
                snippet_text = snippet_text.strip()

                slot_values = targets[3] if len(targets) > 3 else []

                op_snippets[op].append({
                    "convo_id": convo_id,
                    "turn_index": turn_idx,
                    "snippet_text": snippet_text,
                    "action_name": action_name,
                    "slots": list(slot_values) if isinstance(slot_values, list) else [],
                })

    return dict(op_snippets)


def _get_context(
    delexed: list[dict],
    turn_idx: int,
    before: int = 2,
    after: int = 0,
    speaker_filter: str | None = None,
) -> str:
    """Get surrounding turn text."""
    lines = []
    for offset in range(-before, after + 1):
        if offset == 0:
            continue
        idx = turn_idx + offset
        if 0 <= idx < len(delexed):
            turn = delexed[idx]
            if speaker_filter and turn.get("speaker") != speaker_filter:
                continue
            lines.append(_format_turn(turn))
    return "\n".join(lines)


def _format_turn(turn: dict) -> str:
    """Format a turn as readable text."""
    speaker = turn.get("speaker", "unknown")
    text = turn.get("text", "").strip()
    label_map = {"agent": "Agent", "customer": "Customer", "action": "System"}
    label = label_map.get(speaker, speaker)
    return f"[{label}] {text}"


# ═══════════════════════════════════════════════════════════════
# Reference.md generation
# ═══════════════════════════════════════════════════════════════

def build_reference_md(
    subflow: str,
    op_snippets: Dict[str, list[dict]],
    max_snippets_per_op: int = 5,
    transition_cases: dict[str, list[dict[str, Any]]] | None = None,
    max_snippets_per_transition: int = 3,
) -> str:
    """Generate reference.md from operator→snippets mapping.

    Each operator gets up to ``max_snippets_per_op`` snippets, deduplicated
    by snippet text.  Sections have HTML anchors so skill.md can link to them.
    """
    transition_cases = transition_cases or {}
    max_snippets_per_transition = max(1, min(max_snippets_per_transition, 3))
    transition_by_source: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for key, cases in transition_cases.items():
        if " -> " in key:
            source, target = key.split(" -> ", 1)
            transition_by_source[source].append((target, cases or []))

    def display_action(operator: str) -> str:
        parts = operator.split(":", 1)
        return parts[1] if len(parts) == 2 else operator

    def unique(values: list[str], limit: int) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            value = str(value or "").strip()
            if not value or value[:160] in seen:
                continue
            seen.add(value[:160])
            result.append(value)
            if len(result) >= limit:
                break
        return result

    lines = [
        f"# Reference: {subflow}", "",
        f"Transition-centered dialogue evidence for `{subflow}`.",
        "Values in snippets are reference examples, not executable slot values.", "",
    ]
    source_ids = set(op_snippets) | set(transition_by_source)
    for source in sorted(source_ids):
        source_label = display_action(source)
        anchor = source_label.replace(":", "-").replace(" ", "-").lower()
        lines.extend([f'<a id="operator-{anchor}"></a>', f"## {source_label}", ""])
        transitions = sorted(transition_by_source.get(source, []), key=lambda item: item[0])
        source_snippets = [item.get("snippet_text", "") for item in op_snippets.get(source, [])]
        if not transitions:
            lines.extend(["### Terminal action evidence", ""])
            for snippet in unique(source_snippets, 1):
                lines.extend(["```text", snippet, "```", ""])
            continue
        for target, cases in transitions:
            target_label = display_action(target)
            lines.extend([f"### {source_label} -> {target_label}", ""])
            snippets = unique([case.get("context", "") for case in cases], max_snippets_per_transition)
            if not snippets:
                snippets = unique(source_snippets, 1)
            # Keep the transition contract explicit even when a malformed or
            # unusually sparse conversation has no recoverable text.  The
            # section must still contain one evidence item; the note is
            # intentionally non-executable and cannot be mistaken for a slot
            # value or a learned routing rule.
            if not snippets:
                snippets = ["No dialogue snippet was available for this observed transition."]
            for index, snippet in enumerate(snippets, 1):
                lines.extend([f"#### Example {index}", "", "```text", snippet, "```", ""])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Skill.md generation from subgraph (with pathways + branches)
# ═══════════════════════════════════════════════════════════════

def build_skill_md_from_subgraph(
    subflow: str,
    subgraph: dict,
    op_snippets: Dict[str, list[dict]],
    use_llm: bool = True,
) -> str:
    """Generate skill.md from subgraph (pathways, branches, edges).

    Unlike the flat vertex-list approach, this uses the subgraph structure
    to describe the action flow with branching conditions.
    """
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])
    pathways = subgraph.get("pathways", [])
    branches = subgraph.get("branch_points", [])
    coverage_pct = subgraph.get("coverage_pct", 0)
    num_sessions = subgraph.get("n_sessions", 0)

    if use_llm:
        prompt = _build_subgraph_skill_prompt(
            subflow, nodes, edges, pathways, branches, op_snippets,
            coverage_pct, num_sessions,
        )
        return _compile_skill_with_retry(prompt, subflow)

    return _build_skill_md_from_subgraph_fallback(
        subflow, nodes, edges, pathways, branches, op_snippets,
        coverage_pct, num_sessions,
    )


def build_skill_md_from_backbone(
    subflow: str,
    subgraph: dict,
    op_snippets: Dict[str, list[dict]],
    use_llm: bool = True,
    transition_induction: dict[str, Any] | None = None,
    seed_skill: str | None = None,
) -> str:
    """Compile one complete backbone skill from all retained graph evidence."""
    if use_llm:
        nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
        required_actions = [
            nodes[node_id]["label"]
            for node_id in subgraph.get("backbone", {}).get("compilation_order", [])
            if node_id in nodes
        ]
        return _compile_backbone_skill_with_transition_mcp(
            subflow,
            subgraph,
            op_snippets,
            transition_induction,
            required_actions,
            seed_skill=seed_skill,
        )
    return _build_skill_md_from_backbone_fallback(subflow, subgraph)


def materialize_progressive_disclosure(skill_md: str) -> tuple[str, str, str]:
    """Split compiled action/slot knowledge into bounded runtime resources.

    The backbone and routing policy remain in ``skill.md``.  Per-action rules
    and ordered slot policies are persisted separately so inference can fetch
    only the few sections relevant to the current dialogue.
    """
    def section(start: str, end: str) -> str:
        match = re.search(
            rf"(?ms)^{re.escape(start)}\s*$\n?(.*?)(?=^{re.escape(end)}\s*$|\Z)",
            skill_md,
        )
        return match.group(1).strip() if match else ""

    action_body = section("### Action Rules", "## Slot Discipline")
    slot_body = section("## Slot Policies", "## Reference")
    action_rules = "# Action Rules\n\n" + (action_body or "No action rules were compiled.") + "\n"
    slot_policies = "# Slot Policies\n\n" + (slot_body or "No slot policies were compiled.") + "\n"

    compact = re.sub(
        r"(?ms)^### Action Rules\s*$.*?(?=^## Slot Discipline\s*$)",
        "### Action Rules\n"
        "- Use `retrieve_action_rule` when the applicable action procedure is uncertain. "
        "The full per-action rules are stored in `action_rules.md`.\n\n",
        skill_md,
    )
    compact = re.sub(
        r"(?ms)^## Slot Policies\s*$.*?(?=^## Reference\s*$|\Z)",
        "## Slot Policies\n"
        "- Use `retrieve_slot_policy` after selecting an action when ordered value "
        "sources, reuse, or missing-value behavior are uncertain. The full policies "
        "are stored in `slot_policies.md`.\n\n",
        compact,
    )
    return compact, action_rules, slot_policies


def build_skill_md_from_unordered_backbone(
    subflow: str,
    subgraph: dict,
    op_snippets: Dict[str, list[dict]],
    transition_induction: dict[str, Any] | None = None,
    use_llm: bool = True,
) -> str:
    """Compile a flat induced-transition control for the organization ablation."""
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    required_actions = [node["label"] for node in nodes.values()]
    if use_llm:
        return _compile_skill_with_retry(
            _build_unordered_backbone_skill_prompt(
                subflow, subgraph, op_snippets, transition_induction,
            ),
            subflow,
            validator=lambda text: _validate_unordered_backbone_skill(text, required_actions),
        )
    return _build_skill_md_from_unordered_backbone_fallback(
        subflow, subgraph, transition_induction,
    )


def _stable_unordered(items: list[Any], subflow: str, key: Callable[[Any], str]) -> list[Any]:
    """Return a reproducible non-semantic ordering for the flat control."""
    return sorted(
        items,
        key=lambda item: hashlib.sha256(
            f"{subflow}::organization-ablation::{key(item)}".encode("utf-8")
        ).hexdigest(),
    )


def _build_unordered_backbone_skill_prompt(
    subflow: str,
    subgraph: dict,
    op_snippets: Dict[str, list[dict]],
    transition_induction: dict[str, Any] | None,
) -> str:
    """Render induced edge semantics without exposing a workflow organization."""
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    node_blocks: list[str] = []
    for node in _stable_unordered(list(nodes.values()), subflow, lambda item: item["id"]):
        contract = node.get("slot_contract", {})
        positions = "; ".join(
            f"arg{item['position']}: types={','.join(item.get('value_types') or ['value'])}; "
            f"observed_sources={','.join(item.get('source_types') or ['unresolved'])}; "
            f"required_rate={item.get('required_rate', 0):.0%}"
            for item in contract.get("positions", [])
        ) or "no observed slot values"
        snippets = op_snippets.get(node["id"], [])
        evidence = snippets[0]["snippet_text"][:280] if snippets else "(no local snippet)"
        node_blocks.append(
            f"Action: {node['label']}\n"
            f"Observed slot contract: count={node.get('observed_slot_counts', [0])}; {positions}\n"
            f"Reference-only dialogue evidence (do not copy slot values): {evidence}"
        )

    edge_cards: list[dict[str, str]] = []
    for source, rules in (transition_induction or {}).get("rules_by_source", {}).items():
        for rule in rules or []:
            target = str(rule.get("target", ""))
            if not target:
                continue
            edge_cards.append({
                "source": str(source),
                "target": target,
                "status": str(rule.get("status", "underspecified")),
                "condition": str(rule.get("condition") or ""),
            })
    edge_blocks: list[str] = []
    for card in _stable_unordered(
        edge_cards, subflow, lambda item: f"{item['source']}->{item['target']}",
    ):
        source = nodes.get(card["source"], {"label": _short_label(card["source"])})
        target = nodes.get(card["target"], {"label": _short_label(card["target"])})
        condition = card["condition"] or "insufficient observable evidence to state a unique trigger"
        edge_blocks.append(
            f"Induced transition card: {source['label']} -> {target['label']}\n"
            f"Induction status: {card['status']}\n"
            f"Induced transition condition: {condition}"
        )

    allowed_actions = [node["label"] for node in nodes.values()]
    return f"""You are compiling a deliberately flat customer-service skill for a controlled ablation.
The supplied evidence contains all mined action nodes and all pre-induced edge-level transition cards, but it is intentionally unordered.

## Strict Control Condition
- Allowed actions, and only allowed actions: {', '.join(allowed_actions)}
- Write exactly one `#### ` action card for every allowed action.
- Preserve the supplied induced transition cards; do not invent actions, edges, conditions, business policies, database outcomes, or semantic slot names.
- Do NOT infer, name, or present a main path, backbone, decision hierarchy, edge priority, branch/retry category, route grouping, or global state machine. Do not reorder the cards into a workflow.
- Keep one flat `## Unordered Interaction Cards` section. Each action card may list its induced transitions in the evidence order below. No card may claim that it is the normal or preferred next step.
- Slot values in node cards and dialogue snippets are REFERENCE-ONLY evidence.
  They describe observed usage patterns and must never be hard-coded into the
  executable skill or emitted as default action arguments. At runtime, use only
  values explicitly present in the current dialogue state. A few-shot example
  is allowed only in a clearly labelled illustrative section, must use masked
  placeholders such as `<VALUE_1>`, and must never become a literal rule.
- Slots must be real values available in the current dialogue. Never put field
  names, placeholders, or demonstration values into action calls. Mention only
  the observed source categories and missing-value behavior when evidence makes
  this clear.

## Skill ID
{subflow}

## Unordered Node Cards
{chr(10).join(node_blocks) if node_blocks else '(none)'}

## Unordered Induced Transition Cards
{chr(10).join(edge_blocks) if edge_blocks else '(none)'}

## Required Output
Return only Markdown in this form:

```markdown
# Skill: {subflow}

## Intent
[One concise description of the customer request.]

## Unordered Interaction Cards
#### `action-a`
- Slots: ...
- Induced transition: `action-a` -> `action-b` when ...

#### `every-other-action`
- Slots: ...
- Induced transitions: ...

## Reference
- Consult `reference.md` for dialogue wording when uncertain.
```
"""


def _build_skill_md_from_unordered_backbone_fallback(
    subflow: str,
    subgraph: dict,
    transition_induction: dict[str, Any] | None = None,
) -> str:
    """No-LLM rendering for inspection-only organization ablations."""
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    lines = [f"# Skill: {subflow}", "", "## Intent", "", f"Handle `{subflow}` requests using observed interactions.", "", "## Unordered Interaction Cards", ""]
    for node in _stable_unordered(list(nodes.values()), subflow, lambda item: item["id"]):
        lines.extend([f"#### `{node['label']}`", ""])
        contract = node.get("slot_contract", {})
        lines.append(f"- Slots: {contract.get('min_slots', 0)}-{contract.get('max_slots', 0)} ordered real value(s).")
        rules = (transition_induction or {}).get("rules_by_source", {}).get(node["id"], [])
        for rule in _stable_unordered(
            list(rules), subflow,
            lambda item: f"{node['id']}->{item.get('target', '')}",
        ):
            target_id = str(rule.get("target", ""))
            target = nodes.get(target_id, {"label": _short_label(target_id)})
            condition = str(rule.get("condition") or "insufficient observable evidence")
            lines.append(f"- Induced transition: `{node['label']}` -> `{target['label']}` when {condition}.")
        lines.append("")
    lines.extend(["## Reference", "", "- Consult `reference.md` for dialogue wording when uncertain."])
    return "\n".join(lines)


def _validate_unordered_backbone_skill(skill: str, required_actions: list[str]) -> None:
    """Reject controls that reconstruct the organized compiler's structure."""
    _validate_backbone_action_coverage(skill, required_actions)
    if "## Unordered Interaction Cards" not in skill:
        raise ValueError("unordered control omitted its flat interaction-card section")
    forbidden_headings = re.compile(
        r"^#{1,6}\s+(?:Main Path|Workflow|State Machine|Decision Point|Recovery And Retry|Backbone)\b",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if forbidden_headings.search(skill):
        raise ValueError("unordered control reconstructed a structured routing section")
    if re.search(r"\bpriority\s*\d+\b", skill, flags=re.IGNORECASE):
        raise ValueError("unordered control introduced edge priorities")


_ACTION_RULES_START = "<!-- ACTION_RULES_START -->"
_ACTION_RULES_END = "<!-- ACTION_RULES_END -->"
_TRANSITION_RULES_START = "<!-- TRANSITION_RULES_START -->"
_TRANSITION_RULES_END = "<!-- TRANSITION_RULES_END -->"
_ROUTING_SECTION_START = "<!-- ROUTING_SECTION_START -->"
_ROUTING_SECTION_END = "<!-- ROUTING_SECTION_END -->"


def _route_source_marker(source_id: str) -> str:
    """Invisible coverage marker for one routing decision point."""
    return f"<!-- ROUTE_SOURCE:{source_id} -->"


def _route_edge_marker(edge_id: str) -> str:
    """Legacy per-edge marker used by the older branch-patch compiler."""
    return f"<!-- ROUTE_EDGE:{edge_id} -->"


def _validate_structured_seed_skill(skill: str) -> None:
    if skill.count(_ROUTING_SECTION_START) != 1 or skill.count(_ROUTING_SECTION_END) != 1:
        raise ValueError("structured seed skill must preserve the ROUTING_SECTION anchors exactly once")
    if skill.index(_ROUTING_SECTION_START) >= skill.index(_ROUTING_SECTION_END):
        raise ValueError("ROUTING_SECTION anchors are out of order")
    if "### Primary Route" not in skill:
        raise ValueError("structured seed skill omitted the primary route")


def _replace_routing_section(skill: str, content: str) -> str:
    if skill.count(_ROUTING_SECTION_START) != 1 or skill.count(_ROUTING_SECTION_END) != 1:
        raise ValueError("skill does not contain a unique ROUTING_SECTION region")
    start = skill.index(_ROUTING_SECTION_START) + len(_ROUTING_SECTION_START)
    end = skill.index(_ROUTING_SECTION_END)
    if start > end:
        raise ValueError("ROUTING_SECTION markers are out of order")
    return skill[:start] + "\n\n" + content.strip() + "\n\n" + skill[end:]


def _transition_induction_source_ids(
    transition_induction: dict[str, Any] | None,
) -> set[str]:
    rules_by_source = (transition_induction or {}).get("rules_by_source", {})
    return {
        str(source)
        for source, rules in rules_by_source.items()
        if any(isinstance(rule, dict) and rule.get("target") for rule in rules or [])
    }


def _build_transition_write_prompt(
    subflow: str,
    current_skill: str,
    transition_induction: dict[str, Any] | None,
    nodes: dict[str, dict[str, Any]],
) -> str:
    """Ask the LLM to write induced transitions as natural routing prose."""
    rows: list[str] = []
    rules_by_source = (transition_induction or {}).get("rules_by_source", {})
    for source, rules in rules_by_source.items():
        source_label = nodes.get(source, {}).get("label", _short_label(source))
        source_rows: list[str] = []
        for rule in rules or []:
            if not isinstance(rule, dict) or not rule.get("target"):
                continue
            target = str(rule["target"])
            target_label = nodes.get(target, {}).get("label", _short_label(target))
            status = str(rule.get("status") or "underspecified")
            condition = str(rule.get("condition") or "")
            evidence = str(rule.get("evidence") or "")
            source_rows.append(
                f"transition={source_label} -> {target_label}; "
                f"induction_mode={status}; induced_interpretation={condition or '(uncertain)'}; "
                f"induction_note={evidence or '(none)'}"
            )
        if source_rows:
            rows.append(
                f"{_route_source_marker(str(source))} Decision point: {source_label}\n"
                + "\n".join(source_rows)
            )
    return f"""You are making one constrained routing edit to an existing customer-service skill.

Write the supplied transition induction into the skill's Routing Policies
section as concise, natural-language workflow prose. This is an editorial
integration pass, not a re-induction or a verification pass.

<current_skill>
{current_skill}
</current_skill>

<induced_transition_evidence>
{chr(10).join(rows) if rows else '(no transition evidence)'}
</induced_transition_evidence>

Writing requirements:
- Preserve the complete Backbone Tree and all action rules conceptually.
- Organize related transitions around meaningful backbone decision points.
- Explain normal continuation, alternative routes, retries, loops, and
  rejoining behavior in connected prose. Do not produce one repetitive
  `source -> target: condition` line per edge.
- Use the induced interpretation as evidence, but phrase it naturally and
  conservatively. If the mode is `underspecified` or the interpretation is uncertain,
  say that the route is an observed alternative whose exact trigger is not
  fully identifiable; do not invent a precise condition.
- A self-transition means repeated action / retry behavior. Describe it as a
  retry or continued attempt when supported, not as a new action.
- Do not claim mutual exclusion unless the evidence explicitly supports it.
- Keep reference slot values out of the skill. Describe slot availability or
  reuse patterns, and use only masked placeholders in any illustrative few-shot.
- Preserve every invisible source-decision marker exactly once somewhere in
  the prose. One decision-point paragraph may jointly explain all outgoing
  transitions from that source. Markers are compiler metadata and must not be
  explained to the end user.

<filesystem_mcp>
Use exactly one constrained operation:
```json
{{
  "operations": [
    {{
      "op": "replace_routing_section",
      "content": "natural Markdown prose containing every required invisible source-decision marker"
    }}
  ]
}}
```
It replaces only content between `ROUTING_SECTION_START` and
`ROUTING_SECTION_END`. It cannot change the intent, Backbone Tree, action
rules, slot policies, or reference-use sections.
</filesystem_mcp>

Return only one valid JSON object, without Markdown fences.
"""


def _parse_transition_write(raw: str, expected_source_ids: set[str]) -> str:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I | re.S).strip()
    parsed = json.loads(text)
    operations = parsed.get("operations") if isinstance(parsed, dict) else None
    if not isinstance(operations, list) or len(operations) != 1:
        raise ValueError("transition write must return exactly one filesystem-MCP operation")
    operation = operations[0]
    if not isinstance(operation, dict) or operation.get("op") != "replace_routing_section":
        raise ValueError("transition write returned an unsupported filesystem-MCP operation")
    content = str(operation.get("content") or "").strip()
    if not content:
        raise ValueError("transition write returned empty routing prose")
    found = set(re.findall(r"<!-- ROUTE_SOURCE:(.*?) -->", content))
    if found != expected_source_ids:
        raise ValueError(
            f"transition write decision coverage mismatch; missing={sorted(expected_source_ids - found)}, "
            f"unexpected={sorted(found - expected_source_ids)}"
        )
    return content


def build_backbone_seed_skill(
    subflow: str,
    subgraph: dict[str, Any],
    op_snippets: Dict[str, list[dict]],
    required_actions: list[str] | None = None,
) -> str:
    """Compile the reusable backbone seed before transition induction."""
    if required_actions is None:
        nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
        required_actions = [
            nodes[node_id]["label"]
            for node_id in subgraph.get("backbone", {}).get("compilation_order", [])
            if node_id in nodes
        ]
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    root = str(subgraph.get("backbone", {}).get("root", "ROOT"))
    required_backbone_edges = [
        (
            "ROOT" if str(edge.get("source")) == root else nodes.get(
                str(edge.get("source")), {}
            ).get("label", str(edge.get("source"))),
            nodes.get(str(edge.get("target")), {}).get("label", str(edge.get("target"))),
        )
        for edge in subgraph.get("backbone", {}).get("edges", [])
        if edge.get("target")
    ]
    return _compile_skill_with_retry(
        _build_backbone_skill_prompt(subflow, subgraph, op_snippets, None),
        subflow,
        validator=lambda text: _validate_backbone_skill(
            text, required_actions, required_backbone_edges,
        ),
    )


def _compile_backbone_skill_with_transition_mcp(
    subflow: str,
    subgraph: dict[str, Any],
    op_snippets: Dict[str, list[dict]],
    transition_induction: dict[str, Any] | None,
    required_actions: list[str],
    seed_skill: str | None = None,
) -> str:
    """Compile a seed, then write transition prose through the constrained MCP."""
    seed = seed_skill or build_backbone_seed_skill(
        subflow, subgraph, op_snippets, required_actions,
    )
    expected_source_ids = _transition_induction_source_ids(transition_induction)
    if not expected_source_ids:
        return seed

    from llm import chat
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    prompt = _build_transition_write_prompt(subflow, seed, transition_induction, nodes)
    last_error: Exception | None = None
    for attempt in range(1, _SKILL_LLM_MAX_RETRIES + 1):
        try:
            content = _parse_transition_write(chat(prompt, temperature=0.0).strip(), expected_source_ids)
            return _replace_routing_section(seed, content)
        except Exception as exc:
            last_error = exc
            if attempt < _SKILL_LLM_MAX_RETRIES:
                delay = _SKILL_LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"  LLM transition write failed for {subflow} "
                    f"(attempt {attempt}/{_SKILL_LLM_MAX_RETRIES}): {exc}; "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
    raise RuntimeError(
        f"LLM transition write failed for {subflow} after "
        f"{_SKILL_LLM_MAX_RETRIES} attempts: {last_error}"
    )


def _build_structured_seed_prompt(subflow: str, subgraph: dict[str, Any]) -> str:
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    main_path = subgraph.get("backbone", {}).get("main_path", [])
    main_labels = [nodes[node_id]["label"] for node_id in main_path if node_id in nodes]
    return f"""Write the stable backbone seed of an evidence-grounded customer-service skill.

This is the first stage of a two-stage compiler. Write the user-facing task
goal, a concise runtime-state note, and only the primary route. Do not enumerate
action-level preconditions, post-states, raw edges, branches, or retries here.
A later routing synthesis stage adds those as organized decision policies.

<backbone>
Skill ID: {subflow}
Primary route: {' -> '.join(main_labels) or '(single-action skill)'}
</backbone>

Return only Markdown and preserve the two HTML anchors exactly:

```markdown
# Skill: {subflow}

## Intent
[One concise sentence describing the customer request this skill handles.]

## Runtime State
- Track only state needed to choose routes, such as completed actions, selected entities, supplied credential types, and explicit failure signals.

## Workflow
### Primary Route
1. `action-a` -> `action-b` -> ...

### Routing Policies
{_ROUTING_SECTION_START}
{_ROUTING_SECTION_END}

## Reference Use
- Consult `reference.md` only when a route condition or action-specific dialogue pattern is uncertain.
```
"""


def _render_route_plan_evidence(
    plan: dict[str, Any],
    transition_cases: dict[str, list[dict[str, Any]]] | None,
) -> str:
    parts: list[str] = []
    transition_cases = transition_cases or {}
    for cluster in plan.get("clusters", []):
        parts.append(
            f"## Decision anchor: {cluster['anchor_label']} ({cluster['anchor']})\n"
            f"Normal main-path continuation: {cluster.get('normal_next_label') or '(none)'}"
        )
        for route in cluster.get("routes", []):
            parts.append(
                "- edge_id=" + route["edge_id"]
                + f"; {route['source_label']} -> {route['target_label']}"
                + f"; route_type={route['route_type']}"
                + f"; likely_rejoin={route['likely_rejoin_label'] or '(none)'}"
                + f"; suggested_route={' -> '.join(route['suggested_route_labels']) or '(none)'}"
            )
            case_key = f"{route['source']} -> {route['target']}"
            cases = transition_cases.get(case_key, [])
            if cases:
                parts.append("  Raw training-session contexts for this transition:")
                for case in cases:
                    parts.append(
                        "  ```text\n"
                        + str(case.get("context") or "")[:900]
                        + "\n  ```"
                    )
    return "\n".join(parts) if parts else "(No retained non-main transitions.)"


def _build_routing_synthesis_prompt(
    subflow: str,
    current_skill: str,
    branch_route_plan: dict[str, Any],
    transition_cases: dict[str, list[dict[str, Any]]] | None,
) -> str:
    edge_ids = branch_route_plan.get("selected_edge_ids", [])
    edge_markers = "\n".join("- " + _route_edge_marker(edge_id) for edge_id in edge_ids) or "- (none)"
    evidence = _render_route_plan_evidence(branch_route_plan, transition_cases)
    return f"""Organize retained graph branches into a concise routing policy for a
customer-service skill. The primary route in the current skill is fixed.

Do not produce a flat list of source-target rules or a per-action specification.
Organize the evidence around its main-path decision anchors. State the normal
continuation once per anchor, group related alternatives into short routes, and
explicitly say where a recovery route rejoins an existing/main-path action.
Put retry or loop behavior under `### Recovery And Retry`. Preserve overlap and
only claim mutual exclusion when the raw examples clearly support it. Derive
transition triggers conservatively from the raw training-session contexts; if
they are ambiguous, describe the customer situation broadly rather than
inventing a precise business rule. Keep the document compact and explain
control-flow logic rather than every action's pre/post-condition.

<current_skill>
{current_skill}
</current_skill>

<deterministic_branch_route_plan>
{evidence}
</deterministic_branch_route_plan>

<coverage_contract>
Every selected graph edge must have exactly one invisible marker in the routing
content. Put a marker beside the decision/route it supports. These are compiler
validation metadata, not user-facing text.
Required markers:
{edge_markers}
</coverage_contract>

<filesystem_mcp>
Use exactly one constrained operation:
```json
{{
  "operations": [
    {{
      "op": "replace_routing_section",
      "content": "Markdown beginning with ### Decision Point: ... and/or ### Recovery And Retry ..."
    }}
  ]
}}
```
It replaces only content between ROUTING_SECTION anchors. It cannot change the
intent, runtime state, primary route, or reference-use sections.
</filesystem_mcp>

Return ONLY one valid JSON object, without Markdown fences.
"""


def _parse_routing_patch(raw: str, expected_edge_ids: set[str]) -> str:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I | re.S).strip()
    parsed = json.loads(text)
    operations = parsed.get("operations") if isinstance(parsed, dict) else None
    if not isinstance(operations, list) or len(operations) != 1:
        raise ValueError("routing synthesis must return exactly one operation")
    operation = operations[0]
    if not isinstance(operation, dict) or operation.get("op") != "replace_routing_section":
        raise ValueError("unsupported filesystem-MCP routing operation")
    content = str(operation.get("content") or "").strip()
    if not content:
        raise ValueError("routing synthesis returned empty content")
    found = set(re.findall(r"<!-- ROUTE_EDGE:(.*?) -->", content))
    if found != expected_edge_ids:
        missing = sorted(expected_edge_ids - found)
        unexpected = sorted(found - expected_edge_ids)
        raise ValueError(f"routing coverage mismatch; missing={missing}, unexpected={unexpected}")
    return content


def _compile_structured_backbone_skill(
    subflow: str,
    subgraph: dict[str, Any],
    branch_route_plan: dict[str, Any] | None,
    transition_cases: dict[str, list[dict[str, Any]]] | None,
) -> str:
    """Compile a concise backbone skill with one global routing synthesis pass."""
    if branch_route_plan is None:
        from skill_mining.branch_route_planning import build_branch_route_plan
        branch_route_plan = build_branch_route_plan(subgraph)
    seed = _compile_skill_with_retry(
        _build_structured_seed_prompt(subflow, subgraph),
        subflow,
        validator=_validate_structured_seed_skill,
    )
    expected_edge_ids = set(branch_route_plan.get("selected_edge_ids", []))
    if not expected_edge_ids:
        return seed

    from llm import chat
    print(
        f"  Structured routing synthesis for {subflow}: "
        f"{len(branch_route_plan.get('clusters', []))} decision anchors, "
        f"{len(expected_edge_ids)} retained non-main edges, 1 LLM call"
    )
    prompt = _build_routing_synthesis_prompt(
        subflow, seed, branch_route_plan, transition_cases,
    )
    last_error: Exception | None = None
    for attempt in range(1, _SKILL_LLM_MAX_RETRIES + 1):
        try:
            content = _parse_routing_patch(chat(prompt, temperature=0.0).strip(), expected_edge_ids)
            return _replace_routing_section(seed, content)
        except Exception as exc:
            last_error = exc
            if attempt < _SKILL_LLM_MAX_RETRIES:
                delay = _SKILL_LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"  LLM routing synthesis failed for {subflow} "
                    f"(attempt {attempt}/{_SKILL_LLM_MAX_RETRIES}): {exc}; retrying in {delay:.1f}s"
                )
                time.sleep(delay)
    raise RuntimeError(
        f"LLM routing synthesis failed for {subflow} after "
        f"{_SKILL_LLM_MAX_RETRIES} attempts: {last_error}"
    )


def _action_heading(label: str) -> str:
    return f"#### `{label}`"


def _action_rule_labels(skill: str) -> set[str]:
    return set(re.findall(r"^####\s+`([^`]+)`\s*$", skill, flags=re.MULTILINE))


def _validate_backbone_action_coverage(skill: str, required_actions: list[str]) -> None:
    headings = set(re.findall(r"^#{3,6}\s+`([^`]+)`\s*$", skill, flags=re.MULTILINE))
    missing = [action for action in required_actions if action not in headings]
    if missing:
        raise ValueError(
            "compiled skill omitted retained action rules: " + ", ".join(missing)
        )


def _validate_backbone_skill(
    skill: str,
    required_actions: list[str],
    required_backbone_edges: list[tuple[str, str]] | None = None,
) -> None:
    """Validate action coverage and preservation of the complete tree section."""
    _validate_backbone_action_coverage(skill, required_actions)
    if "### Backbone Tree" not in skill:
        raise ValueError("compiled skill omitted the complete Backbone Tree")
    if "### Action Rules" not in skill:
        raise ValueError("compiled skill omitted the Action Rules section")
    if "## Slot Policies" not in skill:
        raise ValueError("compiled skill omitted the Slot Policies section")
    for source, target in required_backbone_edges or []:
        edge_text = f"`{source}` -> `{target}`"
        if edge_text not in skill:
            raise ValueError(f"compiled skill omitted backbone relation: {edge_text}")
    if skill.count(_ROUTING_SECTION_START) != 1 or skill.count(_ROUTING_SECTION_END) != 1:
        raise ValueError("compiled skill omitted the unique routing MCP region")


def _validate_main_path_skill(skill: str, main_labels: list[str]) -> None:
    if skill.count(_ACTION_RULES_START) != 1 or skill.count(_ACTION_RULES_END) != 1:
        raise ValueError("main-path skill must preserve the ACTION_RULES anchors exactly once")
    if skill.index(_ACTION_RULES_START) >= skill.index(_ACTION_RULES_END):
        raise ValueError("ACTION_RULES anchors are out of order")
    if skill.count(_TRANSITION_RULES_START) != 1 or skill.count(_TRANSITION_RULES_END) != 1:
        raise ValueError("main-path skill must preserve the TRANSITION_RULES anchors exactly once")
    if skill.index(_TRANSITION_RULES_START) >= skill.index(_TRANSITION_RULES_END):
        raise ValueError("TRANSITION_RULES anchors are out of order")
    missing = [label for label in main_labels if label not in _action_rule_labels(skill)]
    if missing:
        raise ValueError(f"main-path skill omitted action rules: {', '.join(missing)}")


def _upsert_action_rule(skill: str, action: str, content: str) -> str:
    """Apply the only filesystem-MCP mutation accepted during refinement."""
    if not content.strip().startswith(_action_heading(action)):
        raise ValueError(f"upsert content must start with {_action_heading(action)!r}")
    if skill.count(_ACTION_RULES_START) != 1 or skill.count(_ACTION_RULES_END) != 1:
        raise ValueError("skill does not contain a unique ACTION_RULES region")

    start_marker = skill.index(_ACTION_RULES_START)
    end_marker = skill.index(_ACTION_RULES_END)
    if start_marker >= end_marker:
        raise ValueError("ACTION_RULES markers are out of order")

    block = content.strip() + "\n\n"
    heading = _action_heading(action)
    match = re.search(
        rf"^{re.escape(heading)}\s*$.*?(?=^####\s+`|{re.escape(_ACTION_RULES_END)})",
        skill,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match:
        return skill[:match.start()] + block + skill[match.end():]

    insertion = skill.index(_ACTION_RULES_END)
    prefix = "" if skill[insertion - 1:insertion] == "\n" else "\n"
    return skill[:insertion] + prefix + block + skill[insertion:]


def _transition_rule_key(source: str, target: str) -> str:
    return f"<!-- EDGE_RULE:{source}=>{target} -->"


def _upsert_transition_rule(
    skill: str,
    source: str,
    target: str,
    content: str,
) -> str:
    """Apply one edge-scoped update without touching node/action rules."""
    if not content.strip().startswith("##### `"):
        raise ValueError("transition content must begin with a level-5 Markdown heading")
    if skill.count(_TRANSITION_RULES_START) != 1 or skill.count(_TRANSITION_RULES_END) != 1:
        raise ValueError("skill does not contain a unique TRANSITION_RULES region")
    start_marker = skill.index(_TRANSITION_RULES_START)
    end_marker = skill.index(_TRANSITION_RULES_END)
    if start_marker >= end_marker:
        raise ValueError("TRANSITION_RULES markers are out of order")

    key = _transition_rule_key(source, target)
    block = key + "\n" + content.strip() + "\n\n"
    position = skill.find(key, start_marker, end_marker)
    if position >= 0:
        following = re.search(
            rf"^<!-- EDGE_RULE:.*? -->\s*$|^{re.escape(_TRANSITION_RULES_END)}\s*$",
            skill[position + len(key):],
            flags=re.MULTILINE,
        )
        if not following:
            raise ValueError("could not locate the end of the existing transition rule")
        end = position + len(key) + following.start()
        return skill[:position] + block + skill[end:]

    insertion = skill.index(_TRANSITION_RULES_END)
    prefix = "" if skill[insertion - 1:insertion] == "\n" else "\n"
    return skill[:insertion] + prefix + block + skill[insertion:]


def _branch_action_sources(subgraph: dict[str, Any]) -> list[str]:
    """Return off-main-path nodes that need their own node/action rule."""
    backbone = subgraph.get("backbone", {})
    main_path = backbone.get("main_path", [])
    order = backbone.get("compilation_order", [])
    main_nodes = set(main_path)
    return [node_id for node_id in order if node_id not in main_nodes]


def _branch_transition_edges(subgraph: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every retained non-main edge, including retries and returns."""
    main_path = subgraph.get("backbone", {}).get("main_path", [])
    main_pairs = set(zip(main_path, main_path[1:]))
    local = subgraph.get("local_transitions", {})
    edges = [
        edge
        for source, transitions in local.items()
        for edge in transitions
        if (source, edge.get("target")) not in main_pairs
    ]
    return sorted(edges, key=lambda edge: (edge["source"], edge.get("priority", 1), edge["target"]))


def _render_action_evidence(
    node_id: str,
    nodes: dict[str, dict[str, Any]],
    op_snippets: Dict[str, list[dict]],
) -> str:
    node = nodes[node_id]
    lines = [
        f"Source action: {node['label']} ({node_id})",
        "Observed ordered slot contract: " + json.dumps(node.get("slot_contract", {}), ensure_ascii=False),
    ]
    snippets = op_snippets.get(node_id, [])
    if snippets:
        lines.append("Representative dialogue evidence:")
        lines.append("```text\n" + snippets[0]["snippet_text"][:420] + "\n```")
    return "\n".join(lines)


def _build_main_path_skill_prompt(
    subflow: str,
    subgraph: dict[str, Any],
    op_snippets: Dict[str, list[dict]],
) -> tuple[str, list[str]]:
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    main_path = subgraph.get("backbone", {}).get("main_path", [])
    main_labels = [nodes[node_id]["label"] for node_id in main_path if node_id in nodes]
    evidence: list[str] = []
    for node_id in main_path:
        snippets = op_snippets.get(node_id, [])
        if snippets:
            evidence.append(
                f"**{nodes[node_id]['label']}**\n```text\n{snippets[0]['snippet_text'][:280]}\n```"
            )
    prompt = f"""You are writing the seed version of an evidence-grounded customer-service skill.

This is round 1 of a controlled multi-round compiler. Write ONLY the primary
execution spine below. Do not invent branch, retry, or alternative transitions:
those are added later by restricted patch calls. Preserve every HTML anchor
verbatim because a local filesystem-MCP adapter uses them to update the skill.

<skill_scope>
Skill ID: {subflow}
Sessions: {subgraph.get('n_sessions', 0)}
Primary execution spine: {' -> '.join(main_labels) or '(single-action skill)'}
</skill_scope>

<main_path_evidence>
{chr(10).join(evidence) if evidence else '(No representative snippets supplied.)'}
</main_path_evidence>

<requirements>
- Include every main-path action exactly once under Action Rules.
- For each rule, give concise preconditions, ordered real-value slot discipline,
  and post-state. Do not put transition rules in an action block.
- Include every adjacent main-path edge exactly once under Transition Rules.
  Each edge must name its source and target action and state its priority-1
  transition condition conservatively.
- Do not include actions outside the primary execution spine yet.
- Do not use concrete example values as reusable slot values.
</requirements>

Return only Markdown in this exact skeleton. Keep all four anchors unchanged.

```markdown
# Skill: {subflow}

## Intent
[One concise sentence.]

## Workflow
### Main Path
1. `action-a` -> `action-b` -> ...

## State Machine
- Track `last_completed_action`, `account_selected`, `credential_types`, `credential_count`, and `failure_signal`.

### Action Rules
{_ACTION_RULES_START}
#### `action-a`
- Preconditions: ...
- Slots: ordered real values only.
- Post-state: `last_completed_action=action-a`.
{_ACTION_RULES_END}

### Transition Rules
{_TRANSITION_RULES_START}
##### `action-a` -> `action-b`
- Priority 1: when [observed main-path condition], transition from `action-a` to `action-b`.
{_TRANSITION_RULES_END}

## Slot Discipline
- Use only real values available in the current dialogue state.
- Do not emit schema labels or placeholders as slot values.

## Reference
- Consult `reference.md` for action-specific dialogue evidence.
```
"""
    return prompt, main_labels


def _build_branch_patch_prompt(
    subflow: str,
    current_skill: str,
    node_id: str,
    evidence: str,
) -> str:
    return f"""You are adding or refining one action rule in a customer-service skill.

The current skill is an accepted artifact. Preserve its intent, main path,
state variables, slot discipline, and every action rule other than the focused
source action. This round describes only the action's preconditions, ordered
slot discipline, and post-state. Do NOT add, remove, or describe transitions:
each selected graph edge is updated separately in the Transition Rules section.

<current_skill>
{current_skill}
</current_skill>

<branch_evidence>
{evidence}
</branch_evidence>

<filesystem_mcp>
The runtime provides a constrained local editing tool. It is the only way to
modify the skill in this round:

Tool: `upsert_action_rule`
Input JSON:
{{
  "operations": [
    {{
      "op": "upsert_action_rule",
      "action": "{_short_label(node_id)}",
      "content": "the complete Markdown block beginning with #### `ACTION`"
    }}
  ]
}}

The tool replaces only the action-rule block whose heading is
`#### `ACTION`` inside `ACTION_RULES_START` / `ACTION_RULES_END`, or inserts
that block there if it is missing. It cannot edit any other part of the
document. Submit exactly one operation for the focused source action
`{_short_label(node_id)}`.
</filesystem_mcp>

Return ONLY one valid JSON tool call. Do not wrap it in Markdown fences.
"""


def _parse_branch_patch(raw: str, expected_action: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I | re.S).strip()
    parsed = json.loads(text)
    operations = parsed.get("operations") if isinstance(parsed, dict) else None
    if not isinstance(operations, list) or len(operations) != 1:
        raise ValueError("branch refinement must return exactly one operation")
    operation = operations[0]
    if not isinstance(operation, dict) or operation.get("op") != "upsert_action_rule":
        raise ValueError("unsupported filesystem-MCP operation")
    action = str(operation.get("action") or "").strip()
    content = str(operation.get("content") or "").strip()
    if action != expected_action:
        raise ValueError(f"branch refinement targeted {action!r}, expected {expected_action!r}")
    if not content:
        raise ValueError("branch refinement returned empty action content")
    return {"action": action, "content": content}


def _apply_branch_patch_with_retry(
    subflow: str,
    current_skill: str,
    node_id: str,
    evidence: str,
) -> str:
    from llm import chat

    expected_action = _short_label(node_id)
    prompt = _build_branch_patch_prompt(subflow, current_skill, node_id, evidence)
    last_error: Exception | None = None
    for attempt in range(1, _SKILL_LLM_MAX_RETRIES + 1):
        try:
            patch = _parse_branch_patch(chat(prompt, temperature=0.0).strip(), expected_action)
            updated = _upsert_action_rule(current_skill, patch["action"], patch["content"])
            if expected_action not in _action_rule_labels(updated):
                raise ValueError("filesystem-MCP patch did not create the focused action rule")
            return updated
        except Exception as exc:
            last_error = exc
            if attempt < _SKILL_LLM_MAX_RETRIES:
                delay = _SKILL_LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"  LLM branch refinement failed for {subflow}/{expected_action} "
                    f"(attempt {attempt}/{_SKILL_LLM_MAX_RETRIES}): {exc}; retrying in {delay:.1f}s"
                )
                time.sleep(delay)
    raise RuntimeError(
        f"LLM branch refinement failed for {subflow}/{expected_action} after "
        f"{_SKILL_LLM_MAX_RETRIES} attempts: {last_error}"
    )


def _render_transition_evidence(
    edge: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    induced_rules: dict[str, list[dict[str, Any]]],
    op_snippets: Dict[str, list[dict]],
) -> str:
    source = edge["source"]
    target = edge["target"]
    source_label = nodes.get(source, {"label": _short_label(source)})["label"]
    target_label = nodes.get(target, {"label": _short_label(target)})["label"]
    lines = [
        f"Selected edge ID: {source} => {target}",
        f"Transition: {source_label} -> {target_label}",
        f"kind={edge.get('kind', 'branch')}; priority={edge.get('priority', 1)}; "
        f"support={edge.get('support', 0)}; P={edge.get('probability', 0)}",
        "Observed condition: " + _format_observed_condition(edge.get("condition", {})),
    ]
    for rule in induced_rules.get(source, []):
        if rule.get("target") == target:
            lines.append(
                "Joint transition induction (authoritative): "
                f"condition={rule['condition']}; priority={rule['priority']}; relation={rule['relation']}"
            )
            break
    snippets = op_snippets.get(source, [])
    if snippets:
        lines.append("Representative source-action evidence:")
        lines.append("```text\n" + snippets[0]["snippet_text"][:420] + "\n```")
    return "\n".join(lines)


def _build_transition_patch_prompt(
    subflow: str,
    current_skill: str,
    edge: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    evidence: str,
) -> str:
    source = edge["source"]
    target = edge["target"]
    source_label = nodes.get(source, {"label": _short_label(source)})["label"]
    target_label = nodes.get(target, {"label": _short_label(target)})["label"]
    return f"""You are recording one selected graph edge in an accepted customer-service skill.

The edge is a first-class workflow fact, not merely part of its source action.
Preserve all existing skill content. Add or replace only this exact directed
edge rule. The target may already be a main-path action or another existing
node: name that target explicitly instead of duplicating its action rule. Do
not reverse the edge, merge it with another edge, or invent a new target.

When this edge returns to a main-path action or overlaps another transition,
state a compatible guard plus the supplied priority/relation. Never claim
mutual exclusion unless the evidence says `exclusive`; use ordered priority or
fallback wording for overlapping/retry behavior.

<current_skill>
{current_skill}
</current_skill>

<selected_edge_evidence>
{evidence}
</selected_edge_evidence>

<filesystem_mcp>
The constrained local editing tool is the only allowed mutation in this round.

Tool: `upsert_transition_rule`
Input JSON:
{{
  "operations": [
    {{
      "op": "upsert_transition_rule",
      "source": "{source}",
      "target": "{target}",
      "content": "the complete Markdown block beginning with ##### `{source_label}` -> `{target_label}`"
    }}
  ]
}}

The tool updates only the uniquely keyed edge `{source} => {target}` inside
`TRANSITION_RULES_START` / `TRANSITION_RULES_END`. It cannot change action
rules, main-path text, or any other edge. Submit exactly one operation.
</filesystem_mcp>

Return ONLY one valid JSON tool call. Do not wrap it in Markdown fences.
"""


def _parse_transition_patch(raw: str, expected_source: str, expected_target: str) -> dict[str, str]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I | re.S).strip()
    parsed = json.loads(text)
    operations = parsed.get("operations") if isinstance(parsed, dict) else None
    if not isinstance(operations, list) or len(operations) != 1:
        raise ValueError("transition refinement must return exactly one operation")
    operation = operations[0]
    if not isinstance(operation, dict) or operation.get("op") != "upsert_transition_rule":
        raise ValueError("unsupported filesystem-MCP transition operation")
    source = str(operation.get("source") or "").strip()
    target = str(operation.get("target") or "").strip()
    content = str(operation.get("content") or "").strip()
    if source != expected_source or target != expected_target:
        raise ValueError(
            f"transition refinement targeted {source!r} => {target!r}, expected "
            f"{expected_source!r} => {expected_target!r}"
        )
    if not content:
        raise ValueError("transition refinement returned empty edge content")
    return {"source": source, "target": target, "content": content}


def _apply_transition_patch_with_retry(
    subflow: str,
    current_skill: str,
    edge: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    evidence: str,
) -> str:
    from llm import chat

    source, target = edge["source"], edge["target"]
    prompt = _build_transition_patch_prompt(subflow, current_skill, edge, nodes, evidence)
    last_error: Exception | None = None
    for attempt in range(1, _SKILL_LLM_MAX_RETRIES + 1):
        try:
            patch = _parse_transition_patch(chat(prompt, temperature=0.0).strip(), source, target)
            return _upsert_transition_rule(current_skill, **patch)
        except Exception as exc:
            last_error = exc
            if attempt < _SKILL_LLM_MAX_RETRIES:
                delay = _SKILL_LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"  LLM edge refinement failed for {subflow}/{_short_label(source)} "
                    f"-> {_short_label(target)} (attempt {attempt}/{_SKILL_LLM_MAX_RETRIES}): "
                    f"{exc}; retrying in {delay:.1f}s"
                )
                time.sleep(delay)
    raise RuntimeError(
        f"LLM edge refinement failed for {subflow}/{source}=>{target} after "
        f"{_SKILL_LLM_MAX_RETRIES} attempts: {last_error}"
    )


def _build_branch_batch_patch_prompt(
    subflow: str,
    current_skill: str,
    action_sources: list[str],
    edges: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
    op_snippets: Dict[str, list[dict]],
    induced_rules: dict[str, list[dict[str, Any]]],
) -> str:
    action_blocks = [
        _render_action_evidence(node_id, nodes, op_snippets)
        for node_id in action_sources
        if node_id in nodes
    ]
    edge_blocks = [
        _render_transition_evidence(edge, nodes, induced_rules, op_snippets)
        for edge in edges
        if edge["source"] in nodes and edge["target"] in nodes
    ]
    expected_actions = [nodes[node_id]["label"] for node_id in action_sources if node_id in nodes]
    expected_edges = [f"{edge['source']} => {edge['target']}" for edge in edges]
    action_evidence_text = "\n".join("---\n" + block for block in action_blocks) or "(none)"
    edge_evidence_text = "\n".join("---\n" + block for block in edge_blocks) or "(none)"
    required_edges_text = "\n".join("- " + edge for edge in expected_edges) or "- (none)"
    return f"""You are performing the branch-refinement round of a two-stage
customer-service skill compiler.

The main-path seed skill below is already accepted. In this one joint pass,
integrate every listed off-main action node and every listed selected graph edge.
Consider all edge guards together before writing: preserve the main path, keep
returns to existing/main-path actions explicit, and use the jointly induced
priority/relation whenever supplied. Do not invent nodes, edges, policy, or
database outcomes. Do not call branches exclusive unless their evidence says
`exclusive`; overlapping guards must remain ordered by priority or fallback.

<current_skill>
{current_skill}
</current_skill>

<off_main_action_evidence>
{action_evidence_text}
</off_main_action_evidence>

<selected_non_main_edge_evidence>
{edge_evidence_text}
</selected_non_main_edge_evidence>

<required_coverage>
You must submit exactly one action operation for each action: {', '.join(expected_actions) or '(none)'}.
You must submit exactly one transition operation for each directed edge:
{required_edges_text}
</required_coverage>

<filesystem_mcp>
The runtime applies only these constrained local operations:

```json
{{
  "operations": [
    {{
      "op": "upsert_action_rule",
      "action": "ACTION_LABEL",
      "content": "complete block starting with #### `ACTION_LABEL`"
    }},
    {{
      "op": "upsert_transition_rule",
      "source": "FULL_SOURCE_NODE_ID",
      "target": "FULL_TARGET_NODE_ID",
      "content": "complete block starting with ##### `SOURCE_ACTION` -> `TARGET_ACTION`"
    }}
  ]
}}
```

`upsert_action_rule` may alter only that action's block in ACTION_RULES.
`upsert_transition_rule` may alter only the uniquely keyed `(source, target)`
block in TRANSITION_RULES. Thus an edge that returns to an already-defined
main-path action must be added as a transition operation, never by duplicating
or rewriting the target action. Do not return any other operation.
</filesystem_mcp>

Return ONLY one valid JSON object. Do not wrap it in Markdown fences.
"""


def _parse_branch_batch_patch(
    raw: str,
    expected_actions: set[str],
    expected_edges: set[tuple[str, str]],
    nodes: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I | re.S).strip()
    parsed = json.loads(text)
    operations = parsed.get("operations") if isinstance(parsed, dict) else None
    if not isinstance(operations, list):
        raise ValueError("branch batch must return an operations list")
    action_ops: dict[str, dict[str, str]] = {}
    edge_ops: dict[tuple[str, str], dict[str, str]] = {}
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("branch batch contains a non-object operation")
        op = operation.get("op")
        if op == "upsert_action_rule":
            action = str(operation.get("action") or "").strip()
            content = str(operation.get("content") or "").strip()
            if action not in expected_actions or action in action_ops:
                raise ValueError(f"unexpected or duplicate action operation: {action!r}")
            if not content.startswith(_action_heading(action)):
                raise ValueError(f"action operation has an invalid content heading: {action!r}")
            action_ops[action] = {"op": op, "action": action, "content": content}
        elif op == "upsert_transition_rule":
            source = str(operation.get("source") or "").strip()
            target = str(operation.get("target") or "").strip()
            content = str(operation.get("content") or "").strip()
            pair = (source, target)
            if pair not in expected_edges or pair in edge_ops:
                raise ValueError(f"unexpected or duplicate edge operation: {source!r} => {target!r}")
            source_label = nodes[source]["label"]
            target_label = nodes[target]["label"]
            if not content.startswith(f"##### `{source_label}` -> `{target_label}`"):
                raise ValueError(f"edge operation has an invalid content heading: {source!r} => {target!r}")
            edge_ops[pair] = {
                "op": op, "source": source, "target": target, "content": content,
            }
        else:
            raise ValueError(f"unsupported filesystem-MCP operation: {op!r}")
    if set(action_ops) != expected_actions:
        raise ValueError("branch batch omitted one or more required action operations")
    if set(edge_ops) != expected_edges:
        raise ValueError("branch batch omitted one or more required edge operations")
    return list(action_ops.values()) + list(edge_ops.values())


def _apply_branch_batch_patch_with_retry(
    subflow: str,
    current_skill: str,
    action_sources: list[str],
    edges: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
    op_snippets: Dict[str, list[dict]],
    induced_rules: dict[str, list[dict[str, Any]]],
) -> str:
    """Request all branch edits jointly, then apply each validated operation locally."""
    if not action_sources and not edges:
        return current_skill
    from llm import chat

    expected_actions = {nodes[node_id]["label"] for node_id in action_sources if node_id in nodes}
    expected_edges = {
        (edge["source"], edge["target"])
        for edge in edges
        if edge["source"] in nodes and edge["target"] in nodes
    }
    prompt = _build_branch_batch_patch_prompt(
        subflow, current_skill, action_sources, edges, nodes, op_snippets, induced_rules,
    )
    last_error: Exception | None = None
    for attempt in range(1, _SKILL_LLM_MAX_RETRIES + 1):
        try:
            operations = _parse_branch_batch_patch(
                chat(prompt, temperature=0.0).strip(), expected_actions, expected_edges, nodes,
            )
            updated = current_skill
            for operation in operations:
                if operation["op"] == "upsert_action_rule":
                    updated = _upsert_action_rule(updated, operation["action"], operation["content"])
                else:
                    updated = _upsert_transition_rule(
                        updated, operation["source"], operation["target"], operation["content"],
                    )
            return updated
        except Exception as exc:
            last_error = exc
            if attempt < _SKILL_LLM_MAX_RETRIES:
                delay = _SKILL_LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"  LLM joint branch refinement failed for {subflow} "
                    f"(attempt {attempt}/{_SKILL_LLM_MAX_RETRIES}): {exc}; retrying in {delay:.1f}s"
                )
                time.sleep(delay)
    raise RuntimeError(
        f"LLM joint branch refinement failed for {subflow} after "
        f"{_SKILL_LLM_MAX_RETRIES} attempts: {last_error}"
    )


def _compile_backbone_skill_iteratively(
    subflow: str,
    subgraph: dict[str, Any],
    op_snippets: Dict[str, list[dict]],
    transition_induction: dict[str, Any] | None,
) -> str:
    """Seed main path, then add off-spine nodes and selected edges separately."""
    seed_prompt, main_labels = _build_main_path_skill_prompt(subflow, subgraph, op_snippets)
    skill = _compile_skill_with_retry(
        seed_prompt,
        subflow,
        validator=lambda text: _validate_main_path_skill(text, main_labels),
    )
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    induced_rules = (transition_induction or {}).get("rules_by_source", {})
    sources = _branch_action_sources(subgraph)
    edges = _branch_transition_edges(subgraph)
    print(
        f"  Joint branch refinement for {subflow}: "
        f"{len(sources)} action nodes, {len(edges)} selected edges, 1 LLM call"
    )
    return _apply_branch_batch_patch_with_retry(
        subflow, skill, sources, edges, nodes, op_snippets, induced_rules,
    )


def _format_observed_condition(condition: dict[str, Any]) -> str:
    if condition.get("kind") == "session_entry":
        return "at session entry"
    facts: list[str] = []
    if condition.get("account_selected"):
        facts.append("an account has already been selected")
    if condition.get("failure_signal"):
        facts.append("a prior failure or retry signal is present")
    count = condition.get("min_credential_count")
    if isinstance(count, int) and count > 0:
        facts.append(f"at least {count} credential type(s) are observed")
    types = condition.get("common_credential_types") or []
    if types:
        facts.append("observed credentials include " + ", ".join(types))
    return "; ".join(facts) if facts else "when this transition is observed in similar sessions"


def _render_backbone_tree(subgraph: dict[str, Any]) -> str:
    """Render the complete arborescence as a compact ASCII tree for the LLM."""
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    backbone = subgraph.get("backbone", {})
    root = backbone.get("root", "ROOT")
    children: dict[str, list[str]] = defaultdict(list)
    for edge in backbone.get("edges", []):
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if target in nodes:
            children[source].append(target)
    order = {node_id: index for index, node_id in enumerate(backbone.get("compilation_order", []))}
    for source in children:
        children[source].sort(key=lambda node_id: (order.get(node_id, 10**9), nodes[node_id].get("label", node_id)))

    lines = ["ROOT"]
    visited: set[str] = set()

    def walk(source: str, prefix: str) -> None:
        child_nodes = [node_id for node_id in children.get(source, []) if node_id not in visited]
        for index, node_id in enumerate(child_nodes):
            visited.add(node_id)
            is_last = index == len(child_nodes) - 1
            connector = "`-- " if is_last else "|-- "
            label = nodes[node_id].get("label", node_id)
            lines.append(prefix + connector + f"`{label}`")
            walk(node_id, prefix + ("    " if is_last else "|   "))

    walk(root, "")
    # Keep disconnected artifacts visible instead of silently omitting them.
    for node_id in backbone.get("compilation_order", []):
        if node_id in nodes and node_id not in visited:
            lines.append("`-- " + f"`{nodes[node_id].get('label', node_id)}` (unattached)")
    return "\n".join(lines)


def _render_backbone_edge_table(subgraph: dict[str, Any]) -> str:
    """Render every selected backbone edge, including its support score."""
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    rows = []
    for edge in subgraph.get("backbone", {}).get("edges", []):
        source = str(edge.get("source", "ROOT"))
        target = str(edge.get("target", ""))
        source_label = "ROOT" if source == subgraph.get("backbone", {}).get("root", "ROOT") else nodes.get(source, {}).get("label", source)
        target_label = nodes.get(target, {}).get("label", target)
        rows.append(
            f"- `{source_label}` -> `{target_label}`; "
            f"support={edge.get('support', 0)}; score={edge.get('score', 0)}"
        )
    return "\n".join(rows) if rows else "(empty backbone)"


def _build_backbone_skill_prompt(
    subflow: str,
    subgraph: dict,
    op_snippets: Dict[str, list[dict]],
    transition_induction: dict[str, Any] | None = None,
) -> str:
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    backbone = subgraph.get("backbone", {})
    order = backbone.get("compilation_order", [])
    induced_rules = (transition_induction or {}).get("rules_by_source", {})

    allowed_actions = [nodes[node_id]["label"] for node_id in order if node_id in nodes]
    backbone_tree = _render_backbone_tree(subgraph)
    backbone_edge_table = _render_backbone_edge_table(subgraph)
    action_blocks: list[str] = []
    for node_id in order:
        node = nodes.get(node_id)
        if not node:
            continue
        slots = node.get("observed_slot_counts", [])
        contract = node.get("slot_contract", {})
        position_contract = "; ".join(
            f"arg{item['position']}: {', '.join(item.get('value_types') or ['value'])} "
            f"(required rate={item.get('required_rate', 0):.0%}; observed sources="
            f"{', '.join(item.get('source_types') or ['unresolved'])})"
            for item in contract.get("positions", [])
        )
        action_blocks.append(
            f"### {node['label']}\n"
            f"Ordered slot contract: count={slots or [0]}; {position_contract or 'no observed slot values'}\n"
            f"Reference-only slot examples (pattern evidence; never hard-code or reuse these values): "
            f"{json.dumps(node.get('slot_examples', [])[:3], ensure_ascii=False)}"
        )
        if induced_rules.get(node_id):
            rendered = "\n".join(
                f"- -> {nodes.get(rule['target'], {}).get('label', rule['target'])}; "
                f"mode={rule.get('status', 'underspecified')}; "
                f"condition={rule.get('condition') or '(insufficient observable evidence)'}"
                for rule in induced_rules[node_id]
            )
            action_blocks[-1] += "\nTransition evidence available for the later routing write pass:\n" + rendered
        else:
            action_blocks[-1] += "\nTransition evidence available for the later routing write pass:\n- No observed outgoing transition."

    evidence_blocks: list[str] = []
    for node_id in order:
        snippets = op_snippets.get(node_id, [])
        if snippets:
            evidence_blocks.append(
                f"**{nodes.get(node_id, {}).get('label', node_id)}**\n"
                f"```text\n{snippets[0]['snippet_text'][:280]}\n```\n"
                "These are reference examples only; abstract any slot values before writing rules."
            )

    return f"""You are compiling a compact, evidence-grounded customer-service skill.

## Contract
- Allowed actions, and only allowed actions: {', '.join(allowed_actions)}
- Coverage is mandatory: write exactly one `#### ` action-rule heading for EVERY
  allowed action listed above, including actions in any backbone branch. Do not
  omit a retained action merely because it belongs to a branch or recovery route.
- Do not reduce the backbone to a single main path. The complete maximum
  spanning backbone below is the structural organization of the skill and
  every retained backbone node must remain visible in the Backbone Tree.
- The routing prose will be written by a later constrained filesystem-MCP pass.
  In this seed pass, describe the action inventory and backbone organization;
  do not turn transition evidence into a rigid per-edge rule table.
- Slot examples and reference snippets are evidence, not executable values.
  Never hard-code any observed/reference slot value into the skill. At runtime,
  use only values explicitly available in the current dialogue state. If a
  few-shot illustration is useful, mask values as `<VALUE_1>`, `<VALUE_2>`,
  etc., label it as illustrative, and never turn it into a default argument.
- Slots must be REAL values grounded in the current dialogue state. Never put
  field names, placeholders, or example values into an action slot.
- Mine a reusable slot policy for every action with slots. For each ordered
  argument, state: (1) allowed source(s): current customer utterance, prior
  dialogue state, or scenario facts; (2) when it becomes usable; (3) what to
  do when it is missing (ask/verify, defer the action, or follow an
  evidence-backed alternative); and (4) whether it is newly collected or
  safely reused from an earlier action. Use only the observed source evidence
  and dialogue examples below. Do not invent semantic field names or turn an
  example-specific value into a rule.
- Use the complete backbone tree to explain parent-child organization, branch
  attachment, and rejoining relations for each action's transition cards. Do
  not claim that branches are mutually exclusive unless their induced guards
  are.
- Be concise and natural. Explain the workflow as a coherent procedure rather
  than a list of mechanically repeated edge rules.

## Skill
Skill ID: {subflow}
Sessions: {subgraph.get('n_sessions', 0)}
All actions are retained; the maximum spanning backbone determines the global
tree organization. It is not a single linear route.

## Maximum Spanning Backbone (Directed MST / Arborescence)
Tree view:
{backbone_tree}

Backbone edge table:
{backbone_edge_table}

## Per-Action Pre-Induced Transition Evidence
{chr(10).join(action_blocks) if action_blocks else '(none)'}

## Reference-Only Dialogue Evidence (Not Executable Values)
{chr(10).join(evidence_blocks) if evidence_blocks else '(see reference.md)'}

## Required Output
Write only this Markdown document:

```markdown
# Skill: {subflow}

## Intent
[One short evidence-grounded sentence.]

## Workflow
### Backbone Tree
```text
ROOT
└── `action-a`
    ├── `action-b`
    └── `action-c`
```
Reproduce the complete retained maximum spanning backbone as a compact tree.
Do not collapse it into one main path. Preserve every backbone node and parent-child edge.
Use ASCII connectors such as pipe-dash-dash and backtick-dash-dash; do not use Unicode box-drawing characters.

### Backbone Edges
- `ROOT` -> `action-a`
- `action-a` -> `action-b`
[List EVERY edge from the supplied Backbone edge table exactly once. This is
the authoritative parent-child placement contract for actions in the tree.]

### Routing Policies
<!-- ROUTING_SECTION_START -->
<!-- ROUTING_SECTION_END -->

### Action Rules
#### `action-a`
- Slots: ordered real values only; `arg1` is ...
- Position: parent `...`; backbone children `...`.
- Role: describe its contribution to the overall workflow in natural language.
- Routing details are written in the Routing Policies section.

#### `every-other-retained-action`
- [Repeat one concise rule for every remaining allowed action, including branch and retry actions.]

## Slot Discipline
- Use only real values available in the current dialogue state.
- Preserve the action's observed slot order and do not emit schema labels.
- Reference examples are not defaults. Never copy their literal slot values.
- Few-shot illustrations, if included, must use masked values such as `<VALUE_1>`.

## Slot Policies
#### `action-a`
- `arg1`: source(s) ..., usable when ..., missing behavior ..., reuse rule ...

## Reference
- `action-a`: see `reference.md`.
```
"""


def induce_transition_rules(
    subflow: str,
    subgraph: dict[str, Any],
    edge_cases: dict[str, list[dict[str, Any]]],
    max_retries: int = 3,
    skill_context: str | None = None,
) -> dict[str, Any]:
    """Induce continuation modes using local cases and the global skill draft."""
    from llm import chat

    def parse_json_object(raw: str) -> dict[str, Any] | None:
        """Best-effort parse for workflow responses that contain JSON noise.

        Workflow APIs occasionally return markdown fences, literal control
        characters, or a truncated JSON object. The transition compiler must
        degrade to conservative underspecified rules in those cases rather
        than fail an otherwise valid subflow.
        """
        text = str(raw or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
        candidates = [text]
        start = text.find("{")
        if start >= 0:
            candidates.append(text[start:])
        for candidate in candidates:
            for value in (candidate, re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", candidate)):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    continue
        return None

    nodes = {node["id"]: node["label"] for node in subgraph.get("nodes", [])}
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in subgraph.get("edges", []):
        outgoing[edge["source"]].append(edge)
    groups: list[str] = []
    allowed: dict[str, set[str]] = {}
    for source, edges in sorted(outgoing.items()):
        if not edges:
            continue
        allowed[source] = {edge["target"] for edge in edges}
        lines = [f"### Source: {source} ({nodes.get(source, source)})"]
        for edge in sorted(edges, key=lambda item: (-item["support"], item["target"])):
            key = f"{source} -> {edge['target']}"
            lines.append(f"Target: {edge['target']} ({nodes.get(edge['target'], edge['target'])})")
            for case in edge_cases.get(key, []):
                inter_action = str(case.get("inter_action_dialogue") or "").strip()
                lines.append(
                    "Dialogue between source action and target action:\n"
                    + (inter_action if inter_action else "(no intervening utterance)")
                )
                lines.append(
                    "Earlier dialogue context (only use when needed to interpret "
                    "the intervening exchange):\n"
                    + str(case.get("context", ""))[:1200]
                )
        groups.append("\n".join(lines))

    graph_overview = []
    for source, edges in sorted(outgoing.items()):
        targets = ", ".join(
            f"{edge['target']} (support={edge.get('support', 0)})"
            for edge in sorted(edges, key=lambda item: (-item.get("support", 0), item["target"]))
        )
        graph_overview.append(f"{source} -> {targets}")
    global_skill = str(skill_context or "(No previously compiled skill draft is available.)")

    discovery_prompt = f"""Discover continuation modes for a session-mined action graph.

The graph overview and current skill draft provide global context. Use them to
understand how a local transition fits the whole workflow, including its
backbone position, likely rejoin points, and neighboring decisions. Do not
rewrite the skill and do not invent graph edges.

For EACH source, compare ALL outgoing target groups jointly. The primary
evidence is the raw dialogue BETWEEN the source action and the target action.
Use it to identify interaction events: an agent proposal/question, a user
acceptance/rejection/correction, newly supplied information, an unresolved
request, or another exchange that explains why the next action differs. Use
the earlier prefix only to interpret that exchange. Do not treat the action
label, slot values, or a precomputed state variable as the explanation.
Treat literal values as instance-specific. Do not force an explanation into a
variable, boolean, or formal predicate. A valid interpretation may be ordinary
natural language describing how the intervening utterances differ, or may say
that the route is only supported by the preceding dialogue and lacks a stable
general trigger.
For each target choose exactly one status: `distinguishable` when the supplied
dialogue gives a meaningful clue, `ordered_fallback` when it is only valid
after a normal route cannot proceed, or `underspecified` when the evidence does
not uniquely explain the difference. For `underspecified`, leave `interpretation` empty
and preserve the uncertainty in `evidence`. Do not add targets.

<global_graph_overview>
{chr(10).join(graph_overview) or '(no retained graph edges)'}
</global_graph_overview>

<existing_skill_draft>
{global_skill}
</existing_skill_draft>

{chr(10).join(groups)}

Return ONLY JSON in this schema:
{{"modes_by_source": {{
  "SOURCE_ID": [{{"target": "TARGET_ID", "status": "distinguishable|ordered_fallback|underspecified", "interpretation": "...", "evidence": "..."}}]
}}}}"""

    def clean_modes(payload: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        raw_modes = payload.get("modes_by_source", {}) if isinstance(payload, dict) else {}
        result: dict[str, list[dict[str, str]]] = {}
        for source, targets in allowed.items():
            rows = raw_modes.get(source, []) if isinstance(raw_modes, dict) else []
            parsed = {str(row.get("target")): row for row in rows if isinstance(row, dict)}
            result[source] = []
            for target in sorted(targets):
                row = parsed.get(target, {})
                status = str(row.get("status") or "underspecified").lower()
                if status not in {"distinguishable", "ordered_fallback", "underspecified"}:
                    status = "underspecified"
                interpretation = str(
                    row.get("interpretation") or row.get("guard") or ""
                ).strip() if status != "underspecified" else ""
                result[source].append({
                    "target": target,
                    "status": status,
                    "condition": interpretation,
                    "evidence": str(row.get("evidence") or "").strip(),
                })
        return result

    def underspecified_modes() -> dict[str, list[dict[str, str]]]:
        return {
            source: [
                {
                    "target": target,
                    "status": "underspecified",
                    "condition": "",
                    "evidence": "",
                }
                for target in sorted(targets)
            ]
            for source, targets in allowed.items()
        }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            discovery_raw = chat(discovery_prompt, temperature=0.0).strip()
            discovery = parse_json_object(discovery_raw)
            if discovery is None:
                print(
                    f"  Transition induction warning for {subflow}: "
                    f"discovery returned invalid or truncated JSON; using conservative fallback"
                )
                return {
                    "rules_by_source": underspecified_modes(),
                    "discovery_raw_output": discovery_raw,
                    "verifier_raw_output": "",
                    "parse_warning": "invalid_or_truncated_discovery_json",
                }
            # Keep the single joint induction result as evidence for the
            # later filesystem-MCP writing pass. Do not run a second verifier:
            # the writer should express uncertainty naturally instead of
            # forcing every edge through another rigid classification step.
            rules_by_source = clean_modes(discovery)
            return {
                "rules_by_source": rules_by_source,
                "discovery_raw_output": discovery_raw,
                "verifier_raw_output": "",
                "verification_skipped": True,
            }
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(_SKILL_LLM_RETRY_BASE_DELAY * (2 ** attempt))
    raise RuntimeError(f"Continuation-mode induction failed for {subflow}: {last_error}")


def _build_skill_md_from_backbone_fallback(subflow: str, subgraph: dict) -> str:
    """Deterministic compact rendering used by tests or offline inspection."""
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    backbone = subgraph.get("backbone", {})
    order = backbone.get("compilation_order", [])
    local = subgraph.get("local_transitions", {})
    lines = [
        f"# Skill: {subflow}", "", "## Intent", "",
        f"Handle `{subflow}` requests using the observed action workflow.",
        "", "## Workflow", "", "### Backbone Tree", "", "```text",
        _render_backbone_tree(subgraph), "```",
        "", "Preserve the complete maximum spanning backbone; do not collapse it into one route.",
    ]
    lines.extend(["", "### Action Rules", ""])
    for node_id in order:
        node = nodes.get(node_id)
        if not node:
            continue
        lines.extend([f"#### `{node['label']}`", ""])
        contract = node.get("slot_contract", {})
        lines.append(
            f"- Slot contract: {contract.get('min_slots', 0)}-{contract.get('max_slots', 0)} ordered real value(s)."
        )
        lines.append("- Place this action according to its parent and children in the Backbone Tree.")
        transitions = local.get(node_id, [])
        if not transitions:
            lines.append("- No retained outgoing transition.")
        for edge in transitions:
            target = nodes.get(edge["target"], {"label": _short_label(edge["target"])})
            lines.append(f"- Observed continuation: `{node['label']}` -> `{target['label']}`.")
        lines.append("")
    lines.extend([
        "## Slot Discipline", "",
        "- Use only real values available in the current dialogue state.",
        "- Do not emit schema labels or placeholders as slot values.",
        "", "## Reference", "",
    ])
    for node_id in order:
        node = nodes.get(node_id)
        if node:
            lines.append(f"- `{node['label']}`: see `reference.md`.")
    return "\n".join(lines)


def _compile_skill_with_retry(
    prompt: str,
    subflow: str,
    validator: Callable[[str], None] | None = None,
) -> str:
    """Compile skill markdown; retry failures instead of silently falling back."""
    from llm import chat

    last_error: Exception | None = None
    for attempt in range(1, _SKILL_LLM_MAX_RETRIES + 1):
        try:
            text = chat(prompt, temperature=0.0).strip()
            if not text:
                raise RuntimeError("LLM returned an empty response")
            if text.startswith("```markdown"):
                text = text[len("```markdown"):].strip()
            if text.startswith("```"):
                text = text[3:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
            if not text:
                raise RuntimeError("LLM returned an empty markdown document")
            if validator is not None:
                validator(text)
            return text
        except Exception as exc:
            last_error = exc
            if attempt < _SKILL_LLM_MAX_RETRIES:
                delay = _SKILL_LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"  LLM skill compile failed for {subflow} "
                    f"(attempt {attempt}/{_SKILL_LLM_MAX_RETRIES}): {exc}; "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)

    raise RuntimeError(
        f"LLM skill compilation failed for {subflow} after "
        f"{_SKILL_LLM_MAX_RETRIES} attempts: {last_error}"
    )


def _build_subgraph_skill_prompt(
    subflow: str,
    nodes: list,
    edges: list,
    pathways: list,
    branches: list,
    op_snippets: Dict[str, list[dict]],
    coverage_pct: float,
    num_sessions: int,
) -> str:
    """Build LLM prompt from subgraph structure."""
    # Node list
    node_lines = []
    for n in nodes:
        node_lines.append(f"  - `{n['id']}` (freq={n['frequency']})")

    # Edge list
    edge_lines = []
    for e in edges:
        edge_lines.append(f"  - `{e['source']}` → `{e['target']}` (weight={e['weight']})")

    # Main pathway
    path_lines = []
    for i, p in enumerate(pathways):
        steps_str = " → ".join(
            s.get("next", s["node"]) if "next" in s else s["node"]
            for s in p["steps"]
        )
        path_lines.append(f"  Pathway {i+1} (weight={p['total_weight']}): {steps_str}")

    # Branch points
    branch_lines = []
    for bp in branches:
        targets = []
        for b in bp["branches"]:
            targets.append(f"`{b['target']}` (weight={b['weight']})")
        branch_lines.append(f"  At `{bp['node']}`: go to {' or '.join(targets)}")

    # Snippet examples
    snippet_text = ""
    if op_snippets:
        sample_ops = list(op_snippets.keys())
        parts = []
        for op in sample_ops:
            snips = op_snippets[op][:1]
            if snips:
                parts.append(f"**{op}**:\n```\n{snips[0]['snippet_text'][:250]}\n```")
        snippet_text = "\n\n".join(parts) if parts else ""

    return f"""You are documenting a customer service skill from a weighted action graph.
Write a skill card in Markdown that an AI agent can follow.

## Skill Context
- **Skill ID**: `{subflow}`
- **Coverage**: {coverage_pct:.0f}% ({num_sessions} sessions)

## Action Graph
### Nodes (operators)
{chr(10).join(node_lines) if node_lines else "(none)"}

### Edges (transitions, weight = occurrence count)
{chr(10).join(edge_lines) if edge_lines else "(none)"}

### Main Pathway (highest-weight path)
{chr(10).join(path_lines) if path_lines else "(none)"}

### Branch Points (decision points with multiple outgoing paths)
{chr(10).join(branch_lines) if branch_lines else "(none)"}

## Example Dialogue Snippets
{snippet_text or '(see reference.md)'}

## Output Format
Write a Markdown document with this structure:

```markdown
# Skill: {subflow}

## Intent
[1-2 sentences describing the customer need]

## Triggers
- keyword/phrases the customer might say

## Workflow
### Main Path
[Step-by-step description of the main pathway through the graph]

### Branch: [condition A]
[When condition A is met, take this path. Describe transitions between operators.]
**Transitions**: `op1` → `op2` → `op3`

### Branch: [condition B]
...

## Decision Points
[For each branch point, describe WHEN to choose which path. Use edge weights
as indicators of how common each path is.]

## Slot Policies
[For each operator that takes slots, describe each ordered argument's allowed
source (current customer utterance, prior dialogue state, or scenario facts),
when it becomes available, what to do when it is missing/unverified, and when
an earlier value may be reused. Do not write example-specific values or invent
slot field names.]

## Operator Reference
[Link each operator to its reference snippet. Format:]
- `op_name` — [brief description]. See [reference.md#op-name](reference.md#op-name)
```

## Important
- Derive branching CONDITIONS from operator names, edge patterns, and snippet context.
- Higher edge weight = more common transition. Use this to prioritise the main path.
- Treat slot values as stateful resources: infer source, availability,
  missing-value behavior, and safe reuse, not only their output order.
- Write ONLY the Markdown document, no extra commentary."""


def _build_skill_md_from_subgraph_fallback(
    subflow: str,
    nodes: list,
    edges: list,
    pathways: list,
    branches: list,
    op_snippets: Dict[str, list[dict]],
    coverage_pct: float,
    num_sessions: int,
) -> str:
    """Template-based skill.md from subgraph (no LLM)."""
    lines = [f"# Skill: {subflow}", "", f"*Coverage: {coverage_pct:.0f}% ({num_sessions} sessions)*", ""]

    # Intent
    lines.extend(["## Intent", "",
                  f"Handle customer requests related to `{subflow}`.", ""])

    # Workflow — main pathway
    lines.append("## Workflow")
    lines.append("")
    if pathways:
        lines.append("### Main Path")
        lines.append("")
        p = pathways[0]
        for i, step in enumerate(p["steps"], 1):
            node_label = step.get("label", step["node"])
            nxt = step.get("next", "")
            nxt_label = ""
            if nxt:
                for n in nodes:
                    if n["id"] == nxt:
                        nxt_label = n.get("label", nxt)
                        break
            arrow = f" → `{nxt_label}`" if nxt_label else ""
            lines.append(f"{i}. **`{node_label}`**{arrow}")
        lines.append("")

    # Branch points
    if branches:
        lines.append("### Decision Points")
        lines.append("")
        for bp in branches:
            lines.append(f"At **`{bp['label']}`**:")
            for b in bp["branches"]:
                lines.append(f"- → `{b['label']}` (weight={b['weight']})")
            lines.append("")

    # Edges summary
    if edges:
        lines.append("### All Transitions")
        lines.append("")
        for e in edges:
            lines.append(f"- `{_short_label(e['source'])}` → `{_short_label(e['target'])}` (w={e['weight']})")
        lines.append("")

    # Operator reference
    lines.append("## Operator Reference")
    lines.append("")
    for n in nodes:
        nid = n["id"]
        label = n.get("label", nid)
        anchor = label.replace(":", "-").replace(" ", "-").lower()
        has_snippets = "yes" if nid in op_snippets and op_snippets[nid] else "no"
        ref_link = f"operator-{anchor}"
        lines.append(f"- **`{label}`** — freq={n['frequency']}. "
                     f"[See snippets →](reference.md#{ref_link}) "
                     f"({has_snippets} snippets)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Skill.md generation (legacy — flat vertex list)
# ═══════════════════════════════════════════════════════════════

def build_skill_md_prompt(
    subflow: str,
    operators: list[str],
    op_snippets: Dict[str, list[dict]],
    coverage_pct: float,
    num_sessions: int,
) -> str:
    """Build prompt for LLM to generate skill.md (flat list version)."""
    ops_clean = []
    for op in operators:
        parts = op.split(":", 1)
        ops_clean.append(parts[1] if len(parts) == 2 else op)

    snippet_examples = []
    for op, snippets in op_snippets.items():
        if snippets:
            snippet_examples.append(f"**{op}**:\n```\n{snippets[0]['snippet_text'][:300]}\n```")
    snippets_text = "\n\n".join(snippet_examples) if snippet_examples else "(no snippets available)"

    return f"""You are documenting a customer service skill for an AI agent. Write a skill card in Markdown.

## Skill Context
- **Skill ID**: `{subflow}`
- **Coverage**: {coverage_pct:.0f}% ({num_sessions} training sessions)
- **Key Actions**: {', '.join(ops_clean)}

## Example Dialogue Snippets
{snippets_text}

## Output Format
```markdown
# Skill: {subflow}
## Intent
[1-2 sentence description]
## Triggers
- keyword 1
## Actions
1. **action** — description
## Strategy
[2-3 sentences]
## Expected Outcome
[what customer gets]
```

Write ONLY the Markdown, no extra commentary."""


def generate_skill_md_llm(
    subflow: str,
    operators: list[str],
    op_snippets: Dict[str, list[dict]],
    coverage_pct: float,
    num_sessions: int,
) -> str:
    """Use LLM to generate skill.md content (legacy flat version)."""
    prompt = build_skill_md_prompt(subflow, operators, op_snippets, coverage_pct, num_sessions)
    return _compile_skill_with_retry(prompt, subflow)


def build_skill_md_fallback(
    subflow: str,
    operators: list[str],
    op_snippets: Dict[str, list[dict]],
    coverage_pct: float,
    num_sessions: int,
) -> str:
    """Generate skill.md without LLM (fallback using raw operators)."""
    ops_clean = []
    for op in operators:
        parts = op.split(":", 1)
        ops_clean.append(parts[1] if len(parts) == 2 else op)

    lines = [
        f"# Skill: {subflow}", "",
        "## Intent",
        f"Handle customer requests related to `{subflow}`.", "",
        "## Triggers",
    ]
    name_parts = subflow.replace("_", " ").split()
    for p in name_parts[:5]:
        lines.append(f"- customer mentions \"{p}\"")
    lines.extend(["", "## Actions"])
    for i, op in enumerate(ops_clean, 1):
        lines.append(f"{i}. **{op}**")
    lines.extend([
        "", "## Strategy",
        f"Follow the action sequence above. Refer to `reference.md` for snippets.",
        "", "## Expected Outcome",
        f"Customer's `{subflow}` request is resolved.", "",
        f"---",
        f"*Coverage: {coverage_pct:.0f}% ({num_sessions} sessions)*",
    ])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Main writer
# ═══════════════════════════════════════════════════════════════

def write_skill_and_reference(
    subflow: str,
    skill_info: dict,
    conversations: list[dict],
    output_dir: Path,
    use_llm: bool = True,
    subgraph: dict | None = None,
) -> Tuple[Path, Path]:
    """Generate skill.md and reference.md for one subflow.

    If ``subgraph`` is provided, uses pathways + branches + edges to generate
    a richer skill.md with branching conditions.
    """
    operators = skill_info.get("selected_vertices", [])
    # If subgraph provided, extract operators from subgraph nodes
    if not operators and subgraph:
        operators = [n["id"] for n in subgraph.get("nodes", [])]
    coverage_pct = skill_info.get("coverage_pct", subgraph.get("coverage_pct", 0) if subgraph else 0)
    num_sessions = skill_info.get("num_sessions", subgraph.get("n_sessions", 0) if subgraph else 0)

    safe_name = subflow.replace("/", "_").replace("\\", "_").replace(":", "_")[:50]
    intent_dir = output_dir / safe_name
    intent_dir.mkdir(parents=True, exist_ok=True)

    # Extract dialogue snippets
    op_snippets = _find_operator_snippets(conversations, subflow, operators)

    # Generate skill.md — use subgraph if available
    if subgraph and (subgraph.get("pathways") or subgraph.get("edges")):
        skill_md = build_skill_md_from_subgraph(
            subflow, subgraph, op_snippets, use_llm=use_llm,
        )
    elif use_llm:
        skill_md = generate_skill_md_llm(
            subflow, operators, op_snippets, coverage_pct, num_sessions,
        )
        if not skill_md:
            skill_md = build_skill_md_fallback(
                subflow, operators, op_snippets, coverage_pct, num_sessions,
            )
    else:
        skill_md = build_skill_md_fallback(
            subflow, operators, op_snippets, coverage_pct, num_sessions,
        )

    skill_path = intent_dir / "skill.md"
    skill_path.write_text(skill_md, encoding="utf-8")

    # Generate transition-centered reference evidence. The graph miner is
    # deterministic here, so this does not add an LLM call.
    from skill_mining.backbone_workflow_mining import sample_transition_cases
    transition_cases = sample_transition_cases(
        subflow, conversations, max_cases_per_edge=3,
    )
    reference_md = build_reference_md(
        subflow, op_snippets, max_snippets_per_transition=3,
        transition_cases=transition_cases,
    )
    ref_path = intent_dir / "reference.md"
    ref_path.write_text(reference_md, encoding="utf-8")

    n_ops_with_snippets = sum(1 for s in op_snippets.values() if s)
    has_subgraph = "subgraph" if subgraph and subgraph.get("edges") else "flat"
    print(f"  {subflow}: {len(operators)} ops ({has_subgraph}), {n_ops_with_snippets} with snippets")

    return skill_path, ref_path


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-intent skill.md + reference.md from vertex sets"
    )
    parser.add_argument("--skills", required=True,
                        help="per_subflow_vertex_subsets.json path")
    parser.add_argument("--split", default="train",
                        help="ABCD split for snippet extraction")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Limit ABCD conversations loaded")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: outputs/skills)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM summary, use fallback template for skill.md")
    parser.add_argument("--max-intents", type=int, default=None,
                        help="Limit number of intents to process")
    parser.add_argument("--subgraph", default=None,
                        help="per_subflow_subgraphs.json from subgraph_mining.py "
                             "(enables pathway-based skill.md with branching)")
    args = parser.parse_args()

    skills_path = Path(args.skills)
    if not skills_path.exists():
        print(f"Error: {skills_path} not found")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load skills
    print(f"Loading skills from {skills_path}...")
    data = json.loads(skills_path.read_text(encoding="utf-8"))
    # Handle different formats
    if "intent_skills" in data:
        per_subflow = data["intent_skills"]
    elif "per_intent" in data:
        per_subflow = data["per_intent"]
    else:
        per_subflow = data
    print(f"  {len(per_subflow)} subflows")

    # Load ABCD conversations
    print(f"Loading ABCD {args.split} split...")
    conversations = load_abcd_data(args.split)
    if args.max_sessions:
        conversations = conversations[:args.max_sessions]
    print(f"  {len(conversations)} conversations")

    # Generate per subflow
    intents = sorted(per_subflow.items(), key=lambda x: -x[1].get("num_sessions", 0))
    if args.max_intents:
        intents = intents[:args.max_intents]

    # Load subgraphs if provided
    subgraphs: dict[str, dict] = {}
    if args.subgraph:
        sg_path = Path(args.subgraph)
        if sg_path.exists():
            subgraphs = json.loads(sg_path.read_text(encoding="utf-8"))
            print(f"Loaded {len(subgraphs)} subgraphs from {sg_path}")

    use_llm = not args.no_llm
    print(f"\nGenerating skill.md + reference.md for {len(intents)} intents "
          f"({'LLM' if use_llm else 'template'} mode, "
          f"subgraph={'yes' if subgraphs else 'no'})...")

    generated: list[dict] = []
    for subflow, skill_info in intents:
        sg = subgraphs.get(subflow)
        skill_path, ref_path = write_skill_and_reference(
            subflow, skill_info, conversations, out_dir,
            use_llm=use_llm, subgraph=sg,
        )
        generated.append({
            "subflow": subflow,
            "skill_md": str(skill_path),
            "reference_md": str(ref_path),
            "num_operators": len(skill_info.get("selected_vertices", [])),
        })

    # Index
    index_lines = ["# Skill Index", ""]
    for g in generated:
        index_lines.append(f"- [{g['subflow']}]({g['subflow']}/skill.md) "
                           f"({g['num_operators']} ops) "
                           f"— [reference]({g['subflow']}/reference.md)")
    (out_dir / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print(f"\nDone. Output: {out_dir}")
    print(f"  INDEX.md           — skill index")
    print(f"  {{intent}}/skill.md    — skill description")
    print(f"  {{intent}}/reference.md — dialogue snippets")
    print(f"  Generated {len(generated)} skill sets")


if __name__ == "__main__":
    main()
