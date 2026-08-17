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
) -> str:
    """Generate reference.md from operator→snippets mapping.

    Each operator gets up to ``max_snippets_per_op`` snippets, deduplicated
    by snippet text.  Sections have HTML anchors so skill.md can link to them.
    """
    lines: list[str] = []
    lines.append(f"# Reference: {subflow}")
    lines.append("")
    lines.append(f"Dialogue snippets for the `{subflow}` skill. "
                 f"Each section shows one operator with example contexts.")
    lines.append("")

    for op, snippets in sorted(op_snippets.items()):
        parts = op.split(":", 1)
        display_op = parts[1] if len(parts) == 2 else op
        anchor = display_op.replace(":", "-").replace(" ", "-").lower()

        lines.append(f'<a id="operator-{anchor}"></a>')
        lines.append(f"## {display_op}")
        lines.append("")

        # Deduplicate by snippet text, keep most diverse
        seen_texts: set[str] = set()
        unique_snippets = []
        for snip in snippets:
            key = snip["snippet_text"][:100].strip()
            if key not in seen_texts:
                seen_texts.add(key)
                unique_snippets.append(snip)

        for i, snip in enumerate(unique_snippets[:max_snippets_per_op], 1):
            lines.append(f"### Example {i} (convo={snip['convo_id']}, turn={snip['turn_index']})")
            lines.append("")
            lines.append("```text")
            lines.append(snip["snippet_text"])
            lines.append("```")
            lines.append("")

        if len(unique_snippets) > max_snippets_per_op:
            lines.append(f"*({len(unique_snippets) - max_snippets_per_op} more snippets not shown)*")
            lines.append("")

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
) -> str:
    """Compile a backbone skill in a main-path seed round plus branch patches.

    The first LLM call writes only the primary execution spine.  Subsequent
    calls receive one source action's retained local transitions and make a
    restricted ``upsert_action_rule`` filesystem-style tool call.  The runtime
    applies that call to exactly one Markdown action-rule block, so a branch
    refinement cannot erase unrelated rules already accepted into the skill.
    """
    if use_llm:
        return _compile_backbone_skill_iteratively(
            subflow, subgraph, op_snippets, transition_induction,
        )
    return _build_skill_md_from_backbone_fallback(subflow, subgraph)


_ACTION_RULES_START = "<!-- ACTION_RULES_START -->"
_ACTION_RULES_END = "<!-- ACTION_RULES_END -->"
_TRANSITION_RULES_START = "<!-- TRANSITION_RULES_START -->"
_TRANSITION_RULES_END = "<!-- TRANSITION_RULES_END -->"


def _action_heading(label: str) -> str:
    return f"#### `{label}`"


def _action_rule_labels(skill: str) -> set[str]:
    return set(re.findall(r"^####\s+`([^`]+)`\s*$", skill, flags=re.MULTILINE))


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
{chr(10).join('---\n' + block for block in action_blocks) if action_blocks else '(none)'}
</off_main_action_evidence>

<selected_non_main_edge_evidence>
{chr(10).join('---\n' + block for block in edge_blocks) if edge_blocks else '(none)'}
</selected_non_main_edge_evidence>

<required_coverage>
You must submit exactly one action operation for each action: {', '.join(expected_actions) or '(none)'}.
You must submit exactly one transition operation for each directed edge:
{chr(10).join('- ' + edge for edge in expected_edges) or '- (none)'}
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


def _build_backbone_skill_prompt(
    subflow: str,
    subgraph: dict,
    op_snippets: Dict[str, list[dict]],
    transition_induction: dict[str, Any] | None = None,
) -> str:
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    backbone = subgraph.get("backbone", {})
    main_path = backbone.get("main_path", [])
    order = backbone.get("compilation_order", [])
    local = subgraph.get("local_transitions", {})
    induced_rules = (transition_induction or {}).get("rules_by_source", {})

    allowed_actions = [nodes[node_id]["label"] for node_id in order if node_id in nodes]
    main_labels = [nodes[node_id]["label"] for node_id in main_path if node_id in nodes]
    action_blocks: list[str] = []
    for node_id in order:
        node = nodes.get(node_id)
        if not node:
            continue
        transitions = local.get(node_id, [])
        transition_lines = []
        for edge in transitions:
            target = nodes.get(edge["target"], {"label": _short_label(edge["target"])})
            transition_lines.append(
                f"- {node['label']} -> {target['label']} "
                f"[priority={edge.get('priority', 1)}; {edge['kind']}; support={edge['support']}; "
                f"P={edge['probability']}; observed state: "
                f"{_format_observed_condition(edge.get('condition', {}))}]"
            )
        slots = node.get("observed_slot_counts", [])
        contract = node.get("slot_contract", {})
        position_contract = "; ".join(
            f"arg{item['position']}: {', '.join(item.get('value_types') or ['value'])} "
            f"(required rate={item.get('required_rate', 0):.0%})"
            for item in contract.get("positions", [])
        )
        action_blocks.append(
            f"### {node['label']}\n"
            f"Ordered slot contract: count={slots or [0]}; {position_contract or 'no observed slot values'}\n"
            f"Transitions:\n" + ("\n".join(transition_lines) if transition_lines else "- No retained outgoing transition.")
        )
        if induced_rules.get(node_id):
            rendered = "\n".join(
                f"- priority {rule['priority']}: -> {nodes.get(rule['target'], {}).get('label', rule['target'])}; "
                f"condition={rule['condition']}; relation={rule['relation']}"
                for rule in induced_rules[node_id]
            )
            action_blocks[-1] += "\nJoint transition induction:\n" + rendered

    evidence_blocks: list[str] = []
    for node_id in order:
        snippets = op_snippets.get(node_id, [])
        if snippets:
            evidence_blocks.append(
                f"**{nodes.get(node_id, {}).get('label', node_id)}**\n"
                f"```text\n{snippets[0]['snippet_text'][:280]}\n```"
            )

    return f"""You are compiling a compact, evidence-grounded customer-service skill.

## Contract
- Allowed actions, and only allowed actions: {', '.join(allowed_actions)}
- The Main Path MUST follow this backbone order exactly: {' -> '.join(main_labels) or '(no multi-step path)'}
- For each action, use only its listed local transitions. Do not add actions,
  transitions, database outcomes, or policy requirements absent from evidence.
- A transition condition is an observed state pattern, not a universal business
  rule. Phrase it conservatively (for example, "when an account is already
  selected and the needed credentials are available").
- Slots must be REAL values grounded in the current dialogue state. Never put
  field names, placeholders, or example values into an action slot.
- Define an explicit State Machine section using only these runtime-observable
  variables: `last_completed_action`, `account_selected`,
  `credential_types`, `credential_count`, and `failure_signal`. For every
  action rule, state a precondition, an ordered slot contract, and a post-state
  update. Use `last_completed_action=<action>` as the default post-state update.
- Evaluate each action's outgoing rules in the listed priority order. Do not
  claim that branches are mutually exclusive unless their observed guards are.
- Be concise: one short main path and at most one short transition rule per
  retained outgoing edge. Do not repeat the global workflow under every node.
- When Joint transition induction is present, it is authoritative for branch
  conditions, relation (exclusive, overlap, or fallback), and priority. Do not
  replace it with conditions inferred from a single example.

## Skill
Skill ID: {subflow}
Sessions: {subgraph.get('n_sessions', 0)}
All actions are retained; the backbone only determines the primary order.

## Main Backbone
{' -> '.join(main_labels) or '(single-action skill)'}

## Per-Action Local Transition Evidence
{chr(10).join(action_blocks) if action_blocks else '(none)'}

## Representative Dialogue Evidence
{chr(10).join(evidence_blocks) if evidence_blocks else '(see reference.md)'}

## Required Output
Write only this Markdown document:

```markdown
# Skill: {subflow}

## Intent
[One short evidence-grounded sentence.]

## Workflow
### Main Path
1. `action-a` -> `action-b` -> ...

## State Machine
- Variables: ...

### Action Rules
#### `action-a`
- Preconditions: ...
- Slots: ordered real values only; `arg1` is ...
- Post-state: ...
- Priority 1: when [observed condition], transition to `action-b`.

## Slot Discipline
- Use only real values available in the current dialogue state.
- Preserve the action's observed slot order and do not emit schema labels.

## Reference
- `action-a`: see `reference.md`.
```
"""


def induce_transition_rules(
    subflow: str,
    subgraph: dict[str, Any],
    edge_cases: dict[str, list[dict[str, Any]]],
    max_retries: int = 3,
) -> dict[str, Any]:
    """Jointly infer outgoing-edge guards for each source action with an LLM."""
    from llm import chat

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
            lines.append(
                f"Target: {edge['target']} ({nodes.get(edge['target'], edge['target'])}); "
                f"support={edge['support']}; P={edge['probability']}; observed={edge.get('condition', {})}"
            )
            for case in edge_cases.get(key, []):
                lines.append(
                    f"Case state={case.get('state', {})}\n"
                    f"Case context:\n{case.get('context', '')[:500]}"
                )
        groups.append("\n".join(lines))

    prompt = f"""Infer compact transition guards for a mined customer-service action graph.

For EACH source action, compare ALL listed outgoing target cases jointly. Do
not infer a condition from one target in isolation. Conditions may use only
runtime-observable dialogue state (previous actions, entity availability,
customer request, explicit failure signal), never concrete example values.

For every listed target output: a concise condition, an integer priority, and
one relation label: `exclusive`, `overlap`, or `fallback`. Use `exclusive` only
when the cases support mutual exclusion. When guards overlap, preserve the
overlap label and resolve with priority; do not pretend they are disjoint.
Do not add actions or targets not present below.

{chr(10).join(groups)}

Return ONLY JSON in this schema:
{{"rules_by_source": {{
  "SOURCE_ID": [
    {{"target": "TARGET_ID", "condition": "...", "priority": 1,
      "relation": "exclusive|overlap|fallback"}}
  ]
}}}}
"""

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = chat(prompt, temperature=0.0).strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
            parsed = json.loads(text)
            raw_rules = parsed.get("rules_by_source", {}) if isinstance(parsed, dict) else {}
            rules_by_source: dict[str, list[dict[str, Any]]] = {}
            for source, rules in raw_rules.items():
                if source not in allowed or not isinstance(rules, list):
                    continue
                cleaned = []
                for rule in rules:
                    target = str(rule.get("target") or "")
                    relation = str(rule.get("relation") or "overlap").lower()
                    if target not in allowed[source] or relation not in {"exclusive", "overlap", "fallback"}:
                        continue
                    cleaned.append({
                        "target": target,
                        "condition": str(rule.get("condition") or "observed dialogue state supports this transition").strip(),
                        "priority": max(1, int(rule.get("priority", len(cleaned) + 1))),
                        "relation": relation,
                    })
                if cleaned:
                    rules_by_source[source] = sorted(cleaned, key=lambda item: (item["priority"], item["target"]))
            if rules_by_source:
                return {"rules_by_source": rules_by_source, "raw_output": raw}
            raise ValueError("LLM returned no valid transition rules")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(_SKILL_LLM_RETRY_BASE_DELAY * (2 ** attempt))
    raise RuntimeError(f"Transition induction failed for {subflow}: {last_error}")


def _build_skill_md_from_backbone_fallback(subflow: str, subgraph: dict) -> str:
    """Deterministic compact rendering used by tests or offline inspection."""
    nodes = {node["id"]: node for node in subgraph.get("nodes", [])}
    backbone = subgraph.get("backbone", {})
    main_path = backbone.get("main_path", [])
    order = backbone.get("compilation_order", [])
    local = subgraph.get("local_transitions", {})
    lines = [f"# Skill: {subflow}", "", "## Intent", "", f"Handle `{subflow}` requests using the observed action workflow.", "", "## Workflow", "", "### Main Path", ""]
    labels = [nodes[node_id]["label"] for node_id in main_path if node_id in nodes]
    lines.append(" -> ".join(f"`{label}`" for label in labels) if labels else "Use the retained action rules below.")
    lines.extend([
        "", "## State Machine", "",
        "- Track `last_completed_action`, `account_selected`, `credential_types`, `credential_count`, and `failure_signal`.",
        "", "### Action Rules", "",
    ])
    for node_id in order:
        node = nodes.get(node_id)
        if not node:
            continue
        lines.extend([f"#### `{node['label']}`", ""])
        contract = node.get("slot_contract", {})
        lines.append(
            f"- Slot contract: {contract.get('min_slots', 0)}-{contract.get('max_slots', 0)} ordered real value(s)."
        )
        lines.append(f"- Post-state: `last_completed_action={node['label']}`.")
        transitions = local.get(node_id, [])
        if not transitions:
            lines.append("- No retained outgoing transition.")
        for edge in transitions:
            target = nodes.get(edge["target"], {"label": _short_label(edge["target"])})
            lines.append(
                f"- [priority {edge.get('priority', 1)}; {edge['kind']}] When {_format_observed_condition(edge.get('condition', {}))}, "
                f"transition to `{target['label']}`."
            )
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

## Operator Reference
[Link each operator to its reference snippet. Format:]
- `op_name` — [brief description]. See [reference.md#op-name](reference.md#op-name)
```

## Important
- Derive branching CONDITIONS from operator names, edge patterns, and snippet context.
- Higher edge weight = more common transition. Use this to prioritise the main path.
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

    # Generate reference.md (max 5 diverse snippets per operator)
    reference_md = build_reference_md(subflow, op_snippets, max_snippets_per_op=5)
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
