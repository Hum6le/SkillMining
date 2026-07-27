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
from typing import Any, Dict, List, Optional, Tuple

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


def _compile_skill_with_retry(prompt: str, subflow: str) -> str:
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
