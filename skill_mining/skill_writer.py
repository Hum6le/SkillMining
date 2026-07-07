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
) -> str:
    """Generate reference.md content from operator→snippets mapping."""
    lines: list[str] = []
    lines.append(f"# Reference: {subflow}")
    lines.append("")
    lines.append(f"Dialogue snippets extracted from training data for the `{subflow}` skill.")
    lines.append("Each section corresponds to one key operator (action).")
    lines.append("")

    for op, snippets in sorted(op_snippets.items()):
        # Clean up operator name for display
        parts = op.split(":", 1)
        display_op = parts[1] if len(parts) == 2 else op

        lines.append(f"## {display_op}")
        lines.append("")

        for i, snip in enumerate(snippets, 1):
            lines.append(f"### Example {i} (convo={snip['convo_id']}, turn={snip['turn_index']})")
            lines.append("")
            lines.append("```text")
            lines.append(snip["snippet_text"])
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Skill.md generation (LLM summary)
# ═══════════════════════════════════════════════════════════════

def build_skill_md_prompt(
    subflow: str,
    operators: list[str],
    op_snippets: Dict[str, list[dict]],
    coverage_pct: float,
    num_sessions: int,
) -> str:
    """Build prompt for LLM to generate skill.md."""
    ops_clean = []
    for op in operators:
        parts = op.split(":", 1)
        ops_clean.append(parts[1] if len(parts) == 2 else op)

    # Collect a few example snippets
    snippet_examples = []
    for op, snippets in list(op_snippets.items())[:5]:
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
Write a Markdown document with this EXACT structure:

```markdown
# Skill: {subflow}

## Intent
[A 1-2 sentence description of what the customer needs]

## Triggers
- [keyword/phrase 1]
- [keyword/phrase 2]
- [keyword/phrase 3]

## Actions
[Ordered list of actions the agent should take, with brief descriptions]
1. **action_name** — what this step does
2. **action_name** — what this step does
...

## Strategy
[2-3 sentences on how to handle this request type effectively]

## Expected Outcome
[What the customer should receive by the end]
```

Write ONLY the Markdown, no extra commentary."""


def generate_skill_md_llm(
    subflow: str,
    operators: list[str],
    op_snippets: Dict[str, list[dict]],
    coverage_pct: float,
    num_sessions: int,
) -> str:
    """Use LLM to generate skill.md content."""
    prompt = build_skill_md_prompt(
        subflow, operators, op_snippets, coverage_pct, num_sessions,
    )
    try:
        from llm_utils import ds_api_retry
        raw = ds_api_retry(prompt)
        text = raw.get("data", {}).get("text", "") if isinstance(raw, dict) else str(raw)
        # Clean up markdown code fences
        text = text.strip()
        if text.startswith("```markdown"):
            text = text[len("```markdown"):].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text
    except Exception as e:
        print(f"  LLM error for {subflow}: {e}")
        return ""


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
        f"# Skill: {subflow}",
        "",
        "## Intent",
        f"Handle customer requests related to `{subflow}`.",
        "",
        "## Triggers",
    ]
    # Basic triggers from the subflow name
    name_parts = subflow.replace("_", " ").split()
    for p in name_parts[:5]:
        lines.append(f"- customer mentions \"{p}\"")

    lines.extend([
        "",
        "## Actions",
    ])
    for i, op in enumerate(ops_clean[:15], 1):
        lines.append(f"{i}. **{op}**")

    lines.extend([
        "",
        "## Strategy",
        f"Follow the action sequence above. Refer to `reference.md` for example dialogue snippets.",
        "",
        "## Expected Outcome",
        f"Customer's `{subflow}` request is resolved.",
        "",
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
) -> Tuple[Path, Path]:
    """Generate skill.md and reference.md for one subflow.

    Returns:
        (skill_md_path, reference_md_path)
    """
    operators = skill_info.get("selected_vertices", [])
    coverage_pct = skill_info.get("coverage_pct", 0)
    num_sessions = skill_info.get("num_sessions", 0)

    # Safe directory name
    safe_name = subflow.replace("/", "_").replace("\\", "_").replace(":", "_")[:50]
    intent_dir = output_dir / safe_name
    intent_dir.mkdir(parents=True, exist_ok=True)

    # Extract dialogue snippets
    op_snippets = _find_operator_snippets(conversations, subflow, operators)

    # Generate skill.md
    if use_llm:
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

    # Generate reference.md
    reference_md = build_reference_md(subflow, op_snippets)
    ref_path = intent_dir / "reference.md"
    ref_path.write_text(reference_md, encoding="utf-8")

    # Summary
    n_ops_with_snippets = sum(1 for s in op_snippets.values() if s)
    print(f"  {subflow}: {len(operators)} ops, {n_ops_with_snippets} with snippets")

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

    use_llm = not args.no_llm
    print(f"\nGenerating skill.md + reference.md for {len(intents)} intents "
          f"({'LLM' if use_llm else 'template'} mode)...")

    generated: list[dict] = []
    for subflow, skill_info in intents:
        skill_path, ref_path = write_skill_and_reference(
            subflow, skill_info, conversations, out_dir, use_llm=use_llm,
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
