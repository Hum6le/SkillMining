#!/usr/bin/env python3
r"""Skill Metadata Generator — LLM 将 per-subflow 算子集总结为 skill cards。

输入：per_subflow_vertex_subsets.json（从 abcd_session_hg.py 产出）
输出：skill_metadata.json — 每个 subflow 一张可被 agent 理解和选择的 skill card

用法：
  python skill_mining/skill_metadata.py \
    --input skill_mining/output/abcd_session_hg/per_subflow_vertex_subsets.json \
    --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_SKILL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SKILL_DIR.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SKILL_DIR) in sys.path:
    sys.path.remove(str(_SKILL_DIR))
sys.path.insert(0, str(_SKILL_DIR))

from llm_utils import ds_api_retry

_OUTPUT_DIR = _SKILL_DIR / "output" / "abcd_intent"


def extract_workflow_text(res: Any) -> str:
    """从 API 响应中提取文本。"""
    data = res.get("data") if isinstance(res, dict) else res
    if isinstance(data, dict):
        for key in ("res", "output", "text", "result", "data"):
            if key in data:
                value = data[key]
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    for sk in ("res", "text", "output", "result"):
                        if sk in value and isinstance(value[sk], str):
                            return value[sk]
        return json.dumps(data, ensure_ascii=False)
    return str(data).strip()


def call_llm(prompt: str) -> str:
    res = ds_api_retry(prompt)
    return extract_workflow_text(res)


def generate_skill_metadata(
    per_subflow_skills: Dict[str, dict],
    batch_size: int = 5,
) -> Dict[str, dict]:
    """为每个 subflow 的算子集生成 skill card metadata。

    每批处理 batch_size 个 subflow，LLM 一次性生成多张 skill card。

    Returns:
        {subflow_name: {intent, triggers, key_actions, strategy, expected_outcome}}
    """
    subflows = sorted(per_subflow_skills.items(), key=lambda x: -x[1].get("num_sessions", 0))
    all_metadata: Dict[str, dict] = {}

    for batch_start in range(0, len(subflows), batch_size):
        batch = subflows[batch_start:batch_start + batch_size]

        # 构建批量 prompt
        skill_descriptions = []
        for sf_name, sf_info in batch:
            ops = sf_info.get("selected_vertices", [])
            ops_clean = []
            for op in ops:
                parts = op.split(":", 1)
                ops_clean.append(parts[1] if len(parts) == 2 else op)
            skill_descriptions.append(
                f"### {sf_name}\n"
                f"Sessions: {sf_info.get('num_sessions', '?')}\n"
                f"Coverage: {sf_info.get('coverage_pct', '?')}%\n"
                f"Key Actions: {', '.join(ops_clean)}\n"
            )

        prompt = f"""You are creating skill cards for a customer service AI agent. Each skill card describes a specific type of customer request and the actions the agent should take.

Below are skill definitions mined from real customer service dialogues. For each one, write a concise skill card with:
1. **intent**: A short (5-15 word) description of what the customer wants
2. **triggers**: 3-5 keywords/phrases the customer might say that indicate this skill is needed
3. **key_actions**: The recommended action sequence (from the provided actions, in logical order)
4. **strategy**: 2-3 sentences on how to handle this type of request effectively
5. **expected_outcome**: What the customer should receive by the end

## Skills to Describe

{chr(10).join(skill_descriptions)}

## Output Format
Output a JSON object with one key per skill name:

```json
{{
  "subflow_name": {{
    "intent": "...",
    "triggers": ["...", "..."],
    "key_actions": ["...", "..."],
    "strategy": "...",
    "expected_outcome": "..."
  }}
}}
```

Output ONLY the JSON object, no other text."""

        print(f"  Generating metadata for {len(batch)} skills ({batch_start+1}-{batch_start+len(batch)}/{len(subflows)})...")
        try:
            raw = call_llm(prompt)
            # 提取 JSON
            json_match = _extract_json(raw)
            if json_match:
                batch_metadata = json.loads(json_match)
                all_metadata.update(batch_metadata)
            else:
                print(f"    Warning: could not parse JSON from LLM output")
                print(f"    Raw (first 300 chars): {raw[:300]}")
        except Exception as e:
            print(f"    Error: {e}")

    return all_metadata


def _extract_json(text: str) -> str | None:
    """从 LLM 输出中提取 JSON 块。"""
    import re
    # 尝试 ```json ... ``` 格式
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # 尝试直接的 JSON 对象
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return None


def format_skill_cards_prompt(metadata: Dict[str, dict]) -> str:
    """将所有 skill cards 格式化为可注入 agent prompt 的文本。

    Agent 会看到这份菜单，自行选择最合适的 skill。
    """
    if not metadata:
        return ""

    cards: list[str] = []
    cards.append("## Available Skills")
    cards.append(
        "Below are the skills you can use. Read the conversation, "
        "select the ONE skill that best matches the customer's need, "
        "then follow that skill's actions and strategy to respond.\n"
    )

    for i, (name, card) in enumerate(sorted(metadata.items()), 1):
        triggers = ", ".join(f'"{t}"' for t in card.get("triggers", [])[:5])
        actions = " → ".join(card.get("key_actions", [])[:8])
        cards.append(
            f"### [{i}] {card.get('intent', name)}\n"
            f"- **Skill ID**: `{name}`\n"
            f"- **When to use**: customer mentions {triggers}\n"
            f"- **Actions**: {actions}\n"
            f"- **Strategy**: {card.get('strategy', '')}\n"
            f"- **Expected outcome**: {card.get('expected_outcome', '')}\n"
        )

    return "\n".join(cards)


def main():
    parser = argparse.ArgumentParser(
        description="Generate skill metadata cards from per-subflow vertex sets"
    )
    parser.add_argument("--input", required=True,
                        help="per_subflow_vertex_subsets.json path")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of skills per LLM call")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading skills from {input_path}...")
    data = json.loads(input_path.read_text(encoding="utf-8"))

    # 兼容两种格式：裸 dict 或带 wrapper key
    if "intent_skills" in data:
        per_subflow = data["intent_skills"]
    elif "per_intent" in data:
        per_subflow = data["per_intent"]
    else:
        per_subflow = data

    print(f"  {len(per_subflow)} subflows")

    # 生成 metadata
    print("Generating skill metadata via LLM...")
    metadata = generate_skill_metadata(per_subflow, batch_size=args.batch_size)
    print(f"  Generated {len(metadata)} skill cards")

    # 保存
    metadata_path = out_dir / "skill_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Metadata saved → {metadata_path}")

    # 格式化为 agent prompt 文本
    prompt_text = format_skill_cards_prompt(metadata)
    prompt_path = out_dir / "skill_cards_prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    print(f"Prompt text saved → {prompt_path} ({len(prompt_text.splitlines())} lines)")

    # 统计
    print(f"\nSkill cards summary:")
    for name, card in sorted(metadata.items()):
        print(f"  {name:35s}  intent: {card.get('intent', '?')[:60]}")


if __name__ == "__main__":
    main()
