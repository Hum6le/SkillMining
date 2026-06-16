"""LLM Judge 提示词模板。"""

from __future__ import annotations

from .config import SCORING_DIMENSIONS

# ═══════════════════════════════════════════════════════════════════
# Judge 提示词模板
# ═══════════════════════════════════════════════════════════════════
JUDGE_BASE_PROMPT = """{role}

You are evaluating an AI agent's performance in a task-oriented dialogue (ToD) scenario. You will be given:
1. The user's goal description
2. The dialogue between the user and the AI agent
3. The agent's predicted output (inform slots, request slots, booking info)

Rate the agent on ALL of the following dimensions using a 1-5 Likert scale:

{dimensions}

Scoring rubric (integer scores):
- 1: Very poor / fundamentally failed
- 2: Below average / significant issues
- 3: Acceptable / meets basic requirements
- 4: Good / solid performance
- 5: Excellent / near-perfect

Pay special attention to the **{focus_dimension}** dimension, as this is your core area of expertise.

Return ONLY a valid JSON object with the following structure:
{{
  "scores": {{
{score_template}
  }},
  "reasoning": "<Brief explanation of your scores, emphasizing the {focus_dimension} dimension>"
}}"""


def build_judge_prompt(role: str, focus_dim: str) -> str:
    """Build the full judge system prompt from role and focus dimension."""
    dims_desc_lines: list[str] = []
    score_template_lines: list[str] = []
    for dim_key, dim_info in SCORING_DIMENSIONS.items():
        low, high = dim_info["range"]
        dims_desc_lines.append(
            f"- **{dim_key}** ({low}-{high}): {dim_info['description']}"
        )
        score_template_lines.append(f'    "{dim_key}": <integer {low}-{high}>')
    score_template_lines.append('    "overall": <integer 1-5>')

    dims_desc = "\n".join(dims_desc_lines)
    score_template = "\n".join(score_template_lines)

    return JUDGE_BASE_PROMPT.format(
        role=role,
        focus_dimension=focus_dim,
        dimensions=dims_desc,
        score_template=score_template,
    )


# ═══════════════════════════════════════════════════════════════════
# Combiner 提示词模板
# ═══════════════════════════════════════════════════════════════════
COMBINER_PROMPT = """{role}

Several specialist judges have independently evaluated the same task-oriented dialogue. Each has a different area of expertise.

Your task:
1. Carefully review all judges' scores and reasoning
2. Identify points of agreement and disagreement among them
3. Synthesize their evaluations into a final, balanced set of scores
4. Explain your reasoning, especially how you resolved any disagreements

Return ONLY a valid JSON object:
{{
  "scores": {{
{score_template}
  }},
  "reasoning": "<Your synthesis and final evaluation reasoning>",
  "judge_agreement": "<Which dimensions had consensus, and where did judges diverge?>"
}}"""


def build_combiner_prompt(role: str) -> str:
    """Build the combiner system prompt."""
    score_template_lines: list[str] = []
    for dim_key, dim_info in SCORING_DIMENSIONS.items():
        low, high = dim_info["range"]
        score_template_lines.append(f'    "{dim_key}": <integer {low}-{high}>')
    score_template_lines.append('    "overall": <integer 1-5>')

    return COMBINER_PROMPT.format(
        role=role,
        score_template="\n".join(score_template_lines),
    )


# ═══════════════════════════════════════════════════════════════════
# 对话格式化 —— 将 ToD 数据结构转为 Judge 输入
# ═══════════════════════════════════════════════════════════════════
def format_dialogue_for_judge(
    goal_description: str,
    turns_text: str,
    inform_slots: dict,
    request_slots: dict,
    booking: dict,
) -> str:
    """Format a ToD dialogue + agent prediction for judge evaluation."""
    parts: list[str] = []

    parts.append("## User Goal")
    parts.append(goal_description)
    parts.append("")

    parts.append("## Dialogue")
    parts.append(turns_text)
    parts.append("")

    parts.append("## Agent Predictions")
    if inform_slots:
        parts.append("### Inform Slots (information provided to user)")
        for domain, slots in inform_slots.items():
            slot_str = ", ".join(f"{k}={v}" for k, v in slots.items())
            parts.append(f"- {domain}: {slot_str}")
    else:
        parts.append("### Inform Slots: (none)")

    if request_slots:
        parts.append("### Request Slots (information requested from user)")
        for domain, slots in request_slots.items():
            parts.append(f"- {domain}: {', '.join(slots)}")
    else:
        parts.append("### Request Slots: (none)")

    if booking:
        parts.append("### Booking Info")
        for domain, info in booking.items():
            ref = info.get("reference", "N/A")
            parts.append(f"- {domain}: reference={ref}")
    else:
        parts.append("### Booking Info: (none)")

    return "\n".join(parts)


def build_judge_user_message(
    goal_description: str,
    turns_text: str,
    inform_slots: dict,
    request_slots: dict,
    booking: dict,
) -> str:
    """Build the user message (the dialogue to evaluate) for a judge."""
    formatted = format_dialogue_for_judge(
        goal_description, turns_text, inform_slots, request_slots, booking
    )
    return f"Please evaluate the following task-oriented dialogue:\n\n---\n{formatted}\n---"


def build_combiner_user_message(
    goal_description: str,
    turns_text: str,
    inform_slots: dict,
    request_slots: dict,
    booking: dict,
    judge_results: list[dict],
) -> str:
    """Build the user message for the combiner containing judge outputs."""
    formatted = format_dialogue_for_judge(
        goal_description, turns_text, inform_slots, request_slots, booking
    )

    judge_outputs: list[str] = []
    for jr in judge_results:
        judge_outputs.append(
            f"### {jr['judge_name']} (Focus: {jr['focus_dimension']})\n"
            f"Scores: {jr['scores']}\n"
            f"Reasoning: {jr['reasoning']}"
        )

    return (
        f"Dialogue to evaluate:\n---\n{formatted}\n---\n\n"
        f"Independent judge evaluations:\n\n"
        + "\n\n".join(judge_outputs)
    )
