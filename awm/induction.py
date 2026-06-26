"""Workflow induction for MultiWOZ — adapted from AWM/mind2web/online_induction.py.

Mirrors the exact flow:
1. Collect agent trajectories from results (``get_trajectory``)
2. Format as examples (``format_examples``)
3. Build prompt: INSTRUCTION + ONE_SHOT + examples
4. LLM generates workflow text
5. Filter/save workflow

For MultiWOZ, trajectories are ReAct agent logs showing the
thought→action→observation loop.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
# Prompt templates (mirrors instruction_action.txt + one_shot_action.txt)
# ══════════════════════════════════════════════════════════════════

_INDUCTION_INSTRUCTION = """You are an expert at analyzing task-oriented dialogue (ToD) agent trajectories \
and extracting reusable workflow patterns.

Given agent trajectories from MultiWOZ dialogues (each showing the agent's \
thought process, tool calls, and observations), identify:

1. **Success Patterns**: What sequence of actions reliably leads to task completion?
2. **Failure Patterns**: What mistakes does the agent make and how to avoid them?
3. **Domain-Specific Strategies**: For each domain (hotel, train, restaurant, attraction, \
taxi, hospital, police), what are the key steps?

## Output Format
Output ONLY workflow patterns in this format. No introduction, no summary.

### [Domain] - [Pattern Name]
**When**: [condition that triggers this pattern]
**Do**: [step-by-step actions]
**Avoid**: [common mistakes]

### [Domain] - [Another Pattern]
..."""

_INDUCTION_ONESHOT = """## Example: Inducing Workflows from Trajectories

Given these agent trajectories:

Query #1: Find a cheap guesthouse in the east with parking. Book for 3 people, 2 nights.
Actions and Environments:
[thought] Need to query hotel DB with constraints: type=guesthouse, price_range=cheap, area=east, parking=yes
[action] query_db {"domain": "hotel", "constraints": {"type": "guesthouse", "price range": "cheap", "area": "east", "parking": "yes"}}
[observation] Found: Alexander B&B, cheap, east, parking=yes
[thought] Hotel matches. Inform user and ask about booking.
[action] inform hotel name="Alexander B&B", area="east", price_range="cheap", parking="yes"
[observation] User wants to book for 3 people, 2 nights starting Tuesday.
[action] book hotel name="Alexander B&B", book_day="tuesday", book_stay="2", book_people="3"
[observation] Booking confirmed. Reference: ABC123XYZ
[action] inform hotel reference="ABC123XYZ"

Query #2: Find police station and get address, phone, postcode.
Actions and Environments:
[thought] Query police database for any station.
[action] query_db {"domain": "police", "constraints": {}}
[observation] Found: Parkside Police Station, Parkside, Cambridge, 01223358966, cb11jg
[thought] Got all info. Inform address, phone, postcode. Also need to request postcode explicitly.
[action] inform police name="Parkside Police Station", address="Parkside, Cambridge", phone="01223358966"
[action] request police postcode
[observation] Postcode is cb11jg.
[thought] All done.
[action] inform police postcode="cb11jg"

# Summary Workflows

### Hotel - Complete Booking Workflow
**When**: User requests a hotel with specific constraints and wants to book
**Do**: 1. Query hotel DB with ALL constraints 2. Present the matching hotel 3. Collect booking details 4. Book and provide reference number
**Avoid**: Forgetting to include all constraint attributes in the query; omitting the booking reference

### Police - Information Retrieval
**When**: User needs police station contact info
**Do**: 1. Query police DB 2. Inform name, address, phone 3. Request postcode if needed 4. Inform postcode
**Avoid**: Not explicitly requesting postcode when the goal requires it"""


# ══════════════════════════════════════════════════════════════════
# Trajectory formatting (mirrors get_trajectory + format_examples)
# ══════════════════════════════════════════════════════════════════

def _get_trajectory(log_path: str) -> list[dict]:
    """Parse a ReAct agent log file into trajectory steps.

    Mirrors ``get_trajectory()`` from online_induction.py.
    Reads a .md log file and extracts thought/action/observation triples.
    """
    if not os.path.exists(log_path):
        return []
    text = Path(log_path).read_text(encoding="utf-8")
    steps = []
    current = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("[thought]") or line.startswith("**Think"):
            if current:
                steps.append(current)
            current = {"thought": line}
        elif line.startswith("[action]") or line.startswith("ACTION:"):
            current["action"] = line
        elif line.startswith("[observation]") or line.startswith("OBSERVATION:"):
            current["observation"] = line
            steps.append(current)
            current = {}
    if current and "action" in current:
        steps.append(current)
    return steps


def _format_trajectory_steps(steps: list[dict]) -> str:
    """Format trajectory steps as text — mirrors action_reprs in format_examples."""
    lines = []
    for s in steps:
        for key in ("thought", "action", "observation"):
            if key in s:
                lines.append(s[key])
    return "\n".join(lines)


def _format_examples(
    cases: list[dict],
    prefix: str | None = None,
    suffix: str = "# Summary Workflows",
) -> str:
    """Format dialogue results for the induction prompt.

    Mirrors ``format_examples()`` from AWM/mind2web/utils/data.py.
    Each case is a dict with keys: dialogue_id, domains, goal, trajectory_steps.
    """
    lines = []
    for i, case in enumerate(cases):
        lines.append(f"Query #{i+1}: {case.get('goal', '')[:300]}")
        lines.append("Actions and Environments:")
        traj_text = case.get("trajectory_text", "")
        if traj_text:
            lines.append(traj_text[:3000])
        else:
            lines.append(f"Predicted: {json.dumps(case.get('prediction', {}))}")
        lines.append("")
    prompt = "\n".join(lines)
    if prefix:
        prompt = prefix + "\n" + prompt
    if suffix:
        prompt += "\n\n" + suffix
    return prompt


# ══════════════════════════════════════════════════════════════════
# Main induction function (mirrors online_induction.py main())
# ══════════════════════════════════════════════════════════════════

def induce_workflows(
    dialogues: list,
    predictions: list,
    eval_results: list[dict],
    model: str = "deepseek-chat",
    api_key: str | None = None,
    base_url: str | None = None,
    trajectory_dir: str | None = None,
    existing_workflow: str = "",
) -> str:
    """Induce workflow patterns from a batch of dialogues.

    Mirrors ``online_induction.py main()``:
    1. Collect trajectories
    2. Format as examples with format_examples()
    3. Build prompt: INSTRUCTION + ONE_SHOT + examples
    4. LLM generates workflow text
    5. Return workflow text (caller saves to file)

    Args:
        dialogues: List of Dialogue objects.
        predictions: List of Prediction objects.
        eval_results: Per-dialogue metrics (list of dicts with success, info_rate, etc.).
        model: LLM model name.
        api_key: API key.
        base_url: API base URL.
        trajectory_dir: Directory with agent trajectory .md files.
        existing_workflow: Previous workflow text (for incremental update context).

    Returns:
        Induced workflow text string.
    """
    # ── 1. Collect trajectories (mirrors get_trajectory loop) ──
    cases = []
    for dialogue, pred, metrics in zip(dialogues, predictions, eval_results):
        traj_text = ""
        if trajectory_dir:
            safe_id = dialogue.dialogue_id.replace("/", "_").replace("\\", "_")
            traj_path = os.path.join(trajectory_dir, f"{safe_id}.md")
            steps = _get_trajectory(traj_path)
            traj_text = _format_trajectory_steps(steps)

        cases.append({
            "dialogue_id": dialogue.dialogue_id,
            "domains": list(dialogue.domains),
            "goal": dialogue.goal.description,
            "trajectory_text": traj_text,
            "prediction": {
                "inform_slots": pred.inform_slots,
                "request_slots": pred.request_slots,
                "booking": pred.booking,
            },
            "success": metrics.get("success", False),
            "info_rate": metrics.get("info_rate", 0),
        })

    # ── 2. Format examples (mirrors format_examples) ────────────
    examples_text = _format_examples(cases)

    # ── 3. Build prompt: INSTRUCTION + ONE_SHOT + examples ──────
    domains = list(set(d for c in cases for d in c["domains"] if d != "general"))
    domain_str = ", ".join(domains) if domains else "all"

    prompt_parts = [
        _INDUCTION_INSTRUCTION,
        _INDUCTION_ONESHOT,
        f"## Agent Trajectories — Domain(s): {domain_str}\n{examples_text}",
    ]
    if existing_workflow.strip():
        prompt_parts.append(
            f"\n## Existing Workflow (update with new insights)\n{existing_workflow[:2000]}"
        )

    user_message = "\n\n".join(prompt_parts)

    # ── 4. LLM call (mirrors client.chat.completions.create) ────
    from llm import chat
    return chat(
        user_message,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
    )
