"""AWM Workflow Induction for MultiWOZ.

The core AWM operation: given a batch of agent trajectories (successes
and failures), call an LLM to abstract reusable workflow patterns.

Mirrors ``AWM/mind2web/online_induction.py`` adapted for ToD dialogues.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
# Induction prompt
# ══════════════════════════════════════════════════════════════════

_INDUCTION_SYSTEM = """You are an expert at analyzing task-oriented dialogue (ToD) agent \
trajectories and extracting reusable workflow patterns.

Given a set of agent trajectories (each showing the agent's actions, \
observations, and final outcome), identify:

1. **Success Patterns**: What sequence of actions reliably leads to task completion?
   What heuristics does the agent use correctly?
2. **Failure Patterns**: What mistakes does the agent make? What should be avoided?
3. **Domain-Specific Strategies**: For each domain (hotel, train, restaurant, etc.),
   what are the key steps and common pitfalls?

## Output Format

Output ONLY the workflow patterns in this format. No introduction, no summary.

### [Domain] - [Short Pattern Name]
**When**: [condition that triggers this pattern]
**Do**: [step-by-step actions the agent should take]
**Avoid**: [common mistakes to avoid]

### [Domain] - [Another Pattern Name]
...
"""


def _format_trajectory(dialogue, prediction, eval_metrics, trajectory_log: str = "") -> str:
    """Format one dialogue result for the induction prompt."""
    success = "SUCCESS" if eval_metrics.get("success") else "FAILURE"
    domains = ", ".join(dialogue.domains)
    goal = dialogue.goal.description[:300]
    ir = eval_metrics.get("info_rate", 0)
    inform = f"{eval_metrics.get('inform_correct', '?')}/{eval_metrics.get('inform_total', '?')}"
    request = f"{eval_metrics.get('request_correct', '?')}/{eval_metrics.get('request_total', '?')}"

    lines = [
        f"## Dialogue: {dialogue.dialogue_id} [{success}]",
        f"Domains: {domains}",
        f"Goal: {goal}",
        f"Metrics: IR={ir:.2f}  inform={inform}  request={request}",
    ]
    if trajectory_log:
        # Truncate to avoid blowing up the prompt
        truncated = trajectory_log[:2000]
        lines.append(f"Trajectory:\n{truncated}")
    else:
        # Fallback: show prediction summary
        lines.append(f"Predicted inform: {json.dumps(prediction.inform_slots)}")
        lines.append(f"Predicted request: {json.dumps(prediction.request_slots)}")
        lines.append(f"Booking: {json.dumps(prediction.booking)}")

    return "\n".join(lines)


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
    """Induce workflow patterns from a batch of agent trajectories.

    This is the core AWM operation.  It calls an LLM to analyze a batch
    of dialogue results and extract reusable workflow patterns.

    Args:
        dialogues: List of Dialogue objects.
        predictions: List of Prediction objects.
        eval_results: List of per-dialogue metric dicts (from evaluate_predictions).
        model: LLM model name.
        api_key: API key (resolved from env if None).
        base_url: API base URL.
        trajectory_dir: Optional directory containing trajectory .md files.
        existing_workflow: Current workflow text (for incremental update).

    Returns:
        Induced workflow pattern text.  Empty string if LLM call fails.
    """
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    url = base_url or os.environ.get("OPENAI_BASE_URL", None)
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    # ── Build the examples section ─────────────────────────────
    examples_parts = []
    for dialogue, pred, metrics in zip(dialogues, predictions, eval_results):
        traj_log = ""
        if trajectory_dir:
            safe_id = dialogue.dialogue_id.replace("/", "_").replace("\\", "_")
            traj_path = os.path.join(trajectory_dir, f"{safe_id}.md")
            if os.path.exists(traj_path):
                traj_log = Path(traj_path).read_text(encoding="utf-8")
        examples_parts.append(_format_trajectory(dialogue, pred, metrics, traj_log))

    examples_text = "\n\n".join(examples_parts)

    # ── Build full prompt ─────────────────────────────────────
    user_parts = [f"## Agent Trajectories ({len(dialogues)} dialogues)\n"]
    user_parts.append(examples_text)

    if existing_workflow.strip():
        user_parts.append("\n## Existing Workflow Patterns\n")
        user_parts.append(existing_workflow)
        user_parts.append("\nUpdate the patterns above with new insights from these trajectories.")

    user_parts.append(
        "\n\nExtract workflow patterns from these trajectories. "
        "Focus on actionable heuristics that would help an agent succeed."
    )

    # ── Call LLM ──────────────────────────────────────────────
    client = OpenAI(api_key=key, **(dict(base_url=url) if url else {}))

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _INDUCTION_SYSTEM},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            max_tokens=2048,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [AWM induction] LLM call failed: {exc}")
        return ""
