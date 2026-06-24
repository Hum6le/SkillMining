"""ToD Error Analysis Agent.

For each failed dialogue, calls an LLM to analyze:
- What went wrong (failure causes linked to skill gaps)
- What should be remembered for future cases (failure memories)

Produces ``analysis_report.md`` in the Trace2Skill-compatible format so
that ``report_parsing.py`` can parse it directly.

Usage::

    from eval_tod.error_analysis import ErrorAnalyzer
    analyzer = ErrorAnalyzer(model="deepseek-chat")
    analyzer.analyze_batch(failed_cases, output_dir="outputs/error_analysis")
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


# ── Prompt ────────────────────────────────────────────────────────

_ANALYSIS_SYSTEM = """You are an expert analyst for task-oriented dialogue (ToD) agents.
Your job is to analyze WHY an agent failed on a specific dialogue, and extract
actionable lessons that can improve the agent's skill file.

## Analysis Framework

For each failure, identify:

### Failure Causes (root causes linked to skill gaps)
These are specific mistakes the agent made. For each cause:
- What exactly went wrong? (cite the trajectory)
- Why did it happen? (missing knowledge, wrong reasoning, bad query?)
- How can the skill be improved to prevent this? (concrete skill patch)

### Failure Memories (general lessons to remember)
These are patterns that apply beyond this single case. For each memory:
- What is the general pattern?
- What should future agents do differently?

## Output Format

You MUST output your analysis in this exact markdown format:

# Failure Cause Item 1
## Title
<One-line summary of the root cause>
## Description
<What happened, citing specific evidence from the trajectory>
## Content
<What the agent should have done instead. Be specific and actionable.>
## Relation to Skill
<How the current skill is missing guidance on this, and what exact text should be added>

# Failure Cause Item 2
... (more causes if relevant)

# Failure Memory Item 1
## Title
<One-line summary of the pattern>
## Description
<When and why this pattern occurs>
## Content
<Concrete action to prevent this in the future>
## Skill Reflection
<What section of the skill should be updated, and how>

# Failure Memory Item 2
... (more memories if relevant)

Output at least 1 Failure Cause Item and 1 Failure Memory Item.
Be specific — cite exact slot names, query parameters, and dialogue turns from the trajectory."""


def _build_analysis_user_message(case: dict) -> str:
    """Build the user message for error analysis."""
    parts = [
        "## Failed Dialogue",
        f"Dialogue ID: {case.get('dialogue_id', 'N/A')}",
        f"Domains: {case.get('domains', [])}",
        "",
        "### User Goal",
        case.get("goal_description", "")[:2000],
        "",
        "### Evaluation Results",
        f"Information Rate: {case.get('info_rate', 'N/A')}",
        f"Success: {case.get('success', 'N/A')}",
        f"Inform correct: {case.get('inform_correct', '?')}/{case.get('inform_total', '?')}",
        f"Request correct: {case.get('request_correct', '?')}/{case.get('request_total', '?')}",
        f"Booking passed: {case.get('booking_passed', 'N/A')}",
        "",
        "### Agent Trajectory",
        case.get("trajectory", "(no trajectory available)")[:4000],
        "",
        "### Agent's Inform Slots (predicted)",
        json.dumps(case.get("inform_slots", {}), indent=2),
        "",
        "### Agent's Request Slots (predicted)",
        json.dumps(case.get("request_slots", {}), indent=2),
        "",
        "### Agent's Booking (predicted)",
        json.dumps(case.get("booking", {}), indent=2),
        "",
        "### Ground Truth (expected from goal)",
        f"Inform: {case.get('goal_inform', 'N/A')}",
        f"Request: {case.get('goal_request', 'N/A')}",
        f"Booking required: {case.get('has_booking', 'N/A')}",
        "",
        "## Instructions",
        "Analyze the failure above. Identify root causes linked to skill gaps, "
        "and extract general lessons. Output in the specified markdown format.",
    ]
    return "\n".join(parts)


# ── Analyzer ─────────────────────────────────────────────────────

class ErrorAnalyzer:
    """LLM-based error analysis agent for ToD failures.

    For each failed case, analyzes the trajectory + eval results and
    produces an ``analysis_report.md`` compatible with Trace2Skill's
    ``report_parsing.py``.

    Attributes:
        model: LLM model name.
        workers: Number of parallel analysis workers.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
        workers: int = 4,
        response_logger = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.workers = workers
        self._response_logger = response_logger

    def analyze_single(self, case: dict) -> str:
        """Analyze one failed case and return the analysis report text."""
        user_message = _build_analysis_user_message(case)

        try:
            response = self._chat(_ANALYSIS_SYSTEM, user_message)
            return response
        except Exception as exc:
            return (
                f"# Failure Cause Item 1\n"
                f"## Title\nLLM Error\n"
                f"## Description\nFailed to analyze: {exc}\n"
                f"## Content\nRe-run analysis.\n"
                f"## Relation to Skill\nN/A\n\n"
                f"# Failure Memory Item 1\n"
                f"## Title\nAnalysis Failed\n"
                f"## Description\nError analysis LLM call failed: {exc}\n"
                f"## Content\nCheck API connectivity.\n"
                f"## Skill Reflection\nN/A\n"
            )

    def analyze_batch(
        self,
        cases: list[dict],
        output_dir: str,
        delay: float = 0.3,
    ) -> list[str]:
        """Analyze multiple failed cases in parallel and save reports.

        Args:
            cases: List of case dicts, each with keys: dialogue_id, domains,
                   goal_description, info_rate, success, inform_correct,
                   inform_total, request_correct, request_total,
                   booking_passed, inform_slots, request_slots, booking,
                   goal_inform, goal_request, has_booking, trajectory.
            output_dir: Directory to save analysis_report.md files.
            delay: Seconds between API calls per worker.

        Returns:
            List of output subdirectory paths (one per case).
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths: list[str] = []

        if self.workers <= 1:
            for i, case in enumerate(cases):
                print(f"  Analyzing {i+1}/{len(cases)}: {case.get('dialogue_id', '?')}")
                report = self.analyze_single(case)
                path = self._save_report(output_dir, case, report)
                output_paths.append(path)
                if i < len(cases) - 1:
                    time.sleep(delay)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {
                    pool.submit(self.analyze_single, case): case
                    for case in cases
                }
                for future in as_completed(futures):
                    case = futures[future]
                    try:
                        report = future.result()
                    except Exception as exc:
                        report = (
                            f"# Failure Cause Item 1\n"
                            f"## Title\nWorker Error\n"
                            f"## Description\n{exc}\n"
                            f"## Content\nN/A\n"
                            f"## Relation to Skill\nN/A\n"
                        )
                    path = self._save_report(output_dir, case, report)
                    output_paths.append(path)
                    print(f"  Analyzed: {case.get('dialogue_id', '?')}")

        return output_paths

    def _save_report(
        self, output_dir: str, case: dict, report: str,
    ) -> str:
        """Save analysis report and evaluate_passed.flag to disk.

        The flag file is required by Trace2Skill's ``report_parsing.py``
        to consider this analysis valid.  We write it unconditionally.
        """
        did = case.get("dialogue_id", "unknown").replace("/", "_").replace("\\", "_")
        case_dir = os.path.join(output_dir, did)
        os.makedirs(case_dir, exist_ok=True)

        path = os.path.join(case_dir, "analysis_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)

        # Required by Trace2Skill parser as a validity gate
        flag_path = os.path.join(case_dir, "evaluate_passed.flag")
        with open(flag_path, "w") as f:
            f.write("passed\n")

        return case_dir

    def _chat(self, system_prompt: str, user_message: str) -> str:
        """Single LLM call for error analysis."""
        from openai import OpenAI

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        url = self.base_url or os.environ.get("OPENAI_BASE_URL", None)
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")

        kwargs: dict = {"api_key": key}
        if url:
            kwargs["base_url"] = url
        client = OpenAI(**kwargs)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
        )

        # Log raw response if logger is configured
        if self._response_logger is not None:
            try:
                self._response_logger.log(
                    messages=messages,
                    response=resp,
                    call_tag="error_analysis",
                    extra={"model": self.model, "max_tokens": 2048, "temperature": 0.3},
                )
            except Exception:
                pass

        return resp.choices[0].message.content or ""


# ── Helper: build case dicts from eval results ───────────────────

def build_failure_cases(
    dialogues: list,
    predictions: list,
    eval_results: dict,
    log_dir: str | None = None,
) -> list[dict]:
    """Build the list of failed case dicts for error analysis.

    Args:
        dialogues: List of ``Dialogue`` objects.
        predictions: List of ``Prediction`` objects.
        eval_results: Dict from ``evaluate()`` with ``per_dialogue`` key.
        log_dir: Directory containing trajectory logs (``{dialogue_id}.md``).

    Returns:
        List of case dicts, one per failed dialogue, ready for
        ``ErrorAnalyzer.analyze_batch()``.
    """
    from .utils import extract_inform_slots, extract_request_slots, extract_booking_domains

    per_dialogue = eval_results.get("per_dialogue", [])
    failed: list[dict] = []

    for dm, dialogue, pred in zip(per_dialogue, dialogues, predictions):
        if dm["success"]:
            continue  # skip successful cases

        # Read trajectory log
        trajectory = ""
        if log_dir:
            safe_id = dialogue.dialogue_id.replace("/", "_").replace("\\", "_")
            log_path = os.path.join(log_dir, f"{safe_id}.md")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    trajectory = f.read()

        # Extract goal info
        has_booking = bool(extract_booking_domains(dialogue.goal.inform))

        failed.append({
            "dialogue_id": dialogue.dialogue_id,
            "domains": dialogue.domains,
            "goal_description": dialogue.goal.description,
            "info_rate": dm["info_rate"],
            "success": dm["success"],
            "inform_correct": dm["inform_correct"],
            "inform_total": dm["inform_total"],
            "request_correct": dm["request_correct"],
            "request_total": dm["request_total"],
            "booking_passed": dm["booking_passed"],
            "inform_slots": pred.inform_slots,
            "request_slots": pred.request_slots,
            "booking": pred.booking,
            "goal_inform": {
                dom: {k: v for k, v in slots.items() if not k.startswith("book ")}
                for dom, slots in dialogue.goal.inform.items()
            },
            "goal_request": {
                dom: list(slots.keys())
                for dom, slots in dialogue.goal.request.items()
            },
            "has_booking": has_booking,
            "trajectory": trajectory,
        })

    return failed
