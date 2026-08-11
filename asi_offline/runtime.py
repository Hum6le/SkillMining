"""Label-hidden ABCD runtime for a frozen ASIoffline action library."""

from __future__ import annotations

from pathlib import Path

from awm import MemoryStore, WorkflowStore
from eval_tod.abcd.agent import ABCDAgent


_RUNTIME_POLICY = """## ASIoffline Runtime
The programmatic action library below was induced offline from fixed training
trajectories and is frozen. Infer relevance from the current dialogue only; no
scenario or subflow label is available. Use a function's action order and
argument roles as procedural guidance, but emit exactly one canonical primitive
ABCD action with its ordered current-dialogue slot values for each target turn.
Never emit a composite function name as the action and never copy an induction
example's values.
"""


def build_asi_workflow(skill_library: str) -> WorkflowStore:
    workflow = WorkflowStore()
    workflow.replace(_RUNTIME_POLICY + "\n\n" + skill_library.strip())
    return workflow


def load_asi_library(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def create_asi_offline_abcd_agent(
    skill_library: str,
    *,
    model: str = "deepseek-chat",
    api_key: str | None = None,
    base_url: str | None = None,
    delay: float = 0.3,
    response_logger=None,
) -> ABCDAgent:
    """Build the frozen-library agent used only for held-out ABCD evaluation."""
    return ABCDAgent(
        model=model,
        api_key=api_key,
        base_url=base_url,
        workflow=build_asi_workflow(skill_library),
        memory=MemoryStore(),
        delay=delay,
        expose_scenario_labels=False,
        response_logger=response_logger,
    )
