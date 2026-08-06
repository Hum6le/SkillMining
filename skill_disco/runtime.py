"""Test-time ABCD adapter for a rendered offline SKILL-DISCO library."""

from __future__ import annotations

from pathlib import Path

from awm import MemoryStore, WorkflowStore
from eval_tod.abcd.agent import ABCDAgent


_RUNTIME_POLICY = """## Procedural Skill Runtime
The following skills were distilled offline from successful training trajectories.
Infer the relevant skill from the current dialogue only; no scenario label is available.
Use a skill only when its declared preconditions are supported. Bind every parameter from the
current dialogue, preserve backend action order, and fall back to ordinary reasoning when no skill fits.
The library is procedural guidance, not a source of customer-specific values.
"""


def build_skill_disco_workflow(skill_library: str) -> WorkflowStore:
    """Wrap rendered pseudocode in the existing workflow injection interface."""
    workflow = WorkflowStore()
    workflow.replace(_RUNTIME_POLICY + "\n\n" + skill_library.strip())
    return workflow


def load_skill_library(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def create_skill_disco_abcd_agent(
    skill_library: str,
    *,
    model: str = "deepseek-chat",
    api_key: str | None = None,
    base_url: str | None = None,
    delay: float = 0.3,
    response_logger=None,
) -> ABCDAgent:
    """Create a label-hidden ABCD agent augmented only by generated skills."""
    return ABCDAgent(
        model=model,
        api_key=api_key,
        base_url=base_url,
        workflow=build_skill_disco_workflow(skill_library),
        memory=MemoryStore(),
        delay=delay,
        expose_scenario_labels=False,
        response_logger=response_logger,
    )
