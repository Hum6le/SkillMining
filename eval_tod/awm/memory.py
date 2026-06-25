"""AWM Workflow Memory for MultiWOZ.

Core concept: accumulate LLM-induced workflow patterns (action heuristics)
from agent trajectories.  The workflow text grows over time and is injected
into the agent's system prompt as domain knowledge.

Unlike simple few-shot exemplar storage, AWM abstracts patterns from
both successes AND failures to produce general, reusable guidance.
"""

from __future__ import annotations

import os
from pathlib import Path


class WorkflowStore:
    """Accumulates LLM-induced workflow patterns across runs.

    The workflow is a plain text file.  Each call to ``induce()``
    appends new patterns.  At inference time, ``format_prompt()``
    injects the current workflow into the agent's system prompt.

    Attributes:
        text: The full workflow text.
        history: List of (trajectory_summary, induced_pattern) tuples
                 for debugging / rollback.
    """

    def __init__(self, path: str | None = None):
        self._lines: list[str] = []
        self.history: list[dict] = []
        if path and os.path.exists(path):
            self._lines = Path(path).read_text(encoding="utf-8").strip().split("\n")

    @property
    def text(self) -> str:
        return "\n".join(self._lines).strip()

    def update(self, pattern: str):
        """Replace the entire workflow with a new pattern block."""
        if pattern.strip():
            self._lines.append(pattern.strip())

    def format_prompt(self) -> str:
        """Format workflow text for prompt injection.

        Returns empty string if no workflow has been induced yet.
        """
        if not self._lines:
            return ""
        header = ("## Induced Workflow Patterns\n"
                  "The following patterns were extracted from past agent "
                  "trajectories. Use them to guide your decisions.\n\n")
        return header + "\n\n".join(self._lines) + "\n"

    def save(self, path: str):
        """Persist workflow to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        Path(path).write_text(self.text, encoding="utf-8")

    def load(self, path: str):
        """Load workflow from disk."""
        if os.path.exists(path):
            self._lines = Path(path).read_text(encoding="utf-8").strip().split("\n")

    def __len__(self) -> int:
        return len(self._lines)

    def __bool__(self) -> bool:
        return bool(self._lines) and any(l.strip() for l in self._lines)
