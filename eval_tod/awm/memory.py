"""AWM Memory Store for MultiWOZ.

Stores successful dialogue trajectories as exemplars.  At inference time,
retrieves the most relevant exemplars (by domain overlap) and formats them
as few-shot prompts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class MemoryStore:
    """Accumulates and retrieves successful dialogue exemplars.

    Each exemplar is a compact record of a dialogue that the agent
    successfully completed: ``{goal, domains, trajectory_summary}``.
    At retrieval time, exemplars with overlapping domains are preferred.

    Attributes:
        exemplars: List of stored exemplar dicts.
        max_exemplars: Maximum number of exemplars to retrieve per query.
    """

    def __init__(self, max_exemplars: int = 3):
        self.exemplars: list[dict] = []
        self.max_exemplars = max_exemplars

    def add(self, dialogue, prediction, trajectory: str = ""):
        """Store a successful dialogue as an exemplar.

        Args:
            dialogue: ``Dialogue`` object.
            prediction: ``Prediction`` object.
            trajectory: Optional trajectory log text (agent's reasoning trace).
        """
        self.exemplars.append({
            "dialogue_id": dialogue.dialogue_id,
            "domains": list(dialogue.domains),
            "goal": dialogue.goal.description[:500],
            "inform_slots": prediction.inform_slots,
            "request_slots": prediction.request_slots,
            "booking": prediction.booking,
            "trajectory": trajectory[:2000],
        })

    def retrieve(self, domains: list[str], k: int | None = None) -> list[dict]:
        """Retrieve exemplars most relevant to the given domains.

        Exemplars are ranked by domain overlap (Jaccard-like: number of
        shared domains).  The top-k are returned.

        Args:
            domains: Target dialogue's domain list.
            k: Number of exemplars to return (default: ``self.max_exemplars``).

        Returns:
            List of exemplar dicts, most relevant first.
        """
        if not self.exemplars:
            return []

        k = k or self.max_exemplars
        domain_set = set(domains)

        scored = []
        for ex in self.exemplars:
            ex_domains = set(ex.get("domains", []))
            overlap = len(domain_set & ex_domains)
            scored.append((overlap, ex))

        scored.sort(key=lambda x: -x[0])
        return [ex for _, ex in scored[:k] if _ > 0] or [ex for _, ex in scored[:k]]

    def format_prompt(self, domains: list[str], k: int | None = None) -> str:
        """Build a few-shot prompt section from retrieved exemplars.

        Args:
            domains: Target dialogue's domain list.
            k: Number of exemplars to include.

        Returns:
            Formatted string suitable for injection into a system/user prompt,
            or empty string if no exemplars are available.
        """
        exemplars = self.retrieve(domains, k)
        if not exemplars:
            return ""

        lines = ["## Past Successful Examples"]
        for i, ex in enumerate(exemplars, 1):
            lines.append(f"\n### Example {i}: {', '.join(ex['domains'])}")
            lines.append(f"Goal: {ex['goal'][:300]}")
            if ex.get("trajectory"):
                lines.append(f"Trajectory:\n{ex['trajectory'][:1000]}")
        lines.append("\n---\n")
        return "\n".join(lines)

    def load(self, path: str):
        """Load exemplars from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            self.exemplars = json.load(f)

    def save(self, path: str):
        """Save exemplars to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.exemplars, f, indent=2, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self.exemplars)


# ══════════════════════════════════════════════════════════════════
# Workflow accumulator
# ══════════════════════════════════════════════════════════════════

class WorkflowStore:
    """Accumulates workflow text (action patterns) across runs.

    The workflow is a plain text file that gets appended to as the LLM
    induces new patterns from successful trajectories.  It is injected
    into the agent's prompt as domain knowledge.

    This mirrors AWM's ``workflow_path`` concept.
    """

    def __init__(self, path: str | None = None):
        self._lines: list[str] = []
        if path and os.path.exists(path):
            self._lines = Path(path).read_text(encoding="utf-8").strip().split("\n")

    @property
    def text(self) -> str:
        return "\n".join(self._lines)

    def append(self, pattern: str):
        """Append a new workflow pattern."""
        self._lines.append(pattern)

    def format_prompt(self) -> str:
        """Format workflow text for prompt injection."""
        if not self._lines:
            return ""
        return "## Workflow Patterns\n" + "\n".join(self._lines) + "\n"

    def save(self, path: str):
        Path(path).write_text(self.text, encoding="utf-8")

    def __bool__(self) -> bool:
        return bool(self._lines)
