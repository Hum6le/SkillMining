"""AWM Memory for MultiWOZ — adapted from AWM/mind2web/memory.py.

Two memory types:
1. **Workflow text** (mirrors ``args.workflow_path``): LLM-induced patterns
   loaded from a .txt file, injected as a user message.
2. **Concrete exemplars** (mirrors ``args.memory_path/exemplars.json``):
   successful dialogue trajectories stored as JSON, retrieved by domain overlap.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path


class MemoryStore:
    """Stores and retrieves concrete successful exemplars.

    Mirrors ``get_exemplars()`` from AWM/mind2web/memory.py:
    - Exemplars are stored as ``[{dialogue_id, domains, goal, trajectory, ...}]``
    - Retrieval filters by domain overlap and returns top-k
    - Format as few-shot messages for prompt injection

    Attributes:
        exemplars: List of stored exemplar dicts.
        max_exemplars: Max number to retrieve (mirrors ``retrieve_top_k``).
    """

    def __init__(self, max_exemplars: int = 3):
        self.exemplars: list[dict] = []
        self.max_exemplars = max_exemplars

    # ── Storage ──────────────────────────────────────────────

    def add(self, dialogue, prediction, trajectory: str = ""):
        """Store one successful dialogue as an exemplar."""
        self.exemplars.append({
            "dialogue_id": dialogue.dialogue_id,
            "domains": list(dialogue.domains),
            "goal": dialogue.goal.description[:500],
            "trajectory": trajectory[:2000],
            "inform_slots": prediction.inform_slots,
            "request_slots": prediction.request_slots,
            "booking": prediction.booking,
        })

    # ── Retrieval (mirrors get_exemplars filtering logic) ─────

    def retrieve(self, domains: list[str], k: int | None = None) -> list[dict]:
        """Retrieve exemplars filtered by domain overlap.

        Mirrors the hierarchical filtering in AWM's ``get_exemplars()``:
        rank by number of shared domains, return top-k.
        """
        if not self.exemplars:
            return []
        k = k or self.max_exemplars
        domain_set = set(domains)
        scored = [(len(domain_set & set(e.get("domains", []))), e)
                  for e in self.exemplars]
        scored.sort(key=lambda x: -x[0])
        # At least one domain must overlap
        filtered = [e for s, e in scored if s > 0]
        if not filtered:
            filtered = [e for s, e in scored[:k]]
        return filtered[:k]

    def format_prompt(self, domains: list[str], k: int | None = None) -> str:
        """Format retrieved exemplars as few-shot prompt section."""
        exemplars = self.retrieve(domains, k)
        if not exemplars:
            return ""
        lines = ["## Past Successful Examples"]
        for i, ex in enumerate(exemplars, 1):
            lines.append(f"\n### Example {i}: {', '.join(ex['domains'])}")
            lines.append(f"Goal: {ex['goal'][:300]}")
            if ex.get("trajectory"):
                lines.append(f"Trajectory:\n{ex['trajectory'][:1000]}")
        return "\n".join(lines) + "\n"

    # ── I/O ──────────────────────────────────────────────────

    def load(self, path: str):
        if os.path.exists(path):
            self.exemplars = json.loads(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        Path(path).write_text(json.dumps(self.exemplars, indent=2, ensure_ascii=False), encoding="utf-8")

    def __len__(self) -> int:
        return len(self.exemplars)


class WorkflowStore:
    """Workflow text accumulator — mirrors AWM's workflow_path .txt file.

    The workflow file is a plain text file that gets overwritten/updated
    after each induction call.  At inference time, the full text is
    injected into the agent prompt (mirrors the ``workflow_text`` in
    ``get_exemplars()``).
    """

    def __init__(self, path: str | None = None):
        self._text = ""
        if path and os.path.exists(path):
            self._text = Path(path).read_text(encoding="utf-8")

    @property
    def text(self) -> str:
        return self._text.strip()

    def update(self, new_workflow: str):
        """Replace workflow with newly induced text (mirrors online_induction.py output)."""
        self._text = new_workflow.strip()

    def format_prompt(self) -> str:
        """Format workflow for prompt injection — mirrors the user message in get_exemplars."""
        if not self._text:
            return ""
        return (
            "## Induced Workflow Patterns\n"
            "The following patterns were extracted from past agent trajectories. "
            "Follow these strategies when applicable.\n\n"
            + self._text + "\n"
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        Path(path).write_text(self._text, encoding="utf-8")

    def load(self, path: str):
        if os.path.exists(path):
            self._text = Path(path).read_text(encoding="utf-8")

    def __bool__(self) -> bool:
        return bool(self._text.strip())

    def __len__(self) -> int:
        return len(self._text.splitlines()) if self._text else 0
