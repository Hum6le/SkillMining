"""Reference loading utilities for prompt-time retrieval.

There are two local reference formats in this repo:

* ``skill_mining`` writes a sibling ``reference.md`` next to ``skill.md``.
* Trace2Skill skill folders expose references only as ``references/*.md``.

AWM exemplars are intentionally not handled here; they are JSON memory records
retrieved and formatted by ``awm.memory.MemoryStore``.
"""

from __future__ import annotations

from pathlib import Path


def _read_markdown_files(candidates: list[Path]) -> str:
    parts: list[str] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists() or not candidate.is_file():
            continue
        seen.add(candidate)
        text = candidate.read_text(encoding="utf-8").strip()
        if text:
            parts.append(f"# Source: {candidate.name}\n\n{text}")
    return "\n\n---\n\n".join(parts)


def load_skill_mining_reference(root: str | Path | None) -> str:
    """Load ``skill_mining`` sibling ``reference.md`` files.

    This matches ``skill_mining/skill_writer.py``, which emits
    ``{subflow}/skill.md`` and ``{subflow}/reference.md``.
    """
    if not root:
        return ""

    path = Path(root)
    candidates: list[Path] = []

    if path.is_file():
        if path.name.lower() == "skill.md":
            candidates.append(path.parent / "reference.md")
        elif path.suffix.lower() == ".md":
            candidates.append(path)
    elif path.is_dir():
        candidates.append(path / "reference.md")
        for skill_file in sorted(path.glob("*/SKILL.md")):
            candidates.append(skill_file.parent / "reference.md")

    return _read_markdown_files(candidates)


def load_trace2skill_references(root: str | Path | None) -> str:
    """Load official Trace2Skill ``references/*.md`` files.

    This mirrors ``Trace2Skill/skill_evolver/skill_evolving_agent.py``:
    the evolver reads ``SKILL.md`` plus files under ``references/`` only.
    It deliberately ignores sibling ``reference.md`` files.
    """
    if not root:
        return ""

    path = Path(root)
    candidates: list[Path] = []

    if path.is_file():
        candidates.extend(sorted((path.parent / "references").glob("*.md")))
    elif path.is_dir():
        candidates.extend(sorted((path / "references").glob("*.md")))
        for skill_file in sorted(path.glob("*/SKILL.md")):
            candidates.extend(sorted((skill_file.parent / "references").glob("*.md")))

    return _read_markdown_files(candidates)


def load_markdown_references(root: str | Path | None) -> str:
    """Backward-compatible alias for the ``skill_mining`` reference format."""
    return load_skill_mining_reference(root)
