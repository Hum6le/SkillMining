"""Skill-preloaded ToD agent.

Wraps ``ToolBasedTodAgent`` with skill discovery and injection.  Mirrors the
``CLISkillPreloadedAgent`` pattern from Trace2Skill: reads SKILL.md files
from a skills directory and prepends them to the agent's system prompt.

Usage::

    from eval_tod.kb import MultiWOZKB
    from eval_tod.agent_skill import SkillPreloadedAgent

    kb = MultiWOZKB("data/eval/multiwoz21/data/data")
    agent = SkillPreloadedAgent(kb=kb, skills_dir="eval_tod/skills")
    predictions = agent.generate_predictions(dialogues)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .agent_tool import ToolBasedTodAgent
from .kb import MultiWOZKB
from .schemas import Dialogue, Prediction


# ── Skill discovery ──────────────────────────────────────────────

def discover_skills(skills_dir: str) -> list[dict[str, str]]:
    """Find all SKILL.md files under ``skills_dir``.

    Returns:
        List of ``{name, description, file_path, content}`` dicts.
    """
    skills: list[dict[str, str]] = []
    if not os.path.isdir(skills_dir):
        return skills

    for entry in sorted(os.listdir(skills_dir)):
        skill_dir = os.path.join(skills_dir, entry)
        skill_file = os.path.join(skill_dir, "SKILL.md")
        if not (os.path.isdir(skill_dir) and os.path.exists(skill_file)):
            continue

        with open(skill_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract YAML frontmatter
        name = entry
        description = ""
        fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if fm_match:
            fm = fm_match.group(1)
            nm = re.search(r'^name:\s*["\']?([^"\'\n]+)', fm, re.MULTILINE)
            ds = re.search(r'^description:\s*["\']?([^"\'\n]+)', fm, re.MULTILINE)
            if nm:
                name = nm.group(1).strip()
            if ds:
                description = ds.group(1).strip()
            # Strip frontmatter from content
            content = content[fm_match.end():].lstrip("\n")

        skills.append({
            "name": name,
            "description": description,
            "file_path": skill_file,
            "content": content,
        })

    return skills


def render_skill_section(skills: list[dict[str, str]], skills_dir: str) -> str:
    """Render loaded skills into a section for the system prompt."""
    if not skills:
        return ""

    lines = [
        "## Loaded Skills",
        "",
        "The following domain-specific guidance is pre-loaded. Follow these",
        "instructions when they apply to your task. The skill takes precedence",
        "over your general knowledge when there is a conflict.",
        "",
    ]

    for skill in skills:
        lines.append(f"### Skill: {skill['name']}")
        if skill["description"]:
            lines.append(f"*{skill['description']}*")
        lines.append("")
        lines.append(skill["content"])
        lines.append("")

    lines.extend([
        "---",
        "",
        "**Skill Authority**: When a loaded skill above has guidance for your",
        "operation, its instructions take precedence over general knowledge.",
        "",
    ])

    return "\n".join(lines)


# ── Agent ─────────────────────────────────────────────────────────

class SkillPreloadedAgent(ToolBasedTodAgent):
    """ToolBasedTodAgent with skill content pre-loaded in the system prompt.

    Discovers SKILL.md files from ``skills_dir`` at init time and prepends
    them to the agent's system prompt.  The ReAct loop and KB tool work
    exactly as in the base agent.

    Attributes:
        skills: List of discovered skill dicts.
        skills_dir: Path to the skills directory.
    """

    def __init__(
        self,
        kb: MultiWOZKB,
        skills_dir: str = "eval_tod/skills",
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
        max_turns: int = 8,
        delay: float = 0.3,
        ontology_path: str | None = None,
        log_dir: str | None = None,
        response_logger = None,
    ):
        self.skills_dir = os.path.abspath(skills_dir)
        self.skills = discover_skills(self.skills_dir)

        if not self.skills:
            print(f"Warning: No skills discovered in {self.skills_dir}")

        skill_section = render_skill_section(self.skills, self.skills_dir)

        super().__init__(
            kb=kb,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_turns=max_turns,
            delay=delay,
            ontology_path=ontology_path,
            log_dir=log_dir,
            extra_system_prompt=skill_section,
            response_logger=response_logger,
        )

    def reload_skills(self) -> None:
        """Re-discover skills from disk (useful after evolution)."""
        self.skills = discover_skills(self.skills_dir)
        skill_section = render_skill_section(self.skills, self.skills_dir)
        self.extra_system_prompt = skill_section
        print(f"Reloaded {len(self.skills)} skill(s) from {self.skills_dir}")
