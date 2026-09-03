"""ExpeL-style experiential learning for ABCD.

The official ExpeL repository learns natural-language rules from successful
and failed trajectories.  ABCD has no interactive environment success flag,
so this adapter defines a successful experience as a dialogue with joint AST
equal to one and keeps the shared turn-level runner as the environment.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eval_tod.abcd.agent import ABCDAgent, _build_abcd_turn_trajectory


@dataclass
class _Rule:
    text: str
    strength: int = 2


@dataclass
class ExpeLRuleStore:
    """Persistent rule pool using ExpeL's operation semantics."""

    max_rules: int = 20
    rules: list[_Rule] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(
            f"{i}. {rule.text}" for i, rule in enumerate(self.rules, start=1)
        )

    def _find(self, text: str) -> int | None:
        needle = text.strip().lower()
        for i, rule in enumerate(self.rules):
            if rule.text.lower() in needle or needle in rule.text.lower():
                return i
        return None

    def apply(self, operations: list[tuple[str, str]]) -> None:
        # Match the official implementation: removals first, then agreement,
        # edits, and additions, with weak rules pruned afterwards.
        for operation, text in operations:
            kind, _, number = operation.partition(" ")
            index = self._find(text)
            if kind == "REMOVE" and index is not None:
                self.rules[index].strength -= 1
            elif kind == "AGREE" and index is not None:
                self.rules[index].strength += 1
            elif kind == "EDIT" and number.isdigit() and 1 <= int(number) <= len(self.rules):
                self.rules[int(number) - 1].text = text.strip()
                self.rules[int(number) - 1].strength += 1
            elif kind == "ADD" and self._find(text) is None and text.strip():
                self.rules.append(_Rule(text=text.strip().rstrip("." ) + "."))

        self.rules = [rule for rule in self.rules if rule.strength > 0]
        self.rules.sort(key=lambda rule: rule.strength, reverse=True)
        self.rules = self.rules[: self.max_rules]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"max_rules": self.max_rules, "rules": [asdict(r) for r in self.rules]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExpeLRuleStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(max_rules=int(payload.get("max_rules", 20)))
        store.rules = [_Rule(**item) for item in payload.get("rules", [])]
        return store


def _parse_operations(raw: str) -> list[tuple[str, str]]:
    """Parse the official ExpeL operation format defensively."""
    pattern = re.compile(r"\b(REMOVE|EDIT|ADD|AGREE)(?:\s+(\d+))?\s*:\s*(.+)", re.I)
    operations = []
    for match in pattern.finditer(raw or ""):
        kind = match.group(1).upper()
        number = match.group(2) or ""
        text = match.group(3).strip().splitlines()[0].strip()
        if text and not any(token in text.upper() for token in ("<RULE", "EXISTING RULE")):
            operations.append((f"{kind} {number}".strip(), text))
    return operations[:4]


class ExpeLABCDAgent(ABCDAgent):
    """ABCD agent with ExpeL rules injected into the system prompt."""

    def __init__(self, *args, rule_store: ExpeLRuleStore | None = None, **kwargs):
        self.rule_store = rule_store or ExpeLRuleStore()
        super().__init__(*args, **kwargs)

    def _build_system_prompt(
        self,
        scenario: dict[str, Any],
        context: str = "",
        candidate_actions: list[str] | None = None,
    ) -> str:
        base = super()._build_system_prompt(
            scenario,
            context=context,
            candidate_actions=candidate_actions,
        )
        if not self.rule_store.rules:
            return base
        return (
            "## ExpeL Insights\n"
            "These are general lessons extracted from previous ABCD experiences. "
            "Apply them to the current dialogue, but never copy another dialogue's private slot values.\n"
            f"{self.rule_store.text}\n\n{base}"
        )

    def set_rule_store(self, rule_store: ExpeLRuleStore) -> None:
        self.rule_store = rule_store

    def induce_rules(
        self,
        conversations: list[dict[str, Any]],
        turn_results: list[dict[str, Any]],
        metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract and apply rules from complete success/failure trajectories."""
        from llm import chat

        metric_by_id = {
            str(conv.get("convo_id", "?")): metric
            for conv, metric in zip(conversations, metrics)
        }
        experiences = []
        for conv in conversations:
            cid = str(conv.get("convo_id", "?"))
            rows = [r for r in turn_results if str(r.get("convo_id")) == cid]
            metric = metric_by_id[cid]
            trajectory = _build_abcd_turn_trajectory(conv, rows)
            label = "SUCCESS" if metric.get("action_total", 0) and metric.get("ast_score", 0) >= 1.0 else "FAILURE"
            # ExpeL receives episode-level outcome feedback. Per-turn
            # correctness labels and the dense AST score would be extra
            # supervision beyond the original success/failure protocol.
            for turn in trajectory:
                for key in ("action_correct", "slot_correct", "ast_correct"):
                    turn.pop(key, None)
            experiences.append(
                f"### {label} convo={cid} {conv.get('scenario', {}).get('flow', '?')}/{conv.get('scenario', {}).get('subflow', '?')}\n"
                f"{json.dumps(trajectory, ensure_ascii=False, indent=2)}"
            )

        if not experiences:
            return {"raw_output": "", "operations": [], "rules": self.text}
        prompt = (
            "You are the insight extraction stage of ExpeL, adapted to ABCD customer-service dialogues.\n"
            "Compare the complete successful and failed trajectories below. Infer concise, general rules about "
            "state transitions, action selection, slot ordering, and response timing. Do not mention dialogue IDs "
            "or private values. Return at most four lines using exactly:\n"
            "ADD: rule.\nAGREE 1: existing rule.\nEDIT 1: rewritten rule.\nREMOVE 1: existing rule.\n\n"
            f"Existing rules:\n{self.rule_store.text or '(none)'}\n\n"
            + "\n\n".join(experiences[:20])
        )
        raw = chat(
            prompt,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.0,
            response_logger=self._response_logger,
            call_tag="expel_rule_induction",
        ).strip()
        operations = _parse_operations(raw)
        self.rule_store.apply(operations)
        return {"raw_output": raw, "operations": operations, "rules": self.rule_store.text}
