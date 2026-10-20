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


_ABCD_EXPEL_INSTRUCTION = """You are an expert at analyzing task-oriented dialogue agent trajectories for the ABCD benchmark.

The agent must solve the customer's task while producing the exact backend action
and ordered slot-value list required on every action-bearing turn, and a helpful
natural-language response. A response can sound reasonable and still fail if the
backend action or ordered slot values are wrong.

Analyze the successful and failed trajectories together in this same call:

1. SUCCESS cases are positive evidence. Preserve their reliable action sequence,
   action-to-slot alignment, slot ordering, and response timing.
2. FAILURE cases are negative evidence. Identify what should be avoided or repaired,
   but do not infer a rule from a failure alone.
3. Prefer rules grounded in observable dialogue context, user requests, prior turns,
   action transitions, and slot availability. Do not copy conversation-specific
   values, IDs, or private entities into a reusable rule.
4. State a rule generally enough to transfer to another ABCD conversation. Mention
   the relevant action and slot behavior when that is necessary for correctness.

Return only rule operations, with at most four operations total:
ADD: a new reusable rule.
AGREE 1: an existing rule that is confirmed by the evidence.
EDIT 1: a corrected replacement for existing rule 1.
REMOVE 1: an existing rule that is contradicted by the evidence.

Do not output an introduction, analysis, Markdown heading, JSON, or dialogue IDs."""


_ABCD_EXPEL_ONESHOT = """Example of the desired level of abstraction:

Successful pattern: after the user supplies a missing booking detail, the agent
uses the booking action with the complete ordered slot list before confirming.
Failed pattern: the agent gives a plausible response but omits or reorders a
required slot, so the backend action is incorrect.

ADD: When an action requires multiple slots, preserve the dataset-defined slot order and fill each slot from the current dialogue before generating the response."""


@dataclass
class _Rule:
    text: str
    strength: int = 2


RULE_UPDATE_PROTOCOL = 'snapshot_numbered_operations_clean_replacements_v2'


def _clean_rule(text: str, previous: str | None = None) -> str:
    """Strip edit-control wrappers, preserving actual procedural instructions."""
    text = text.strip()
    text = re.sub(r'^(?:rule|replacement)\s*:\s*', '', text, flags=re.I)
    text = re.sub(r'^(?:replace\s+)?rule\s+\d+\s+with\s*:?\s*', '', text, flags=re.I)
    text = text.strip()
    if text.startswith('`') and text.endswith('`'):
        text = text[1:-1].strip()
    if previous:
        old, sep, new = text.partition(' -> ')
        if sep and old.strip().rstrip('.') == previous.strip().rstrip('.'):
            text = new.strip()
        match = re.fullmatch(r'Replace ["\u201c](.+)["\u201d] with ["\u201c](.+)["\u201d]', text, re.I|re.S)
        if match and match.group(1).strip().rstrip('.') == previous.strip().rstrip('.'):
            text = match.group(2).strip()
    return text.rstrip('.')+'.' if text else ''


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
        # Numbers refer to the rule list shown to the inducer, before any
        # operation in this batch. Do not let ADD/EDIT retarget later updates.
        snapshot = list(self.rules)
        order = {'REMOVE':0, 'AGREE':1, 'EDIT':2, 'ADD':3}
        for operation, text in sorted(operations, key=lambda row: order.get(row[0].split()[0], 9)):
            kind, _, number = operation.partition(" ")
            target = None
            if number.isdigit():
                if 1 <= int(number) <= len(snapshot):
                    target = snapshot[int(number)-1]
            elif kind in {'REMOVE','AGREE'}:
                index = self._find(text)
                target = self.rules[index] if index is not None else None
            if kind == "REMOVE" and target is not None:
                target.strength -= 1
            elif kind == "AGREE" and target is not None:
                target.strength += 1
            elif kind == "EDIT" and target is not None:
                cleaned = _clean_rule(text, target.text)
                if cleaned:
                    target.text = cleaned
                    target.strength += 1
            elif kind == "ADD":
                cleaned = _clean_rule(text)
                if cleaned and not any(r.text.casefold() == cleaned.casefold() for r in self.rules):
                    self.rules.append(_Rule(text=cleaned))

        unique = {}
        for rule in self.rules:
            if rule.strength <= 0:
                continue
            key = rule.text.casefold()
            if key in unique:
                unique[key].strength = max(unique[key].strength, rule.strength)
            else:
                unique[key] = rule
        self.rules = list(unique.values())
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
        numbered_reference = bool(number) and kind in {'REMOVE','AGREE'}
        if text and (numbered_reference or not any(token in text.upper() for token in ("<RULE", "EXISTING RULE"))):
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
            _ABCD_EXPEL_INSTRUCTION
            + "\n\n"
            + _ABCD_EXPEL_ONESHOT
            + "\n\n"
            + "## Existing ExpeL Rules\n"
            + (self.rule_store.text or "(none)")
            + "\n\n"
            + "## Joint Success/Failure Evidence\n"
            + "The following cases are one induction batch. Compare them jointly; "
              "do not process each case in isolation.\n\n"
            + "\n\n".join(experiences[:20])
            + "\n\n## Required Synthesis\n"
              "Prefer a small number of high-confidence rules that explain the "
              "observable difference between successful and failed behavior."
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
