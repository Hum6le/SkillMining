"""Episode feedback and persistent verbal memory for ABCD Reflexion."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eval_tod.abcd.agent import ABCDAgent, _build_abcd_turn_trajectory


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_-]{2,}", text.lower()))


def _query(conversation: dict[str, Any]) -> str:
    """Only customer language is a retrieval key; no scenario or gold labels."""
    return " ".join(
        str(turn.get("text", ""))
        for turn in conversation.get("delexed", [])
        if turn.get("speaker") == "customer"
    )[:1000]


def _redact_training_values(text: str, conversation: dict[str, Any]) -> str:
    """Keep raw customer-specific training values out of transferable memory."""
    values: set[str] = set()
    for turn in conversation.get("delexed", []):
        targets = turn.get("targets", [])
        if len(targets) > 3 and targets[1] == "take_action" and isinstance(targets[3], list):
            values.update(str(value).strip() for value in targets[3])
    scenario = conversation.get("scenario", {})
    for section_name in ("personal", "order"):
        section = scenario.get(section_name, {})
        if isinstance(section, dict):
            values.update(str(value).strip() for value in section.values()
                          if isinstance(value, (str, int)))
    for value in sorted((value for value in values if len(value) >= 2 or value.isdigit()),
                        key=len, reverse=True):
        text = re.sub(
            rf"(?<!\w){re.escape(value)}(?!\w)", "[customer_value]", text,
            flags=re.IGNORECASE,
        )
    return text


@dataclass(frozen=True)
class Reflection:
    conversation_id: str
    trial_index: int
    query: str
    text: str
    ast_score: float


@dataclass
class ReflectionStore:
    """Raw self-reflections, without ExpeL-style rule consolidation."""

    reflections: list[Reflection] = field(default_factory=list)

    @classmethod
    def from_trials(cls, trials: list[dict[str, Any]]) -> "ReflectionStore":
        return cls([
            Reflection(
                conversation_id=str(row["conversation_id"]),
                trial_index=int(row["trial_index"]),
                query=str(row.get("query", "")),
                text=str(row["reflection"]).strip(),
                ast_score=float(row.get("metrics", {}).get("ast_score", 0)),
            )
            for row in trials if str(row.get("reflection", "")).strip()
        ])

    @classmethod
    def load(cls, path: str | Path) -> "ReflectionStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Reflection(**row) for row in payload.get("reflections", [])])

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps({
            "schema_version": 1,
            "reflections": [asdict(item) for item in self.reflections],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)

    def select(self, context: str, *, conversation_id: str, same_conversation: bool,
               limit: int = 3) -> list[Reflection]:
        if limit <= 0:
            return []
        candidates = [item for item in self.reflections if
                      (item.conversation_id == conversation_id) == same_conversation]
        if same_conversation:
            return candidates[-limit:]
        query_tokens = _tokens(context)
        ranked = sorted(enumerate(candidates), key=lambda pair: (
            -len(query_tokens & _tokens(pair[1].query)), -pair[0],
        ))
        return [item for _, item in ranked[:limit]]


class ReflexionABCDAgent(ABCDAgent):
    """Shared ABCD actor with bounded reflection text in its system prompt."""

    def __init__(self, *args, reflection_store: ReflectionStore | None = None,
                 same_conversation: bool = False, reflection_limit: int = 3,
                 **kwargs):
        self.reflection_store = reflection_store or ReflectionStore()
        self.same_conversation = same_conversation
        self.reflection_limit = reflection_limit
        self._active_conversation_id = ""
        super().__init__(*args, **kwargs)

    def predict_all_turns(self, conversation: dict[str, Any], *args, **kwargs) -> list[dict]:
        self._active_conversation_id = str(conversation.get("convo_id", ""))
        try:
            return super().predict_all_turns(conversation, *args, **kwargs)
        finally:
            self._active_conversation_id = ""

    def _build_system_prompt(self, scenario: dict[str, Any], context: str = "",
                             candidate_actions: list[str] | None = None) -> str:
        base = super()._build_system_prompt(
            scenario, context=context, candidate_actions=candidate_actions,
        )
        selected = self.reflection_store.select(
            context, conversation_id=self._active_conversation_id,
            same_conversation=self.same_conversation, limit=self.reflection_limit,
        )
        if not selected:
            return base
        notes = "\n".join(f"{index}. {item.text[:800]}" for index, item in enumerate(selected, 1))
        return (
            "<reflexion_memory>\n"
            "Lessons from earlier training attempts. Apply them to the visible dialogue; "
            "never copy another customer's slot values.\n"
            f"{notes}\n</reflexion_memory>\n\n{base}"
        )


def build_reflection_feedback(conversation: dict[str, Any],
                              turn_results: list[dict[str, Any]],
                              metrics: dict[str, Any]) -> dict[str, Any]:
    """Give the train-only reflector exact action and ordered-slot feedback."""
    trajectory = _build_abcd_turn_trajectory(conversation, turn_results)
    errors = [{
        "turn_index": row["turn_index"],
        "dialogue_prefix": row["context"][-600:],
        "predicted_action": row["predicted_action"],
        "expected_action": row["gold_action"],
        "predicted_slots": row["predicted_slots"],
        "expected_slots": row["gold_slots"],
        "action_correct": row["action_correct"],
        "slot_correct": row["slot_correct"],
    } for row in trajectory if row["turn_type"] == "action" and not row["ast_correct"]]
    return {
        "ast_score": float(metrics.get("ast_score", 0)),
        "action_correct": int(metrics.get("action_correct", 0)),
        "action_total": int(metrics.get("action_total", 0)),
        "errors": errors[:4],
        "remaining_errors": max(0, len(errors) - 4),
    }


def generate_reflection(conversation: dict[str, Any], trial_index: int,
                        feedback: dict[str, Any], previous: list[Reflection],
                        *, model: str = "deepseek-chat", response_logger=None) -> str:
    """Amplify train feedback into one reusable verbal lesson."""
    from llm import chat

    prompt = (
        "You are the self-reflection component of a customer-service agent. "
        "A completed training attempt made backend action or ordered-slot errors. "
        "Write one concise, actionable lesson for the next attempt. Ground it "
        "in the correct action and ordered slots in the training feedback. "
        "Explain the mistake and correction, and avoid customer "
        "names, IDs, concrete slot values, and dataset labels. Do not invent "
        "unobserved environment outcomes. Return plain text only, at most 120 words.\n\n"
        f"Trial: {trial_index}\n"
        f"Customer request: {_query(conversation)[:600]}\n"
        f"Previous reflections: {json.dumps([item.text for item in previous[-3:]], ensure_ascii=False)}\n"
        f"Training feedback: {json.dumps(feedback, ensure_ascii=False)}"
    )
    reflection = chat(
        prompt, model=model, temperature=0.0, response_logger=response_logger,
        call_tag="reflexion_self_reflection",
    ).strip()
    if not reflection:
        raise RuntimeError("Reflexion returned an empty self-reflection")
    return _redact_training_values(reflection, conversation)[:1200]


def customer_query(conversation: dict[str, Any]) -> str:
    return _query(conversation)
