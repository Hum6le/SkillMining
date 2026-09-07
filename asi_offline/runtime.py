"""Label-hidden ABCD runtime for a frozen ASIoffline action library."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from awm import MemoryStore, WorkflowStore
from eval_tod.abcd.agent import ABCDAgent
from eval_tod.abcd.action_schema import canonicalize_prediction


_RUNTIME_POLICY = """## ASIoffline Runtime
The programmatic action library below was induced offline from fixed training
trajectories and is frozen. Infer relevance from the current dialogue only; no
scenario or subflow label is available. Use a function's action order and
argument roles as procedural guidance, but emit exactly one canonical primitive
ABCD action with its ordered current-dialogue slot values for each target turn.
The optimization and decision target is joint ABCD AST, not action accuracy
alone: the action name and the complete ordered slot-value list must be right
together. If the action is right but any slot is missing, extra, incorrect, or
out of order, the prediction is not AST-correct. Use the current dialogue to
bind every argument before emitting the action.
Never emit a composite function name as the action and never copy an induction
example's values.
For a multi-step induced function, use the dialogue history to determine which
primitive step has already been completed and emit only the next applicable
primitive action. Bind every ordered slot value from the current dialogue;
`slot_1`, `slot_2`, and similar names are parameter references, never literal
ABCD slot values. A function is guidance, not a mandatory route: do not force a
step when the current user utterance or dialogue context does not support it.
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


_BATCHED_ACTION_PROMPT = """<task>
You are evaluating an ASI programmatic action library on one ABCD dialogue.
The library contains reusable multi-step functions, but the evaluator needs
the canonical primitive backend action at each marked action turn.
</task>

<causal_checkpoints>
{checkpoints}
</causal_checkpoints>

<instruction>
For every marked action turn, execute or resume the applicable ASI function
and emit exactly one primitive action for that turn. Preserve the order of
function steps and bind every slot to a real value grounded in the dialogue
prefix available at that checkpoint. Do not emit a composite function name,
slot names, placeholders, or values copied from an example.

Return ONLY valid JSON in this form:
{{
  "predictions": [
    {{"turn_index": 3, "action": "canonical-action", "slots": ["value"]}}
  ],
  "responses": [
    {{"turn_index": 2, "response": "brief agent response"}}
  ]
}}

Use the exact turn_index shown in each checkpoint. Include an action entry for
every marked action turn, even when the action is empty. Include response
entries only for marked agent utterance turns. Never add entries for other
turns. A missing action must be represented as action="" and slots=[].
</instruction>"""


def _original_turn(conversation: dict[str, Any], index: int, turn: dict[str, Any]) -> str:
    """Return the original text while tolerating the two ABCD storage forms."""
    original = conversation.get("original", [])
    if isinstance(original, list) and index < len(original):
        row = original[index]
        if isinstance(row, dict):
            return str(row.get("text", "") or "").strip()
        return str(row or "").strip()
    return str(turn.get("text", "") or "").strip()


def _is_action_turn(turn: dict[str, Any]) -> bool:
    targets = turn.get("targets", [])
    return len(targets) >= 2 and targets[1] == "take_action"


def _build_causal_checkpoints(conversation: dict[str, Any]) -> tuple[str, list[int], list[int]]:
    """Render all prediction targets with a prefix for causal step recovery."""
    delexed = list(conversation.get("delexed", []))
    action_indices = [i for i, turn in enumerate(delexed) if _is_action_turn(turn)]
    response_indices = [
        i for i, turn in enumerate(delexed)
        if turn.get("speaker") == "agent" and _original_turn(conversation, i, turn)
    ]
    target_indices = sorted(set(action_indices) | set(response_indices))
    sections: list[str] = []
    for index in target_indices:
        prefix: list[str] = []
        for previous_index in range(index):
            previous = delexed[previous_index]
            text = _original_turn(conversation, previous_index, previous)
            if not text:
                continue
            speaker = str(previous.get("speaker", "unknown"))
            label = {"agent": "Agent", "customer": "Customer", "action": "System"}.get(speaker, speaker)
            prefix.append(f"[{label}] {text}")
        turn = delexed[index]
        target_kind = "ACTION" if _is_action_turn(turn) else "RESPONSE"
        current = _original_turn(conversation, index, turn)
        sections.append(
            f"### turn_index={index} target={target_kind}\n"
            f"Dialogue prefix:\n{chr(10).join(prefix) or '(empty)'}\n"
            f"Current turn text: {current or '(backend action turn)'}"
        )
    return "\n\n".join(sections), action_indices, response_indices


def _extract_json_object(raw: str) -> dict[str, Any]:
    payload = str(raw or "").strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.I | re.S).strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def generate_batched_conversation_predictions(
    agent: ABCDAgent,
    conversation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Predict one complete conversation in one LLM call.

    The returned rows intentionally use the same flat turn-result contract as
    ``ABCDAgent.generate_all_turn_predictions``. This makes the existing AST
    mapper/evaluator reusable while moving ASI's function-step expansion to a
    single conversation-level request.
    """
    from llm import chat

    checkpoints, action_indices, response_indices = _build_causal_checkpoints(conversation)
    if not checkpoints:
        return []
    workflow = agent.workflow.format_prompt()
    system = (
        _RUNTIME_POLICY
        + "\n\n"
        + workflow
        + "\n\nThe dialogue prefix attached to each checkpoint is authoritative."
    )
    raw = chat(
        [{"role": "system", "content": system}, {
            "role": "user",
            "content": _BATCHED_ACTION_PROMPT.format(checkpoints=checkpoints),
        }],
        model=agent.model,
        api_key=agent.api_key,
        base_url=agent.base_url,
        temperature=0.7,
        response_logger=getattr(agent, "_response_logger", None),
    ) or ""
    payload = _extract_json_object(raw)
    by_index: dict[int, dict[str, Any]] = {}
    for item in payload.get("predictions", []) if isinstance(payload.get("predictions", []), list) else []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("turn_index"))
        except (TypeError, ValueError):
            continue
        action, suffix_slots = canonicalize_prediction(
            item.get("action", ""), item.get("slots", []), agent.action_schema
        )[:2]
        by_index[index] = {
            "predicted_action": action,
            "predicted_slots": list(suffix_slots),
        }
    responses = {
        int(item.get("turn_index")): str(item.get("response", "") or "")
        for item in payload.get("responses", [])
        if isinstance(item, dict) and str(item.get("turn_index", "")).lstrip("-").isdigit()
    }

    convo_id = str(conversation.get("convo_id", "?"))
    rows: list[dict[str, Any]] = []
    for index in sorted(set(action_indices) | set(response_indices)):
        row = {
            "convo_id": convo_id,
            "turn_index": index,
            "target_type": "action" if index in action_indices else "utterance",
            "prediction": responses.get(index, ""),
            "predicted_action": None,
            "predicted_slots": [],
            "batched_prediction": True,
        }
        if index in action_indices:
            prediction = by_index.get(index, {})
            row["predicted_action"] = prediction.get("predicted_action")
            row["predicted_slots"] = prediction.get("predicted_slots", [])
        rows.append(row)
    return rows
