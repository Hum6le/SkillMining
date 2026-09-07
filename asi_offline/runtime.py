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
For every marked action turn, choose or resume the applicable ASI function and
return its 1-based step_index. The parser will recover the primitive action
name from the function catalog, so do not rewrite the action name. Preserve
function-step order and bind every slot to a real value grounded in the
dialogue prefix available at that checkpoint. Do not emit slot names,
placeholders, or values copied from an example.

Return ONLY valid JSON in this form:
{{
  "predictions": [
    {{"turn_index": 3, "function": "induced_function", "step_index": 1, "slots": ["value"]}}
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


def _original_turn(
    conversation: dict[str, Any], index: int, turn: dict[str, Any]
) -> tuple[str, str]:
    """Return raw speaker/text while tolerating ABCD storage forms."""
    original = conversation.get("original", [])
    if isinstance(original, list) and index < len(original):
        row = original[index]
        if isinstance(row, dict):
            return (
                str(row.get("speaker", turn.get("speaker", "unknown"))),
                str(row.get("text", "") or "").strip(),
            )
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            return str(row[0]), str(row[1] or "").strip()
    return (
        str(turn.get("speaker", "unknown")),
        str(turn.get("text", "") or "").strip(),
    )


def _is_action_turn(turn: dict[str, Any]) -> bool:
    targets = turn.get("targets", [])
    return len(targets) >= 2 and targets[1] == "take_action"


def _build_causal_checkpoints(conversation: dict[str, Any]) -> tuple[str, list[int], list[int]]:
    """Render all prediction targets with a prefix for causal step recovery."""
    delexed = list(conversation.get("delexed", []))
    action_indices = [i for i, turn in enumerate(delexed) if _is_action_turn(turn)]
    response_indices = [
        i for i, turn in enumerate(delexed)
        if _original_turn(conversation, i, turn)[0] == "agent"
        and _original_turn(conversation, i, turn)[1]
    ]
    target_indices = sorted(set(action_indices) | set(response_indices))
    sections: list[str] = []
    for index in target_indices:
        prefix: list[str] = []
        for previous_index in range(index):
            previous = delexed[previous_index]
            speaker, text = _original_turn(conversation, previous_index, previous)
            if not text:
                continue
            label = {"agent": "Agent", "customer": "Customer", "action": "System"}.get(speaker, speaker)
            prefix.append(f"[{label}] {text}")
        turn = delexed[index]
        target_kind = "ACTION" if _is_action_turn(turn) else "RESPONSE"
        sections.append(
            f"### turn_index={index} target={target_kind}\n"
            f"Dialogue prefix:\n{chr(10).join(prefix) or '(empty)'}\n"
            "Current target text is hidden; predict from the prefix only."
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


def parse_asi_library(skill_library: str) -> dict[str, list[dict[str, Any]]]:
    """Parse the frozen ASI markdown into a deterministic function catalog."""
    functions: dict[str, list[dict[str, Any]]] = {}
    current: str | None = None
    heading = re.compile(r"^## Induced Action:\s*([a-z][a-z0-9_]*)\(", re.I)
    step = re.compile(
        r"^\s*(\d+)\.\s*Primitive action:\s*([^;]+);\s*"
        r"ordered slot parameters:\s*\[(.*?)\]\s*$", re.I
    )
    for line in str(skill_library or "").splitlines():
        match = heading.match(line)
        if match:
            current = match.group(1).strip()
            functions.setdefault(current, [])
            continue
        if current is None:
            continue
        match = step.match(line)
        if not match:
            continue
        functions[current].append({
            "step_index": int(match.group(1)),
            "action": match.group(2).strip(),
            "arguments": [
                value.strip() for value in match.group(3).split(",") if value.strip()
            ],
        })
    return {name: steps for name, steps in functions.items() if steps}


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
    function_catalog = parse_asi_library(workflow)
    catalog_lines = ["## Parsed ASI function catalog"]
    for name, steps in function_catalog.items():
        rendered_steps = ", ".join(
            f"{item['step_index']}:{item['action']}({','.join(item['arguments']) or 'none'})"
            for item in steps
        )
        catalog_lines.append(f"- {name}: {rendered_steps}")
    system = (
        workflow
        + "\n\n"
        + "\n".join(catalog_lines)
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
        call_tag="asi_batched_conversation_eval",
    ) or ""
    payload = _extract_json_object(raw)
    by_index: dict[int, dict[str, Any]] = {}
    expected_indices = set(action_indices)
    unexpected_indices: list[int] = []
    invalid_items = 0
    unresolved_items = 0
    for item in payload.get("predictions", []) if isinstance(payload.get("predictions", []), list) else []:
        if not isinstance(item, dict):
            invalid_items += 1
            continue
        try:
            index = int(item.get("turn_index"))
        except (TypeError, ValueError):
            invalid_items += 1
            continue
        if index not in expected_indices:
            unexpected_indices.append(index)
            continue
        function_name = str(item.get("function", "")).strip()
        action_value = item.get("action", "")
        if function_name:
            try:
                step_index = int(item.get("step_index"))
            except (TypeError, ValueError):
                unresolved_items += 1
                continue
            selected_step = next(
                (
                    step for step in function_catalog.get(function_name, [])
                    if step["step_index"] == step_index
                ),
                None,
            )
            if selected_step is None:
                unresolved_items += 1
                continue
            action_value = selected_step["action"]
        action, parsed_slots, validation = canonicalize_prediction(
            action_value, item.get("slots", []), agent.action_schema
        )
        if not validation.get("valid", True):
            unresolved_items += 1
        by_index[index] = {
            "predicted_action": action,
            "predicted_slots": list(parsed_slots),
            "function": function_name,
            "step_index": item.get("step_index"),
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
    missing_indices = sorted(expected_indices - set(by_index))
    for row in rows:
        row["batched_diagnostics"] = {
            "missing_action_turns": missing_indices,
            "unexpected_action_turns": sorted(set(unexpected_indices)),
            "invalid_prediction_items": invalid_items,
            "unresolved_function_steps": unresolved_items,
            "request_granularity": "one_request_per_conversation",
            "expected_action_turns": len(action_indices),
            "returned_action_turns": len(by_index),
        }
    return rows
