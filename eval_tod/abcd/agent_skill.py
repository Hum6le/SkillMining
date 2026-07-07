#!/usr/bin/env python3
"""Skill-Selecting Agent for ABCD — self-selects skills + generates actions + NL response.

Unlike the base ABCDAgent (which receives a pre-selected single workflow), this agent
receives ALL available skill cards and chooses which one to use based on the conversation.
It outputs structured predictions that support full evaluation: text metrics (ROUGE/BLEU/BERT)
via response_text, and AST/CDS via per-turn action predictions.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from eval_tod.evaluate import AbstractTodAgent
from eval_tod.schemas import Prediction
from eval_tod.abcd.schemas import ABCDPrediction, ABCDTurnPrediction
from awm.memory import MemoryStore, WorkflowStore


# ── Prompt templates ─────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a customer service agent for an online retail company.
You help customers with returns, order issues, account questions, shipping problems,
and general inquiries.

## How You Work
1. Read the conversation history and identify the customer's core need.
2. Choose the ONE skill from "Available Skills" that best matches.
3. Plan the action sequence you will follow (referencing the skill's Key Actions).
4. Generate a helpful, natural response (1-3 sentences).

## Scenario Context
{subflow_description}

## Customer Info
{customer_info}

## Order Info
{order_info}
"""

_TASK_PROMPT = """## Conversation So Far
{context}

## Instruction
First select a skill, then plan actions, then generate the response.

Output EXACTLY in this format:
```
SKILL: <skill_id>
ACTIONS:
- <action_name>
- <action_name>: <slot1>, <slot2>
RESPONSE:
<your natural language response here>
```

- SKILL: the `skill_id` from the Available Skills list (e.g. `recover_password`).
- ACTIONS: the sequence of system actions to take. Each line is either just an action name, or an action name with comma-separated slot values after `:`.
- RESPONSE: the final agent utterance to the customer."""


class SkillSelectingAgent(AbstractTodAgent):
    """Agent that self-selects skills from a menu of available skill cards.

    Attributes:
        skill_cards_prompt: Pre-formatted "Available Skills" text injected
            into the system prompt.  Set via ``set_skill_cards()``.
        workflow: Accumulated workflow patterns from training (optional).
        memory: Successful exemplars for few-shot prompting (optional).
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
        max_turns: int = 1,
        memory: MemoryStore | None = None,
        workflow: WorkflowStore | None = None,
        delay: float = 0.3,
        response_logger=None,
    ):
        from llm import resolve_config
        cfg = resolve_config(api_key=api_key, base_url=base_url, model=model)
        self.model = cfg["model"]
        self.api_key = cfg["api_key"]
        self.base_url = cfg["base_url"]
        self.max_turns = max_turns
        self.delay = delay
        self.memory = memory if memory is not None else MemoryStore()
        self.workflow = workflow if workflow is not None else WorkflowStore()
        self._response_logger = response_logger

        # Skill cards — set via set_skill_cards()
        self._skill_cards_prompt: str = ""
        self._skill_metadata: Dict[str, dict] = {}
        # Track selections for evaluation
        self.selection_log: List[Dict[str, str]] = []

    def set_skill_cards(self, skill_cards_prompt: str, metadata: Dict[str, dict] | None = None):
        """Set the skill cards to make available to the agent.

        Args:
            skill_cards_prompt: Formatted "Available Skills" text.
            metadata: Raw skill metadata for logging/analysis.
        """
        self._skill_cards_prompt = skill_cards_prompt
        self._skill_metadata = metadata or {}

    # ── AbstractTodAgent interface ──────────────────────────────

    def generate_predictions(
        self, dialogues: list[dict[str, Any]], verbose: bool = True,
    ) -> list[Prediction]:
        """Generate predictions for ABCD conversations.

        Returns plain Prediction objects with response_text filled
        (for text metrics compatibility).
        """
        predictions: list[Prediction] = []
        total = len(dialogues)

        for i, conv in enumerate(dialogues):
            convo_id = str(conv.get("convo_id", i))
            if verbose:
                flow = conv.get("scenario", {}).get("flow", "?")
                subflow = conv.get("scenario", {}).get("subflow", "?")
                print(f"  [{i+1}/{total}] convo={convo_id}  {flow}/{subflow}")

            result = self._predict_single(conv)
            predictions.append(result["prediction"])
            self.selection_log.append(result["selection_log"])

            if i < total - 1:
                time.sleep(self.delay)

        return predictions

    def generate_abcd_predictions(
        self, dialogues: list[dict[str, Any]], verbose: bool = True,
    ) -> list[ABCDPrediction]:
        """Generate ABCD-format predictions (with per-turn actions) for AST/CDS.

        Returns ABCDPrediction objects with per-turn action/slot predictions.
        """
        predictions: list[ABCDPrediction] = []
        total = len(dialogues)

        for i, conv in enumerate(dialogues):
            convo_id = str(conv.get("convo_id", i))
            if verbose:
                flow = conv.get("scenario", {}).get("flow", "?")
                subflow = conv.get("scenario", {}).get("subflow", "?")
                print(f"  [{i+1}/{total}] convo={convo_id}  {flow}/{subflow}")

            result = self._predict_single(conv)
            predictions.append(result["abcd_prediction"])
            self.selection_log.append(result["selection_log"])

            if i < total - 1:
                time.sleep(self.delay)

        return predictions

    def generate_all_predictions(
        self, dialogues: list[dict[str, Any]], verbose: bool = True,
    ) -> tuple[list[Prediction], list[ABCDPrediction]]:
        """Generate both prediction formats in one pass.

        Returns:
            (text_predictions, abcd_predictions) tuple.
        """
        text_preds: list[Prediction] = []
        abcd_preds: list[ABCDPrediction] = []
        total = len(dialogues)

        for i, conv in enumerate(dialogues):
            convo_id = str(conv.get("convo_id", i))
            if verbose:
                flow = conv.get("scenario", {}).get("flow", "?")
                subflow = conv.get("scenario", {}).get("subflow", "?")
                print(f"  [{i+1}/{total}] convo={convo_id}  {flow}/{subflow}")

            result = self._predict_single(conv)
            text_preds.append(result["prediction"])
            abcd_preds.append(result["abcd_prediction"])
            self.selection_log.append(result["selection_log"])

            if i < total - 1:
                time.sleep(self.delay)

        return text_preds, abcd_preds

    # ── Core prediction logic ────────────────────────────────────

    def _predict_single(self, conversation: dict[str, Any]) -> dict:
        """Generate prediction with skill selection + actions + response.

        Returns dict with keys: prediction, abcd_prediction, selection_log.
        """
        convo_id = str(conversation.get("convo_id", "?"))
        scenario = conversation.get("scenario", {})
        delexed = conversation.get("delexed", [])

        # Build system prompt
        system = self._build_system_prompt(scenario)

        # Build conversation history
        history_lines: list[str] = []
        for turn in delexed:
            spk = turn.get("speaker", "unknown")
            txt = turn.get("text", "").strip()
            if not txt:
                continue
            label = {"agent": "Agent", "customer": "Customer", "action": "System"}.get(spk, spk)
            history_lines.append(f"[{label}] {txt}")

        context = "\n".join(history_lines)

        # Build messages
        from llm import chat

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _TASK_PROMPT.format(context=context)},
        ]

        raw_output = ""
        try:
            raw_output = chat(
                messages,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.3,
                max_tokens=512,
                response_logger=self._response_logger,
            ).strip()
        except Exception as exc:
            print(f"    LLM error convo={convo_id}: {exc}")

        # Parse output
        parsed = _parse_structured_output(raw_output, convo_id, delexed)

        return {
            "prediction": Prediction(
                dialogue_id=f"abcd-{convo_id}",
                inform_slots={},
                request_slots={},
                booking={},
                response_text=parsed["response_text"],
            ),
            "abcd_prediction": ABCDPrediction(
                conversation_id=convo_id,
                turns=parsed["turns"],
            ),
            "selection_log": {
                "convo_id": convo_id,
                "selected_skill": parsed["selected_skill"],
                "ground_truth_subflow": str(scenario.get("subflow", "")),
                "raw_output": raw_output[:500],
            },
        }

    def _build_system_prompt(self, scenario: dict[str, Any]) -> str:
        """Build system prompt with skill cards + workflow + memory + scenario."""
        flow = scenario.get("flow", "unknown")
        subflow = scenario.get("subflow", "unknown")
        personal = scenario.get("personal", {})
        order = scenario.get("order", {})

        subflow_desc = f"Flow: {flow} / Subflow: {subflow}"

        cust_parts = []
        if personal.get("customer_name"):
            cust_parts.append(f"Name: {personal['customer_name']}")
        if personal.get("member_level"):
            cust_parts.append(f"Membership: {personal['member_level']}")
        customer_info = ", ".join(cust_parts) if cust_parts else "Not available"

        ord_parts = []
        if order.get("order_id"):
            ord_parts.append(f"Order ID: {order['order_id']}")
        if order.get("payment_method"):
            ord_parts.append(f"Payment: {order['payment_method']}")
        order_info = ", ".join(ord_parts) if ord_parts else "Not available"

        base = _SYSTEM_PROMPT.format(
            subflow_description=subflow_desc,
            customer_info=customer_info,
            order_info=order_info,
        )

        # Build prompt layers (top to bottom)
        layers: list[str] = []

        # 1. Workflow (from AWM training)
        wf = self.workflow.format_prompt()
        if wf:
            layers.append(wf)

        # 2. Skill cards (from hypergraph mining)
        if self._skill_cards_prompt:
            layers.append(self._skill_cards_prompt)

        # 3. Exemplars (from AWM memory)
        ex = self.memory.format_prompt([flow, subflow])
        if ex:
            layers.append(ex)

        # 4. Base system prompt
        layers.append(base)

        return "\n\n".join(layers)


# ── Output parsing ──────────────────────────────────────────────

def _parse_structured_output(
    raw: str, convo_id: str, delexed: list[dict],
) -> dict:
    """Parse the agent's structured output: SKILL + ACTIONS + RESPONSE.

    Returns:
        {selected_skill, response_text, turns: [ABCDTurnPrediction]}
    """
    selected_skill = ""
    response_text = ""
    action_lines: list[str] = []

    # Extract SKILL
    m = re.search(r"SKILL:\s*(\S+)", raw)
    if m:
        selected_skill = m.group(1).strip()

    # Extract ACTIONS block
    m = re.search(r"ACTIONS:\s*\n(.*?)(?:RESPONSE:|$)", raw, re.DOTALL)
    if m:
        actions_text = m.group(1).strip()
        for line in actions_text.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                action_lines.append(line[1:].strip())

    # Extract RESPONSE
    m = re.search(r"RESPONSE:\s*\n(.*?)$", raw, re.DOTALL)
    if m:
        response_text = m.group(1).strip()
        # Clean up quotes
        response_text = response_text.strip('"').strip("'")

    # If no structured output found, treat entire output as response
    if not response_text and not selected_skill:
        response_text = raw.strip()

    # Build ABCDTurnPrediction objects
    turns: list[ABCDTurnPrediction] = _actions_to_turn_predictions(
        action_lines, delexed,
    )

    return {
        "selected_skill": selected_skill,
        "response_text": response_text,
        "turns": turns,
    }


def _actions_to_turn_predictions(
    action_lines: list[str], delexed: list[dict],
) -> list[ABCDTurnPrediction]:
    """Map parsed action lines to ABCDTurnPrediction objects.

    Aligns predicted actions to the action turns in the dialogue.
    """
    # Identify action turn indices in the conversation
    action_turn_indices: list[int] = []
    for i, turn in enumerate(delexed):
        targets = turn.get("targets", [])
        if len(targets) >= 2 and targets[1] == "take_action":
            action_turn_indices.append(i)

    predictions: list[ABCDTurnPrediction] = []

    # Map predicted actions to action turns
    for pred_idx, action_line in enumerate(action_lines):
        if pred_idx >= len(action_turn_indices):
            break

        turn_idx = action_turn_indices[pred_idx]

        # Parse "action_name: slot1, slot2" or just "action_name"
        parts = action_line.split(":", 1)
        action_name = parts[0].strip()
        slot_values: list[str] = []
        if len(parts) == 2:
            slot_values = [s.strip() for s in parts[1].split(",") if s.strip()]

        predictions.append(ABCDTurnPrediction(
            turn_index=turn_idx,
            turn_type="action",
            predicted_action=action_name if action_name else None,
            predicted_slots=slot_values if slot_values else None,
        ))

    return predictions


# ── Selection accuracy ──────────────────────────────────────────

def compute_selection_accuracy(selection_log: List[Dict[str, str]]) -> dict:
    """Compute skill selection accuracy from agent logs.

    Args:
        selection_log: List of per-dialogue selection records with keys:
            ``selected_skill``, ``ground_truth_subflow``.

    Returns:
        {accuracy, correct, total, per_subflow: {total, correct}}
    """
    correct = 0
    total = 0
    per_subflow: Dict[str, dict] = {}

    for entry in selection_log:
        gt = entry.get("ground_truth_subflow", "")
        sel = entry.get("selected_skill", "")
        total += 1
        is_correct = (sel == gt)
        if is_correct:
            correct += 1

        if gt not in per_subflow:
            per_subflow[gt] = {"total": 0, "correct": 0}
        per_subflow[gt]["total"] += 1
        if is_correct:
            per_subflow[gt]["correct"] += 1

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "per_subflow": per_subflow,
    }
