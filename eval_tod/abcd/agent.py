"""Generative ABCD agent — end-to-end: context → natural language response.

No utterance candidate pool, no action prediction.  The agent reads
dialogue history and directly generates the next agent utterance.

Integrates with the AWM pipeline: workflow patterns and successful
exemplars are injected into the system prompt.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval_tod.evaluate import AbstractTodAgent
from eval_tod.schemas import Prediction
from awm.memory import MemoryStore, WorkflowStore


# ── Prompt templates ──────────────────────────────────────────

_SYSTEM_PROMPT = """You are a customer service agent for an online retail company.
You help customers with returns, order issues, account questions, shipping
problems, and general inquiries.

## How You Work
Read the conversation history below and generate the NEXT agent response.
Your reply should be:
- Natural and conversational (like a real human agent)
- Helpful and informative
- Consistent with company policies mentioned in the workflow guidance
- Short and to the point (1-3 sentences)

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
Generate the next agent response.  Reply with ONLY the response text, nothing else."""


# ── Agent ──────────────────────────────────────────────────────

class ABCDAgent(AbstractTodAgent):
    """Generative agent for ABCD — direct response generation.

    Architecture:
    - Reads conversation history (all turns up to current point)
    - Injects workflow patterns + exemplars (AWM)
    - Generates natural language response via LLM
    - No utterance candidates, no action prediction, no KB queries
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

    # ── AbstractTodAgent interface ─────────────────────────

    def generate_predictions(
        self, dialogues: list[dict[str, Any]], verbose: bool = True,
    ) -> list[Prediction]:
        """Generate responses for a list of ABCD conversations.

        Args:
            dialogues: ABCD conversation dicts (from load_abcd_data).

        Returns:
            List of Prediction objects with response_text filled.
        """
        predictions: list[Prediction] = []
        total = len(dialogues)

        for i, conv in enumerate(dialogues):
            convo_id = str(conv.get("convo_id", i))
            if verbose:
                flow = conv.get("scenario", {}).get("flow", "?")
                subflow = conv.get("scenario", {}).get("subflow", "?")
                print(f"  [{i+1}/{total}] convo={convo_id}  {flow}/{subflow}")

            pred = self._predict_single(conv)
            predictions.append(pred)

            if i < total - 1:
                time.sleep(self.delay)

        return predictions

    def _predict_single(self, conversation: dict[str, Any]) -> Prediction:
        """Generate response for the last agent turn in a conversation."""
        convo_id = str(conversation.get("convo_id", "?"))
        scenario = conversation.get("scenario", {})
        delexed = conversation.get("delexed", [])

        # Build system prompt
        system = self._build_system_prompt(scenario)

        # Build conversation history (all turns up to the last)
        history_lines: list[str] = []
        last_agent_idx = -1
        for i, turn in enumerate(delexed):
            spk = turn.get("speaker", "unknown")
            txt = turn.get("text", "").strip()
            if not txt:
                continue
            label = {"agent": "Agent", "customer": "Customer", "action": "System"}.get(spk, spk)
            history_lines.append(f"[{label}] {txt}")
            if spk == "agent":
                last_agent_idx = i

        context = "\n".join(history_lines[:-1]) if len(history_lines) > 1 else history_lines[0] if history_lines else ""

        # Build messages
        from llm import chat

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _TASK_PROMPT.format(context=context)},
        ]

        response_text = ""
        try:
            response_text = chat(
                messages,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.7,
                max_tokens=256,
                response_logger=self._response_logger,
            ).strip()
            # Clean up quotes if the model wraps in them
            response_text = response_text.strip('"').strip("'")
        except Exception as exc:
            print(f"    LLM error convo={convo_id}: {exc}")

        return Prediction(
            dialogue_id=f"abcd-{convo_id}",
            inform_slots={},
            request_slots={},
            booking={},
            response_text=response_text,
        )

    def _build_system_prompt(self, scenario: dict[str, Any]) -> str:
        """Build system prompt with scenario context + workflow + memory."""
        flow = scenario.get("flow", "unknown")
        subflow = scenario.get("subflow", "unknown")
        personal = scenario.get("personal", {})
        order = scenario.get("order", {})

        # Scenario description
        subflow_desc = f"Flow: {flow} / Subflow: {subflow}"

        # Customer info
        cust_parts = []
        if personal.get("customer_name"):
            cust_parts.append(f"Name: {personal['customer_name']}")
        if personal.get("member_level"):
            cust_parts.append(f"Membership: {personal['member_level']}")
        customer_info = ", ".join(cust_parts) if cust_parts else "Not available"

        # Order info
        ord_parts = []
        if order.get("order_id"):
            ord_parts.append(f"Order ID: {order['order_id']}")
        if order.get("payment_method"):
            ord_parts.append(f"Payment: {order['payment_method']}")
        products = order.get("products", "")
        if products:
            # products is a stringified list of dicts; clean it up
            try:
                import ast
                prod_list = ast.literal_eval(products) if isinstance(products, str) else products
                names = [p.get("product_type", p.get("brand", "?")) for p in prod_list]
                ord_parts.append(f"Items: {', '.join(names)}")
            except Exception:
                pass
        order_info = ", ".join(ord_parts) if ord_parts else "Not available"

        base = _SYSTEM_PROMPT.format(
            subflow_description=subflow_desc,
            customer_info=customer_info,
            order_info=order_info,
        )

        # Inject workflow + exemplars (AWM)
        extra_parts = []
        wf = self.workflow.format_prompt()
        if wf:
            extra_parts.append(wf)
        # For memory, we need domains — use flow as domain
        ex = self.memory.format_prompt([flow, subflow])
        if ex:
            extra_parts.append(ex)

        if extra_parts:
            base = "\n".join(extra_parts) + "\n\n" + base

        return base

    def update_memory(self, dialogues, predictions, eval_results: list[dict]):
        """Store successful dialogues as exemplars (for AWM)."""
        for conv, pred, metrics in zip(dialogues, predictions, eval_results):
            if metrics.get("success") or metrics.get("info_rate", 0) > 0.8:
                scenario = conv.get("scenario", {})
                self.memory.add_dict({
                    "dialogue_id": f"abcd-{conv.get('convo_id', '?')}",
                    "domains": [scenario.get("flow", "?"), scenario.get("subflow", "?")],
                    "goal": f"{scenario.get('flow', '?')}/{scenario.get('subflow', '?')}",
                    "trajectory": pred.response_text[:1000],
                })

    def save_memory(self, path: str):
        self.memory.save(path)

    def load_memory(self, path: str):
        self.memory.load(path)

    def save_workflow(self, path: str):
        self.workflow.save(path)

    def load_workflow(self, path: str):
        self.workflow.load(path)
