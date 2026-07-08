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

# Ensure project root is FIRST on path (before any conflicting parent dirs)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
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

_TASK_PROMPT_WITH_ACTION = """## Conversation So Far
{context}

## Instruction
First identify what system action should be taken, then generate the agent response.

Output format:
```
ACTION: <action_name>
SLOTS: <slot1>, <slot2>
RESPONSE: <the agent response text>
```

- ACTION: the name of the system action to take (e.g. pull-up-account, verify-identity, send-link).
  If no action is needed, write "none".
- SLOTS: comma-separated slot values for the action. If no slots, write "none".
- RESPONSE: the natural language agent utterance (1-3 sentences)."""


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

    def predict_all_turns(
        self, conversation: dict[str, Any], verbose: bool = False,
        predict_actions: bool = False,
    ) -> list[dict]:
        """Predict EVERY agent turn in a conversation, not just the last.

        Args:
            predict_actions: If True, also predict system actions + slots
                for AST/CDS evaluation.  The output format changes to
                ``ACTION: ...\\nSLOTS: ...\\nRESPONSE: ...``.

        Returns:
            List of dicts with keys: turn_index, context, reference, prediction,
            (and predicted_action, predicted_slots if predict_actions=True).
        """
        convo_id = str(conversation.get("convo_id", "?"))
        scenario = conversation.get("scenario", {})
        delexed = conversation.get("delexed", [])
        system = self._build_system_prompt(scenario)
        results: list[dict] = []

        agent_indices = [
            i for i, t in enumerate(delexed)
            if t.get("speaker") == "agent" and t.get("text", "").strip()
        ]

        task_template = _TASK_PROMPT_WITH_ACTION if predict_actions else _TASK_PROMPT

        for agent_num, turn_idx in enumerate(agent_indices, 1):
            context_lines: list[str] = []
            for i in range(turn_idx):
                t = delexed[i]
                spk = t.get("speaker", "unknown")
                txt = t.get("text", "").strip()
                if not txt:
                    continue
                label_map = {"agent": "Agent", "customer": "Customer", "action": "System"}
                context_lines.append(f"[{label_map.get(spk, spk)}] {txt}")

            context = "\n".join(context_lines)
            reference = str(delexed[turn_idx].get("text", "")).strip()
            if not context or not reference:
                continue

            from llm import chat
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": task_template.format(context=context)},
            ]

            raw_output = ""
            try:
                raw_output = chat(
                    messages, model=self.model, api_key=self.api_key,
                    base_url=self.base_url, temperature=0.7, max_tokens=384,
                    response_logger=self._response_logger,
                ).strip()
            except Exception as exc:
                if verbose:
                    print(f"    LLM error convo={convo_id} turn={turn_idx}: {exc}")

            entry = {
                "convo_id": convo_id,
                "turn_index": turn_idx,
                "agent_turn_num": agent_num,
                "total_agent_turns": len(agent_indices),
                "subflow": str(scenario.get("subflow", "")),
                "flow": str(scenario.get("flow", "")),
                "context": context,
                "reference": reference,
                "prediction": raw_output,
            }

            if predict_actions:
                action, slots, resp = _parse_action_response(raw_output)
                entry["predicted_action"] = action
                entry["predicted_slots"] = slots
                entry["prediction"] = resp
            else:
                entry["prediction"] = raw_output.strip().strip('"').strip("'")

            results.append(entry)

        return results

    def generate_all_turn_predictions(
        self, conversations: list[dict[str, Any]], verbose: bool = True,
        predict_actions: bool = False,
    ) -> list[dict]:
        """Generate turn-level predictions for all conversations.

        Args:
            predict_actions: If True, each turn also predicts action+slots.

        Returns a flat list of all turn-level prediction dicts.
        """
        all_results: list[dict] = []
        total = len(conversations)
        for i, conv in enumerate(conversations):
            convo_id = str(conv.get("convo_id", i))
            flow = conv.get("scenario", {}).get("flow", "?")
            subflow = conv.get("scenario", {}).get("subflow", "?")
            if verbose:
                print(f"  [{i+1}/{total}] convo={convo_id}  {flow}/{subflow}")

            results = self.predict_all_turns(conv, verbose=verbose,
                                              predict_actions=predict_actions)
            all_results.extend(results)

            if verbose:
                print(f"    {len(results)} agent turns predicted")

            if i < total - 1:
                import time
                time.sleep(self.delay)

        return all_results

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

    def induce(
        self,
        dialogues,
        predictions,
        eval_results: list[dict],
    ) -> str:
        """Induce workflow patterns with LLM-managed update.

        Shows the LLM both the agent's generated response AND the ground-truth
        action sequence, so it can learn what the correct action pattern is.
        """
        from llm import chat

        # Build batch summary with ground-truth actions
        lines = ["## New Batch Summary"]
        for conv, pred, metrics in zip(dialogues, predictions, eval_results):
            scenario = conv.get("scenario", {})
            flow = scenario.get("flow", "?")
            subflow = scenario.get("subflow", "?")
            convo_id = conv.get("convo_id", "?")

            # Extract ground-truth action sequence
            gt_actions = _extract_action_sequence(conv)
            gt_str = " → ".join(gt_actions) if gt_actions else "(no actions)"

            lines.append(
                f"### {flow}/{subflow} (convo={convo_id})\n"
                f"- Ground Truth Actions: {gt_str}\n"
                f"- Agent Generated: {pred.response_text[:200] if pred.response_text else '(empty)'}\n"
            )

        existing = self.workflow.text if self.workflow else "(empty — first batch)"

        prompt = (
            "You maintain a living knowledge base of customer service workflow patterns. "
            "Given the existing workflow and a new batch with ground-truth action sequences "
            "and agent responses, produce the UPDATED workflow.\n\n"
            "## How to Use the Data\n"
            "- **Ground Truth Actions**: the CORRECT sequence of system actions for this dialogue. "
            "Use these to learn what the proper workflow should be.\n"
            "- **Agent Generated**: what the agent actually said. Compare against ground truth "
            "to identify gaps — does the agent's response align with the correct actions?\n\n"
            "## Update Rules\n"
            "- **Add** patterns from ground-truth action sequences that are not yet covered.\n"
            "- **Refine** patterns if the agent's response doesn't align with correct actions.\n"
            "- **Merge** patterns that are duplicates or very similar.\n"
            "- **Delete** patterns that consistently don't match the ground truth.\n"
            "- Keep the workflow concise and actionable (aim for 10-20 patterns max).\n"
            "- Preserve patterns that are still valid even if not seen in this batch.\n\n"
            + "\n".join(lines[:60])
            + "\n\n## Existing Workflow\n"
            + existing
            + "\n\n## Output: Updated Workflow\n"
            "Output the COMPLETE updated workflow (not a diff). Use this format:\n\n"
            "### [Flow/Subflow] - [Pattern Name]\n"
            "**When**: [condition that triggers this pattern]\n"
            "**Do**: [concrete strategy — what actions to take, what to say]\n"
            "**Avoid**: [common mistakes to avoid]\n"
        )

        updated = ""
        try:
            updated = chat(
                prompt,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.0,
                max_tokens=3072,
            ).strip()
        except Exception as exc:
            print(f"  [ABCD induce] LLM error: {exc}")
            return ""

        if updated.strip():
            old_lines = len(existing.splitlines()) if existing and existing != "(empty — first batch)" else 0
            self.workflow.replace(updated)
            n_lines = len(updated.splitlines())
            print(f"  [AWM] Workflow updated: {n_lines} lines (was {old_lines})")

        return updated

    def update_memory(self, dialogues, predictions, eval_results: list[dict]):
        """Store successful dialogues as exemplars — based on AST score > 0.5."""
        for conv, pred, metrics in zip(dialogues, predictions, eval_results):
            ast = metrics.get("ast_score", 0)
            if ast > 0.5 or metrics.get("success") or metrics.get("info_rate", 0) > 0.8:
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


# ── Output parsing helpers ────────────────────────────────────

def _extract_action_sequence(conv: dict) -> list[str]:
    """Extract ground-truth action sequence from an ABCD conversation.

    Returns list of action names (e.g. ``['pull-up-account', 'verify-identity', 'send-link']``).
    """
    actions: list[str] = []
    for turn in conv.get("delexed", []):
        targets = turn.get("targets", [])
        if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
            action_name = str(targets[2])
            slots = targets[3] if len(targets) > 3 else []
            if slots and isinstance(slots, list) and len(slots) > 0:
                action_name += ":" + ",".join(str(s)[:20] for s in slots)
            actions.append(action_name)
    return actions


def _parse_action_response(raw: str) -> tuple[str, list[str], str]:
    """Parse ``ACTION: ...\\nSLOTS: ...\\nRESPONSE: ...`` output.

    Returns (action_name, slot_list, response_text).
    """
    import re
    action = ""
    slots: list[str] = []
    response = raw  # fallback: whole text

    m = re.search(r"ACTION:\s*(.+)", raw)
    if m:
        action = m.group(1).strip()
        if action.lower() == "none":
            action = ""

    m = re.search(r"SLOTS:\s*(.+)", raw)
    if m:
        slots_text = m.group(1).strip()
        if slots_text.lower() != "none":
            slots = [s.strip() for s in slots_text.split(",") if s.strip()]

    m = re.search(r"RESPONSE:\s*\n?(.*?)$", raw, re.DOTALL)
    if m:
        response = m.group(1).strip().strip('"').strip("'")

    return action, slots, response


def turn_results_to_abcd_predictions(
    turn_results: list[dict],
    conversations: list[dict],
) -> list:
    """Convert turn-level prediction dicts to ABCDPrediction objects for AST/CDS.

    Groups per-turn action predictions back into per-conversation
    ABCDPrediction objects that ``evaluate_abcd()`` expects.
    """
    from .schemas import ABCDPrediction, ABCDTurnPrediction

    # Index conversations by convo_id
    conv_index: dict[str, dict] = {}
    for conv in conversations:
        cid = str(conv.get("convo_id", "?"))
        conv_index[cid] = conv

    # Group turn results by convo_id
    by_convo: dict[str, list[dict]] = {}
    for r in turn_results:
        cid = r["convo_id"]
        by_convo.setdefault(cid, []).append(r)

    predictions: list[ABCDPrediction] = []
    for cid, turns in by_convo.items():
        conv = conv_index.get(cid)
        if conv is None:
            continue

        delexed = conv.get("delexed", [])
        turn_preds: list[ABCDTurnPrediction] = []

        # Each agent turn predicts actions that follow until the next agent turn.
        # Action turns between agent A and agent B all get A's prediction.
        agent_preds: dict[int, str] = {}
        agent_slots: dict[int, str] = {}
        for r in sorted(turns, key=lambda x: x["turn_index"]):
            pa = r.get("predicted_action", "")
            ps = r.get("predicted_slots", [])
            if pa:
                agent_preds[r["turn_index"]] = pa
                agent_slots[r["turn_index"]] = ps

        agent_indices = sorted(agent_preds.keys())

        for turn_idx, turn in enumerate(delexed):
            targets = turn.get("targets", [])
            if len(targets) < 3 or targets[1] != "take_action":
                continue

            # Nearest preceding agent turn's prediction applies here
            pred_action = ""
            pred_slots: list[str] = []
            for ai in agent_indices:
                if ai < turn_idx:
                    pred_action = agent_preds[ai]
                    pred_slots = agent_slots[ai]
                else:
                    break  # agent_indices is sorted, stop once ai >= turn_idx

            turn_preds.append(ABCDTurnPrediction(
                turn_index=turn_idx,
                turn_type="action",
                predicted_action=pred_action if pred_action else None,
                predicted_slots=list(pred_slots) if pred_slots else None,
            ))

        predictions.append(ABCDPrediction(
            conversation_id=cid,
            turns=turn_preds,
        ))

    return predictions


def compute_per_dialogue_ast(
    conversations: list[dict],
) -> list[dict]:
    """Compute per-dialogue AST score for induction feedback.

    Compares ground-truth action turns in each conversation.  Callers should
    run predict_all_turns(predict_actions=True) first, then call this with
    the resulting turn dicts to compute per-dialogue AST scores.

    Returns list of dicts: {ast_score, action_correct, action_total}.
    """
    from .data import extract_ground_truth

    scores: list[dict] = []
    for conv in conversations:
        truths = extract_ground_truth(conv)
        gt_action_turns = [
            t for t in truths if t.turn_type == "action" and t.action_name
        ]
        total = len(gt_action_turns)
        # Without predictions, report totals only
        scores.append({
            "ast_score": 0.0,
            "action_correct": 0,
            "action_total": total,
        })
    return scores


def compute_ast_from_turn_results(
    conversations: list[dict],
    turn_results: list[dict],
) -> list[dict]:
    """Compute per-dialogue AST from turn-level predictions (predict_actions=True).

    Groups turn_results by convo_id, compares predicted_action against
    ground-truth action turns.

    Returns list of dicts (aligned with conversations): {ast_score, action_correct, action_total}.
    """
    from .data import extract_ground_truth

    # Group turn results by convo_id
    by_convo: dict[str, list[dict]] = {}
    for r in turn_results:
        cid = r["convo_id"]
        by_convo.setdefault(cid, []).append(r)

    scores: list[dict] = []
    for conv in conversations:
        cid = str(conv.get("convo_id", "?"))
        truths = extract_ground_truth(conv)
        gt_actions = {
            t.turn_index: t
            for t in truths if t.turn_type == "action" and t.action_name
        }

        turns = by_convo.get(cid, [])
        # Map predictions to action turns: for each action turn, find the
        # closest preceding agent turn prediction
        correct = 0
        for turn_idx, gt in gt_actions.items():
            # Find best matching prediction (the agent turn just before this action)
            best_action = ""
            for r in turns:
                if r["turn_index"] < turn_idx and "predicted_action" in r:
                    best_action = r["predicted_action"]
            if best_action and best_action == gt.action_name:
                correct += 1

        total = len(gt_actions)
        scores.append({
            "ast_score": correct / max(total, 1),
            "action_correct": correct,
            "action_total": total,
        })

    return scores
