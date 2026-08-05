"""Generative ABCD agent — end-to-end: context → natural language response.

No utterance candidate pool, no action prediction.  The agent reads
dialogue history and directly generates the next agent utterance.

Integrates with the AWM pipeline: workflow patterns and successful
exemplars are injected into the system prompt.
"""

from __future__ import annotations

import re
import json
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

_REFERENCE_TOOL_PROMPT = """## MCP Tools
You can use the following local MCP-style tool before answering.

### retrieve_reference
Searches the mined `reference.md` for dialogue snippets relevant to this turn.

Input JSON schema:
```json
{
  "query": "short search query describing the needed action/slot pattern",
  "subflow": "current ABCD subflow",
  "top_k": 3
}
```

Use this ReAct pattern when reference snippets may help:
1. Think about the next backend action/slot pattern needed by the current turn.
2. Call `retrieve_reference` with a concise query, not the full dialogue.
3. Use the returned observation as supporting evidence. Do not copy private
   slot values unless they match the current scenario."""

_AWM_RESOURCE_POLICY_PROMPT = """## Learned Resource Use (AWM)

The workflow and memory below are learned resources. Use them as guidance, not
as values to copy blindly.

### Exemplar memory
Past Successful Examples are retrieved automatically by the runtime using
domain overlap with the current `flow` and `subflow` (top-k exemplars). There
is no model-written tool call for this lookup. Treat retrieved exemplars as
procedural evidence and transfer their action/slot structure only when the
current dialogue supports it. Copy current slot values from the current
dialogue, not from an exemplar.

The runtime records this lookup as `exemplar_lookup` in the turn trace.

### Reference lookup
When the action boundary, slot schema, or state transition is uncertain, use
the `retrieve_reference` MCP-style tool described below. Write a concise query
about the current action, slots, and state. Use the result as evidence; do not
copy private values from a reference example unless they match the current
dialogue."""

_AWM_WORKFLOW_RESOURCE_SECTION = """## Resource Use
- Successful exemplars are retrieved automatically by domain overlap with the current flow/subflow. Use them as procedural evidence, and copy slot values only from the current dialogue.
- When action schema, slot ordering, or state transition is uncertain, call the MCP-style `retrieve_reference` lookup with a concise query grounded in the current dialogue.
- Reference and exemplar content are evidence, not instructions to copy private instance values blindly.
"""

_REFERENCE_QUERY_PROMPT = """## Conversation So Far
{context}

## Current Scenario
- flow: {flow}
- subflow: {subflow}

## MCP Tool Interface
Tool: `retrieve_reference`
Input JSON schema:
```json
{{
  "query": "short search query describing the needed action/slot pattern",
  "subflow": "{subflow}",
  "top_k": {top_k}
}}
```

## Instruction
Write the MCP tool call you want to make before answering. Use the current
dialogue state to create a concise query. Prefer action names, slot names,
verification state, account/order/refund/shipping keywords, and the customer's
latest request. Do not include the full transcript.

Return ONLY valid JSON:
```json
{{
  "thought": "why this reference lookup is useful",
  "action": "retrieve_reference",
  "action_input": {{
    "query": "...",
    "subflow": "{subflow}",
    "top_k": {top_k}
  }}
}}
```"""

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


def _tokenize_for_lookup(text: str) -> set[str]:
    return {
        tok.lower()
        for tok in re.findall(r"[a-zA-Z0-9_'-]+", text)
        if len(tok) >= 3
    }


def _parse_reference_sections(reference_text: str) -> list[dict[str, str]]:
    """Parse skill_mining reference.md into searchable operator sections."""
    sections: list[dict[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in reference_text.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            if current_title and current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append({"title": current_title, "body": body})
            current_title = line[3:].strip()
            current_lines = []
            continue
        if current_title:
            current_lines.append(line)

    if current_title and current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"title": current_title, "body": body})

    return sections


def _canonical_reference_title(title: str) -> str:
    """Collapse legacy slot-specific reference titles to an action name."""
    return title.split(":", 1)[0].strip()


def _get_original_turn(
    conversation: dict[str, Any],
    turn_index: int,
    fallback_turn: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Return the raw speaker/text for a turn aligned with ``delexed``.

    ABCD stores labels and targets in ``delexed`` but keeps the real entity
    values in ``original``.  Runtime prompts should use the latter so slot
    values are observable; action labels still come from ``delexed``.
    """
    original = conversation.get("original") or []
    if 0 <= turn_index < len(original):
        raw_turn = original[turn_index]
        if isinstance(raw_turn, dict):
            return (
                str(raw_turn.get("speaker", "unknown")),
                str(raw_turn.get("text", "")).strip(),
            )
        if isinstance(raw_turn, (list, tuple)) and len(raw_turn) >= 2:
            return str(raw_turn[0]), str(raw_turn[1]).strip()

    fallback_turn = fallback_turn or {}
    return (
        str(fallback_turn.get("speaker", "unknown")),
        str(fallback_turn.get("text", "")).strip(),
    )


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
        reference_text: str | None = None,
        reference_top_k: int = 3,
        reference_max_chars: int = 1800,
        delay: float = 0.3,
        response_logger=None,
        expose_scenario_labels: bool = True,
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
        self.reference_text = reference_text or ""
        self.reference_sections = _parse_reference_sections(self.reference_text)
        self.reference_top_k = max(0, reference_top_k)
        self.reference_max_chars = max(200, reference_max_chars)
        self._response_logger = response_logger
        self.expose_scenario_labels = expose_scenario_labels
        self._last_exemplar_lookup: dict[str, Any] = {
            "tool": "retrieve_exemplar",
            "executed": False,
            "status": "not_started",
            "query": {},
            "selected_exemplars": [],
        }

    def set_reference_text(self, reference_text: str | None) -> None:
        """Replace prompt-time mined-reference material."""
        self.reference_text = reference_text or ""
        self.reference_sections = _parse_reference_sections(self.reference_text)

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
            spk, txt = _get_original_turn(conversation, i, turn)
            if not txt:
                continue
            label = {"agent": "Agent", "customer": "Customer", "action": "System"}.get(spk, spk)
            history_lines.append(f"[{label}] {txt}")
            if spk == "agent":
                last_agent_idx = i

        context = "\n".join(history_lines[:-1]) if len(history_lines) > 1 else history_lines[0] if history_lines else ""

        reference_plan = self._plan_reference_lookup(context, scenario, verbose=False)
        reference_lookup = self._lookup_reference(
            reference_plan.get("query_text", ""),
            context,
            scenario,
            top_k=reference_plan.get("top_k"),
        )
        prompt_context = context
        if reference_lookup["observation"]:
            prompt_context += "\n\n" + reference_lookup["observation"]

        # Build messages
        from llm import chat

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _TASK_PROMPT.format(context=prompt_context)},
        ]

        response_text = ""
        try:
            response_text = chat(
                messages,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.7,
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

        # Keep both target types.  Agent turns are used for response metrics;
        # action turns are separate ABCD targets and must be predicted at
        # their own turn index for AST.  Mapping every action back to the
        # previous agent utterance loses alignment when several utterances
        # occur before one backend action.
        agent_indices = [
            i for i, t in enumerate(delexed)
            if t.get("speaker") == "agent" and t.get("text", "").strip()
        ]
        action_indices = [
            i for i, t in enumerate(delexed)
            if len(t.get("targets", [])) >= 3
            and t.get("targets", [None, None])[1] == "take_action"
        ]
        target_indices = sorted(set(agent_indices) | set(action_indices))

        task_template = _TASK_PROMPT_WITH_ACTION if predict_actions else _TASK_PROMPT

        for target_num, turn_idx in enumerate(target_indices, 1):
            context_lines: list[str] = []
            for i in range(turn_idx):
                t = delexed[i]
                spk, txt = _get_original_turn(conversation, i, t)
                if not txt:
                    continue
                label_map = {"agent": "Agent", "customer": "Customer", "action": "System"}
                context_lines.append(f"[{label_map.get(spk, spk)}] {txt}")

            context = "\n".join(context_lines)
            reference = str(delexed[turn_idx].get("text", "")).strip()
            _, reference_original = _get_original_turn(
                conversation, turn_idx, delexed[turn_idx]
            )
            if not context or not reference:
                continue

            reference_plan = self._plan_reference_lookup(context, scenario, verbose=verbose)
            reference_lookup = self._lookup_reference(
                reference_plan.get("query_text", ""),
                context,
                scenario,
                top_k=reference_plan.get("top_k"),
            )
            prompt_context = context
            if reference_lookup["observation"]:
                prompt_context += "\n\n" + reference_lookup["observation"]

            from llm import chat
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": task_template.format(context=prompt_context)},
            ]

            raw_output = ""
            try:
                raw_output = chat(
                    messages, model=self.model, api_key=self.api_key,
                    base_url=self.base_url, temperature=0.7,
                    response_logger=self._response_logger,
                ).strip()
            except Exception as exc:
                if verbose:
                    print(f"    LLM error convo={convo_id} turn={turn_idx}: {exc}")

            entry = {
                "convo_id": convo_id,
                "turn_index": turn_idx,
                "agent_turn_num": (
                    agent_indices.index(turn_idx) + 1
                    if turn_idx in agent_indices else None
                ),
                "target_type": "action" if turn_idx in action_indices else "utterance",
                "total_agent_turns": len(agent_indices),
                "subflow": str(scenario.get("subflow", "")),
                "flow": str(scenario.get("flow", "")),
                "context": context,
                "context_view": "original",
                "workflow_injected": bool(self.workflow.text.strip()),
                "workflow_chars": len(self.workflow.text),
                "memory_exemplars": len(self.memory),
                "reference_lookup": reference_lookup,
                "exemplar_lookup": getattr(self, "_last_exemplar_lookup", {}),
                "react_trace": [
                    {
                        "turn": 1,
                        "thought": "Use the runtime's domain-overlap exemplar lookup as procedural evidence.",
                        "action": "retrieve_exemplar",
                        "action_input": self._last_exemplar_lookup.get("query", {}),
                        "executed": self._last_exemplar_lookup.get("executed", False),
                        "status": self._last_exemplar_lookup.get("status", "unknown"),
                        "selected_exemplars": self._last_exemplar_lookup.get("selected_exemplars", []),
                    },
                    {
                        "turn": 2,
                        "thought": "Plan a concise MCP retrieve_reference query for the current dialogue state.",
                        "action": "llm_plan_reference_query",
                        "action_input": {
                            "messages": reference_plan.get("messages", []),
                        },
                        "observation": reference_plan.get("raw_output", ""),
                        "parsed_tool_call": reference_plan.get("tool_call", {}),
                        "fallback_used": reference_plan.get("fallback_used", False),
                    },
                    {
                        "turn": 3,
                        "thought": reference_plan.get("thought", ""),
                        "action": "retrieve_reference",
                        "action_input": reference_lookup["query"],
                        "executed": reference_lookup.get("executed", False),
                        "status": reference_lookup.get("status", "unknown"),
                        "observation": reference_lookup["observation"],
                        "selected_sections": reference_lookup["selected_sections"],
                    },
                    {
                        "turn": 4,
                        "thought": "Predict the backend action, ordered slots, and response using the workflow, scenario, dialogue context, and retrieved snippets.",
                        "action": "llm_generate",
                        "action_input": {
                            "messages": messages,
                        },
                        "observation": raw_output,
                    },
                ],
                "reference": reference,
                "reference_original": reference_original,
                "prediction": raw_output,
            }

            if predict_actions:
                action, slots, resp = _parse_action_response(raw_output)
                # Debug: log first few parses
                if target_num == 1 and len(results) == 0:
                    print(f"  [DEBUG predict_actions] raw(200): {raw_output[:200]}")
                    print(f"  [DEBUG predict_actions] parsed: action={action!r} slots={slots!r} resp={resp[:80]!r}")
                entry["predicted_action"] = action
                entry["predicted_slots"] = slots
                entry["prediction"] = resp
            else:
                entry["prediction"] = raw_output.strip().strip('"').strip("'")

            results.append(entry)

        return results

    def _recent_context_excerpt(self, context: str, max_lines: int = 6) -> str:
        lines = [line for line in context.splitlines() if line.strip()]
        return "\n".join(lines[-max_lines:])

    def _parse_reference_tool_call(
        self,
        raw_output: str,
        scenario: dict[str, Any],
        context: str,
    ) -> dict[str, Any]:
        """Parse the model's MCP-style retrieve_reference tool call."""
        payload = raw_output.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", payload, re.DOTALL)
        if fenced:
            payload = fenced.group(1)
        else:
            start = payload.find("{")
            end = payload.rfind("}")
            payload = payload[start:end + 1] if start != -1 and end > start else ""

        fallback_query = " ".join(
            self._recent_context_excerpt(context, max_lines=4).split()
        )[:240]
        if not fallback_query and self.expose_scenario_labels:
            fallback_query = str(scenario.get("subflow", ""))

        visible_subflow = (
            str(scenario.get("subflow", ""))
            if self.expose_scenario_labels else ""
        )

        try:
            parsed = json.loads(payload)
        except Exception:
            return {
                "thought": "Fallback: the model did not return parseable MCP JSON.",
                "action": "retrieve_reference",
                "action_input": {
                    "query": fallback_query,
                    "subflow": visible_subflow,
                    "top_k": self.reference_top_k,
                },
                "fallback_used": True,
            }

        action_input = parsed.get("action_input", {})
        if not isinstance(action_input, dict):
            action_input = {}
        query_text = str(action_input.get("query") or parsed.get("query") or "").strip()
        if not query_text:
            query_text = fallback_query
            fallback_used = True
        else:
            fallback_used = False
        try:
            top_k = int(action_input.get("top_k", self.reference_top_k))
        except Exception:
            top_k = self.reference_top_k

        return {
            "thought": str(parsed.get("thought", "")).strip(),
            "action": "retrieve_reference",
            "action_input": {
                "query": query_text,
                "subflow": str(action_input.get("subflow") or visible_subflow),
                "top_k": max(1, top_k),
            },
            "fallback_used": fallback_used,
        }

    def _plan_reference_lookup(
        self,
        context: str,
        scenario: dict[str, Any],
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Ask the model to produce an MCP-style reference lookup query."""
        if not self.reference_sections or self.reference_top_k <= 0:
            return {
                "thought": "No reference sections are available.",
                "query_text": "",
                "top_k": self.reference_top_k,
                "raw_output": "",
                "tool_call": {},
                "messages": [],
                "fallback_used": False,
            }

        if self.expose_scenario_labels:
            flow = str(scenario.get("flow", ""))
            subflow = str(scenario.get("subflow", ""))
        else:
            flow = ""
            subflow = ""
        recent_context = self._recent_context_excerpt(context)
        user_prompt = _REFERENCE_QUERY_PROMPT.format(
            context=recent_context,
            flow=flow,
            subflow=subflow,
            top_k=self.reference_top_k,
        )
        messages = [
            {"role": "system", "content": self._build_system_prompt(scenario)},
            {"role": "user", "content": user_prompt},
        ]

        raw_output = ""
        try:
            from llm import chat
            raw_output = chat(
                messages,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.0,
                response_logger=self._response_logger,
            ).strip()
        except Exception as exc:
            if verbose:
                print(f"    reference query planning error: {exc}")

        tool_call = self._parse_reference_tool_call(raw_output, scenario, context)
        action_input = tool_call.get("action_input", {})
        return {
            "thought": tool_call.get("thought", ""),
            "query_text": str(action_input.get("query", "")),
            "top_k": int(action_input.get("top_k", self.reference_top_k)),
            "raw_output": raw_output,
            "tool_call": tool_call,
            "messages": messages,
            "fallback_used": bool(tool_call.get("fallback_used", False)),
        }

    def _lookup_reference(
        self,
        query_text: str,
        context: str,
        scenario: dict[str, Any],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Return compact reference snippets relevant to the model-written query."""
        requested_top_k = max(1, int(top_k or self.reference_top_k))
        visible_subflow = (
            str(scenario.get("subflow", ""))
            if self.expose_scenario_labels else ""
        )
        query = {
            "query_text": query_text,
            "subflow": visible_subflow,
            "top_k": requested_top_k,
            "max_chars": self.reference_max_chars,
            "reference_available": bool(self.reference_sections),
        }
        if not self.reference_sections or requested_top_k <= 0:
            return {
                "tool": "retrieve_reference",
                "query": query,
                "executed": False,
                "status": "no_reference_loaded",
                "selected_sections": [],
                "observation": "",
            }

        subflow = (
            str(scenario.get("subflow", "")).replace("_", " ")
            if self.expose_scenario_labels else ""
        )
        query_tokens = _tokenize_for_lookup(query_text + " " + subflow)
        context_tokens = _tokenize_for_lookup(self._recent_context_excerpt(context, max_lines=4))
        if not query_tokens:
            query_tokens = context_tokens | _tokenize_for_lookup(subflow)
        query["query_tokens"] = sorted(query_tokens)[:80]
        if not query_tokens:
            return {
                "tool": "retrieve_reference",
                "query": query,
                "executed": False,
                "status": "empty_query",
                "selected_sections": [],
                "observation": "",
            }

        best_by_action: dict[str, tuple[float, dict[str, str]]] = {}
        for section in self.reference_sections:
            title = section["title"]
            body = section["body"]
            canonical_title = _canonical_reference_title(title)
            title_tokens = _tokenize_for_lookup(title.replace("-", " "))
            canonical_tokens = _tokenize_for_lookup(canonical_title.replace("-", " "))
            body_tokens = _tokenize_for_lookup(body[:1200])
            score = 0.0
            score += 4.0 * len(query_tokens & title_tokens)
            score += 5.0 * len(query_tokens & canonical_tokens)
            score += 1.0 * len(query_tokens & body_tokens)
            score += 0.25 * len(context_tokens & body_tokens)
            # Common action names are especially useful for AST guidance.
            if any(tok in canonical_tokens for tok in ("verify", "identity", "account", "order")):
                score += 0.25
            if score > 0:
                current = best_by_action.get(canonical_title)
                if current is None or score > current[0]:
                    best_by_action[canonical_title] = (score, section)

        if best_by_action:
            scored = list(best_by_action.values())
        else:
            seen_actions: set[str] = set()
            scored = []
            for section in self.reference_sections:
                canonical_title = _canonical_reference_title(section["title"])
                if canonical_title in seen_actions:
                    continue
                seen_actions.add(canonical_title)
                scored.append((0.0, section))
                if len(scored) >= requested_top_k:
                    break

        scored.sort(key=lambda item: (-item[0], _canonical_reference_title(item[1]["title"]), item[1]["title"]))
        parts = [
            "## Reference Lookup Results",
            "Use these mined dialogue snippets as examples; do not copy private slot values unless they match the current scenario.",
        ]
        used_chars = sum(len(p) for p in parts)
        selected_sections: list[dict[str, Any]] = []
        for _, section in scored[:requested_top_k]:
            display_title = _canonical_reference_title(section["title"])
            snippet = section["body"].strip()
            if len(snippet) > 600:
                snippet = snippet[:600].rstrip() + "\n..."
            block = f"\n### {display_title}\n{snippet}"
            if used_chars + len(block) > self.reference_max_chars:
                remaining = self.reference_max_chars - used_chars
                if remaining > 120:
                    parts.append(block[:remaining].rstrip() + "\n...")
                    selected_sections.append({
                        "title": section["title"],
                        "canonical_title": display_title,
                        "truncated": True,
                    })
                break
            parts.append(block)
            selected_sections.append({
                "title": section["title"],
                "canonical_title": display_title,
                "truncated": False,
            })
            used_chars += len(block)

        return {
            "tool": "retrieve_reference",
            "query": query,
            "executed": True,
            "status": "matched" if selected_sections else "no_match",
            "selected_sections": selected_sections,
            "observation": "\n".join(parts).strip(),
        }

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

        # Scenario labels are optional: mixed training must infer the task
        # from the dialogue instead of receiving the dataset category.
        subflow_desc = (
            f"Flow: {flow} / Subflow: {subflow}"
            if self.expose_scenario_labels
            else "Infer the customer's task from the conversation; no task label is provided."
        )

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

        # Inject the resource policy plus workflow and exemplars (AWM).
        extra_parts = [_AWM_RESOURCE_POLICY_PROMPT]
        wf = self.workflow.format_prompt()
        if wf:
            extra_parts.append(wf)
        # For memory, we need domains — use flow as domain
        memory_domains = [flow, subflow] if self.expose_scenario_labels else []
        selected_exemplars = (
            self.memory.retrieve(memory_domains, k=self.memory.max_exemplars)
            if self.memory else []
        )
        overlap_count = sum(
            1 for exemplar in selected_exemplars
            if set(memory_domains) & set(exemplar.get("domains", []))
        )
        if not self.memory:
            lookup_status = "no_memory_loaded"
        elif not selected_exemplars:
            lookup_status = "no_exemplar_match"
        elif overlap_count:
            lookup_status = "matched_domain_overlap"
        else:
            lookup_status = "fallback_top_k"
        self._last_exemplar_lookup = {
            "tool": "retrieve_exemplar",
            "executed": bool(self.memory),
            "status": lookup_status,
            "query": {
                "domains": memory_domains,
                "top_k": self.memory.max_exemplars if self.memory else 0,
                "strategy": "domain_overlap",
            },
            "selected_exemplars": [
                {
                    "dialogue_id": exemplar.get("dialogue_id", ""),
                    "domains": exemplar.get("domains", []),
                }
                for exemplar in selected_exemplars
            ],
        }
        ex = self.memory.format_prompt(
            memory_domains,
            include_metadata=self.expose_scenario_labels,
        )
        if ex:
            extra_parts.append(ex)

        if extra_parts:
            base = "\n".join(extra_parts) + "\n\n" + base

        if self.reference_sections and self.reference_top_k > 0:
            base = base + "\n\n" + _REFERENCE_TOOL_PROMPT

        return base

    def induce(
        self,
        dialogues,
        predictions,
        eval_results: list[dict],
        turn_results: list[dict] | None = None,
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

            header = (
                f"### {flow}/{subflow} (convo={convo_id})"
                if self.expose_scenario_labels
                else f"### Conversation {convo_id}"
            )
            trajectory = ""
            if turn_results is not None:
                rows = [
                    row for row in turn_results
                    if str(row.get("convo_id", "")) == str(convo_id)
                ]
                trajectory = _format_abcd_turn_trajectory(conv, rows)
            lines.append(
                f"{header}\n"
                f"- Ground Truth Actions: {gt_str}\n"
                f"- Agent Generated (last turn): {pred.response_text[:200] if pred.response_text else '(empty)'}\n"
                + (f"- Full Turn Trajectory:\n{trajectory}\n" if trajectory else "")
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
            "to identify gaps — does the agent's response align with the correct actions?\n"
            "- **Full Turn Trajectory**: use every turn's context, predicted action, "
            "predicted slots, response, and `ast_correct` field. Do not infer the "
            "workflow only from the final response.\n\n"
            "## Update Rules\n"
            "- **Add** patterns from ground-truth action sequences that are not yet covered.\n"
            "- **Refine** patterns if the agent's response doesn't align with correct actions.\n"
            "- **Merge** patterns that are duplicates or very similar.\n"
            "- **Delete** patterns that consistently don't match the ground truth.\n"
            "- Keep the workflow concise and actionable (aim for 10-20 patterns max).\n"
            "- Preserve patterns that are still valid even if not seen in this batch.\n\n"
            "## Resource Use Requirement\n"
            "Include a concise `## Resource Use` section in the updated workflow. "
            "State that successful exemplars are automatically retrieved by current "
            "flow/subflow domain overlap and must be used as procedural evidence, "
            "not copied for private slot values. State that `retrieve_reference` "
            "is the MCP-style lookup for uncertain action schemas, slot patterns, "
            "or state transitions, and that its query must be concise and grounded "
            "in the current dialogue.\n\n"
            + "\n".join(lines[:60])
            + "\n\n## Existing Workflow\n"
            + existing
            + "\n\n## Output: Updated Workflow\n"
            + "Output the COMPLETE updated workflow (not a diff). Use this format:\n\n"
            + (
                "### [Flow/Subflow] - [Pattern Name]\n"
                if self.expose_scenario_labels
                else "### [Pattern Name]\n"
            )
            + "**When**: [condition that triggers this pattern]\n" +
            "**Do**: [concrete strategy — what actions to take, what to say]\n"
            "**Avoid**: [common mistakes to avoid]\n"
        )

        updated = ""
        for attempt in range(3):
            try:
                updated = chat(
                    prompt,
                    model=self.model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    temperature=0.0,
                ).strip()
                if updated:
                    break
                print(f"  [ABCD induce] empty response, retry {attempt + 1}/3")
            except Exception as exc:
                print(f"  [ABCD induce] LLM error ({attempt + 1}/3): {exc}")
            if attempt < 2:
                time.sleep(2 ** attempt)

        if not updated:
            print("  [ABCD induce] failed after 3 attempts; workflow unchanged")
            return ""

        # Make resource usage part of the persisted workflow contract even if
        # the induction model omits the requested section.
        if "## Resource Use" not in updated:
            updated = updated.rstrip() + "\n\n" + _AWM_WORKFLOW_RESOURCE_SECTION.strip()

        if updated.strip():
            old_lines = len(existing.splitlines()) if existing and existing != "(empty — first batch)" else 0
            self.workflow.replace(updated)
            n_lines = len(updated.splitlines())
            print(f"  [AWM] Workflow updated: {n_lines} lines (was {old_lines})")

        return updated

    def update_memory(
        self,
        dialogues,
        predictions,
        eval_results: list[dict],
        turn_results: list[dict] | None = None,
    ):
        """Store successful dialogues as exemplars — based on AST score > 0.5."""
        for conv, pred, metrics in zip(dialogues, predictions, eval_results):
            ast = metrics.get("ast_score", 0)
            if ast > 0.5 or metrics.get("success") or metrics.get("info_rate", 0) > 0.8:
                scenario = conv.get("scenario", {})
                domains = (
                    [scenario.get("flow", "?"), scenario.get("subflow", "?")]
                    if self.expose_scenario_labels
                    else []
                )
                rows = []
                if turn_results is not None:
                    rows = [
                        row for row in turn_results
                        if str(row.get("convo_id", ""))
                        == str(conv.get("convo_id", "?"))
                    ]
                trajectory = (
                    _format_abcd_turn_trajectory(conv, rows)
                    if rows else pred.response_text[:1000]
                )
                self.memory.add_dict({
                    "dialogue_id": f"abcd-{conv.get('convo_id', '?')}",
                    "domains": domains,
                    "goal": (
                        f"{scenario.get('flow', '?')}/{scenario.get('subflow', '?')}"
                        if self.expose_scenario_labels
                        else "customer-service dialogue"
                    ),
                    "trajectory": trajectory[:4000],
                    "trajectory_turns": _build_abcd_turn_trajectory(conv, rows),
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


def _build_abcd_turn_trajectory(
    conversation: dict[str, Any],
    turn_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join generated records with every ABCD utterance/action turn."""
    result_by_index = {
        int(row["turn_index"]): row
        for row in turn_results
        if isinstance(row, dict) and "turn_index" in row
    }
    delexed = conversation.get("delexed", [])
    trajectory: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(delexed):
        row = result_by_index.get(turn_index, {})
        speaker = str(turn.get("speaker", "unknown"))
        targets = turn.get("targets", [])
        is_action = len(targets) >= 3 and targets[1] == "take_action"
        turn_type = "action" if is_action else (
            "utterance" if speaker == "agent" else "customer"
        )

        context = str(row.get("context", "")).strip()
        if not context:
            context_lines: list[str] = []
            for previous_index in range(turn_index):
                previous = delexed[previous_index]
                _, text = _get_original_turn(conversation, previous_index, previous)
                if not text:
                    continue
                label = {
                    "agent": "Agent",
                    "customer": "Customer",
                    "action": "System",
                }.get(str(previous.get("speaker", "unknown")), "unknown")
                context_lines.append(f"[{label}] {text}")
            context = "\n".join(context_lines)

        predicted_action = str(row.get("predicted_action", "") or "").strip()
        predicted_slots = [str(value) for value in (row.get("predicted_slots") or [])]
        if is_action and not predicted_action:
            # Backward-compatible alignment for legacy turn results that only
            # contain agent rows.  New runs contain direct action rows.
            for previous_index in sorted(result_by_index):
                if previous_index >= turn_index:
                    break
                previous = result_by_index[previous_index]
                if previous.get("target_type") != "action" and previous.get("predicted_action"):
                    predicted_action = str(previous["predicted_action"]).strip()
                    predicted_slots = [
                        str(value) for value in (previous.get("predicted_slots") or [])
                    ]
        gold_action = str(targets[2]) if is_action and targets[2] else None
        gold_slots = (
            list(targets[3])
            if is_action and len(targets) > 3 and isinstance(targets[3], list)
            else []
        )

        ast_correct: bool | None = None
        action_correct: bool | None = None
        slot_correct: bool | None = None
        if is_action:
            action_correct = predicted_action == gold_action
            slot_correct = predicted_slots == [str(value) for value in gold_slots]
            ast_correct = bool(action_correct and slot_correct)

        trajectory.append({
            "turn_index": turn_index,
            "speaker": speaker,
            "turn_type": turn_type,
            "context": context,
            "predicted_action": predicted_action or None,
            "predicted_slots": predicted_slots,
            "gold_action": gold_action,
            "gold_slots": [str(value) for value in gold_slots],
            "action_correct": action_correct,
            "slot_correct": slot_correct,
            "ast_correct": ast_correct,
            "response": str(row.get("prediction", "") or "").strip(),
            "reference": str(row.get("reference_original") or row.get("reference", "") or "").strip(),
        })

    return trajectory


def _format_abcd_turn_trajectory(
    conversation: dict[str, Any],
    turn_results: list[dict[str, Any]],
    max_chars: int = 4000,
) -> str:
    """Render a bounded prompt view of a structured ABCD trajectory."""
    rows = _build_abcd_turn_trajectory(conversation, turn_results)
    lines: list[str] = []
    for row in rows:
        ast = "N/A" if row["ast_correct"] is None else str(row["ast_correct"])
        slots = ", ".join(row["predicted_slots"]) if row["predicted_slots"] else "none"
        context = row["context"]
        if len(context) > 1600:
            context = context[-1600:]
        response = row["response"]
        if len(response) > 600:
            response = response[:600] + "..."
        gold_lines = (
            f"  gold_action: {row['gold_action']}\n"
            f"  gold_slots: {', '.join(row['gold_slots']) if row['gold_slots'] else 'none'}\n"
            if row["turn_type"] == "action" else ""
        )
        lines.append(
            f"- turn_index: {row['turn_index']} | speaker: {row['speaker']} | "
            f"turn_type: {row['turn_type']} | ast_correct: {ast}\n"
            f"  context: {context}\n"
            f"  predicted_action: {row['predicted_action'] or 'none'}\n"
            f"  predicted_slots: {slots}\n"
            + gold_lines
            + f"  response: {response or '(empty)'}"
        )
    return "\n".join(lines)[:max_chars]


def _parse_action_response(raw: str) -> tuple[str, list[str], str]:
    """Parse ``ACTION: ...\\nSLOTS: ...\\nRESPONSE: ...`` output.

    Returns (action_name, slot_list, response_text).
    """
    import json
    import re
    action = ""
    slots: list[str] = []
    response = raw  # fallback: whole text

    # Accept JSON and fenced JSON as well as the documented line format.
    # Some OpenAI-compatible endpoints ignore the line-format instruction
    # and return a structured object instead.
    payload = raw.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload,
                         flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        obj = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        obj = None
    if isinstance(obj, dict):
        action_value = obj.get("action", obj.get("predicted_action", ""))
        slot_value = obj.get("slots", obj.get("predicted_slots", []))
        response_value = obj.get("response", obj.get("prediction", raw))
        action = str(action_value or "").strip()
        if action.lower() in {"none", "null", "no_action", "no-action"}:
            action = ""
        if isinstance(slot_value, list):
            slots = [str(s).strip() for s in slot_value if str(s).strip()]
        elif slot_value and str(slot_value).lower() != "none":
            slots = [s.strip() for s in str(slot_value).split(",") if s.strip()]
        response = str(response_value or "").strip().strip('"').strip("'")
        return action, slots, response

    m = re.search(r"^\s*ACTION\s*:\s*(.+?)\s*$", raw,
                  re.IGNORECASE | re.MULTILINE)
    if m:
        action = m.group(1).strip()
        if action.lower() == "none":
            action = ""

    m = re.search(r"^\s*SLOTS\s*:\s*(.+?)\s*$", raw,
                  re.IGNORECASE | re.MULTILINE)
    if m:
        slots_text = m.group(1).strip()
        if slots_text.lower() != "none":
            slots = [s.strip() for s in slots_text.split(",") if s.strip()]

    m = re.search(r"^\s*RESPONSE\s*:\s*\n?(.*?)$", raw,
                  re.IGNORECASE | re.MULTILINE | re.DOTALL)
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
    total_agent_preds = 0
    total_action_turns = 0
    total_mapped = 0

    missing_conversations = 0
    # Preserve the input conversation order and emit an empty prediction for
    # a dialogue with no generated agent turns.  This keeps callers that zip
    # conversations with predictions aligned instead of silently shifting all
    # subsequent AST results.
    for conv in conversations:
        cid = str(conv.get("convo_id", "?"))
        turns = by_convo.get(cid, [])
        if not turns:
            missing_conversations += 1

        delexed = conv.get("delexed", [])
        turn_preds: list[ABCDTurnPrediction] = []

        agent_preds: dict[int, str] = {}
        agent_slots: dict[int, list[str]] = {}
        direct_action_preds: dict[int, tuple[str, list[str]]] = {}
        for r in sorted(turns, key=lambda x: x["turn_index"]):
            pa = r.get("predicted_action", "")
            ps = r.get("predicted_slots", [])
            if pa:
                idx = int(r["turn_index"])
                if r.get("target_type") == "action":
                    direct_action_preds[idx] = (pa, list(ps or []))
                else:
                    agent_preds[idx] = pa
                    agent_slots[idx] = list(ps or [])
        total_agent_preds += len(agent_preds)

        agent_indices = sorted(agent_preds.keys())

        for turn_idx, turn in enumerate(delexed):
            targets = turn.get("targets", [])
            if len(targets) < 3 or targets[1] != "take_action":
                continue
            total_action_turns += 1

            if turn_idx in direct_action_preds:
                pred_action, pred_slots = direct_action_preds[turn_idx]
            else:
                pred_action = ""
                pred_slots = []
                for ai in agent_indices:
                    if ai < turn_idx:
                        pred_action = agent_preds[ai]
                        pred_slots = agent_slots[ai]
                    else:
                        break

            if pred_action:
                total_mapped += 1

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

    total_direct = sum(
        1 for r in turn_results
        if r.get("target_type") == "action" and r.get("predicted_action")
    )
    total_parsed = sum(1 for r in turn_results if r.get("predicted_action"))
    print(f"  [AST mapping] {len(predictions)} convs, "
          f"{total_parsed}/{len(turn_results)} parsed actions "
          f"({total_direct} direct action targets), "
          f"{total_agent_preds} agent predictions, "
          f"{total_action_turns} action turns, "
          f"{total_mapped} mapped ({100*total_mapped/max(total_action_turns,1):.0f}%), "
          f"{missing_conversations} convs without generated turns")

    return predictions


def compute_ast_from_turn_results(
    conversations: list[dict],
    turn_results: list[dict],
) -> list[dict]:
    """Compute per-dialogue AST — uses the SAME path as evaluate_abcd().

    Calls turn_results_to_abcd_predictions → compute_ast to guarantee
    consistency with the full evaluation pipeline.
    """
    from .data import extract_ground_truth
    from .metrics import compute_ast as _compute_ast

    abcd_preds = turn_results_to_abcd_predictions(turn_results, conversations)
    scores: list[dict] = []
    for conv, pred in zip(conversations, abcd_preds):
        truths = extract_ground_truth(conv)
        result = _compute_ast(truths, pred, conversation_id=pred.conversation_id)
        scores.append({
            "ast_score": result.joint_accuracy,
            "action_correct": result.joint_correct,
            "action_total": result.num_action_turns,
        })
    return scores
