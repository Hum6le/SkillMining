"""KB-backed ReAct agent for ToD prediction.

Unlike the single-shot ``TodPredictionAgent``, this agent operates in a
ReAct think-act-observe loop:

1. **Think**: the LLM reasons about what information it needs.
2. **Act**: it calls ``query_db(domain, constraints)`` to search the KB.
3. **Observe**: it reads the returned entities.
4. **Repeat** until it has enough information, then outputs predictions.

This mirrors the Trace2Skill agent pattern (``bash`` tool → spreadsheet),
but with a domain-specific ``query_db`` tool.

Architecture::

    goal + ontology
         │
         ▼
    ┌─────────────┐     query_db(hotel, ...)
    │   LLM       │ ──────────────────────→  KB
    │  (ReAct)    │ ←──────────────────────  (entities)
    └─────────────┘     "Found 3 hotels..."
         │
         ▼
    Predictions: {inform_slots, request_slots, booking}
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

from .data_loader import load_multiwoz21
from .kb import MultiWOZKB
from .schemas import Dialogue, Prediction

# ── tool schema for the LLM ────────────────────────────────────

TOOL_DEFINITION = """## Available Tool

You have access to the following tool:

### query_db
Search the knowledge base for entities matching constraints.

Parameters:
- domain (string, required): one of hotel, restaurant, train, taxi, attraction, hospital, police
- constraints (object): {slot_name: desired_value}. Use exact slot names from the ontology.
  Example: {"area": "centre", "price range": "cheap"}

Returns a formatted list of matching entities with all their slot values.

To call the tool, output EXACTLY:
ACTION: query_db
ARGUMENTS: {"domain": "hotel", "constraints": {"area": "centre", "price range": "cheap"}}

The tool response will be appended as an Observation."""

# ── system prompt ───────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a task-oriented dialogue (ToD) agent. You help users find and book services.

## How You Work
1. Read the user's goal carefully — identify what they want (inform constraints) and what they need to know (request slots).
2. Use the `query_db` tool to search for matching entities. You may call it multiple times for different domains or different constraint combinations.
3. When you have enough information, output your final predictions.

## Using query_db
- Call it with the exact slot names from the ontology.
- If you're unsure about a constraint, try a broader search first, then refine.
- For booking tasks (hotel, restaurant, train), you don't need to actually book — just find matching entities and predict what would be informed.

## When to Finish
When you have found relevant entities and determined what should be informed to the user, output:
ACTION: FINAL
ARGUMENTS: {"inform_slots": {...}, "request_slots": {...}, "booking": {...}}

The FINAL output format:
{
  "inform_slots": {
    "hotel": {"name": "Ashley Hotel", "price range": "cheap", "area": "centre"}
  },
  "request_slots": {
    "hotel": ["address", "phone"]
  },
  "booking": {
    "hotel": {"reference": "PLACEHOLDER"}
  }
}

## Rules
- Use EXACT slot names from the ontology.
- For categorical slots, normalise values to match the ontology's allowed values.
- Only inform what the KB actually contains. Do not hallucinate.
- For booking: include a "reference" key with value "PLACEHOLDER" (we don't need real booking codes for evaluation).
- If multiple entities match, pick the first/best one to inform."""


def _build_task_prompt(
    goal_description: str,
    goal_inform: dict,
    goal_request: dict,
    ontology_text: str,
) -> str:
    """Build the initial user message with the goal and ontology."""
    parts = [
        "## User Goal",
        goal_description,
        "",
        "### Goal Constraints (what the user wants — use these to query KB)",
    ]
    for domain, slots in goal_inform.items():
        non_book = {k: v for k, v in slots.items() if not k.startswith("book ")}
        book = {k: v for k, v in slots.items() if k.startswith("book ")}
        if non_book:
            parts.append(f"  [{domain}] inform: {json.dumps(non_book)}")
        if book:
            parts.append(f"  [{domain}] booking needs: {json.dumps(book)}")
    parts.append("")
    parts.append("### Goal Requests (what the user wants to know — infer from goal)")
    for domain, slots in goal_request.items():
        if slots:
            parts.append(f"  [{domain}] request: {list(slots.keys())}")
        else:
            parts.append(f"  [{domain}] request: (any available info)")
    parts.append("")
    parts.append("## Slot Ontology")
    parts.append(ontology_text)
    parts.append("")
    parts.append(TOOL_DEFINITION)
    parts.append("")
    parts.append(
        "Begin by querying the KB for each domain in the goal. "
        "Then output your FINAL predictions."
    )
    return "\n".join(parts)


# ── response parsing ───────────────────────────────────────────

_ACTION_RE = re.compile(r"ACTION:\s*(query_db|FINAL)", re.IGNORECASE)
_ARGS_RE = re.compile(r"ARGUMENTS:\s*(\{.*\})", re.DOTALL)


def _parse_action(response: str) -> tuple[str | None, dict | None]:
    """Extract action type and arguments from LLM response."""
    action_match = _ACTION_RE.search(response)
    if not action_match:
        return None, None
    action_type = action_match.group(1).lower()

    args_match = _ARGS_RE.search(response)
    args = {}
    if args_match:
        try:
            args = json.loads(args_match.group(1))
        except json.JSONDecodeError:
            pass

    return action_type, args


# ── KB tool execution ──────────────────────────────────────────

def _execute_query_db(kb: MultiWOZKB, args: dict) -> str:
    """Execute a query_db call and return formatted results."""
    domain = args.get("domain", "")
    constraints = args.get("constraints", {})

    if domain not in kb.domains:
        return f"Error: unknown domain '{domain}'. Available: {kb.domains}"

    return kb.query_formatted(domain, constraints, max_results=5)


# ── Agent ───────────────────────────────────────────────────────

class ToolBasedTodAgent:
    """ReAct agent that uses a KB query tool to produce predictions.

    The agent loops up to ``max_turns`` times.  Each turn the LLM can
    either call ``query_db`` or output ``FINAL`` predictions.  The KB
    results are fed back as observations.

    Attributes:
        kb: The MultiWOZ knowledge base.
        model: LLM model name.
        max_turns: Max ReAct turns before forced stop.
    """

    def __init__(
        self,
        kb: MultiWOZKB,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
        max_turns: int = 8,
        delay: float = 0.3,
        ontology_path: str | None = None,
        log_dir: str | None = None,
        extra_system_prompt: str = "",
        response_logger = None,
    ):
        self.kb = kb
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_turns = max_turns
        self.delay = delay
        self.ontology_path = ontology_path
        self.log_dir = log_dir
        self.extra_system_prompt = extra_system_prompt  # skill content injected here
        self._response_logger = response_logger

        # Build ontology text for prompt (same filtered approach as TodPredictionAgent)
        from .agent import _load_ontology, _build_ontology_text
        slot_defs = _load_ontology(ontology_path)
        self._ontology_text = _build_ontology_text(slot_defs)

    def predict_single(self, dialogue: Dialogue) -> Prediction:
        """Run the ReAct loop for one dialogue and produce predictions.

        Returns:
            ``Prediction`` with extracted slots.
        """
        from .judge.llm_client import call_llm_structured

        clean_goal = re.sub(r"<span[^>]*>|</span>", "", dialogue.goal.description)
        task_prompt = _build_task_prompt(
            goal_description=clean_goal,
            goal_inform=dialogue.goal.inform,
            goal_request=dialogue.goal.request,
            ontology_text=self._ontology_text,
        )

        # Build system prompt (with optional skill injection)
        system_content = _SYSTEM_PROMPT
        if self.extra_system_prompt:
            system_content = self.extra_system_prompt + "\n\n" + system_content

        # Build message history for multi-turn chat
        messages: list[dict] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": task_prompt},
        ]

        # Trajectory log
        trajectory_lines: list[str] = [
            f"# Trajectory: {dialogue.dialogue_id}",
            f"Domains: {', '.join(dialogue.domains)}",
            f"",
            f"## System Prompt (skill injected: {bool(self.extra_system_prompt)})",
            "```",
            system_content[:3000] + ("..." if len(system_content) > 3000 else ""),
            "```",
            f"",
            f"## Task",
            "```",
            task_prompt[:2000] + ("..." if len(task_prompt) > 2000 else ""),
            "```",
            f"",
        ]

        raw_response = ""
        for turn in range(1, self.max_turns + 1):
            # Call LLM
            try:
                raw_response = self._chat(messages)
            except Exception as exc:
                print(f"    LLM error turn {turn}: {exc}")
                break

            action_type, args = _parse_action(raw_response)

            # Log this turn
            trajectory_lines.append(f"## Turn {turn}")
            trajectory_lines.append(f"")
            trajectory_lines.append(f"### LLM Response")
            trajectory_lines.append("```")
            trajectory_lines.append(raw_response[:1500])
            trajectory_lines.append("```")
            trajectory_lines.append(f"  Parsed action: {action_type}")

            if action_type == "final":
                trajectory_lines.append(f"")
                trajectory_lines.append(f"### Prediction")
                trajectory_lines.append("```json")
                trajectory_lines.append(json.dumps(args, indent=2))
                trajectory_lines.append("```")
                self._save_trajectory(dialogue.dialogue_id, trajectory_lines)
                return Prediction(
                    dialogue_id=dialogue.dialogue_id,
                    inform_slots=args.get("inform_slots", {}),
                    request_slots=args.get("request_slots", {}),
                    booking=args.get("booking", {}),
                )

            elif action_type == "query_db":
                observation = _execute_query_db(self.kb, args)
                trajectory_lines.append(f"")
                trajectory_lines.append(f"### Observation")
                trajectory_lines.append("```")
                trajectory_lines.append(observation[:1000])
                trajectory_lines.append("```")
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": f"Observation:\n{observation}"})

            else:
                trajectory_lines.append(f"  (parse error — reminding LLM)")
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": (
                    "I couldn't parse your action. Please output either:\n"
                    'ACTION: query_db\nARGUMENTS: {"domain": "...", "constraints": {...}}\n\n'
                    "or:\n"
                    'ACTION: FINAL\nARGUMENTS: {"inform_slots": {...}, "request_slots": {...}, "booking": {...}}'
                )})

        # Max turns exceeded
        trajectory_lines.append(f"")
        trajectory_lines.append(f"## Result: MAX TURNS EXCEEDED")
        trajectory_lines.append(f"Last raw response: {raw_response[:500]}")
        self._save_trajectory(dialogue.dialogue_id, trajectory_lines)

        try:
            return Prediction(
                dialogue_id=dialogue.dialogue_id,
                inform_slots=json.loads(
                    re.search(r'"inform_slots"\s*:\s*(\{[^}]+\})', raw_response, re.DOTALL)
                ) if re.search(r'"inform_slots"', raw_response) else {},
            )
        except Exception:
            pass

        return Prediction(dialogue_id=dialogue.dialogue_id)

    def _save_trajectory(self, dialogue_id: str, lines: list[str]) -> None:
        """Save trajectory log to disk if log_dir is configured."""
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        safe_id = dialogue_id.replace("/", "_").replace("\\", "_")
        path = os.path.join(self.log_dir, f"{safe_id}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _chat(self, messages: list[dict]) -> str:
        """Send messages to the LLM and return the response text."""
        from openai import OpenAI

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        url = self.base_url or os.environ.get("OPENAI_BASE_URL", None)
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")

        client_kwargs: dict = {"api_key": key}
        if url:
            client_kwargs["base_url"] = url
        client = OpenAI(**client_kwargs)

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
        )

        # Log raw response if logger is configured
        if self._response_logger is not None:
            try:
                self._response_logger.log(
                    messages=messages,
                    response=resp,
                    call_tag="agent_chat",
                    extra={"model": self.model, "max_tokens": 1024, "temperature": 0.3},
                )
            except Exception:
                pass

        return resp.choices[0].message.content or ""

    def generate_predictions(
        self, dialogues: list[Dialogue], verbose: bool = True,
    ) -> list[Prediction]:
        predictions: list[Prediction] = []
        total = len(dialogues)
        for i, dialogue in enumerate(dialogues):
            if verbose:
                print(f"  [{i+1}/{total}] {dialogue.dialogue_id} ({', '.join(dialogue.domains)})")
            pred = self.predict_single(dialogue)
            predictions.append(pred)
            if i < total - 1:
                time.sleep(self.delay)
        return predictions

    def run_and_save(
        self, dialogues: list[Dialogue], output_path: str, verbose: bool = True,
    ) -> list[Prediction]:
        import json as _json
        preds = self.generate_predictions(dialogues, verbose=verbose)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pred_dicts = [
            {"dialogue_id": p.dialogue_id, "inform_slots": p.inform_slots,
             "request_slots": p.request_slots, "booking": p.booking}
            for p in preds
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            _json.dump(pred_dicts, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to: {output_path} ({len(pred_dicts)} items)")
        return preds


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def build_parser() -> "argparse.ArgumentParser":
    import argparse
    p = argparse.ArgumentParser(description="KB-backed ReAct ToD agent")
    p.add_argument("--data_path", required=True)
    p.add_argument("--db_dir", default="data/eval/multiwoz21/data/data")
    p.add_argument("--output", required=True)
    p.add_argument("--split", default=None, choices=["train", "validation", "test"])
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--max_turns", type=int, default=6)
    p.add_argument("--delay", type=float, default=0.3)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Loading KB from: {args.db_dir}")
    kb = MultiWOZKB(args.db_dir)
    print(f"  Domains: {kb.domains}")
    for d in kb.domains:
        print(f"    {d}: {kb.domain_size(d)} entities")

    print(f"Loading dialogues from: {args.data_path}")
    dialogues = load_multiwoz21(args.data_path, split=args.split)
    end = args.end_idx or len(dialogues)
    dialogues = dialogues[args.start_idx:end]
    print(f"Processing {len(dialogues)} dialogues")

    agent = ToolBasedTodAgent(
        kb=kb, model=args.model, max_turns=args.max_turns, delay=args.delay,
    )
    preds = agent.run_and_save(dialogues, args.output)

    total_inf = sum(sum(len(s) for s in p.inform_slots.values()) for p in preds)
    total_req = sum(sum(len(s) for s in p.request_slots.values()) for p in preds)
    total_bk = sum(1 for p in preds if p.booking)
    print(f"\nSummary: {len(preds)} dialogues, inform={total_inf}, request={total_req}, booking={total_bk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
