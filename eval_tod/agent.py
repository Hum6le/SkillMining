"""Simplest ToD prediction agent.

A single-LLM-call agent that reads a complete task-oriented dialogue (goal +
turns) and extracts structured predictions: what slots were informed, what
slots were requested, and any booking reference numbers.

Usage::

    from eval_tod.agent import TodPredictionAgent
    from eval_tod.data_loader import load_multiwoz21

    dialogues = load_multiwoz21("data/eval/multiwoz21", split="test")
    agent = TodPredictionAgent(model="deepseek-chat")
    predictions = agent.generate_predictions(dialogues)
    agent.save_predictions(predictions, "outputs/preds.json")
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

from .data import load_multiwoz21, load_dataset
from .judge.llm_client import call_llm_structured
from .schemas import Dialogue, Prediction


# ═══════════════════════════════════════════════════════════════════
# Goal-relevant slot whitelist
# ═══════════════════════════════════════════════════════════════════

# Slots that appear in goal.inform across all MultiWOZ dialogues.
_GOAL_INFORM_SLOTS: dict[str, set[str]] = {
    "attraction": {"area", "name", "type"},
    "hospital":   {"department"},
    "hotel":      {"area", "internet", "name", "parking", "price range", "stars", "type"},
    "restaurant": {"area", "food", "name", "price range"},
    "taxi":       {"arrive by", "departure", "destination", "leave at"},
    "train":      {"arrive by", "day", "departure", "destination", "leave at"},
}

# Slots that appear in goal.request across all MultiWOZ dialogues.
_GOAL_REQUEST_SLOTS: dict[str, set[str]] = {
    "attraction": {"address", "area", "entrance fee", "phone", "postcode", "type"},
    "hospital":   {"address", "phone", "postcode"},
    "hotel":      {"address", "area", "internet", "parking", "phone", "postcode", "price range", "stars", "type"},
    "police":     {"address", "phone", "postcode"},
    "restaurant": {"address", "area", "food", "phone", "postcode", "price range"},
    "taxi":       {"phone", "type"},
    "train":      {"arrive by", "duration", "leave at", "price", "train id"},
}

# Booking sub-slots that appear in goal.inform.
_BOOKING_SUB_SLOTS: dict[str, set[str]] = {
    "hotel":      {"book day", "book people", "book stay"},
    "restaurant": {"book day", "book people", "book time"},
    "train":      {"book people"},
}


# ═══════════════════════════════════════════════════════════════════
# Ontology loader
# ═══════════════════════════════════════════════════════════════════

def _load_ontology(path: str | None = None) -> dict[str, dict[str, dict]]:
    """Load ontology, keeping only slots that appear in goals."""
    if path is None:
        candidates = [
            "data/eval/multiwoz21/data/data/ontology.json",
            os.path.join(os.path.dirname(__file__), "..", "data", "eval",
                         "multiwoz21", "data", "data", "ontology.json"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    slot_defs: dict[str, dict[str, dict]] = {}
    for domain, info in raw.get("domains", {}).items():
        all_slots = info.get("slots", {})
        keep = (_GOAL_INFORM_SLOTS.get(domain, set())
                | _GOAL_REQUEST_SLOTS.get(domain, set())
                | _BOOKING_SUB_SLOTS.get(domain, set()))

        filtered = {}
        for sname, sinfo in all_slots.items():
            if sname in keep:
                filtered[sname] = dict(sinfo)

        if filtered:
            slot_defs[domain] = filtered

    return slot_defs


# ═══════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════

_PREAMBLE = (
    "You are a dialogue analysis specialist. Read the task-oriented dialogue "
    "and extract structured information about what the SYSTEM communicated.\n\n"
    "## Slot Reference (use these EXACT names)\n"
    "[categorical] slots have fixed allowed values — normalize to match one.\n\n"
)

_TASK = (
    "\n## Your Task\n\n"
    "### 1. inform_slots\n"
    "Slot-value pairs the SYSTEM explicitly stated. Use only names from the "
    "reference above. For categorical slots, match allowed values exactly.\n\n"
    "### 2. request_slots\n"
    "Slot names the SYSTEM asked the user for "
    "(e.g. \"what area?\" -> \"area\", \"how many people?\" -> \"book people\").\n\n"
    "### 3. booking\n"
    "For each domain with a successful booking, include {\"reference\": \"CODE\"} "
    "using the booking code the system gave (e.g. \"7GAWK763\").\n\n"
    "## Output Format\n"
    '{"inform_slots": {"hotel": {"name": "...", "price range": "cheap"}}, '
    '"request_slots": {"hotel": ["area"]}, '
    '"booking": {"hotel": {"reference": "ABC123"}}}\n\n'
    "## Rules\n"
    "- Use ONLY slot names from the reference. No invented names.\n"
    "- For categorical slots, output lowercase values from the allowed list.\n"
    "- Only what the system ACTUALLY said. Better to miss than hallucinate.\n"
    "- Omit domains that have no inform/request slots.\n"
)


def _build_ontology_text(slot_defs: dict[str, dict[str, dict]]) -> str:
    """Build a compact slot reference."""
    lines: list[str] = []
    for domain in sorted(slot_defs):
        slots = slot_defs[domain]
        lines.append(f"### {domain}")
        for sname, info in sorted(slots.items()):
            desc = info.get("description", "")
            vals = info.get("possible_values", [])
            if vals:
                vlist = ", ".join(str(v) for v in vals)
                lines.append(f"  {sname}: {desc}  [allowed: {vlist}]")
            else:
                lines.append(f"  {sname}: {desc}  [free-text]")
        lines.append("")
    return "\n".join(lines)


def build_system_prompt(slot_defs: dict[str, dict[str, dict]]) -> str:
    """Build the complete system prompt with ontology."""
    return _PREAMBLE + _build_ontology_text(slot_defs) + _TASK


# ═══════════════════════════════════════════════════════════════════
# User message builder
# ═══════════════════════════════════════════════════════════════════

def _build_user_message(dialogue: Dialogue) -> str:
    """Build the user message containing the dialogue to analyze."""
    clean_goal = re.sub(r"<span[^>]*>|</span>", "", dialogue.goal.description)

    turns_text = []
    for turn in dialogue.turns:
        speaker = "USER" if turn.speaker == "user" else "SYSTEM"
        turns_text.append(f"[{speaker}] {turn.utterance}")

    return "\n".join([
        "## User Goal",
        clean_goal,
        "",
        "## Dialogue",
        "\n".join(turns_text),
        "",
        "## Instruction",
        "Extract what the SYSTEM communicated. Return ONLY the JSON object.",
    ])


# ═══════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════

class TodPredictionAgent:
    """Single-call LLM agent that extracts structured predictions from dialogues.

    Loads a filtered MultiWOZ ontology (only goal-relevant slots) and injects
    it into the system prompt.  The prompt includes exact slot names and
    allowed values for categorical slots.

    Attributes:
        model: LLM model name.
        delay: Seconds between API calls.
        slot_defs: Filtered ontology slot definitions.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
        delay: float = 0.3,
        ontology_path: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.delay = delay
        self.slot_defs = _load_ontology(ontology_path)
        self._system_prompt = build_system_prompt(self.slot_defs)

    def predict_single(self, dialogue: Dialogue) -> Prediction:
        user_message = _build_user_message(dialogue)
        try:
            raw = call_llm_structured(
                system_prompt=self._system_prompt,
                user_message=user_message,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except Exception as exc:
            print(f"  ERROR [{dialogue.dialogue_id}]: {exc}")
            raw = {}
        return Prediction(
            dialogue_id=dialogue.dialogue_id,
            inform_slots=raw.get("inform_slots", {}),
            request_slots=raw.get("request_slots", {}),
            booking=raw.get("booking", {}),
        )

    def generate_predictions(
        self, dialogues: list[Dialogue], verbose: bool = True,
    ) -> list[Prediction]:
        predictions: list[Prediction] = []
        total = len(dialogues)
        for i, dialogue in enumerate(dialogues):
            if verbose:
                print(f"  [{i+1}/{total}] {dialogue.dialogue_id} "
                      f"({', '.join(dialogue.domains)})")
            predictions.append(self.predict_single(dialogue))
            if i < total - 1:
                time.sleep(self.delay)
        return predictions

    def run_and_save(
        self, dialogues: list[Dialogue], output_path: str, verbose: bool = True,
    ) -> list[Prediction]:
        predictions = self.generate_predictions(dialogues, verbose=verbose)
        self.save_predictions(predictions, output_path)
        return predictions

    @staticmethod
    def save_predictions(predictions: list[Prediction], output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pred_dicts = [
            {"dialogue_id": p.dialogue_id, "inform_slots": p.inform_slots,
             "request_slots": p.request_slots, "booking": p.booking}
            for p in predictions
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pred_dicts, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to: {output_path} ({len(pred_dicts)} items)")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def build_parser() -> "argparse.ArgumentParser":
    import argparse
    p = argparse.ArgumentParser(description="Generate ToD predictions")
    p.add_argument("--data_path", required=True, help="MultiWOZ data dir or JSON file")
    p.add_argument("--output", required=True, help="Path to save predictions JSON")
    p.add_argument("--split", default=None, choices=["train", "validation", "test"])
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--delay", type=float, default=0.3)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Loading dialogues from: {args.data_path}")
    dialogues = load_multiwoz21(args.data_path, split=args.split)
    end = args.end_idx or len(dialogues)
    dialogues = dialogues[args.start_idx:end]
    print(f"Processing {len(dialogues)} dialogues ({args.start_idx}:{end})")

    agent = TodPredictionAgent(model=args.model, delay=args.delay)
    preds = agent.run_and_save(dialogues, args.output)

    total_inf = sum(sum(len(s) for s in p.inform_slots.values()) for p in preds)
    total_req = sum(sum(len(s) for s in p.request_slots.values()) for p in preds)
    total_bk  = sum(1 for p in preds if p.booking)
    print(f"\nSummary: {len(preds)} dialogues, "
          f"inform={total_inf}, request={total_req}, booking={total_bk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
