"""AWM Agent for MultiWOZ.

Wraps ``ToolBasedTodAgent`` with AWM memory retrieval.  Before each
dialogue, the agent retrieves relevant past exemplars from the memory
store and injects them into the system prompt as few-shot guidance.

Usage::

    from eval_tod.awm import AWMAgent, MemoryStore
    from eval_tod.kb import MultiWOZKB
    from eval_tod.data import load_multiwoz21
    from eval_tod import evaluate_predictions

    kb = MultiWOZKB("data/eval/multiwoz21/data/data")
    memory = MemoryStore()
    agent = AWMAgent(kb=kb, memory=memory)

    dialogues = load_multiwoz21("data/.../dialogues.json", split="test")
    predictions = agent.generate_predictions(dialogues)
    result = evaluate_predictions(dialogues, predictions)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root importable
_TRACE2SKILL = Path(__file__).resolve().parent.parent.parent / "Trace2Skill"
_PROJECT_ROOT = _TRACE2SKILL.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm import resolve_config
from eval_tod.agent_tool import ToolBasedTodAgent
from eval_tod.schemas import Dialogue, Prediction
from eval_tod.evaluate import AbstractTodAgent
from eval_tod.kb import MultiWOZKB

from .memory import MemoryStore, WorkflowStore


class AWMAgent(AbstractTodAgent):
    """AWM-enhanced ToD agent for MultiWOZ.

    Extends the ReAct loop with:
    1. **Memory retrieval** — fetches similar past successful dialogues
       from ``MemoryStore`` and injects them as few-shot exemplars.
    2. **Workflow injection** — appends accumulated workflow patterns
       (action heuristics) into the system prompt.

    The underlying agent loop is still ``ToolBasedTodAgent``; AWM only
    enriches the prompt.

    Attributes:
        kb: MultiWOZ knowledge base.
        memory: ``MemoryStore`` with past exemplars.
        workflow: ``WorkflowStore`` with accumulated patterns.
        model: LLM model name.
        max_turns: Max ReAct loop turns per dialogue.
    """

    def __init__(
        self,
        kb: MultiWOZKB,
        memory: MemoryStore | None = None,
        workflow: WorkflowStore | None = None,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
        max_turns: int = 8,
        log_dir: str | None = None,
        response_logger=None,
    ):
        # Resolve API credentials
        cfg = resolve_config(api_key=api_key, base_url=base_url, model=model)
        self.model = cfg["model"]
        self.api_key = cfg["api_key"]
        self.base_url = cfg["base_url"]

        self.kb = kb
        self.memory = memory or MemoryStore()
        self.workflow = workflow or WorkflowStore()
        self.max_turns = max_turns
        self.log_dir = log_dir
        self._response_logger = response_logger

    # ── AbstractTodAgent interface ──────────────────────────────

    def generate_predictions(self, dialogues: list[Dialogue]) -> list[Prediction]:
        """Run AWM agent on a list of dialogues.

        For each dialogue:
        1. Retrieve relevant exemplars from memory
        2. Build an enriched system prompt (memory + workflow)
        3. Run the ReAct agent loop
        4. On success, add the dialogue to memory
        """
        predictions: list[Prediction] = []
        total = len(dialogues)

        for i, dialogue in enumerate(dialogues):
            print(f"  [{i+1}/{total}] {dialogue.dialogue_id} "
                  f"({', '.join(dialogue.domains)})  mem={len(self.memory)}")

            # Build enriched prompt
            memory_prompt = self.memory.format_prompt(dialogue.domains)
            workflow_prompt = self.workflow.format_prompt()
            extra_prompt = (workflow_prompt + memory_prompt).strip()

            # Create agent with AWM-enriched prompt
            agent = ToolBasedTodAgent(
                kb=self.kb,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                max_turns=self.max_turns,
                log_dir=self.log_dir,
                extra_system_prompt=extra_prompt,
                response_logger=self._response_logger,
            )

            pred = agent.predict_single(dialogue)
            predictions.append(pred)

        return predictions

    # ── Memory update ───────────────────────────────────────────

    def update_memory(
        self,
        dialogues: list[Dialogue],
        predictions: list[Prediction],
        eval_results: dict,
    ):
        """Add successful dialogues to memory.

        Iterates through per-dialogue evaluation results and stores
        exemplars for any dialogue that succeeded.

        Args:
            dialogues: Ground-truth dialogues.
            predictions: Agent predictions.
            eval_results: Dict from ``evaluate_predictions()`` with
                          ``per_dialogue`` key.
        """
        per_dialogue = eval_results.get("per_dialogue", [])
        for dm, dialogue, pred in zip(per_dialogue, dialogues, predictions):
            if dm.get("success"):
                self.memory.add(dialogue, pred)

    def save_memory(self, path: str):
        """Persist memory to disk."""
        self.memory.save(path)

    def load_memory(self, path: str):
        """Load memory from disk."""
        self.memory.load(path)
