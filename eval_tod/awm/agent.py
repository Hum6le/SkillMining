"""AWM Agent for MultiWOZ.

Wraps ``ToolBasedTodAgent`` with AWM workflow injection.  After each
batch of dialogues, calls ``induce_workflows()`` to extract patterns
from the trajectories and appends them to a ``WorkflowStore``.

Usage::

    from eval_tod.awm import AWMAgent, WorkflowStore
    from eval_tod.kb import MultiWOZKB

    kb = MultiWOZKB("data/eval/multiwoz21/data/data")
    workflow = WorkflowStore()
    agent = AWMAgent(kb=kb, workflow=workflow)

    # Run a batch
    predictions = agent.generate_predictions(dialogues)

    # Induce workflows from this batch
    agent.induce(dialogues, predictions, eval_results)
    workflow.save("outputs/awm_workflow.txt")
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

from .memory import WorkflowStore


class AWMAgent(AbstractTodAgent):
    """AWM-enhanced ToD agent for MultiWOZ.

    Core AWM loop (per batch):
    1. Inject current workflow into agent's system prompt
    2. Run agent on the batch
    3. Evaluate results
    4. Call ``induce_workflows()`` to extract new patterns from trajectories
    5. Append to WorkflowStore for future batches

    Attributes:
        kb: MultiWOZ knowledge base.
        workflow: ``WorkflowStore`` with accumulated patterns.
        model: LLM model name.
        max_turns: Max ReAct loop turns per dialogue.
    """

    def __init__(
        self,
        kb: MultiWOZKB,
        workflow: WorkflowStore | None = None,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
        max_turns: int = 8,
        log_dir: str | None = None,
        response_logger=None,
    ):
        cfg = resolve_config(api_key=api_key, base_url=base_url, model=model)
        self.model = cfg["model"]
        self.api_key = cfg["api_key"]
        self.base_url = cfg["base_url"]

        self.kb = kb
        self.workflow = workflow if workflow is not None else WorkflowStore()
        self.max_turns = max_turns
        self.log_dir = log_dir
        self._response_logger = response_logger

    # ── AbstractTodAgent interface ──────────────────────────────

    def generate_predictions(self, dialogues: list[Dialogue]) -> list[Prediction]:
        """Run AWM agent on a list of dialogues.

        For each dialogue, injects the current workflow into the
        system prompt before running the ReAct loop.
        """
        predictions: list[Prediction] = []
        total = len(dialogues)
        workflow_text = self.workflow.format_prompt()

        for i, dialogue in enumerate(dialogues):
            has_wf = "yes" if self.workflow else "no"
            print(f"  [{i+1}/{total}] {dialogue.dialogue_id} "
                  f"({', '.join(dialogue.domains)})  wf={has_wf}")

            agent = ToolBasedTodAgent(
                kb=self.kb,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                max_turns=self.max_turns,
                log_dir=self.log_dir,
                extra_system_prompt=workflow_text,
                response_logger=self._response_logger,
            )

            pred = agent.predict_single(dialogue)
            predictions.append(pred)

        return predictions

    # ── AWM induction ───────────────────────────────────────────

    def induce(
        self,
        dialogues: list[Dialogue],
        predictions: list[Prediction],
        eval_results: list[dict],
        trajectory_dir: str | None = None,
    ) -> str:
        """Induce workflow patterns from a batch and update the store.

        Calls the LLM to analyze this batch's trajectories, extracts
        workflow patterns, and appends them to ``self.workflow``.

        Args:
            dialogues: Ground-truth dialogues.
            predictions: Agent predictions.
            eval_results: Per-dialogue metrics from evaluate_predictions().
            trajectory_dir: Directory with agent trajectory .md files.

        Returns:
            The induced pattern text (empty string on failure).
        """
        from .induction import induce_workflows

        pattern = induce_workflows(
            dialogues=dialogues,
            predictions=predictions,
            eval_results=eval_results,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            trajectory_dir=trajectory_dir,
            existing_workflow=self.workflow.text,
        )

        if pattern.strip():
            self.workflow.update(pattern)
            print(f"  [AWM] Induced {len(pattern.splitlines())} lines of workflow patterns")

        return pattern

    def save_workflow(self, path: str):
        self.workflow.save(path)

    def load_workflow(self, path: str):
        self.workflow.load(path)
