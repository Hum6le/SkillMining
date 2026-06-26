"""AWM Agent for MultiWOZ — adapted from AWM/mind2web/memory.py eval_sample().

Follows the original AWM agent pattern:
1. Load exemplars from MemoryStore (mirrors get_exemplars)
2. Load workflow text from WorkflowStore (mirrors workflow_path)
3. Build prompt: system_message + exemplars + workflow + current_query
4. Run ReAct loop with ToolBasedTodAgent
5. After batch: induce new workflows from trajectories
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# awm/ is at project root; Trace2Skill/ is a sibling
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm import resolve_config
from eval_tod.agent_tool import ToolBasedTodAgent
from eval_tod.schemas import Dialogue, Prediction
from eval_tod.evaluate import AbstractTodAgent
from eval_tod.kb import MultiWOZKB

from .memory import MemoryStore, WorkflowStore


class AWMAgent(AbstractTodAgent):
    """AWM agent for MultiWOZ — mirrors AWM/mind2web/memory.py pattern.

    Architecture (matching original AWM):
    - **MemoryStore**: concrete exemplars (successful trajectories)
    - **WorkflowStore**: LLM-induced workflow patterns (.txt file)
    - **Agent loop**: system_prompt + exemplars + workflow + current_query → LLM

    Pipeline usage (mirrors AWM/mind2web/pipeline.py online mode)::

        agent = AWMAgent(kb=kb)

        for batch_idx, batch in enumerate(batches):
            # 1. Agent runs on batch (using current workflow + exemplars)
            preds = agent.generate_predictions(batch)

            # 2. Evaluate
            result = evaluate_predictions(batch, preds)

            # 3. Induce new workflows from batch (mirrors online_induction.py)
            agent.induce(batch, preds, result["per_dialogue"])

            # 4. Update exemplar memory with successes
            agent.update_memory(batch, preds, result["per_dialogue"])

            # 5. Save
            agent.save_workflow(f"outputs/awm_workflow_step_{batch_idx}.txt")
            agent.save_memory("outputs/awm_exemplars.json")
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
        cfg = resolve_config(api_key=api_key, base_url=base_url, model=model)
        self.model = cfg["model"]
        self.api_key = cfg["api_key"]
        self.base_url = cfg["base_url"]

        self.kb = kb
        self.memory = memory if memory is not None else MemoryStore()
        self.workflow = workflow if workflow is not None else WorkflowStore()
        self.max_turns = max_turns
        self.log_dir = log_dir
        self._response_logger = response_logger

    # ── AbstractTodAgent interface ──────────────────────────────

    def generate_predictions(self, dialogues: list[Dialogue]) -> list[Prediction]:
        """Run agent on dialogues.

        For each dialogue, builds prompt with exemplar few-shots + workflow
        patterns (mirrors eval_sample's sys_message + demo_message + query).
        """
        predictions: list[Prediction] = []
        total = len(dialogues)

        for i, dialogue in enumerate(dialogues):
            has_ex = "yes" if self.memory else "no"
            has_wf = "yes" if self.workflow else "no"
            print(f"  [{i+1}/{total}] {dialogue.dialogue_id} "
                  f"({', '.join(dialogue.domains)})  ex={has_ex} wf={has_wf}")

            # Build enriched prompt (mirrors sys_message + demo_message + query)
            prompt_parts = []
            wf = self.workflow.format_prompt()
            if wf:
                prompt_parts.append(wf)
            ex = self.memory.format_prompt(dialogue.domains)
            if ex:
                prompt_parts.append(ex)
            extra_prompt = "\n".join(prompt_parts).strip()

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

    # ── Memory update (mirrors building exemplars.json) ─────────

    def update_memory(self, dialogues, predictions, eval_results: list[dict]):
        """Store successful dialogues as exemplars."""
        for dm, dialogue, pred in zip(eval_results, dialogues, predictions):
            if dm.get("success"):
                self.memory.add(dialogue, pred)

    def save_memory(self, path: str):
        """Save exemplars to JSON (mirrors exemplars.json)."""
        self.memory.save(path)

    def load_memory(self, path: str):
        """Load exemplars from JSON."""
        self.memory.load(path)

    # ── Workflow induction (mirrors online_induction.py) ────────

    def induce(
        self,
        dialogues,
        predictions,
        eval_results: list[dict],
        trajectory_dir: str | None = None,
    ) -> str:
        """Induce workflows from a batch (mirrors online_induction.py main()).

        Calls the LLM to analyze trajectories and extract workflow patterns,
        then updates self.workflow with the result.
        """
        from .induction import induce_workflows

        pattern = induce_workflows(
            dialogues=dialogues,
            predictions=predictions,
            eval_results=eval_results,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            trajectory_dir=trajectory_dir or self.log_dir,
            existing_workflow=self.workflow.text if self.workflow else "",
        )

        if pattern.strip():
            self.workflow.update(pattern)
            n_lines = len(pattern.splitlines())
            print(f"  [AWM] Induced workflow: {n_lines} lines")

        return pattern

    def save_workflow(self, path: str):
        """Save workflow to .txt file (mirrors workflow_path)."""
        self.workflow.save(path)

    def load_workflow(self, path: str):
        """Load workflow from .txt file."""
        self.workflow.load(path)
