"""SGD adaptation of the ABCD online AWM training protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from awm.memory import MemoryStore, WorkflowStore
from eval_tod.sgd_adapter import gold_policy, score_policies, visible_context


class SGDAWMAgent:
    """Induce a family-level workflow from train rollouts and verified examples.

    The predictor is shared with the standard SGD evaluator; AWM only adds
    training-time induction and retrieved resources to its prompts.
    """

    def __init__(
        self, *, family: str, model: str, ontology: dict, predictor: Callable,
        workflow_max_chars: int = 8000, exemplar_max_chars: int = 3000,
    ) -> None:
        self.family = family
        self.model = model
        self.ontology = ontology
        self.predictor = predictor
        self.workflow = WorkflowStore()
        self.memory = MemoryStore()
        self.workflow_max_chars = workflow_max_chars
        self.exemplar_max_chars = exemplar_max_chars

    def predict(self, dialogue: dict, turn_index: int) -> tuple[dict, dict]:
        context = visible_context(dialogue, turn_index)
        workflow_text = self.workflow.format_prompt(
            query_text=context, max_chars=self.workflow_max_chars,
        )
        exemplar_text = self.memory.format_prompt(
            [self.family], max_chars=self.exemplar_max_chars,
        )
        return self.predictor(
            dialogue, turn_index, model=self.model, ontology=self.ontology,
            frequency_graph=None, workflow_text=workflow_text,
            exemplar_text=exemplar_text,
        )

    @staticmethod
    def _trajectory_row(dialogue: dict, turn_index: int, prediction: dict) -> dict:
        gold = gold_policy(dialogue, turn_index)
        exact = score_policies([gold], [prediction])["turn_joint_ast"] == 1.0
        return {
            "turn_index": turn_index,
            "context": visible_context(dialogue, turn_index)[-1000:],
            "gold": gold,
            "prediction": prediction,
            "joint_correct": exact,
        }

    def train_batch(
        self, dialogues: list[dict], *, batch_index: int,
        trace_path: Path | None = None,
    ) -> dict:
        """Roll out against train turns, update workflow, then add verified memory.

        The resource snapshot is fixed throughout a batch, as in ABCD AWM.
        Gold labels are accessed only after each prediction has been generated.
        """
        from llm import chat

        examples = []
        successful = []
        gold_rows, pred_rows = [], []
        for dialogue in dialogues:
            rows = []
            for turn_index in dialogue["target_indices"]["policy_turn_indices"]:
                prediction, _ = self.predict(dialogue, turn_index)
                row = self._trajectory_row(dialogue, turn_index, prediction)
                rows.append(row)
                gold_rows.append(row["gold"])
                pred_rows.append(prediction)
                if row["joint_correct"]:
                    successful.append((dialogue, row))
            if rows:
                examples.append({
                    "dialogue_id": dialogue["dialogue_id"],
                    "turns": rows,
                })
        if trace_path is not None:
            with trace_path.open("a", encoding="utf-8") as stream:
                for example in examples:
                    for row in example["turns"]:
                        stream.write(json.dumps({
                            "batch_index": batch_index,
                            "dialogue_id": example["dialogue_id"],
                            **row,
                        }, ensure_ascii=False) + "\n")

        # Include both failed predictions and the corresponding train gold, so
        # induction can distinguish a bad branch from a missing slot binding.
        compact = []
        for dialogue in examples:
            rows = dialogue["turns"]
            failures = [row for row in rows if not row["joint_correct"]]
            successes = [row for row in rows if row["joint_correct"]]
            selected = failures[:1] + successes[:1]
            if not selected:
                selected = rows[:1]
            compact.append({
                "dialogue_id": dialogue["dialogue_id"],
                "policy_turns": len(rows),
                "incorrect_turns": len(failures),
                "sampled_turns": [{**row, "context": row["context"][-350:]}
                                  for row in selected],
            })
        prompt = (
            "You maintain an AWM workflow for SGD task-oriented dialogue. "
            "Given a new batch of TRAINING rollouts and gold policy labels, output "
            "the COMPLETE updated workflow, not a diff. Keep valid old patterns; "
            "add missing conditional branches, refine incorrect ones, merge duplicates, "
            "and remove contradicted rules. Aim for 10-20 concise patterns.\n"
            "Each turn's policy is one optional CALL(service, method, parameters) "
            "plus an UNORDERED set of dialogue acts; never linearize simultaneous acts. "
            "Specify when to call an API, how to bind required parameters from prior "
            "user utterances, and what REQUEST/INFORM/OFFER/etc. act set to emit "
            "after the observation. Generalize across service versions by using the "
            "available schema, and never memorize concrete private values. "
            "Do not claim a DB observation is visible before a correct API call.\n\n"
            "Existing workflow:\n" + (self.workflow.text or "(empty)")[:12000]
            + "\n\nNew batch (context is visible before each target turn):\n"
            + json.dumps(compact, ensure_ascii=False)[:30000]
            + "\n\nOutput only the complete workflow as Markdown."
        )
        updated = chat(prompt, model=self.model, temperature=0,
                       call_tag="sgd_awm_induction").strip()
        if updated:
            self.workflow.replace(updated)
        for dialogue, row in successful:
            self.memory.add_dict({
                "dialogue_id": f"sgd-{dialogue['dialogue_id']}-turn-{row['turn_index']}",
                "domains": [self.family],
                "goal": f"{self.family} policy turn",
                "trajectory": json.dumps({
                    "context": row["context"], "verified_policy": row["gold"],
                }, ensure_ascii=False),
            })
        return {
            "batch_index": batch_index,
            "dialogues": len(dialogues),
            "rollout": score_policies(gold_rows, pred_rows),
            "successful_turns_added": len(successful),
            "workflow_lines": len(self.workflow),
            "memory_exemplars": len(self.memory),
            "induced": bool(updated),
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.workflow.save(str(directory / "awm_workflow.txt"))
        self.memory.save(str(directory / "awm_exemplars.json"))

    def load(self, directory: Path) -> None:
        workflow = directory / "awm_workflow.txt"
        memory = directory / "awm_exemplars.json"
        if not workflow.is_file() or not memory.is_file():
            raise FileNotFoundError(f"Missing AWM resources in {directory}")
        self.workflow.load(str(workflow))
        self.memory.load(str(memory))
