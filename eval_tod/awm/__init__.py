"""AWM (Agent Workflow Memory) for MultiWOZ.

Core AWM loop:
1. Agent runs on a batch of dialogues
2. Evaluate results
3. ``induce_workflows()`` calls LLM to extract workflow patterns from trajectories
4. Patterns accumulated in ``WorkflowStore``
5. Next batch: workflow injected into agent's system prompt

Usage::

    from eval_tod.awm import AWMAgent, WorkflowStore
    from eval_tod.kb import MultiWOZKB

    kb = MultiWOZKB("data/eval/multiwoz21/data/data")
    workflow = WorkflowStore()
    agent = AWMAgent(kb=kb, workflow=workflow)

    # Batch loop
    for batch in batches:
        preds = agent.generate_predictions(batch)
        result = evaluate_predictions(batch, preds)
        agent.induce(batch, preds, result["per_dialogue"])
        agent.save_workflow("outputs/awm_workflow.txt")
"""

from .memory import WorkflowStore
from .agent import AWMAgent
from .induction import induce_workflows

__all__ = ["AWMAgent", "WorkflowStore", "induce_workflows"]
