"""AWM (Agent Workflow Memory) for MultiWOZ.

Adapted from AWM/mind2web/.  Core idea:
- Memory stores successful dialogue trajectories as exemplars
- Workflow accumulates domain-specific action patterns via LLM induction
- Agent retrieves relevant memories + workflow at inference time

Usage::

    from eval_tod.awm import AWMAgent, MemoryStore
    from eval_tod.kb import MultiWOZKB

    kb = MultiWOZKB("data/eval/multiwoz21/data/data")
    memory = MemoryStore()
    agent = AWMAgent(kb=kb, memory=memory)
    predictions = agent.generate_predictions(dialogues)
"""

from .memory import MemoryStore
from .agent import AWMAgent

__all__ = ["AWMAgent", "MemoryStore"]
