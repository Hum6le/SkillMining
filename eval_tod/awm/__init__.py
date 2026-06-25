"""AWM (Agent Workflow Memory) for MultiWOZ — adapted from AWM/mind2web/.

Two modes (mirroring AWM/mind2web/pipeline.py):
- **offline**: pre-induce workflows from training data, then evaluate
- **online**: interleave batch inference + workflow induction

Core components:
- ``AWMAgent`` — ReAct agent with workflow/exemplar injection
- ``MemoryStore`` — stores concrete exemplars (successful dialogue trajectories)
- ``WorkflowStore`` — accumulates LLM-induced workflow patterns
- ``induce_workflows()`` — LLM analyzes trajectories → extracts patterns
"""

from .memory import MemoryStore, WorkflowStore
from .agent import AWMAgent
from .induction import induce_workflows

__all__ = ["AWMAgent", "MemoryStore", "WorkflowStore", "induce_workflows"]
