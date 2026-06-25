"""ToD Evaluation Module.

Two-layer API:

**Core** (framework-agnostic, any agent):
  - ``evaluate_predictions(dialogues, predictions)`` -- the main entry point
  - ``load_dataset`` / ``load_and_split`` -- data loading
  - ``AbstractTodAgent`` -- agent interface to implement

**Convenience** (file-based, quick eval):
  - ``evaluate(dataset_name, data_path, predictions_path)``

Quick start::

    from eval_tod import evaluate_predictions
    from eval_tod.data import load_dataset

    dialogues = load_dataset("multiwoz21", data_path, split="test")
    predictions = my_agent.generate_predictions(dialogues)
    result = evaluate_predictions(dialogues, predictions)
    print(result["aggregate"]["info_rate"])

Plug in your own agent::

    from eval_tod import AbstractTodAgent

    class MyAgent(AbstractTodAgent):
        def generate_predictions(self, dialogues):
            ...  # your logic here

    result = evaluate_predictions(dialogues, MyAgent().generate_predictions(dialogues))
"""

from .data import (
    build_batches,
    list_available_splits,
    load_and_split,
    load_dataset,
    load_multiwoz21,
    load_predictions,
    register_loader,
    split_train_val,
)
from .evaluate import (
    AbstractTodAgent,
    evaluate,
    evaluate_predictions,
    print_summary,
)
from .metrics import (
    DEFAULT_LLM_DIMENSIONS,
    compute_aggregate_metrics,
    compute_dialogue_metrics,
    compute_information_rate,
    compute_success,
    llm_judge_evaluate,
)
from .schemas import (
    AggregateMetrics,
    Dialogue,
    DialogueMetrics,
    Goal,
    Prediction,
    Turn,
)

from .agent import TodPredictionAgent
from .agent_skill import SkillPreloadedAgent
from .kb import MultiWOZKB

__all__ = [
    # ── Core evaluation ──
    "evaluate_predictions",       # generic: (dialogues, preds) -> results
    "evaluate",                   # convenience: loads from files
    "print_summary",
    # ── Agent interface ──
    "AbstractTodAgent",           # ABC to implement for new methods
    "TodPredictionAgent",
    "SkillPreloadedAgent",
    # ── Data ──
    "load_dataset",
    "load_multiwoz21",
    "load_predictions",
    "list_available_splits",
    "register_loader",
    "load_and_split",
    "split_train_val",
    "build_batches",
    # ── Metrics ──
    "compute_information_rate",
    "compute_success",
    "compute_dialogue_metrics",
    "compute_aggregate_metrics",
    "llm_judge_evaluate",
    "DEFAULT_LLM_DIMENSIONS",
    # ── KB ──
    "MultiWOZKB",
    # ── Schemas ──
    "Dialogue",
    "Goal",
    "Turn",
    "Prediction",
    "DialogueMetrics",
    "AggregateMetrics",
]
