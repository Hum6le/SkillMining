"""ToD Evaluation Module.

Provides evaluation metrics for Task-oriented Dialogue (ToD) agent outputs:

- **Information Rate**: Slot-level precision — what fraction of goal slots
  (inform + request) were correctly handled.
- **Success Rate**: Binary per-dialogue pass/fail.
- **LLM-as-a-Judge**: Multi-agent LLM-based evaluation system (adapted from
  ``LLMasaJudge``). Uses 5 specialist judges + 1 combiner to score dialogues
  on task_completion, slot_accuracy, dialogue_fluency, helpfulness, efficiency.

Usage::

    from eval_tod import evaluate, load_dataset, load_predictions

    # Quick evaluation
    result = evaluate(
        dataset_name="multiwoz21",
        data_path="data/eval/multiwoz21",
        predictions_path="outputs/predictions.json",
        split="test",
    )
    print(f"Info Rate:  {result['aggregate']['info_rate']:.4f}")
    print(f"Success Rate: {result['aggregate']['success_rate']:.4f}")

    # With LLM Judge
    result = evaluate(
        dataset_name="multiwoz21",
        data_path="data/eval/multiwoz21",
        predictions_path="outputs/predictions.json",
        split="test",
        llm_judge=True,
        llm_model="deepseek-chat",
        llm_judge_sample_size=50,  # optional cost control
    )

Or from the command line::

    python -m eval_tod.cli --dataset multiwoz21 --data_path data/eval/multiwoz21 --predictions preds.json
    python -m eval_tod.cli --dataset multiwoz21 --data_path data/eval/multiwoz21 --predictions preds.json --llm_judge
"""

from .agent import TodPredictionAgent
from .data_loader import (
    load_dataset,
    load_multiwoz21,
    load_predictions,
    list_available_splits,
)
from .evaluate import evaluate, print_summary
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

__all__ = [
    # Main entry points
    "evaluate",
    "print_summary",
    # Agent
    "TodPredictionAgent",
    # Data loading
    "load_dataset",
    "load_multiwoz21",
    "load_predictions",
    "list_available_splits",
    # Metrics
    "DEFAULT_LLM_DIMENSIONS",
    "compute_information_rate",
    "compute_success",
    "compute_dialogue_metrics",
    "compute_aggregate_metrics",
    "llm_judge_evaluate",
    # Schemas
    "Dialogue",
    "Goal",
    "Turn",
    "Prediction",
    "DialogueMetrics",
    "AggregateMetrics",
]
