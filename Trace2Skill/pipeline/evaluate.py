"""Pipeline evaluation utilities.

Provides ``_run_validation`` -- run agent on a validation/test set
and produce metrics.
"""

from __future__ import annotations

import os

from .config import PipelineConfig


def _run_validation(
    config: PipelineConfig,
    model: str,
    api_key: str,
    base_url: str,
    kb,
    skills_dir: str,
    dialogues: list,
    out,
    label: str,
    agent=None,
    response_logger=None,
) -> dict:
    """Run evaluation on a validation or test set.

    Args:
        config: Pipeline configuration.
        model, api_key, base_url: LLM settings.
        kb: MultiWOZKB instance.
        skills_dir: Path to skill directory to evaluate.
        dialogues: List of Dialogue objects.
        out: Output root directory.
        label: Label prefix for output files (e.g. "val_step_0005").
        agent: Optional existing agent to reuse (will reload skills).
        response_logger: Optional ResponseLogger for raw LLM output.

    Returns:
        Dict with keys: metrics, eval_result, predictions_path, eval_path.
    """
    from eval_tod.agent_skill import SkillPreloadedAgent
    from eval_tod import evaluate as eval_func

    preds_dir = out / "val_predictions"
    evals_dir = out / "val_evals"
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(evals_dir, exist_ok=True)

    preds_path = preds_dir / f"{label}.json"
    eval_path = evals_dir / f"{label}.json"

    if agent is not None:
        agent.skills_dir = skills_dir
        agent.log_dir = str(out / "val_trajectories" / label)
        os.makedirs(agent.log_dir, exist_ok=True)
        agent.reload_skills()
    else:
        agent = SkillPreloadedAgent(
            kb=kb,
            skills_dir=skills_dir,
            model=model,
            max_turns=config.max_turns,
            log_dir=str(out / "val_trajectories" / label),
            api_key=api_key,
            base_url=base_url,
            response_logger=response_logger,
        )

    preds = agent.run_and_save(
        dialogues=dialogues,
        output_path=str(preds_path),
    )

    eval_result = eval_func(
        dataset_name=config.dataset_name,
        data_path=str(config.resolved_data_path),
        predictions_path=str(preds_path),
        split=config.val_split or config.split,
        output_path=str(eval_path),
        llm_judge=config.llm_judge,
        llm_model=model,
        llm_judge_sample_size=min(config.llm_judge_sample, len(dialogues))
            if config.llm_judge_sample > 0 else None,
        llm_api_key=api_key,
        llm_base_url=base_url,
    )

    agg = eval_result["aggregate"]
    metrics = {
        "label": label,
        "num_dialogues": len(dialogues),
        "info_rate": agg["info_rate"],
        "success_rate": agg["success_rate"],
        "num_success": agg.get("num_success", 0),
        "num_fail": agg.get("num_fail", 0),
    }
    if eval_result.get("llm_judge"):
        metrics["llm_judge"] = eval_result["llm_judge"]

    return {
        "metrics": metrics,
        "eval_result": eval_result,
        "predictions_path": str(preds_path),
        "eval_path": str(eval_path),
    }
