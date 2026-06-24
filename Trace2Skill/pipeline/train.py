"""Training iteration: agent run, evaluation, error analysis, skill evolution.

Provides:
- ``_run_training_iteration`` -- one full batch iteration
- ``_run_oneshot_pipeline`` -- original one-shot pipeline (stages 1-6)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .config import PipelineConfig, PipelineResult, _TRACE2SKILL, _PROJECT_ROOT

# Ensure project root + Trace2Skill on sys.path
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_TRACE2SKILL) not in sys.path:
    sys.path.insert(0, str(_TRACE2SKILL))

from llm import get_client, resolve_config


def _run_training_iteration(
    config: PipelineConfig,
    model: str,
    api_key: str,
    base_url: str,
    kb,
    evolved_skills_dir: Path,
    evolved_skill_dir: Path,
    batch: list,
    batch_idx: int,
    out: Path,
    agent=None,
    response_logger=None,
) -> dict:
    """Run one training iteration: agent -> eval -> error analysis -> evolution.

    Args:
        config: Pipeline configuration.
        model, api_key, base_url: LLM settings.
        kb: MultiWOZKB instance.
        evolved_skills_dir: Parent skills directory the agent reads from.
        evolved_skill_dir: Specific skill subdir that gets evolved in-place.
        batch: List of Dialogue objects for this batch.
        batch_idx: 1-based batch number.
        out: Output root directory.
        agent: Optional existing agent to reuse.
        response_logger: Optional ResponseLogger for raw LLM output.

    Returns:
        Dict with keys: metrics, had_failures, changelog, llm_calls.
    """
    from eval_tod.agent_skill import SkillPreloadedAgent
    from eval_tod import evaluate as eval_func
    from eval_tod.error_analysis import ErrorAnalyzer, build_failure_cases
    from eval_tod.schemas import Prediction

    label = f"batch_{batch_idx:04d}"
    batch_preds_dir = out / "batch_predictions"
    batch_evals_dir = out / "batch_evals"
    os.makedirs(batch_preds_dir, exist_ok=True)
    os.makedirs(batch_evals_dir, exist_ok=True)

    # ── 1. Run agent on this batch ──
    log_dir = str(out / "trajectories" / label)
    os.makedirs(log_dir, exist_ok=True)

    if agent is not None:
        agent.skills_dir = str(evolved_skills_dir)
        agent.log_dir = log_dir
        agent.reload_skills()
    else:
        agent = SkillPreloadedAgent(
            kb=kb,
            skills_dir=str(evolved_skills_dir),
            model=model,
            max_turns=config.max_turns,
            log_dir=log_dir,
            api_key=api_key,
            base_url=base_url,
            response_logger=response_logger,
        )

    preds_path = batch_preds_dir / f"{label}.json"
    preds = agent.run_and_save(
        dialogues=batch,
        output_path=str(preds_path),
    )

    # ── 2. Evaluate batch ──
    eval_path = batch_evals_dir / f"{label}.json"
    batch_eval = eval_func(
        dataset_name=config.dataset_name,
        data_path=str(config.resolved_data_path),
        predictions_path=str(preds_path),
        split=config.split,
        output_path=str(eval_path),
        llm_judge=False,  # skip judge per batch for speed
        llm_model=model,
        llm_api_key=api_key,
        llm_base_url=base_url,
    )

    agg = batch_eval["aggregate"]
    metrics = {
        "batch_idx": batch_idx,
        "num_dialogues": len(batch),
        "info_rate": agg["info_rate"],
        "success_rate": agg["success_rate"],
        "num_success": agg.get("num_success", 0),
        "num_fail": agg.get("num_fail", 0),
        "num_failures": agg.get("num_fail", 0),
    }

    # ── 3. Build failure cases ──
    with open(preds_path, "r", encoding="utf-8") as f:
        pred_dicts = json.load(f)
    pred_objs = [Prediction(**p) for p in pred_dicts]

    failed_cases = build_failure_cases(
        batch, pred_objs, batch_eval, log_dir=log_dir,
    )

    if not failed_cases:
        print(f"  Batch {batch_idx}: 0/{len(batch)} failed -- skipping error analysis & evolution")
        return {
            "metrics": metrics,
            "had_failures": False,
            "changelog": [],
            "llm_calls": 0,
        }

    print(f"  Batch {batch_idx}: {len(failed_cases)}/{len(batch)} failed -- analyzing errors...")

    # ── 4. Error analysis ──
    error_dir = str(out / "error_analysis" / label)
    analyzer = ErrorAnalyzer(
        model=model,
        workers=config.workers_analysis,
        api_key=api_key,
        base_url=base_url,
        response_logger=response_logger,
    )
    analyzer.analyze_batch(failed_cases, output_dir=error_dir)

    # ── 5. Parse error analysis ──
    parsed_path = out / "error_analysis" / f"{label}_parsed.json"
    subprocess.run(
        [
            sys.executable,
            str(_TRACE2SKILL / "analysis" / "parse_error_analysis_outputs.py"),
            "--input_dir", error_dir,
            "--output", str(parsed_path),
        ],
        cwd=str(_TRACE2SKILL),
        check=True,
    )

    if not parsed_path.exists():
        print(f"  Warning: parsed error analysis not found, skipping evolution")
        return {
            "metrics": metrics,
            "had_failures": True,
            "changelog": [],
            "llm_calls": 0,
        }

    # ── 6. Skill evolution ──
    from skill_evolver.parallel_evolving_agent import ParallelSkillEvolver

    with open(parsed_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        print(f"  No records parsed -- skipping evolution")
        return {
            "metrics": metrics,
            "had_failures": True,
            "changelog": [],
            "llm_calls": 0,
        }

    evolver_client = get_client(
        model=model, api_key=api_key, base_url=base_url,
        cache_tag=f"evolver_batch_{batch_idx}",
        response_logger=response_logger,
    )

    intermediates_dir = out / "intermediates" / label
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    evo = config.evolution
    evolver = ParallelSkillEvolver(
        client=evolver_client,
        skill_dir=str(evolved_skill_dir),
        batch_size=evo.batch_size,
        merge_batch_size=evo.merge_batch_size,
        max_workers=evo.max_workers,
        max_merge_levels=evo.max_merge_levels,
        temperature=evo.temperature,
        max_tokens=evo.max_tokens,
        verbose=True,
        dry_run=evo.dry_run,
        prompt_variant=evo.prompt_variant,
        output_dir=intermediates_dir,
        parse_failure_dir=out / "parse_failures",
        max_skill_lines=evo.max_skill_lines,
        skip_translation=evo.skip_translation,
        patch_pipeline=evo.patch_pipeline,
    )

    evolver_result = evolver.run(records, input_mode="records")

    # Write changelog
    changelog_entries = evolver_result.get("changelog", [])
    cumulative_patch = evolver_result.get("cumulative_patch", "")
    change_log_path = out / "batch_changelogs" / f"{label}.log"
    os.makedirs(change_log_path.parent, exist_ok=True)
    change_log_lines = [
        f"Change Log -- Batch {batch_idx} (Parallel Evolution):",
        f"MAP patches: {len(evolver_result.get('patches', []))}",
        f"LLM calls: {evolver_result.get('total_llm_calls', 0)}",
        "",
    ]
    if changelog_entries:
        change_log_lines.append("Changes:")
        for entry in changelog_entries:
            change_log_lines.append(f"  - {entry}")
    change_log_lines.append("")
    change_log_lines.append("Overall Diff (final vs original):")
    change_log_lines.append("```diff")
    change_log_lines.append(cumulative_patch)
    change_log_lines.append("```")
    change_log_path.write_text("\n".join(change_log_lines), encoding="utf-8")

    llm_calls = evolver_result.get("total_llm_calls", 0)
    print(f"  Batch {batch_idx}: {len(evolver_result.get('edits', []))} edits applied, "
          f"{llm_calls} LLM calls")

    return {
        "metrics": metrics,
        "had_failures": True,
        "changelog": changelog_entries,
        "llm_calls": llm_calls,
    }


def _run_oneshot_pipeline(
    config: PipelineConfig,
    model: str,
    api_key: str,
    base_url: str,
    out: Path,
    kb,
    dialogues: list,
    split_counts: dict,
    start: int,
    end: int,
    total_in_split: int,
    response_logger=None,
) -> PipelineResult:
    """Run the original one-shot pipeline (stages 1-6) on a single set of dialogues."""
    import shutil
    from eval_tod.agent_skill import SkillPreloadedAgent
    from eval_tod import evaluate as eval_func

    evo = config.evolution

    # ── Stage 1: Run seed agent ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1: Generate predictions with seed skill")
    print("=" * 60)

    log_dir = str(out / "trajectories")
    os.makedirs(log_dir, exist_ok=True)

    agent = SkillPreloadedAgent(
        kb=kb,
        skills_dir=str(config.resolved_skill_dir),
        model=model,
        max_turns=config.max_turns,
        log_dir=log_dir,
        api_key=api_key,
        base_url=base_url,
        response_logger=response_logger,
    )

    preds = agent.run_and_save(
        dialogues=dialogues,
        output_path=str(out / "predictions_seed.json"),
    )

    # ── Stage 2: Evaluate seed ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Evaluate seed skill predictions")
    print("=" * 60)

    seed_eval = eval_func(
        dataset_name=config.dataset_name,
        data_path=str(config.resolved_data_path),
        predictions_path=str(out / "predictions_seed.json"),
        split=config.split,
        output_path=str(out / "eval_seed.json"),
        llm_judge=config.llm_judge,
        llm_model=model,
        llm_judge_sample_size=min(config.llm_judge_sample, len(dialogues))
            if config.llm_judge_sample > 0 else None,
        llm_api_key=api_key,
        llm_base_url=base_url,
    )

    agg = seed_eval["aggregate"]
    print(f"  Seed IR: {agg['info_rate']:.4f}, Success: {agg['success_rate']:.4f}")
    if config.llm_judge and seed_eval.get("llm_judge"):
        js = seed_eval["llm_judge"]
        if js:
            print(f"  Seed Judge: {', '.join(f'{k}={v:.2f}' for k, v in js.items())}")

    # ── Stage 3: Error analysis ─────────────────────────────────
    if config.smoke_test:
        print("\n" + "=" * 60)
        print("STAGE 3-6: SKIPPED (smoke test -- no LLM calls)")
        print("=" * 60)
        failed_cases = []
        evolved_eval = seed_eval
        evolved_skill_dir = Path()
    else:
        print("\n" + "=" * 60)
        print("STAGE 3: Error analysis on failed dialogues")
        print("=" * 60)

        from eval_tod.error_analysis import ErrorAnalyzer, build_failure_cases
        from eval_tod.schemas import Prediction

        with open(out / "predictions_seed.json", "r", encoding="utf-8") as f:
            pred_dicts = json.load(f)
        pred_objs = [Prediction(**p) for p in pred_dicts]

        failed_cases = build_failure_cases(
            dialogues, pred_objs, seed_eval, log_dir=log_dir,
        )
        print(f"  Failed dialogues: {len(failed_cases)}/{len(dialogues)}")

        if failed_cases:
            analyzer = ErrorAnalyzer(
                model=model,
                workers=config.workers_analysis,
                api_key=api_key,
                base_url=base_url,
                response_logger=response_logger,
            )
            error_dir = str(out / "error_analysis")
            analyzer.analyze_batch(failed_cases, output_dir=error_dir)
        else:
            print("  No failures to analyze -- skill is perfect!")

        # ── Stage 4: Parse error analysis ───────────────────────────
        if failed_cases:
            print("\n" + "=" * 60)
            print("STAGE 4: Parse error analysis reports")
            print("=" * 60)

            subprocess.run(
                [
                    sys.executable,
                    str(_TRACE2SKILL / "analysis" / "parse_error_analysis_outputs.py"),
                    "--input_dir", str(out / "error_analysis"),
                    "--output", str(out / "error_analysis_parsed.json"),
                ],
                cwd=str(_TRACE2SKILL),
                check=True,
            )

        # ── Stage 5: Skill evolution ────────────────────────────────
        if failed_cases:
            print("\n" + "=" * 60)
            print("STAGE 5: Skill evolution (MAP -> REDUCE -> APPLY)")
            print("=" * 60)

            evolved_skills_dir = out / "evolved_skills"
            evolved_skills_dir.mkdir(parents=True, exist_ok=True)
            evolved_skill_dir = evolved_skills_dir / config.skill_subdir
            shutil.copytree(
                config.resolved_skill_dir / config.skill_subdir,
                evolved_skill_dir,
                dirs_exist_ok=True,
            )

            from skill_evolver.parallel_evolving_agent import ParallelSkillEvolver

            with open(out / "error_analysis_parsed.json", "r", encoding="utf-8") as f:
                records = json.load(f)

            evolver_client = get_client(
                model=model, api_key=api_key, base_url=base_url,
                cache_tag="evolver",
                response_logger=response_logger,
            )

            intermediates_dir = out / "intermediates"
            intermediates_dir.mkdir(parents=True, exist_ok=True)

            evolver = ParallelSkillEvolver(
                client=evolver_client,
                skill_dir=str(evolved_skill_dir),
                batch_size=evo.batch_size,
                merge_batch_size=evo.merge_batch_size,
                max_workers=evo.max_workers,
                max_merge_levels=evo.max_merge_levels,
                temperature=evo.temperature,
                max_tokens=evo.max_tokens,
                verbose=True,
                dry_run=evo.dry_run,
                prompt_variant=evo.prompt_variant,
                output_dir=intermediates_dir,
                parse_failure_dir=out / "parse_failures",
                max_skill_lines=evo.max_skill_lines,
                skip_translation=evo.skip_translation,
                patch_pipeline=evo.patch_pipeline,
            )

            evolver_result = evolver.run(records, input_mode="records")

            # Write changelog
            changelog_entries = evolver_result.get("changelog", [])
            cumulative_patch = evolver_result.get("cumulative_patch", "")
            change_log_path = out / "change.log"
            change_log_lines = [
                "Change Log (Parallel Evolution):",
                f"MAP patches: {len(evolver_result.get('patches', []))}",
                f"LLM calls: {evolver_result.get('total_llm_calls', 0)}",
                "",
            ]
            if changelog_entries:
                change_log_lines.append("Changes:")
                for entry in changelog_entries:
                    change_log_lines.append(f"  - {entry}")
            change_log_lines.append("")
            change_log_lines.append("Overall Diff (final vs original):")
            change_log_lines.append("```diff")
            change_log_lines.append(cumulative_patch)
            change_log_lines.append("```")
            change_log_path.write_text("\n".join(change_log_lines), encoding="utf-8")

            print(f"\n  Edits applied: {len(evolver_result.get('edits', []))}")
            print(f"  LLM calls:     {evolver_result.get('total_llm_calls', 0)}")

        # ── Stage 6: Re-evaluate with evolved skill ─────────────────
        if failed_cases:
            print("\n" + "=" * 60)
            print("STAGE 6: Evaluate with evolved skill")
            print("=" * 60)

            evolved_agent = SkillPreloadedAgent(
                kb=kb,
                skills_dir=str(evolved_skills_dir),
                model=model,
                max_turns=config.max_turns,
                log_dir=str(out / "trajectories_evolved"),
                api_key=api_key,
                base_url=base_url,
                response_logger=response_logger,
            )

            evolved_preds = evolved_agent.run_and_save(
                dialogues=dialogues,
                output_path=str(out / "predictions_evolved.json"),
            )

            evolved_eval = eval_func(
                dataset_name=config.dataset_name,
                data_path=str(config.resolved_data_path),
                predictions_path=str(out / "predictions_evolved.json"),
                split=config.split,
                output_path=str(out / "eval_evolved.json"),
                llm_judge=config.llm_judge,
                llm_model=model,
                llm_judge_sample_size=min(config.llm_judge_sample, len(dialogues))
                    if config.llm_judge_sample > 0 else None,
                llm_api_key=api_key,
                llm_base_url=base_url,
            )

            agg_ev = evolved_eval["aggregate"]
            print(f"\n  Evolved IR:      {agg_ev['info_rate']:.4f}  (seed: {agg['info_rate']:.4f})")
            print(f"  Evolved Success: {agg_ev['success_rate']:.4f}  (seed: {agg['success_rate']:.4f})")
            if config.llm_judge and evolved_eval.get("llm_judge"):
                js = evolved_eval["llm_judge"]
                if js:
                    print(f"  Evolved Judge:   {', '.join(f'{k}={v:.2f}' for k, v in js.items())}")
        else:
            evolved_eval = seed_eval
            evolved_skill_dir = Path()

    # ── Done ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"Output: {out}")
    if failed_cases:
        print(f"Evolved skill: {evolved_skill_dir}")
    print(f"{'='*60}")

    return PipelineResult(
        output_dir=out,
        seed_predictions_path=out / "predictions_seed.json",
        seed_eval_path=out / "eval_seed.json",
        evolved_predictions_path=out / "predictions_evolved.json" if failed_cases else Path(),
        evolved_eval_path=out / "eval_evolved.json" if failed_cases else Path(),
        evolved_skill_dir=evolved_skill_dir if failed_cases else Path(),
        seed_eval=seed_eval,
        evolved_eval=evolved_eval,
        num_dialogues=len(dialogues),
        num_failed=len(failed_cases),
        had_failures=bool(failed_cases),
    )
