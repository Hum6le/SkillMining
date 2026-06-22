#!/usr/bin/env python3
"""ToD Skill Evolution Pipeline — single entry point.

Usage:
    # From project root:
    python -m Trace2Skill.pipeline_tod

    # As a library:
    from Trace2Skill.pipeline_tod import PipelineConfig, run_pipeline
    result = run_pipeline(PipelineConfig(
        skill_dir="eval_tod/skills",
        data_path="data/eval/multiwoz21/dummy_data.json",
        db_dir="data/eval/multiwoz21/data/data",
        model="deepseek-chat",
        api_key="sk-...",
        base_url="https://api.deepseek.com",
    ))
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
_TRACE2SKILL = Path(__file__).resolve().parent
_PROJECT_ROOT = _TRACE2SKILL.parent

# Make project root importable (for eval_tod and llm modules)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_TRACE2SKILL) not in sys.path:
    sys.path.insert(0, str(_TRACE2SKILL))

from llm import get_client, resolve_config


# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════


@dataclass
class EvolutionConfig:
    """Settings for the skill evolution phase (MAP → REDUCE → APPLY)."""

    batch_size: int = 1
    merge_batch_size: int = 5
    max_workers: int = 4
    max_merge_levels: int = 5
    temperature: float = 0.3
    max_tokens: int | None = None
    max_skill_lines: int = 500
    max_verification_rounds: int = 3
    patch_pipeline: str = "json"  # "json" or "markdown"
    prompt_variant: str = "generic"
    skip_translation: bool = False
    dry_run: bool = False
    # Per-phase model overrides (None = use main model)
    map_model: str | None = None
    merge_model: str | None = None
    translation_model: str | None = None


@dataclass
class PipelineConfig:
    """Full pipeline configuration.

    All relative paths are resolved against the project root
    (parent of Trace2Skill/).
    """

    # Paths (relative to project root, or absolute)
    skill_dir: str = "eval_tod/skills"  # parent dir containing skill subdirs
    data_path: str = "data/eval/multiwoz21/dummy_data.json"
    db_dir: str = "data/eval/multiwoz21/data/data"
    output_dir: str = "outputs/tod_pipeline"

    # Dataset
    split: str | None = None  # "train", "validation", "test", or None for all
    start: int = 0
    end: int = 10
    seed: int = 41

    # Agent
    model: str = "deepseek-chat"
    api_key: str | None = None  # None = resolve from config / env
    base_url: str | None = None  # None = resolve from config / env
    max_turns: int = 6
    workers_agent: int = 1
    workers_analysis: int = 4

    # LLM Judge
    llm_judge: bool = True
    llm_judge_sample: int = 5

    # Evolution
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    # ── derived paths ──
    @property
    def skill_subdir(self) -> str:
        return "tod"

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    @property
    def resolved_skill_dir(self) -> Path:
        return self._resolve(self.skill_dir)

    @property
    def resolved_output_dir(self) -> Path:
        return self._resolve(self.output_dir)

    @property
    def resolved_data_path(self) -> Path:
        return self._resolve(self.data_path)

    @property
    def resolved_db_dir(self) -> Path:
        return self._resolve(self.db_dir)


# ══════════════════════════════════════════════════════════════════
# Pipeline result
# ══════════════════════════════════════════════════════════════════


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""

    output_dir: Path
    seed_predictions_path: Path
    seed_eval_path: Path
    evolved_predictions_path: Path
    evolved_eval_path: Path
    evolved_skill_dir: Path
    seed_eval: dict
    evolved_eval: dict
    num_dialogues: int
    num_failed: int
    had_failures: bool


# ══════════════════════════════════════════════════════════════════
# Main pipeline entry point
# ══════════════════════════════════════════════════════════════════


def run_pipeline(config: PipelineConfig | None = None, **kwargs) -> PipelineResult:
    """Run the full skill evolution pipeline.

    Args:
        config: PipelineConfig with all settings. If None, defaults are used.
        **kwargs: Override individual config fields (e.g. model="gpt-4o").

    Returns:
        PipelineResult with paths to all outputs and evaluation summaries.
    """
    if config is None:
        config = PipelineConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # ── Resolve API config ──────────────────────────────────────
    cfg = resolve_config(
        api_key=config.api_key, base_url=config.base_url, model=config.model,
    )
    model, api_key, base_url = cfg["model"], cfg["api_key"], cfg["base_url"]

    evo = config.evolution

    # ── Prepare output directory ────────────────────────────────
    out = config.resolved_output_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Run seed agent ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1: Generate predictions with seed skill")
    print("=" * 60)

    from eval_tod.kb import MultiWOZKB
    from eval_tod.agent_skill import SkillPreloadedAgent
    from eval_tod.data_loader import load_multiwoz21

    kb = MultiWOZKB(str(config.resolved_db_dir))
    print(f"KB loaded: {kb.domains}")

    dialogues = load_multiwoz21(
        str(config.resolved_data_path), split=config.split,
    )
    dialogues = dialogues[config.start : config.end]
    print(f"Dialogues: {len(dialogues)}")

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
    )

    preds = agent.run_and_save(
        dialogues=dialogues,
        output_path=str(out / "predictions_seed.json"),
    )

    # ── Stage 2: Evaluate seed ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Evaluate seed skill predictions")
    print("=" * 60)

    from eval_tod import evaluate as eval_func

    seed_eval = eval_func(
        dataset_name="multiwoz21",
        data_path=str(config.resolved_data_path),
        predictions_path=str(out / "predictions_seed.json"),
        split=config.split,
        output_path=str(out / "eval_seed.json"),
        llm_judge=config.llm_judge,
        llm_model=model,
        llm_judge_sample_size=min(config.llm_judge_sample, len(dialogues)),
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
        )
        error_dir = str(out / "error_analysis")
        analyzer.analyze_batch(failed_cases, output_dir=error_dir)
    else:
        print("  No failures to analyze — skill is perfect!")

    # ── Stage 4: Parse error analysis ───────────────────────────
    if failed_cases:
        print("\n" + "=" * 60)
        print("STAGE 4: Parse error analysis reports")
        print("=" * 60)

        import subprocess
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
        print("STAGE 5: Skill evolution (MAP → REDUCE → APPLY)")
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

        result = evolver.run(records, input_mode="records")

        # Write changelog
        changelog_entries = result.get("changelog", [])
        cumulative_patch = result.get("cumulative_patch", "")
        change_log_path = out / "change.log"
        change_log_lines = [
            "Change Log (Parallel Evolution):",
            f"MAP patches: {len(result.get('patches', []))}",
            f"LLM calls: {result.get('total_llm_calls', 0)}",
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

        print(f"\n  Edits applied: {len(result.get('edits', []))}")
        print(f"  LLM calls:     {result.get('total_llm_calls', 0)}")

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
        )

        evolved_preds = evolved_agent.run_and_save(
            dialogues=dialogues,
            output_path=str(out / "predictions_evolved.json"),
        )

        evolved_eval = eval_func(
            dataset_name="multiwoz21",
            data_path=str(config.resolved_data_path),
            predictions_path=str(out / "predictions_evolved.json"),
            split=config.split,
            output_path=str(out / "eval_evolved.json"),
            llm_judge=config.llm_judge,
            llm_model=model,
            llm_judge_sample_size=min(config.llm_judge_sample, len(dialogues)),
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


# ══════════════════════════════════════════════════════════════════
# Script entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    result = run_pipeline()
    print(f"\nSeed:     IR={result.seed_eval['aggregate']['info_rate']:.3f}  "
          f"SR={result.seed_eval['aggregate']['success_rate']:.3f}")
    if result.had_failures:
        print(f"Evolved:  IR={result.evolved_eval['aggregate']['info_rate']:.3f}  "
              f"SR={result.evolved_eval['aggregate']['success_rate']:.3f}")
