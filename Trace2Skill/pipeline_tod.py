#!/usr/bin/env python3
"""ToD Skill Evolution Pipeline -- single entry point.

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

    Dataset splitting:
      MultiWOZ 2.1 has pre-defined splits: train (8438), validation
      (1000), test (1000).  Use ``split`` to select one, or leave as
      ``None`` to load all.  For smoke testing (no LLM calls), set
      ``smoke_test=True`` — this forces dummy data, disables the judge
      and evolution, and limits dialogues to a tiny slice.
    """

    # Paths (relative to project root, or absolute)
    skill_dir: str = "eval_tod/skills"  # parent dir containing skill subdirs

    # Default points to the real MultiWOZ 2.1 dialogues (with splits).
    # Set smoke_test=True to use dummy_data.json instead.
    data_path: str = "data/eval/multiwoz21/data/data/dialogues.json"
    db_dir: str = "data/eval/multiwoz21/data/data"
    output_dir: str = "outputs/tod_pipeline"

    # Dataset
    #   split: one of "train", "validation", "test", or None for all.
    #          MultiWOZ 2.1 split sizes: train=8438 val=1000 test=1000
    split: str | None = "test"
    start: int = 0
    end: int | None = None  # None = all dialogues in split
    seed: int = 41
    dataset_name: str = "multiwoz21"  # dataset key (currently only multiwoz21)

    # Agent
    model: str = "deepseek-chat"
    api_key: str | None = None  # None = resolve from config / env
    base_url: str | None = None  # None = resolve from config / env
    max_turns: int = 6
    workers_agent: int = 1
    workers_analysis: int = 4

    # LLM Judge
    llm_judge: bool = True
    llm_judge_sample: int = 50  # per-split default

    # Evolution
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    # Smoke test -- no LLM calls, minimal data
    smoke_test: bool = False

    # ── Batch training ──────────────────────────────────────────
    # When batch_training=True, the pipeline iterates over training
    # dialogues in batches, evolving the skill after each batch.
    batch_training: bool = False        # enable iterative batch-based evolution
    batch_size: int = 50                # training dialogues per batch
    checkpoint_every: int | None = None  # save skill snapshot every N batches
    val_every: int | None = None        # run validation every N batches
    val_split: str | None = None        # explicit val split (e.g. "validation")
    test_split: str | None = None       # explicit test split (e.g. "test")
    max_batches: int | None = None      # cap total batches (None = all)
    val_fraction: float = 0.2           # hold-out fraction from training for val
    seed_split: int = 42                # random seed for train/val split
    resume_from: str | None = None      # resume from checkpoint path

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

    # ── smoke-test overrides ──
    def apply_smoke_test(self) -> None:
        """Force safe no-LLM defaults suitable for a quick smoke test.

        Idempotent — calling multiple times has no extra effect.
        """
        if self.smoke_test:
            return  # already applied
        self.smoke_test = True
        self.data_path = "data/eval/multiwoz21/dummy_data.json"
        self.split = None  # dummy data has a mix; don't filter
        self.start = 0
        self.end = 3  # only 3 dialogues
        self.llm_judge = False
        self.llm_judge_sample = 0
        self.evolution.dry_run = True  # skip actual LLM evolver calls


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

    # Batch training trajectory (empty/None for oneshot mode)
    num_batches: int = 0
    batch_metrics: list = field(default_factory=list)
    val_history: list = field(default_factory=list)
    checkpoint_dir: Path | None = None


# ══════════════════════════════════════════════════════════════════
# Batch training helpers
# ══════════════════════════════════════════════════════════════════


def _split_train_val(
    dialogues: list, val_fraction: float, seed: int,
) -> tuple[list, list]:
    """Randomly split dialogues into train and validation sets.

    Args:
        dialogues: List of Dialogue objects.
        val_fraction: Fraction to hold out for validation (0 < val_fraction < 1).
        seed: Random seed for reproducibility.

    Returns:
        (train_dialogues, val_dialogues) tuple.
    """
    import random
    rng = random.Random(seed)
    indices = list(range(len(dialogues)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_fraction))
    val_indices = set(indices[:n_val])
    train_indices = indices[n_val:]
    train = [dialogues[i] for i in sorted(train_indices)]
    val = [dialogues[i] for i in sorted(val_indices)]
    return train, val


def _build_batches(
    dialogues: list, batch_size: int, max_batches: int | None = None,
) -> list[list]:
    """Split dialogues into batches of batch_size.

    Args:
        dialogues: List of Dialogue objects.
        batch_size: Number of dialogues per batch.
        max_batches: If set, only return the first N batches.

    Returns:
        List of batches (each batch is a list of Dialogue objects).
    """
    batches = [
        dialogues[i:i + batch_size]
        for i in range(0, len(dialogues), batch_size)
    ]
    if max_batches is not None:
        batches = batches[:max_batches]
    return batches


def _save_batch_checkpoint(
    skill_dir: Path,
    checkpoint_root: Path,
    step: int,
    metrics: dict,
    changelog: list[str] | None = None,
    val_metrics: dict | None = None,
) -> Path:
    """Save a checkpoint of the current skill state.

    Copies the skill directory to ``checkpoint_root/step_{step:04d}/``
    and writes ``_checkpoint_meta.json`` with metadata.

    Args:
        skill_dir: Path to the current evolved skill directory.
        checkpoint_root: Root directory for all checkpoints.
        step: Current batch number.
        metrics: Per-batch metrics dict.
        changelog: Evolution changelog entries (optional).
        val_metrics: Most recent validation metrics (optional).

    Returns:
        Path to the saved checkpoint directory.
    """
    from datetime import datetime

    dest = checkpoint_root / f"step_{step:04d}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(skill_dir, dest)

    meta = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "changelog": changelog or [],
        "val_metrics": val_metrics,
    }
    (dest / "_checkpoint_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    log.info("Checkpoint saved: step_%04d", step)
    return dest


def _resume_from_checkpoint(
    checkpoint_path: str, target_skill_dir: Path,
) -> int:
    """Restore skill state from a checkpoint and return the step number.

    Copies the checkpoint's skill files into ``target_skill_dir``
    (overwriting existing files) and returns the step number so the
    pipeline can skip already-processed batches.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        target_skill_dir: Directory to restore the skill into.

    Returns:
        The step number from which to resume (already-completed batches).
    """
    src = Path(checkpoint_path)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    meta_file = src / "_checkpoint_meta.json"
    step = 0
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        step = meta.get("step", 0)

    # Copy skill files into target
    if target_skill_dir.exists():
        shutil.rmtree(target_skill_dir)
    shutil.copytree(src, target_skill_dir, ignore=shutil.ignore_patterns("_checkpoint_meta.json"))

    print(f"  Resumed from checkpoint: step_{step:04d}")
    return step


def _stage0_load_and_split(config: PipelineConfig):
    """Load dataset and split into train/val/test for batch training.

    Strategy:
    1. Try explicit test_split; if not found, hold out 20% of data as test.
    2. Try explicit val_split; if not found, hold out val_fraction from
       remaining data as val.
    3. The rest is train.

    Returns:
        (train_dialogues, val_dialogues, test_dialogues, split_counts)
    """
    from eval_tod.data_loader import load_dataset, list_available_splits

    data_path = str(config.resolved_data_path)
    split_counts = list_available_splits(data_path)
    avail = list(split_counts.keys())

    # Load the main data pool
    main_split = config.split or "train"
    main_split = main_split if main_split in split_counts else None

    all_dialogues = load_dataset(
        config.dataset_name, data_path, split=main_split,
    )
    all_dialogues = all_dialogues[config.start:(config.end or len(all_dialogues))]

    # ── Test set ──
    test_split_name = config.test_split or "test"
    if test_split_name in split_counts and test_split_name != (main_split or ""):
        # Load from explicit separate split
        test_dialogues = load_dataset(
            config.dataset_name, data_path, split=test_split_name,
        )
        test_dialogues = test_dialogues[config.start:(config.end or len(test_dialogues))]
        remaining = all_dialogues
    else:
        # Hold out a test fraction from the main pool
        test_dialogues, remaining = _split_train_val(
            all_dialogues, config.val_fraction, config.seed_split,
        )

    # ── Validation set ──
    val_split_name = config.val_split
    if val_split_name and val_split_name in split_counts and val_split_name != (main_split or ""):
        val_dialogues = load_dataset(
            config.dataset_name, data_path, split=val_split_name,
        )
        val_dialogues = val_dialogues[config.start:(config.end or len(val_dialogues))]
        train_raw = remaining
    else:
        # Hold out val_fraction from remaining data as val
        train_raw, val_dialogues = _split_train_val(
            remaining, config.val_fraction, config.seed_split,
        )

    return train_raw, val_dialogues, test_dialogues, split_counts


def _run_validation(
    config: PipelineConfig,
    model: str,
    api_key: str,
    base_url: str,
    kb,
    skills_dir: str,
    dialogues: list,
    out: Path,
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
    import subprocess
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
    """Run the original one-shot pipeline (stages 1-6) on a single set of dialogues.

    Extracted from run_pipeline to keep backward compatibility when
    batch_training=False.
    """
    evo = config.evolution

    # ── Stage 1: Run seed agent ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1: Generate predictions with seed skill")
    print("=" * 60)

    from eval_tod.agent_skill import SkillPreloadedAgent

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

    from eval_tod import evaluate as eval_func

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


# ══════════════════════════════════════════════════════════════════
# Main pipeline entry point
# ══════════════════════════════════════════════════════════════════


def run_pipeline(config: PipelineConfig | None = None, **kwargs) -> PipelineResult:
    """Run the full skill evolution pipeline.

    Two modes:
    - **One-shot** (default, ``batch_training=False``): the original
      6-stage pipeline on a single split.
    - **Batch training** (``batch_training=True``): iterates over
      training dialogues in batches, evolving the skill after each
      batch, with periodic checkpointing and validation.

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

    # ── Smoke-test overrides (applied first) ────────────────────
    if config.smoke_test:
        config.apply_smoke_test()

    # ── Resolve API config ──────────────────────────────────────
    cfg = resolve_config(
        api_key=config.api_key, base_url=config.base_url, model=config.model,
    )
    model, api_key, base_url = cfg["model"], cfg["api_key"], cfg["base_url"]

    # ── Prepare timestamped output directory ────────────────────
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = config.resolved_output_dir / timestamp
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput dir: {out}")

    # ── Response logger ─────────────────────────────────────────
    from eval_tod.response_logger import ResponseLogger
    response_logger = ResponseLogger(str(out / "llm_responses"))

    # ══════════════════════════════════════════════════════════════
    # STAGE 0: Load dataset, report splits, select dialogues
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STAGE 0: Load dataset and split")
    print("=" * 60)

    from eval_tod.kb import MultiWOZKB
    from eval_tod.data_loader import load_dataset, list_available_splits

    kb = MultiWOZKB(str(config.resolved_db_dir))
    print(f"KB loaded: {kb.domains}")

    split_counts = list_available_splits(str(config.resolved_data_path))
    print(f"  Dataset:     {config.dataset_name}")
    print(f"  Data path:   {config.resolved_data_path}")
    print(f"  DB dir:      {config.resolved_db_dir}")
    print(f"  All splits:  {dict(split_counts)}")
    print(f"  Total:       {sum(split_counts.values())} dialogues")

    # ── Branch: batch training vs oneshot ───────────────────────
    if config.batch_training and not config.smoke_test:
        return _run_batch_training_pipeline(
            config, model, api_key, base_url, out, kb, split_counts, response_logger,
        )
    else:
        # Original one-shot flow
        if config.split and config.split not in split_counts:
            avail = list(split_counts.keys())
            raise ValueError(
                f"Requested split '{config.split}' not found in data. "
                f"Available: {avail}"
            )

        dialogues = load_dataset(
            config.dataset_name,
            str(config.resolved_data_path),
            split=config.split,
        )
        total_in_split = len(dialogues)

        start = config.start
        end = config.end if config.end is not None else total_in_split
        dialogues = dialogues[start:end]

        print(f"  Selected split: {config.split or 'all'}")
        print(f"    Full split:    {total_in_split} dialogues")
        print(f"    Slice:         [{start}:{end}] -> {len(dialogues)} dialogues")
        print(f"    Domains in slice: "
              f"{sorted(set(d for dg in dialogues for d in dg.domains))}")

        if config.smoke_test:
            print(f"  [SMOKE TEST] no LLM calls, minimal data")

        return _run_oneshot_pipeline(
            config, model, api_key, base_url, out, kb,
            dialogues, split_counts, start, end, total_in_split,
            response_logger,
        )


def _run_batch_training_pipeline(
    config: PipelineConfig,
    model: str,
    api_key: str,
    base_url: str,
    out: Path,
    kb,
    split_counts: dict,
    response_logger=None,
) -> PipelineResult:
    """Run the iterative batch-training pipeline.

    Stage 0: Load and split train/val/test
    Stage 1: Seed baseline on validation set
    Stage 2: Batch training loop with checkpointing
    Stage 3: Final test evaluation
    """
    from eval_tod.agent_skill import SkillPreloadedAgent

    # ── Stage 0 (batch): Load and split ─────────────────────────
    print("\n  [BATCH TRAINING MODE]")
    train_raw, val_dialogues, test_dialogues, _ = _stage0_load_and_split(config)

    print(f"  Train:            {len(train_raw)} dialogues")
    print(f"  Validation:       {len(val_dialogues)} dialogues")
    print(f"  Test:             {len(test_dialogues)} dialogues")

    batches = _build_batches(train_raw, config.batch_size, config.max_batches)
    print(f"  Batches:          {len(batches)} (batch_size={config.batch_size})")
    print(f"  Checkpoint every: {config.checkpoint_every or 'never'}")
    print(f"  Validate every:   {config.val_every or 'never'}")

    # ── Prepare evolved skill directory ─────────────────────────
    evolved_skills_dir = out / "evolved_skills"
    evolved_skill_dir = evolved_skills_dir / config.skill_subdir
    # Start from seed skill
    if evolved_skill_dir.exists():
        shutil.rmtree(evolved_skill_dir)
    shutil.copytree(
        config.resolved_skill_dir / config.skill_subdir,
        evolved_skill_dir,
    )

    # Resume from checkpoint if requested
    start_batch = 0
    if config.resume_from:
        start_batch = _resume_from_checkpoint(config.resume_from, evolved_skill_dir)

    # ── Stage 1 (batch): Seed baseline on validation ────────────
    print("\n" + "=" * 60)
    print("STAGE 1: Seed skill baseline on validation set")
    print("=" * 60)

    val_result = _run_validation(
        config, model, api_key, base_url, kb,
        str(config.resolved_skill_dir),  # seed skill
        val_dialogues, out, "seed_baseline",
        response_logger=response_logger,
    )
    seed_val_metrics = val_result["metrics"]
    print(f"  Seed val IR: {seed_val_metrics['info_rate']:.4f}, "
          f"Success: {seed_val_metrics['success_rate']:.4f}")

    # ── Also evaluate seed on test for comparison ───────────────
    test_result_seed = _run_validation(
        config, model, api_key, base_url, kb,
        str(config.resolved_skill_dir),  # seed skill
        test_dialogues, out, "test_seed_baseline",
        response_logger=response_logger,
    )
    seed_test_metrics = test_result_seed["metrics"]
    print(f"  Seed test IR: {seed_test_metrics['info_rate']:.4f}, "
          f"Success: {seed_test_metrics['success_rate']:.4f}")

    # ── Create reusable agent ───────────────────────────────────
    agent = SkillPreloadedAgent(
        kb=kb,
        skills_dir=str(evolved_skills_dir),
        model=model,
        max_turns=config.max_turns,
        log_dir=str(out / "trajectories"),
        api_key=api_key,
        base_url=base_url,
        response_logger=response_logger,
    )

    # ── Stage 2: Batch training loop ────────────────────────────
    print("\n" + "=" * 60)
    print(f"STAGE 2: Batch training ({len(batches)} batches, starting at batch {start_batch+1})")
    print("=" * 60)

    val_history: list[dict] = [seed_val_metrics]
    batch_metrics: list[dict] = []
    checkpoint_dir = out / "checkpoints"
    total_failed = 0
    total_llm_calls = 0

    for batch_idx in range(start_batch, len(batches)):
        batch = batches[batch_idx]
        epoch_num = batch_idx + 1

        print(f"\n{'─'*40}")
        print(f"  Batch {epoch_num}/{len(batches)}: {len(batch)} dialogues")
        print(f"{'─'*40}")

        iter_result = _run_training_iteration(
            config, model, api_key, base_url, kb,
            evolved_skills_dir, evolved_skill_dir,
            batch, epoch_num, out, agent=agent,
            response_logger=response_logger,
        )

        batch_metrics.append(iter_result["metrics"])
        total_failed += iter_result["metrics"].get("num_failures", 0)
        total_llm_calls += iter_result.get("llm_calls", 0)

        # ── Checkpoint ──────────────────────────────────────────
        if config.checkpoint_every and epoch_num % config.checkpoint_every == 0:
            _save_batch_checkpoint(
                evolved_skill_dir, checkpoint_dir, epoch_num,
                iter_result["metrics"],
                changelog=iter_result.get("changelog", []),
                val_metrics=val_history[-1] if val_history else None,
            )

        # ── Validation ──────────────────────────────────────────
        if config.val_every and epoch_num % config.val_every == 0:
            print(f"\n  Running validation at batch {epoch_num}...")
            val_result = _run_validation(
                config, model, api_key, base_url, kb,
                str(evolved_skills_dir),
                val_dialogues, out, f"val_step_{epoch_num:04d}",
                response_logger=response_logger,
            )
            vm = val_result["metrics"]
            val_history.append(vm)
            delta_ir = vm["info_rate"] - seed_val_metrics["info_rate"]
            delta_sr = vm["success_rate"] - seed_val_metrics["success_rate"]
            print(f"  Val IR: {vm['info_rate']:.4f} (delta={delta_ir:+.4f}), "
                  f"SR: {vm['success_rate']:.4f} (delta={delta_sr:+.4f})")
            if vm.get("llm_judge"):
                js = vm["llm_judge"]
                print(f"  Val Judge: {', '.join(f'{k}={v:.2f}' for k, v in js.items())}")

    # Final checkpoint
    if config.checkpoint_every:
        _save_batch_checkpoint(
            evolved_skill_dir, checkpoint_dir, len(batches),
            batch_metrics[-1] if batch_metrics else {},
            changelog=[],
            val_metrics=val_history[-1] if val_history else None,
        )

    # ── Stage 3: Final test evaluation ──────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 3: Final test evaluation with evolved skill")
    print("=" * 60)

    test_result_final = _run_validation(
        config, model, api_key, base_url, kb,
        str(evolved_skills_dir),
        test_dialogues, out, "test_final",
        response_logger=response_logger,
    )
    final_test_metrics = test_result_final["metrics"]

    delta_ir = final_test_metrics["info_rate"] - seed_test_metrics["info_rate"]
    delta_sr = final_test_metrics["success_rate"] - seed_test_metrics["success_rate"]
    print(f"\n  Seed    test IR: {seed_test_metrics['info_rate']:.4f}, "
          f"SR: {seed_test_metrics['success_rate']:.4f}")
    print(f"  Evolved test IR: {final_test_metrics['info_rate']:.4f}, "
          f"SR: {final_test_metrics['success_rate']:.4f}")
    print(f"  Delta:           IR={delta_ir:+.4f}, SR={delta_sr:+.4f}")

    # ── Save training trajectory ────────────────────────────────
    trajectory = {
        "seed_val_metrics": seed_val_metrics,
        "seed_test_metrics": seed_test_metrics,
        "final_test_metrics": final_test_metrics,
        "num_batches": len(batches),
        "total_failed": total_failed,
        "total_llm_calls": total_llm_calls,
        "batch_metrics": batch_metrics,
        "val_history": val_history,
    }
    (out / "training_trajectory.json").write_text(
        json.dumps(trajectory, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # ── Done ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"Output: {out}")
    print(f"Batches: {len(batches)}, Total failures: {total_failed}")
    print(f"Seed    test: IR={seed_test_metrics['info_rate']:.4f}  "
          f"SR={seed_test_metrics['success_rate']:.4f}")
    print(f"Evolved test: IR={final_test_metrics['info_rate']:.4f}  "
          f"SR={final_test_metrics['success_rate']:.4f}")
    print(f"Checkpoints:  {checkpoint_dir}")
    print(f"{'='*60}")

    return PipelineResult(
        output_dir=out,
        seed_predictions_path=Path(test_result_seed["predictions_path"]),
        seed_eval_path=Path(test_result_seed["eval_path"]),
        evolved_predictions_path=Path(test_result_final["predictions_path"]),
        evolved_eval_path=Path(test_result_final["eval_path"]),
        evolved_skill_dir=evolved_skill_dir,
        seed_eval=test_result_seed["eval_result"],
        evolved_eval=test_result_final["eval_result"],
        num_dialogues=len(train_raw),
        num_failed=total_failed,
        had_failures=total_failed > 0,
        num_batches=len(batches),
        batch_metrics=batch_metrics,
        val_history=val_history,
        checkpoint_dir=checkpoint_dir,
    )


# ══════════════════════════════════════════════════════════════════
# Script entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="ToD Skill Evolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (no LLM calls, minimal data):
  python -m Trace2Skill.pipeline_tod --smoke-test

  # One-shot on test split:
  python -m Trace2Skill.pipeline_tod --split test --end 50

  # Batch training with checkpointing:
  python -m Trace2Skill.pipeline_tod --batch-training --batch-size 50 \\
      --checkpoint-every 10 --val-every 5 --max-batches 50

  # Batch training with explicit val/test splits:
  python -m Trace2Skill.pipeline_tod --batch-training \\
      --split train --val-split validation --test-split test \\
      --batch-size 50 --val-every 5 --end 200
""".strip(),
    )
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run smoke test with dummy data, no LLM judge, no evolution")
    ap.add_argument("--split", default=None,
                    choices=["train", "validation", "test"],
                    help="Data split (default: test)")
    ap.add_argument("--start", type=int, default=0,
                    help="Start index for dialogue slice")
    ap.add_argument("--end", type=int, default=None,
                    help="End index (default: all in split)")
    ap.add_argument("--model", default="deepseek-chat",
                    help="LLM model name")
    ap.add_argument("--no-judge", action="store_true",
                    help="Disable LLM judge")
    ap.add_argument("--judge-sample", type=int, default=50,
                    help="LLM judge sample size")
    ap.add_argument("--output-dir", default="outputs/tod_pipeline",
                    help="Output directory")
    ap.add_argument("--data-path", default=None,
                    help="Override data path (default: real MultiWOZ data)")
    ap.add_argument("--max-turns", type=int, default=6,
                    help="Max conversation turns per dialogue")

    # ── Batch training args ─────────────────────────────────────
    ap.add_argument("--batch-training", action="store_true",
                    help="Enable iterative batch-based skill evolution")
    ap.add_argument("--batch-size", type=int, default=50,
                    help="Number of training dialogues per batch")
    ap.add_argument("--checkpoint-every", type=int, default=None,
                    help="Save skill checkpoint every N batches")
    ap.add_argument("--val-every", type=int, default=None,
                    help="Run validation every N batches")
    ap.add_argument("--val-split", default=None,
                    help="Explicit validation split (e.g. 'validation')")
    ap.add_argument("--test-split", default=None,
                    help="Explicit test split (e.g. 'test')")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Maximum number of training batches")
    ap.add_argument("--resume-from", default=None,
                    help="Resume from a checkpoint directory")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig()

    if args.smoke_test:
        config.apply_smoke_test()
    else:
        # Apply CLI overrides
        if args.split is not None:
            config.split = args.split
        config.start = args.start
        if args.end is not None:
            config.end = args.end
        config.model = args.model
        config.output_dir = args.output_dir
        config.max_turns = args.max_turns
        if args.data_path is not None:
            config.data_path = args.data_path
        if args.no_judge:
            config.llm_judge = False
        config.llm_judge_sample = args.judge_sample

        # Batch training overrides
        config.batch_training = args.batch_training
        config.batch_size = args.batch_size
        config.checkpoint_every = args.checkpoint_every
        config.val_every = args.val_every
        if args.val_split:
            config.val_split = args.val_split
        if args.test_split:
            config.test_split = args.test_split
        config.max_batches = args.max_batches
        config.resume_from = args.resume_from

    result = run_pipeline(config)
    if config.batch_training and not config.smoke_test:
        print(f"\nSeed    test: IR={result.seed_eval['aggregate']['info_rate']:.3f}  "
              f"SR={result.seed_eval['aggregate']['success_rate']:.3f}")
        print(f"Evolved test: IR={result.evolved_eval['aggregate']['info_rate']:.3f}  "
              f"SR={result.evolved_eval['aggregate']['success_rate']:.3f}")
    else:
        print(f"\nSeed:     IR={result.seed_eval['aggregate']['info_rate']:.3f}  "
              f"SR={result.seed_eval['aggregate']['success_rate']:.3f}")
        if result.had_failures:
            print(f"Evolved:  IR={result.evolved_eval['aggregate']['info_rate']:.3f}  "
                  f"SR={result.evolved_eval['aggregate']['success_rate']:.3f}")
