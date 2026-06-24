"""Pipeline configuration and result dataclasses."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
# pipeline/  is one level below Trace2Skill/
_PIPELINE_DIR = Path(__file__).resolve().parent
_TRACE2SKILL = _PIPELINE_DIR.parent
_PROJECT_ROOT = _TRACE2SKILL.parent

# Make project root importable (for eval_tod and llm modules)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_TRACE2SKILL) not in sys.path:
    sys.path.insert(0, str(_TRACE2SKILL))


# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════


@dataclass
class EvolutionConfig:
    """Settings for the skill evolution phase (MAP -> REDUCE -> APPLY)."""

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
      ``smoke_test=True`` -- this forces dummy data, disables the judge
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

        Idempotent -- calling multiple times has no extra effect.
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
