"""ToD Skill Evolution Pipeline — modular package.

Usage::

    from Trace2Skill.pipeline import PipelineConfig, run_pipeline
    result = run_pipeline(PipelineConfig(batch_training=True, ...))

Or from CLI::

    python -m Trace2Skill.pipeline.main --smoke-test
    python -m Trace2Skill.pipeline.main --batch-training --split train

Backward-compatible::

    python -m Trace2Skill.pipeline_tod  # still works
"""

from .config import EvolutionConfig, PipelineConfig, PipelineResult
from .main import run_pipeline

__all__ = [
    "EvolutionConfig",
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline",
]
