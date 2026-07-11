#!/usr/bin/env python3
"""ToD Skill Evolution Pipeline -- backward-compatible re-export.

This file is kept for backward compatibility.  All logic now lives in
the ``Trace2Skill.pipeline`` package.

Usage (both styles work)::

    # New style
    from Trace2Skill.pipeline import PipelineConfig, run_pipeline

    # Old style (still works)
    from Trace2Skill.pipeline_tod import PipelineConfig, run_pipeline

CLI::

    # New style
    python -m Trace2Skill.pipeline.main --smoke-test

    # Old style (still works)
    python -m Trace2Skill.pipeline_tod --smoke-test
"""

from Trace2Skill.pipeline.config import EvolutionConfig, PipelineConfig, PipelineResult
from Trace2Skill.pipeline.main import main as run_cli, run_pipeline

__all__ = [
    "EvolutionConfig",
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline",
]

if __name__ == "__main__":
    run_cli()
