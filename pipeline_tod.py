#!/usr/bin/env python3
"""Thin wrapper -- delegates to Trace2Skill.pipeline_tod.

Usage:
    python pipeline_tod.py              # same as before
    python -m Trace2Skill.pipeline_tod  # equivalent
    from Trace2Skill.pipeline_tod import PipelineConfig, run_pipeline  # library use
"""

from Trace2Skill.pipeline_tod import (
    EvolutionConfig,
    PipelineConfig,
    PipelineResult,
    run_cli,
    run_pipeline,
)

if __name__ == "__main__":
    run_cli()
