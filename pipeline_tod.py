#!/usr/bin/env python3
"""Thin wrapper -- delegates to Trace2Skill.pipeline_tod.

Usage:
    python pipeline_tod.py              # same as before
    python -m Trace2Skill.pipeline_tod  # equivalent
    from Trace2Skill.pipeline_tod import PipelineConfig, run_pipeline  # library use
"""

import importlib.util
import sys
from pathlib import Path

# Explicitly load Trace2Skill/pipeline_tod.py with a distinct module name
# to avoid clashing with this wrapper file's own name.
_TRACE2SKILL = Path(__file__).resolve().parent / "Trace2Skill"
_spec = importlib.util.spec_from_file_location(
    "_trace2skill_pipeline_tod",
    str(_TRACE2SKILL / "pipeline_tod.py"),
)
_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["_trace2skill_pipeline_tod"] = _pipeline
_spec.loader.exec_module(_pipeline)

PipelineConfig = _pipeline.PipelineConfig
EvolutionConfig = _pipeline.EvolutionConfig
PipelineResult = _pipeline.PipelineResult
run_pipeline = _pipeline.run_pipeline

if __name__ == "__main__":
    run_pipeline()
