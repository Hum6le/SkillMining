#!/usr/bin/env python3
"""Thin wrapper — delegates to Trace2Skill.pipeline_tod.

Usage:
    python pipeline_tod.py              # same as before
    python -m Trace2Skill.pipeline_tod  # equivalent
    from Trace2Skill.pipeline_tod import PipelineConfig, run_pipeline  # library use
"""

import importlib
import sys
from pathlib import Path

# Ensure Trace2Skill/pipeline_tod.py is importable (avoiding name clash
# with this wrapper file, which has the same basename).
_TRACE2SKILL = Path(__file__).resolve().parent / "Trace2Skill"
if str(_TRACE2SKILL) not in sys.path:
    sys.path.insert(0, str(_TRACE2SKILL))

_pipeline = importlib.import_module("pipeline_tod")
PipelineConfig = _pipeline.PipelineConfig
EvolutionConfig = _pipeline.EvolutionConfig
PipelineResult = _pipeline.PipelineResult
run_pipeline = _pipeline.run_pipeline

if __name__ == "__main__":
    run_pipeline()
