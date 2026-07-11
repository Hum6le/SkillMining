#!/usr/bin/env python3
"""Canonical script entry for the Trace2Skill ToD pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Trace2Skill.pipeline.main import main


if __name__ == "__main__":
    main()
