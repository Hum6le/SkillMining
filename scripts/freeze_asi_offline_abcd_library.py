#!/usr/bin/env python3
"""Freeze validated ASIoffline functions into a prompt-ready ABCD library."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from asi_offline import render_asi_library


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze an ASIoffline ABCD function library.")
    parser.add_argument("--accepted-functions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source_path = Path(args.accepted_functions)
    records = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("--accepted-functions must be a JSON list")
    library = render_asi_library(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ASI_ACTIONS.md").write_text(library.rendered_text, encoding="utf-8")
    (output_dir / "library.json").write_text(
        json.dumps(library.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "method": "asioffline-abcd",
        "stage": "frozen_action_library",
        "accepted_functions_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "accepted_function_definitions": len(records),
        "frozen_unique_functions": len(library.functions),
        "duplicate_definitions_skipped": len(library.duplicate_names),
        "runtime_policy": {
            "test_time_skill_updates": False,
            "scenario_labels_exposed": False,
            "composite_function_emitted_as_abcd_action": False,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Froze {len(library.functions)} ASI functions -> {output_dir / 'ASI_ACTIONS.md'}")


if __name__ == "__main__":
    main()
