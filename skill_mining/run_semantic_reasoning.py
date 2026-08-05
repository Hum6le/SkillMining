#!/usr/bin/env python3
"""Run model-agnostic semantic reasoning over mined skill motifs.

All model configuration stays inside the repository-level ``llm.chat`` API.
This runner only supplies grounded evidence and parses the returned JSON.

Example:
    python skill_mining/run_semantic_reasoning.py \
        --input skill_mining/output/abcd_hierarchical_skills_by_subflow.json \
        --output skill_mining/output/abcd_hierarchical_skills_reasoned.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hierarchical_skill_mining import render_reasoning_prompt
from llm import chat


REASONING_FIELDS = (
    "semantic_name",
    "goal",
    "parameters",
    "preconditions",
    "postconditions",
    "branches",
    "confidence",
    "unknowns",
)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from a model response."""
    text = (text or "").strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_reasoning(value: dict[str, Any] | None) -> dict[str, Any]:
    """Keep the semantic artifact stable even when the model omits fields."""
    value = value or {}
    normalized = {field: value.get(field) for field in REASONING_FIELDS}
    normalized["semantic_name"] = normalized["semantic_name"] or "unknown"
    normalized["goal"] = normalized["goal"] or "unknown"
    for field in ("parameters", "preconditions", "postconditions", "branches", "unknowns"):
        if not isinstance(normalized[field], list):
            normalized[field] = []
    try:
        normalized["confidence"] = float(normalized["confidence"] or 0.0)
    except (TypeError, ValueError):
        normalized["confidence"] = 0.0
    normalized["confidence"] = max(0.0, min(1.0, normalized["confidence"]))
    return normalized


def reason_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    prompt = render_reasoning_prompt(evidence)
    error = None
    try:
        raw = chat(prompt, temperature=0.0)
    except Exception as exc:
        raw = ""
        error = f"{type(exc).__name__}: {exc}"
    parsed = extract_json_object(raw)
    result = normalize_reasoning(parsed)
    result["parse_status"] = "error" if error else (
        "ok" if parsed is not None else "invalid_or_empty_response"
    )
    if error:
        result["error"] = error
    result["raw_response"] = raw
    return result


def run(input_path: Path, output_path: Path) -> dict[str, Any]:
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(source.get("skills"), dict):
        results: dict[str, Any] = {}
        for key, skill in source["skills"].items():
            results[key] = reason_evidence(skill["agent_evidence"])
        output = {
            "algorithm": source.get("algorithm"),
            "input": str(input_path),
            "reasoning_api": "llm.chat",
            "semantic_reasoning": results,
            "source_statistics": source.get("statistics", {}),
        }
    else:
        output = {
            "algorithm": source.get("algorithm"),
            "input": str(input_path),
            "reasoning_api": "llm.chat",
            "semantic_reasoning": reason_evidence(source["agent_evidence"]),
            "source_statistics": source.get("statistics", {}),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.input, args.output)
    reasoning = result["semantic_reasoning"]
    count = len(reasoning) if isinstance(reasoning, dict) and "parse_status" not in reasoning else 1
    print(json.dumps({"reasoned_motifs": count}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
