"""Audit evidence flow and patch application in a Trace2Skill-hybrid run.

Run against the run root, e.g.:
  python scripts/audit_trace2skill_hybrid.py /path/to/run \
      --json-out /path/to/run/hybrid_audit.json \
      --markdown-out /path/to/run/hybrid_audit.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


RESOURCE_NAMES = (
    "tod_reference.md",
    "tod_action_rules.md",
    "tod_slot_policies.md",
)
FORBIDDEN_TRAJECTORY_KEYS = {
    "react_trace", "reference_lookup", "action_selection", "action_card_lookup",
}
ANALYSIS_TAGS = {"trace2skill_error_analysis", "trace2skill_success_analysis"}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _files(root: Path, patterns: Iterable[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        found.update(p for p in root.rglob(pattern) if p.is_file())
    return sorted(found)


def _contains_key(value: Any, keys: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys:
                found.append(str(key))
            found.extend(_contains_key(item, keys))
    elif isinstance(value, list):
        for item in value:
            found.extend(_contains_key(item, keys))
    return found


def _message_text(record: dict[str, Any]) -> str:
    messages = record.get("messages", [])
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(m.get("content", "")) if isinstance(m, dict) else str(m)
        for m in messages
    )


def _patch_stats(paths: list[Path]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    edits = 0
    operations: dict[str, int] = {}
    for path in paths:
        payload = _read_json(path)
        row: dict[str, Any] = {"path": str(path), "parseable_json": isinstance(payload, dict)}
        if isinstance(payload, dict):
            raw_edits = payload.get("edits", [])
            if isinstance(raw_edits, list):
                edits += len(raw_edits)
                for edit in raw_edits:
                    if isinstance(edit, dict):
                        op = str(edit.get("op", "<missing>"))
                        operations[op] = operations.get(op, 0) + 1
            row["edit_count"] = len(raw_edits) if isinstance(raw_edits, list) else None
            row["reasoning_chars"] = len(str(payload.get("reasoning", "")))
            row["changelog_entries"] = len(payload.get("changelog_entries", [])) if isinstance(payload.get("changelog_entries"), list) else None
        files.append(row)
    return {"file_count": len(paths), "total_edits": edits, "ops": operations, "files": files}


def _load_response_prompts(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in _files(run_dir, ("*_prompt.json",)):
        record = _read_json(path)
        if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
            continue
        rows.append({
            "path": str(path), "tag": str(record.get("call_tag", "")),
            "text": _message_text(record), "record": record,
        })
    return rows


def _audit_batch(batch_dir: Path, response_prompts: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_path = batch_dir / "trajectory_evidence.json"
    payload = _read_json(evidence_path)
    records = payload if isinstance(payload, list) else []
    ids = {str(r.get("conversation_id", "")) for r in records if isinstance(r, dict)}
    turn_rows = [
        turn for row in records if isinstance(row, dict)
        for turn in row.get("trajectory", []) if isinstance(turn, dict)
    ]
    trajectory_chars = sum(
        len(str(turn.get(key, ""))) for turn in turn_rows for key in ("context", "prediction")
    )
    forbidden = sorted(set(_contains_key(turn_rows, FORBIDDEN_TRAJECTORY_KEYS)))

    analysis_calls = []
    for row in response_prompts:
        if row["tag"] not in ANALYSIS_TAGS:
            continue
        text = row["text"]
        matched_ids = sorted(
            cid for cid in ids
            if cid and re.search(
                rf"(?<![A-Za-z0-9])(?:abcd[-_])?{re.escape(cid)}(?![A-Za-z0-9])",
                text,
                flags=re.IGNORECASE,
            )
        )
        if matched_ids:
            analysis_calls.append({
                "tag": row["tag"], "path": row["path"], "matched_conversation_ids": matched_ids,
                "has_graph_context_marker": "graph_context" in text,
                "has_first_divergence": "first_divergence" in text,
                "has_gold_action": "gold_action" in text or "Golden action" in text,
                "has_gold_slots": "gold_slots" in text or "Golden slot" in text,
                "has_complete_react_trace_marker": "react_trace" in text,
                "prompt_chars": len(text),
            })

    parsed_analysis = []
    for path in _files(batch_dir, ("error_analysis_parsed.json", "success_analysis_parsed.json")):
        data = _read_json(path)
        rows = data if isinstance(data, list) else []
        parsed_analysis.append({
            "path": str(path), "record_count": len(rows),
            "records_with_ast_evidence": sum(bool(r.get("ast_evidence")) for r in rows if isinstance(r, dict)),
            "records_with_memory_items": sum(
                len(r.get("items", [])) for r in rows if isinstance(r, dict) and isinstance(r.get("items", []), list)
            ),
        })

    evolution_root = batch_dir / "evolution"
    prompt_samples = _files(batch_dir, ("*.md",))
    map_prompts = [p for p in prompt_samples if "prompt_samples" in p.parts and "map" in p.parts]
    reduce_prompts = [p for p in prompt_samples if "prompt_samples" in p.parts and ("reduce" in p.parts or "merge" in p.parts)]
    translation_prompts = [p for p in prompt_samples if "prompt_samples" in p.parts and "translation" in p.parts]
    map_semantic_paths = _files(evolution_root, ("patch_*.md",))
    map_semantic_paths = [p for p in map_semantic_paths if "map_semantic" in p.parts]
    reduce_semantic_paths = _files(evolution_root, ("merged_*.md",))
    reduce_semantic_paths = [p for p in reduce_semantic_paths if any(part.lower().startswith("merge_level_") for part in p.parts)]
    map_text = "\n".join(_read_text(p) for p in map_prompts)
    map_patch_paths = [p for p in _files(evolution_root, ("patch_*.json",)) if "map_patches" in p.parts]
    reduce_patch_paths = [p for p in _files(evolution_root, ("merged_*.json",)) if any(part.lower().startswith("merge_level_") for part in p.parts)]
    final_paths = _files(evolution_root, ("final_patch.json", "final_semantic_patch.md"))
    translated_paths = _files(evolution_root, ("translated_final_patch.json",))
    translated_semantic_paths = _files(evolution_root, ("translated_final_semantic_patch.md",))
    applied_paths = _files(evolution_root, ("applied_diffs.patch",))
    parse_failures = _files(evolution_root, ("*_parse_failed.md",))
    summaries = [x for p in _files(batch_dir, ("batch_summary.json",)) if isinstance((x := _read_json(p)), dict)]

    return {
        "batch_dir": str(batch_dir),
        "evidence": {
            "conversation_count": len(records), "conversation_ids": sorted(ids),
            "trajectory_turn_count": len(turn_rows), "prefix_plus_prediction_chars": trajectory_chars,
            "forbidden_trajectory_fields": forbidden,
        },
        "analysis": {
            "saved_raw_prompt_calls_matched": len(analysis_calls), "calls": analysis_calls,
            "parsed_analysis_artifacts": parsed_analysis,
            "unique_conversations_in_analysis_prompts": len({cid for call in analysis_calls for cid in call["matched_conversation_ids"]}),
            "evidence_gap_ids": sorted(ids - {cid for call in analysis_calls for cid in call["matched_conversation_ids"]}),
        },
        "map": {
            "prompt_sample_count": len(map_prompts),
            "prompt_samples": [str(p) for p in map_prompts],
            "resource_names_mentioned": {name: name in map_text for name in RESOURCE_NAMES},
            "patches": _patch_stats(map_patch_paths),
            "semantic_patch_files": [str(p) for p in map_semantic_paths],
        },
        "reduce": {
            "prompt_sample_count": len(reduce_prompts), "prompt_samples": [str(p) for p in reduce_prompts],
            "merged_patch_files": _patch_stats(reduce_patch_paths),
            "semantic_patch_files": [str(p) for p in reduce_semantic_paths],
            "parse_failure_count": len(parse_failures), "parse_failure_paths": [str(p) for p in parse_failures],
        },
        "translation": {
            "prompt_sample_count": len(translation_prompts), "prompt_samples": [str(p) for p in translation_prompts],
            "translated_semantic_patch_paths": [str(p) for p in translated_semantic_paths],
        },
        "application": {
            "final_patch_paths": [str(p) for p in final_paths],
            "final_patch_stats": _patch_stats([p for p in final_paths if p.suffix == ".json"]),
            "translated_patch_paths": [str(p) for p in translated_paths],
            "translated_patch_stats": _patch_stats(translated_paths),
            "applied_diff_paths": [str(p) for p in applied_paths],
            "applied_diff_chars": sum(len(_read_text(p)) for p in applied_paths),
            "applied_diff_contains_tod_resource_paths": any("references/tod_" in _read_text(p) for p in applied_paths),
        },
        "batch_summaries": summaries,
    }


def audit_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    evidence_paths = sorted(run_dir.rglob("trajectory_evidence.json"))
    response_prompts = _load_response_prompts(run_dir)
    batches = [_audit_batch(path.parent, response_prompts) for path in evidence_paths]
    logs = [p for p in _files(run_dir, ("*.log", "*.txt")) if p.stat().st_size < 50_000_000]
    log_text = "\n".join(_read_text(path) for path in logs)
    return {
        "run_dir": str(run_dir), "hybrid_batches_found": len(batches),
        "llm_response_prompt_files_found": len(response_prompts),
        "analysis_prompt_tags_found": {
            tag: sum(row["tag"] == tag for row in response_prompts) for tag in sorted(ANALYSIS_TAGS)
        },
        "log_files": [str(p) for p in logs],
        "log_markers": {
            "reduce_merge_failed": log_text.count("merge failed"),
            "reduce_first_patch_fallback": log_text.count("forced merge failed, returning first patch"),
            "reduce_preserve_all_fallback": log_text.count("deterministically preserving all"),
            "apply_skipped_edits": log_text.count("edit(s) skipped by programmatic apply"),
        },
        "batches": batches,
        "limitations": [
            "Analysis prompts are associated with batches by conversation ID text matching; confirm IDs are unique across batches.",
            "Only persisted prompt/patch/diff artifacts can be audited; missing artifacts are reported as absent, not inferred.",
            "Resource filename visibility proves prompt inclusion, not that the model used the resource correctly.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Trace2Skill Hybrid Audit: {report['run_dir']}", "",
        f"Batches: {report['hybrid_batches_found']}",
        f"Saved raw LLM prompts: {report['llm_response_prompt_files_found']}",
        f"Analysis calls: {report['analysis_prompt_tags_found']}",
        f"Run log markers: {report['log_markers']}", "",
    ]
    if not report["batches"]:
        lines.append("No `trajectory_evidence.json` was found. Run this on the output run root, not the code checkout.")
    for batch in report["batches"]:
        ev, analysis, map_info, reduce, app = (batch[k] for k in ("evidence", "analysis", "map", "reduce", "application"))
        lines.extend([
            f"## {batch['batch_dir']}", "",
            f"- Evidence: {ev['conversation_count']} conversations, {ev['trajectory_turn_count']} turns, {ev['prefix_plus_prediction_chars']} chars; forbidden fields: {ev['forbidden_trajectory_fields'] or 'none'}",
            f"- Analysis: {analysis['saved_raw_prompt_calls_matched']} raw calls, {analysis['unique_conversations_in_analysis_prompts']}/{ev['conversation_count']} conversations represented; parsed outputs: {analysis['parsed_analysis_artifacts']}; unmatched: {analysis['evidence_gap_ids']}",
            f"- MAP: {map_info['prompt_sample_count']} prompt samples, resource mentions {map_info['resource_names_mentioned']}, JSON patches {map_info['patches']['file_count']} / {map_info['patches']['total_edits']} edits; semantic patches {len(map_info['semantic_patch_files'])}",
            f"- REDUCE: {reduce['prompt_sample_count']} prompt samples, JSON merged patches {reduce['merged_patch_files']['file_count']} / {reduce['merged_patch_files']['total_edits']} edits; semantic patches {len(reduce['semantic_patch_files'])}; {reduce['parse_failure_count']} parse failures",
            f"- TRANSLATE: {batch['translation']['prompt_sample_count']} prompt samples; semantic translated artifacts {len(batch['translation']['translated_semantic_patch_paths'])}",
            f"- Apply: final patch {app['final_patch_stats']['file_count']} file(s), translated {app['translated_patch_stats']['file_count']} file(s) / {app['translated_patch_stats']['total_edits']} edits; applied diff {app['applied_diff_chars']} chars",
            "",
        ])
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Hybrid run output root")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    report = audit_run(args.run_dir)
    markdown = render_markdown(report)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
