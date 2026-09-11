"""Analyze Trace2Skill module contributions and migration cost.

This is deliberately evidence-driven. A Trace2Skill run or an existing rollout
directory can provide measured artifacts; without one, the script still emits
a static inventory. Seed rollout is treated as a fixed input, never as an
ablation. Batch deltas are marked as *proxies* because they are not causal
leave-one-out ablations.

Examples:
  python scripts/analyze_trace2skill_modules.py
  python scripts/analyze_trace2skill_modules.py outputs/abcd_trace2skill_x
  python scripts/analyze_trace2skill_modules.py --rollouts outputs/foo/rollouts
  python scripts/analyze_trace2skill_modules.py RUN --output report.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

MODULES = {
    "seed_rollout": {
        "label": "Seed rollout",
        "paths": ["scripts/run_trace2skill_abcd.py"],
        "signals": ["seed_train_eval.json", "seed_test_eval.json"],
        "migration": "Keep as the frozen baseline and use its AST traces as input evidence.",
        "priority": "required",
    },
    "verified_failure_analysis": {
        "label": "Verified AST failure analysis",
        "paths": ["scripts/run_trace2skill_abcd.py", "Trace2Skill/analysis"],
        "signals": ["error_analysis", "failed_cases", "corrections"],
        "migration": "Port only the local AST mismatch/correction extractor; feed corrections into offline candidate induction.",
        "priority": "high",
    },
    "success_memory": {
        "label": "AST-success memory",
        "paths": ["scripts/run_trace2skill_abcd.py", "Trace2Skill/skill_evolver/success_evolving_agent.py"],
        "signals": ["success_analysis", "successful_cases"],
        "migration": "Use strict successful spans as positive prototypes and cluster them before LLM summarization.",
        "priority": "high",
    },
    "map_reduce_evolution": {
        "label": "MAP/REDUCE skill evolution",
        "paths": ["Trace2Skill/skill_evolver", "scripts/run_trace2skill_abcd.py"],
        "signals": ["parsed_error_analysis", "parsed_success_analysis", "changelog"],
        "migration": "Replace free-form folder edits with constrained patches to the offline library/specification.",
        "priority": "medium",
    },
    "iterative_batch_update": {
        "label": "Iterative batch update",
        "paths": ["scripts/run_trace2skill_abcd.py"],
        "signals": ["batch_history.json", "pre_skill_lines", "post_skill_lines"],
        "migration": "Apply updates at offline mining checkpoints, then validate once on held-out conversations.",
        "priority": "high",
    },
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _metric(obj: Any, *keys: str) -> float | None:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def _find_run(path: Path | None, subflow: str | None = None) -> Path | None:
    if path is None:
        return None
    path = path.resolve()
    if (path / "summary.json").is_file() and (not subflow or subflow in path.parts):
        return path
    candidates = sorted(path.rglob("summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if subflow:
        candidates = [p for p in candidates if subflow in p.parent.parts or subflow in p.parent.name]
    return candidates[0].parent if candidates else None


def _find_rollouts(path: Path | None, subflow: str | None = None) -> list[Path]:
    """Find rollout JSONs written by local Trace2Skill/online-refine scripts."""
    if path is None:
        return []
    path = path.resolve()
    if path.is_file() and path.suffix.lower() == ".json":
        return [path]
    patterns = ("seed_train_turns.json", "turns.json", "*.json")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.rglob(pattern))
    unique = {p.resolve() for p in files if p.is_file()}
    if subflow:
        unique = {p for p in unique if subflow in p.parts or subflow in p.parent.name}
    return sorted(unique, key=lambda p: str(p))


def _rollout_stats(files: list[Path]) -> dict[str, Any]:
    stats = {"files": [str(p) for p in files], "json_files": len(files),
             "records": 0, "turn_records": 0, "conversation_ids": set(),
             "seed_files": 0, "batch_files": 0}
    for path in files:
        payload = _read_json(path)
        if payload is None:
            continue
        if isinstance(payload, dict) and any(key.endswith("_metrics") for key in payload):
            stats.setdefault("metric_files", 0)
            stats["metric_files"] += 1
            continue
        if "seed_train_turns" in path.name:
            stats["seed_files"] += 1
        if "batch_" in path.name or "train_batches" in path.parts:
            stats["batch_files"] += 1
        rows = payload if isinstance(payload, list) else payload.get("turns", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            continue
        stats["records"] += 1
        stats["turn_records"] += len(rows)
        for row in rows:
            if isinstance(row, dict):
                for key in ("convo_id", "conversation_id", "dialogue_id", "instance_id"):
                    if row.get(key) is not None:
                        stats["conversation_ids"].add(str(row[key]))
                        break
    stats["conversation_ids"] = len(stats["conversation_ids"])
    return stats


def _usage(run: Path) -> dict[str, Any]:
    data = _read_json(run / "llm_usage.json") or {}
    # Support both current split_usage_summary and older flat counters.
    text = json.dumps(data, ensure_ascii=False)
    numbers = [int(x) for x in re.findall(r'"(?:total_tokens|prompt_tokens|completion_tokens|calls)"\s*:\s*(\d+)', text)]
    return {"file": str(run / "llm_usage.json") if data else None,
            "numeric_fields": len(numbers), "numeric_sum": sum(numbers), "raw": data}


def analyze(run: Path | None, rollout_files: list[Path] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "method": "trace2skill_module_contribution_analysis",
        "repo_root": str(ROOT),
        "run_dir": str(run) if run else None,
        "causal_warning": "Batch deltas are observational proxies. Causal contribution requires rerunning with one module disabled and identical seeds/data.",
        "modules": [],
        "ablation_scope": ["verified_failure_analysis", "success_memory", "map_reduce_evolution", "iterative_batch_update"],
        "fixed_inputs": ["seed_rollout", "existing rollout trajectory files"],
    }
    summary = _read_json(run / "summary.json") if run else None
    history = _read_json(run / "batch_history.json") if run else None
    if not isinstance(history, list) and isinstance(summary, dict):
        history = summary.get("batch_history", [])
    history = history if isinstance(history, list) else []
    rollout_stats = _rollout_stats(rollout_files or [])

    seed_test = _metric(summary, "seed_test", "summary", "ast_joint") if summary else None
    evolved_test = _metric(summary, "evolved_test", "summary", "ast_joint") if summary else None
    if seed_test is None and run:
        seed_test = _metric(_read_json(run / "seed_test_eval.json"), "summary", "ast_joint")
    if evolved_test is None and run:
        evolved_test = _metric(_read_json(run / "evolved_test_eval.json"), "summary", "ast_joint")

    batch_rows = []
    for row in history:
        ev = row.get("eval", {}) if isinstance(row, dict) else {}
        ast = _metric(ev, "summary", "ast_joint")
        batch_rows.append({"batch": row.get("batch"), "ast_joint_before_update": ast,
                           "failed_cases": row.get("failed_cases", 0),
                           "successful_cases": row.get("successful_cases", 0),
                           "changelog_entries": len(row.get("changelog", []) or []),
                           "skill_line_delta": (row.get("post_skill_lines") - row.get("pre_skill_lines"))
                           if isinstance(row.get("post_skill_lines"), int) and isinstance(row.get("pre_skill_lines"), int) else None})

    for name, spec in MODULES.items():
        evidence = {"files_present": [], "counts": {}}
        # Always report repository implementation points, even for a static run.
        for rel in spec["paths"]:
            path = ROOT / rel
            if path.exists():
                evidence["files_present"].append(rel)
        for rel in spec["signals"]:
            if run:
                matches = list(run.rglob(rel)) if "." not in rel else [run / rel]
                matches = [p for p in matches if p.exists()]
                if matches:
                    evidence["files_present"].extend(str(p.relative_to(run)) for p in matches[:20])
            if name == "verified_failure_analysis":
                evidence["counts"]["failed_cases"] = sum(int(r.get("failed_cases", 0)) for r in history)
            elif name == "success_memory":
                evidence["counts"]["successful_cases"] = sum(int(r.get("successful_cases", 0)) for r in history)
            elif name == "map_reduce_evolution":
                evidence["counts"]["changelog_entries"] = sum(len(r.get("changelog", []) or []) for r in history)
        module = {"name": name, "label": spec["label"], "priority": spec["priority"],
                  "migration": spec["migration"], "evidence": evidence}
        if name == "iterative_batch_update":
            module["evidence"]["batches"] = len(history)
            module["evidence"]["batch_rows"] = batch_rows
        if name == "seed_rollout":
            module["measured_seed_test_ast_joint"] = seed_test
            module["ablation"] = False
            module["role"] = "fixed input/baseline"
        if name in {"verified_failure_analysis", "success_memory", "map_reduce_evolution", "iterative_batch_update"}:
            module["measured_final_test_ast_joint"] = evolved_test
            module["proxy_gain_over_seed"] = (evolved_test - seed_test) if evolved_test is not None and seed_test is not None else None
            module["ablation"] = True
        report["modules"].append(module)

    report["run_metrics"] = {"seed_test_ast_joint": seed_test, "evolved_test_ast_joint": evolved_test,
                             "observed_gain": evolved_test - seed_test if evolved_test is not None and seed_test is not None else None,
                             "batches": len(history), "rollouts": rollout_stats,
                             "llm_usage": _usage(run) if run else None}
    report["recommended_offline_ablation_order"] = [
        {"ablation": "failure_analysis", "why": "Most directly converts observable AST errors into corrective candidates; cheap and locally verifiable."},
        {"ablation": "success_memory", "why": "Tests whether positive spans improve coverage beyond failure-only mining."},
        {"ablation": "iterative_update", "why": "Compare one-shot offline consolidation against checkpointed updates."},
        {"ablation": "map_reduce_evolution", "why": "Highest LLM/editing cost; test after evidence selection is established."},
    ]
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Trace2Skill module contribution report", "", f"Run: `{report['run_dir'] or 'static repository inventory'}`", "",
             "## Measured overview", "", "| Metric | Value |", "|---|---:|"]
    for key, value in report["run_metrics"].items():
        if key != "llm_usage": lines.append(f"| {key} | {value} |")
    lines += ["", "## Module evidence", "", "| Module | Priority | Evidence | Migration |", "|---|---|---|---|"]
    for m in report["modules"]:
        ev = m["evidence"].get("counts", {})
        if "batches" in m["evidence"]: ev["batches"] = m["evidence"]["batches"]
        lines.append(f"| {m['label']} | {m['priority']} | `{json.dumps(ev, ensure_ascii=False)}` | {m['migration']} |")
    lines += ["", "## Recommended ablation order", ""]
    lines += [f"{i}. **{x['ablation']}**: {x['why']}" for i, x in enumerate(report["recommended_offline_ablation_order"], 1)]
    lines += ["", "Caution: batch deltas are proxies, not causal module effects. Disable one module per rerun with fixed data, seed, and evaluation suite."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, help="Trace2Skill run dir, subflow dir, or parent containing summary.json")
    parser.add_argument("--subflow", help="Select the newest summary.json belonging to this subflow below RUN")
    parser.add_argument("--rollouts", type=Path, help="Existing rollout JSON, rollouts directory, or run root; no new rollout is executed")
    parser.add_argument("--output", type=Path, help="JSON output path; Markdown is written beside it")
    args = parser.parse_args()
    run = _find_run(args.run, args.subflow)
    rollout_root = args.rollouts or args.run or run
    rollout_files = _find_rollouts(rollout_root, args.subflow)
    report = analyze(run, rollout_files)
    output = args.output or ((report["run_dir"] and Path(report["run_dir"]) / "trace2skill_module_report.json") or ROOT / "outputs" / "trace2skill_module_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(output), "markdown": str(output.with_suffix('.md')), "run_dir": report["run_dir"], "observed_gain": report["run_metrics"]["observed_gain"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
