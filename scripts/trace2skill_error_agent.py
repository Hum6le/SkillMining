#!/usr/bin/env python3
"""Multi-round, file-reading error-analysis agent for a failed hybrid case.

Example:
  python scripts/trace2skill_error_agent.py \
    --run-dir outputs/test_trace2skill_hybrid_0922 \
    --conversation-id 12345 --model qwen3.6-flash
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SYSTEM_PROMPT = """You are a skeptical research error-analysis agent investigating why Trace2Skill failed to improve a particular ToD action prediction. You have read-only file tools. Plan your next evidence-gathering step, inspect artifacts, and only then propose a method change.

Do not assume the implementation is wrong because a result is unchanged. Separate evidence from hypotheses. Investigate relevant possibilities when artifacts permit: (1) the training rollout is not the evaluated failure; (2) failure localization or trajectory alignment is wrong; (3) analyzer sees diagnosis but not decisive prefix/current prediction/gold labels; (4) analysis output is lost or malformed before MAP; (5) MAP sees the evidence but patch is generic, contradictory, or edits the wrong file; (6) REDUCE/translation/apply drops or weakens a useful edit; (7) the action-card runtime path differs from the skill files being evolved; (8) evaluation/parser/scoring mismatch; (9) sample is irreducible from current artifacts.

The current user's runtime concept is a unified per-action card containing both action rules and slot-binding policy. Distinguish that runtime card from separate source/reference files and from Trace2Skill's evolvable skill folder; do not conflate them.

On every turn return exactly one JSON object, no markdown fence. For an inspection step:
{"action":"read_file","path":"run/relative/path","start_line":1,"max_lines":120}
{"action":"list_files","path":"run/relative/dir","glob":"*.json","max_items":80}
{"action":"search_text","path":"run/relative/dir","query":"literal or regex","max_files":40}
When enough evidence is gathered, return:
{"action":"finish","diagnosis":"...","evidence":[{"claim":"...","files":["..."],"quotes_or_values":"..."}],"uncertainties":["..."],"proposal":{"name":"...","mechanism":"...","why_trace2skill_misses_it":"...","minimal_implementation":"...","test":"...","ablation":"..."}}

Use one tool action per turn. Read at least two distinct files before finishing. Prefer the exact case's analysis prompt/output, the MAP prompt/patch that consumed its analysis, then final/translated patch and applied diff; also inspect prediction/runtime traces where available. Never invent file contents. If evidence is absent, say so and request the smallest next artifact."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _json_from_response(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1))
    left, right = text.find("{"), text.rfind("}")
    if left >= 0 and right > left:
        candidates.append(text[left:right + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return None


def _find_evidence_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("trajectory_evidence.json"))
    if not candidates:
        raise FileNotFoundError(f"No trajectory_evidence.json under {run_dir}")
    if len(candidates) > 1:
        raise ValueError(
            "Multiple hybrid batches found; pass --evidence-json with the desired batch's trajectory_evidence.json"
        )
    return candidates[0]


def _select_case(evidence_path: Path, conversation_id: str) -> dict[str, Any]:
    records = _load_json(evidence_path)
    if not isinstance(records, list):
        raise ValueError(f"Invalid evidence list: {evidence_path}")
    for row in records:
        if isinstance(row, dict) and str(row.get("conversation_id")) == str(conversation_id):
            return row
    raise KeyError(f"conversation_id {conversation_id!r} not found in {evidence_path}")


def _relative_virtual(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


class ReadOnlyFileTools:
    def __init__(self, run_dir: Path, repo_dir: Path):
        self.roots = {"run": run_dir.resolve(), "repo": repo_dir.resolve()}

    def _resolve(self, virtual: str) -> tuple[Path, str]:
        prefix, sep, relative = virtual.partition("/")
        if not sep or prefix not in self.roots:
            raise ValueError("Use a virtual path beginning with run/ or repo/")
        root = self.roots[prefix]
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            raise ValueError("Path escapes the allowed root")
        return target, prefix

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        action = request.get("action")
        target, _ = self._resolve(str(request.get("path", "")))
        if action == "read_file":
            if not target.is_file():
                return {"error": "not a file", "path": str(request.get("path"))}
            start = max(1, int(request.get("start_line", 1)))
            limit = min(250, max(1, int(request.get("max_lines", 120))))
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, min(len(lines), start + limit - 1) + 1))
            return {"path": str(request.get("path")), "total_lines": len(lines), "shown_from": start, "content": body[:40000]}
        if action == "list_files":
            if not target.is_dir():
                return {"error": "not a directory", "path": str(request.get("path"))}
            pattern = str(request.get("glob", "*"))
            limit = min(120, max(1, int(request.get("max_items", 80))))
            items = sorted(target.rglob(pattern))[:limit]
            return {"path": str(request.get("path")), "items": [
                str(request.get("path")).rstrip("/") + "/" + p.relative_to(target).as_posix()
                + ("/" if p.is_dir() else "") for p in items
            ]}
        if action == "search_text":
            if not target.is_dir():
                return {"error": "search path must be a directory"}
            query = str(request.get("query", ""))
            if not query or len(query) > 200:
                return {"error": "query must contain 1-200 characters"}
            regex = re.compile(query, re.IGNORECASE)
            max_files = min(60, max(1, int(request.get("max_files", 40))))
            extensions = {".md", ".txt", ".json", ".jsonl", ".patch", ".log", ".py"}
            matches = []
            for path in sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in extensions)[:max_files]:
                try:
                    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if regex.search(line):
                            matches.append({
                                "path": str(request.get("path")).rstrip("/") + "/" + path.relative_to(target).as_posix(),
                                "line": line_no, "text": line[:600],
                            })
                            if len(matches) >= 80:
                                break
                except OSError:
                    continue
                if len(matches) >= 80:
                    break
            return {"query": query, "matches": matches}
        return {"error": f"unknown action: {action}"}


def _seed_context(run_dir: Path, evidence_path: Path, case: dict[str, Any]) -> dict[str, Any]:
    batch_dir = evidence_path.parent
    trajectory = case.get("trajectory", [])
    divergence = case.get("first_divergence") or {}
    selected_turns = []
    turn_id = divergence.get("action_turn_index") if isinstance(divergence, dict) else None
    if isinstance(trajectory, list):
        if turn_id is not None:
            selected_turns = [t for t in trajectory if isinstance(t, dict) and abs(int(t.get("turn_index", -999)) - int(turn_id)) <= 2]
        if not selected_turns:
            selected_turns = [t for t in trajectory if isinstance(t, dict)][:4]
    compact_turns = [{k: t.get(k) for k in ("turn_index", "context", "prediction", "predicted_action", "predicted_slots") if k in t} for t in selected_turns]
    likely = []
    for subdir in ("error_analysis", "success_analysis", "evolution", "failure_logs", "success_logs"):
        path = batch_dir / subdir
        if path.exists():
            likely.append(_relative_virtual(path, run_dir) + "/")
    likely.extend([
        _relative_virtual(batch_dir / "error_analysis_parsed.json", run_dir),
        _relative_virtual(batch_dir / "success_analysis_parsed.json", run_dir),
        _relative_virtual(batch_dir / "batch_summary.json", run_dir),
        _relative_virtual(run_dir / "trace2skill_hybrid_skill" / "SKILL.md", run_dir),
    ])
    return {
        "run_dir": str(run_dir), "batch_dir": str(batch_dir),
        "evidence_file": _relative_virtual(evidence_path, run_dir),
        "conversation_id": str(case.get("conversation_id", "")),
        "error_type": case.get("error_type"), "first_divergence": divergence,
        "graph_context": case.get("graph_context"), "action_sequence": case.get("action_sequence"),
        "near_divergence_prefix_prediction": compact_turns,
        "likely_artifact_paths": likely,
    }


def run_agent(
    *, run_dir: Path, repo_dir: Path, evidence_path: Path, conversation_id: str,
    model: str, max_rounds: int,
) -> dict[str, Any]:
    from llm import chat

    case = _select_case(evidence_path, conversation_id)
    seed = _seed_context(run_dir, evidence_path, case)
    tools = ReadOnlyFileTools(run_dir, repo_dir)
    observations: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    raw_rounds: list[dict[str, Any]] = []
    for round_index in range(1, max_rounds):
        user_content = json.dumps({
            "round": round_index, "max_rounds": max_rounds,
            "selected_failure_case": seed,
            "observations_so_far": observations[-8:],
            "instruction": "Choose exactly one next read-only file action, or finish if evidence is sufficient.",
        }, ensure_ascii=False)
        raw = chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            model=model, temperature=0.1, call_tag="trace2skill_error_agent",
        )
        decision = _json_from_response(raw)
        raw_rounds.append({"round": round_index, "raw_response": raw, "decision": decision})
        if decision is None:
            observations.append({"round": round_index, "tool_result": {"error": "Model response was not valid JSON", "raw": raw[:4000]}})
            continue
        if decision.get("action") == "finish":
            read_paths = {
                row.get("request", {}).get("path")
                for row in observations
                if row.get("request", {}).get("action") == "read_file"
                and isinstance(row.get("tool_result", {}).get("content"), str)
            }
            if len(read_paths) >= 2:
                final = decision
                break
            observations.append({"round": round_index, "tool_result": {"error": "Read at least two distinct files before finishing", "attempted_finish": decision}})
            continue
        result = tools.execute(decision)
        observations.append({"round": round_index, "request": decision, "tool_result": result})
    if final is None:
        # One final synthesis call gets the evidence gathered in the last tool round.
        raw = chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps({
                "selected_failure_case": seed, "observations_so_far": observations[-10:],
                "instruction": "Synthesize only if at least two distinct files were successfully read. Otherwise return a structured diagnosis saying evidence is insufficient and list the smallest missing artifact; do not invent a cause.",
            }, ensure_ascii=False)}], model=model, temperature=0.1,
            call_tag="trace2skill_error_agent_final",
        )
        raw_rounds.append({"round": "final", "raw_response": raw, "decision": _json_from_response(raw)})
        final = _json_from_response(raw)
    return {
        "model": model, "seed": seed, "observations": observations,
        "final": final, "rounds": raw_rounds,
    }


def render_report(result: dict[str, Any]) -> str:
    final = result.get("final") or {}
    lines = [
        f"# Trace2Skill Error Analysis: {result['seed']['conversation_id']}", "",
        f"- Run: `{result['seed']['run_dir']}`",
        f"- Type: `{result['seed'].get('error_type')}`",
        f"- Evidence files read: {len(result.get('observations', []))}", "",
        "## Diagnosis", "", str(final.get("diagnosis", "No structured diagnosis returned.")), "",
        "## Evidence", "",
    ]
    for item in final.get("evidence", []) if isinstance(final.get("evidence"), list) else []:
        lines.append(f"- **{item.get('claim', 'Claim')}**: {item.get('quotes_or_values', '')} (files: {item.get('files', [])})")
    lines.extend(["", "## Uncertainties", ""])
    for item in final.get("uncertainties", []) if isinstance(final.get("uncertainties"), list) else []:
        lines.append(f"- {item}")
    proposal = final.get("proposal") if isinstance(final.get("proposal"), dict) else {}
    lines.extend(["", "## Proposal", ""])
    for key, title in (
        ("name", "Name"), ("mechanism", "Mechanism"),
        ("why_trace2skill_misses_it", "Why Trace2Skill misses it"),
        ("minimal_implementation", "Minimal implementation"),
        ("test", "Test"), ("ablation", "Ablation"),
    ):
        lines.extend([f"### {title}", "", str(proposal.get(key, "Not supplied.")), ""])
    lines.extend(["## Read Log", ""])
    for row in result.get("observations", []):
        request = row.get("request", {})
        lines.append(f"- Round {row['round']}: `{request.get('action')}` `{request.get('path')}`")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Run output root")
    parser.add_argument("--evidence-json", type=Path, help="Optional exact trajectory_evidence.json")
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--repo-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    evidence_path = args.evidence_json.resolve() if args.evidence_json else _find_evidence_file(run_dir)
    if not (3 <= args.max_rounds <= 20):
        parser.error("--max-rounds must be between 3 and 20")
    result = run_agent(
        run_dir=run_dir, repo_dir=args.repo_dir.resolve(), evidence_path=evidence_path,
        conversation_id=args.conversation_id, model=args.model, max_rounds=args.max_rounds,
    )
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "error_agent" / str(args.conversation_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "analysis.md").write_text(render_report(result), encoding="utf-8")
    print(render_report(result), end="")
    print(f"\nSaved: {output_dir / 'analysis.md'}")


if __name__ == "__main__":
    main()
