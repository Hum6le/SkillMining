#!/usr/bin/env python3
"""Read-only AWM transport/output audit. Stdlib only; no imports of LLM/config.

Prints metadata and counts, NEVER prompts, edit bodies, credentials or raw tails.
Provider chunks and assembled validation responses are listed separately: no
unreliable nearest-file call matching is inferred.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def extract_text(record):
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return ""
    if isinstance(record.get("choices"), list) and record["choices"]:
        choice = record["choices"][0]
        if isinstance(choice, dict):
            value = extract_text(choice.get("message"))
            if value:
                return value
    for key in ("raw_response", "content", "text", "answer", "data", "res", "output", "result"):
        if key in record:
            value = extract_text(record[key])
            if value:
                return value
    return ""


def metadata(record):
    """Only expose known termination/timing/token fields, not arbitrary values."""
    reasons, durations, tokens = [], [], []
    def walk(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                location = f"{path}.{key}"
                if key in {"finish_reason", "stop_reason", "termination_reason"} and isinstance(child, (str, type(None))):
                    reasons.append({"path": location, "value": child[:80] if isinstance(child, str) else None})
                if key in {"elapsed_ms", "duration_ms", "latency_ms", "elapsed_seconds", "duration_seconds", "latency_seconds"} and isinstance(child, (int, float)):
                    durations.append({"path": location, "value": child})
                if key in {"prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "total_tokens"} and isinstance(child, (int, float)):
                    tokens.append({"path": location, "value": child})
                walk(child, location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
    walk(record, "response")
    return {"termination_fields": reasons, "explicit_duration_fields": durations, "token_fields": tokens,
            "finish_reason_available": bool(reasons), "duration_available": bool(durations)}


def output_shape(text):
    """Count only structurally observed edits; never treat guesses as writes."""
    lines = text.lstrip().splitlines(keepends=True)
    result = {"chars": len(text), "ends_with_newline": text.endswith(("\n", "\r")),
              "has_exact_header": bool(lines and lines[0].rstrip("\r\n") == "AWM_EDIT_PLAN_V1"),
              "has_final_end_plan": bool(lines and lines[-1].strip() == "END_PLAN"),
              "end_edit_markers": sum(line.rstrip("\r\n") == "END_EDIT" for line in lines),
              "end_plan_markers": sum(line.rstrip("\r\n") == "END_PLAN" for line in lines)}
    if not result["has_exact_header"]:
        try:
            plan = json.loads(text)
            operations = plan.get("operations") if isinstance(plan, dict) else None
            if isinstance(operations, list):
                result.update(format="json", proposed_edits=len(operations),
                              missing_op_edits=[i for i, op in enumerate(operations, 1)
                                                if not isinstance(op, dict) or not isinstance(op.get("op"), str)])
                return result
        except (ValueError, TypeError):
            pass
        result.update(format="nonframed_or_suffix", proposed_edits=None)
        return result
    edits, cursor = [], 1
    while cursor < len(lines):
        header = lines[cursor].rstrip("\r\n")
        if header == "END_PLAN":
            break
        if not header.startswith("EDIT "):
            result["scan_stopped"] = "expected_edit_header_or_end_plan"
            break
        try:
            edit = json.loads(header[5:])
        except ValueError:
            result["scan_stopped"] = "incomplete_or_invalid_metadata"
            break
        if not isinstance(edit, dict):
            result["scan_stopped"] = "metadata_not_object"
            break
        cursor += 1
        body = []
        while cursor < len(lines) and lines[cursor].rstrip("\r\n") not in {"END_EDIT", "END_PLAN"}:
            body.append(lines[cursor])
            cursor += 1
        closed = cursor < len(lines) and lines[cursor].rstrip("\r\n") == "END_EDIT"
        value = "".join(body)
        if closed:
            value = value[:-2] if value.endswith("\r\n") else value[:-1] if value.endswith(("\n", "\r")) else value
        op = edit.get("op")
        edits.append({"edit": len(edits) + 1,
                      "op": op if op in ("replace", "delete", "insert_before", "insert_after") else "<missing/invalid>",
                      "body_chars_received": len(value), "closed": closed})
        if not closed:
            result["scan_stopped"] = "open_edit_at_end" if cursor == len(lines) else "end_plan_before_end_edit"
            break
        cursor += 1
    result.update(format="framed", observed_edit_headers=len(edits),
                  completed_edits=sum(edit["closed"] for edit in edits), edits=edits,
                  edit_op_counts=dict(Counter(edit["op"] for edit in edits)))
    return result


def source_settings(project):
    """Static source audit only: no config execution or credential disclosure."""
    result = {"project": str(project), "config_static_only": True, "files": []}
    config = project / "config.py"
    if config.is_file():
        try:
            tree = ast.parse(config.read_text(encoding="utf-8-sig"))
            safe, dynamic = {}, []
            allowed = {"provider", "model", "timeout", "max_tokens", "max_completion_tokens", "max_output_tokens"}
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(isinstance(target, ast.Name) and target.id == "LLM_CONFIG" for target in targets) and isinstance(node.value, ast.Dict):
                        for key, value in zip(node.value.keys, node.value.values):
                            if isinstance(key, ast.Constant) and key.value in allowed:
                                if isinstance(value, ast.Constant):
                                    safe[key.value] = value.value
                                else:
                                    dynamic.append(key.value)
            result["config_literal_settings"] = safe
            result["config_dynamic_keys_not_evaluated"] = dynamic
        except (OSError, ValueError, SyntaxError):
            result["config_read_error"] = "could_not_parse_config"
    for relative in ("llm.py", "llm_new.py", "utils/llm_utils.py", "skill_mining/llm_utils.py"):
        path = project / relative
        if not path.is_file():
            continue
        entry = {"path": str(path), "timeout_expressions": []}
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for call in ast.walk(node):
                        if isinstance(call, ast.Call):
                            for keyword in call.keywords:
                                if keyword.arg == "timeout":
                                    entry["timeout_expressions"].append({"function": node.name,
                                        "line": call.lineno, "expression": ast.unparse(keyword.value)})
            # Report actual literal delay schedules only, never an arbitrary dict.
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_RATE_LIMIT_RETRIES" for t in node.targets):
                    try:
                        delays = ast.literal_eval(node.value)
                        if isinstance(delays, (list, tuple)) and all(isinstance(x, (int, float)) for x in delays):
                            entry["rate_limit_retry_delays_seconds"] = delays
                    except (ValueError, TypeError):
                        pass
        except (OSError, ValueError, SyntaxError):
            entry["source_read_error"] = "could_not_parse_source"
        result["files"].append(entry)
    return result


def audit(directory, recent=3):
    log_root = directory / "llm_responses"
    rejected = list(log_root.rglob("*awm_induction_validation_error_response.json"))
    if not rejected:
        raise ValueError("No rejected induction response logs found")
    session = max(rejected, key=lambda path: path.stat().st_mtime_ns).parent
    def inspect(path):
        record = read_json(path)
        prompt_path = path.with_name(path.name.removesuffix("_response.json") + "_prompt.json")
        prompt = read_json(prompt_path) if prompt_path.is_file() else {}
        extra = prompt.get("extra", {})
        return {"file": str(path), "timestamp": record.get("timestamp"),
                "record_keys": sorted(record), "shape": output_shape(extract_text(record)),
                "metadata": metadata(record), "attempt_metadata": {key: extra[key] for key in (
                    "attempt", "plan_attempt", "continuation_calls", "chunk_chars", "prefix_chars",
                    "response_chars", "response_mode", "continuable_incomplete", "error") if key in extra}}
    def latest(pattern):
        return sorted(session.glob(pattern), key=lambda path: int(path.name.split("_", 1)[0]))[-recent:]
    return {"read_only": True, "session": str(session),
            "assembled_rejected_responses": [inspect(path) for path in latest("*awm_induction_validation_error_response.json")],
            "provider_response_chunks_unmatched": [inspect(path) for path in latest("*_skill_induction_response.json")],
            "limitations": ["Rows are recent events, not guaranteed to be the same batch. Provider chunks are not automatically matched to assembled plans.",
                "These logger timestamps are written AFTER the call returns; their difference is NOT LLM request latency.",
                "Missing finish_reason/duration means NOT LOGGED, not stop/timeout/length.",
                "Edit/body counts describe generated proposals, not applied changes. A rejected atomic plan is not partially written.",
                "HTTP timeout, model output-token budget and local continuation-count budget are distinct limits."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--recent", type=int, default=3)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.recent < 1:
        parser.error("--recent must be positive")
    try:
        report = audit(args.directory, args.recent)
        report["client_configuration_audit"] = source_settings(args.project_root)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Audit failed ({type(exc).__name__}); check directory and JSON logs.\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
