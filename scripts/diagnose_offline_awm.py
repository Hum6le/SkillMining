"""Read-only offline-AWM diagnostics. Standard library only; prints JSON to stdout.

Never prints dialogue text, replacement text, or full prompts/responses.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


FILES = {"skill.md", "action_rules.md", "slot_policies.md"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def catalog_from_prompt(prompt):
    """Only inspect the current catalog, NEVER IDs echoed in rejected plans."""
    marker = "## Current Editable Snapshot\n"
    if marker not in prompt:
        return {}
    snapshot = prompt.split(marker, 1)[1].split("\n\n## Read-only Reference", 1)[0]
    catalog, name = {}, None
    for line in snapshot.splitlines():
        if line.startswith("## ") and line[3:].strip() in FILES:
            name = line[3:].strip()
        match = re.match(r"^\[([^\]\s]+)\] ", line)
        if match and name:
            catalog[match[1]] = name
    return catalog


def response_text(data):
    if isinstance(data.get("raw_response"), str):
        return data["raw_response"]
    choices = data.get("choices") or []
    if choices:
        return (choices[0].get("message") or {}).get("content") or ""
    return ""


def inspect_plan(raw, catalog, other_ids):
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        plan = json.loads(text)
    except ValueError as exc:
        return {"status": "invalid_json", "error": str(exc), "response_chars": len(raw)}
    if not isinstance(plan, dict) or not isinstance(plan.get("operations"), list):
        return {"status": "invalid_plan_shape", "response_chars": len(raw)}
    issues, legacy = [], 0
    for index, op in enumerate(plan["operations"], 1):
        if not isinstance(op, dict):
            issues.append({"edit": index, "reason": "non_object_operation"})
            continue
        if "target_id" not in op:
            legacy += 1
            continue
        for field in ("target_id", "end_target_id"):
            if field not in op:
                continue
            target, name = op[field], op.get("file")
            if not isinstance(target, str):
                reason = "non_text_id"
            elif (re.fullmatch(r"\[[0-9a-f]{16}:L[1-9][0-9]*\]", target)
                  and target[1:-1] in catalog):
                reason = ("display_brackets_only" if catalog[target[1:-1]] == name
                          else "display_brackets_and_wrong_file")
            elif target in catalog:
                if catalog[target] == name:
                    continue
                reason = "wrong_file"
            elif target in other_ids:
                reason = "id_seen_in_other_logged_snapshot"
            elif re.fullmatch(r"[0-9a-f]{16}:L[1-9][0-9]*", target):
                revision = target.split(":", 1)[0]
                reason = ("unknown_line_in_current_revision" if any(
                    key.startswith(revision + ":") for key in catalog) else "unknown_revision")
            else:
                reason = "malformed_or_placeholder_id"
            issues.append({"edit": index, "field": field, "file": name,
                           "target_id": target if isinstance(target, str) else None,
                           "reason": reason, "actual_file": catalog.get(target) if isinstance(target, str) else None})
    return {"status": "address_issues" if issues else "addresses_match_or_legacy",
            "operation_count": len(plan["operations"]), "legacy_operations": legacy,
            "issue_count": len(issues), "issues": issues[:8], "response_chars": len(raw)}


def checkpoint_report(directory, state):
    failures = state.get("failed_batches", [])
    indices = sorted({b["batch_index"] for b in failures if type(b.get("batch_index")) is int})
    cursor = state.get("completed_batches")
    result = {"processed_batch_cursor": cursor, "failure_records": len(failures),
              "failed_batch_numbers": [i + 1 for i in indices],
              "last_error_counts": dict(Counter(b.get("error", "unknown") for b in failures))}
    tail = bool(indices) and type(cursor) is int and indices == list(range(indices[0], cursor))
    result["failures_are_contiguous_processed_tail"] = tail
    if not indices:
        return result
    path = directory / "batches" / f"batch_{indices[0] + 1:04d}.json"
    result["first_failed_rollout_cache"] = str(path)
    if path.is_file():
        try:
            identity = read_json(path).get("identity", {})
            expected = {"scope_hash": digest(state["scope"]),
                        "workflow_hash": digest(state["workflow"].strip()),
                        "cards_hash": digest(state["cards"]), "index": indices[0]}
            checks = {key: identity.get(key) == value for key, value in expected.items()}
            result["current_state_matches_first_failed_cache"] = checks
            result["tail_restart_precheck_passed"] = tail and all(checks.values())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["cache_read_error"] = str(exc)
    result["note"] = "Precheck only; no cursor/result/state changed. Missing historical logs are not evidence of absence."
    return result


def diagnose(directory, examples=6, latest_run=False, recent=None):
    state = read_json(directory / "awm_state.json")
    log_root = directory / "llm_responses"
    selected_run = None
    if latest_run:
        # UUID names do not encode chronology. Use the latest *induction*
        # artifact, not evaluation/rollout logs or lexical UUID ordering.
        induction_logs = list(log_root.rglob("*awm_induction_*_prompt.json"))
        if induction_logs:
            selected_run = max(induction_logs, key=lambda p: p.stat().st_mtime_ns).parent
            log_root = selected_run
        else:
            raise ValueError("No induction logs found; cannot select latest run")
    paths = sorted(log_root.rglob("*awm_induction_validation_error_prompt.json"),
                   key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
    available_attempts = len(paths)
    if recent is not None:
        paths = paths[:recent]
    counts, reasons, samples, errors = Counter(), Counter(), [], []
    # Collect ID metadata only, not full prompt bodies, across logged attempts.
    all_ids = set()
    for path in paths:
        try:
            record = read_json(path)
            prompt = "\n".join(m.get("content", "") for m in record.get("messages", []) if isinstance(m, dict))
            all_ids.update(catalog_from_prompt(prompt))
        except (OSError, ValueError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    for path in paths:
        try:
            record = read_json(path)
            prompt = "\n".join(m.get("content", "") for m in record.get("messages", []) if isinstance(m, dict))
            catalog = catalog_from_prompt(prompt)
            response_path = path.with_name(path.name.removesuffix("_prompt.json") + "_response.json")
            if not response_path.is_file():
                counts["missing_response_log"] += 1
                continue
            result = inspect_plan(response_text(read_json(response_path)), catalog, all_ids)
            result.update(prompt_log=str(path), response_log=str(response_path),
                          timestamp=record.get("timestamp"),
                          prompt_has_bracket_fix_instructions="Catalog brackets are display delimiters" in prompt,
                          attempt=record.get("extra", {}).get("attempt"),
                          validator_error=record.get("extra", {}).get("error"),
                          catalog_lines=len(catalog),
                          catalog_lines_per_file=dict(Counter(catalog.values())))
            feedback = record.get("extra", {}).get("edit_repair_context", {})
            result["validator_address_feedback"] = {key: feedback[key] for key in (
                "edit_index", "execution_index", "file", "target_id", "end_target_id",
                "target_present", "end_target_present", "actual_file", "address_checks") if key in feedback}
            counts[result["status"]] += 1
            for issue in result.get("issues", []):
                reasons[issue["reason"]] += 1
            # Representative examples by status/reason, then fill remaining slots.
            signature = (result["status"], tuple(sorted({i["reason"] for i in result.get("issues", [])})))
            if signature not in {sig for sig, _ in samples}:
                samples.append((signature, result))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return {"directory": str(directory), "read_only": True,
            "selected_log_run": str(selected_run) if selected_run else "all",
            "available_rejected_attempts_in_selection": available_attempts,
            "recent_limit": recent,
            "checkpoint": checkpoint_report(directory, state),
            "logged_rejected_attempts": len(paths), "attempt_status_counts": dict(counts),
            "address_issue_counts_first_8_per_attempt": dict(reasons),
            "examples": [row for _, row in samples[:examples]], "log_read_errors": errors[:10],
            "limitations": "Attempt counts are not batch counts. Logs have no reliable batch ID; no automatic mapping inferred. Address checks do not validate edit semantics, overlaps or headings. JSON errors alone do not prove truncation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="offline_awm output directory")
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--latest-run", action="store_true", help="Inspect only the most recently written induction run")
    parser.add_argument("--recent", type=int, help="Inspect only the N most recent rejected attempts")
    args = parser.parse_args()
    try:
        if args.recent is not None and args.recent <= 0:
            parser.error("--recent must be positive")
        report = diagnose(args.directory, max(0, args.examples), args.latest_run, args.recent)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Diagnostic failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
