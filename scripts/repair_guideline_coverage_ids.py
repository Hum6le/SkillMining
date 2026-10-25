"""Repair failed coverage responses in place, preserving original failures.

Duplicate IDs are removed only within an evidence list. Unknown IDs are never
guessed or dropped: failed batches are rejudged with exact short aliases, then
mapped to the original frozen IDs and passed through the original validator.
The source, extracted rules and input-freeze files are not changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.extract_skill_knowledge import Gate, dump, read, sha
from scripts.benchmark_abcd_skill_guidelines import load_inputs, export_results, review_template
from scripts.run_abcd_guideline_batch import METHODS, aggregate, collect_coverage
from scripts.repair_guideline_citations import configure, extraction_locations, wait_for_batch
from skill_mining.guideline_coverage import coverage_messages, validate_assessments

VERSION = "coverage_id_repair_v1"
ID_FIELDS = ("support_rule_ids", "conflict_rule_ids", "condition_rule_ids")
ROW_FIELDS = ("reference_id", "status", "reason", *ID_FIELDS)


def parse(raw):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw or "").strip(), flags=re.I | re.S)
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("assessments"), list):
        raise ValueError("Expected JSON object with assessments list")
    return value


def normalize_ids(raw, units, rules, unit_map=None, rule_map=None):
    value = deepcopy(parse(raw))
    allowed = {r["rule_id"] for r in rules}
    duplicates = 0
    for row in value["assessments"]:
        if not isinstance(row, dict):
            raise ValueError("Expected an assessment object")
        if unit_map is not None:
            rid = row.get("reference_id")
            if not isinstance(rid, str) or rid not in unit_map:
                raise ValueError("Unknown reference alias: " + repr(rid))
            row["reference_id"] = unit_map[rid]
        for field in ID_FIELDS:
            ids = row.get(field)
            if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
                raise ValueError("Expected string ID list in " + field)
            if rule_map is not None:
                unknown = [x for x in ids if x not in rule_map]
                if unknown:
                    raise ValueError("Unknown rule alias in " + field + ": " + repr(unknown))
                ids = [rule_map[x] for x in ids]
            unknown = [x for x in ids if x not in allowed]
            if unknown:
                raise ValueError("Unknown rule ID in " + field + ": " + repr(unknown))
            unique = list(dict.fromkeys(ids))
            duplicates += len(ids) - len(unique)
            row[field] = unique
    canonical = json.dumps({"assessments": [{k: r[k] for k in ROW_FIELDS} for r in value["assessments"]]}, ensure_ascii=False)
    return canonical, validate_assessments(canonical, units, rules), duplicates


def alias_messages(units, rules, error=None):
    unit_map = {f"U{i}": u["reference_id"] for i, u in enumerate(units, 1)}
    rule_map = {f"R{i}": r["rule_id"] for i, r in enumerate(rules, 1)}
    alias_units = [{**u, "reference_id": alias} for alias, u in zip(unit_map, units)]
    alias_rules = [{**r, "rule_id": alias} for alias, r in zip(rule_map, rules)]
    messages = coverage_messages(alias_units, alias_rules)
    messages[0]["content"] += " IDs use exact local aliases U1/U2 for reference units and R1/R2 for candidate rules. Copy only the supplied aliases. Never invent IDs. Keep each ID at most once in each list."
    payload = json.loads(messages[1]["content"])
    payload["allowed_reference_ids"] = list(unit_map)
    payload["allowed_rule_ids"] = list(rule_map)
    if error:
        payload["previous_validation_error"] = str(error)[:1500]
    messages[1]["content"] = json.dumps(payload, ensure_ascii=False)
    return messages, unit_map, rule_map


def audit_locations(batch, flow, state):
    if state.get("audit_dir"):
        return Path(state["audit_dir"])
    flat = batch / "flows" / flow
    if (flat / "coverage/protocol.json").exists():
        return flat  # literal-quote repaired batch layout
    if (flat / "repaired/coverage/protocol.json").exists():
        return flat / "repaired"
    return flat / "run"


def batch_origin(batch):
    if (batch / "batch_config.json").exists():
        return read(batch / "batch_config.json"), batch / "batch_config.json"
    config_path = batch / "citation_repair_config.json"
    config = read(config_path)
    old_path = Path(config["original_batch_dir"]) / "batch_config.json"
    if sha(old_path) != config["original_batch_config_sha256"]:
        raise ValueError("Original batch config changed")
    return read(old_path), config_path


def diagnose(saved, rules):
    allowed = {r["rule_id"] for r in rules}
    output = []
    for attempt in saved.get("attempts", []):
        try:
            value = parse(attempt.get("raw", ""))
            for row in value["assessments"]:
                for field in ID_FIELDS:
                    ids = row.get(field, [])
                    if isinstance(ids, list) and all(isinstance(x, str) for x in ids):
                        unknown = list(dict.fromkeys(x for x in ids if x not in allowed))
                        duplicate_count = len(ids) - len(set(ids))
                        if unknown or duplicate_count:
                            output.append({"reference_id": row.get("reference_id"), "field": field,
                                           "unknown_ids": unknown, "duplicate_count": duplicate_count})
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return output


def repair_one(path, method, flow, units, rules, args, gate, client):
    saved = read(path) if path.exists() else {"status": "missing", "attempts": []}
    if saved.get("status") == "complete":
        validate_assessments(saved["raw"], units, rules)
        return "cached"
    repair_dir = path.parent.parent / "id_repairs" / path.stem
    repair_dir.mkdir(parents=True, exist_ok=True)
    if not (repair_dir / "original.json").exists():
        dump(repair_dir / "original.json", saved)
    dump(repair_dir / "diagnostics.json", diagnose(saved, rules))
    errors = []
    canonical = None
    mode = None
    removed = 0
    for attempt in reversed(saved.get("attempts", [])):
        try:
            canonical, _, removed = normalize_ids(attempt.get("raw", ""), units, rules)
            mode = "duplicate_id_deduplication" if removed else "revalidated_prior_raw"
            break
        except (ValueError, KeyError, TypeError):
            continue
    events = []
    counter = max([int(p.stem.rsplit("_", 1)[1]) for p in repair_dir.glob("request_*.json")] or [0])

    def judge(selected):
        nonlocal counter, removed
        error = None
        for _ in range(args.retries):
            messages, unit_map, rule_map = alias_messages(selected, rules, error)
            if len(json.dumps(messages, ensure_ascii=False)) > args.max_prompt_chars:
                raise ValueError("Prompt exceeds budget; candidate rules were not truncated")
            gate.wait()
            counter += 1
            raw = ""
            try:
                llm, workflow = client
                kwargs = {"model": args.model, "temperature": 0., "call_tag": "abcd_guideline_coverage_id_repair"}
                if workflow is not None:
                    kwargs["config"] = workflow
                raw = llm.chat(messages, **kwargs)
                _, checked, deduped = normalize_ids(raw, selected, rules, unit_map, rule_map)
                removed += deduped
                event = {"ok": True, "messages": messages, "raw_response": raw,
                         "unit_alias_map": unit_map, "rule_alias_map": rule_map, "duplicates_removed": deduped}
                dump(repair_dir / f"request_{counter:04d}.json", event)
                events.append({"ok": True, "request_file": f"request_{counter:04d}.json"})
                return checked
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                dump(repair_dir / f"request_{counter:04d}.json", {"ok": False, "messages": messages, "raw_response": raw,
                    "unit_alias_map": unit_map, "rule_alias_map": rule_map, "error": error})
                errors.append({"error": error, "raw": raw, "request_file": f"request_{counter:04d}.json"})
        if len(selected) > 1:
            midpoint = len(selected) // 2
            return judge(selected[:midpoint]) + judge(selected[midpoint:])
        raise RuntimeError("Single-unit coverage still failed: " + str(error))

    try:
        if canonical is None:
            if not rules:
                raw = json.dumps({"assessments": [{"reference_id": u["reference_id"], "status": "absent",
                    "reason": "Complete extraction produced no eligible explicit rules for this method/flow.",
                    "support_rule_ids": [], "conflict_rule_ids": [], "condition_rule_ids": []} for u in units]})
                canonical, _, _ = normalize_ids(raw, units, rules)
                mode = "empty_candidate_set"
            else:
                checked = judge(units)
                canonical = json.dumps({"assessments": [{k: r[k] for k in ROW_FIELDS} for r in checked]}, ensure_ascii=False)
                mode = "short_alias_rejudgment"
        checked = validate_assessments(canonical, units, rules)
        result = {"status": "complete", "method": method, "flow": flow, "reference_ids": [u["reference_id"] for u in units],
            "raw": canonical, "assessments": checked, "attempts": saved.get("attempts", []) + errors,
            "repair": {"version": VERSION, "mode": mode, "duplicates_removed": removed,
                       "original_file_sha256": sha(repair_dir / "original.json"), "script_sha256": sha(Path(__file__)),
                       "request_events": events, "evidence_semantics": "ID lists are sets; only duplicates removed. Unknown IDs require rejudgment."}}
        dump(path, result)
        return mode
    except Exception as exc:
        dump(path, {**saved, "status": "failed", "method": method, "flow": flow,
                    "reference_ids": [u["reference_id"] for u in units],
                    "attempts": saved.get("attempts", []) + errors,
                    "repair_error": f"{type(exc).__name__}: {exc}"})
        return "failed"


def run(args):
    batch = args.batch_dir.resolve()
    with wait_for_batch(batch):
        original, origin_path = batch_origin(batch)
        ref = read(batch / "reference_snapshot.json")
        if original["reference"] != ref:
            raise ValueError("Batch reference changed")
        previous = read(batch / "progress.json")["flows"]
        input_freezes = {}
        for flow in ref["subflows"]:
            freeze = audit_locations(batch, flow, previous.get(flow, {})) / "coverage/input_freeze.json"
            if freeze.exists():
                input_freezes[flow] = sha(freeze)
        contract = {"version": VERSION, "origin_config_sha256": sha(origin_path), "script_sha256": sha(Path(__file__)),
                    "coverage_core_sha256": sha(ROOT / "skill_mining/guideline_coverage.py"),
                    "reference_sha256": sha(batch / "reference_snapshot.json"), "provider": original["provider"], "model": original["model"],
                    "input_freezes": input_freezes}
        config_path = batch / "coverage_id_repair_config.json"
        if config_path.exists() and read(config_path) != contract:
            raise ValueError("Coverage-repair code/inputs changed; inspect before resuming")
        dump(config_path, contract)
        backup = batch / "coverage_id_repair_original"
        backup.mkdir(exist_ok=True)
        for name in ("aggregate_summary.json", "progress.json", "by_subflow.csv", "assessments.json"):
            if (batch / name).exists() and not (backup / name).exists():
                shutil.copyfile(batch / name, backup / name)
        states = deepcopy(previous)
        all_rows = {}
        client = None
        gate = Gate(args.qps)
        args.model = original["model"]
        for flow in ref["subflows"]:
            try:
                audit = audit_locations(batch, flow, previous.get(flow, {}))
                source, rules_dir = extraction_locations(audit)
                ref_path = audit / "reference/reference.json"
                local_ref, methods, candidates = load_inputs(ref_path, source, rules_dir, "procedural_rule")
                # This validates all frozen inputs and every already-successful raw response.
                all_rows[flow] = collect_coverage(audit, flow, ref, provider=original["provider"], model=original["model"], require_complete=False)
                protocol = read(audit / "coverage/protocol.json")
                units = [u for u in local_ref["units"] if u["eligibility"] == "eligible"]
                tasks, paths = [], []
                for method in methods:
                    for offset in range(0, len(units), protocol["batch_size"]):
                        path = audit / "coverage/batches" / f"{method}_{flow}_{offset:04d}.json"
                        paths.append(path)
                        if not path.exists() or read(path).get("status") != "complete":
                            tasks.append((path, method, units[offset:offset + protocol["batch_size"]]))
                if not tasks:
                    states[flow] = {"status": "reused", "audit_dir": str(audit)}
                    print(f"REUSED {flow}: all coverage batches verified", flush=True)
                    continue
                if tasks:
                    print(f"REPAIR {flow}: {len(tasks)} failed/missing batches; {len(paths) - len(tasks)} successful batches retained", flush=True)
                    states[flow] = {"status": "running", "phase": "coverage_id_repair", "audit_dir": str(audit)}
                    aggregate(batch, ref, all_rows, states)
                    if client is None:
                        extraction_config = read(rules_dir / "extraction_config.json")
                        client = configure({"provider": protocol["provider"], "model": protocol["model"],
                                            "workflow_id": protocol.get("workflow_id") or extraction_config.get("workflow_id")})
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        future_tasks = {pool.submit(repair_one, p, m, flow, selected, candidates[m, flow], args, gate, client): p for p, m, selected in tasks}
                        for future in as_completed(future_tasks):
                            print(f"{flow}/{future_tasks[future].name}: {future.result()}", flush=True)
                checked, summary = export_results(audit / "coverage", local_ref, METHODS, candidates, [p for p in paths if p.exists()])
                review_template(audit / "coverage", local_ref, METHODS, checked)
                all_rows[flow] = [{**r, "method": r["method"], "flow": flow} for r in checked]
                if not summary["complete"]:
                    raise RuntimeError("Some coverage batches still failed; inspect coverage/id_repairs/*/request_*.json")
                states[flow] = {"status": "completed", "audit_dir": str(audit)}
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                states[flow] = {"status": "failed", "error": str(exc)}
                print(f"FAILED {flow}: {exc}", flush=True)
            finally:
                aggregate(batch, ref, all_rows, states)
        dump(batch / "coverage_id_repair_summary.json", {"version": VERSION, "flows": states,
             "complete": all(s["status"] in ("completed", "reused") for s in states.values())})
        if any(s["status"] == "failed" for s in states.values()):
            raise RuntimeError("Some subflows still failed; successful batches retained. See progress.json and id_repairs diagnostics.")
        print("Coverage complete: " + str(batch / "aggregate_summary.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--qps", type=float, default=1.8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-prompt-chars", type=int, default=400000)
    args = parser.parse_args()
    if args.workers < 1 or args.qps <= 0 or args.retries < 1 or args.max_prompt_chars < 1:
        parser.error("Positive workers/QPS/retries/prompt budget required")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print("Coverage repair stopped: " + str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
