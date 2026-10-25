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
from scripts import benchmark_abcd_skill_guidelines as benchmark
from scripts import repair_guideline_citations as citation_repair
from scripts.run_abcd_guideline_batch import METHODS, aggregate, collect_coverage
from scripts.repair_guideline_citations import configure, extraction_locations, wait_for_batch
from skill_mining.guideline_coverage import coverage_messages, validate_assessments

VERSION = "coverage_id_repair_v1"
LEGACY_RESOLVER_SCRIPTS = {
    "71e8bde40f607fd01a9f3691be7f190eb2a49beb0b268b3edecd48f2f89d060a",
    "8f0a1ebfe7589a9747aebe78296ff5b131e5b84777b8b6a7b9864cb78fb4bf36",
    "137e52b7b6b5b208db0e0fc999008c5e2eb8c2adcab2e610535cba9949516182",
    "900c0c453d9124d78be7981c07487f81a318215992d498f5d943c63fbd218686",
}
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
    markers = ("latest_extraction.json", "rules/summary.json", "sources/inventory.json",
               "reference/reference.json", "coverage/protocol.json")
    def is_audit(path):
        return any((path / marker).is_file() for marker in markers)
    if state.get("audit_dir"):
        explicit = Path(state["audit_dir"])
        if is_audit(explicit):
            return explicit
        raise ValueError("Recorded audit directory has no saved audit artifacts: " + str(explicit))
    flat = batch / "flows" / flow
    # An unfinished extraction has no coverage/protocol.json yet. It still uses
    # the flat layout; never invent a /run suffix and mask its actual error.
    for candidate in (flat, flat / "repaired", flat / "run"):
        if is_audit(candidate):
            return candidate
    # Reused successful audits may live entirely outside this batch directory.
    backup = batch / "coverage_id_repair_original/progress.json"
    if backup.exists():
        old = read(backup).get("flows", {}).get(flow, {})
        if old.get("audit_dir") and is_audit(Path(old["audit_dir"])):
            return Path(old["audit_dir"])
    citation_config = batch / "citation_repair_config.json"
    if citation_config.exists():
        original = Path(read(citation_config)["original_batch_dir"])
        old_progress = original / "progress.json"
        if old_progress.exists():
            old = read(old_progress).get("flows", {}).get(flow, {})
            if old.get("audit_dir") and is_audit(Path(old["audit_dir"])):
                return Path(old["audit_dir"])
        for candidate in (original / "flows" / flow / "repaired", original / "flows" / flow / "run"):
            if is_audit(candidate):
                return candidate
    raise FileNotFoundError("No saved audit found for " + flow + "; checked flat, repaired, run and prior-stage locations")


def require_complete_extraction(audit):
    source, rules = extraction_locations(audit)
    summary_path = rules / "summary.json"
    summary = read(summary_path)
    if not summary.get("complete"):
        raise ValueError("Extraction is still incomplete at " + str(summary_path)
                         + f"; successful={summary.get('successful_chunks')}, failed={summary.get('failed_chunks')}; "
                         + json.dumps(summary.get("failed_chunk_details", []), ensure_ascii=False))
    return source, rules


def save_repair_contract(config_path, contract, *, newly_initialized=()):
    if config_path.exists():
        previous = read(config_path)
        if previous != contract:
            old_fixed = {k: v for k, v in previous.items() if k not in {"script_sha256", "input_freezes"}}
            new_fixed = {k: v for k, v in contract.items() if k not in {"script_sha256", "input_freezes"}}
            old_inputs, new_inputs = previous.get("input_freezes", {}), contract.get("input_freezes", {})
            inputs_preserved = all(new_inputs.get(k) == v for k, v in old_inputs.items())
            additions_allowed = set(new_inputs) - set(old_inputs) <= set(newly_initialized)
            same_or_legacy = previous.get("script_sha256") in LEGACY_RESOLVER_SCRIPTS or previous.get("script_sha256") == contract["script_sha256"]
            if not same_or_legacy or old_fixed != new_fixed or not inputs_preserved or not additions_allowed:
                raise ValueError("Coverage-repair code/inputs changed; inspect before resuming")
            history = config_path.with_name("coverage_id_repair_config.before_resolver_fix.json")
            if not history.exists():
                dump(history, previous)
            print("Updated compatible resume metadata; existing frozen inputs and grading semantics unchanged", flush=True)
    dump(config_path, contract)


def resume_extraction(audit, batch, flow, expected, args):
    source, rules = extraction_locations(audit)
    config = read(rules / "extraction_config.json")
    if not config.get("source_audit_dir"):
        raise ValueError("Unfinished extraction requires its quote-repair source_audit_dir; inspect " + str(rules / "summary.json"))
    if not hasattr(citation_repair, "normalize_rule_container"):
        raise ValueError("Upload both updated repair_guideline_citations.py and repair_guideline_coverage_ids.py")
    original_audit = Path(config["source_audit_dir"])
    candidates = [original_audit / "reference/reference.json", audit / "reference/reference.json"]
    citation_config = batch / "citation_repair_config.json"
    if citation_config.exists():
        original_batch = Path(read(citation_config)["original_batch_dir"])
        candidates.insert(0, original_batch / "flows" / flow / "run/reference/reference.json")
    reference = next((p for p in candidates if p.exists() and sha(p) == config["reference_sha256"]), None)
    if reference is None:
        raise ValueError("Original frozen reference not found for " + flow)
    official = read(reference)
    if official["source_hashes"] != expected["source_hashes"] or official["subflows"] != [flow] or official["units"] != [u for u in expected["units"] if u["flow"] == flow]:
        raise ValueError("Extraction resume reference differs from the batch reference")
    summary = read(rules / "summary.json")
    backup = rules / "summary.before_container_resume.json"
    if not backup.exists():
        dump(backup, summary)
    print(f"RESUME EXTRACTION {flow}: {summary.get('failed_chunks')} failed chunks; successful chunks retained", flush=True)
    citation_repair.repair_extraction(original_audit, audit, reference, args, evaluate=False)
    dump(audit / "extraction_container_resume.json", {"flow": flow,
        "helper_sha256": sha(ROOT / "scripts/repair_guideline_citations.py"),
        "old_summary_sha256": sha(backup), "summary_sha256": sha(rules / "summary.json"),
        "source_inventory_sha256": sha(source / "inventory.json")})


def initialize_coverage(audit, original, args):
    source, rules = require_complete_extraction(audit)
    if (audit / "coverage/protocol.json").exists():
        return False
    if any((audit / "coverage/batches").glob("*.json")):
        raise ValueError("Coverage batches exist without their frozen protocol; refusing to guess their inputs")
    extraction_config = read(rules / "extraction_config.json")
    from types import SimpleNamespace
    benchmark.evaluate(SimpleNamespace(reference=audit / "reference/reference.json", source_dir=source,
        rules_dir=rules, output_dir=audit / "coverage", origins="procedural_rule", provider=original["provider"],
        model=original["model"], workflow_id=extraction_config.get("workflow_id"), batch_size=4,
        workers=args.workers, qps=args.qps, max_prompt_chars=args.max_prompt_chars, max_batches=None, retries=args.retries,
        dry_run=True))
    return True


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
        save_repair_contract(config_path, contract)
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
            audit = None
            try:
                audit = audit_locations(batch, flow, previous.get(flow, {}))
                _, existing_rules = extraction_locations(audit)
                extraction_done = read(existing_rules / "summary.json").get("complete")
                if not extraction_done or not (audit / "reference/reference.json").exists():
                    states[flow] = {"status": "running", "phase": "resume_extraction", "audit_dir": str(audit)}
                    aggregate(batch, ref, all_rows, states)
                    resume_extraction(audit, batch, flow, ref, args)
                source, rules_dir = require_complete_extraction(audit)
                if initialize_coverage(audit, original, args):
                    contract["input_freezes"][flow] = sha(audit / "coverage/input_freeze.json")
                    save_repair_contract(config_path, contract, newly_initialized={flow})
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
                if audit is not None:
                    states[flow]["audit_dir"] = str(audit)
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
