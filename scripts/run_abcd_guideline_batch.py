"""Run saved-skill guideline audits across independent ABCD coarse subflows.

Use a JSON manifest for method directory templates, per-flow overrides and
completed audit reuse. --dry-run scans/verifies sources without LLM calls.
Requires the v4 audit tools plus repair_abcd_guideline_extraction.py.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.extract_skill_knowledge import dump, read, read_lines, scan, sha
from scripts.benchmark_abcd_skill_guidelines import load_inputs, reference
from skill_mining.guideline_coverage import FINE_TITLES, VERSION, build_reference, summarize, validate_assessments
from skill_mining.knowledge_extraction import OURS_CORE_NAMES, discover_documents

METHODS = ["awm", "ours", "trace2skill"]
TOOLS = ["scripts/extract_skill_knowledge.py", "skill_mining/knowledge_extraction.py",
         "scripts/benchmark_abcd_skill_guidelines.py", "skill_mining/guideline_coverage.py",
         "scripts/repair_abcd_guideline_extraction.py"]


@contextmanager
def exclusive_output(output):
    """OS lock is released even after a crash; no stale PID guessing."""
    output.mkdir(parents=True, exist_ok=True)
    stream = (output / ".runner.lock").open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            if not stream.seek(0, os.SEEK_END):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another batch is using this output directory") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another batch is using this output directory") from exc
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def template():
    return {
        "methods": {
            "ours": {flow: None for flow in FINE_TITLES},
            "awm": "outputs/full_abcd_2026-09-29_14-20-14/awm/{flow}",
            "trace2skill": "outputs/full_abcd_2026-09-09_13-29-43/trace2skill/{flow}/abcd_trace2skill_*",
        },
        "overrides": {},
        "reuse": {"shipping_issue": "outputs/abcd_guideline_shipping_core_jsonfix_v1"},
        "notes": "Ours runs may be in unrelated directories: fill methods.ours.<flow> with each final run directory. Null is intentionally unconfigured. Reused flows do not need source paths. Each method accepts either a directory template or a per-flow mapping. Paths are relative to the project root. {flow} expands to a coarse subflow; globs must match exactly one directory. overrides may contain flow -> method -> exact directory. reuse requires a complete three-method audit with the same model/provider/reference protocol.",
    }


def path_at_root(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_resource(value, flow):
    pattern = str(path_at_root(value.format(flow=flow)))
    found = [Path(p).resolve() for p in glob.glob(pattern) if Path(p).is_dir()]
    if len(found) != 1:
        raise ValueError(f"{flow}: expected exactly one directory for {pattern!r}; found {len(found)}")
    return found[0]


def collect_coverage(audit, flow, expected, *, provider, model, require_complete=True):
    """Revalidate frozen inputs and raw batches; never modify an old audit."""
    audit = Path(audit)
    if (audit / "latest_extraction.json").exists():
        latest = read(audit / "latest_extraction.json")
        source, rules = path_at_root(latest["source_dir"]), path_at_root(latest["rules_dir"])
    else:
        source, rules = audit / "sources", audit / "rules"
    coverage = audit / "coverage"
    ref_path = audit / "reference/reference.json"
    ref, methods, candidates = load_inputs(ref_path, source, rules, "procedural_rule")
    if sorted(methods) != METHODS or ref["subflows"] != [flow]:
        raise ValueError(f"Reused audit must contain exactly three methods and only {flow}")
    if ref["source_hashes"] != expected["source_hashes"]:
        raise ValueError("Reused official reference sources differ")
    target = [u for u in expected["units"] if u["flow"] == flow]
    if ref["units"] != target:
        raise ValueError("Reused reference units/classification differ")
    protocol = read(coverage / "protocol.json")
    if protocol["version"] != VERSION or protocol["provider"] != provider or protocol["model"] != model or protocol["origins"] != "procedural_rule":
        raise ValueError("Reused coverage protocol/provider/model differs")
    for filename, actual in (("reference_sha256", sha(ref_path)), ("documents_sha256", sha(source / "documents.jsonl")),
                             ("rules_sha256", sha(rules / "rules.jsonl"))):
        if protocol[filename] != actual:
            raise ValueError("Reused input differs: " + filename)
    frozen = read(coverage / "input_freeze.json")
    for name in ("protocol.json", "reference_snapshot.json", "candidate_snapshot.json", "input_manifest.json"):
        if frozen.get(name) != sha(coverage / name):
            raise ValueError("Reused frozen snapshot changed: " + name)
    if read(coverage / "reference_snapshot.json") != ref:
        raise ValueError("Reused reference snapshot differs")
    saved_candidates = {(x["method"], x["flow"]): x["rules"] for x in read(coverage / "candidate_snapshot.json")}
    if saved_candidates != candidates:
        raise ValueError("Reused candidate snapshot differs")
    units = [u for u in ref["units"] if u["eligibility"] == "eligible"]
    rows = []
    size = protocol["batch_size"]
    for method in METHODS:
        for index in range(0, len(units), size):
            path = coverage / "batches" / f"{method}_{flow}_{index:04d}.json"
            saved = read(path) if path.exists() else {}
            if saved.get("status") != "complete":
                if require_complete:
                    raise ValueError("Coverage batch missing or failed: " + str(path))
                continue
            checked = validate_assessments(saved["raw"], units[index:index + size], candidates[method, flow])
            rows += [{**row, "method": method, "flow": flow, "review_status": "model_proposal_pending_review"} for row in checked]
    if require_complete and not summarize(ref, METHODS, rows)["complete"]:
        raise ValueError("Reused coverage is incomplete")
    return rows


def make_plan(manifest, flows, ref, *, provider, model):
    if set(manifest.get("methods", {})) != set(METHODS):
        raise ValueError("manifest.methods must contain exactly ours, awm, trace2skill")
    for key in ("overrides", "reuse"):
        if set(manifest.get(key, {})) - set(FINE_TITLES):
            raise ValueError("Unknown flow in manifest." + key)
    for method, value in manifest["methods"].items():
        if isinstance(value, dict) and set(value) - set(FINE_TITLES):
            raise ValueError("Unknown flow in manifest.methods." + method)
    plans, errors = {}, []
    for flow in flows:
        try:
            if flow in manifest.get("reuse", {}):
                audit = path_at_root(manifest["reuse"][flow])
                assessments = collect_coverage(audit, flow, ref, provider=provider, model=model)
                plans[flow] = {"reuse": str(audit), "assessments": len(assessments),
                               "protocol_sha256": sha(audit / "coverage/protocol.json"),
                               "freeze_sha256": sha(audit / "coverage/input_freeze.json")}
                continue
            overrides = manifest.get("overrides", {}).get(flow, {})
            if set(overrides) - set(METHODS):
                raise ValueError("Unknown method in flow overrides")
            own = {}
            for method in METHODS:
                setting = manifest["methods"][method]
                setting = setting.get(flow) if isinstance(setting, dict) else setting
                setting = overrides.get(method, setting)
                if not isinstance(setting, str) or not setting.strip():
                    raise ValueError(f"Missing final resource directory for {method}/{flow}; fill methods.{method}.{flow} or a per-flow override")
                directory = resolve_resource(setting, flow)
                options = {"families": {flow}, "family_override": flow}
                if method == "ours":
                    options["resource_names"] = OURS_CORE_NAMES
                docs, _ = discover_documents(method, directory, **options)
                if not docs:
                    raise ValueError(f"No eligible final resources for {method}: {directory}")
                if method == "ours":
                    locations = [d["source_file"] for d in docs] + [a["file"] for d in docs for a in d["aliases"]]
                    for name in OURS_CORE_NAMES:
                        if sum(Path(p).name.lower() == name for p in locations) > 1:
                            raise ValueError(f"Multiple {name} resources under {directory}; select one final arm")
                own[method] = {"root": str(directory), "documents": len(docs),
                    "resources": [{k: d[k] for k in ("document_id", "source_file", "file_sha256", "text_sha256")} for d in docs]}
            plans[flow] = {"methods": own}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{flow}: {exc}")
    if errors:
        raise ValueError("Preflight failed before any LLM calls:\n" + "\n".join(errors))
    return plans


def aggregate(output, ref, all_rows, states):
    rows = [row for flow in ref["subflows"] for row in all_rows.get(flow, [])]
    result = summarize(ref, METHODS, rows)
    table = []
    for flow in ref["subflows"]:
        small_ref = {**ref, "subflows": [flow], "units": [u for u in ref["units"] if u["flow"] == flow]}
        small = summarize(small_ref, METHODS, all_rows.get(flow, []))
        for method in METHODS:
            own = small["methods"][method]
            cond = own["conditional_action"]
            result["methods"][method]["by_flow"][flow]["conditional_action"] = cond
            table.append({"subflow": flow, "method": method, "status": states[flow]["status"],
                "assessed": own["assessed"], "reference_units": own["reference_units"], "covered": own["covered"],
                "coverage": own["strict_coverage"], "conditional_units": cond["reference_units"],
                "conditional_covered": cond["covered"], "conditional_coverage": cond["strict_coverage"], "conflicting": own["conflicting"]})
    for method in METHODS:
        values = [g["conditional_action"]["strict_coverage"] for g in result["methods"][method]["by_flow"].values()
                  if g["conditional_action"]["reference_units"]]
        result["methods"][method]["conditional_flow_macro_coverage"] = sum(values) / len(values) if values and None not in values else None
    result.update(label_status="model_proposal_pending_review", flows=states,
                  interpretation="Fixed-reference saved-skill coverage; reused and fresh raw batches revalidated. Missing flows stay in the denominator; incomplete global scores are null.")
    dump(output / "aggregate_summary.json", result)
    dump(output / "assessments.json", rows)
    with (output / "by_subflow.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    dump(output / "progress.json", {"runner_pid": os.getpid(), "completed_flows": sum(s["status"] in ("completed", "reused") for s in states.values()),
        "total_flows": len(states), "failed_flows": [f for f, s in states.items() if s["status"] == "failed"],
        "flows": states, "assessed": result["assessed"], "expected_assessments": result["expected_assessments"], "complete": result["complete"]})


def execute(args):
    manifest = read(args.manifest)
    flows = list(FINE_TITLES) if args.all or not args.subflows else [f.strip() for f in args.subflows.split(",") if f.strip()]
    raw = {name: read(args.data_dir / (name + ".json")) for name in ("guidelines", "ontology", "kb")}
    ref = build_reference(raw["guidelines"], raw["ontology"], raw["kb"], flows)
    for tool in TOOLS:
        if not (ROOT / tool).is_file():
            raise FileNotFoundError("Missing audit dependency: " + tool)
    help_text = subprocess.check_output([sys.executable, str(ROOT / TOOLS[0]), "extract", "--help"], text=True, cwd=ROOT)
    if "--reuse-matching-chunks" not in help_text:
        raise ValueError("extract_skill_knowledge.py must be v4 or later")
    plan = make_plan(manifest, flows, ref, provider=args.provider, model=args.model)
    if args.dry_run:
        print(json.dumps({"llm_calls": 0, "flows": flows, "eligible_units": sum(u["eligibility"] == "eligible" for u in ref["units"]),
                          "conditional_units": sum(u["eligibility"] == "eligible" and u["kind"] == "conditional_action" for u in ref["units"]),
                          "plan": plan}, ensure_ascii=False, indent=2))
        return
    output = args.output_dir.resolve()
    contract = {"reference": ref, "plan": plan, "provider": args.provider, "model": args.model,
                "workflow_id": args.workflow_id, "chunk_chars": args.chunk_chars, "batch_size": args.batch_size,
                "tools_sha256": {tool: sha(ROOT / tool) for tool in TOOLS}}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "batch_config.json").exists() and read(output / "batch_config.json") != contract:
        raise ValueError("Batch inputs/code/settings changed; choose a new output directory")
    dump(output / "batch_config.json", contract)
    dump(output / "reference_snapshot.json", ref)
    (output / "logs").mkdir(exist_ok=True)
    env = dict(os.environ)
    if args.provider == "existing":
        env["SKILLMINING_LLM_BACKEND"] = "workflow"
        if args.workflow_id:
            env["SKILLMINING_WORKFLOW_ID"] = args.workflow_id
        subprocess.run([sys.executable, "-c", "import llm; c=llm.get_llm_config(); "
                        "assert c.get('provider') == 'workflow' and c.get('workflow_id'), 'Workflow config missing'; "
                        "print('Workflow backend OK:', llm.chat.__module__)"], cwd=ROOT, env=env, check=True)
    common = ["--provider", args.provider, "--model", args.model, "--workers", str(args.workers), "--qps", str(args.qps)]
    if args.workflow_id:
        common += ["--workflow-id", args.workflow_id]
    states = {f: {"status": "queued"} for f in flows}
    all_rows = {}
    aggregate(output, ref, all_rows, states)
    for flow in flows:
        log_path = output / "logs" / (flow + ".log")
        current = output / "flows" / flow / "run"
        repaired = current.parent / "repaired"
        audit = None
        try:
            if "reuse" in plan[flow]:
                audit = Path(plan[flow]["reuse"])
                all_rows[flow] = collect_coverage(audit, flow, ref, provider=args.provider, model=args.model)
                states[flow] = {"status": "reused", "audit_dir": str(audit)}
                print(f"REUSED {flow}: {len(all_rows[flow])} assessments", flush=True)
                continue
            states[flow] = {"status": "running", "phase": "scan", "log": str(log_path)}
            aggregate(output, ref, all_rows, states)
            if not (current / "reference/reference.json").exists():
                reference(SimpleNamespace(output_dir=current / "reference", all=False, subflows=flow, data_dir=args.data_dir))
            if not (current / "sources/inventory.json").exists():
                scan(SimpleNamespace(run=[m + "=" + plan[flow]["methods"][m]["root"] for m in METHODS],
                    output_dir=current / "sources", families=flow, family_override=[m + "=" + flow for m in METHODS],
                    snapshot="final", ours_resources="core", chunk_chars=args.chunk_chars, overlap=min(200, args.chunk_chars // 4), archive=None))
            docs = read_lines(current / "sources/documents.jsonl")
            for method in METHODS:
                actual = [{k: d[k] for k in ("document_id", "source_file", "file_sha256", "text_sha256")} for d in docs if d["method"] == method]
                if actual != plan[flow]["methods"][method]["resources"]:
                    raise ValueError("Scanned sources differ from preflight: " + method)
            with log_path.open("a", encoding="utf-8") as log:
                def child(command):
                    log.write("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
                    log.flush()
                    return subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if not (repaired / "repair_config.json").exists():
                    states[flow]["phase"] = "extract"
                    aggregate(output, ref, all_rows, states)
                    done = child([sys.executable, "-u", str(ROOT / TOOLS[0]), "extract", "--source-dir", str(current / "sources"),
                                  "--output-dir", str(current / "rules"), *common])
                    if done.returncode not in (0, 2) or not (current / "rules/summary.json").exists():
                        raise RuntimeError(f"Extraction exited {done.returncode}; see {log_path}")
                if (repaired / "repair_config.json").exists() or not read(current / "rules/summary.json")["complete"]:
                    states[flow]["phase"] = "repair_and_evaluate"
                    aggregate(output, ref, all_rows, states)
                    done = child([sys.executable, "-u", str(ROOT / TOOLS[-1]), "--audit-dir", str(current), "--reference", str(current / "reference/reference.json"),
                                  "--output-dir", str(repaired), "--chunk-chars", str(args.chunk_chars), "--workers", str(args.workers), "--qps", str(args.qps)])
                    audit = repaired
                else:
                    states[flow]["phase"] = "evaluate"
                    aggregate(output, ref, all_rows, states)
                    done = child([sys.executable, "-u", str(ROOT / "scripts/benchmark_abcd_skill_guidelines.py"), "evaluate",
                        "--reference", str(current / "reference/reference.json"), "--source-dir", str(current / "sources"),
                        "--rules-dir", str(current / "rules"), "--output-dir", str(current / "coverage"), "--batch-size", str(args.batch_size), *common])
                    audit = current
                if done.returncode:
                    raise RuntimeError(f"Audit exited {done.returncode}; see {log_path}")
            all_rows[flow] = collect_coverage(audit, flow, ref, provider=args.provider, model=args.model)
            states[flow] = {"status": "completed", "audit_dir": str(audit), "log": str(log_path)}
            print(f"COMPLETED {flow}: {len(all_rows[flow])} assessments", flush=True)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
            states[flow] = {"status": "failed", "error": str(exc), "log": str(log_path)}
            if audit and (audit / "coverage/protocol.json").exists():
                try:
                    all_rows[flow] = collect_coverage(audit, flow, ref, provider=args.provider, model=args.model, require_complete=False)
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            print(f"FAILED {flow}: {exc}", flush=True)
        finally:
            aggregate(output, ref, all_rows, states)
    if any(s["status"] == "failed" for s in states.values()):
        raise RuntimeError("Some subflows failed; successful calls are cached. See progress.json and per-flow logs; rerun the same command to resume.")
    print("All subflows complete: " + str(output / "aggregate_summary.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--write-template", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/abcd_guideline_all_v1"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true")
    group.add_argument("--subflows")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/eval/abcd/data")
    parser.add_argument("--provider", choices=["existing", "shubiaobiao"], default="existing")
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--workflow-id")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--qps", type=float, default=1.8)
    parser.add_argument("--chunk-chars", type=int, default=1600)
    # The existing repair helper also evaluates four reference units per batch.
    parser.set_defaults(batch_size=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    try:
        if args.write_template:
            if args.write_template.exists():
                raise FileExistsError(args.write_template)
            dump(args.write_template, template())
            print("Edit method directory templates: " + str(args.write_template))
            return
        if args.status:
            print(json.dumps(read(args.output_dir / "progress.json"), ensure_ascii=False, indent=2))
            return
        if not args.manifest:
            parser.error("--manifest is required for preflight/run")
        if args.workers < 1 or args.qps <= 0 or args.chunk_chars < 500 or args.batch_size < 1:
            parser.error("positive workers/QPS/batch-size and chunk-chars >= 500 required")
        if args.dry_run:
            execute(args)
        else:
            with exclusive_output(args.output_dir.resolve()):
                execute(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print("Batch stopped: " + str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
