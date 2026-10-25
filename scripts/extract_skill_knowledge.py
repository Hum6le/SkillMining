"""Portable server scan/export and provenance-checked LLM skill extraction.

Scanning requires Python stdlib only. Existing company llm.py is reused by
default; local extraction can explicitly select the Shubiaobiao provider.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tarfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skill_mining.knowledge_extraction import (VERSION, OURS_CORE_NAMES, discover_documents, chunk_document,
    extraction_messages, validate_response, merge_rules, hash_text)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def write_lines(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    roots, docs, skipped = {}, [], []
    overrides = {}
    for item in getattr(args, "family_override", None) or []:
        if "=" not in item:
            raise ValueError("Use --family-override method=coarse_flow")
        method, family = item.split("=", 1)
        if not method or not family or method in overrides:
            raise ValueError("Family overrides must be nonempty and unique")
        overrides[method] = family
    for item in args.run:
        if "=" not in item:
            raise ValueError("Use --run method=/path/to/method/run")
        method, directory = item.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", method) or method in roots:
            raise ValueError("Method labels must be unique alphanumeric names")
        roots[method] = str(Path(directory).resolve())
        options = {"snapshot": args.snapshot, "families": set(args.families.split(",")) if args.families else None}
        if method.casefold() == "ours" and getattr(args, "ours_resources", "core") == "core" and args.snapshot == "final":
            options["resource_names"] = OURS_CORE_NAMES
        if method in overrides:
            options["family_override"] = overrides[method]
        found, omissions = discover_documents(method, directory, **options)
        if not found:
            raise ValueError("No eligible resources found: " + method + " (check path/family/snapshot)")
        docs.extend(found)
        skipped.extend(omissions)
    if set(overrides) - set(roots):
        raise ValueError("Family override names a method absent from --run")
    requested = set(args.families.split(",")) if args.families else None
    if requested:
        for method in roots:
            missing = requested - {d["family"] for d in docs if d["method"] == method}
            if missing:
                raise ValueError(f"Missing requested families for {method}: {sorted(missing)}")
    chunks = [chunk for doc in docs for chunk in chunk_document(doc, args.chunk_chars, args.overlap)]
    # Round-robin method order ensures a small --max-chunks smoke includes methods fairly.
    per_method = {method: [c for c in chunks if c["method"] == method] for method in sorted(roots)}
    chunks = [group[i] for i in range(max(map(len, per_method.values()))) for group in per_method.values() if i < len(group)]
    write_lines(args.output_dir / "documents.jsonl", docs)
    write_lines(args.output_dir / "chunks.jsonl", chunks)
    inventory = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "roots": roots, "snapshot": args.snapshot, "chunk_chars": args.chunk_chars, "overlap": args.overlap,
        "counts": {method: {"documents": sum(d["method"] == method for d in docs),
            "chunks": len(per_method[method]), "source_chars": sum(len(d["text"]) for d in docs if d["method"] == method),
            "families": sorted({d["family"] for d in docs if d["method"] == method})} for method in sorted(roots)},
        "documents_sha256": sha(args.output_dir / "documents.jsonl"), "chunks_sha256": sha(args.output_dir / "chunks.jsonl"),
        "family_overrides": overrides,
        "ours_resources": getattr(args, "ours_resources", "core"),
        "ours_final_allowlist": sorted(OURS_CORE_NAMES) if getattr(args, "ours_resources", "core") == "core" and args.snapshot == "final" else None,
        "exclusions": "prompts, predictions, logs, raw exemplar memories, checkpoints, code, env files, seed resources; final excludes offline/offline_artifact",
        "resources": [{k: d[k] for k in ("method", "family", "snapshot", "source_file", "json_pointer", "file_sha256")} for d in docs],
        "duplicate_policy": "identical resource texts deduplicated only within method/family/snapshot; aliases retained",
        "skipped": skipped, "llm_calls": 0}
    dump(args.output_dir / "inventory.json", inventory)
    print(json.dumps(inventory["counts"], ensure_ascii=False, indent=2))
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        if args.archive.exists():
            raise FileExistsError(args.archive)
        with tarfile.open(args.archive, "w:gz") as archive:
            for name in ("documents.jsonl", "chunks.jsonl", "inventory.json"):
                archive.add(args.output_dir / name, arcname=args.output_dir.name + "/" + name)
        print("SOURCE-ONLY archive: " + str(args.archive))


class Gate:
    def __init__(self, qps):
        self.interval, self.next = 1 / qps, 0.
        self.lock = threading.Lock()
    def wait(self):
        with self.lock:
            delay = max(0., self.next - time.monotonic())
            self.next = max(self.next, time.monotonic()) + self.interval
        if delay:
            time.sleep(delay)


def extract(args):
    inventory = read(args.source_dir / "inventory.json")
    for filename, key in (("documents.jsonl", "documents_sha256"), ("chunks.jsonl", "chunks_sha256")):
        if sha(args.source_dir / filename) != inventory[key]:
            raise ValueError("Frozen inventory changed: " + filename)
    docs = {d["document_id"]: d for d in read_lines(args.source_dir / "documents.jsonl")}
    all_chunks = read_lines(args.source_dir / "chunks.jsonl")
    chunks = all_chunks[:args.max_chunks] if args.max_chunks else all_chunks
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("chunks", "raw"):
        (args.output_dir / name).mkdir(exist_ok=True)
    # An explicit workflow selection must dispatch before importing the common
    # llm.py entry point; otherwise its OpenAI default may already be bound.
    if args.provider == "existing" and args.workflow_id:
        os.environ["SKILLMINING_LLM_BACKEND"] = "workflow"
    # Import API clients only after source inventory checks; no secrets logged.
    import llm
    if args.provider == "shubiaobiao":
        from eval_tod.sgd_local_config import configure, DEFAULT_ENV
        provider = configure(args.env_file or DEFAULT_ENV, args.model)
    else:
        provider = {"provider": "existing_llm_py", "model": args.model}
    workflow_config = None
    if args.workflow_id:
        from config import LLM_CONFIG
        workflow_config = {**LLM_CONFIG, "provider": "workflow", "workflow_id": args.workflow_id}
        if not callable(getattr(llm, "get_llm_config", None)):
            raise ValueError("--workflow-id requires company workflow-aware llm.py")
    contract = {"version": VERSION, "documents_sha256": inventory["documents_sha256"],
        "chunks_sha256": inventory["chunks_sha256"], "provider": args.provider, "model": args.model,
        "workflow_id": args.workflow_id, "prompt_sha256": hash_text(json.dumps(extraction_messages({"text": ""}))),
        "code_sha256": sha(Path(__file__)), "core_sha256": sha(ROOT / "skill_mining/knowledge_extraction.py")}
    config_path = args.output_dir / "extraction_config.json"
    if config_path.exists() and read(config_path) != contract:
        raise ValueError("Extraction config/source changed; use a new output directory")
    reuse_dir = getattr(args, "reuse_responses_from", None)
    reuse_contract = None
    reusable_chunks = None
    if reuse_dir:
        if reuse_dir.resolve() == args.output_dir.resolve():
            raise ValueError("Revalidation requires a separate new output directory")
        reuse_contract = read(reuse_dir / "extraction_config.json")
        matching = getattr(args, "reuse_matching_chunks", False)
        keys = ("provider", "model", "workflow_id", "prompt_sha256")
        if not matching:
            keys += ("documents_sha256", "chunks_sha256")
        for key in keys:
            if reuse_contract.get(key) != contract.get(key):
                raise ValueError("Cannot reuse responses with different " + key)
        if matching:
            old_source = Path(read(reuse_dir / "summary.json")["source_dir"])
            old_inventory = read(old_source / "inventory.json")
            for filename, key in (("documents.jsonl", "documents_sha256"), ("chunks.jsonl", "chunks_sha256")):
                if sha(old_source / filename) != old_inventory[key] or old_inventory[key] != reuse_contract[key]:
                    raise ValueError("Cached source inventory changed: " + filename)
            old_docs = {d["document_id"]: d for d in read_lines(old_source / "documents.jsonl")}
            old_chunks = {c["chunk_id"]: c for c in read_lines(old_source / "chunks.jsonl")}
            # Exact chunk equality includes text, offsets, identity and node aliases.
            # Document aliases can shrink when excluded duplicate files disappear.
            comparable = lambda d: {k: v for k, v in d.items() if k != "aliases"}
            reusable_chunks = {c["chunk_id"] for c in all_chunks
                if old_chunks.get(c["chunk_id"]) == c and c["document_id"] in old_docs
                and comparable(docs[c["document_id"]]) == comparable(old_docs[c["document_id"]])}
    dump(config_path, contract)
    gate = Gate(args.qps)
    def one(chunk):
        path = args.output_dir / "chunks" / (chunk["chunk_id"] + ".json")
        doc = docs[chunk["document_id"]]
        if path.exists():
            old = read(path)
            if old.get("status") == "complete":
                # Revalidate literal spans, including after copying to another machine.
                validate_response(old["raw_response"], chunk, doc)
                return old
        if reuse_dir and (reusable_chunks is None or chunk["chunk_id"] in reusable_chunks):
            old_path = reuse_dir / "chunks" / (chunk["chunk_id"] + ".json")
            if old_path.exists():
                old = read(old_path)
                candidates = [old["raw_response"]] if old.get("raw_response") else []
                failures = sorted((reuse_dir / "raw").glob(chunk["chunk_id"] + "_attempt_*.json"),
                                  key=lambda p: int(p.stem.rsplit("_", 1)[1]), reverse=True)
                candidates += [read(p).get("raw_response", "") for p in failures]
                for cached_raw in candidates:
                    try:
                        rules = validate_response(cached_raw, chunk, doc)
                    except (ValueError, TypeError, KeyError):
                        continue
                    result = {"chunk_id": chunk["chunk_id"], "status": "complete", "rules": rules,
                        "raw_response": cached_raw, "attempts": old.get("attempts", []),
                        "revalidated_response": {"source_dir": str(reuse_dir), "response_sha256": hash_text(cached_raw),
                            "prior_config_sha256": sha(reuse_dir / "extraction_config.json"),
                            "prior_status": old.get("status"), "new_llm_calls": 0}}
                    dump(path, result)
                    return result
        messages = extraction_messages(chunk)
        attempts = []
        if path.exists():
            attempts = read(path).get("attempts", [])
        result = None
        for attempt in range(args.retries):
            gate.wait()
            raw = ""
            try:
                kwargs = {"model": args.model, "temperature": 0., "call_tag": "skill_knowledge_extract"}
                if workflow_config is not None:
                    kwargs["config"] = workflow_config
                raw = llm.chat(messages, **kwargs)
                rules = validate_response(raw, chunk, doc)
                attempts.append({"attempt": len(attempts)+1, "ok": True,
                    "estimated_prompt_tokens": (len(json.dumps(messages, ensure_ascii=False))+3)//4,
                    "estimated_completion_tokens": (len(str(raw))+3)//4})
                result = {"chunk_id": chunk["chunk_id"], "status": "complete", "rules": rules,
                          "raw_response": raw, "attempts": attempts}
                break
            except Exception as exc:
                attempts.append({"attempt": len(attempts)+1, "ok": False, "error": type(exc).__name__ + ": " + str(exc)[:400],
                    "estimated_prompt_tokens": (len(json.dumps(messages, ensure_ascii=False))+3)//4,
                    "estimated_completion_tokens": (len(str(raw))+3)//4})
                dump(args.output_dir / "raw" / f"{chunk['chunk_id']}_attempt_{len(attempts)}.json",
                    {"messages": messages, "raw_response": raw, "error": attempts[-1]["error"]})
                # Invalid outputs stay on disk; repair never invents a source quote.
                messages = extraction_messages(chunk) + [{"role": "assistant", "content": str(raw)},
                    {"role": "user", "content": "Validation failed: " + attempts[-1]["error"]
                     + ". Return a complete corrected JSON object. Use valid local source line numbers. Omit unsupported claims; use null for absent details."}]
                if attempt + 1 < args.retries:
                    time.sleep(min(2 ** attempt, 8))
        if result is None:
            result = {"chunk_id": chunk["chunk_id"], "status": "failed", "rules": [], "attempts": attempts}
        dump(path, result)
        return result
    statuses = []
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(one, chunk) for chunk in chunks]
            for future in as_completed(futures):
                result = future.result()
                statuses.append(result)
                dump(args.output_dir / "progress.json", {"completed_chunks": len(statuses), "requested_chunks": len(chunks),
                    "total_inventory_chunks": len(all_chunks), "failed_chunks": sum(r["status"] == "failed" for r in statuses),
                    "elapsed_seconds": time.monotonic() - started})
                print(f"{result['status'].upper()} {result['chunk_id']} ({len(statuses)}/{len(chunks)}), rules={len(result['rules'])}", flush=True)
    finally:
        tracker = getattr(llm, "get_usage_summary", None)
        if tracker:
            dump(args.output_dir / "llm_usage_this_process.json", tracker())
    # Include all cached chunks, so a later partial resume cannot erase past results.
    results = [read(args.output_dir / "chunks" / (c["chunk_id"] + ".json"))
               for c in all_chunks if (args.output_dir / "chunks" / (c["chunk_id"] + ".json")).exists()]
    rules = merge_rules([r for result in results if result["status"] == "complete" for r in result["rules"]])
    write_lines(args.output_dir / "rules.jsonl", rules)
    write_lines(args.output_dir / "procedural_rules.jsonl", [r for r in rules if r["origin"] == "procedural_rule"])
    counts = {}
    for method in inventory["counts"]:
        own = [r for r in rules if r["method"] == method]
        counts[method] = {"mechanically_deduplicated_rules": len(own), "origins": dict(Counter(r["origin"] for r in own)),
                          "categories": dict(Counter(r["category"] for r in own)),
                          "note": "counts are extraction descriptors, not correctness/coverage scores"}
    complete = len(results) == len(all_chunks) and all(r["status"] == "complete" for r in results)
    chunk_index = {c["chunk_id"]: c for c in all_chunks}
    failure_details = [{"chunk_id": r["chunk_id"], "method": chunk_index[r["chunk_id"]]["method"],
        "source_file": docs[chunk_index[r["chunk_id"]]["document_id"]]["source_file"],
        "last_error": r.get("attempts", [{}])[-1].get("error", "unknown") if r.get("attempts") else "unknown"}
        for r in results if r["status"] == "failed"]
    summary = {"complete": complete, "inventory_chunks": len(all_chunks), "successful_chunks": sum(r["status"] == "complete" for r in results),
        "failed_chunks": sum(r["status"] == "failed" for r in results), "methods": counts,
        "failed_chunk_details": failure_details,
        "source_dir": str(args.source_dir), "source_inventory_sha256": sha(args.source_dir / "inventory.json"),
        "revalidated_cached_chunks": sum("revalidated_response" in r for r in results),
        "literal_citations_verified": True, "semantic_support": "not evaluated; human review pending",
        "train_grounding": "not evaluated; requires separate TRAIN/guideline audit",
        "all_attempts": sum(len(r["attempts"]) for r in results),
        "estimated_tokens_including_cached_attempts": sum(a.get("estimated_prompt_tokens", 0)+a.get("estimated_completion_tokens", 0) for r in results for a in r["attempts"])}
    dump(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def review(args):
    rules = read_lines(args.extraction_dir / "rules.jsonl")
    selected = [r for r in rules if r["origin"] in args.origins.split(",")]
    rng = random.Random(args.seed)
    rng.shuffle(selected)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.per_method:
        counts, subset = Counter(), []
        for rule in selected:
            if counts[rule["method"]] < args.per_method:
                counts[rule["method"]] += 1
                subset.append(rule)
        selected = subset
        rng.shuffle(selected)
    methods = sorted({r["method"] for r in rules})
    # Best effort de-identification: never expose actual file paths or rule IDs.
    mask = re.compile(r"\b(?:" + "|".join(re.escape(x) for x in set(methods + ["AWM", "Trace2Skill", "SkillMiner"])) + r")\b", re.I)
    rows, key = [], []
    for i, rule in enumerate(selected, 1):
        rid = f"R{i:05d}"
        clean = lambda value: mask.sub("[METHOD]", str(value or ""))
        rows.append({"review_id": rid, "family": rule["family"], "snapshot": rule["snapshot"],
            **{k: clean(rule.get(k)) for k in ("category", "rule_text", "action", "condition", "directive", "slot", "value_source", "missing_behavior")},
            "source_quote": clean(rule["citations"][0]["quote"]),
            "text_support_label": "", "train_support_label": "", "specificity_label": "", "review_notes": ""})
        key.append({"review_id": rid, "rule_id": rule["rule_id"], "method": rule["method"], "citations": rule["citations"]})
    with (args.output_dir / "blind_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(rows[0]) if rows else ["review_id"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    dump(args.output_dir / "private_key.json", key)
    dump(args.output_dir / "review_manifest.json", {"seed": args.seed, "origins": args.origins,
        "per_method_limit": args.per_method, "rows": len(rows), "source_rules_sha256": sha(args.extraction_dir / "rules.jsonl"),
        "limitations": "method names/paths removed best effort; phrasing may reveal method. Keep private_key away from reviewers. This is an audit template, no correctness grades computed. Fixed-count sampling is not corpus coverage evaluation."})
    print("Review template: " + str(args.output_dir / "blind_review.csv"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("scan", help="No API calls; package eligible final resources")
    p.add_argument("--run", action="append", required=True, help="method=/path/to/run (repeat)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--families", "--subflows", "--subflow", dest="families", help="Exact family/coarse-flow names, comma-separated")
    p.add_argument("--family-override", action="append", help="method=coarse_flow for a single-flow run with missing metadata; conflicting metadata is rejected")
    p.add_argument("--snapshot", choices=["final", "offline", "all"], default="final")
    p.add_argument("--ours-resources", choices=["core", "all"], default="core",
                   help="Final Ours default: skill.md + online action/slot/transition files; all restores the broad scan")
    p.add_argument("--chunk-chars", type=int, default=8000)
    p.add_argument("--overlap", type=int, default=400)
    p.add_argument("--archive", type=Path)
    p.set_defaults(func=scan)
    p = sub.add_parser("extract", help="LLM extraction with verified exact source spans")
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--provider", choices=["existing", "shubiaobiao"], default="existing")
    p.add_argument("--env-file", type=Path)
    p.add_argument("--workflow-id")
    p.add_argument("--model", default="qwen3.6-flash")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--qps", type=float, default=1.8)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-chunks", type=int, help="Smoke only; incomplete inventory stays marked incomplete")
    p.add_argument("--reuse-responses-from", type=Path,
                   help="Revalidate cached raw responses in a NEW output directory; source/model/prompt must match")
    p.add_argument("--reuse-matching-chunks", action="store_true",
                   help="Allow a rescanned resource subset; reuse only exactly identical chunks/documents and prompts")
    p.set_defaults(func=extract)
    p = sub.add_parser("review", help="Seeded blinded CSV and separate private source key")
    p.add_argument("--extraction-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--origins", default="procedural_rule")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-method", type=int)
    p.set_defaults(func=review)
    args = parser.parse_args()
    if args.command == "extract" and args.reuse_matching_chunks and not args.reuse_responses_from:
        parser.error("--reuse-matching-chunks requires --reuse-responses-from")
    if args.command == "extract" and (args.workers < 1 or args.qps <= 0 or args.retries < 1 or args.max_chunks is not None and args.max_chunks < 1):
        parser.error("Positive workers/qps/retries/max-chunks required")
    result = args.func(args)
    if args.command == "extract" and result["failed_chunks"]:
        print("Extraction failed: see failed_chunk_details in summary.json; coverage was not started.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
