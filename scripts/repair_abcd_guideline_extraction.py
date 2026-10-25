"""Repair failed extraction chunks without repeating successful LLM calls.

Requires the existing v4 audit tools. Only this file needs to be transferred.
Frozen source text and reference units stay unchanged; failed inputs are split
into smaller overlapping windows. All cached source citations are revalidated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skill_mining.knowledge_extraction import chunk_document, hash_text


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def split_chunk(chunk, doc, max_chars):
    if doc["text"][chunk["start"]:chunk["end"]] != chunk["text"]:
        raise ValueError("Chunk text differs from frozen source: " + chunk["chunk_id"])
    if len(chunk["text"]) <= max_chars:
        return [chunk]
    local_doc = {**doc, "text": chunk["text"], "node_context": chunk.get("node_context", {})}
    children = chunk_document(local_doc, max_chars=max_chars, overlap=min(200, max_chars // 4))
    for child in children:
        child["start"] += chunk["start"]
        child["end"] += chunk["start"]
        child["chunk_id"] = "chunk-" + hash_text(child["document_id"] + f"/{child['start']}/{child['end']}")[:20]
    if children[0]["start"] != chunk["start"] or children[-1]["end"] != chunk["end"]:
        raise ValueError("Split did not preserve source boundaries")
    if any(b["start"] > a["end"] for a, b in zip(children, children[1:])):
        raise ValueError("Split left a source gap")
    return children


def prepare_round(previous, target, max_chars):
    previous, target = Path(previous), Path(target)
    source = previous / "sources"
    inventory = read(source / "inventory.json")
    for filename, key in (("documents.jsonl", "documents_sha256"), ("chunks.jsonl", "chunks_sha256")):
        if sha(source / filename) != inventory[key]:
            raise ValueError("Frozen source changed: " + filename)
    docs = {d["document_id"]: d for d in rows(source / "documents.jsonl")}
    original = rows(source / "chunks.jsonl")
    selected, lineage = [], []
    successes = 0
    for chunk in original:
        saved_path = previous / "rules/chunks" / (chunk["chunk_id"] + ".json")
        saved = read(saved_path) if saved_path.exists() else {}
        if saved.get("status") == "complete":
            children = [chunk]
            successes += 1
        else:
            children = split_chunk(chunk, docs[chunk["document_id"]], max_chars)
        selected.extend(children)
        lineage.append({"parent_chunk_id": chunk["chunk_id"], "prior_status": saved.get("status", "missing"),
                        "child_chunk_ids": [c["chunk_id"] for c in children]})
    # Overlap can yield an identical window twice; its single response suffices.
    unique = {}
    for chunk in selected:
        if chunk["chunk_id"] in unique and unique[chunk["chunk_id"]] != chunk:
            raise ValueError("Chunk identity collision")
        unique[chunk["chunk_id"]] = chunk
    selected = list(unique.values())
    target.mkdir(parents=True, exist_ok=False)
    (target / "documents.jsonl").write_bytes((source / "documents.jsonl").read_bytes())
    (target / "chunks.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in selected), encoding="utf-8")
    counts = Counter(c["method"] for c in selected)
    inventory = {**inventory, "counts": {m: {**c, "chunks": counts[m]} for m, c in inventory["counts"].items()},
        "chunks_sha256": sha(target / "chunks.jsonl"),
        "repair": {"previous_audit_dir": str(previous), "previous_inventory_sha256": sha(source / "inventory.json"),
                   "max_failed_chunk_chars": max_chars, "prior_successful_chunks": successes,
                   "chunking": "successful windows unchanged; failed windows subdivided with overlap and no source gaps",
                   "lineage": lineage}}
    dump(target / "inventory.json", inventory)
    print(f"Prepared {len(selected)} windows; {successes} successful windows retained; failed-window limit={max_chars} chars", flush=True)
    return inventory


def run(args):
    previous = args.audit_dir.resolve()
    output = args.output_dir.resolve()
    reference = args.reference.resolve()
    if previous == output or previous in output.parents:
        raise ValueError("Use a separate sibling output directory; do not write inside the old run")
    read(reference)  # Fail early if the supplied reference is missing/invalid.
    config = read(previous / "rules/extraction_config.json")
    for flag, command in (("--reuse-matching-chunks", "extract"),):
        help_text = subprocess.check_output([sys.executable, str(ROOT / "scripts/extract_skill_knowledge.py"), command, "--help"], cwd=ROOT, text=True)
        if flag not in help_text:
            raise ValueError("Existing extract_skill_knowledge.py lacks " + flag + "; v4 tools are required")
    contract = {"previous_audit_dir": str(previous), "previous_inventory_sha256": sha(previous / "sources/inventory.json"),
                "reference": str(reference), "reference_sha256": sha(reference),
                "provider": config["provider"], "model": config["model"], "workflow_id": config.get("workflow_id"),
                "chunk_chars": args.chunk_chars, "max_rounds": args.max_rounds,
                "extractor_sha256": sha(ROOT / "scripts/extract_skill_knowledge.py"),
                "core_sha256": sha(ROOT / "skill_mining/knowledge_extraction.py")}
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "repair_config.json"
    if contract_path.exists() and read(contract_path) != contract:
        raise ValueError("Repair inputs/settings changed; choose a new output directory")
    dump(contract_path, contract)
    env = dict(os.environ)
    if config["provider"] == "existing":
        env["SKILLMINING_LLM_BACKEND"] = "workflow"
        subprocess.run([sys.executable, "-c", "import llm; c=llm.get_llm_config(); "
                        "assert c.get('provider') == 'workflow' and c.get('workflow_id'), 'Workflow config missing'; "
                        "print('Workflow backend OK:', llm.chat.__module__)"], cwd=ROOT, env=env, check=True)
    common = ["--provider", config["provider"], "--model", config["model"],
              "--workers", str(args.workers), "--qps", str(args.qps)]
    if config.get("workflow_id"):
        common += ["--workflow-id", config["workflow_id"]]
    state_path = output / "repair_state.json"
    state = read(state_path) if state_path.exists() else {"round": 1, "previous": str(previous)}
    for number in range(state["round"], args.max_rounds + 1):
        previous = Path(state["previous"])
        current = output / f"round_{number:02d}"
        source, rules_dir = current / "sources", current / "rules"
        limit = max(500, args.chunk_chars // (2 ** (number - 1)))
        if not (source / "inventory.json").exists():
            prepare_round(previous, source, limit)
        print(f"Repair round {number}/{args.max_rounds}: {current}", flush=True)
        command = [sys.executable, "-u", str(ROOT / "scripts/extract_skill_knowledge.py"), "extract",
                   "--source-dir", str(source), "--output-dir", str(rules_dir),
                   "--reuse-responses-from", str(previous / "rules"), "--reuse-matching-chunks", *common]
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode not in (0, 2) or not (rules_dir / "summary.json").exists():
            raise RuntimeError(f"Extraction process failed before a valid summary (exit={result.returncode})")
        summary = read(rules_dir / "summary.json")
        dump(output / "latest_extraction.json", {"round": number, "source_dir": str(source),
            "rules_dir": str(rules_dir), "summary": summary})
        if summary["complete"]:
            # Copy the public reference so this repaired run is self-contained.
            copied = output / "reference"
            if not copied.exists():
                shutil.copytree(reference.parent, copied)
            if sha(copied / reference.name) != contract["reference_sha256"]:
                raise ValueError("Copied reference changed")
            print("Extraction complete; starting guideline coverage evaluation", flush=True)
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/benchmark_abcd_skill_guidelines.py"), "evaluate",
                "--reference", str(copied / reference.name), "--source-dir", str(source),
                "--rules-dir", str(rules_dir), "--output-dir", str(output / "coverage"), *common], cwd=ROOT, env=env, check=True)
            return
        if number < args.max_rounds:
            state = {"round": number + 1, "previous": str(current)}
            dump(state_path, state)
    raise RuntimeError("Some chunks still failed after subdivision; see " + str(output / "latest_extraction.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--qps", type=float, default=1.8)
    parser.add_argument("--chunk-chars", type=int, default=1600)
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()
    if args.chunk_chars < 500 or args.workers < 1 or args.qps <= 0 or args.max_rounds < 1:
        parser.error("chunk-chars >= 500 and positive workers/QPS/max-rounds required")
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Repair stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
