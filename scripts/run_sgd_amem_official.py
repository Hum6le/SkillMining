#!/usr/bin/env python3
"""Pinned official A-MEM memory + the existing SGD shared executor.

TRAIN policy examples create notes sequentially; evaluation is read-only and
parallel. This is an SGD adaptation, not a reproduction of LoCoMo metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amem_adapter.official import OfficialMemory, MemoryCheckpoint, SharedController, provenance, file_sha
from amem_adapter.sgd import PROTOCOL as AMEM_PROTOCOL, train_note, predict
from eval_tod.multiwoz_aligned import atomic_json
from eval_tod.sgd_adapter import iter_family_dialogues, score_policies, metrics_from_counts
from eval_tod.sgd_aligned import PROTOCOL, training_contract, digest
from eval_tod.sgd_extra_baselines import predict_row
from eval_tod.sgd_extra_config import configure
from eval_tod.sgd_local_config import DEFAULT_ENV
from eval_tod.sgd_local_parallel import local_cases
from eval_tod.sgd_parallel import require_value
from eval_tod.sgd_usage import configure_usage, set_phase
from scripts.run_sgd_extra_baselines import families, read, journal_usage, now


def prepare_encoder(args):
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(args.embedding_threads)
    path = Path(args.embedding_model)
    revision = args.embedding_revision
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(args.embedding_model, revision=revision,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"], max_workers=2))
        revision = path.name
    signature = {"requested_model": args.embedding_model, "revision": revision,
        "files": {p.relative_to(path).as_posix(): file_sha(p) for p in sorted(path.rglob("*"))
                  if p.is_file() and p.suffix in {".safetensors", ".json", ".txt"}},
        "device": "cpu"}
    if not any(p.endswith(".safetensors") for p in signature["files"]):
        raise ValueError("Expected real local MiniLM safetensors weights; no substitute embeddings")
    return SentenceTransformer(str(path), device="cpu"), signature


def run_family(args, family, config, encoder_provider):
    out = args.output_dir / family
    out.mkdir(parents=True, exist_ok=True)
    train = list(iter_family_dialogues(args.splits_dir, family, "train"))
    evaluation = list(iter_family_dialogues(args.splits_dir, family, "validation"))
    if args.max_train_dialogues:
        train = train[:args.max_train_dialogues]
    if args.max_eval_dialogues:
        evaluation = evaluation[:args.max_eval_dialogues]
    if {d["dialogue_id"] for d in train} & {d["dialogue_id"] for d in evaluation}:
        raise ValueError("TRAIN/validation overlap")
    train_cases = [(d, i) for d in train for i in d["target_indices"]["policy_turn_indices"]]
    eval_cases = [(d, i) for d in evaluation for i in d["target_indices"]["policy_turn_indices"]]
    if args.max_train_notes:
        train_cases = train_cases[:args.max_train_notes]
    if args.max_eval_turns:
        eval_cases = eval_cases[:args.max_eval_turns]
    sample_id = lambda case: f"{case[0]['dialogue_id']}:{case[1]}"
    with ZipFile(args.archive) as stream:
        ontology_bytes = stream.read("data/ontology.json")
    ontology = json.loads(ontology_bytes)
    ontology["_sgd_training_contract"] = training_contract(train)
    manifest = {"protocol": AMEM_PROTOCOL, "executor_protocol": PROTOCOL, "family": family,
        "model": config["model"], "provider": config["provider"], "base_url": config["base_url"],
        "enable_thinking": config["enable_thinking"], "official": provenance(args.upstream_dir),
        "train_sample_ids": [sample_id(c) for c in train_cases], "evaluation_sample_ids": [sample_id(c) for c in eval_cases],
        "data_sha256": {s: file_sha(args.splits_dir / family / (s + ".jsonl.gz")) for s in ("train", "validation")},
        "ontology_sha256": digest(json.loads(ontology_bytes)), "training_contract_sha256": digest(ontology["_sgd_training_contract"]),
        "retrieve_k": args.retrieve_k, "refresh_index": args.refresh_index,
        "embedding_model": args.embedding_model, "embedding_revision": args.embedding_revision,
        "compare_react": args.compare_react, "smoke": args.smoke,
        "training_order": "dataset_dialogue_order_then_policy_turn; native evolution is sequential",
        "evaluation": "all_policy_turns; no_current_or_future_gold_in_retrieval; frozen_memory",
        "source_sha256": {name: file_sha(ROOT / name) for name in
            ["amem_adapter/official.py", "amem_adapter/sgd.py", "scripts/run_sgd_amem_official.py"]}}
    if config["provider"] == "workflow":
        manifest.update(workflow_pool_method=config["workflow_pool_method"], workflow_settings=config["workflow_settings"])
    previous = read(out / "data_manifest.json")
    if previous is not None and previous != manifest:
        raise ValueError("Resume data/protocol mismatch; use a new output directory")
    if args.dry_run:
        atomic_json(out / "preflight.json", manifest)
        return {"method": "amem_official", "domain": family, "preflight": True,
                "train_notes": len(train_cases), "eval_turns": len(eval_cases)}
    atomic_json(out / "data_manifest.json", manifest)
    complete = read(out / "summary.json", {})
    if complete.get("evaluation_complete"):
        if complete["manifest_sha256"] != digest(manifest) or file_sha(out / "memory.sqlite") != complete["memory_file_sha256"]:
            raise ValueError("Completed A-MEM artifact no longer matches")
        print(f"SKIP verified amem_official/{family}", flush=True)
        return complete
    configure_usage(out)
    controller = SharedController(config["model"], out / "amem_completions.jsonl")
    encoder, encoder_signature = encoder_provider()
    stored_signature = read(out / "embedding_signature.json")
    if stored_signature is not None and stored_signature != encoder_signature:
        raise ValueError("Embedding weights changed across resume")
    atomic_json(out / "embedding_signature.json", encoder_signature)
    memory = OfficialMemory(encoder, controller, refresh_index=args.refresh_index, upstream_dir=args.upstream_dir)
    checkpoint = MemoryCheckpoint(out / "memory.sqlite", digest(manifest))
    completed = checkpoint.restore(memory)
    if completed > len(train_cases):
        raise ValueError("More saved notes than TRAIN targets")
    started = time.time()
    def progress(phase, n, total, **extra):
        row = {"method": "amem_official", "family": family, "phase": phase, "completed": n,
               "total": total, "updated_at": now(), "elapsed_seconds": time.time() - started, **extra}
        atomic_json(out / "progress.json", row)
        atomic_json(args.output_dir / "progress.json", row)
        if n % 20 == 0 or n == total:
            print(f"amem_official/{family} {phase}: {n}/{total}", flush=True)
    set_phase("amem_memory_construction")
    progress("memory_construction", completed, len(train_cases))
    try:
        for number, (dialogue, index) in enumerate(train_cases[completed:], completed + 1):
            memory.add(train_note(dialogue, index), sample_id((dialogue, index)))
            checkpoint.commit(memory, number)
            progress("memory_construction", number, len(train_cases))
    finally:
        checkpoint.close()
    if not memory.system.memories:
        raise ValueError("A-MEM has no TRAIN memories")
    frozen_hash = memory.state_hash()
    database_hash = file_sha(out / "memory.sqlite")
    notes = list(memory.system.memories.values())
    resources = {"type": "official_amem", "protocol": AMEM_PROTOCOL,
        "memory_sha256": frozen_hash, "memory_file_sha256": database_hash,
        "notes": len(notes), "links": sum(len(n.links) for n in notes), "evolution_count": memory.system.evo_cnt,
        "official_failures_current_process": dict(memory.base.FAILURES), "refresh_index": args.refresh_index,
        "retrieve_k": args.retrieve_k, "official": manifest["official"], "embedding": encoder_signature}
    existing_resources = read(out / "resources.json")
    if existing_resources and existing_resources["memory_sha256"] != frozen_hash:
        raise ValueError("TRAIN artifact changed while resuming frozen evaluation")
    atomic_json(out / "resources.json", resources)
    set_phase("testing")
    methods = ["amem_official"] + (["react"] if args.compare_react else [])
    all_metrics, all_errors = {}, {}
    expected_ids = [sample_id(c) for c in eval_cases]
    for method in methods:
        output = out / method
        output.mkdir(exist_ok=True)
        path = output / "predictions.jsonl"
        cached = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []
        prior = {r["sample_id"]: r for r in cached}
        if len(prior) != len(cached) or not prior.keys() <= set(expected_ids):
            raise ValueError("Duplicate/unexpected saved evaluation target")
        if any(r.get("frozen_resource_sha256") != frozen_hash for r in cached):
            raise ValueError("Predictions use a different frozen memory")
        pending = [c for c in eval_cases if sample_id(c) not in prior]
        def evaluate(case):
            d, i = case
            if method == "react":
                return predict_row(d, i, config["model"], ontology, "")
            return predict(memory, controller, d, i, config["model"], ontology, k=args.retrieve_k)
        progress("testing_" + method, len(prior), len(expected_ids))
        with path.open("a", encoding="utf-8") as stream:
            for outcome in local_cases(evaluate, pending, workers=args.workers, directory=out, phase="testing"):
                row = require_value(outcome)
                row["frozen_resource_sha256"] = frozen_hash
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                prior[row["sample_id"]] = row
                progress("testing_" + method, len(prior), len(expected_ids))
        if set(prior) != set(expected_ids):
            raise RuntimeError("Incomplete evaluation")
        rows = [prior[sid] for sid in expected_ids]
        metrics = score_policies([r["gold"] for r in rows], [r["prediction"] for r in rows])
        errors = sum(bool(r["error"]) for r in rows)
        atomic_json(output / "summary.json", {"method": method, "metrics": metrics, "case_errors": errors,
                    "evaluation_complete": True, "memory_sha256": frozen_hash})
        all_metrics[method] = metrics
        all_errors[method] = errors
    if memory.state_hash() != frozen_hash or file_sha(out / "memory.sqlite") != database_hash:
        raise RuntimeError("Memory mutated during held-out evaluation")
    summary = {"method": "amem_official", "domain": family, "evaluation_complete": True,
        "manifest_sha256": digest(manifest), "memory_sha256": frozen_hash, "memory_file_sha256": database_hash,
        "metrics": all_metrics["amem_official"], "methods": all_metrics, "case_errors": all_errors,
        "generation": resources, "llm_usage": journal_usage(out), "memory_frozen_verified": True,
        "finished_at": now(), "smoke": args.smoke}
    atomic_json(out / "summary.json", summary)
    progress("complete", len(expected_ids), len(expected_ids), case_errors=all_errors)
    return summary


def run(args):
    config = configure(args)
    chosen = args.families.split(",") if args.families else families(args.splits_dir)
    if any(f not in families(args.splits_dir) for f in chosen):
        raise ValueError("Use the same nine seen-schema validation families as the main experiments")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "run_config.json", {**config, "families": chosen, "workers": args.workers,
        "smoke": args.smoke, "started_at": now(), "dataset": str(args.splits_dir.resolve())})
    encoder = []
    def encoder_provider():
        if not encoder:
            encoder.append(prepare_encoder(args))
        return encoder[0]
    summaries = []
    try:
        for family in chosen:
            summaries.append(run_family(args, family, config, encoder_provider))
            if args.dry_run:
                continue
            aggregate = {"method": "amem_official", "completed_families": [s["domain"] for s in summaries],
                "expected_families": chosen, "evaluation_complete": len(summaries) == len(chosen),
                "smoke": args.smoke, "methods": {}, "generation": {s["domain"]: s["generation"] for s in summaries}}
            for method in summaries[0]["methods"]:
                counts = Counter()
                for s in summaries:
                    counts.update(s["methods"][method]["counts"])
                aggregate["methods"][method] = {**metrics_from_counts(counts),
                    "case_errors": sum(s["case_errors"][method] for s in summaries)}
            from scripts.llm_usage_utils import merge_usage_summaries
            aggregate["llm_usage"] = {phase: merge_usage_summaries({"total": {}},
                *(s["llm_usage"].get(phase, {}) for s in summaries)) for phase in ("generation", "testing", "total")}
            atomic_json(args.output_dir / "aggregate.json", aggregate)
        if args.dry_run:
            atomic_json(args.output_dir / "preflight_report.json", {"status": "passed", "families": summaries,
                "train_notes": sum(s["train_notes"] for s in summaries),
                "eval_turns": sum(s["eval_turns"] for s in summaries), "llm_calls": 0})
            print("A-MEM preflight passed: " + str(args.output_dir / "preflight_report.json"), flush=True)
        return summaries
    except Exception as exc:
        atomic_json(args.output_dir / "fatal_error.json", {"error": f"{type(exc).__name__}: {exc}", "time": now()})
        raise


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits-dir", type=Path, default=ROOT / "data/eval/sgd/protocol_20pct_seed42_seen_schema_v1")
    p.add_argument("--archive", type=Path, default=ROOT / "data/eval/sgd/data.zip")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--upstream-dir", type=Path, default=ROOT / "third_party/a-mem-official")
    p.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    p.add_argument("--provider", choices=["shubiaobiao", "workflow"], default="shubiaobiao")
    p.add_argument("--workflow-pools", type=Path, default=ROOT / "configs/sgd_workflows.json")
    p.add_argument("--workflow-pool-method", default="awm")
    p.add_argument("--model")
    p.add_argument("--families", default="")
    p.add_argument("--workers", type=int, default=4, help="Evaluation threads; native sequential TRAIN evolution is preserved")
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedding-revision", default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    p.add_argument("--embedding-threads", type=int, default=2)
    p.add_argument("--retrieve-k", type=int, default=10)
    p.add_argument("--refresh-index", action="store_true", help="Official optional immediate refresh; default paper delayed consolidation")
    p.add_argument("--compare-react", action="store_true", help="Evaluate a paired no-memory ReAct diagnostic on identical targets")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-train-dialogues", type=int)
    p.add_argument("--max-train-notes", type=int)
    p.add_argument("--max-eval-dialogues", type=int)
    p.add_argument("--max-eval-turns", type=int)
    p.add_argument("--dry-run", action="store_true", help="Data/source/config checks; no model import, download or API request")
    a = p.parse_args(argv)
    if not 1 <= a.workers <= 10 or a.retrieve_k < 1 or a.embedding_threads < 1:
        p.error("Use 1..10 evaluation workers and positive retrieval/CPU thread counts")
    limits = [a.max_train_dialogues, a.max_train_notes, a.max_eval_dialogues, a.max_eval_turns]
    if any(v is not None and v < 1 for v in limits):
        p.error("Smoke limits must be positive")
    if not a.smoke and any(v is not None for v in limits):
        p.error("Truncation requires --smoke; it cannot be reported as a full run")
    return a


if __name__ == "__main__":
    run(arguments())
