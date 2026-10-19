#!/usr/bin/env python3
"""Trace2Skill MAP/REDUCE/APPLY on the matched MultiWOZ hotel turn protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Trace2Skill"))

from eval_tod.multiwoz_policy import load_hotel_pilot, ontology_from_train, predict_policy
from eval_tod.sgd_adapter import gold_policy, score_policies, visible_context
from eval_tod.sgd_errors import error_summary, record_error
from eval_tod.sgd_llm import sgd_chat_with_retry
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_skill_agent import _json_object
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase
from eval_tod.reference_lookup import load_trace2skill_references


SEED_SKILL = """---
name: multiwoz_trace2skill
description: Evolving turn-level MultiWOZ action and slot policy
---

# MultiWOZ turn policy

Predict the next system turn's unordered set of MultiWOZ dialogue acts.
Each act has an intent, domain, and optional named slot and value. There is no
API CALL in this evaluation. Use only prior dialogue utterances; the current
system utterance, gold acts and future turns are hidden.

## Action choice

- Predict the unordered set of (intent, domain) heads for the next system turn.
- A head may have multiple named slots. General acts have an empty domain.
- Use only the dialogue observed before this system turn.

## Slot and value choice
- Emit one act per applicable named slot; omit value for requests.
- Copy entity values only when grounded in the visible dialogue.
- Never copy literal values from training examples as defaults.
"""


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _predict(case, model: str, ontology: dict, skill_dir: Path, aligned_predictor=None):
    dialogue, turn_index = case
    skill = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    references = load_trace2skill_references(skill_dir)
    predictor = aligned_predictor or predict_policy
    return predictor(dialogue, turn_index, model=model, ontology=ontology,
                          frequency_graph=None, skill_text=skill, reference_text=references)


def _analysis_job(job, model: str, *, dataset_name='MultiWOZ', task_contract=''):
    kind, index, cases, skill, raw_dir = job
    brief = [{"sample_id": row["sample_id"], "context": row["context"],
              "gold": row["gold"], "prediction": row["prediction"],
              **({"api_observation": row['api_observation'],
                  "observation_visible": row.get('observation_visible', False)} if 'api_observation' in row else {}),
              **({"shared_db_observation": row["shared_db_observation"]} if 'shared_db_observation' in row else {})}
             for row in cases]
    prompt = (f"Distill reusable {dataset_name} turn-policy lessons from TRAIN examples. "
              "The gold next-turn acts are visible only for this training analysis. "
              "Explain action-head selection and named-slot/value grounding separately. "
              "Do not memorize dialogue-specific values. Return JSON only: "
              '{"items":[{"title":"...","description":"...","content":"general rule..."}]}.'
              " At most 4 short items; an empty list is allowed. Keep each field "
              "under 100 words and close every JSON bracket.\n"
              + task_contract + f"\nEvidence type: {kind}\nCurrent skill:\n{skill[-8000:]}\n"
              "Train cases:\n" + json.dumps(brief, ensure_ascii=False))
    raw = sgd_chat_with_retry(prompt, model=model, temperature=0,
                              call_tag=f"trace2skill_{dataset_name.lower()}_{kind}_analysis")
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    for attempt in range(3):
        attempts.append(raw)
        _write_json(raw_dir / f"{kind}_{index:05d}.json",
                    {"responses": attempts, "sample_ids": [r["sample_id"] for r in cases]})
        try:
            payload = _json_object(raw)
            items = payload.get("items")
            if not isinstance(items, list):
                raise ValueError("Trace2Skill analysis omitted items")
            break
        except (ValueError, json.JSONDecodeError):
            if attempt == 2:
                raise
        repair = ("Repair the following malformed JSON. Return exactly one valid "
                  'JSON object with an "items" array and at most 4 short entries. '
                  "Preserve general lessons, omit case-specific values. No markdown.\n"
                  + raw[:6000])
        raw = sgd_chat_with_retry(
            repair, model=model, temperature=0,
            call_tag=f"trace2skill_{dataset_name.lower()}_{kind}_analysis_repair")
    clean = []
    for item in items[:4]:
        if isinstance(item, dict) and str(item.get("content", "")).strip():
            clean.append({"type": "failure_memory" if kind == "error" else "success_memory",
                          "title": str(item.get("title", ""))[:180],
                          "description": str(item.get("description", ""))[:800],
                          "content": str(item["content"])[:1200]})
    return {"instance_id": f"{kind}-batch-{index:04d}", "source_file": "train_turns.jsonl",
            "items": clean, "ast_evidence": brief}


class _EvolutionClient:
    def __init__(self, model: str, call_tag='trace2skill_multiwoz_evolution'):
        self.model = model
        self.call_tag = call_tag

    def chat(self, messages, settings=None):
        from llm import chat
        return chat(messages, model=self.model,
                    temperature=getattr(settings, "temperature", 0.3),
                    call_tag=self.call_tag)


def _evolve(records: list[dict], skill_dir: Path, output: Path, model: str,
            workers: int, *, dataset_name='multiwoz', task_contract='') -> dict:
    from skill_evolver.parallel_success_evolving_agent import (
        CombinedParallelSkillEvolver, normalize_mixed_records,
    )
    errors = [r for r in records if r["instance_id"].startswith("error-")]
    successes = [r for r in records if r["instance_id"].startswith("success-")]
    mixed = normalize_mixed_records(errors, successes)
    if not mixed:
        return {"total_llm_calls": 0, "changelog": [], "status": "no_records"}
    task_options = {'task_name': dataset_name.upper(), 'task_contract': task_contract} if task_contract else {}
    evolver = CombinedParallelSkillEvolver(
        client=_EvolutionClient(model, f'trace2skill_{dataset_name.lower()}_evolution'), skill_dir=str(skill_dir), batch_size=8,
        merge_batch_size=5, max_workers=workers, max_merge_levels=5,
        temperature=0.3, verbose=False, dry_run=False,
        prompt_variant="generic", output_dir=output / "evolution",
        parse_failure_dir=output / "parse_failures", max_skill_lines=500,
        skip_translation=False, patch_pipeline="json", **task_options)
    result = evolver.run(mixed, input_mode="records")
    return {"total_llm_calls": result.get("total_llm_calls", 0),
            "changelog": result.get("changelog", []),
            "status": "updated" if result.get("edits") else "no_edits"}


def run(args):
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    data = load_hotel_pilot(args.source, seed=args.seed, fraction=args.fraction)
    train, validation = data["train"], data["validation"]
    aligned_mode = getattr(args, 'runtime_protocol', 'legacy') == 'aligned'
    comparison = args.comparison_root / "awm" / "data_manifest.json"
    if comparison.is_file():
        comparison_manifest = json.loads(comparison.read_text(encoding="utf-8"))
        from eval_tod.multiwoz_aligned import PROTOCOL as aligned_protocol
        if aligned_mode and comparison_manifest.get('protocol') != aligned_protocol:
            raise ValueError('Aligned Trace2Skill requires a freshly trained aligned AWM comparison, not legacy resources')
        matched = comparison_manifest.get('selected_ids')
        if matched is None:
            matched = {'train': comparison_manifest['train_dialogue_ids'], 'validation': list(dict.fromkeys(
                case[0] for case in comparison_manifest['eval_case_ids']))}
        for split in ("train", "validation"):
            if [d["dialogue_id"] for d in data[split]] != matched[split]:
                raise ValueError(f"Trace2Skill {split} dialogue IDs do not match AWM")
    output.mkdir(parents=True)
    manifest = {"protocol": 'multiwoz21_trace2skill_aligned_v2_agent_reference' if aligned_mode else "multiwoz21_hotel_trace2skill_turn_ast_v2",
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "model": args.model, "fraction": args.fraction, "seed": args.seed,
                "workers": args.workers, "batch_dialogues": args.batch_dialogues,
                "train_dialogues": len(train), "validation_dialogues": len(validation),
                "train_turns": sum(len(d["target_indices"]["policy_turn_indices"]) for d in train),
                "validation_turns": sum(len(d["target_indices"]["policy_turn_indices"]) for d in validation),
                "comparison_root": str(args.comparison_root.resolve())}
    _write_json(output / "data_manifest.json", manifest)
    if args.dry_run:
        return manifest
    from llm import resolve_config
    resolve_config(model=args.model)
    configure_usage(output)
    ontology = ontology_from_train(train)
    skill_dir = output / "evolved_skill"
    skill_dir.mkdir()
    seed = SEED_SKILL
    if aligned_mode:
        from eval_tod.multiwoz_aligned import ALIGNED_TASK, COMMON_LEARNING_CONTRACT
        seed = seed.replace('Use only prior dialogue utterances;', 'Use prior dialogue utterances and the shared read-only DB;')
        seed = seed.replace('Copy entity values only when grounded in the visible dialogue.',
                            'Copy entity values only from the visible dialogue or shared read-only DB observation.')
        seed += '\n' + ALIGNED_TASK + '\n' + COMMON_LEARNING_CONTRACT
    (skill_dir / "SKILL.md").write_text(seed, encoding="utf-8")
    batches = [train[i:i + args.batch_dialogues] for i in range(0, len(train), args.batch_dialogues)]
    if args.max_batches:
        batches = batches[:args.max_batches]
    aligned_predictor = None
    if aligned_mode:
        from eval_tod.multiwoz_aligned import AlignedBaselinePredictor, ObservationCache, RuntimeLimits
        store = ObservationCache(args.observation_cache or output / 'observations', args.source.parent, args.model)
        preparation_cases = []
        for batch in batches:
            batch_cases = [(d, i) for d in batch for i in d['target_indices']['policy_turn_indices']]
            preparation_cases.extend(batch_cases[:args.max_cases] if args.max_cases else batch_cases)
        if not (args.max_batches or args.max_cases):
            preparation_cases += [(d, i) for d in validation for i in d['target_indices']['policy_turn_indices']]
        for outcome in ordered_cases(lambda case: store.get(*case), preparation_cases, workers=args.workers,
                                     directory=output, phase='shared_db_planning'):
            require_value(outcome)
        store.allow_generate = False
        aligned_predictor = AlignedBaselinePredictor(store, RuntimeLimits(
            reference_max_rounds=args.reference_max_rounds, reference_context_chars=args.reference_context_chars))
        from dataclasses import asdict
        from eval_tod.multiwoz_matched_executor import PROTOCOL as executor_protocol, digest
        manifest['execution_contract'] = {'executor_protocol': executor_protocol, 'model': args.model,
            'db_sha256': store.db_sha256, 'ontology_sha256': digest(ontology), 'retrieval': asdict(aligned_predictor.limits)}
        _write_json(output / 'data_manifest.json', manifest)
    set_phase("train_rollout")
    with (output / "training_turns.jsonl").open("w", encoding="utf-8") as stream:
        for batch_number, batch in enumerate(batches, 1):
            cases = [(d, i) for d in batch for i in d["target_indices"]["policy_turn_indices"]]
            if args.max_cases:
                cases = cases[:args.max_cases]
            print(f"Trace2Skill batch {batch_number}/{len(batches)}: {len(cases)} turns", flush=True)
            results = ordered_cases(lambda case: _predict(case, args.model, ontology, skill_dir, aligned_predictor),
                                    cases, workers=args.workers, directory=output,
                                    phase=f"train_rollout_batch_{batch_number:04d}")
            rows = []
            for outcome, (dialogue, turn_index) in zip(results, cases):
                gold = gold_policy(dialogue, turn_index)
                try:
                    prediction, diagnostic = require_value(outcome)
                    correct = bool(score_policies([gold], [prediction])["turn_joint_ast"])
                except Exception as exc:
                    record_error(output, "train_rollout", exc,
                                 dialogue_id=dialogue["dialogue_id"], turn_index=turn_index)
                    prediction, correct = {"call": None, "acts": [], "_execution_error": True}, False
                    diagnostic = {}
                row = {"sample_id": f"{dialogue['dialogue_id']}:{turn_index}",
                       "context": visible_context(dialogue, turn_index)[-1800:],
                       "gold": gold, "prediction": prediction, "joint_correct": correct,
                       "batch": batch_number}
                if aligned_mode:
                    row['shared_db_observation'] = diagnostic.get('shared_observation', {}).get('observations', [])
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            set_phase("trace2skill_analysis")
            skill = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
            jobs = []
            for kind, selected in (("error", [r for r in rows if not r["joint_correct"]]),
                                   ("success", [r for r in rows if r["joint_correct"]])):
                for start in range(0, len(selected), 8):
                    jobs.append((kind, batch_number * 1000 + start // 8,
                                 selected[start:start + 8], skill, output / "analysis_raw"))
            records = []
            if jobs:
                analyses = ordered_cases(lambda job: _analysis_job(job, args.model), jobs,
                                         workers=args.workers, directory=output,
                                         phase=f"analysis_batch_{batch_number:04d}")
                for outcome, job in zip(analyses, jobs):
                    try:
                        records.append(require_value(outcome))
                    except Exception as exc:
                        record_error(output, "trace2skill_analysis", exc,
                                     batch=batch_number, job=job[1])
            batch_dir = output / "batches" / f"batch_{batch_number:04d}"
            batch_dir.mkdir(parents=True)
            _write_json(batch_dir / "analysis_records.json", records)
            set_phase("trace2skill_evolution")
            try:
                evolved = _evolve(records, skill_dir, batch_dir, args.model, args.workers)
            except Exception as exc:
                record_error(output, "trace2skill_evolution", exc, batch=batch_number)
                evolved = {"status": "error", "error": repr(exc)}
            _write_json(batch_dir / "summary.json", {"turns": len(rows),
                        "joint_correct": sum(r["joint_correct"] for r in rows),
                        "analysis_records": len(records), "evolution": evolved})
            print(f"Trace2Skill batch {batch_number}: "
                  f"{sum(r['joint_correct'] for r in rows)}/{len(rows)} train turns exact; "
                  f"evolution={evolved['status']}", flush=True)
            set_phase("train_rollout")
    if args.max_batches or args.max_cases:
        return {"status": "smoke_completed", **manifest}
    set_phase("testing")
    cases = [(d, i) for d in validation for i in d["target_indices"]["policy_turn_indices"]]
    predictions = ordered_cases(lambda case: _predict(case, args.model, ontology, skill_dir, aligned_predictor),
                                cases, workers=args.workers, directory=output, phase="testing")
    gold_rows, pred_rows = [], []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for outcome, (dialogue, turn_index) in zip(predictions, cases):
            gold = gold_policy(dialogue, turn_index)
            try:
                prediction, diagnostic = require_value(outcome)
            except Exception as exc:
                record_error(output, "testing", exc,
                             dialogue_id=dialogue["dialogue_id"], turn_index=turn_index)
                prediction, diagnostic = {"call": None, "acts": [], "_execution_error": True}, {"status": "error"}
            gold_rows.append(gold)
            pred_rows.append(prediction)
            stream.write(json.dumps({"dialogue_id": dialogue["dialogue_id"], "turn_index": turn_index,
                                     "gold": gold, "prediction": prediction,
                                     "diagnostic": diagnostic}, ensure_ascii=False) + "\n")
    errors = error_summary(output)
    summary = {**manifest, **score_policies(gold_rows, pred_rows),
               "case_errors": errors, "llm_usage": persist_usage(),
               "execution_status": "completed_with_errors" if errors["total"] else "completed"}
    _write_json(output / "summary.json", summary)
    print(json.dumps({"turn_joint_ast": summary["turn_joint_ast"],
                      "action_set_exact": summary["action_set_exact"],
                      "execution_status": summary["execution_status"]}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/eval/multiwoz21/data/data/dialogues.json")
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=.1)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--batch-dialogues", type=int, default=20)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument('--runtime-protocol', choices=['legacy', 'aligned'], default='legacy')
    parser.add_argument('--observation-cache', type=Path)
    parser.add_argument('--reference-max-rounds', type=int, default=4)
    parser.add_argument('--reference-context-chars', type=int, default=32768)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_dialogues < 1:
        parser.error("workers and batch-dialogues must be positive")
    result = run(args)
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
