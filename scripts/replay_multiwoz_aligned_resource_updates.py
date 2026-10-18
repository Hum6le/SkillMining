#!/usr/bin/env python3
"""Replay explicit TRAIN reflection decisions through corrected typed writes.

Reuses frozen TRAIN predictions and the model's original reflection JSON; it
does not generate new training supervision or rewrite the source experiment.
Only the held-out evaluation is called again after the corrected checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.multiwoz_aligned import MultiWOZAlignedSkillAgent, ObservationCache, RuntimeLimits, atomic_json
from eval_tod.multiwoz_policy import load_hotel_pilot
from eval_tod.sgd_adapter import gold_policy, score_policies
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase
from skill_mining.online_refinement import group_batch_reports
from scripts.run_multiwoz_aligned_pilot import progress


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class ReplayAgent(MultiWOZAlignedSkillAgent):
    def _chat(self, messages, **kwargs):
        if kwargs['call_tag'] != 'online_group_reflection':
            raise ValueError('TRAIN replay may not issue new model requests')
        if not self.replay_decisions:
            raise ValueError('Original reflection decisions exhausted')
        return self.replay_decisions.pop(0)


def run(args):
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    manifest = read(source / 'data_manifest.json')
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('Dataset changed')
    output.mkdir(parents=True, exist_ok=True)
    configure_usage(output)
    ontology = read(source / 'train_ontology.json')
    cache = ObservationCache(source / 'observations', args.dataset.parent, manifest['model'], allow_generate=False)
    limits = RuntimeLimits(**read(source / 'runtime_contract.json')['retrieval'])
    agent = ReplayAgent(family='hotel', model=manifest['model'], ontology=ontology,
        predictor=None, observations=cache, limits=limits, mining_method='trace_cover')
    agent.parallel_workers = args.workers
    agent.load(source / 'offline')
    original_training = read(source / 'ours_training.json')
    assert agent.resource_hash() == original_training['snapshot_sha256'], 'Frozen TRAIN runtime/resources changed'
    records = [json.loads(line) for line in (source / 'ours_training_turns.jsonl').read_text(encoding='utf-8').splitlines()]
    shutil.copytree(source / 'offline', output / 'offline')
    for name in ['ours_training_turns.jsonl', 'online_schedule.json', 'train_ontology.json']:
        shutil.copyfile(source / name, output / name)
    manifest = {**manifest, 'training_mode': 'same_frozen_train_rollouts_and_original_reflection_typed_serialization_repair',
                'source_run': str(source), 'workers': args.workers}
    atomic_json(output / 'data_manifest.json', manifest)
    agent.stage = 'refining'
    agent._accumulate(records)
    reports = []
    for path in sorted(source.glob('diagnosis_[0-9][0-9][0-9][0-9].json')):
        report = agent.normalize_diagnosis(read(path))
        reports.append(report)
        atomic_json(output / path.name, report)
    groups = group_batch_reports(reports, max_reports=16)
    decisions = [read(p) for p in sorted(source.glob('reflection_[0-9][0-9][0-9][0-9].json'))]
    if len(groups) != len(decisions):
        raise ValueError('TRAIN reflection grouping changed')
    for group, decision in zip(groups, decisions):
        assert [r['batch_id'] for r in group] == decision['batch_ids']
    agent.replay_decisions = [d['raw_response'] for d in decisions]
    progress(output, 'typed_resource_update_replay', 0, len(groups))
    agent.reflect_groups(groups, output, edit_attempts=1)
    if agent.replay_decisions:
        raise ValueError('Original decisions not fully replayed')
    agent.stage = 'refined'
    agent.save(output)
    training = {**original_training, 'final_sha256': agent.resource_hash(),
                'reused_frozen_train_rollouts': True, 'new_train_model_calls': 0}
    atomic_json(output / 'ours_training.json', training)
    provenance = {'source_run': str(source), 'new_train_model_calls': 0,
        'unchanged_train_prediction_file_sha256': hashlib.sha256((source / 'ours_training_turns.jsonl').read_bytes()).hexdigest(),
        'original_reflection_raw_sha256': [hashlib.sha256(d['raw_response'].encode()).hexdigest() for d in decisions],
        'repair': 'Unpack explicit composite card.rule and card.payload_policy into native action_rule and slot_policy stores',
        'evaluation': 'Fresh held-out calls through the same executor, original cases and shared DB; source results unchanged'}
    atomic_json(output / 'resource_update_repair.json', provenance)
    loaded = MultiWOZAlignedSkillAgent(family='hotel', model=manifest['model'], ontology=ontology,
        predictor=None, observations=cache, limits=limits, mining_method='trace_cover')
    loaded.load(output)
    assert loaded.resource_hash() == agent.resource_hash()
    lookup = {d['dialogue_id']: d for d in load_hotel_pilot(args.dataset, seed=manifest['seed'], fraction=manifest['fraction'])[manifest['split']]}
    cases = [(lookup[identifier], turn) for identifier, turn in manifest['eval_case_ids']]
    set_phase('testing')
    progress(output, 'evaluation', 0, len(cases))
    outcomes = ordered_cases(lambda case: loaded.predict(*case), cases, workers=args.workers,
                             directory=output, phase='testing')
    results = []
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for n, (outcome, (dialogue, turn)) in enumerate(zip(outcomes, cases), 1):
            prediction, diagnostic = require_value(outcome)
            row = {'dialogue_id': dialogue['dialogue_id'], 'turn_index': turn,
                   'gold': gold_policy(dialogue, turn), 'prediction': prediction, 'diagnostic': diagnostic}
            results.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            progress(output, 'evaluation', n, len(cases))
    summary = {**manifest, 'training': training, 'resource_update_repair': provenance,
        **score_policies([r['gold'] for r in results], [r['prediction'] for r in results]), 'llm_usage': persist_usage()}
    atomic_json(output / 'summary.json', summary)
    progress(output, 'completed', len(results), len(cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/eval/multiwoz21/data/data/dialogues.json')
    parser.add_argument('--workers', type=int, default=5)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception as exc:
        if arguments.output_dir.exists():
            atomic_json(arguments.output_dir / 'run_failed.json', {'error': str(exc), 'type': type(exc).__name__})
        raise
