#!/usr/bin/env python3
"""Retry rolled-back native reflection groups, then evaluate saved resources."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.multiwoz_aligned import ObservationCache, RuntimeLimits, atomic_json
from eval_tod.multiwoz_native import NativeMultiWOZSkillAgent, PROTOCOL
from eval_tod.multiwoz_policy import load_hotel_pilot
from eval_tod.sgd_adapter import gold_policy, score_policies
from eval_tod.sgd_errors import error_summary, record_error
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase
from scripts.replay_multiwoz_reflection import restore_failed_groups, combine_usage
from scripts.run_multiwoz_aligned_pilot import progress
from scripts.run_multiwoz_matched_eval import recover_logged_usage
from skill_mining.online_refinement import group_batch_reports


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare_agent(source, output, dataset, workers):
    manifest = read(source / 'data_manifest.json')
    if manifest['protocol'] != PROTOCOL or hashlib.sha256(dataset.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('Native protocol/dataset changed')
    contract = read(source / 'runtime_contract.json')
    cache = ObservationCache(source / 'observations', dataset.parent, manifest['model'], allow_generate=False)
    agent = NativeMultiWOZSkillAgent(family='hotel', model=manifest['model'],
        ontology=read(source / 'train_ontology.json'), predictor=None, observations=cache,
        limits=RuntimeLimits(**contract['retrieval']), mining_method='trace_cover')
    agent.load(source)
    training = read(source / 'ours_training.json')
    if agent.resource_hash() != training['final_sha256'] or training['failed_rollout_turns']:
        raise ValueError('Completed frozen rollout/checkpoint is not intact')
    reports = [read(path) for path in sorted(source.glob('diagnosis_[0-9][0-9][0-9][0-9].json'))]
    if len(reports) != training['reports'] or any(r['status'] != 'ok' for r in reports):
        raise ValueError('Complete successful saved diagnoses are required')
    groups = group_batch_reports(reports, max_reports=16)
    failures = [json.loads(line) for line in (source / 'case_errors.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if not failures or any(row['phase'] != 'online_reflection' or not row.get('rolled_back') for row in failures):
        raise ValueError('Only explicitly rolled-back reflection groups may be resumed')
    indices = [row['group_index'] for row in failures]
    selected = restore_failed_groups(agent, source, output, groups, indices)
    agent.parallel_workers = workers
    agent.runtime_diagnostics_dir = output
    return agent, manifest, contract, training, selected, indices


def run(args):
    source, output, dataset = args.source_run.resolve(), args.output_dir.resolve(), args.dataset.resolve()
    launch_files = {'launch.json', 'run.log', 'awake.log', 'monitor.log', 'code_snapshot', 'code_snapshot.json'}
    resume = getattr(args, 'resume_checkpoint', False)
    if not resume and output.exists() and any(path.name not in launch_files for path in output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        manifest, contract = read(output / 'data_manifest.json'), read(source / 'runtime_contract.json')
        assert Path(manifest['source_run']).resolve() == source
        assert hashlib.sha256(dataset.read_bytes()).hexdigest() == manifest['source_sha256']
        previous_training, previous_usage = read(source / 'ours_training.json'), read(output / 'previous_attempt_usage.json')
        agent = NativeMultiWOZSkillAgent(family='hotel', model=manifest['model'],
            ontology=read(source / 'train_ontology.json'), predictor=None,
            observations=ObservationCache(source / 'observations', dataset.parent, manifest['model'], allow_generate=False),
            limits=RuntimeLimits(**contract['retrieval']), mining_method='trace_cover')
        agent.load(output)
        if agent.stage != 'refined' or (output / 'predictions.jsonl').exists():
            raise ValueError('Resume requires a completed repaired checkpoint before evaluation')
        committed = [entry['group_index'] for entry in agent.ledger if entry.get('phase') == 'online_reflection']
        if len(committed) != previous_training['reflection_groups'] or len(set(committed)) != len(committed):
            raise ValueError('Resume must contain each committed reflection exactly once')
        configure_usage(output)
    else:
        agent, manifest, contract, previous_training, selected, indices = prepare_agent(source, output, dataset, args.workers)
        # Original usage stays at its source; new calls are combined once.
        recover_logged_usage(source)
        previous_usage = read(source / 'llm_usage.json')
        atomic_json(output / 'previous_attempt_usage.json', previous_usage)
        configure_usage(output)
        shutil.copytree(source / 'offline', output / 'offline', ignore=shutil.ignore_patterns('parallel_cases'))
        for name in ['ours_training_turns.jsonl', 'online_schedule.json', 'train_ontology.json']:
            shutil.copy2(source / name, output / name)
        for path in source.glob('diagnosis_[0-9][0-9][0-9][0-9].json'):
            shutil.copy2(path, output / path.name)
        manifest = {**manifest, 'source_run': str(source), 'workers': args.workers,
            'observation_cache': str(source / 'observations'),
            'reflection_recovery': {'retry_groups': indices, 'committed_groups_retained': previous_training['reflection_groups'] - len(indices),
                'retry_order': 'Rebase rolled-back groups on the final previously committed state',
                'training_predictions_reused': previous_training['rollout_turns'], 'new_train_prediction_calls': 0}}
        atomic_json(output / 'data_manifest.json', manifest)
        progress(output, 'retry_failed_reflection', 0, len(indices))
        agent.stage = 'refining'
        agent.reflect_groups(selected, output, group_indices=indices)
    failures = error_summary(output)
    if failures['total']:
        raise RuntimeError('Failed reflection remains unresolved; successful source resources are unchanged')
    agent.state['refinement_errors'] = failures
    agent.stage = 'refined'
    agent.save(output)
    if {**agent.runtime_contract(), 'cards_ready': agent.cards_ready} != contract:
        raise ValueError('Recovery changed the execution contract')
    training = {**previous_training, 'final_sha256': agent.resource_hash(), 'case_errors': failures,
        'reflection_recovery': manifest['reflection_recovery']}
    atomic_json(output / 'ours_training.json', training)
    data = load_hotel_pilot(dataset, seed=manifest['seed'], fraction=manifest['fraction'])
    assert [d['dialogue_id'] for d in data['train']] == manifest['train_dialogue_ids']
    lookup = {d['dialogue_id']: d for d in data[manifest['split']]}
    cases = [(lookup[identifier], turn) for identifier, turn in manifest['eval_case_ids']]
    # Reload through the same persisted artifact interface used by the main run.
    frozen = NativeMultiWOZSkillAgent(family='hotel', model=manifest['model'],
        ontology=read(source / 'train_ontology.json'), predictor=None, observations=agent.observations,
        limits=agent.limits, mining_method='trace_cover')
    frozen.load(output)
    assert frozen.resource_hash() == training['final_sha256']
    set_phase('testing')
    progress(output, 'evaluation', 0, len(cases))
    outcomes = ordered_cases(lambda case: frozen.predict(*case), cases, workers=args.workers,
                             directory=output, phase='testing')
    rows = []
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for number, (outcome, (dialogue, index)) in enumerate(zip(outcomes, cases), 1):
            try:
                prediction, diagnostic = require_value(outcome)
            except Exception as exc:
                record_error(output, 'testing', exc, dialogue_id=dialogue['dialogue_id'], turn_index=index)
                prediction, diagnostic = {'call': None, 'acts': [], '_execution_error': True}, {'status': 'error'}
            row = {'dialogue_id': dialogue['dialogue_id'], 'turn_index': index,
                   'gold': gold_policy(dialogue, index), 'prediction': prediction, 'diagnostic': diagnostic}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            progress(output, 'evaluation', number, len(cases))
    assert frozen.resource_hash() == training['final_sha256']
    current_usage = persist_usage()
    failures = error_summary(output)
    summary = {**manifest, 'training': training,
        **score_policies([r['gold'] for r in rows], [r['prediction'] for r in rows]),
        'llm_usage': combine_usage(previous_usage, current_usage), 'recovery_usage': current_usage,
        'case_errors': failures, 'execution_status': failures['status']}
    atomic_json(output / 'summary.json', summary)
    if not failures['total']:
        from scripts.audit_multiwoz_aligned_run import audit
        atomic_json(output / 'protocol_audit.json', audit(output))
    progress(output, 'completed', len(rows), len(cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/eval/multiwoz21/data/data/dialogues.json')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--resume-checkpoint', action='store_true',
                        help='Reuse an already repaired checkpoint without repeating reflection')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir / 'run_failed.json', {'error': str(exc), 'type': type(exc).__name__})
        raise
