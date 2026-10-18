#!/usr/bin/env python3
"""Evaluate new offline slot resources on a frozen legacy trained graph.

This is a diagnostic intervention, not a new aligned TRAIN/refine experiment.
All outputs live in a new directory; source checkpoints and scores stay intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_tod.multiwoz_aligned import (
    PROTOCOL, OFFLINE_SLOT_PROTOCOL, MultiWOZAlignedSkillAgent, ObservationCache,
    atomic_json, compile_cards_job, initial_reference_text, initial_slot_policy,
    offline_slot_report, training_profiles,
)
from eval_tod.multiwoz_matched_executor import (
    PROTOCOL as EXECUTOR_PROTOCOL, digest, load_kb, query_records,
)
from eval_tod.multiwoz_policy import MultiWOZStandardSkillAgent, load_hotel_pilot, ontology_from_train
from eval_tod.sgd_adapter import gold_policy, score_policies, visible_context
from eval_tod.sgd_errors import record_error
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_usage import configure_usage, persist_usage
from scripts.run_multiwoz_matched_eval import recover_logged_usage

MODE = 'frozen_legacy_training_with_new_offline_slot_resources_diagnostic'
SHANGHAI = timezone(timedelta(hours=8))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def observation_job(job):
    cache, dialogue, turn = job
    cache.get(dialogue, turn)


def reuse_matched_observations(directory, cases, cache, source_sha256):
    """Verify old context fingerprints and replay every query against today's DB.

    v1 -> v2 changed payload binary normalization, not the context-only DB
    observation. The migration is explicit and records the original provenance.
    """
    manifest = read(directory / 'manifest.json')
    if (manifest['protocol'] not in {'multiwoz21_hotel_shared_db_action_payload_v1', EXECUTOR_PROTOCOL}
            or manifest['source_sha256'] != source_sha256 or manifest['model'] != cache.model):
        raise ValueError('Matched DB cache dataset/model/protocol mismatch')
    if digest(read(directory / 'db_manifest.json')) != cache.db_sha256:
        raise ValueError('Matched DB contents changed')
    kb = load_kb(str(cache.kb_dir))
    source_signature, migrated = digest(manifest), []
    for index, (dialogue, turn) in enumerate(cases):
        path = directory / 'observations' / f'case_{index:04d}.json'
        saved = read(path)
        expected = digest([source_signature, cache.db_sha256, dialogue['dialogue_id'], turn,
                           visible_context(dialogue, turn)])
        observation = saved['observation']
        if (saved['protocol'] != manifest['protocol'] or saved['signature'] != expected
                or observation['context_sha256'] != digest(visible_context(dialogue, turn)[-12000:])):
            raise ValueError(f'Matched observation input mismatch: {path}')
        replayed = []
        for query in observation['plan']['queries']:
            records = query_records(kb, query['domain'], query['constraints'])
            replayed.append({**query, 'match_count': len(records), 'records': records[:5],
                             'records_truncated': len(records) > 5})
        if replayed != observation['observations']:
            raise ValueError(f'Matched DB replay changed: {path}')
        signature = cache.identity(dialogue, turn)
        target = cache.directory / f'{signature}.json'
        payload = {'signature': signature, 'executor_protocol': EXECUTOR_PROTOCOL,
                   'observation': observation, 'cache_migration': {
                       'source': str(path.resolve()), 'source_protocol': saved['protocol'],
                       'source_signature': saved['signature'], 'db_replay_identical': True}}
        if not target.exists():
            atomic_json(target, payload)
        else:
            cache.get(dialogue, turn)
        migrated.append(signature)
    return migrated


def build_diagnostic_agent(legacy, agent, profiles, cards):
    """Retain learned routing/actions, replace slot policies and seed references."""
    if set(cards) != {n['id'] for n in legacy.graph['nodes']}:
        raise ValueError('Diagnostic cards do not cover the frozen graph')
    agent.graph, agent.resources = deepcopy(legacy.graph), deepcopy(legacy.resources)
    agent.state = deepcopy(legacy.state)
    old_slots = deepcopy(agent.state['slot_policies'])
    agent.state['slot_policies'] = {}
    # These histories do not participate in prediction. Keep their originals
    # in the frozen source snapshot rather than copy them into every worker.
    for key in ('evidence_pool', 'patches'):
        value = agent.state.get(key)
        agent.state[key] = {} if isinstance(value, dict) else []
    for bucket in ('edges', 'action_rules'):
        for record in agent.state[bucket].values():
            record.pop('evidence', None)
    by_id = {p['node_id']: p for p in profiles}
    for node_id, card in cards.items():
        agent.resources['action_rules'][node_id] = card['action_rule']
        agent.resources['slot_policies'][node_id] = initial_slot_policy(by_id[node_id], card)
    agent.base_reference = initial_reference_text(profiles)
    agent.base_action_rules = '\n\n'.join(f'## {key}\n{text}' for key, text in agent.resources['action_rules'].items())
    agent.base_slot_policies = '\n\n'.join(f'## {key}\n{text}' for key, text in agent.resources['slot_policies'].items())
    agent.offline_slot_audit = offline_slot_report(profiles, cards, 4)
    if not agent.offline_slot_audit['verified']:
        raise ValueError('Incomplete offline slot coverage')
    agent.cards_ready, agent.stage = True, 'diagnostic_frozen_rebased'
    agent.working_skill, agent.ledger = agent._seed_skill(), []
    for node in agent.graph['nodes']:
        assert agent.action_card(node)['payload_policy'] == agent.resources['slot_policies'][node['id']]
    return {'mode': MODE, 'new_train_turns': 0, 'new_reflection_calls': 0,
            'removed_online_slot_records': old_slots,
            'removed_resolved_slot_policies': sum(s.get('status') == 'resolved' for s in old_slots.values()),
            'retained_resolved_action_rules': sum(s.get('status') == 'resolved' for s in agent.state['action_rules'].values()),
            'retained_reference_notes': len(agent.state['reference_notes']),
            'graph_identical_to_source': digest(agent.graph) == digest(legacy.graph),
            'slot_rule_source': 'New TRAIN-only offline cards; no legacy slot override',
            'reference_source': 'Four diverse TRAIN snippets per node plus frozen learned reference notes'}


def progress(root, phase, completed, total, **extra):
    atomic_json(root / 'progress.json', {'time': datetime.now(SHANGHAI).isoformat(), 'phase': phase,
        'completed': completed, 'total': total, 'workers': 10, **extra})
    print(f'{phase}: {completed}/{total}', flush=True)


def run_phase(root, phase, jobs, function, path_for, workers, *, rounds=2, terminal_predictions=False):
    pending = [job for job in jobs if not path_for(job).exists()]
    started, initial_pending = monotonic(), len(pending)
    progress(root, phase, len(jobs) - len(pending), len(jobs), workers=workers)
    for attempt in range(1, rounds + 1):
        if not pending:
            return
        outcomes = ordered_cases(function, pending, workers=workers, directory=root, phase=phase)
        for outcome, job in zip(outcomes, pending):
            try:
                require_value(outcome)
            except Exception as exc:
                record_error(root, phase, exc, attempt=attempt, output=str(path_for(job)))
            completed = sum(path_for(j).exists() for j in jobs)
            finished = completed - (len(jobs) - initial_pending)
            seconds = monotonic() - started
            eta = seconds / finished * (len(jobs) - completed) if finished else None
            progress(root, phase, completed, len(jobs), round=attempt, workers=workers,
                     elapsed_seconds=round(seconds), eta_seconds=None if eta is None else round(eta))
        pending = [job for job in jobs if not path_for(job).exists()]
    if pending and terminal_predictions:
        for path, signature, dialogue, turn in pending:
            atomic_json(path, {'signature': signature, 'dialogue_id': dialogue['dialogue_id'], 'turn_index': turn,
                'gold': gold_policy(dialogue, turn), 'prediction': {'call': None, 'acts': [], '_execution_error': True},
                'diagnostic': {'status': 'error', 'error': 'Bounded execution retries exhausted; see case_errors.jsonl'}})
    elif pending:
        raise RuntimeError(f'{phase}: {len(pending)} unresolved jobs; successful artifacts are resumable')


def diagnostic_report(root, manifest, records, previous):
    scores = score_policies([r['gold'] for r in records], [r['prediction'] for r in records])
    old = [json.loads(line) for line in previous.read_text(encoding='utf-8').splitlines()]
    if [(r['dialogue_id'], r['turn_index'], r['gold']) for r in old] != [
            (r['dialogue_id'], r['turn_index'], r['gold']) for r in records]:
        raise ValueError('Historical comparison case selection changed')
    old_scores = score_policies([r['gold'] for r in old], [r['prediction'] for r in old])
    shared, old_joint, new_joint = 0, 0, 0
    for before, after in zip(old, records):
        first = score_policies([before['gold']], [before['prediction']])
        second = score_policies([after['gold']], [after['prediction']])
        if first['action_set_exact'] and second['action_set_exact']:
            shared += 1
            old_joint += int(first['turn_joint_ast'])
            new_joint += int(second['turn_joint_ast'])
    result = {**manifest, **scores, 'execution_status': 'completed_with_errors' if any(
        r['prediction'].get('_execution_error') for r in records) else 'completed',
        'historical_ours': old_scores, 'common_action_correct_subset': {
            'turns': shared, 'historical_joint_correct': old_joint, 'diagnostic_joint_correct': new_joint},
        'llm_usage': persist_usage()}
    atomic_json(root / 'summary.json', result)
    lines = ['# MultiWOZ slot/resource diagnostic', '',
        'Frozen legacy trained graph/actions/routing plus new TRAIN-only offline slot cards and four snippets per node. '
        'No new TRAIN rollout or reflection. This is not a fully retrained aligned comparison.', '',
        '| Metric | Historical Ours | New diagnostic |', '|---|---:|---:|']
    for title, key in [('Action exact', 'action_set_exact'), ('Slot exact given action', 'slot_exact_given_action'),
                       ('Joint exact', 'turn_joint_ast')]:
        lines.append(f'| {title} | {old_scores[key]:.2%} | {scores[key]:.2%} |')
    lines += ['', f'On the {shared} turns where both executions choose correct actions: '
              f'historical joint correct = {old_joint}; diagnostic joint correct = {new_joint}.', '',
              'All requested turns remain in the denominator. Source predictions and scores were not overwritten.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def run(args):
    root = args.output_dir.resolve()
    launch_files = {'launch.json', 'run.log', 'code_snapshot', 'code_snapshot.json'}
    if root.exists() and any(p.name not in launch_files for p in root.iterdir()) and not args.resume:
        raise FileExistsError('Use a fresh output directory or --resume')
    root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        failure = root / 'run_failed.json'
        if failure.exists():
            archive = root / 'previous_failures' / f"run_failed_{datetime.now(SHANGHAI).strftime('%Y%m%d_%H%M%S')}.json"
            archive.parent.mkdir(exist_ok=True)
            failure.replace(archive)
        recover_logged_usage(root)
    configure_usage(root)
    original = read(args.source_checkpoint / 'data_manifest.json')
    dataset = Path(original['source'])
    source_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if source_sha != original['source_sha256']:
        raise ValueError('Source dataset changed')
    data = load_hotel_pilot(dataset, seed=original['seed'], fraction=original['fraction'])
    for split in ('train', 'validation'):
        if [d['dialogue_id'] for d in data[split]] != original['selected_ids'][split]:
            raise ValueError('Frozen TRAIN/validation selection changed')
    cases = [(d, t) for d in data['validation'] for t in d['target_indices']['policy_turn_indices']]
    source_hashes = {name: hashlib.sha256((args.source_checkpoint / name).read_bytes()).hexdigest()
        for name in ('data_manifest.json', 'subgraph.json', 'sgd_standard_state.json', 'sgd_skill_state.json')}
    model = original['model']
    manifest = {'protocol': PROTOCOL, 'offline_slot_protocol': OFFLINE_SLOT_PROTOCOL, 'mode': MODE,
        'source_checkpoint': str(args.source_checkpoint.resolve()), 'source_checkpoint_files': source_hashes,
        'source_sha256': source_sha, 'model': model, 'seed': original['seed'], 'fraction': original['fraction'],
        'train_dialogues': len(data['train']), 'train_turns': sum(len(d['target_indices']['policy_turn_indices']) for d in data['train']),
        'validation_turns': len(cases), 'eval_case_ids': [[d['dialogue_id'], t] for d, t in cases],
        'workers': args.workers, 'new_train_turns': 0, 'new_reflection_calls': 0, 'initial_snippets_per_node': 4}
    if (root / 'data_manifest.json').exists() and read(root / 'data_manifest.json') != manifest:
        raise ValueError('Diagnostic resume configuration/source changed')
    atomic_json(root / 'data_manifest.json', manifest)
    frozen = root / 'source_checkpoint'
    frozen.mkdir(exist_ok=True)
    for name in source_hashes:
        if not (frozen / name).exists():
            shutil.copyfile(args.source_checkpoint / name, frozen / name)
        if hashlib.sha256((frozen / name).read_bytes()).hexdigest() != source_hashes[name]:
            raise ValueError('Frozen source snapshot changed')
    ontology = ontology_from_train(data['train'])
    atomic_json(root / 'train_ontology.json', ontology)
    cache = ObservationCache(root / 'observations', dataset.parent, model)
    legacy = MultiWOZStandardSkillAgent(family='hotel', model=model, ontology=ontology, predictor=None)
    legacy.load(frozen)
    for directory in args.observation_cache:
        for path in directory.glob('*.json'):
            saved = read(path)
            if saved.get('executor_protocol') == EXECUTOR_PROTOCOL and path.stem == saved.get('signature'):
                target = cache.directory / path.name
                if not target.exists():
                    atomic_json(target, saved)
    reused = reuse_matched_observations(args.previous_evaluation, cases, cache, source_sha)
    atomic_json(root / 'observation_reuse.json', {'migrated_validation_cases': len(reused),
        'context_and_signature_verified': True, 'db_replay_identical': True,
        'source': str(args.previous_evaluation.resolve())})
    skeleton = training_profiles(data['train'], legacy.graph, ontology)
    selected = {c['sample_id'] for p in skeleton for c in p['train_cases']}
    train_cases = [(d, t) for d in data['train'] for t in d['target_indices']['policy_turn_indices']
                   if f"{d['dialogue_id']}:{t}" in selected]
    jobs = [(cache, d, t) for d, t in train_cases]
    run_phase(root, 'selected_train_db_planning', jobs, observation_job,
              lambda j: cache.directory / f'{cache.identity(j[1], j[2])}.json', args.workers)
    cache.allow_generate = False
    profiles = training_profiles(data['train'], legacy.graph, ontology, observations=cache)
    atomic_json(root / 'offline/train_card_profiles.json', profiles)
    jobs = [(root / 'offline/card_compile' / f"batch_{p['node_id'].split(':')[-1]}.json", [p], model) for p in profiles]
    seeds = [read(path) for directory in args.card_cache for path in directory.glob('batch_*.json')]
    for path, group, _ in jobs:
        signature = digest([PROTOCOL, OFFLINE_SLOT_PROTOCOL, EXECUTOR_PROTOCOL, group, model])
        for saved in seeds:
            if saved.get('signature') == signature and not path.exists():
                atomic_json(path, {**saved, 'diagnostic_cache_reuse': 'Exact profile/model/protocol signature'})
    run_phase(root, 'offline_compile', jobs, compile_cards_job, lambda j: j[0], args.workers)
    cards = {card['node_id']: card for job in jobs for card in compile_cards_job(job)}
    agent = MultiWOZAlignedSkillAgent(family='hotel', model=model, ontology=ontology,
        predictor=None, observations=cache, mining_method='trace_cover')
    migration = build_diagnostic_agent(legacy, agent, profiles, cards)
    atomic_json(root / 'diagnostic_resource_migration.json', migration)
    atomic_json(root / 'offline_slot_audit.json', agent.offline_slot_audit)
    checkpoint = root / 'checkpoint'
    agent.save(checkpoint)
    expected_hash = agent.resource_hash()
    agent.load(checkpoint)
    if agent.resource_hash() != expected_hash:
        raise ValueError('Diagnostic checkpoint save/load changed resources')
    jobs = [(root / 'cases' / f'case_{i:04d}.json', digest([digest(manifest), expected_hash, cache.identity(d, t)]), d, t)
            for i, (d, t) in enumerate(cases)]
    for path, signature, _, _ in jobs:
        if path.exists() and read(path)['signature'] != signature:
            raise ValueError('Cached diagnostic prediction resources changed')
    def evaluate(job):
        path, signature, dialogue, turn = job
        prediction, diagnostic = agent.predict(dialogue, turn)
        atomic_json(path, {'signature': signature, 'dialogue_id': dialogue['dialogue_id'], 'turn_index': turn,
            'gold': gold_policy(dialogue, turn), 'prediction': prediction, 'diagnostic': diagnostic})
    run_phase(root, 'testing', jobs, evaluate, lambda j: j[0], args.workers, terminal_predictions=True)
    records = [read(job[0]) for job in jobs]
    (root / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    recover_logged_usage(root)
    configure_usage(root)
    result = diagnostic_report(root, manifest, records, args.previous_evaluation / 'ours/predictions.jsonl')
    progress(root, result['execution_status'], len(records), len(cases), workers=args.workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path, default=ROOT / 'outputs/multiwoz_hotel_10pct_20261017_222018/ours_retry_11_12')
    parser.add_argument('--previous-evaluation', type=Path, default=ROOT / 'outputs/multiwoz_hotel_matched_20261018_035410')
    parser.add_argument('--observation-cache', type=Path, action='append', default=[])
    parser.add_argument('--card-cache', type=Path, action='append', default=[])
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir / 'run_failed.json', {'error': str(exc), 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()
