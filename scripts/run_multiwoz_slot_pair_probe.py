#!/usr/bin/env python3
"""Paired TRAIN-only resource ablations with identical frozen predicted heads.

The 64 probes are drawn from old action-correct TRAIN predictions, exclude
the previous four initialization snippets, and are diagnostic, not held-out
performance estimates. No validation/test gold is used for rule mining.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_tod.multiwoz_aligned import (
    AgentReferenceSession, MultiWOZAlignedSkillAgent, ObservationCache, atomic_json,
    compile_conditional_cards_job, initial_reference_text, reference_sections,
)
from eval_tod.multiwoz_slot_policy import (
    PROTOCOL, ScopedReferenceSession, contrast_training_profiles, conditional_slot_policy, payload_card,
)
from eval_tod.multiwoz_matched_executor import bind_payload, digest
from eval_tod.multiwoz_policy import load_hotel_pilot, ontology_from_train
from eval_tod.sgd_adapter import gold_policy, score_policies, visible_context
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_skill_agent import policy_heads, policy_token
from eval_tod.sgd_usage import configure_usage, persist_usage
from scripts.run_multiwoz_matched_eval import recover_logged_usage

ARMS = ('A_original', 'B_payload_only', 'C_joint_patterns', 'D_scoped_reference')
SHANGHAI = timezone(timedelta(hours=8))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def select_probe_cases(training_rows, profiles, *, count=64, seed=42):
    initial = {c['sample_id'] for p in profiles for c in p['train_cases']}
    by_label = {p['node_id'].split(':')[-1]: p for p in profiles}
    groups = defaultdict(list)
    for row in training_rows:
        evidence = row['sgd_evidence']
        label = policy_token(evidence['predicted_policy'])
        profile = by_label.get(label)
        if (evidence['action_correct'] and evidence['sample_id'] not in initial and profile
                and len(profile['slot_schema']) >= 2):
            groups[label].append(evidence)
    # Seven recurrent, nontrivial nodes; equal-round selection prevents easy
    # goodbye turns or a single frequent head from dominating the diagnosis.
    labels = sorted(groups, key=lambda key: (-len(groups[key]), key))[:7]
    if len(labels) != 7 or sum(len(groups[label]) for label in labels) < count:
        raise ValueError('Insufficient recurrent TRAIN-only probe candidates')
    for label in labels:
        groups[label].sort(key=lambda r: digest([seed, r['sample_id']]))
    selected = []
    while len(selected) < count:
        for label in labels:
            if groups[label] and len(selected) < count:
                selected.append(groups[label].pop(0))
    return selected


def progress(root, phase, completed, total, workers):
    atomic_json(root / 'progress.json', {'time': datetime.now(SHANGHAI).isoformat(), 'phase': phase,
        'completed': completed, 'total': total, 'workers': workers})
    print(f'{phase}: {completed}/{total}', flush=True)


def phase(root, name, jobs, function, path_for, workers, attempts=2):
    pending = [job for job in jobs if not path_for(job).exists()]
    progress(root, name, len(jobs) - len(pending), len(jobs), workers)
    for attempt in range(1, attempts + 1):
        if not pending:
            return
        for outcome, job in zip(ordered_cases(function, pending, workers=workers, directory=root, phase=name), pending):
            try:
                require_value(outcome)
            except Exception as exc:
                with (root / 'retry_history.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'phase': name, 'round': attempt, 'output': str(path_for(job)), 'error': str(exc)}) + '\n')
            progress(root, name, sum(path_for(j).exists() for j in jobs), len(jobs), workers)
        pending = [job for job in jobs if not path_for(job).exists()]
    if pending:
        raise RuntimeError(f'{name}: {len(pending)} jobs remain; rerun --resume to retain successes')


def slot_keys(policy):
    return Counter((a['intent'], a['domain'], a['slot']) for a in policy['acts'])


def score_arm(records):
    scores = score_policies([r['gold'] for r in records], [r['prediction'] for r in records])
    counts = Counter()
    for row in records:
        gold, pred = slot_keys(row['gold']), slot_keys(row['prediction'])
        extra, missing = pred - gold, gold - pred
        counts['slot_key_exact'] += gold == pred
        counts['extra_slot_acts'] += sum(extra.values())
        counts['missing_slot_acts'] += sum(missing.values())
        counts['slot_key_tp'] += sum((gold & pred).values())
        counts['gold_slot_acts'] += sum(gold.values())
        counts['pred_slot_acts'] += sum(pred.values())
        if gold == pred and row['gold'] != row['prediction']:
            counts['matching_slot_keys_value_errors'] += 1
    denominator = counts['gold_slot_acts'] + counts['pred_slot_acts']
    scores['slot_key_metrics'] = {**dict(counts), 'exact': counts['slot_key_exact'] / len(records),
        'micro_f1': 2 * counts['slot_key_tp'] / denominator if denominator else 0}
    return scores


def compare_records(before, after):
    result = Counter()
    cases = []
    for a, b in zip(before, after):
        assert (a['sample_id'], a['gold'], a['fixed_heads'], a['observation_sha256']) == (
            b['sample_id'], b['gold'], b['fixed_heads'], b['observation_sha256'])
        old = score_policies([a['gold']], [a['prediction']])['turn_joint_ast']
        new = score_policies([b['gold']], [b['prediction']])['turn_joint_ast']
        category = 'both_correct' if old and new else 'improved' if new else 'regressed' if old else 'both_wrong'
        result[category] += 1
        if category in ('improved', 'regressed'):
            cases.append({'sample_id': a['sample_id'], 'change': category,
                'gold': a['gold'], 'before': a['prediction'], 'after': b['prediction']})
    return {'counts': dict(result), 'changed_cases': cases}


def run(args):
    root, source = args.output_dir.resolve(), args.source_diagnostic.resolve()
    launch_files = {'launch.json', 'run.log', 'code_snapshot', 'code_snapshot.json'}
    if root.exists() and any(p.name not in launch_files for p in root.iterdir()) and not args.resume:
        raise FileExistsError('Use a fresh probe directory or --resume')
    root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if (root / 'run_failed.json').exists():
            (root / 'run_failed.json').replace(root / 'previous_run_failed.json')
        recover_logged_usage(root)
    configure_usage(root)
    previous = read(source / 'data_manifest.json')
    source_manifest = read(Path(previous['source_checkpoint']) / 'data_manifest.json')
    dataset = Path(source_manifest['source'])
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != previous['source_sha256']:
        raise ValueError('TRAIN source dataset changed')
    train = load_hotel_pilot(dataset, seed=previous['seed'], fraction=previous['fraction'])['train']
    ontology = ontology_from_train(train)
    original_profiles = read(source / 'offline/train_card_profiles.json')
    training_rows = rows(Path(previous['source_checkpoint']) / 'ours_training_turns.jsonl')
    selected = select_probe_cases(training_rows, original_profiles)
    by_id = {d['dialogue_id']: d for d in train}
    probes = []
    for evidence in selected:
        dialogue, turn = by_id[evidence['conversation_id']], evidence['turn_index']
        heads = policy_heads(evidence['predicted_policy'])
        # TRAIN labels verify the diagnostic stratum; inference gets only the
        # frozen old prediction, prior utterances and context-planned DB.
        assert heads == policy_heads(gold_policy(dialogue, turn))
        probes.append((evidence['sample_id'], dialogue, turn, heads))
    cache = ObservationCache(root / 'observations', dataset.parent, previous['model'])
    legacy = MultiWOZAlignedSkillAgent(family='hotel', model=previous['model'], ontology=ontology,
        predictor=None, observations=cache, mining_method='trace_cover')
    legacy.load(source / 'checkpoint')
    labels = {policy_token(e['predicted_policy']) for e in selected}
    graph = {'nodes': [n for n in legacy.graph['nodes'] if n['label'] in labels]}
    profiles = contrast_training_profiles(train, graph, ontology, exclude_sample_ids=[p[0] for p in probes])
    manifest = {'protocol': PROTOCOL, 'scope': 'TRAIN-only selected action-correct paired resource diagnosis, not held-out performance',
        'source_diagnostic': str(source), 'source_sha256': previous['source_sha256'], 'model': previous['model'],
        'workers': args.workers, 'sample_ids': [p[0] for p in probes], 'probe_count': len(probes),
        'nodes': [n['id'] for n in graph['nodes']], 'initial_snippets_per_node': 4, 'arms': list(ARMS),
        'fixed_heads_source': 'Previously saved TRAIN predictions; action-correct stratum selected using TRAIN labels',
        'source_checkpoint_sha256': digest(read(source / 'checkpoint/sgd_standard_state.json'))}
    if (root / 'data_manifest.json').exists() and read(root / 'data_manifest.json') != manifest:
        raise ValueError('Probe resume inputs changed')
    atomic_json(root / 'data_manifest.json', manifest)
    atomic_json(root / 'probe_cases.json', [{'sample_id': key, 'dialogue_id': d['dialogue_id'], 'turn': t, 'fixed_heads': h}
                                         for key, d, t, h in probes])
    snippet_ids = {c['sample_id'] for p in profiles for c in p['train_cases']}
    all_cases = {(d['dialogue_id'], t): (d, t) for _, d, t, _ in probes}
    for d in train:
        for t in d['target_indices']['policy_turn_indices']:
            if f"{d['dialogue_id']}:{t}" in snippet_ids:
                all_cases[(d['dialogue_id'], t)] = (d, t)
    for d, t in all_cases.values():
        path = source / 'observations' / f'{cache.identity(d, t)}.json'
        if path.exists() and not (cache.directory / path.name).exists():
            atomic_json(cache.directory / path.name, read(path))
    jobs = list(all_cases.values())
    def observe(job):
        cache.get(*job)
    phase(root, 'shared_db', jobs, observe, lambda j: cache.directory / f'{cache.identity(*j)}.json', args.workers)
    cache.allow_generate = False
    profiles = contrast_training_profiles(train, graph, ontology, observations=cache, exclude_sample_ids=[p[0] for p in probes])
    atomic_json(root / 'train_card_profiles.json', profiles)
    jobs = [(root / 'card_compile' / f"batch_{p['node_id'].split(':')[-1]}.json", [p], previous['model']) for p in profiles]
    phase(root, 'conditional_compile', jobs, compile_conditional_cards_job, lambda j: j[0], args.workers)
    compiled = {c['node_id']: c for job in jobs for c in compile_conditional_cards_job(job)}
    old_cards = {n['id']: legacy.action_card(n) for n in graph['nodes']}
    new_cards = {n['id']: {**old_cards[n['id']], 'rule': compiled[n['id']]['action_rule'],
        'payload_policy': conditional_slot_policy(next(p for p in profiles if p['node_id'] == n['id']), compiled[n['id']])}
        for n in graph['nodes']}
    atomic_json(root / 'resource_bundle.json', {'old_cards': old_cards, 'new_cards': new_cards})
    old_text = legacy.reference_text()
    kept = [u['content'] for u in reference_sections(old_text).values()
            if not any(f"TRAIN snippet {node_id} /" in u['title'] for node_id in new_cards)]
    new_text = '\n\n'.join(kept) + '\n\n' + initial_reference_text(profiles)
    (root / 'conditional_reference.md').write_text(new_text, encoding='utf-8')
    node_by_heads = {tuple(sorted(tuple(h) for h in n['action_heads'])): n['id'] for n in graph['nodes']}
    def node_id(heads):
        return node_by_heads[tuple(sorted(tuple(h) for h in heads))]
    jobs = list(enumerate(probes))
    def prepare_reference(job):
        index, (key, dialogue, turn, heads) = job
        obs = cache.get(dialogue, turn)
        context = visible_context(dialogue, turn)[-12000:]
        cid = node_id(heads)
        # Freeze the legacy retrieved context once for A/B/C. D changes only
        # the resource index/reference lookup in addition to C's same card.
        baseline = AgentReferenceSession(old_text, legacy.limits)
        baseline.collect(model=legacy.model, stage='payload', knowledge=json.dumps([old_cards[cid]], ensure_ascii=False),
                         context=context, observation=obs)
        scoped = ScopedReferenceSession(new_text, legacy.limits, nodes=legacy.graph['nodes'], scope_node_ids=[cid])
        scoped.collect(model=legacy.model, stage='payload', knowledge=json.dumps([payload_card(new_cards[cid])], ensure_ascii=False),
                       context=context, observation=obs)
        atomic_json(root / 'references' / f'case_{index:04d}.json', {'sample_id': key,
            'observation_sha256': digest(obs), 'legacy_units': baseline.units, 'legacy_diagnostic': baseline.diagnostic(),
            'scoped_units': scoped.units, 'scoped_diagnostic': scoped.diagnostic()})
    phase(root, 'reference_preparation', jobs, prepare_reference,
          lambda j: root / 'references' / f'case_{j[0]:04d}.json', args.workers)
    def validate_planner(job):
        index, (key, dialogue, turn, heads) = job
        obs = cache.get(dialogue, turn)
        session = ScopedReferenceSession(old_text, legacy.limits, nodes=legacy.graph['nodes'])
        session.collect(model=legacy.model, stage='action', knowledge=legacy.runtime_skill_text(),
                        context=visible_context(dialogue, turn)[-12000:], observation=obs, plan_candidates=True)
        resolved = session.plan['candidate_actions']
        atomic_json(root / 'candidate_plans' / f'case_{index:04d}.json', {'sample_id': key,
            'plan': session.plan, 'diagnostic': session.diagnostic(),
            'all_identifiers_resolve': all(cid in {n['id'] for n in legacy.graph['nodes']} for cid in resolved),
            'frozen_action_in_candidates': node_id(heads) in resolved})
    phase(root, 'candidate_transport', jobs, validate_planner,
          lambda j: root / 'candidate_plans' / f'case_{j[0]:04d}.json', args.workers)
    ablation_jobs = [(arm, index, probe) for arm in ARMS for index, probe in enumerate(probes)]
    def predict(job):
        arm, index, (key, dialogue, turn, heads) = job
        obs = cache.get(dialogue, turn)
        cid = node_id(heads)
        reference = read(root / 'references' / f'case_{index:04d}.json')
        card = old_cards[cid] if arm in ARMS[:2] else new_cards[cid]
        card = card if arm == ARMS[0] else payload_card(card)
        units = reference['scoped_units'] if arm == ARMS[3] else reference['legacy_units']
        knowledge = 'Selected payload resources with frozen action heads:\n' + json.dumps([card], ensure_ascii=False)
        knowledge += '\nFrozen agent-selected reference:\n' + json.dumps(units, ensure_ascii=False)
        prediction, diagnostic = bind_payload(dialogue, turn, heads=heads, observation=obs,
            knowledge=knowledge, ontology=ontology, model=legacy.model)
        atomic_json(root / arm / 'cases' / f'case_{index:04d}.json', {'sample_id': key, 'arm': arm,
            'fixed_heads': heads, 'observation_sha256': digest(obs), 'reference_sha256': digest(units),
            'payload_card_sha256': digest(card), 'action_rule_in_payload': arm == ARMS[0],
            'gold': gold_policy(dialogue, turn), 'prediction': prediction, 'diagnostic': diagnostic})
    phase(root, 'paired_payload', ablation_jobs, predict,
          lambda j: root / j[0] / 'cases' / f'case_{j[1]:04d}.json', args.workers)
    recover_logged_usage(root)
    configure_usage(root)
    records = {arm: [read(root / arm / 'cases' / f'case_{i:04d}.json') for i in range(len(probes))] for arm in ARMS}
    metrics = {arm: score_arm(data) for arm, data in records.items()}
    comparisons = {f'{left}_to_{right}': compare_records(records[left], records[right])
                   for left, right in zip(ARMS, ARMS[1:])}
    plans = [read(root / 'candidate_plans' / f'case_{i:04d}.json') for i in range(len(probes))]
    assert all(p['all_identifiers_resolve'] for p in plans)
    assert all(records['A_original'][i]['reference_sha256'] == records['B_payload_only'][i]['reference_sha256']
               == records['C_joint_patterns'][i]['reference_sha256'] for i in range(len(probes)))
    summary = {**manifest, 'metrics': metrics, 'paired_comparisons': comparisons,
        'candidate_transport': {'cases': len(plans), 'all_identifiers_resolve': True,
            'nonempty_structured_submissions': sum(bool(p['plan']['candidate_actions'] or p['plan'].get('unseen_candidate_heads')) for p in plans),
            'nonempty_candidates': sum(bool(p['plan']['candidate_actions']) for p in plans),
            'frozen_action_in_candidates': sum(p['frozen_action_in_candidates'] for p in plans)},
        'evaluation_errors': 0, 'llm_usage': persist_usage()}
    atomic_json(root / 'summary.json', summary)
    lines = ['# TRAIN-only paired slot/resource probe', '', manifest['scope'], '',
        'All arms share the same 64 frozen predicted head bundles and DB observations. '
        'A/B/C share identical frozen reference content. No validation/test gold used to mine policies.', '',
        '| Arm | Joint/slot-value exact | Slot-key exact | Slot-key F1 | Extra acts | Missing acts |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        score, slots = metrics[arm], metrics[arm]['slot_key_metrics']
        lines.append(f"| {arm} | {score['counts']['joint_exact']}/64 ({score['turn_joint_ast']:.2%}) | "
            f"{slots['slot_key_exact']}/64 | {slots['micro_f1']:.2%} | {slots['extra_slot_acts']} | {slots['missing_slot_acts']} |")
    lines += ['', 'A: old action+slot prose. B: remove action rule from payload. '
        'C: conditional joint-slot patterns with contrastive four-snippet mining. '
        'D: C plus current-action reference index and agent-controlled scope expansion.', '',
        'This stratum is selected using TRAIN action correctness and is not an unbiased or held-out accuracy estimate.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    progress(root, 'completed', len(ablation_jobs), len(ablation_jobs), args.workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-diagnostic', type=Path, default=ROOT / 'outputs/multiwoz_slot_diagnostic_20261018_113434')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir / 'run_failed.json', {'error': str(exc), 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()
