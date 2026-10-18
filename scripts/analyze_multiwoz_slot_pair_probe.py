#!/usr/bin/env python3
"""Read-only independent input audit and error analysis for a finished probe."""
from collections import Counter, defaultdict
import argparse
import json
from math import comb
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_tod.multiwoz_aligned import slot_rule_coverage
from eval_tod.multiwoz_matched_executor import digest
from eval_tod.multiwoz_slot_policy import payload_card
from eval_tod.sgd_adapter import action_counter, slot_signature

ARMS = ('A_original', 'B_payload_only', 'C_joint_patterns', 'D_scoped_reference')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def exact(row):
    return (not row['prediction'].get('_execution_error')
        and action_counter(row['gold']) == action_counter(row['prediction'])
        and slot_signature(row['gold']) == slot_signature(row['prediction']))


def keys(policy):
    return Counter((a['intent'], a['domain'], a['slot']) for a in policy['acts'])


def errors(records):
    result = Counter()
    for row in records:
        extra = keys(row['prediction']) - keys(row['gold'])
        missing = keys(row['gold']) - keys(row['prediction'])
        bucket = ('exact' if exact(row) else 'extra_and_missing' if extra and missing
            else 'extra_only' if extra else 'missing_only' if missing else 'value_only')
        result[bucket] += 1
    return {key: result[key] for key in ('exact', 'extra_only', 'missing_only', 'extra_and_missing', 'value_only')}


def paired(left, right):
    counts = Counter()
    for a, b in zip(left, right):
        before, after = exact(a), exact(b)
        counts['both_correct' if before and after else 'improved' if after else 'regressed' if before else 'both_wrong'] += 1
    discordant = counts['improved'] + counts['regressed']
    counts['discordant'] = discordant
    tail = min(counts['improved'], counts['regressed'])
    return {**dict(counts), 'exact_two_sided_sign_p': min(1.0,
        2 * sum(comb(discordant, i) for i in range(tail + 1)) / 2 ** discordant) if discordant else 1.0}


def analyze(root):
    manifest, summary = read(root / 'data_manifest.json'), read(root / 'summary.json')
    sample_ids = manifest['sample_ids']
    n = len(sample_ids)
    assert n == 64 and len(set(sample_ids)) == n
    records = {arm: [read(root / arm / 'cases' / f'case_{i:04d}.json') for i in range(n)] for arm in ARMS}
    references = [read(root / 'references' / f'case_{i:04d}.json') for i in range(n)]
    cards = read(root / 'resource_bundle.json')
    profiles = read(root / 'train_card_profiles.json')
    profile_by_id = {p['node_id']: p for p in profiles}
    card_id_by_heads = {tuple(sorted(tuple(h) for h in c['heads'])): cid for cid, c in cards['old_cards'].items()}
    for i, sample_id in enumerate(sample_ids):
        base = records[ARMS[0]][i]
        assert base['sample_id'] == sample_id == references[i]['sample_id']
        assert all((row[i]['sample_id'], row[i]['fixed_heads'], row[i]['gold'], row[i]['observation_sha256'])
            == (sample_id, base['fixed_heads'], base['gold'], base['observation_sha256']) for row in records.values())
        assert all(action_counter(row[i]['gold']) == action_counter(row[i]['prediction']) for row in records.values())
        assert all(records[arm][i]['reference_sha256'] == digest(references[i]['legacy_units']) for arm in ARMS[:3])
        assert records[ARMS[3]][i]['reference_sha256'] == digest(references[i]['scoped_units'])
        cid = card_id_by_heads[tuple(sorted(tuple(h) for h in base['fixed_heads']))]
        for arm in ARMS:
            card = cards['old_cards'][cid] if arm in ARMS[:2] else cards['new_cards'][cid]
            projected = card if arm == ARMS[0] else payload_card(card)
            assert records[arm][i]['payload_card_sha256'] == digest(projected)
            assert records[arm][i]['action_rule_in_payload'] == (arm == ARMS[0])
        assert records[ARMS[2]][i]['payload_card_sha256'] == records[ARMS[3]][i]['payload_card_sha256']
    for path in (root / 'card_compile').glob('batch_*.json'):
        for card in read(path)['cards']:
            assert slot_rule_coverage(profile_by_id[card['node_id']], card)['verified']
    assert len(profiles) == 7 and all(len(p['train_cases']) == 4 for p in profiles)
    assert not set(sample_ids) & {c['sample_id'] for p in profiles for c in p['train_cases']}
    source = Path(manifest['source_diagnostic']) / 'checkpoint/sgd_standard_state.json'
    assert digest(read(source)) == manifest['source_checkpoint_sha256']
    for arm, data in records.items():
        assert sum(exact(row) for row in data) == summary['metrics'][arm]['counts']['joint_exact']
    plans = [read(root / 'candidate_plans' / f'case_{i:04d}.json') for i in range(n)]
    assert all(p['plan']['candidate_actions'] or p['plan'].get('unseen_candidate_heads') for p in plans)
    all_ids = {node['id'] for node in read(Path(manifest['source_diagnostic']) / 'checkpoint/subgraph.json')['nodes']}
    assert all(cid in all_ids for p in plans for cid in p['plan']['candidate_actions'])
    grouped = defaultdict(lambda: {arm: [] for arm in ARMS})
    for i, row in enumerate(records[ARMS[0]]):
        head = json.dumps(row['fixed_heads'], ensure_ascii=False)
        for arm in ARMS:
            grouped[head][arm].append(records[arm][i])
    per_node = {node: {'n': len(arms[ARMS[0]]),
        'exact': {arm: sum(exact(row) for row in data) for arm, data in arms.items()}}
        for node, arms in grouped.items()}
    same_inputs = [i for i in range(n) if records[ARMS[2]][i]['reference_sha256'] == records[ARMS[3]][i]['reference_sha256']]
    different_outputs = [sample_ids[i] for i in same_inputs
        if slot_signature(records[ARMS[2]][i]['prediction']) != slot_signature(records[ARMS[3]][i]['prediction'])]
    changed_inputs = [i for i in range(n) if i not in same_inputs]
    reference_stats = {}
    for side in ('legacy', 'scoped'):
        reference_stats[side] = {'turns_with_reference': sum(bool(r[f'{side}_units']) for r in references),
            'retained_units': sum(len(r[f'{side}_units']) for r in references),
            'tool_calls': sum(len(r[f'{side}_diagnostic']['calls']) for r in references)}
    previous_manifest = read(Path(manifest['source_diagnostic']) / 'data_manifest.json')
    training_file = Path(previous_manifest['source_checkpoint']) / 'ours_training_turns.jsonl'
    contexts = {}
    with training_file.open(encoding='utf-8') as stream:
        for line in stream:
            evidence = json.loads(line)['sgd_evidence']
            if evidence['sample_id'] in sample_ids:
                contexts[evidence['sample_id']] = evidence['context'][-12000:]
    observations = {}
    for path in (root / 'observations').glob('*.json'):
        obs = read(path)['observation']
        observations[digest(obs)] = obs
    unobservable_refs = []
    for i, row in enumerate(records[ARMS[0]]):
        visible = (contexts[row['sample_id']] + json.dumps(observations[row['observation_sha256']], ensure_ascii=False)).casefold()
        if any(a['slot'] == 'ref' and a.get('value') and a['value'].casefold() not in visible for a in row['gold']['acts']):
            unobservable_refs.append(i)
    pairs = {f'{a}_to_{b}': paired(records[a], records[b]) for a, b in zip(ARMS, ARMS[1:])}
    pairs['A_original_to_D_scoped_reference'] = paired(records[ARMS[0]], records[ARMS[3]])
    analysis = {'audit_passed': True, 'sample_count': n, 'arms': list(ARMS),
        'controlled_inputs': ['sample_id', 'gold', 'prior_context', 'frozen_predicted_heads', 'DB observation'],
        'frozen_legacy_reference_identical_for_ABC': True, 'conditional_card_identical_for_CD': True,
        'source_checkpoint_hash_unchanged': True, 'all_structured_candidate_ids_resolve': True,
        'all_seven_cards_train_shape_coverage_verified': True, 'four_initial_snippets_each': True,
        'probe_samples_excluded_from_new_snippets': True,
        'error_buckets': {arm: errors(data) for arm, data in records.items()},
        'paired': pairs, 'per_node': per_node, 'reference': reference_stats,
        'unobservable_gold_ref_turns': [sample_ids[i] for i in unobservable_refs],
        'analysis_only_non_unobservable_ref_subset': {'n': n - len(unobservable_refs),
            'exact': {arm: sum(exact(row) for i, row in enumerate(data) if i not in unobservable_refs)
                      for arm, data in records.items()}},
        'CD_equal_generation_inputs': len(same_inputs), 'CD_equal_inputs_different_predictions': different_outputs,
        'CD_equal_inputs_paired': paired([records[ARMS[2]][i] for i in same_inputs],
                                       [records[ARMS[3]][i] for i in same_inputs]),
        'CD_changed_inputs_paired': paired([records[ARMS[2]][i] for i in changed_inputs],
                                         [records[ARMS[3]][i] for i in changed_inputs]),
        'payload_policy_chars_by_node': {cid: {'old': len(card['payload_policy']),
            'new': len(cards['new_cards'][cid]['payload_policy'])} for cid, card in cards['old_cards'].items()},
        'scope': manifest['scope']}
    (root / 'independent_audit.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    analyze(parser.parse_args().root.resolve())
