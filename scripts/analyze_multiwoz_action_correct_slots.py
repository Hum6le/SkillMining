#!/usr/bin/env python3
"""Audit action-correct payload failures and historical TRAIN supervision.

No model calls, no prediction rewriting and no changes to primary scores.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.sgd_adapter import action_counter, score_policies, slot_signature


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def shape(policy):
    return Counter((a['intent'], a['domain'], a['slot']) for a in policy['acts'])


def action_ok(row):
    return not row['prediction'].get('_execution_error') and action_counter(row['gold']) == action_counter(row['prediction'])


def exact(row):
    return action_ok(row) and slot_signature(row['gold']) == slot_signature(row['prediction'])


def category(row):
    if exact(row):
        return 'exact'
    extra, missing = shape(row['prediction']) - shape(row['gold']), shape(row['gold']) - shape(row['prediction'])
    return 'extra_and_missing' if extra and missing else 'extra_only' if extra else 'missing_only' if missing else 'value_only'


def audit_eval(rows):
    correct = [r for r in rows if action_ok(r)]
    buckets = Counter(category(r) for r in correct)
    groups = defaultdict(list)
    for row in correct:
        groups[tuple(sorted(action_counter(row['gold'])))].append(row)
    substantive = [r for r in correct if any(a['intent'] not in {'bye', 'welcome'} for a in r['gold']['acts'])]
    return {'primary': score_policies([r['gold'] for r in rows], [r['prediction'] for r in rows]),
        'action_correct_buckets': dict(buckets),
        'slot_shape_exact': sum(shape(r['gold']) == shape(r['prediction']) for r in correct),
        'slot_name_set_exact': sum(set(shape(r['gold'])) == set(shape(r['prediction'])) for r in correct),
        'action_correct_with_extra_slots': buckets['extra_only'] + buckets['extra_and_missing'],
        'action_correct_with_missing_slots': buckets['missing_only'] + buckets['extra_and_missing'],
        'non_farewell_action_correct': len(substantive), 'non_farewell_joint_correct': sum(exact(r) for r in substantive),
        'by_heads': [{'heads': list(key), 'action_correct': len(data), 'joint_correct': sum(exact(r) for r in data)}
                     for key, data in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))]}


def main(args):
    root, old, output = args.diagnostic.resolve(), args.original.resolve(), args.output.resolve()
    manifest = read(root / 'data_manifest.json')
    source = Path(manifest['source_checkpoint'])
    inputs = [root / 'predictions.jsonl', old / 'ours/predictions.jsonl', source / 'ours_training_turns.jsonl',
              source / 'sgd_standard_state.json']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    current, original = lines(inputs[0]), lines(inputs[1])
    dataset = Path(read(source / 'data_manifest.json')['source'])
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == manifest['source_sha256']
    raw = {d['dialogue_id']: d for d in read(dataset)}
    result = {'analysis_only': True, 'model_calls': 0, 'source_diagnostic': str(root), 'source_original': str(old),
        'diagnostic': audit_eval(current), 'original': audit_eval(original), 'cases': []}
    for i, row in enumerate(current):
        if not action_ok(row) or exact(row):
            continue
        diagnostic = row['diagnostic']
        context = diagnostic['visible_context_for_diagnostics']
        obs = diagnostic['shared_observation']['observations']
        visible = (context + json.dumps(obs, ensure_ascii=False)).casefold()
        unavailable_refs = [a for a in row['gold']['acts'] if a['slot'] == 'ref' and a.get('value')
                            and a['value'].casefold() not in visible]
        extra, missing = shape(row['prediction']) - shape(row['gold']), shape(row['gold']) - shape(row['prediction'])
        cards = diagnostic['selected_action_cards']
        result['cases'].append({'case_index': i, 'sample_id': f"{row['dialogue_id']}:{row['turn_index']}",
            'category': category(row), 'heads': diagnostic['selected_action_heads'],
            'extra_slot_occurrences': list(extra.elements()), 'missing_slot_occurrences': list(missing.elements()),
            'gold': row['gold'], 'prediction': row['prediction'], 'context': context,
            'raw_target_utterance_ANALYSIS_ONLY': raw[row['dialogue_id']]['turns'][row['turn_index']]['utterance'],
            'original_dialogue_id': raw[row['dialogue_id']]['original_id'],
            'unobservable_gold_refs': unavailable_refs, 'selected_cards': cards,
            'retrieved_reference': diagnostic['retrieved_reference'], 'shared_db_observation': obs,
            'gold_train_ids_in_visible_records': {a['value']: any(a['value'].casefold() in json.dumps(o['records']).casefold() for o in obs)
                for a in row['gold']['acts'] if a['slot'] == 'train id' and a.get('value')},
            'all_exact_cards_found': bool(cards)})
    errors = result['cases']
    result['diagnostic'].update(unobservable_ref_error_turns=sum(bool(c['unobservable_gold_refs']) for c in errors),
        non_ref_error_buckets=dict(Counter(c['category'] for c in errors if not c['unobservable_gold_refs'])),
        all_action_correct_errors_read_exact_cards=all(c['all_exact_cards_found'] for c in errors),
        value_only_errors=[{'sample_id': c['sample_id'], 'gold': c['gold'], 'prediction': c['prediction']}
                           for c in errors if c['category'] == 'value_only'])
    train_rows = [r['sgd_evidence'] for r in lines(source / 'ours_training_turns.jsonl')]
    counts, false_ids, examples = Counter(), set(), []
    for evidence in train_rows:
        if not evidence['action_correct']:
            continue
        counts['action_correct'] += 1
        gold, prediction = evidence['gold_policy'], evidence['predicted_policy']
        normalized = deepcopy(prediction)
        for act in normalized['acts']:
            if act.get('value') == '':
                act.pop('value')
        before = slot_signature(gold) == slot_signature(prediction)
        assert bool(evidence['slot_exact_correct']) == before
        after = slot_signature(gold) == slot_signature(normalized)
        counts['strict_slot_exact'] += before
        counts['analysis_only_empty_omission_exact'] += after
        if not before and after:
            false_ids.add(evidence['sample_id'])
            if len(examples) < 4:
                examples.append({k: evidence[k] for k in ('sample_id', 'gold_policy', 'predicted_policy', 'slot_exact_correct')})
    reports_with_false, all_false_reports, proposals = [], [], Counter()
    for path in source.glob('diagnosis_*.json'):
        report = read(path)
        sample_ids = set(report.get('sample_ids', []))
        if sample_ids & false_ids:
            reports_with_false.append(path.name)
        if sample_ids and sample_ids <= false_ids:
            all_false_reports.append(path.name)
            proposals.update(u.get('resource') for u in report.get('candidate_updates', []))
    state = read(source / 'sgd_standard_state.json')['state']
    policies = [{'action': key, 'policy': value['policy']} for key, value in state['slot_policies'].items()
                if value.get('status') == 'resolved']
    result['historical_training'] = {'rollout_turns': len(train_rows), **dict(counts),
        'logged_action_correct_slot_wrong': counts['action_correct'] - counts['strict_slot_exact'],
        'format_only_false_slot_errors': len(false_ids), 'false_error_sample_ids': sorted(false_ids),
        'examples': examples, 'diagnosis_reports_containing_false_errors': reports_with_false,
        'diagnosis_reports_all_false_errors': all_false_reports,
        'all_false_reports_candidate_update_resources': dict(proposals),
        'slot_policy_status_counts': dict(Counter(v.get('status') for v in state['slot_policies'].values())),
        'action_rule_status_counts': dict(Counter(v.get('status') for v in state['action_rules'].values())),
        'resolved_slot_policies': policies,
        'scope': 'Historical source training only; old outputs/labels untouched; current common binder omits empty values.'}
    assert len(errors) == result['diagnostic']['primary']['counts']['action_exact'] - result['diagnostic']['primary']['counts']['joint_exact']
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in inputs)
    result.update(source_files_unchanged=True, source_sha256=hashes)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    brief = {k: v for k, v in result.items() if k not in {'cases', 'historical_training', 'source_sha256'}}
    brief['historical_training'] = {k: v for k, v in result['historical_training'].items()
                                   if k not in {'false_error_sample_ids', 'examples', 'resolved_slot_policies',
                                                'diagnosis_reports_containing_false_errors', 'diagnosis_reports_all_false_errors'}}
    (output / 'summary.json').write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding='utf-8')
    for case in errors:
        (output / f"case_{case['case_index']:04d}.json").write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, default=ROOT / 'outputs/multiwoz_slot_diagnostic_20261018_113434')
    parser.add_argument('--original', type=Path, default=ROOT / 'outputs/multiwoz_hotel_matched_20261018_035410')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/multiwoz_action_correct_slot_audit_20261018')
    main(parser.parse_args())
