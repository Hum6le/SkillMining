"""Audit saved SGD slot-learning evidence without model calls or rescoring edits."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.sgd_adapter import canonical_acts, score_policies, slot_signature
from eval_tod.sgd_skill_agent import policy_token


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def without_values(acts):
    return canonical_acts([{k: v for k, v in act.items() if k != 'value'} for act in acts])


def main():
    training = ROOT/'outputs/sgd_homes_10pct_20261018_153955'
    checkpoint = ROOT/'outputs/sgd_homes_policy_recheck_20261018_191306/repaired_resources'
    evaluation = ROOT/'outputs/sgd_aligned_failure_recovery_20261018_204325'
    output = ROOT/'outputs/sgd_slot_learning_audit_20261018'
    output.mkdir(exist_ok=True)
    predictions = {method: rows(evaluation/method/'predictions.jsonl')
                   for method in ['awm', 'trace2skill', 'ours']}
    evidence = [r['sgd_evidence'] for r in rows(training/'ours/ours_training_turns.jsonl')]
    state = read(checkpoint/'sgd_standard_state.json')['state']
    original_resources = read(training/'ours/sgd_skill_state.json')['resources']
    ledger = read(training/'ours/refinement_ledger.json')
    offline = [r for r in ledger if r.get('phase') == 'offline_compile'
               and r.get('update', {}).get('resource') == 'slot_policies']
    node_metrics = {}
    paired_metrics = {}
    examples = []
    failure_counts = Counter()
    for method, records in predictions.items():
        grouped = defaultdict(Counter)
        for line_number, row in enumerate(records, 1):
            score = score_policies([row['gold']], [row['prediction']])
            node = policy_token(row['gold'])
            count = grouped[node]
            count['total'] += 1
            count['action_correct'] += int(bool(score['action_set_exact']))
            count['payload_correct'] += int(bool(score['turn_joint_ast']))
            if method != 'ours' or not score['action_set_exact'] or score['turn_joint_ast']:
                continue
            gold, pred = row['gold'], row['prediction']
            same_names = (gold['call'] == pred['call']
                          and without_values(gold['acts']) == without_values(pred['acts']))
            presence_only = same_names and all('value' not in a for a in gold['acts'])
            empty_removed = {'call': pred['call'], 'acts': canonical_acts([
                {k: v for k, v in a.items() if k != 'value' or v != ''} for a in pred['acts']])}
            failure_counts['action_correct_payload_wrong'] += 1
            failure_counts['only_value_presence'] += int(presence_only)
            failure_counts['only_extra_empty_value'] += int(empty_removed == gold)
            failure_counts['same_names_values_or_presence_only'] += int(same_names)
            examples.append({'dialogue_id': row['dialogue_id'], 'turn_index': row['turn_index'],
                'prediction_file_line': line_number, 'node': node, 'gold': gold, 'prediction': pred,
                'only_value_presence': presence_only,
                'selected_cards': row['diagnostic'].get('selected_card_ids'),
                'exact_selected_card': row['diagnostic'].get('selected_card_exact'),
                'raw_payload_response': row['diagnostic'].get('act_payload', {}).get('raw_response')})
        node_metrics[method] = dict(grouped)
    for method in ['awm', 'trace2skill']:
        pairs = []
        for ours, other in zip(predictions['ours'], predictions[method]):
            assert (ours['dialogue_id'], ours['turn_index'], ours['gold']) == (
                other['dialogue_id'], other['turn_index'], other['gold'])
            if all(score_policies([r['gold']], [r['prediction']])['action_set_exact']
                   for r in [ours, other]):
                pairs.append((ours, other))
        paired_metrics[method] = {'both_actions_correct': len(pairs),
            'ours_payload_correct': sum(slot_signature(a['gold']) == slot_signature(a['prediction']) for a, b in pairs),
            'other_payload_correct': sum(slot_signature(b['gold']) == slot_signature(b['prediction']) for a, b in pairs)}
    train_nodes = defaultdict(Counter)
    for e in evidence:
        c = train_nodes[e['gold_target']]
        c['total'] += 1
        c['action_correct'] += int(e['action_correct'])
        c['joint_correct'] += int(e['action_correct'] and e['slot_exact_correct'])
        if e['action_correct']:
            c['predicted_empty_value_fields'] += sum(a.get('value') == '' for a in e['predicted_policy']['acts'])
    native = state['slot_policies']
    resolved = {k: v['policy'] for k, v in native.items()
                if v.get('status') == 'resolved' and v.get('policy', '').strip()}
    diagnosis28 = read(training/'ours/diagnosis_0028.json')
    batch = [e for e in evidence if e['sample_id'] in diagnosis28['sample_ids']]
    result = {'training_root': str(training), 'checkpoint': str(checkpoint), 'evaluation_root': str(evaluation),
        'scope': 'Read-only audit. No LLM calls, changed predictions, score normalization or new training.',
        'training': {'scheduled': len(evidence), 'action_correct': sum(e['action_correct'] for e in evidence),
            'action_correct_payload_wrong': sum(e['action_correct'] and not e['slot_exact_correct'] for e in evidence),
            'per_node': dict(train_nodes)},
        'offline': {'slot_updates_proposed': len(offline), 'slot_updates_applied': sum(bool(r['applied']) for r in offline),
            'rejected_keys': [r['update']['key'] for r in offline if not r['applied']],
            'generic_slot_policies_in_original_checkpoint': sum(v == 'Use named slots; ground values in prior context or visible API observation.'
                                                             for v in original_resources['slot_policies'].values())},
        'repaired_online': {'nodes': len(original_resources['slot_policies']),
            'resolved_nonempty_slot_policies': len(resolved), 'resolved_policies': resolved},
        'evaluation': {'turns': len(predictions['ours']), 'per_method_per_node': node_metrics,
            'ours_payload_failures': dict(failure_counts), 'common_action_correct_subsets': paired_metrics},
        'diagnosis28': {'summary': diagnosis28['summary'], 'sample_ids': diagnosis28['sample_ids'],
            'actual_action_correct_payload_wrong': sum(e['action_correct'] and not e['slot_exact_correct'] for e in batch),
            'actual_action_wrong': sum(not e['action_correct'] for e in batch)},
        'ours_action_correct_payload_wrong_cases': examples}
    assert len(predictions['ours']) == 115
    assert failure_counts['action_correct_payload_wrong'] == 43
    assert failure_counts['only_value_presence'] == 22
    (output/'evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(output/'evidence.json'), 'failures': dict(failure_counts),
                      'resolved_policies': len(resolved), 'diagnosis28': result['diagnosis28']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
