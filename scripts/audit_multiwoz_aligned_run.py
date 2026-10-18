#!/usr/bin/env python3
"""Verify TRAIN/eval executor and action-card continuity from saved artifacts."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_tod.multiwoz_aligned import (
    OFFLINE_SLOT_PROTOCOL, CONDITIONAL_SLOT_PROTOCOL, initial_reference_text, initial_slot_policy, offline_slot_report,
)
from eval_tod.multiwoz_slot_policy import conditional_slot_policy
from eval_tod.multiwoz_native import OFFLINE_PROTOCOL as NATIVE_OFFLINE_PROTOCOL, native_resources_report, native_reference


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_offline_resources(root, requested=None):
    offline = root / 'offline'
    profiles = read(offline / 'train_card_profiles.json')
    report = read(offline / 'offline_slot_audit.json')
    assert report == read(root / 'offline_slot_audit.json'), 'Offline slot provenance changed after TRAIN'
    assert report['protocol'] in {OFFLINE_SLOT_PROTOCOL, CONDITIONAL_SLOT_PROTOCOL, NATIVE_OFFLINE_PROTOCOL}
    conditional = report['protocol'] == CONDITIONAL_SLOT_PROTOCOL
    limit = report['initial_snippets_per_node']
    if requested is not None:
        assert requested == {'protocol': report['protocol'], 'initial_snippets_per_node': limit}
    if report['protocol'] == NATIVE_OFFLINE_PROTOCOL:
        resources = read(offline / 'compiled_native_resources.json')
        rebuilt_resources = {'action_rules': {}, 'slot_policies': {}, 'reference': {}}
        for path in sorted((offline / 'card_compile').glob('batch_*.json')):
            for update in read(path)['updates']:
                assert update['key'] not in rebuilt_resources[update['resource']]
                rebuilt_resources[update['resource']][update['key']] = update['text']
        assert rebuilt_resources == resources
        assert report == native_resources_report(profiles, resources, limit) and report['verified']
        cards = read(offline / 'action_cards.json')
        assert set(cards) == {p['node_id'] for p in profiles}
        for profile in profiles:
            key = profile['node_id']
            assert cards[key]['payload_policy'] == resources['slot_policies'][key]
            assert cards[key]['rule'] == resources['action_rules'][key]
            assert 0 < len(profile['train_cases']) <= limit
        for directory in (offline, root):
            assert read(directory / 'sgd_standard_state.json')['base_reference'] == native_reference(profiles, resources)
        return report
    compiled = {}
    for path in sorted((offline / 'card_compile').glob('batch_*.json')):
        for card in read(path)['cards']:
            assert card['node_id'] not in compiled, 'Repeated compiled node'
            compiled[card['node_id']] = card
    cards = read(offline / 'action_cards.json')
    assert set(compiled) == set(cards) == {p['node_id'] for p in profiles}
    rebuilt = offline_slot_report(profiles, compiled, limit)
    rebuilt['protocol'] = report['protocol']
    assert rebuilt == report and report['verified']
    for profile in profiles:
        cases = profile['train_cases']
        assert profile['initial_snippets_per_node'] == limit
        assert 0 < len(cases) <= limit
        assert len({json.dumps([c['context'], c['gold']], sort_keys=True) for c in cases}) == len(cases)
        node_id = profile['node_id']
        assert cards[node_id]['rule'] == compiled[node_id]['action_rule']
        renderer = conditional_slot_policy if conditional else initial_slot_policy
        assert cards[node_id]['payload_policy'] == renderer(profile, compiled[node_id]), \
            'Compiled offline slot rules did not reach persisted card'
    reference = initial_reference_text(profiles)
    for directory in (offline, root):
        assert read(directory / 'sgd_standard_state.json')['base_reference'] == reference, \
            'Initialization snippets did not survive checkpoint save/load'
    return report


def audit(root):
    manifest = read(root / 'data_manifest.json')
    contract = read(root / 'runtime_contract.json')
    offline_contract = read(root / 'offline/runtime_contract.json')
    assert contract == offline_contract, 'Offline/TRAIN/eval execution contract drift'
    assert contract['cards_ready'] is True
    assert contract['runtime_gold_action_history'] is False
    limits = contract['retrieval']
    summary = read(root / 'summary.json')
    training = summary['training']
    assert training['case_errors']['total'] == 0
    train = [json.loads(line)['sgd_evidence'] for line in (root / 'ours_training_turns.jsonl').read_text(encoding='utf-8').splitlines()]
    evaluation = [json.loads(line) for line in (root / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(train) == training['scheduled_turns'] == training['rollout_turns']
    assert len(evaluation) == manifest['eval_turns']
    graph = read(root / 'subgraph.json')
    offline_cards, final_cards = read(root / 'offline/action_cards.json'), read(root / 'action_cards.json')
    nodes = {n['id'] for n in graph['nodes']}
    assert set(offline_cards) == set(final_cards) == nodes
    offline_slots = None
    if manifest.get('offline_resources'):
        offline_slots = audit_offline_resources(root, manifest['offline_resources'])
    for card in final_cards.values():
        if card['rule'].lstrip().startswith('{'):
            try:
                composite = json.loads(card['rule'])
            except json.JSONDecodeError:
                composite = None
            assert not (isinstance(composite, dict) and 'payload_policy' in composite), \
                'Composite card was stored inside action_rule; slot_policy update was lost'
    runtime_contract = {k: v for k, v in contract.items() if k != 'cards_ready'}
    binary_keys = {tuple(key) for key in read(root / 'train_ontology.json')['value_free_acts']}
    evaluated, refs, unknown_bundles = Counter(), [], Counter()
    for phase, rows, card_pool, expected_hash in [('train', train, offline_cards, training['snapshot_sha256']),
                                                 ('evaluation', evaluation, final_cards, training['final_sha256'])]:
        for row in rows:
            diagnostic = row['runtime_diagnostic'] if phase == 'train' else row['diagnostic']
            assert diagnostic.get('status') == 'ok'
            assert diagnostic['runtime_contract'] == runtime_contract
            if contract.get('candidate_transport'):
                assert diagnostic['payload_action_rules_included'] == contract.get('payload_action_rules_included', False)
            assert diagnostic['resource_sha256'] == expected_hash, f'{phase} resource snapshot drift'
            prediction = row['predicted_policy'] if phase == 'train' else row['prediction']
            heads = {tuple(h) for h in diagnostic['selected_action_heads']}
            assert heads == {(a['intent'], a['domain']) for a in prediction['acts']}
            selected = diagnostic['selected_card_ids']
            assert len(selected) <= 1
            if not selected:
                unknown_bundles[phase] += 1
            for node_id, card in zip(selected, diagnostic['selected_action_cards']):
                assert card == card_pool[node_id], 'Executor did not read the current saved action card'
                assert {tuple(h) for h in card['heads']} == heads
            for act in prediction['acts']:
                if (act['intent'], act['domain'], act['slot']) in binary_keys:
                    assert 'value' not in act
            reference = diagnostic['retrieved_reference']
            if contract.get('reference_policy', '').startswith('agent_'):
                lookup = diagnostic['reference_plan_diagnostic']
                assert lookup['mode'] == 'agent' and lookup['limits'] == limits
                for stage in ('action', 'payload'):
                    assert sum(d['stage'] == stage for d in lookup['decisions']) <= limits['reference_max_rounds']
                for units in (diagnostic['selection_reference'], reference):
                    assert len(json.dumps(units, ensure_ascii=False)) <= limits['reference_context_chars']
                    assert len({u['section_id'] for u in units}) == len(units)
                    assert all(u['truncated'] == (len(u['content']) < u['total_chars']) for u in units)
            else:
                assert len(reference) <= limits['reference_top_k']
                assert len(json.dumps(reference, ensure_ascii=False)) <= limits['reference_total_chars']
                assert all(len(r['content']) <= limits['reference_section_chars'] for r in reference)
            assert len({r['content'] for r in reference}) == len(reference)
            refs.append(len(reference))
            evaluated[phase] += 1
    assert not (root / 'card_completion').exists(), 'Evaluation must not rewrite trained cards'
    slot_updates = []
    for path in sorted(root.glob('reflection_[0-9][0-9][0-9][0-9].json')):
        for update in read(path).get('accepted_updates', []):
            if update['resource'] == 'slot_policy' and update.get('status') == 'resolved':
                slot_updates.append(update['action'])
    # Final card view must reflect the last accepted resolved upsert. Multiple
    # groups may intentionally revise a card more than once.
    state = read(root / 'sgd_standard_state.json')['state']
    for card in final_cards.values():
        policy = state['slot_policies'].get(card['label'], {})
        if policy.get('status') == 'resolved':
            assert card['payload_policy'] == policy['policy']
    result = {'verified': True, 'protocol': manifest['protocol'], 'train_eval_contract_identical': True,
        'offline_card_coverage': len(offline_cards), 'final_card_coverage': len(final_cards),
        'offline_slot_resources': None if offline_slots is None else {
            'protocol': offline_slots['protocol'], 'verified': offline_slots['verified'],
            'initial_snippets_per_node': offline_slots['initial_snippets_per_node'],
            'total_initial_snippets': sum(n['initial_snippets'] for n in offline_slots['nodes'].values()),
            'covered_slot_shapes': sum(n.get('covered_slot_shapes', 0) for n in offline_slots['nodes'].values()),
            'inventory_slot_shapes': sum(n.get('inventory_slot_shapes', 0) for n in offline_slots['nodes'].values()),
            'coverage_kind': offline_slots['coverage_kind']},
        'verified_turns': dict(evaluated), 'max_reference_sections': max(refs, default=0),
        'unseen_action_bundles': dict(unknown_bundles), 'resolved_slot_upserts': len(slot_updates),
        'all_current_card_upserts_persisted': True, 'no_eval_time_card_recompile': True,
        'train_snapshot_sha256': training['snapshot_sha256'], 'final_snapshot_sha256': training['final_sha256']}
    (root / 'protocol_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root)
