"""Field-level TRAIN supervision; never normalize predictions or benchmark gold."""
from collections import Counter, defaultdict
from copy import deepcopy
import json


def value_state(act):
    if 'value' not in act:
        return {'presence': 'absent'}
    value = act['value']
    kind = 'null' if value is None else 'empty_string' if value == '' else 'nonempty'
    return {'presence': kind, 'value': deepcopy(value)}


def payload_diff(gold, prediction, *, action_correct):
    """Compare slot multiplicity, presence and exact value as separate facts."""
    if not action_correct:
        return {'applicable': False, 'reason': 'action_wrong; diagnose heads before payload'}
    groups = []
    for policy in [gold, prediction]:
        grouped = defaultdict(list)
        for act in policy['acts']:
            grouped[(act['intent'], act['domain'], act['slot'])].append(value_state(act))
        groups.append(grouped)
    expected, actual = groups
    missing, extra, values = [], [], []
    for key in sorted(expected.keys() | actual.keys()):
        encode = lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False)
        gc, pc = Counter(map(encode, expected[key])), Counter(map(encode, actual[key]))
        # Remove exact matches first, preserving repeated slot multiplicity.
        left = [json.loads(s) for s in sorted((gc-pc).elements())]
        right = [json.loads(s) for s in sorted((pc-gc).elements())]
        name = dict(zip(['intent', 'service', 'slot'], key))
        for g, p in zip(left, right):
            values.append({**name, 'gold': g, 'predicted': p,
                'kind': 'value_presence' if g['presence'] != p['presence'] else 'value_content'})
        if len(left) > len(right):
            missing.append({**name, 'count': len(left)-len(right)})
        if len(right) > len(left):
            extra.append({**name, 'count': len(right)-len(left)})
    calls = []
    gcall, pcall = gold.get('call'), prediction.get('call')
    if (gcall is None) != (pcall is None):
        calls.append({'field': 'call_presence', 'gold': gcall is not None, 'predicted': pcall is not None})
    elif gcall is not None:
        for field in ['service', 'method']:
            if gcall[field] != pcall[field]:
                calls.append({'field': field, 'gold': gcall[field], 'predicted': pcall[field]})
        for key in sorted(gcall['parameters'].keys() | pcall['parameters'].keys()):
            gs = {'presence': 'present', 'value': gcall['parameters'][key]} if key in gcall['parameters'] else {'presence': 'absent'}
            ps = {'presence': 'present', 'value': pcall['parameters'][key]} if key in pcall['parameters'] else {'presence': 'absent'}
            if gs != ps:
                calls.append({'field': 'parameter', 'slot': key, 'gold': gs, 'predicted': ps})
    kinds = sorted({*(['missing_slot'] if missing else []), *(['extra_slot'] if extra else []),
                    *(v['kind'] for v in values), *(['call_payload'] if calls else [])})
    return {'applicable': True, 'error_kinds': kinds, 'missing_slots': missing,
            'extra_slots': extra, 'act_value_differences': values, 'call_differences': calls,
            'note': 'Exact matched items removed first. Remaining repeated values are compared in stable order; pairing is descriptive, not entity alignment.'}


def compact_api_sources(events, context, *, max_entities=4):
    """Bound logged visible API evidence, marking omitted rows explicitly."""
    compact = []
    for event in events:
        result = event.get('result')
        if isinstance(result, list) and all(isinstance(row, dict) for row in result):
            ranked = sorted(enumerate(result), key=lambda pair: (
                -sum(len(str(v)) for k, v in pair[1].items()
                     if k in {'property_name', 'address', 'name'} and str(v) and str(v).lower() in context.lower()),
                pair[0]))[:max_entities]
            view = {'total_entities': len(result), 'shown_entities': [dict(row_index=i, fields=row) for i, row in ranked],
                    'omitted_entities': max(0, len(result)-max_entities),
                    'selection': 'context identifier overlap, then original row order; omission is not absence'}
        else:
            view = deepcopy(result)
        compact.append({**{k: deepcopy(v) for k, v in event.items() if k != 'result'}, 'result': view})
    return compact


def select_train_exemplars(records):
    """Keep successes, correct-head payload failures and head failures distinct."""
    selected, seen = [], set()
    categories = [([r for r in records if r['action_correct'] and r['slot_exact_correct']], 1),
                  ([r for r in records if r['action_correct'] and not r['slot_exact_correct']], 2),
                  ([r for r in records if not r['action_correct']], 1)]
    for candidates, quota in categories:
        # Prefer different payload failure types rather than repeated examples.
        fingerprints = set()
        ordered = []
        for r in candidates:
            key = (r.get('gold_target'), tuple(r.get('slot_diff', {}).get('error_kinds', [])))
            if key not in fingerprints:
                ordered.append(r); fingerprints.add(key)
        ordered += [r for r in candidates if r not in ordered]
        for row in ordered[:quota]:
            if row['sample_id'] in seen:
                continue
            seen.add(row['sample_id'])
            item = {k: deepcopy(row[k]) for k in ['sample_id', 'gold_target', 'gold_policy', 'predicted_policy',
                'action_correct', 'slot_exact_correct', 'slot_diff', 'visible_api_sources'] if k in row}
            item['context'] = row.get('context', '')[-900:]
            selected.append(item)
    return selected
