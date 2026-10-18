"""Freeze the existing whole-dialogue 10% sample into a portable schema profile."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output, schema_scope='seen'):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be separate directories')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new immutable profile directory')
    original = json.loads((source/'INDEX.json').read_text(encoding='utf-8'))
    if original.get('sample', {}).get('fraction') != .1:
        raise ValueError('Expected an existing 10% dialogue sample')
    output.mkdir(parents=True)
    index = {k: original[k] for k in ['source_archive', 'split_policy', 'train_families',
                                     'excluded_families', 'ontology_services', 'sample', 'counting_note']}
    index['source_archive'] = 'data/eval/sgd/data.zip'
    index['sample'] = {k: v for k, v in index['sample'].items() if k != 'source_dir'}
    index['profile'] = {'version': 'sgd_dialogue_10pct_official_splits_v1',
        'schema_scope': schema_scope, 'seed': original['sample']['seed'],
        'sampling': 'Reuse original family-stratified 10% whole-dialogue IDs; no resampling or within-dialogue truncation.',
        'schema_filter': 'Keep evaluation family views only if every target service occurs in sampled TRAIN for that family.' if schema_scope == 'seen' else 'Keep all sampled service versions; includes schema transfer.',
        'filter_order': '10% sample first, schema filter second',
        'source_index_sha256': sha(source/'INDEX.json'), 'evaluation_targets': 'policy',
        'global_act_rule': 'Retain source family-owned turns and single-family global acts; unassigned multi-family global acts remain context only.'}
    index['families'] = {}
    manifest = {'families': {}, 'file_sha256': {}, 'source_file_sha256': {}}
    dialogue_ids = {s: set() for s in ['train', 'validation', 'test']}
    event_ids = {s: set() for s in dialogue_ids}
    unassigned = {s: set() for s in dialogue_ids}
    for family in original['train_families']:
        data = {s: load(source/family/(s+'.jsonl.gz')) for s in dialogue_ids}
        seen = {service for d in data['train'] for service in d['services']}
        index['families'][family] = {}
        manifest['families'][family] = {'train_services': sorted(seen)}
        for split, all_rows in data.items():
            selected = all_rows if split == 'train' or schema_scope == 'all' else [
                d for d in all_rows if set(d['services']) <= seen]
            counts = Counter()
            for d in selected:
                dialogue_ids[split].add(d['dialogue_id'])
                counts['dialogues'] += 1
                for key, field in [('policy_turns', 'policy_turn_indices'), ('api_events', 'api_turn_indices'),
                                   ('domain_act_turns', 'domain_act_turn_indices'), ('global_act_turns', 'global_act_turn_indices')]:
                    counts[key] += len(d['target_indices'][field])
                for turn in d['target_indices']['policy_turn_indices']:
                    event = (d['dialogue_id'], turn)
                    if event in event_ids[split]:
                        raise ValueError(f'Duplicate target turn across family views: {event}')
                    event_ids[split].add(event)
                unassigned[split].update((d['dialogue_id'], i) for i in d['target_indices'].get('unassigned_global_act_turn_indices', []))
            path = output/family/(split+'.jsonl.gz')
            path.parent.mkdir(exist_ok=True)
            if split == 'train':
                path.write_bytes((source/family/path.name).read_bytes())
            else:
                encoded = ''.join(json.dumps(d, ensure_ascii=False, separators=(',', ':'))+'\n' for d in selected).encode()
                path.write_bytes(gzip.compress(encoded, mtime=0))
            index['families'][family][split] = {k: counts[k] for k in
                ['dialogues', 'policy_turns', 'api_events', 'domain_act_turns', 'global_act_turns']}
            manifest['families'][family][split] = {'selected_ids': [d['dialogue_id'] for d in selected],
                'excluded_unseen_schema_ids': [d['dialogue_id'] for d in all_rows if d not in selected]}
            rel = str(path.relative_to(output)).replace('\\', '/')
            manifest['file_sha256'][rel] = sha(path)
            manifest['source_file_sha256'][rel] = sha(source/family/path.name)
    for i, a in enumerate(dialogue_ids):
        for b in list(dialogue_ids)[i+1:]:
            if dialogue_ids[a] & dialogue_ids[b]:
                raise ValueError('Dialogue leakage across official splits')
    index['source_dialogues'] = {s: len(ids) for s, ids in dialogue_ids.items()}
    index['unique_target_turns'] = {s: len(ids) for s, ids in event_ids.items()}
    index['unassigned_global_context_turns'] = {s: len(ids-event_ids[s]) for s, ids in unassigned.items()}
    index['eligible_families'] = {s: [f for f in index['train_families'] if index['families'][f][s]['policy_turns']]
                                  for s in ['validation', 'test']}
    (output/'INDEX.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    manifest['index_sha256'] = sha(output/'INDEX.json')
    manifest['verified_no_split_leakage'] = True
    manifest['verified_no_duplicate_target_turns'] = True
    (output/'SAMPLING_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/eval/sgd/splits_10pct_seed42'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--schema-scope', choices=['seen', 'all'], default='seen')
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.schema_scope)
    print(json.dumps({k: result[k] for k in ['source_dialogues', 'unique_target_turns', 'eligible_families']}, ensure_ascii=False))
