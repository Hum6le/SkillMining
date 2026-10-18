#!/usr/bin/env python3
"""Inspect Homes development errors and freeze dialogue-level split identities."""
from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.sgd_adapter import action_counter, score_policies, score_slot_items, slot_signature
from eval_tod.sgd_aligned import prior_api_history, training_contract
from eval_tod.multiwoz_aligned import atomic_json


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def audit(splits, domain, prediction_file, offline, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest, original_ids, datasets = {}, {}, {}
    for split in ['train', 'validation', 'test']:
        path = splits / domain / (split+'.jsonl.gz')
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            data = [json.loads(line) for line in stream if line.strip()]
        datasets[split] = data
        original_ids[split] = {str(d.get('original_id', d['dialogue_id'])) for d in data}
        cases = [[d['dialogue_id'], i] for d in data for i in d['target_indices']['policy_turn_indices']]
        calls = sum(len(d['target_indices']['api_turn_indices']) for d in data)
        manifest[split] = {'role': {'train':'learning','validation':'development_used_for_diagnosis',
            'test':'out_of_scope_schema_transfer_not_run'}[split], 'dialogues': len(data),
            'policy_turns': len(cases), 'call_turns': calls, 'non_call_turns':len(cases)-calls,
            'services': sorted({s for d in data for s in d['services']}),
            'unassigned_global_turns': sum(len(d['target_indices'].get('unassigned_global_act_turn_indices', [])) for d in data),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'case_ids':cases}
    overlap = {a+'_'+b: len(original_ids[a]&original_ids[b]) for a,b in
        [('train','validation'),('train','test'),('validation','test')]}
    if any(overlap.values()): raise ValueError('Original dialogue overlaps between official splits')
    plan = {'family':domain,'fraction':.1,'seed':42,'sampling_unit':'complete dialogue',
        'target':'all family-owned policy turns, CALL and non-CALL',
        'global_acts':'Single-family global acts included; unowned multi-family global acts counted separately',
        'split_rule':'Official split; sha256(seed NUL split NUL dialogue_id) rank; ceil quota',
        'original_id_overlap':overlap,'splits':manifest,
        'active_evaluation':{'split':'validation','services':manifest['validation']['services'],
            'turns':manifest['validation']['policy_turns'],'fixed_previous_case_ids':True,
            'schema_transfer':'out_of_scope_per_user'}}
    atomic_json(output/'evaluation_dataset.json',plan)
    atomic_json(output/'train_serialization_contract.json',training_contract(datasets['train']))
    predicted = rows(prediction_file)
    actual_ids = [[r['dialogue_id'],r['turn_index']] for r in predicted]
    if actual_ids != manifest['validation']['case_ids']:
        raise ValueError('Prediction identities must match the complete development split')
    dialogue_by_id = {d['dialogue_id']:d for d in datasets['validation']}
    failures, examples, heads = Counter(), [], Counter()
    for row in predicted:
        g,p=row['gold'],row['prediction']
        if action_counter(g) != action_counter(p) or p.get('_execution_error'):continue
        if slot_signature(g)==slot_signature(p):continue
        heads[str(sorted(action_counter(g)))] += 1
        reasons=[]
        if g['call']!=p['call']:reasons.append('call_payload_mismatch')
        gs=Counter((a['intent'],a['domain'],a['slot']) for a in g['acts'])
        ps=Counter((a['intent'],a['domain'],a['slot']) for a in p['acts'])
        if gs-ps:reasons.append('missing_act_slots_or_multiplicity')
        if ps-gs:reasons.append('extra_act_slots_or_multiplicity')
        gp={(a['intent'],a['domain'],a['slot']):'value' in a for a in g['acts']}
        pp={(a['intent'],a['domain'],a['slot']):'value' in a for a in p['acts']}
        if any(gp[k]!=pp[k] for k in gp.keys()&pp.keys()):reasons.append('value_presence_mismatch')
        gv=Counter((a['intent'],a['domain'],a['slot'],'value' in a,a.get('value','')) for a in g['acts'])
        pv=Counter((a['intent'],a['domain'],a['slot'],'value' in a,a.get('value','')) for a in p['acts'])
        if gv!=pv and gs==ps:reasons.append('same_slot_names_wrong_values_or_presence')
        failures.update(reasons)
        d=dialogue_by_id[row['dialogue_id']]
        history=prior_api_history(d,row['turn_index'])
        examples.append({'dialogue_id':row['dialogue_id'],'turn_index':row['turn_index'],
            'reasons':reasons,'gold':g,'prediction':p,
            'has_prior_logged_observation':bool(history)})
    old=json.loads((offline/'sgd_skill_state.json').read_text(encoding='utf-8'))
    rejected=[r for r in old['ledger'] if r.get('phase')=='offline_compile' and not r.get('applied')]
    policies=old['resources']['slot_policies']
    result={'development_only':True,'metrics':score_policies([r['gold'] for r in predicted],[r['prediction'] for r in predicted]),
        'slot_items':score_slot_items([r['gold'] for r in predicted],[r['prediction'] for r in predicted]),
        'action_correct_payload_wrong_turns':len(examples),'overlapping_error_categories':dict(failures),
        'wrong_turns_by_head':dict(heads),'old_offline':{'nodes':len(policies),
            'distinct_slot_policy_bodies':len(set(policies.values())),
            'generic_fallback_policies':sum(v=='Use named slots; ground values in prior context or visible API observation.' for v in policies.values()),
            'rejected_updates':rejected},'cases':examples}
    atomic_json(output/'slot_error_audit.json',result)
    lines=['# SGD Homes: evaluation construction and slot audit','','## Dataset','',
        '| Split | Role | Dialogues | Policy turns | CALL | Non-CALL | Services |',
        '|---|---|---:|---:|---:|---:|---|']
    for split,r in manifest.items():lines.append('| '+' | '.join(str(v) for v in [split,r['role'],r['dialogues'],r['policy_turns'],r['call_turns'],r['non_call_turns'],','.join(r['services'])])+' |')
    lines+=['','Whole-dialogue sampling; official splits retained; original dialogue ID overlap is zero.',
        'All family-owned policy targets count. Unowned multi-family global acts are explicitly recorded in evaluation_dataset.json.',
        'Validation has already been used for development. Test labels were not used to diagnose errors or derive slot rules.',
        '', '## Current repaired checkpoint (development)', '',
        f"Action exact: {result['metrics']['action_set_exact']:.2%}; payload exact given action: {result['metrics']['slot_exact_given_action']:.2%}; joint: {result['metrics']['turn_joint_ast']:.2%}.",
        f"Action-correct, payload-wrong turns: {len(examples)}. Categories below overlap.",'']
    lines += [f'- {key}: {value}' for key,value in failures.items()]
    lines += ['', '## Confirmed implementation issues', '',
        f"- Old offline compiler retained generic fallback policies on {result['old_offline']['generic_fallback_policies']}/{len(policies)} nodes; all nine attempted slot-policy updates used invalid targets.",
        '- Payload generation used preselection candidate cards rather than looking up the final selected node.',
        '- Prior logged API results were missing from later non-CALL prompts, including entity contact facts.',
        '- Selection/binding prompts lacked the complete slot schema and shared TRAIN value-presence/format contract.',
        '- Legacy Ours routing read past gold policy labels, outside its utterance-only input contract.',
        '', '## Implemented protocol', '',
        '- Same schema, logged API prefix, TRAIN serialization statistics, head selector and fixed-head binder for all methods.',
        '- No runtime gold-policy-history routing. Current logged API result remains gated by exact predicted CALL.',
        '- Selected card is looked up after final heads. Reference uses existing agent-controlled search/read/drop/finish.',
        '- Every offline node requires a persisted prose slot policy; invalid IDs trigger retry/failure. Up to four diverse initial TRAIN snippets per node.',
        '- Strict original turn metrics retained; per-slot multiset precision/recall/F1 added for diagnostics.',
        '- Reusing old training resources under the new executor is labeled a diagnostic, not a fully aligned training result.',
        '- Remaining limits: logged-call replay is not a simulator; exact slot sets/entity choices can be underspecified by the prefix.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'splits':{k:{f:r[f] for f in ['dialogues','policy_turns','call_turns']} for k,r in manifest.items()},
            'errors':dict(failures),'report':str(output/'report.md'),'slot_items':result['slot_items']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits-dir',type=Path,default=ROOT/'data/eval/sgd/splits_10pct_seed42')
    p.add_argument('--domain',default='Homes')
    p.add_argument('--predictions',type=Path,default=ROOT/'outputs/sgd_homes_policy_recheck_20261018_191306/evaluation/predictions.jsonl')
    p.add_argument('--offline',type=Path,default=ROOT/'outputs/sgd_homes_10pct_20261018_153955/ours/offline')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.splits_dir,args.domain,args.predictions,args.offline,args.output_dir),ensure_ascii=True))
