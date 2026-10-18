#!/usr/bin/env python3
"""Retry only execution-failed predictions, then rescore every original target.

Never selects cases by action/slot correctness. Successful predictions, source
files, gold labels and case order are preserved. Recovered outputs are accepted
whether right or wrong; this is transport recovery, not answer selection.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import traceback
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from eval_tod.sgd_adapter import iter_family_dialogues,gold_policy,score_policies,score_slot_items
from eval_tod.sgd_aligned import SGDAlignedSkillAgent,predict_policy,training_contract,PROTOCOL
from eval_tod.sgd_parallel import ordered_cases,require_value
from eval_tod.sgd_usage import configure_usage,persist_usage,set_phase
from eval_tod.sgd_errors import record_error,error_summary
from eval_tod.multiwoz_aligned import atomic_json
from scripts.run_multiwoz_trace2skill_pilot import _predict


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def method_directory(root, method):
    if (root/method/'predictions.jsonl').exists():return root/method
    return root/method/method


def baseline_resource_root(root):
    visited=set()
    while root not in visited:
        visited.add(root)
        path=root/'trace2skill/data_manifest.json'
        if path.exists():return Path(read(path)['source_training_resources'])
        path=root/'recovery_manifest.json'
        if not path.exists():break
        root=Path(read(path)['source_run'])
    raise ValueError('Cannot locate source frozen baseline resources')


def run(args):
    source,out=args.source_root.resolve(),args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'comparison.json').exists():raise FileExistsError(out/'comparison.json')
    splits=ROOT/'data/eval/sgd/splits_10pct_seed42'
    train=list(iter_family_dialogues(splits,'Homes','train'))
    validation=list(iter_family_dialogues(splits,'Homes','validation'))
    cases={(d['dialogue_id'],i):(d,i) for d in validation for i in d['target_indices']['policy_turn_indices']}
    with ZipFile(ROOT/'data/eval/sgd/data.zip') as zipped:ontology=json.load(zipped.open('data/ontology.json'))
    ontology['_sgd_training_contract']=training_contract(train)
    configure_usage(out)
    originals,hashes,selected={}, {}, {}
    expected_ids=list(cases)
    for method in ['awm','trace2skill','ours']:
        directory=method_directory(source,method)
        path=directory/'predictions.jsonl'
        originals[method]=rows(path);hashes[method]=sha(path)
        if [(r['dialogue_id'],r['turn_index']) for r in originals[method]]!=expected_ids:
            raise ValueError('Source case order is not the fixed complete validation split')
        if any(r['gold']!=gold_policy(*cases[r['dialogue_id'],r['turn_index']]) for r in originals[method]):
            raise ValueError('Source gold differs from the frozen dataset')
        selected[method]=[r for r in originals[method] if r['prediction'].get('_execution_error')]
        errors=rows(directory/'case_errors.jsonl') if (directory/'case_errors.jsonl').exists() else []
        if {(e['dialogue_id'],e['turn_index']) for e in errors}!={(r['dialogue_id'],r['turn_index']) for r in selected[method]}:
            raise ValueError('Retry selection must match execution errors only')
    manifest={'source_run':str(source),'policy_turns':len(cases),'evaluation_targets':'policy',
        'source_prediction_sha256':hashes,'retry_counts':{m:len(rs) for m,rs in selected.items()},
        'selection_rule':'prediction._execution_error only; no selection by action or slot correctness',
        'successful_predictions':'reuse exactly','gold_and_order':'unchanged',
        'patch':'role_scoped_stage_json_v1; separate reference tool-control from business action schemas',
        'checkpoint':str(args.ours_resource_dir.resolve()),'workers':args.workers}
    atomic_json(out/'recovery_manifest.json',manifest)
    if args.dry_run:return manifest
    agent=SGDAlignedSkillAgent(family='Homes',model=args.model,ontology=ontology,predictor=predict_policy,
        mining_method='trace_cover')
    agent.load(args.ours_resource_dir)
    expected_hashes={r['diagnostic']['resource_sha256'] for r in originals['ours'] if not r['prediction'].get('_execution_error')}
    if expected_hashes!={agent.resource_hash()}:raise ValueError('Recovery checkpoint differs from evaluated frozen resources')
    checkpoint_hash=agent.resource_hash()
    baseline_resources=baseline_resource_root(source)
    evolved=baseline_resources/'trace2skill/evolved_skill'
    summaries={}
    for method in ['awm','trace2skill','ours']:
        directory=out/method;directory.mkdir(exist_ok=True)
        replacements={};failed=selected[method]
        atomic_json(out/'progress.json',{'phase':'recover_'+method,'completed':0,'total':len(failed),
            'unchanged_valid_predictions':len(originals[method])-len(failed)})
        if failed:
            def predict(case):
                if method=='ours':return agent.predict(*case)
                return _predict(case,args.model,ontology,evolved,predict_policy)
            set_phase('testing')
            jobs=[cases[r['dialogue_id'],r['turn_index']] for r in failed]
            values=ordered_cases(predict,jobs,workers=min(args.workers,len(jobs)),directory=directory,phase='testing')
            with (directory/'retry_predictions.jsonl').open('w',encoding='utf-8') as stream:
                for index,(outcome,original) in enumerate(zip(values,failed),1):
                    try:
                        prediction,diagnostic=require_value(outcome)
                        score_policies([original['gold']],[prediction])
                    except Exception as exc:
                        error=record_error(directory,'testing',exc,dialogue_id=original['dialogue_id'],turn_index=original['turn_index'])
                        prediction={'call':None,'acts':[],'_execution_error':True}
                        diagnostic={'executor_protocol':PROTOCOL,'status':'error','error':error['error']}
                    row={**original,'prediction':prediction,'diagnostic':diagnostic}
                    replacements[original['dialogue_id'],original['turn_index']]=row
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    atomic_json(out/'progress.json',{'phase':'recover_'+method,'completed':index,'total':len(failed)})
        merged=[]
        for original in originals[method]:
            key=original['dialogue_id'],original['turn_index']
            row=replacements.get(key,original)
            if key not in replacements and row!=original:raise AssertionError('Successful source prediction changed')
            merged.append(row)
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged),encoding='utf-8')
        if len(merged)!=115:raise ValueError('Every method must retain all 115 targets')
        errors=error_summary(directory)
        summary={'method':method,'protocol':PROTOCOL,'case_errors':errors,'recovery_retried':len(failed),
            'reused_predictions':115-len(failed),'source_execution_errors':len(failed),
            **score_policies([r['gold'] for r in merged],[r['prediction'] for r in merged]),
            'slot_items':score_slot_items([r['gold'] for r in merged],[r['prediction'] for r in merged]),
            'execution_status':errors['status']}
        atomic_json(directory/'summary.json',summary);summaries[method]=summary
    for method in originals:
        if sha(method_directory(source,method)/'predictions.jsonl')!=hashes[method]:raise AssertionError('Source predictions changed')
    if agent.resource_hash()!=checkpoint_hash:raise AssertionError('Recovery updated frozen resources')
    failed=sum(s['case_errors']['total'] for s in summaries.values())
    result={'methods':summaries,'manifest':manifest,'llm_usage':persist_usage(),
        'remaining_execution_errors':failed,'status':'completed_with_errors' if failed else 'completed',
        'checks':{'all_115_targets_retained':True,'only_execution_failed_cases_requeried':True,
                  'successful_source_predictions_unchanged':True,'source_files_unchanged':True,
                  'frozen_resource_hash_unchanged':True}}
    atomic_json(out/'comparison.json',result)
    text=['# SGD Homes: recovered execution failures, all 115 policy turns','',
        'Only failed execution cases were retried. Other predictions, gold labels, checkpoint and order were retained.',
        'The output-contract patch separates reference tool control from business actions; no gold answer or slot rule was added.','',
        '| Method | Action exact | Action F1 | Payload exact given action | Joint | Slot-item F1 | Remaining errors |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method,s in summaries.items():
        vals=[f'{100*s[k]:.2f}%' for k in ['action_set_exact','action_micro_f1','slot_exact_given_action','turn_joint_ast']]
        text.append('| '+' | '.join([method,*vals,f"{100*s['slot_items']['all_turns']['micro_f1']:.2f}%",str(s['case_errors']['total'])])+' |')
    text+=['','Frozen legacy training resources remain in use: this is execution recovery of the earlier diagnostic, not a new aligned training experiment.']
    (out/'comparison.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    atomic_json(out/'progress.json',{'phase':result['status'],'remaining_execution_errors':failed,'turns_per_method':115})
    return {m:{k:s[k] for k in ['action_set_exact','slot_exact_given_action','turn_joint_ast','execution_status']} for m,s in summaries.items()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=ROOT/'outputs/sgd_homes_aligned_diagnostic_20261018_194854')
    p.add_argument('--ours-resource-dir',type=Path,default=ROOT/'outputs/sgd_homes_policy_recheck_20261018_191306/repaired_resources')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model',default='qwen3.6-flash')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.workers<1:p.error('workers must be positive')
    try:print(json.dumps(run(args),ensure_ascii=True))
    except Exception as exc:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output_dir/'run_failed.json',{'error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc()})
        raise
