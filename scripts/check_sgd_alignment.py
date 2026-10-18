#!/usr/bin/env python3
"""Smoke/check the shared SGD executor; never use test labels for development.

Default: up to six representative TRAIN cases, a fresh native offline card set,
and frozen historical AWM/Trace2Skill resources. This is an integration check,
not a final comparison of retrained methods. --phase validation checks all
development targets unless --max-cases is given.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import traceback
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.sgd_adapter import iter_family_dialogues, gold_policy, score_policies, score_slot_items
from eval_tod.sgd_aligned import (SGDAlignedSkillAgent, PROTOCOL, predict_policy,
    training_contract, digest)
from eval_tod.sgd_agent import SGDAWMAgent
from eval_tod.sgd_skill_agent import policy_token
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase
from eval_tod.sgd_errors import record_error, error_summary
from eval_tod.multiwoz_aligned import atomic_json
from scripts.run_multiwoz_trace2skill_pilot import _predict


def run(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists(): raise FileExistsError(out/'summary.json')
    configure_usage(out)
    train = list(iter_family_dialogues(args.splits_dir,args.domain,'train'))
    with ZipFile(args.archive) as zipped:ontology=json.load(zipped.open('data/ontology.json'))
    ontology['_sgd_training_contract'] = training_contract(train)
    ours=SGDAlignedSkillAgent(family=args.domain,model=args.model,ontology=ontology,predictor=predict_policy,
        mining_method='trace_cover')
    ours.parallel_workers=args.workers
    ours.runtime_diagnostics_dir=out/'ours'
    atomic_json(out/'progress.json',{'phase':'prepare_offline_cards'})
    if args.ours_resource_dir:
        ours.load(args.ours_resource_dir)
        ours_alignment = ('aligned_checkpoint' if (args.ours_resource_dir/'aligned_runtime_contract.json').exists()
                          else 'legacy_checkpoint_under_new_runtime_diagnostic')
    else:
        set_phase('offline_compile')
        ours.mine(train,out/'ours/offline',cases_per_node=4)
        ours_alignment='fresh_aligned_offline_only_no_online_refinement'
    awm=SGDAWMAgent(family=args.domain,model=args.model,ontology=ontology,predictor=predict_policy)
    awm.load(args.resource_root/'awm')
    evolved=args.resource_root/'trace2skill/evolved_skill'
    if not (evolved/'SKILL.md').exists():raise FileNotFoundError(evolved/'SKILL.md')
    selected=list(iter_family_dialogues(args.splits_dir,args.domain,args.phase))
    cases=[(d,i) for d in selected for i in d['target_indices']['policy_turn_indices']]
    if args.phase=='train':
        # Representative integration cases chosen on TRAIN labels only.
        diverse, seen=[],set()
        for d,i in cases:
            token=policy_token(gold_policy(d,i))
            if token not in seen:diverse.append((d,i));seen.add(token)
        ids={(d['dialogue_id'],i) for d,i in diverse}
        cases=diverse+[(d,i) for d,i in cases if (d['dialogue_id'],i) not in ids]
    maximum=args.max_cases if args.max_cases is not None else (6 if args.phase=='train' else None)
    if maximum:cases=cases[:maximum]
    manifest={'protocol':PROTOCOL,'phase':args.phase,'model':args.model,'workers':args.workers,
        'evaluation_targets':'policy','case_ids':[[d['dialogue_id'],i] for d,i in cases],
        'train_turns':sum(len(d['target_indices']['policy_turn_indices']) for d in train),
        'train_contract_sha256':digest(ontology['_sgd_training_contract']),
        'source_training_resources':str(args.resource_root.resolve()),'ours_resource_alignment':ours_alignment,
        'baseline_resource_alignment':'legacy_frozen_training_resources_under_new_runtime_diagnostic',
        'result_use':'TRAIN smoke only; not a generalization score' if args.phase=='train'
                     else 'Development diagnostic; historical baseline training was not rerun'}
    atomic_json(out/'data_manifest.json',manifest)
    summaries={}
    for method in args.methods:
        directory=out/method
        directory.mkdir(exist_ok=True)
        def predict(case):
            d,i=case
            if method=='ours':return ours.predict(d,i)
            if method=='awm':return awm.predict(d,i)
            return _predict(case,args.model,ontology,evolved,predict_policy)
        atomic_json(out/'progress.json',{'phase':method,'completed':0,'total':len(cases)})
        set_phase('testing')
        values=ordered_cases(predict,cases,workers=args.workers,directory=directory,phase='testing')
        rows=[]
        for outcome,(d,i) in zip(values,cases):
            try:
                prediction,diagnostic=require_value(outcome)
                score_policies([gold_policy(d,i)],[prediction])
            except Exception as exc:
                if args.phase=='train':raise
                failure=record_error(directory,'testing',exc,dialogue_id=d['dialogue_id'],turn_index=i)
                prediction={'call':None,'acts':[],'_execution_error':True}
                diagnostic={'executor_protocol':PROTOCOL,'status':'error','error':failure['error']}
            rows.append({'dialogue_id':d['dialogue_id'],'turn_index':i,
                'gold':gold_policy(d,i),'prediction':prediction,'diagnostic':diagnostic})
            atomic_json(out/'progress.json',{'phase':method,'completed':len(rows),'total':len(cases)})
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        summary={'method':method,'protocol':PROTOCOL,'turns':len(rows),
            **score_policies([r['gold'] for r in rows],[r['prediction'] for r in rows]),
            'slot_items':score_slot_items([r['gold'] for r in rows],[r['prediction'] for r in rows]),
            'verified_executor_protocols':dict(Counter(r['diagnostic']['executor_protocol'] for r in rows)),
            'case_errors':error_summary(directory)}
        if summary['verified_executor_protocols']!={PROTOCOL:len(rows)}:raise ValueError('Mixed executor protocols')
        atomic_json(directory/'summary.json',summary)
        summaries[method]=summary
    result={'manifest':manifest,'methods':summaries,'llm_usage':persist_usage(),
        'status':'completed_with_errors' if any(s['case_errors']['total'] for s in summaries.values()) else 'completed'}
    atomic_json(out/'summary.json',result)
    atomic_json(out/'progress.json',{'phase':result['status'],'methods':len(summaries),'turns_per_method':len(cases)})
    return {m:{k:s[k] for k in ['turns','action_set_exact','slot_exact_given_action','turn_joint_ast']} for m,s in summaries.items()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',default='Homes')
    p.add_argument('--splits-dir',type=Path,default=ROOT/'data/eval/sgd/splits_10pct_seed42')
    p.add_argument('--archive',type=Path,default=ROOT/'data/eval/sgd/data.zip')
    p.add_argument('--resource-root',type=Path,default=ROOT/'outputs/sgd_homes_10pct_20261018_153955')
    p.add_argument('--ours-resource-dir',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--phase',choices=['train','validation'],default='train')
    p.add_argument('--max-cases',type=int)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--methods',nargs='+',choices=['awm','trace2skill','ours'],default=['awm','trace2skill','ours'])
    p.add_argument('--model',default='qwen3.6-flash')
    args=p.parse_args()
    if args.workers<1 or (args.max_cases is not None and args.max_cases<1):p.error('Limits must be positive')
    try:print(json.dumps(run(args),ensure_ascii=True))
    except Exception as exc:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output_dir/'run_failed.json',{'error':str(exc),'traceback':traceback.format_exc()})
        raise
