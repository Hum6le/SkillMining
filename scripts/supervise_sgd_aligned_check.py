#!/usr/bin/env python3
"""After an Ours API smoke passes, check all three frozen methods on the fixed 115 turns."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from eval_tod.multiwoz_aligned import atomic_json
from scripts.launch_sgd_homes_pilot import start
from scripts.launch_multiwoz_hotel_pilot import background_python


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def run(args):
    root=args.output_dir.resolve()
    root.mkdir(parents=True,exist_ok=True)
    common=[str(background_python()),'-u',str(ROOT/'scripts/check_sgd_alignment.py'),
        '--workers',str(args.workers),'--model','qwen3.6-flash']
    smoke_command=common+['--output-dir',str(root/'smoke'),'--phase','train','--max-cases','1',
        '--methods','ours','--ours-resource-dir',str(args.compiled_offline.resolve())]
    smoke=start(smoke_command,root/'smoke.log')
    atomic_json(root/'progress.json',{'phase':'Ours_API_smoke','pid':smoke.pid})
    if smoke.wait():raise RuntimeError('Ours API smoke failed; no 115-turn diagnostic was launched')
    atomic_json(root/'progress.json',{'phase':'fixed_115_evaluation','workers_per_method':args.workers})
    jobs={}
    def launch(method):
        command=common+['--output-dir',str(root/method),'--phase','validation',
            '--methods',method,'--ours-resource-dir',str(args.refined_ours.resolve())]
        process=start(command,root/(method+'.log'))
        jobs[method]=(process,command)
        atomic_json(root/'jobs.json',{'jobs':{m:{'pid':p.pid,'command':c} for m,(p,c) in jobs.items()},
            'evaluation_targets':'policy','validation_turns':115,'max_concurrent_methods':2,
            'training':'frozen legacy resources; runtime alignment diagnostic only'})
    pending=['ours','awm','trace2skill']
    while pending or any(p.poll() is None for p,c in jobs.values()):
        while pending and sum(p.poll() is None for p,c in jobs.values())<2:
            launch(pending.pop(0))
        progress={}
        for method,(p,c) in jobs.items():
            path=root/method/'progress.json'
            progress[method]=read(path) if path.exists() else {'phase':'starting'}
        atomic_json(root/'progress.json',{'phase':'fixed_115_evaluation','methods':progress})
        time.sleep(3)
    failed=[m for m,(p,c) in jobs.items() if p.returncode]
    if failed:raise RuntimeError('Diagnostic failed for: '+','.join(failed))
    methods,identities={},{}
    for method in jobs:
        methods[method]=read(root/method/method/'summary.json')
        lines=(root/method/method/'predictions.jsonl').read_text(encoding='utf-8').splitlines()
        rows=[json.loads(line) for line in lines if line.strip()]
        identities[method]=[(r['dialogue_id'],r['turn_index'],r['gold']) for r in rows]
    expected=[json.loads(line) for line in (ROOT/'outputs/sgd_homes_policy_recheck_20261018_191306/evaluation/predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    expected=[(r['dialogue_id'],r['turn_index'],r['gold']) for r in expected]
    if len(expected)!=115 or any(ids!=expected for ids in identities.values()):
        raise ValueError('Case IDs/order/gold differ from the user-fixed 115 policy turns')
    result={'methods':methods,'turns':115,'identical_to_previous_evaluation':True,
        'scope':'Shared new runtime, frozen legacy training resources; not a full aligned retraining comparison'}
    atomic_json(root/'comparison.json',result)
    lines=['# SGD Homes: fixed 115-turn runtime diagnostic','',result['scope'],'',
        '| Method | Action exact | Action F1 | Payload exact given action | Joint | Slot-item F1 | Errors |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,row in methods.items():
        values=[f'{row[k]*100:.2f}%' for k in ['action_set_exact','action_micro_f1','slot_exact_given_action','turn_joint_ast']]
        lines.append('| '+' | '.join([m,*values,f"{row['slot_items']['all_turns']['micro_f1']*100:.2f}%",str(row['case_errors']['total'])])+' |')
    lines+=['','All 115 prior Homes_1 targets retained with identical order and gold. No schema transfer test was run.',
        'Fresh offline card compilation and one Ours TRAIN API smoke were checked separately; these new offline cards were not substituted into the frozen legacy refined checkpoint.',
        'Original training/reflection artifacts were preserved. Complete aligned TRAIN runs remain a subsequent experiment.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(root/'progress.json',{'phase':'completed','methods':3,'turns_per_method':115})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--compiled-offline',type=Path,required=True)
    p.add_argument('--refined-ours',type=Path,default=ROOT/'outputs/sgd_homes_policy_recheck_20261018_191306/repaired_resources')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output_dir/'run_failed.json',{'error':str(exc),'type':type(exc).__name__})
        raise
