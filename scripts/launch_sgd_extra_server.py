"""Company workflow batch for ReAct, Skill-DisCo, ExpeL and Reflexion."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_tod.multiwoz_aligned import atomic_json
from scripts.launch_sgd_server_batch import workflow_settings, method_environment
from scripts.run_sgd_extra_baselines import families

METHODS = ['react', 'skill_disco', 'expel', 'reflexion']
METRICS = ['action_set_exact','action_micro_f1','slot_exact_given_action',
           'turn_joint_ast','call_gate_f1','api_joint_ast']


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def collect(out):
    plan = read(out/'batch_plan.json')
    state = read(out/'batch_state.json')
    results = {}
    for method in plan['methods']:
        result = read(out/method/'aggregate.json')
        progress = read(out/method/'progress.json')
        done = result.get('evaluation_complete') and not result.get('smoke')
        status = 'complete' if done else state.get('methods',{}).get(method,{}).get('status','queued')
        if done and result.get('counts',{}).get('turns') != plan['validation_policy_turns']:
            status = 'protocol_mismatch'
        if done and result.get('case_errors'):
            status = 'completed_with_case_errors'
        results[method] = {**result,'status':status,'progress':progress}
    atomic_json(out/'comparison.json', {'methods':results,
                'validation_policy_turns':plan['validation_policy_turns']})
    lines = ['# SGD extra baseline results','',
             f"Validation targets: {plan['validation_policy_turns']}; families: {len(plan['families'])}.",'',
             '| Method | Status | Action exact | Action F1 | Slot given action | Turn joint | Call F1 | API joint |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for method, result in results.items():
        done = result['status'] in ['complete','completed_with_case_errors']
        scores = [f'{result[k]:.2%}' if done else '--' for k in METRICS]
        lines.append('| '+' | '.join([method,result['status']]+scores)+' |')
    lines += ['', 'ExpeL uses local ABCD-aligned TRAIN gold policy/reference supervision (v2). '
              'All-policy targets and the shared aligned evaluator are used. Error cases remain in denominators.', '']
    (out/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    return {'batch':state,'methods':results}


def command(args, method):
    cmd = [sys.executable,'-u',str(ROOT/'scripts/run_sgd_extra_baselines.py'),
           '--provider','workflow','--workflow-pools',str(args.workflow_pools.resolve()),
           '--workflow-pool-method',args.workflow_pool_method,'--method',method,
           '--workers',str(args.workers),'--splits-dir',str(args.splits_dir.resolve()),
           '--archive',str(args.archive.resolve()),'--output-dir',str((args.output_dir/method).resolve())]
    if args.model:
        cmd += ['--model',args.model]
    return cmd


def main():
    import psutil
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--splits-dir',type=Path,default=ROOT/'data/eval/sgd/protocol_20pct_seed42_seen_schema_v1')
    p.add_argument('--archive',type=Path,default=ROOT/'data/eval/sgd/data.zip')
    p.add_argument('--workflow-pools',type=Path,default=ROOT/'configs/sgd_workflows.json')
    p.add_argument('--workflow-pool-method',default='awm')
    p.add_argument('--workers',type=int,default=10)
    p.add_argument('--model')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--status',action='store_true')
    p.add_argument('--run-batch',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();args.provider='workflow'
    out=args.output_dir.resolve();args.output_dir=out
    if args.status:
        if not (out/'batch_plan.json').exists():p.error('No batch_plan.json in this output directory')
        print(json.dumps(collect(out),ensure_ascii=False,indent=2));return
    if not 1 <= args.workers <= 10:p.error('Use 1..10 workers')
    settings=workflow_settings(args,args.workflow_pool_method)
    if args.workers > settings['max_inflight']:p.error('Workers exceed configured shared inflight cap')
    index=read(args.splits_dir/'INDEX.json')
    if index.get('profile',{}).get('schema_scope') != 'seen':p.error('Use the seen-schema profile')
    chosen=families(args.splits_dir)
    expected=sum(index['families'][f]['validation']['policy_turns'] for f in chosen)
    plan={'methods':METHODS,'families':chosen,'method_execution':'sequential',
          'workers':args.workers,'workflow_settings':settings,'splits_dir':str(args.splits_dir.resolve()),
          'profile_sha256':__import__('hashlib').sha256((args.splits_dir/'INDEX.json').read_bytes()).hexdigest(),
          'model':args.model,'archive':str(args.archive.resolve()),
          'train_policy_turns':sum(index['families'][f]['train']['policy_turns'] for f in chosen),
          'validation_policy_turns':expected}
    if args.dry_run:
        print(json.dumps({**plan,'model_calls':0,'commands':{m:command(args,m) for m in METHODS}},indent=2));return
    if not args.run_batch:
        if out.exists() and any(out.iterdir()) and not args.resume:
            p.error('Output exists; use a new directory or --resume')
        prior=read(out/'launches.json').get('batch',{})
        if prior:
            try:
                if abs(psutil.Process(prior['pid']).create_time()-prior['create_time']) < .01:
                    p.error('Batch coordinator is already running')
            except (psutil.Error,KeyError):pass
        if args.resume and read(out/'batch_plan.json') != plan:
            p.error('Resume plan differs; use the original settings')
        out.mkdir(parents=True,exist_ok=True);atomic_json(out/'batch_plan.json',plan)
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--run-batch','--output-dir',str(out),
             '--splits-dir',str(args.splits_dir.resolve()),'--archive',str(args.archive.resolve()),
             '--workflow-pools',str(args.workflow_pools.resolve()),'--workflow-pool-method',args.workflow_pool_method,
             '--workers',str(args.workers)]
        if args.model:cmd+=['--model',args.model]
        env=method_environment(args,args.workflow_pool_method,os.environ)
        kwargs={'creationflags':subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
        with (out/'batch_coordinator.log').open('a',encoding='utf-8') as stream:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,**kwargs)
        atomic_json(out/'launches.json',{'batch':{'pid':child.pid,'create_time':psutil.Process(child.pid).create_time(),'command':cmd}})
        print(json.dumps({'started_pid':child.pid,'output':str(out),'methods':METHODS}));return
    state={'pid':os.getpid(),'status':'running','methods':{m:{'status':'queued'} for m in METHODS}}
    for method in METHODS:
        result=read(out/method/'aggregate.json')
        if result.get('evaluation_complete') and not result.get('smoke') and result.get('counts',{}).get('turns')==expected:
            state['methods'][method]={'status':'complete','reused':True}
            continue
        state['current_method']=method;state['methods'][method]={'status':'running'}
        atomic_json(out/'batch_state.json',state)
        with (out/f'{method}.log').open('a',encoding='utf-8') as stream:
            child=subprocess.Popen(command(args,method),cwd=ROOT,
                env=method_environment(args,args.workflow_pool_method,os.environ),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state['methods'][method]['pid']=child.pid;atomic_json(out/'batch_state.json',state)
            code=child.wait()
        result=read(out/method/'aggregate.json')
        status=('failed' if code or not result.get('evaluation_complete') else
                'completed_with_case_errors' if result.get('case_errors') else 'complete')
        state['methods'][method]={'status':status,'exit_code':code}
        atomic_json(out/'batch_state.json',state);collect(out)
    state.update(current_method=None,status='complete' if all(v['status']=='complete' for v in state['methods'].values()) else 'completed_with_errors')
    atomic_json(out/'batch_state.json',state);collect(out)


if __name__=='__main__':main()
