#!/usr/bin/env python3
"""Run local SGD baselines in two queues or three independent four-thread lanes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_tod.multiwoz_aligned import atomic_json
from eval_tod.sgd_local_config import configure, DEFAULT_ENV
from scripts.run_sgd_extra_baselines import families

KEYS = ['action_set_exact','action_micro_f1','slot_exact_given_action',
        'turn_joint_ast','call_gate_f1','api_joint_ast']
HISTORICAL = {
    'awm': [.4262691377921031,.5067264573991032,.34215500945179583,
            .14585012087026591,.5056497175141242,.18501170960187355],
    'trace2skill': [.37711522965350525,.4764525993883792,.24786324786324787,
                   .09347300564061241,.44170096021947874,.22716627634660422],
    'ours': [.5149073327961321,.5852932420363204,.5602503912363067,
             .2884770346494762,.6099110546378653,.4028103044496487]}
NAMES = {'react':'ReAct','reflexion':'Reflexion','skill_disco':'Skill-DisCo',
         'expel':'ExpeL','awm':'AWM','trace2skill':'Trace2Skill','ours':'Ours'}
LANES = [['react','reflexion'], ['skill_disco','expel']]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def running_method(method, output):
    """Adopt an exact existing run without interrupting its in-flight calls."""
    import psutil
    matches = []
    for process in psutil.process_iter(['cmdline']):
        cmd = process.info['cmdline'] or []
        if not any(Path(arg).name == 'run_sgd_extra_baselines.py' for arg in cmd):
            continue
        try:
            same = (cmd[cmd.index('--method')+1] == method and
                    Path(cmd[cmd.index('--output-dir')+1]).resolve() == output.resolve())
        except (ValueError, IndexError):
            continue
        if same:
            matches.append(process)
    if len(matches) > 1:
        raise RuntimeError(f'Multiple live processes for {method}: {[p.pid for p in matches]}')
    return matches[0] if matches else None


def collect(root):
    methods = {}
    overrides = read(root/'method_run_overrides.json', {})
    for method, values in HISTORICAL.items():
        methods[method] = {**dict(zip(KEYS, values)), 'status':'historical_reported',
            'source':'author_supplied_company_10pct_aggregate', 'turns':1241,
            'model':None, 'llm_usage':None,
            'note':'Raw predictions, model metadata and usage were not supplied with these historical scores.'}
    for method in ['react','reflexion','skill_disco','expel']:
        directory = Path(overrides[method]['output_dir']) if method in overrides else root/method
        result = read(directory/'aggregate.json', {})
        state = read(directory/'progress.json', {})
        complete = result.get('evaluation_complete', False) and not result.get('smoke', False)
        methods[method] = {**result, 'status': ('complete' if complete else
            'failed' if (directory/'fatal_error.json').exists() else 'running' if state else 'queued'),
            'source':'local_shubiaobiao_qwen3.6-flash', 'progress':state,
            'output_dir':str(directory.resolve())}
        if method in overrides:
            methods[method]['rerun'] = overrides[method]
            methods[method]['previous_run'] = read(root/method/'aggregate.json', {})
        if complete and result['counts']['turns'] != 1241:
            methods[method]['status'] = 'protocol_mismatch'
    report = {'dataset':'1/10 SGD', 'split':'seen-schema official validation',
        'families':9, 'train_policy_turns':8269, 'validation_policy_turns':1241,
        'methods':methods, 'updated_at':now(),
        'note':'New runs share aligned heads/payload executor and all policy targets. '
               'Historical values are author-reported aggregates; model identity and raw-case alignment '
               'cannot be re-audited from those aggregates. Slot exact is conditional on exact actions.'}
    atomic_json(root/'comparison.json', report)
    lines = ['# 1/10 SGD comparison', '',
        'Seen-schema validation: 9 families; 8,269 TRAIN policy turns; 1,241 validation policy turns.', '',
        '| Method | Status | Action exact | Action micro F1 | Slot exact given action | Turn joint | Call gate F1 | API joint |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for method in ['react','reflexion','awm','expel','trace2skill','skill_disco','ours']:
        row = methods[method]
        scores = [f'{100*row[k]:.2f}%' if k in row and row['status'] in ['complete','historical_reported']
                  else '--' for k in KEYS]
        lines.append('| '+ ' | '.join([NAMES[method],row['status']]+scores)+' |')
    lines += ['', 'Slot exact is conditioned on the entire action-head set being correct. '
        'Turn joint requires the complete action set and payload, including non-call policy turns.', '',
        'Historical AWM / Trace2Skill / Ours scores were supplied by the author from the company 10% run. '
        'Their raw predictions, model metadata and token accounting are unavailable locally; '
        'they are not presented as re-executed or newly audited results.', '',
        'Local totals include all recorded attempts and retries. Error cases remain in metric denominators.', '']
    for method, entry in overrides.items():
        lines += [f"{NAMES[method]} rerun: {entry.get('note', '')} Output: {entry['output_dir']}.", '']
    (root/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--env-file',type=Path,default=DEFAULT_ENV)
    p.add_argument('--splits-dir',type=Path,default=ROOT/'data/eval/sgd/protocol_10pct_seed42_seen_schema_v1')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--collect-only',action='store_true')
    p.add_argument('--parallel-methods',action='store_true',
                   help='Three independent lanes: Reflexion, Skill-DisCo and ExpeL (4 threads each)')
    p.add_argument('--adopt-running',action='store_true',
                   help='Adopt matching existing method processes instead of launching duplicates')
    p.add_argument('--reflexion-workers',type=int,default=4)
    args=p.parse_args()
    if not 1 <= args.reflexion_workers <= 8:
        p.error('--reflexion-workers must be between 1 and 8')
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.collect_only:
        collect(out); return
    import psutil
    config=configure(args.env_file)
    chosen=families(args.splits_dir)
    assert len(chosen)==9, chosen
    index=read(args.splits_dir/'INDEX.json')
    assert sum(index['families'][f]['train']['policy_turns'] for f in chosen)==8269
    assert sum(index['families'][f]['validation']['policy_turns'] for f in chosen)==1241
    lanes = [['react','reflexion'], ['skill_disco'], ['expel']] if args.parallel_methods else LANES
    overrides = read(out/'method_run_overrides.json', {})
    method_dirs = {m: Path(overrides[m]['output_dir']) if m in overrides else out/m
                   for m in ['react','reflexion','skill_disco','expel']}
    method_workers = {m: (args.reflexion_workers if m == 'reflexion' else 4)
                      for m in ['react','reflexion','skill_disco','expel']}
    concurrent = sum(max([method_workers[m] for m in lane
                         if not read(method_dirs[m]/'aggregate.json',{}).get('evaluation_complete')] or [0])
                     for lane in lanes)
    plan={'lanes':lanes,'threads_per_method':4,'workers_by_method':method_workers,
        'max_concurrent_requests':concurrent,
        'config':config,'families':chosen,'splits_dir':str(args.splits_dir.resolve()),
        'train_policy_turns':8269,'validation_policy_turns':1241,
        'available_memory_gb':round(psutil.virtual_memory().available/2**30,2)}
    if args.preflight:
        print(json.dumps(plan,ensure_ascii=False,indent=2)); return
    # Exclusive controller lease; stale files do not authorize a second live controller.
    lock=out/'controller.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        owner=read(lock,{})
        try:
            alive=abs(psutil.Process(owner['pid']).create_time()-owner['create_time'])<.01
        except (psutil.Error,KeyError):
            alive=False
        if alive:
            raise RuntimeError(f"Controller is already running: PID {owner['pid']}")
        lock.unlink()
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w',encoding='utf-8') as stream:
        json.dump({'pid':os.getpid(),'create_time':psutil.Process().create_time()},stream)
    state={'status':'running','pid':os.getpid(),'started_at':now(),'plan':plan,
           'lanes':[{'methods':lane,'status':'queued'} for lane in lanes]}
    mutex=threading.Lock()
    def persist():
        state['updated_at']=now()
        atomic_json(out/'batch_state.json',state)
    persist()
    collect(out)
    def lane(index):
        records=[]
        for method in lanes[index]:
            method_dir = method_dirs[method]
            complete = read(method_dir/'aggregate.json', {})
            if complete.get('evaluation_complete') and not complete.get('smoke'):
                records.append({'method':method,'exit_code':0,'finished_at':now(),
                                'reused_complete':True})
                continue
            adopted = running_method(method, method_dir)
            if adopted is not None and not args.adopt_running:
                raise RuntimeError(f'{method} is already running: PID {adopted.pid}; use --adopt-running')
            cmd=[sys.executable,str(ROOT/'scripts/run_sgd_extra_baselines.py'),
                '--method',method,'--output-dir',str(method_dir),'--workers',str(method_workers[method]),
                '--env-file',str(args.env_file.resolve()),'--splits-dir',str(args.splits_dir.resolve())]
            if method in overrides:
                cmd += ['--comparison-root',str(out)]
            log=out/(f'{method}.rerun.log' if method in overrides else f'{method}.log')
            with log.open('a',encoding='utf-8') as stream:
                child=adopted or subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8'},
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                with mutex:
                    state['lanes'][index]={'methods':lanes[index],'current_method':method,
                        'pid':child.pid,'status':'running','records':records,'started_at':now(),
                        'adopted':adopted is not None, 'workers':method_workers[method],
                        'output_dir':str(method_dir)}
                    persist()
                code=child.wait()
                if code is None:
                    code=0 if read(method_dir/'aggregate.json',{}).get('evaluation_complete') else 1
            records.append({'method':method,'exit_code':code,'finished_at':now()})
            with mutex:
                collect(out)
        with mutex:
            state['lanes'][index].update(status='complete',records=records)
            persist()
        return records
    try:
        with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
            futures=[pool.submit(lane,index) for index in range(len(lanes))]
            while not all(f.done() for f in futures):
                wait(futures,timeout=10)
                with mutex:
                    persist()
            records=[record for f in futures for record in f.result()]
        state['status']='complete' if all(r['exit_code']==0 for r in records) else 'completed_with_failures'
        state['finished_at']=now()
        persist(); collect(out)
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
