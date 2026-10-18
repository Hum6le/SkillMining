#!/usr/bin/env python3
"""Hidden, memory-gated Homes 10% run: Ours x4 alongside AWM -> Trace2Skill x4."""
import argparse
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.launch_multiwoz_hotel_pilot import background_python, credential_env, DEFAULT_ENV


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def commands(root, splits, model, workers):
    common = ['--domain', 'Homes', '--splits-dir', str(splits),
        '--archive', str(ROOT / 'data/eval/sgd/data.zip'), '--split', 'validation',
        '--eval-targets', 'policy',
        '--model', model, '--workers', str(workers)]
    return {method: [str(background_python()), '-u', str(ROOT / 'scripts' / (
            'run_sgd_trace2skill_pilot.py' if method == 'trace2skill' else 'run_sgd_domain_eval.py')),
            *common, '--output-dir', str(root / method),
            *([] if method == 'trace2skill' else ['--method', method]),
            *(['--ours-refinement-mode', 'standard', '--ours-cases-per-node', '4'] if method == 'ours' else [])]
            for method in ['ours', 'awm', 'trace2skill']}


def start(command, log, env=None):
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    with log.open('ab', buffering=0) as stream:
        return subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=stream, creationflags=flags)


def finish(root, jobs, plan):
    rows = {}
    for method in ['awm', 'trace2skill', 'ours']:
        path = root / method / 'summary.json'
        if path.exists():
            summary = json.loads(path.read_text(encoding='utf-8'))
            rows[method] = {key: summary.get(key) for key in ['action_set_exact',
                'action_micro_f1', 'slot_exact_given_action', 'turn_joint_ast',
                'api_joint_ast', 'counts', 'case_errors', 'execution_status', 'llm_usage']}
    result = {'data_manifest': plan['data_manifest'], 'jobs': jobs, 'methods': rows,
        'all_completed': all(j['status'] in ['completed', 'completed_with_errors'] for j in jobs.values())}
    write(root / 'comparison.json', result)
    lines = ['# SGD Homes 1/10', '', '| Method | Action exact | Slot exact given action | Turn joint | Status |',
        '|---|---:|---:|---:|---|']
    for method in ['awm', 'trace2skill', 'ours']:
        row = rows.get(method, {})
        values = [f"{100 * row[key]:.2f}%" if isinstance(row.get(key), (int, float)) else '-'
                  for key in ['action_set_exact', 'slot_exact_given_action', 'turn_joint_ast']]
        lines.append('| ' + ' | '.join([method, *values, jobs[method]['status']]) + ' |')
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def supervise(root):
    import psutil
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    jobs = {method: {'status': 'queued', 'workers': plan['workers']} for method in plan['commands']}
    running = {}
    while True:
        now = datetime.now(timezone.utc).isoformat()
        for method, process in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            path = root / method / 'summary.json'
            status = 'failed'
            if returncode == 0 and path.exists():
                summary = json.loads(path.read_text(encoding='utf-8'))
                if summary['counts']['turns'] == plan['data_manifest']['validation_turns']:
                    status = summary.get('execution_status', 'completed')
            jobs[method].update(status=status, returncode=returncode, finished_at=now)
            del running[method]
            print(f'{method}: {status}', flush=True)
        available = psutil.virtual_memory().available / 2**30
        capacity = plan.get('max_active_methods', 1 if plan['smoke'] else 2)
        if len(running) < capacity and available >= plan['memory_floor_gib']:
            for method in ['ours', 'awm', 'trace2skill']:
                if jobs[method]['status'] != 'queued':
                    continue
                if (method == 'trace2skill' and plan.get('schedule_mode', 'baseline_queue') == 'baseline_queue'
                        and jobs['awm']['status'] in ['queued', 'running']):
                    continue
                process = start(plan['commands'][method], root / (method + '.log'))
                running[method] = process
                jobs[method].update(status='running', pid=process.pid, started_at=now)
                print(f'{method}: started pid={process.pid}; workers={plan["workers"]}', flush=True)
                break  # Observe actual memory before starting the other lane.
        memory = {'time': now, 'available_gib': round(available, 3),
            'memory_percent': psutil.virtual_memory().percent,
            'running_methods': list(running), 'new_worker_limit': capacity * plan['workers']}
        with (root / 'memory.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(memory) + '\n')
        write(root / 'jobs.json', {'time': now, 'jobs': jobs, 'memory': memory})
        if all(job['status'] not in ['queued', 'running'] for job in jobs.values()):
            finish(root, jobs, plan)
            return
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--memory-floor-gib', type=float, default=1.1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--schedule', choices=['baseline_queue', 'parallel'], default='baseline_queue')
    parser.add_argument('--supervise', type=Path)
    args = parser.parse_args()
    if args.supervise:
        supervise(args.supervise.resolve())
        return
    if args.workers < 1:
        parser.error('workers must be positive')
    now = datetime.now(timezone(timedelta(hours=8)))
    label = 'smoke' if args.smoke else '10pct'
    root = (args.output_dir or ROOT / 'outputs' / f'sgd_homes_{label}_{now:%Y%m%d_%H%M%S}').resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True)
    splits = ROOT / 'data/eval/sgd/splits_10pct_seed42'
    from eval_tod.sgd_adapter import iter_family_dialogues, evaluation_turn_indices
    selected = {s: list(iter_family_dialogues(splits, 'Homes', s)) for s in ['train', 'validation']}
    if args.smoke:
        splits = root / 'smoke_splits'
        for split, rows in selected.items():
            row = rows[0]
            row['target_indices']['policy_turn_indices'] = row['target_indices']['policy_turn_indices'][:2 if split == 'train' else 1]
            selected[split] = [row]
            path = splits / 'Homes' / (split + '.jsonl.gz')
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'dataset': 'SGD', 'family': 'Homes', 'fraction': .1, 'seed': 42,
        'split': 'validation', 'smoke': args.smoke,
        'evaluation_targets': 'policy',
        **{s + '_dialogues': len(rows) for s, rows in selected.items()},
        **{s + '_turns': sum(len(d['target_indices']['policy_turn_indices']) for d in rows) for s, rows in selected.items()},
        'dialogue_ids': {s: [d['dialogue_id'] for d in rows] for s, rows in selected.items()},
        'file_sha256': {s: hashlib.sha256((splits / 'Homes' / (s + '.jsonl.gz')).read_bytes()).hexdigest() for s in selected},
        'protocol': 'Existing SGD optional API CALL plus unordered named acts; shared predict_policy and scorer. Ours uses standard refinement, four initial snippets per node.'}
    manifest['validation_policy_turns'] = manifest['validation_turns']
    manifest['validation_turns'] = sum(len(evaluation_turn_indices(d, 'policy')) for d in selected['validation'])
    if not args.smoke:
        assert [manifest[k] for k in ['train_dialogues', 'train_turns', 'validation_dialogues', 'validation_turns']] == [85, 507, 18, 115]
    env, model = credential_env(DEFAULT_ENV)
    env.update(PYTHONUNBUFFERED='1', SKILLMINING_ENABLE_THINKING='false')
    capacity = 1 if args.smoke else (3 if args.schedule == 'parallel' else 2)
    schedule_text = ('All three methods independently use four-worker pools, gated by available memory.'
        if args.schedule == 'parallel' else 'AWM then Trace2Skill share one four-worker lane; Ours has another four-worker lane.')
    plan = {'workers': args.workers, 'model': model, 'thinking': False, 'smoke': args.smoke,
        'schedule_mode': args.schedule, 'max_active_methods': capacity,
        'memory_floor_gib': args.memory_floor_gib, 'data_manifest': manifest,
        'schedule': schedule_text,
        'commands': commands(root, splits, model, args.workers)}
    write(root / 'plan.json', plan)
    snapshots = [*ROOT.glob('eval_tod/*.py'), *ROOT.glob('skill_mining/*.py'), ROOT / 'llm.py',
        *[ROOT / 'scripts' / name for name in ['run_sgd_domain_eval.py', 'run_sgd_trace2skill_pilot.py',
            'run_multiwoz_trace2skill_pilot.py', 'launch_sgd_homes_pilot.py', 'monitor_sgd_homes_pilot.py']],
        *ROOT.glob('Trace2Skill/skill_evolver/*.py')]
    hashes = {}
    for source in snapshots:
        if source.is_file():
            relative = source.relative_to(ROOT)
            target = root / 'code_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write(root / 'code_snapshot.json', hashes)
    process = start([str(background_python()), '-u', str(Path(__file__).resolve()), '--supervise', str(root)],
        root / 'supervisor.log', env)
    launch = {'pid': process.pid, 'output_root': str(root), 'workers_per_method': args.workers,
        'peak_new_workers': capacity * args.workers, 'started_at': now.isoformat(),
        'model': model, 'smoke': args.smoke, 'schedule': plan['schedule']}
    if not args.smoke:
        helper = start([str(background_python()), str(ROOT / 'scripts/keep_experiment_awake.py'),
            '--pid', str(process.pid), '--script', 'launch_sgd_homes_pilot.py'], root / 'awake.log')
        launch['awake_helper_pid'] = helper.pid
        write(ROOT / 'outputs/latest_sgd_homes_run.json', launch)
    write(root / 'launch.json', launch)
    if not args.smoke:
        monitor = start([str(background_python()), '-u', str(ROOT / 'scripts/monitor_sgd_homes_pilot.py'),
            '--root', str(root), '--watch'], root / 'monitor.log')
        launch['monitor_pid'] = monitor.pid
        write(root / 'launch.json', launch)
        write(ROOT / 'outputs/latest_sgd_homes_run.json', launch)
    print(json.dumps(launch, ensure_ascii=False))


if __name__ == '__main__':
    main()
