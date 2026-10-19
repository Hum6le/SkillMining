"""Portable background SGD batch: three methods, official seen-schema splits."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.multiwoz_aligned import atomic_json


def environment(path, require_key):
    from dotenv import dotenv_values
    values = dotenv_values(path) if path.is_file() else {}
    env = os.environ.copy()
    key = values.get('OPENAI_API_KEY') or values.get('SHUBIAOBIAO_API_KEY') or env.get('OPENAI_API_KEY')
    base = values.get('OPENAI_BASE_URL') or values.get('MIND2WEB_API_BASE_URL') or env.get('OPENAI_BASE_URL')
    if require_key and (not key or not base):
        raise ValueError(f'Configure OPENAI_API_KEY and OPENAI_BASE_URL in {path}')
    if key: env['OPENAI_API_KEY'] = key
    if base: env['OPENAI_BASE_URL'] = base
    env.update(PYTHONUNBUFFERED='1', PYTHONUTF8='1', SKILLMINING_ENABLE_THINKING='false',
        SKILLMINING_LLM_TIMEOUT_SECONDS='90', SKILLMINING_LLM_MAX_RETRIES='0',
        SKILLMINING_RETRY_EMPTY_LLM='1')
    return env


def workflow_settings(args, method):
    if getattr(args, 'provider', 'openai') != 'workflow':
        return None
    config = json.loads(args.workflow_pools.read_text(encoding='utf-8'))
    selected = config['methods'][method]
    ids = selected['workflow_ids']
    if not ids or any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Each method needs unique nonempty workflow IDs')
    qps, workers = config.get('qps_per_workflow', 3), selected.get('case_threads', 160)
    if not isinstance(qps, int) or qps < 1 or not isinstance(workers, int) or workers < 2:
        raise ValueError('Workflow QPS must be positive; threaded mode needs at least two threads')
    database = Path(config.get('rate_database', 'outputs/workflow_rate.sqlite'))
    if not database.is_absolute(): database = ROOT/database
    domains = selected.get('domain_workers', 1)
    if not isinstance(domains, int) or domains < 1: raise ValueError('domain_workers must be positive')
    return {'workflow_ids': ids, 'qps': qps, 'workers': workers, 'domain_workers': domains,
            'database': str(database.resolve())}


def method_environment(args, method, env):
    env = env.copy()
    settings = workflow_settings(args, method)
    if settings:
        env.update(SKILLMINING_LLM_BACKEND='workflow', SKILLMINING_SGD_EXECUTOR='thread',
            SKILLMINING_WORKFLOW_POOL_IDS=','.join(settings['workflow_ids']),
            SKILLMINING_WORKFLOW_ID=settings['workflow_ids'][0],
            SKILLMINING_WORKFLOW_QPS=str(settings['qps']),
            SKILLMINING_WORKFLOW_RATE_DB=settings['database'])
    return env


def phase_commands(args, method):
    index = json.loads((args.splits_dir/'INDEX.json').read_text(encoding='utf-8'))
    if index.get('profile', {}).get('schema_scope') != 'seen':
        raise ValueError('This batch requires the explicit seen-schema profile')
    validation = index['eligible_families']['validation']
    test = index['eligible_families']['test']
    if args.domains:
        requested = args.domains.split(',')
        if any(f not in index['train_families'] for f in requested): raise ValueError('Unknown family')
        validation = [f for f in validation if f in requested]
        test = [f for f in test if f in requested]
    shared, extra = [f for f in test if f in validation], [f for f in test if f not in validation]
    workers = args.ours_workers if method == 'ours' else args.baseline_workers
    workflow = workflow_settings(args, method)
    if workflow: workers = workflow['workers']
    phases = [('validation', validation, False), ('test_reuse', shared, True), ('test_new', extra, False)]
    if args.stage == 'validation': phases = phases[:1]
    commands = []
    for phase, domains, frozen in phases:
        if not domains: continue
        output = args.output_dir/method/phase
        command = [sys.executable, '-u', str(ROOT/'scripts/run_full_sgd_experiments.py'),
            '--method', method, '--domains', ','.join(domains), '--execution-protocol', 'aligned',
            '--eval-targets', 'policy', '--split', 'validation' if phase == 'validation' else 'test',
            '--splits-dir', str(args.splits_dir.resolve()), '--archive', str(args.archive.resolve()),
            '--model', args.model, '--parallelism', 'within-domain', '--workers', str(workers)]
        if workflow:
            command += ['--workflow-ids', ','.join(workflow['workflow_ids']),
                        '--domain-workers', str(workflow['domain_workers'])]
        command += ['--resume-run' if args.resume and (output/'workflow_load_plan.json').exists() else '--output-dir', str(output.resolve())]
        if frozen:
            name = 'trace2skill' if method == 'trace2skill' else method
            command += ['--'+name+'-resource-root', str((args.output_dir/method/'validation'/method).resolve())]
        if method == 'ours':
            command += ['--ours-refinement-mode', 'standard', '--ours-cases-per-node', '4']
        commands.append({'phase': phase, 'families': domains, 'output': str(output),
                         'frozen_training_resources': frozen, 'command': command})
    return commands


def run_method(args, env):
    env = method_environment(args, args.run_method, env)
    phases = phase_commands(args, args.run_method)
    state = {'method': args.run_method, 'phases': []}
    path = args.output_dir/args.run_method/'state.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    for phase in phases:
        state.update(status='running', current_phase=phase['phase'])
        atomic_json(path, state)
        log = path.parent/(phase['phase']+'.log')
        with log.open('ab') as stream:
            result = subprocess.run(phase['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT)
        state['phases'].append({**phase, 'returncode': result.returncode})
        if result.returncode:
            state['status'] = 'failed_or_case_errors'
            atomic_json(path, state)
            return result.returncode
    state['status'] = 'completed'
    atomic_json(path, state)
    return 0


def status(root):
    result = {}
    for method in ['awm', 'trace2skill', 'ours']:
        path = root/method/'state.json'
        result[method] = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status': 'not_started'}
        result[method]['completed_domain_tasks'] = len(list((root/method).glob('*/*/*/summary.json')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits-dir', type=Path, default=ROOT/'data/eval/sgd/protocol_10pct_seed42_seen_schema_v1')
    parser.add_argument('--archive', type=Path, default=ROOT/'data/eval/sgd/data.zip')
    parser.add_argument('--env-file', type=Path, default=ROOT/'configs/sgd_server.env')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='qwen3.6-flash')
    parser.add_argument('--ours-workers', type=int, default=24)
    parser.add_argument('--baseline-workers', type=int, default=12)
    parser.add_argument('--provider', choices=['openai', 'workflow'], default='openai')
    parser.add_argument('--workflow-pools', type=Path, default=ROOT/'configs/sgd_workflows.json')
    parser.add_argument('--stage', choices=['validation', 'all'], default='validation')
    parser.add_argument('--domains', help='Optional comma-separated subset; omit for all eligible families')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--run-method', choices=['awm', 'trace2skill', 'ours'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.status:
        print(json.dumps(status(args.output_dir), ensure_ascii=False, indent=2)); return
    if min(args.ours_workers, args.baseline_workers) < 1: parser.error('Worker counts must be positive')
    env = environment(args.env_file, not args.dry_run and args.provider == 'openai')
    if args.run_method:
        raise SystemExit(run_method(args, env))
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError('Use a new output directory or --resume')
    if args.resume and (args.output_dir/'launches.json').exists():
        import psutil
        previous = json.loads((args.output_dir/'launches.json').read_text(encoding='utf-8'))
        for job in previous.values():
            try:
                command = psutil.Process(job['pid']).cmdline()
                if '--run-method' in command and str(args.output_dir.resolve()) in command:
                    raise RuntimeError('A coordinator for this batch is still running; use --status')
            except psutil.NoSuchProcess:
                pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'model': args.model,
        'profile_sha256': __import__('hashlib').sha256((args.splits_dir/'INDEX.json').read_bytes()).hexdigest(),
        'provider': args.provider,
        'workflow_pools': {m: workflow_settings(args, m) for m in ['awm', 'trace2skill', 'ours']},
        'workers_total': sum(workflow_settings(args, m)['workers']*workflow_settings(args, m)['domain_workers'] if args.provider == 'workflow' else
            args.ours_workers if m == 'ours' else args.baseline_workers for m in ['awm', 'trace2skill', 'ours']),
        'methods': {m: phase_commands(args, m) for m in ['awm', 'trace2skill', 'ours']},
        'scope': 'One family at a time per method; parallel cases; three method coordinators; sequential native skill writes.'}
    atomic_json(args.output_dir/'batch_plan.json', plan)
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'model_calls': 0, 'plan': str(args.output_dir/'batch_plan.json'),
            'workers_total': plan['workers_total'], 'tasks': {m:sum(len(p['families']) for p in ps) for m, ps in plan['methods'].items()}})); return
    launches = {}
    for method in plan['methods']:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-method', method,
            '--splits-dir', str(args.splits_dir.resolve()), '--archive', str(args.archive.resolve()),
            '--env-file', str(args.env_file.resolve()), '--output-dir', str(args.output_dir.resolve()),
            '--model', args.model, '--ours-workers', str(args.ours_workers),
            '--baseline-workers', str(args.baseline_workers), '--stage', args.stage]
        command += ['--provider', args.provider, '--workflow-pools', str(args.workflow_pools.resolve())]
        if args.domains: command += ['--domains', args.domains]
        if args.resume: command += ['--resume']
        kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
        with (args.output_dir/(method+'_coordinator.log')).open('ab') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=method_environment(args, method, env), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, **kwargs)
        launches[method] = {'pid': process.pid, 'command': command}
    atomic_json(args.output_dir/'launches.json', launches)
    print(json.dumps({'started': launches, 'output': str(args.output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
