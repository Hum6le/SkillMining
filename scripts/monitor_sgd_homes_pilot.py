#!/usr/bin/env python3
"""Report Homes job queues, all worker completions, results, and memory."""
import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def lines(path):
    return sum(1 for line in path.open(encoding='utf-8') if line.rstrip().endswith('}')) if path.exists() else 0


def snapshot(root):
    import psutil
    jobs, launch, plan = read(root / 'jobs.json'), read(root / 'launch.json'), read(root / 'plan.json')
    scope = read(root / 'evaluation_scope.json').get('evaluation_targets', 'policy')
    call_plan = read(root / 'call_only_plan.json') if scope == 'call' else {}
    if call_plan:
        plan = call_plan
    methods = {}
    for method in ['awm', 'trace2skill', 'ours']:
        directory = root / method
        phases = list(directory.glob('parallel_cases/*'))
        latest = max(phases, key=lambda p: p.stat().st_ctime, default=None)
        train_files = [file for phase in phases if phase.name.startswith('train_rollout')
                       for file in phase.glob('worker*/case_*_usage.json')]
        eval_files = [file for phase in phases if phase.name.startswith('testing_')
                      for file in phase.glob('worker*/case_*_usage.json')]
        summary = read(directory / 'call_only_summary.json') if call_plan else read(directory / 'summary.json')
        methods[method] = {**jobs.get('jobs', {}).get(method, {'status': 'queued'}),
            'graph_nodes': len(read(directory / 'offline/subgraph.json').get('nodes', [])) if method == 'ours' else None,
            'latest_worker_phase': latest.name if latest else None,
            'train_worker_cases_finished': len(train_files), 'train_turns': plan.get('data_manifest', {}).get('train_turns'),
            'validation_worker_cases_finished': (summary.get('counts', {}).get('turns', len(eval_files))),
            'validation_turns': plan.get('data_manifest', {}).get('validation_turns'),
            'original_policy_worker_cases_finished': len(eval_files),
            'diagnoses': len(list(directory.glob('diagnosis_[0-9][0-9][0-9][0-9].json'))),
            'reflections': len(list(directory.glob('reflection_[0-9][0-9][0-9][0-9].json'))),
            'trace2skill_batches_finished': len(list(directory.glob('batches/batch_*/summary.json'))),
            'case_error_records': lines(directory / 'case_errors.jsonl'),
            'result': {k: summary[k] for k in ['action_set_exact', 'slot_exact_given_action', 'turn_joint_ast',
                'backend_call_metrics', 'execution_status'] if k in summary}}
    return {'root': str(root), 'supervisor_alive': psutil.pid_exists(launch.get('pid', -1)),
        'evaluation_targets': scope,
        'memory_available_gib': round(psutil.virtual_memory().available / 2**30, 2), 'methods': methods,
        'comparison_ready': (root / 'comparison.json').exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    root = args.root or Path(read(ROOT / 'outputs/latest_sgd_homes_run.json')['output_root'])
    while True:
        result = snapshot(root)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.watch or result['comparison_ready'] or not result['supervisor_alive']:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
