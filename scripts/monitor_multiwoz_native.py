#!/usr/bin/env python3
"""Progress for the native-policy standard run; no API calls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def line_count(path):
    if not path.exists():
        return 0
    with path.open(encoding='utf-8') as stream:
        return sum(1 for line in stream if line.rstrip().endswith('}'))


def snapshot(root):
    import psutil
    launch, manifest, progress = read(root / 'launch.json'), read(root / 'data_manifest.json'), read(root / 'progress.json')
    completed = {}
    parallel_roots = [root / 'parallel_cases', root / 'offline/parallel_cases']
    for phase in ('shared_db_planning', 'offline_compile', 'train_rollout', 'online_diagnosis', 'testing'):
        folders = [folder for parent in parallel_roots for folder in parent.glob(phase + '_*')]
        latest = max(folders, key=lambda folder: folder.stat().st_ctime, default=None)
        completed[phase] = len(list(latest.glob('worker*/case_*_usage.json'))) if latest else 0
    n = line_count(root / 'ours_training_turns.jsonl')
    train_complete = (root / 'ours_training.json').exists()
    layout = read(root / 'execution_layout.json')
    result = {'root': str(root), 'pid': launch.get('pid'), 'process_alive': psutil.pid_exists(launch.get('pid', -1)),
        'workers': layout.get('case_workers', launch.get('workers')), 'progress': progress,
        'compiled_nodes': len(list((root / 'offline/card_compile').glob('batch_*.json'))),
        'graph_nodes': len(read(root / 'offline/subgraph.json').get('nodes', [])),
        'observation_cache_entries': len(list(Path(manifest.get('observation_cache', root / 'observations')).glob('*.json'))),
        'reused_train_rollouts': n if manifest.get('source_run') else 0,
        'train_rollouts_recorded': n, 'train_rollouts_finished_in_workers': completed['train_rollout'],
        'train_turns': manifest.get('available_train_turns'),
        'diagnosis_reports': len(list(root.glob('diagnosis_[0-9][0-9][0-9][0-9].json'))),
        'reflection_groups_processed': len(list(root.glob('reflection_[0-9][0-9][0-9][0-9].json'))),
        'standard_refine_completed': train_complete,
        'evaluation_turns_recorded': line_count(root / 'predictions.jsonl'), 'eval_turns': manifest.get('eval_turns'),
        'worker_cases_completed': completed, 'failure': read(root / 'run_failed.json'),
        'results_ready': (root / 'protocol_audit.json').exists()}
    if launch.get('started_at'):
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(launch['started_at'])).total_seconds()
        result['elapsed_minutes'] = round(elapsed / 60, 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=20)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('Interval must be at least one second')
    root = args.root or Path(read(Path(__file__).resolve().parent.parent / 'outputs/latest_multiwoz_native_run.json')['output_root'])
    while True:
        result = snapshot(root)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.watch or not result['process_alive'] or result['failure'] or result['results_ready']:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
