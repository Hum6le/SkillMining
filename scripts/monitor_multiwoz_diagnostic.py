#!/usr/bin/env python3
"""Show live diagnostic progress from files, without API requests."""
import argparse
import json
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def snapshot(root):
    progress = read(root / 'progress.json')
    result = {'root': str(root), 'progress': progress,
        'compiled_nodes': sum(len(read(p).get('cards', [])) for p in (root / 'offline/card_compile').glob('batch_*.json')),
        'evaluation_turns': len(list((root / 'cases').glob('case_*.json'))),
        'observation_cache_entries': len(list((root / 'observations').glob('*.json'))),
        'failure': read(root / 'run_failed.json'), 'results_ready': (root / 'summary.json').exists()}
    launch = read(root / 'launch.json')
    if launch.get('pid'):
        import psutil
        result['process_alive'] = psutil.pid_exists(launch['pid'])
        result['pid'] = launch['pid']
    if result['results_ready']:
        summary = read(root / 'summary.json')
        result['metrics'] = {key: summary[key] for key in ('action_set_exact', 'slot_exact_given_action', 'turn_joint_ast')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=15)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be at least 1 second')
    pointer = Path(__file__).resolve().parent.parent / 'outputs/latest_multiwoz_slot_diagnostic.json'
    root = args.root or Path(read(pointer)['output_root'])
    while True:
        result = snapshot(root)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.watch or result['results_ready'] or result['failure'] or result.get('process_alive') is False:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
