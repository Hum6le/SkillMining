#!/usr/bin/env python3
"""Read paired TRAIN probe progress without model calls."""
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
    launch = read(root / 'launch.json')
    result = {'root': str(root), 'progress': read(root / 'progress.json'),
        'compiled_nodes': len(list((root / 'card_compile').glob('batch_*.json'))),
        'prepared_references': len(list((root / 'references').glob('case_*.json'))),
        'candidate_plans': len(list((root / 'candidate_plans').glob('case_*.json'))),
        'arm_turns': {arm: len(list((root / arm / 'cases').glob('case_*.json'))) for arm in
            ('A_original', 'B_payload_only', 'C_joint_patterns', 'D_scoped_reference')},
        'failure': read(root / 'run_failed.json'), 'results_ready': (root / 'summary.json').exists()}
    if launch.get('pid'):
        import psutil
        result['process_alive'] = psutil.pid_exists(launch['pid'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=15)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('Interval must be at least one second')
    pointer = Path(__file__).resolve().parent.parent / 'outputs/latest_multiwoz_slot_pair_probe.json'
    root = args.root or Path(read(pointer)['output_root'])
    while True:
        result = snapshot(root)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.watch or result['results_ready'] or result['failure'] or result.get('process_alive') is False:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
