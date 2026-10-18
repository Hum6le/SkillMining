"""Offline transport simulation; no credentials, workflow service or LLM calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workflow_request_pool import WorkflowRequestPool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--requests', type=int, default=90)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--latency-seconds', type=float, default=.5)
    args = parser.parse_args()
    if min(args.requests, args.workers) < 1 or args.latency_seconds < 0: parser.error('Invalid simulation size')
    if args.output_dir.exists(): raise FileExistsError('Use a new simulation directory')
    args.output_dir.mkdir(parents=True)
    pool = WorkflowRequestPool(args.output_dir/'rates.sqlite', [f'mock-{i}' for i in range(5)], 3)
    start = time.perf_counter()
    def request(index):
        with pool.request('offline_mock'):
            time.sleep(args.latency_seconds)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(request, range(args.requests)))
    elapsed = time.perf_counter()-start
    with sqlite3.connect(pool.database) as db:
        rows = db.execute('SELECT workflow,started FROM workflow_requests ORDER BY started').fetchall()
    for workflow in pool.ids:
        stamps = [t for w,t in rows if w == workflow]
        assert all(stamps[i+3]-stamps[i] >= 1 for i in range(len(stamps)-3))
    span = rows[-1][1]-rows[0][1]
    report = {'simulation_only': True, 'real_api_calls': 0, 'requests': args.requests,
        'case_threads': args.workers, 'simulated_transport_seconds': args.latency_seconds,
        'elapsed_seconds': round(elapsed, 3), 'rolling_3_qps_per_id_verified': True,
        'steady_admission_qps_excluding_initial_burst': round(max(0,args.requests-15)/span,3) if span else None,
        'workflows': pool.summary()}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__': main()
