"""Migrate company settings to sequential methods, <=10 inflight and 2 QPS/ID."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import psutil
from workflow_request_pool import WorkflowRequestPool
from scripts.stop_sgd_server_batch import matching_processes


def configure(path):
    active = matching_processes(ROOT/'outputs')
    if active:
        raise RuntimeError(f'Stop existing SGD batches first; active PIDs: {[p.pid for p in active]}')
    path = Path(path).resolve()
    original = path.read_bytes()
    config = json.loads(original.decode('utf-8-sig'))
    for method in ['awm', 'trace2skill', 'ours']:
        ids = config['methods'][method]['workflow_ids']
        if not 1 <= len(ids) <= 5 or any(not isinstance(i,str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Each method requires one to five unique workflow IDs')
        config['methods'][method].update(case_threads=10, domain_workers=1)
    database = Path(config.get('rate_database', 'outputs/workflow_rate.sqlite'))
    if not database.is_absolute(): database = ROOT/database
    # Do not retire requests owned by another still-running caller.
    if database.is_file():
        with sqlite3.connect(database) as db:
            columns = {r[1] for r in db.execute('PRAGMA table_info(workflow_requests)')}
            if {'owner_pid','owner_created'} <= columns:
                for pid,created in db.execute('SELECT DISTINCT owner_pid,owner_created FROM workflow_requests '
                    'WHERE status="inflight" AND owner_pid IS NOT NULL'):
                    try:
                        process = psutil.Process(pid)
                        if process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE:
                            raise RuntimeError(f'A request owner is still running: PID {pid}')
                    except psutil.NoSuchProcess:
                        pass
    config.update(qps_per_workflow=2, min_interval_seconds=.55, max_inflight=10, max_active_workflows=5)
    pools = [WorkflowRequestPool(database, config['methods'][m]['workflow_ids'], 2,
        min_interval_seconds=.55, max_inflight=10, max_active_workflows=5)
        for m in ['awm','trace2skill','ours']]
    ids = sorted({w for pool in pools for w in pool.ids})
    with pools[0].connect() as db:
        db.execute('BEGIN IMMEDIATE')
        placeholders = ','.join('?' for _ in ids)
        retired = db.execute('UPDATE workflow_requests SET status="abandoned",finished=?, '
            'error_kind="legacy_batch_stopped" WHERE status="inflight" AND owner_pid IS NULL '
            f'AND workflow IN ({placeholders})', (time.time(), *ids)).rowcount
    shared = pools[0].global_summary()
    if shared['current_inflight']:
        raise RuntimeError('Other requests still occupy the shared database; stop their callers first')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = path.with_name(path.name+'.backup_'+stamp)
    backup.write_bytes(original)
    temporary = path.with_name(path.name+'.updating')
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)
    return {'config':str(path), 'backup':str(backup), 'method_execution':'sequential',
        'qps_per_workflow':2, 'min_interval_seconds':.55, 'max_inflight':10,
        'max_active_workflows':5, 'retired_unowned_legacy_requests':retired, 'model_calls':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sgd_workflows.json')
    args = parser.parse_args()
    print(json.dumps(configure(args.config), ensure_ascii=False, indent=2))
