"""Request-level workflow routing and host-wide admission limits.

One SQLite database on LOCAL disk is shared by all participating processes.
It stores timing/status metadata only, never prompts, responses or credentials.
The pooled llm_new transport sends exactly one HTTP attempt per admission;
every retry obtains a new permit. Legacy workspace helpers are not called.
"""
from contextlib import contextmanager
from bisect import bisect_right
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time

_LOCAL = threading.local()
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


class WorkflowRequestPool:
    def __init__(self, database, workflow_ids, qps=3, *, clock=time.time, sleep=time.sleep):
        self.database = Path(database).resolve()
        with _LOCKS_GUARD:
            self.lock = _LOCKS.setdefault((os.getpid(), str(self.database)), threading.RLock())
        self.ids = tuple(workflow_ids)
        if not self.ids or len(set(self.ids)) != len(self.ids) or any(not str(i).strip() for i in self.ids):
            raise ValueError('Workflow IDs must be nonempty and unique')
        if not isinstance(qps, int) or qps < 1:
            raise ValueError('Workflow QPS must be a positive integer')
        self.qps, self.clock, self.sleep = qps, clock, sleep
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS workflow_limits '
                       '(workflow TEXT PRIMARY KEY, qps INTEGER NOT NULL, starts TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS workflow_requests '
                       '(id INTEGER PRIMARY KEY, workflow TEXT NOT NULL, started REAL NOT NULL, '
                       'finished REAL, status TEXT NOT NULL, call_tag TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS workflow_started ON workflow_requests(workflow, started)')
            for workflow in self.ids:
                db.execute('INSERT OR IGNORE INTO workflow_limits VALUES (?, ?, ?)', (workflow, qps, '[]'))
                if db.execute('SELECT qps FROM workflow_limits WHERE workflow=?', (workflow,)).fetchone()[0] != qps:
                    raise ValueError('The shared database already has a different QPS for this workflow')

    @contextmanager
    def connect(self):
        # A connection is never shared across threads or inherited across fork.
        # Avoid hundreds of threads competing in SQLite's busy backoff loop.
        # SQLite still arbitrates between independent method processes.
        with self.lock:
            db = sqlite3.connect(self.database, timeout=30)
            try:
                db.execute('PRAGMA synchronous=NORMAL')
                with db:
                    yield db
            finally:
                db.close()

    def acquire(self, call_tag='chat'):
        while True:
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                now = self.clock()
                candidates = []
                for workflow in self.ids:
                    stamps = json.loads(db.execute('SELECT starts FROM workflow_limits WHERE workflow=?', (workflow,)).fetchone()[0])
                    # Small boundary margin protects a rolling one-second quota.
                    recent = [stamp for stamp in stamps if stamp > now-1.005]
                    due = recent[-self.qps]+1.005 if len(recent) >= self.qps else now
                    candidates.append((due, stamps[-1] if stamps else -math.inf, workflow, recent))
                due, _, workflow, recent = min(candidates, key=lambda c: (c[0], c[1], self.ids.index(c[2])))
                if due <= now:
                    recent.append(now)
                    db.execute('UPDATE workflow_limits SET starts=? WHERE workflow=?',
                               (json.dumps(recent[-self.qps:]), workflow))
                    cursor = db.execute('INSERT INTO workflow_requests(workflow,started,status,call_tag) VALUES (?,?,?,?)',
                                        (workflow, now, 'inflight', call_tag))
                    return workflow, cursor.lastrowid
                delay = due-now
            # Never hold a database transaction while waiting or running HTTP.
            self.sleep(max(.001, delay))

    def finish(self, request_id, status):
        with self.connect() as db:
            db.execute('UPDATE workflow_requests SET finished=?,status=? WHERE id=?',
                       (self.clock(), status, request_id))

    @contextmanager
    def request(self, call_tag='chat'):
        workflow, request_id = self.acquire(call_tag)
        try:
            yield workflow
        except BaseException:
            self.finish(request_id, 'failed')
            raise
        else:
            self.finish(request_id, 'completed')

    def summary(self, window=60):
        now = self.clock()
        result = {}
        with self.connect() as db:
            for workflow in self.ids:
                count, completed, failed, inflight, latency = db.execute(
                    'SELECT count(*),sum(status="completed"),sum(status="failed"),'
                    'sum(status="inflight"),avg(finished-started) '
                    'FROM workflow_requests WHERE workflow=? AND started>?', (workflow, now-window)).fetchone()
                first = db.execute('SELECT min(started) FROM workflow_requests WHERE workflow=?', (workflow,)).fetchone()[0]
                duration = min(window, max(0, now-first)) if first is not None else 0
                stamps = [r[0] for r in db.execute('SELECT started FROM workflow_requests '
                          'WHERE workflow=? AND started>? ORDER BY started', (workflow, now-window))]
                peak = max((i-bisect_right(stamps, stamp-1)+1 for i, stamp in enumerate(stamps)), default=0)
                result[workflow] = {'admitted': count, 'completed': completed or 0, 'failed': failed or 0,
                    'unfinished': inflight or 0, 'window_seconds': round(duration, 3),
                    'admitted_qps': round(count/duration, 3) if duration >= 1 else None,
                    'peak_admissions_in_rolling_second': peak,
                    'mean_transport_seconds': round(latency, 3) if latency is not None else None,
                    'qps_limit': self.qps}
        return result


def from_environment():
    ids = [item.strip() for item in os.getenv('SKILLMINING_WORKFLOW_POOL_IDS', '').split(',') if item.strip()]
    if not ids:
        return None
    database = os.getenv('SKILLMINING_WORKFLOW_RATE_DB', '').strip()
    if not database:
        raise ValueError('Workflow pooling requires one shared SKILLMINING_WORKFLOW_RATE_DB path')
    key = (os.getpid(), database, tuple(ids), int(os.getenv('SKILLMINING_WORKFLOW_QPS', '3')))
    if getattr(_LOCAL, 'key', None) != key:
        _LOCAL.pool = WorkflowRequestPool(database, ids, key[-1])
        _LOCAL.key = key
    return _LOCAL.pool
