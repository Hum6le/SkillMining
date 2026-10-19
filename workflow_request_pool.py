"""Request-level workflow routing and host-wide admission limits.

One SQLite database on LOCAL disk is shared by all participating processes.
It stores timing/status metadata only, never prompts, responses or credentials.
The pooled transport sends exactly one HTTP attempt per admission; every retry
obtains a new permit. Admission is paced per ID, with shared rate-limit backoff.
This controls client starts; a remote receive-time quota still needs monitoring.
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
import psutil

_LOCAL = threading.local()
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


class WorkflowRequestPool:
    def __init__(self, database, workflow_ids, qps=3, *, clock=time.time, sleep=time.sleep,
                 min_interval_seconds=None, max_inflight=10, max_active_workflows=5):
        self.database = Path(database).resolve()
        with _LOCKS_GUARD:
            self.lock = _LOCKS.setdefault((os.getpid(), str(self.database)), threading.RLock())
        self.ids = tuple(workflow_ids)
        if not self.ids or len(set(self.ids)) != len(self.ids) or any(not str(i).strip() for i in self.ids):
            raise ValueError('Workflow IDs must be nonempty and unique')
        if not isinstance(qps, int) or qps < 1:
            raise ValueError('Workflow QPS must be a positive integer')
        interval = 1.08/qps if min_interval_seconds is None else min_interval_seconds
        if (isinstance(interval, bool) or not isinstance(interval, (int, float))
                or not math.isfinite(interval) or interval < 1.08/qps-1e-12):
            raise ValueError('Workflow interval must be finite and at least 1.08 / QPS')
        interval = max(interval, 1.08/qps)
        if (type(max_inflight) is not int or not 1 <= max_inflight <= 10
                or type(max_active_workflows) is not int or not 1 <= max_active_workflows <= 5):
            raise ValueError('Shared model limits require 1..10 inflight requests and 1..5 active workflows')
        self.owner = (os.getpid(), psutil.Process().create_time())
        self.qps, self.clock, self.sleep = qps, clock, sleep
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS workflow_limits '
                       '(workflow TEXT PRIMARY KEY, qps INTEGER NOT NULL, starts TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS workflow_requests '
                       '(id INTEGER PRIMARY KEY, workflow TEXT NOT NULL, started REAL NOT NULL, '
                       'finished REAL, status TEXT NOT NULL, call_tag TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS workflow_started ON workflow_requests(workflow, started)')
            if 'error_kind' not in {r[1] for r in db.execute('PRAGMA table_info(workflow_requests)')}:
                db.execute('ALTER TABLE workflow_requests ADD COLUMN error_kind TEXT')
            columns = {r[1] for r in db.execute('PRAGMA table_info(workflow_requests)')}
            for column, kind in [('owner_pid', 'INTEGER'), ('owner_created', 'REAL')]:
                if column not in columns:
                    db.execute(f'ALTER TABLE workflow_requests ADD COLUMN {column} {kind}')
            db.execute('CREATE INDEX IF NOT EXISTS workflow_status ON workflow_requests(status)')
            db.execute('CREATE TABLE IF NOT EXISTS workflow_model_limit '
                       '(singleton INTEGER PRIMARY KEY, max_inflight INTEGER NOT NULL, '
                       'max_active_workflows INTEGER NOT NULL, peak_inflight INTEGER NOT NULL)')
            db.execute('INSERT OR IGNORE INTO workflow_model_limit VALUES (1, ?, ?, 0)',
                       (max_inflight, max_active_workflows))
            db.execute('UPDATE workflow_model_limit SET max_inflight=min(max_inflight, ?), '
                       'max_active_workflows=min(max_active_workflows, ?) WHERE singleton=1',
                       (max_inflight, max_active_workflows))
            db.execute('CREATE TABLE IF NOT EXISTS workflow_pacing '
                       '(workflow TEXT PRIMARY KEY, min_interval REAL NOT NULL, '
                       'blocked_until REAL NOT NULL, last_throttled REAL, throttle_streak INTEGER NOT NULL)')
            for workflow in self.ids:
                db.execute('INSERT OR IGNORE INTO workflow_limits VALUES (?, ?, ?)', (workflow, qps, '[]'))
                stored_qps = db.execute('SELECT qps FROM workflow_limits WHERE workflow=?', (workflow,)).fetchone()[0]
                if qps > stored_qps:
                    raise ValueError('The shared database already has a different QPS for this workflow')
                if qps < stored_qps:
                    db.execute('UPDATE workflow_limits SET qps=? WHERE workflow=?', (qps, workflow))
                db.execute('INSERT OR IGNORE INTO workflow_pacing VALUES (?, ?, 0, NULL, 0)',
                           (workflow, interval))
                # A status reader or another process cannot silently loosen a
                # stricter interval already selected for this shared ID.
                db.execute('UPDATE workflow_pacing SET min_interval=max(min_interval, ?) WHERE workflow=?',
                           (interval, workflow))

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
                self._release_dead_owners(db, now)
                max_inflight, max_active = db.execute('SELECT max_inflight,max_active_workflows '
                    'FROM workflow_model_limit WHERE singleton=1').fetchone()
                active = dict(db.execute('SELECT workflow,count(*) FROM workflow_requests '
                                         'WHERE status="inflight" GROUP BY workflow'))
                candidates = []
                for workflow in self.ids:
                    if sum(active.values()) >= max_inflight or (workflow not in active and len(active) >= max_active):
                        continue
                    qps, stamps_text = db.execute('SELECT qps,starts FROM workflow_limits WHERE workflow=?', (workflow,)).fetchone()
                    stamps = json.loads(stamps_text)
                    interval, blocked_until = db.execute(
                        'SELECT min_interval,blocked_until FROM workflow_pacing WHERE workflow=?',
                        (workflow,)).fetchone()
                    # Small boundary margin protects a rolling one-second quota.
                    recent = [stamp for stamp in stamps if stamp > now-1.005]
                    due = recent[-qps]+1.005 if len(recent) >= qps else now
                    # Never reserve future slots: after any late wake-up or
                    # idle period there is no backlog of permits to burst.
                    due = max(due, blocked_until, stamps[-1]+interval if stamps else now)
                    candidates.append((due, stamps[-1] if stamps else -math.inf, workflow, recent, qps))
                due, _, workflow, recent, qps = min(candidates, key=lambda c: (c[0], c[1], self.ids.index(c[2]))) if candidates else (now+.05, 0, None, [], 0)
                if due <= now:
                    recent.append(now)
                    db.execute('UPDATE workflow_limits SET starts=? WHERE workflow=?',
                               (json.dumps(recent[-qps:]), workflow))
                    cursor = db.execute('INSERT INTO workflow_requests(workflow,started,status,call_tag,owner_pid,owner_created) VALUES (?,?,?,?,?,?)',
                                        (workflow, now, 'inflight', call_tag, *self.owner))
                    db.execute('UPDATE workflow_model_limit SET peak_inflight=max(peak_inflight, ?) WHERE singleton=1',
                               (sum(active.values())+1,))
                    return workflow, cursor.lastrowid
                delay = due-now
            # Never hold a database transaction while waiting or running HTTP.
            self.sleep(max(.001, delay))

    @staticmethod
    def _release_dead_owners(db, now):
        owners = db.execute('SELECT DISTINCT owner_pid,owner_created FROM workflow_requests '
                            'WHERE status="inflight" AND owner_pid IS NOT NULL').fetchall()
        for pid, created in owners:
            try:
                process = psutil.Process(pid)
                live = process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                live = False
            except psutil.AccessDenied:
                live = True  # Cannot establish exit; retain the occupied slots.
            if not live:
                db.execute('UPDATE workflow_requests SET status="abandoned",finished=?,error_kind="owner_exited" '
                           'WHERE status="inflight" AND owner_pid=? AND owner_created=?', (now, pid, created))

    def global_summary(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._release_dead_owners(db, self.clock())
            cap, workflows, peak = db.execute('SELECT max_inflight,max_active_workflows,peak_inflight '
                                             'FROM workflow_model_limit WHERE singleton=1').fetchone()
            active = dict(db.execute('SELECT workflow,count(*) FROM workflow_requests WHERE status="inflight" GROUP BY workflow'))
            legacy = db.execute('SELECT count(*) FROM workflow_requests WHERE status="inflight" AND owner_pid IS NULL').fetchone()[0]
            return {'max_inflight': cap, 'current_inflight': sum(active.values()), 'peak_inflight': peak,
                    'max_active_workflows': workflows, 'active_workflows': active,
                    'unowned_legacy_inflight': legacy, 'database': str(self.database)}

    def finish(self, request_id, status, *, error=None):
        limited = getattr(error, 'status_code', None) == 429
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            now = self.clock()
            workflow, started = db.execute('SELECT workflow,started FROM workflow_requests WHERE id=?',
                                           (request_id,)).fetchone()
            db.execute('UPDATE workflow_requests SET finished=?,status=?,error_kind=? WHERE id=?',
                       (now, status, 'rate_limit' if limited else None, request_id))
            if limited:
                streak = db.execute('SELECT throttle_streak FROM workflow_pacing WHERE workflow=?',
                                    (workflow,)).fetchone()[0]
                cooldown = min(30., 2.*2**min(streak, 4))
                db.execute('UPDATE workflow_pacing SET blocked_until=max(blocked_until, ?),'
                           'last_throttled=?,throttle_streak=throttle_streak+1 WHERE workflow=?',
                           (now+cooldown, now, workflow))
            elif status == 'completed':
                # Successes admitted before the throttle may still be in
                # flight; they must not clear the shared backoff state.
                db.execute('UPDATE workflow_pacing SET throttle_streak=0 WHERE workflow=? '
                           'AND (last_throttled IS NULL OR ?>last_throttled)', (workflow, started))

    @contextmanager
    def request(self, call_tag='chat'):
        workflow, request_id = self.acquire(call_tag)
        try:
            yield workflow
        except BaseException as exc:
            self.finish(request_id, 'failed', error=exc)
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
                interval, blocked_until = db.execute(
                    'SELECT min_interval,blocked_until FROM workflow_pacing WHERE workflow=?',
                    (workflow,)).fetchone()
                throttles = db.execute('SELECT count(*) FROM workflow_requests WHERE workflow=? '
                    'AND finished>? AND error_kind="rate_limit"', (workflow, now-window)).fetchone()[0]
                result[workflow] = {'admitted': count, 'completed': completed or 0, 'failed': failed or 0,
                    'unfinished': inflight or 0, 'window_seconds': round(duration, 3),
                    'admitted_qps': round(count/duration, 3) if duration >= 1 else None,
                    'peak_admissions_in_rolling_second': peak,
                    'minimum_admission_gap_seconds': round(min((b-a for a,b in zip(stamps, stamps[1:])), default=math.inf), 6)
                        if len(stamps) > 1 else None,
                    'client_min_interval_seconds': round(interval, 6),
                    'paced_qps_ceiling': round(1/interval, 3),
                    'rate_limit_responses_in_window': throttles,
                    'cooldown_remaining_seconds': round(max(0., blocked_until-now), 3),
                    'mean_transport_seconds': round(latency, 3) if latency is not None else None,
                    'qps_limit': db.execute('SELECT qps FROM workflow_limits WHERE workflow=?', (workflow,)).fetchone()[0]}
        return result


def from_environment():
    ids = [item.strip() for item in os.getenv('SKILLMINING_WORKFLOW_POOL_IDS', '').split(',') if item.strip()]
    if not ids:
        return None
    database = os.getenv('SKILLMINING_WORKFLOW_RATE_DB', '').strip()
    if not database:
        raise ValueError('Workflow pooling requires one shared SKILLMINING_WORKFLOW_RATE_DB path')
    interval_text = os.getenv('SKILLMINING_WORKFLOW_MIN_INTERVAL_SECONDS', '').strip()
    interval = float(interval_text) if interval_text else None
    qps = int(os.getenv('SKILLMINING_WORKFLOW_QPS', '2'))
    cap = int(os.getenv('SKILLMINING_WORKFLOW_MAX_INFLIGHT', '10'))
    workflows = int(os.getenv('SKILLMINING_WORKFLOW_MAX_ACTIVE_IDS', '5'))
    key = (os.getpid(), database, tuple(ids), qps, interval, cap, workflows)
    if getattr(_LOCAL, 'key', None) != key:
        _LOCAL.pool = WorkflowRequestPool(database, ids, qps, min_interval_seconds=interval,
                                        max_inflight=cap, max_active_workflows=workflows)
        _LOCAL.key = key
    return _LOCAL.pool
