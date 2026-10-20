"""Bounded local API threads with isolated accounting; no workflow routing."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from uuid import uuid4

from eval_tod.sgd_parallel import _thread_case
from eval_tod.sgd_usage import merge_worker_usage, mark_incomplete_usage


def local_cases(function, items, *, workers, directory, phase, progress=None):
    if workers < 1:
        raise ValueError('workers must be positive')
    root = Path(directory)/'parallel_cases'/f'{phase}_{uuid4().hex[:10]}'
    root.mkdir(parents=True, exist_ok=True)
    iterator = iter(enumerate(items))
    pending = deque()
    started = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='sgd-local') as pool:
        def submit_one():
            value = next(iterator, None)
            if value is None:
                return False
            index, item = value
            pending.append(pool.submit(_thread_case, function, index, item, root, phase))
            return True
        for _ in range(2*workers):
            if not submit_one():
                break
        while pending:
            result = pending.popleft().result()
            try:
                merge_worker_usage(result.pop('usage'))
            except Exception as exc:
                mark_incomplete_usage(phase)
                result['usage_accounting_error'] = repr(exc)
            completed += 1
            if progress:
                progress(completed, time.time()-started)
            yield result
            submit_one()
