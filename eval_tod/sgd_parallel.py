"""Bounded, ordered SGD case execution with process-local workflow routing."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextvars import ContextVar
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback
from uuid import uuid4

_FUNCTION = None
_PHASE = "testing"
_WORKER_DIR = None
_CASE_DIRECTORY = ContextVar('sgd_case_directory', default=None)


def diagnostics_directory(default):
    return _CASE_DIRECTORY.get() or (_WORKER_DIR if _WORKER_DIR is not None else default)


def threaded_execution():
    return os.getenv('SKILLMINING_SGD_EXECUTOR', 'process') == 'thread'


def _thread_case(function, index, item, directory, phase):
    import llm
    from eval_tod.sgd_usage import case_usage_scope, configure_usage, set_phase, usage_payload
    case_dir = directory/f'case_{index:08d}'
    token = _CASE_DIRECTORY.set(case_dir)
    try:
        with llm.usage_scope(), case_usage_scope():
            configure_usage(case_dir)
            set_phase(phase)
            result = _execute(function, item)
            result['usage'] = usage_payload()
            result['worker_directory'] = str(case_dir)
            return result
    finally:
        _CASE_DIRECTORY.reset(token)


def _ordered_threads(function, items, workers, workflow_ids, directory, phase):
    import llm
    from eval_tod.sgd_usage import merge_worker_usage, mark_incomplete_usage
    if not callable(getattr(llm, 'usage_scope', None)):
        raise RuntimeError('Threaded SGD requires the updated workflow-aware llm_new.py as llm.py (usage_scope)')
    from workflow_request_pool import from_environment
    pool_config = from_environment()
    if pool_config is None:
        raise ValueError('Threaded SGD requires the request-level workflow ID pool and shared rate database')
    if workflow_ids and set(workflow_ids) != set(pool_config.ids):
        raise ValueError('Case workflow IDs differ from the request pool')
    directory = Path(directory)/'parallel_cases'/f'{phase}_{uuid4().hex[:12]}'
    directory.mkdir(parents=True, exist_ok=True)
    print(f'SGD {phase}: {workers} case threads; {len(pool_config.ids)} request-routed workflow IDs', flush=True)
    iterator, pending, ready = iter(enumerate(items)), {}, {}
    next_output, exhausted = 0, False
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='sgd-case') as executor:
        while pending or ready or not exhausted:
            while not exhausted and len(pending)+len(ready) < 2*workers:
                item = next(iterator, None)
                if item is None:
                    exhausted = True
                    break
                index, value = item
                future = executor.submit(_thread_case, function, index, value, directory, phase)
                pending[future] = index
            if pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        mark_incomplete_usage(phase)
                        result = {'value': None, 'error': f'Thread failure: {exc}',
                                  'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}
                    else:
                        try:
                            merge_worker_usage(result.pop('usage'))
                        except Exception as exc:
                            result['usage_accounting_error'] = f'{type(exc).__name__}: {exc}'
                            mark_incomplete_usage(phase)
                    ready[index] = result
            while next_output in ready:
                yield ready.pop(next_output)
                next_output += 1
                if next_output % 100 == 0:
                    print(f'SGD {phase}: completed {next_output} cases in input order', flush=True)


def _initialize(function, ids, counter, directory, phase):
    global _FUNCTION, _PHASE, _WORKER_DIR
    from eval_tod.sgd_usage import configure_usage
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    if isinstance(function, bytes):
        import cloudpickle
        function = cloudpickle.loads(function)
    _FUNCTION, _PHASE = function, phase
    if ids:
        os.environ["SKILLMINING_WORKFLOW_ID"] = ids[index]
    _WORKER_DIR = Path(directory) / f"worker{index}"
    configure_usage(_WORKER_DIR)


def _execute(function, item):
    try:
        return {"value": function(item), "error": None}
    except Exception as exc:
        return {"value": None, "error": f"{type(exc).__name__}: {exc}",
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}


def _task(index_item):
    from eval_tod.sgd_usage import reset_case_usage, set_phase, usage_payload
    index, item = index_item
    reset_case_usage()
    set_phase(_PHASE)
    result = _execute(_FUNCTION, item)
    result["usage"] = usage_payload()
    result["worker_directory"] = str(_WORKER_DIR)
    # Keep each case's accounting recoverable even if the parent is interrupted.
    (_WORKER_DIR / f"case_{index:08d}_usage.json").write_text(
        json.dumps(result["usage"], ensure_ascii=False), encoding="utf-8")
    return result


def ordered_cases(function, items, *, workers=1, workflow_ids=(), directory, phase):
    """Yield results in input order; at most 2*workers cases are outstanding.

    Fork preserves the compiled agent. Spawn serializes the closure with
    cloudpickle. Each worker owns one workflow ID and never shares mutable
    state, native provider counters, or usage files with another worker.
    """
    from eval_tod.sgd_usage import merge_worker_usage, mark_incomplete_usage
    if workers < 1:
        raise ValueError('SGD workers must be positive')
    if threaded_execution() and workers > 1:
        yield from _ordered_threads(function, items, workers, workflow_ids, directory, phase)
        return
    if workers == 1:
        for item in items:
            yield _execute(function, item)
        return
    if workers < 1 or (workflow_ids and len(workflow_ids) != workers):
        raise ValueError("SGD workers must match the number of workflow IDs")
    use_fork = "fork" in mp.get_all_start_methods()
    context = mp.get_context("fork" if use_fork else "spawn")
    if not use_fork:
        try:
            import cloudpickle
        except ImportError as exc:
            raise RuntimeError("Spawn-based workers require cloudpickle") from exc
        function = cloudpickle.dumps(function)
    counter = context.Value("i", 0)
    directory = Path(directory) / "parallel_cases" / f"{phase}_{uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    print(f"SGD {phase}: {workers} within-domain processes ({context.get_start_method()}); "
          f"workflow_ids={list(workflow_ids)}", flush=True)
    iterator = iter(enumerate(items))
    pending = deque()
    completed = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=_initialize, initargs=(function, list(workflow_ids), counter, str(directory), phase)) as pool:
        for _ in range(workers * 2):
            item = next(iterator, None)
            if item is None:
                break
            try:
                pending.append(pool.submit(_task, item))
            except Exception as exc:
                mark_incomplete_usage(phase)
                pending.append(_FailedFuture(exc))
        while pending:
            try:
                result = pending.popleft().result()
            except Exception as exc:
                # A crashed process may have consumed tokens that were never
                # returned. Preserve this uncertainty instead of claiming zero.
                mark_incomplete_usage(phase)
                result = {"value": None, "error": f"Worker failure: {exc}", "traceback": traceback.format_exc()}
            else:
                try:
                    merge_worker_usage(result.pop("usage"))
                except Exception as exc:
                    # Preserve the prediction if parent-side accounting fails.
                    # The worker's case usage file remains available to audit.
                    result['usage_accounting_error'] = f'{type(exc).__name__}: {exc}'
                    mark_incomplete_usage(phase)
            yield result
            completed += 1
            if completed % 100 == 0 or not pending:
                print(f"SGD {phase}: completed {completed} cases in input order", flush=True)
            item = next(iterator, None)
            if item is not None:
                try:
                    pending.append(pool.submit(_task, item))
                except Exception as exc:
                    # Broken pools cannot accept later cases, but each case
                    # still receives an explicit failure and stays accounted for.
                    mark_incomplete_usage(phase)
                    pending.append(_FailedFuture(exc))


class _FailedFuture:
    def __init__(self, error):
        self.error = error

    def result(self):
        raise self.error


def require_value(result):
    if result["error"]:
        raise RuntimeError(result["error"] + "\n" + result.get("traceback", ""))
    return result["value"]
