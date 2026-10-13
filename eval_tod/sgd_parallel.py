"""Bounded, ordered SGD case execution with process-local workflow routing."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback
from uuid import uuid4

_FUNCTION = None
_PHASE = "testing"
_WORKER_DIR = None


def diagnostics_directory(default):
    return _WORKER_DIR if _WORKER_DIR is not None else default


def _initialize(function, ids, counter, directory, phase):
    global _FUNCTION, _PHASE, _WORKER_DIR
    from eval_tod.sgd_usage import configure_usage
    with counter.get_lock():
        index = counter.value
        counter.value += 1
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

    Fork preserves the compiled agent and injected predictor without pickling
    closures. Each worker owns one workflow ID and never shares mutable state,
    native provider counters, or usage files with another worker.
    """
    from eval_tod.sgd_usage import merge_worker_usage, mark_incomplete_usage
    if workers == 1:
        for item in items:
            yield _execute(function, item)
        return
    if workers < 1 or (workflow_ids and len(workflow_ids) != workers):
        raise ValueError("SGD workers must match the number of workflow IDs")
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("SGD within-domain parallelism requires Linux/fork; use --workers 1 on this platform")
    context = mp.get_context("fork")
    counter = context.Value("i", 0)
    directory = Path(directory) / "parallel_cases" / f"{phase}_{uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    print(f"SGD {phase}: {workers} within-domain processes; workflow_ids={list(workflow_ids)}", flush=True)
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
                merge_worker_usage(result.pop("usage"))
            except Exception as exc:
                # A crashed process may have consumed tokens that were never
                # returned. Preserve this uncertainty instead of claiming zero.
                mark_incomplete_usage(phase)
                result = {"value": None, "error": f"Worker failure: {exc}", "traceback": traceback.format_exc()}
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
