from contextlib import nullcontext
import json
import multiprocessing as mp
import os
from types import SimpleNamespace
import time

import pytest

from eval_tod.sgd_parallel import ordered_cases, require_value


def _spawn_probe(index):
    from eval_tod.sgd_usage import record_attempt
    time.sleep(.05)
    record_attempt({}, "input", "reply", tag="mock", success=True)
    if index == 2:
        raise ValueError("one failed case")
    return {"index": index, "workflow": os.environ["SKILLMINING_WORKFLOW_ID"], "pid": os.getpid()}


def test_real_worker_process_isolation_and_usage_on_all_platforms(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from eval_tod import sgd_parallel as parallel, sgd_usage
    context = mp.get_context("spawn")
    counter = context.Value("i", 0)
    with ProcessPoolExecutor(max_workers=2, mp_context=context, initializer=parallel._initialize,
            initargs=(_spawn_probe, ["ID1", "ID2"], counter, str(tmp_path / "workers"), "testing")) as pool:
        results = list(pool.map(parallel._task, enumerate(range(8))))
    assert len(results) == 8 and results[2]["error"]
    good = [require_value(r) for r in results if not r["error"]]
    assert [v["index"] for v in good] == [0, 1, 3, 4, 5, 6, 7]
    assert {v["workflow"] for v in good} == {"ID1", "ID2"}
    assert len({v["pid"] for v in good}) == 2
    sgd_usage.configure_usage(tmp_path / "parent")
    for result in results:
        sgd_usage.merge_worker_usage(result["usage"])
    assert sgd_usage.usage_payload()["testing"]["total"]["calls"] == 8


def test_serial_cases_preserve_order_and_isolate_errors(tmp_path):
    def case(index):
        if index == 1:
            raise ValueError("bad case")
        return index
    results = list(ordered_cases(case, range(3), directory=tmp_path, phase="testing"))
    assert require_value(results[0]) == 0
    assert results[1]["error"] == "ValueError: bad case"
    assert require_value(results[2]) == 2


def test_worker_routes_workflow_and_reports_per_case_usage(monkeypatch, tmp_path):
    from eval_tod import sgd_parallel as parallel, sgd_usage
    from eval_tod.sgd_llm import sgd_chat_with_retry
    monkeypatch.setenv("SKILLMINING_WORKFLOW_ID", "parent")
    monkeypatch.setattr("llm.get_usage_summary", lambda: {})
    monkeypatch.setattr("llm.chat", lambda *args, **kwargs: "OK")
    monkeypatch.setattr(parallel, "_FUNCTION", None)
    monkeypatch.setattr(parallel, "_WORKER_DIR", None)
    monkeypatch.setattr(parallel, "_PHASE", "testing")
    counter = SimpleNamespace(value=0, get_lock=lambda: nullcontext())
    def case(index):
        sgd_chat_with_retry("request", call_tag="mock")
        return (index, os.environ["SKILLMINING_WORKFLOW_ID"])
    parallel._initialize(case, ["ID1"], counter, str(tmp_path / "children"), "train_rollout")
    first = parallel._task((0, 8))
    second = parallel._task((1, 9))
    assert first["value"] == (8, "ID1")
    assert first["usage"]["generation"]["total"]["calls"] == 1
    assert second["usage"]["generation"]["total"]["calls"] == 1
    assert (tmp_path / "children" / "worker0" / "case_00000001_usage.json").is_file()
    sgd_usage.configure_usage(tmp_path / "parent")
    sgd_usage.merge_worker_usage(first["usage"])
    sgd_usage.merge_worker_usage(second["usage"])
    assert sgd_usage.usage_payload()["generation"]["total"]["calls"] == 2
    sgd_usage.mark_incomplete_usage("train_rollout")
    sgd_usage.configure_usage(tmp_path / "parent")
    assert not sgd_usage.usage_payload()["accounting_complete"]


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="Linux/fork integration")
def test_real_processes_use_distinct_ids_and_ordered_complete_accounting(monkeypatch, tmp_path):
    from eval_tod import sgd_usage
    from eval_tod.sgd_llm import sgd_chat_with_retry
    monkeypatch.setattr("llm.get_usage_summary", lambda: {})
    monkeypatch.setattr("llm.chat", lambda *args, **kwargs: "OK")
    sgd_usage.configure_usage(tmp_path)
    def case(index):
        time.sleep(.04)
        sgd_chat_with_retry("request", call_tag="mock")
        if index == 2:
            raise ValueError("one failed case")
        return {"index": index, "workflow": os.environ["SKILLMINING_WORKFLOW_ID"], "pid": os.getpid()}
    results = list(ordered_cases(case, range(8), workers=2, workflow_ids=["ID1", "ID2"],
                                directory=tmp_path, phase="testing"))
    assert len(results) == 8 and results[2]["error"]
    good = [require_value(r) for r in results if not r["error"]]
    assert [v["index"] for v in good] == [0, 1, 3, 4, 5, 6, 7]
    assert {v["workflow"] for v in good} == {"ID1", "ID2"}
    assert len({v["pid"] for v in good}) == 2
    assert sgd_usage.usage_payload()["testing"]["total"]["calls"] == 8


def test_resume_can_change_only_execution_layout(monkeypatch, tmp_path):
    from scripts import run_full_sgd_experiments as runner
    splits, output = tmp_path / "splits", tmp_path / "output"
    splits.mkdir()
    index = {"train_families": ["A", "B"], "families": {
        name: {"test": {"policy_turns": 1}} for name in ("A", "B")}}
    (splits / "INDEX.json").write_text(json.dumps(index))
    for name in ("A", "B"):
        (splits / name).mkdir()
        for split in ("train", "test"):
            (splits / name / f"{split}.jsonl.gz").write_bytes(b"mock input")
    common = ["--method", "ours", "--split", "test", "--splits-dir", str(splits),
              "--archive", str(tmp_path / "data.zip"), "--workers", "2", "--workflow-ids", "ID1,ID2"]
    monkeypatch.setattr(runner.sys, "argv", ["runner", *common, "--output-dir", str(output),
                                            "--parallelism", "across-domains", "--dry-run"])
    runner.main()
    old_plan = (output / "workflow_load_plan.json").read_bytes()
    # Simulate the original plan schema with no new execution fields.
    legacy = json.loads(old_plan)
    legacy.pop("parallelism")
    legacy.pop("case_workers")
    (output / "workflow_load_plan.json").write_text(json.dumps(legacy))
    old_plan = (output / "workflow_load_plan.json").read_bytes()
    active = 0
    commands = []
    def run(command, **kwargs):
        nonlocal active
        active += 1
        assert active == 1
        commands.append(command)
        time.sleep(.01)
        target = command[command.index("--output-dir") + 1]
        from pathlib import Path
        (Path(target) / "summary.json").write_text(json.dumps({"counts": {"turns": 1}}))
        active -= 1
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.sys, "argv", ["runner", *common, "--resume-run", str(output)])
    runner.main()
    assert len(commands) == 2
    assert all(cmd[cmd.index("--workflow-ids") + 1] == "ID1,ID2" for cmd in commands)
    assert all(cmd[cmd.index("--workers") + 1] == "2" for cmd in commands)
    assert (output / "workflow_load_plan.json").read_bytes() == old_plan
    plan = json.loads((output / "execution_plan.json").read_text(encoding="utf-8"))
    assert plan["parallelism"] == "within-domain" and len(plan["workers"]) == 1
