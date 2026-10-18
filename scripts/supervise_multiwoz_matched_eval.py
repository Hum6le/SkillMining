#!/usr/bin/env python3
"""Hidden, resumable smoke/full supervisor for the corrected MultiWOZ run."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.multiwoz_matched_executor import write_json
from scripts.launch_multiwoz_hotel_pilot import background_python


def terminate_tree(child):
    import psutil
    try:
        parent = psutil.Process(child.pid)
        parent.suspend()
        processes = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        process.kill()


def main(args):
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "run.log"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    for stage in ("smoke", "full"):
        success = False
        for attempt in range(1, 4):
            command = [str(background_python()), str(ROOT / "scripts/run_multiwoz_matched_eval.py"),
                "--source-root", str(args.source_root.resolve()), "--output-root", str(root),
                "--workers", str(args.workers)]
            if stage == "smoke":
                command += ["--max-eval-cases", "2"]
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"\n{datetime.now().isoformat()} supervisor stage={stage} attempt={attempt}\n")
                stream.flush()
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
                status = {"stage": stage, "attempt": attempt, "supervisor_pid": os.getpid(),
                    "runner_pid": child.pid, "workers_total": args.workers,
                    "status": "running", "started_at": datetime.now().isoformat()}
                write_json(root / "supervisor_status.json", status)
                last_progress = time.time()
                observed = None
                while child.poll() is None:
                    progress = root / "progress.json"
                    stamp = progress.stat().st_mtime_ns if progress.exists() else None
                    if stamp != observed:
                        observed, last_progress = stamp, time.time()
                    status["checked_at"] = datetime.now().isoformat()
                    write_json(root / "supervisor_status.json", status)
                    if time.time() - last_progress > 25 * 60:
                        terminate_tree(child)
                        with (root / "supervisor_interruptions.jsonl").open("a", encoding="utf-8") as event:
                            event.write(json.dumps({"time": datetime.now().isoformat(), "pid": child.pid,
                                "reason": "No completed-job progress for 25 minutes; resume cached successes"}) + "\n")
                        break
                    time.sleep(10)
                code = child.wait(timeout=15)
                expected = root / ("smoke_comparison.json" if stage == "smoke" else "RESULTS_READY.md")
                if code == 0 and expected.exists():
                    success = True
                    break
                status.update(status="retrying", exit_code=code)
                write_json(root / "supervisor_status.json", status)
            time.sleep(10)
        if not success:
            status.update(status="failed", finished_at=datetime.now().isoformat())
            write_json(root / "supervisor_status.json", status)
            (root / "RUN_FAILED.md").write_text(
                f"Stage {stage} failed after three resumable attempts. Inspect run.log and retry_history.jsonl. Successful case caches are preserved.\n", encoding="utf-8")
            raise SystemExit(1)
    status.update(status="completed", finished_at=datetime.now().isoformat())
    write_json(root / "supervisor_status.json", status)
    (ROOT / "outputs/LATEST_MULTIWOZ_RESULTS.md").write_text(
        f"Completed corrected evaluation: [results]({root.as_posix()}/final_comparison.md).\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    main(parser.parse_args())
