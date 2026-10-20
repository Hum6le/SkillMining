"""Wait for ExpeL to finish, then resume Reflexion with eight API threads."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_tod.multiwoz_aligned import atomic_json
from scripts.launch_sgd_extra_baselines import running_method


def complete(result):
    return (result.get('evaluation_complete') and not result.get('smoke')
            and result.get('counts', {}).get('turns') == 1241)


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    status_file = out/'reflexion_scale_up.json'
    lock = out/'reflexion_scale_up.lock'
    if lock.exists():
        owner = read(lock)
        try:
            alive = abs(psutil.Process(owner['pid']).create_time()-owner['create_time']) < .01
        except (psutil.Error, KeyError):
            alive = False
        if alive:
            raise RuntimeError(f"Scale-up watcher already running: {owner['pid']}")
        lock.unlink()
    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'create_time': psutil.Process().create_time()}, stream)
    def save(status, **extra):
        atomic_json(status_file, {'status': status, 'pid': os.getpid(),
            'target_workers': 8, 'updated_at': datetime.now(timezone.utc).isoformat(), **extra})
    try:
        original = read(out/'batch_state.json')
        if original.get('status') != 'running':
            save('cancelled', reason='Batch is not running'); return
        controller = psutil.Process(original['pid'])
        controller_created = controller.create_time()
        while True:
            state = read(out/'batch_state.json')
            if state.get('status') == 'paused' or state.get('pid') != controller.pid:
                save('cancelled', reason='User paused or controller changed'); return
            if complete(read(out/'reflexion/aggregate.json')):
                save('unnecessary', reason='Reflexion already complete'); return
            try:
                live = controller.is_running() and controller.create_time() == controller_created
            except psutil.Error:
                live = False
            if not live:
                save('cancelled', reason='Original controller stopped'); return
            if complete(read(out/'expel/aggregate.json')):
                # The aggregate is written shortly before method exit. Wait
                # for its process to exit before reallocating the four slots.
                if running_method('expel', out/'expel') is None:
                    break
            save('waiting_for_expel', controller_pid=controller.pid)
            time.sleep(10)
        reflexion = running_method('reflexion', out/'reflexion')
        if reflexion is None:
            save('cancelled', reason='Reflexion is not running'); return
        save('switching', previous_workers=4, previous_reflexion_pid=reflexion.pid)
        # Stop only this controller and this method. Checkpoints are flushed
        # after every TRAIN policy turn; in-flight uncheckpointed work may repeat.
        controller.terminate(); controller.wait(timeout=15)
        children = reflexion.children(recursive=True)
        reflexion.terminate()
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs([reflexion]+children, timeout=15)
        for process in alive:
            process.kill(); process.wait(timeout=5)
        cmd = [sys.executable, '-u', str(ROOT/'scripts/launch_sgd_extra_baselines.py'),
               '--output-dir', str(out), '--parallel-methods', '--adopt-running',
               '--reflexion-workers', '8', '--env-file', original['plan']['config']['env_file'],
               '--splits-dir', original['plan']['splits_dir']]
        with (out/'controller.scale8.stdout.log').open('a', encoding='utf-8') as stdout, \
                (out/'controller.scale8.stderr.log').open('a', encoding='utf-8') as stderr:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=stderr,
                env={**os.environ, 'PYTHONUNBUFFERED':'1', 'PYTHONIOENCODING':'utf-8'},
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for _ in range(30):
            state = read(out/'batch_state.json')
            if complete(read(out/'reflexion/aggregate.json')):
                save('unnecessary', reason='Reflexion completed during handoff'); return
            lane = next((l for l in state.get('lanes', [])
                         if l.get('current_method') == 'reflexion'), {})
            if state.get('pid') == child.pid and lane.get('workers') == 8:
                save('scaled', controller_pid=child.pid, reflexion_pid=lane['pid'],
                     previous_workers=4, concurrent_request_cap=8)
                return
            if child.poll() is not None:
                raise RuntimeError('Replacement controller exited; inspect controller.scale8.stderr.log')
            time.sleep(1)
        raise RuntimeError('Replacement controller did not report eight Reflexion workers')
    except Exception as exc:
        save('failed', error=repr(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
