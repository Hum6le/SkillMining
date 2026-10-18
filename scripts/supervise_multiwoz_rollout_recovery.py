#!/usr/bin/env python3
"""Handoff after the complete paid rollout; preserve successes and retry failures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.multiwoz_matched_executor import write_json
from scripts.launch_multiwoz_hotel_pilot import background_python, credential_env, DEFAULT_ENV


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    launch = json.loads((root / 'launch.json').read_text(encoding='utf-8'))
    source_pid = launch['pid']
    expected = json.loads((root / 'data_manifest.json').read_text(encoding='utf-8'))['available_train_turns']
    supervisor = {'pid': __import__('os').getpid(), 'source_pid': source_pid,
        'status': 'waiting_for_complete_rollout', 'scheduled_turns': expected}
    write_json(root / 'rollout_recovery_supervisor.json', supervisor)
    source = root / 'ours_training_turns.jsonl'
    position, pending, count = 0, b'', 0
    while True:
        current = json.loads((root / 'launch.json').read_text(encoding='utf-8'))
        if current['pid'] != source_pid:
            raise ValueError('Source PID changed; refusing to interfere with a replacement run')
        if source.exists():
            with source.open('rb') as stream:
                stream.seek(position)
                data = pending + stream.read()
                position = stream.tell()
            parts = data.split(b'\n')
            pending = parts.pop()
            count += sum(bool(p.strip()) for p in parts)
        if count >= expected:
            break
        if not psutil.pid_exists(source_pid):
            raise RuntimeError(f'Source exited before complete rollout: {count}/{expected}')
        time.sleep(.5)
    if count != expected or pending.strip():
        raise ValueError('Rollout count differs from the full schedule')
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    failures = [row for row in rows if row.get('status') == 'error']
    if not failures:
        supervisor.update(status='no_failures_source_continues')
        write_json(root / 'rollout_recovery_supervisor.json', supervisor)
        return
    if any(root.glob('diagnosis_*.json')) or (root / 'ours_training.json').exists():
        raise ValueError('Diagnosis already committed results; refusing a late handoff')
    parent = psutil.Process(source_pid)
    command = parent.cmdline()
    if (not any(Path(arg).name == 'run_multiwoz_aligned_pilot.py' for arg in command)
            or Path(command[command.index('--output-dir') + 1]).resolve() != root):
        raise ValueError('PID is not the expected runner for this output directory')
    # All scheduled results are durable. Prevent new diagnosis submissions
    # while handing the unchanged frozen predictions to the recovery runner.
    parent.suspend()
    diagnosed = list((root / 'parallel_cases').glob('online_diagnosis_*'))
    processes = [*parent.children(recursive=True), parent]
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        process.kill()
    now = datetime.now(timezone.utc).isoformat()
    handoff = {'time': now, 'source_pid': source_pid, 'rollout_turns': count,
        'failed_turns': len(failures), 'diagnosis_processes_may_have_started': bool(diagnosed),
        'reason': 'Complete rollout preserved; replay successful predictions without API calls, retry only failed cases before standard diagnosis/reflection.'}
    write_json(root / 'rollout_handoff.json', handoff)
    if diagnosed:
        with (root / 'supervisor_interruptions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(handoff) + '\n')
    env, _ = credential_env(DEFAULT_ENV)
    env.update(PYTHONUNBUFFERED='1', SKILLMINING_ENABLE_THINKING='false',
               SKILLMINING_LLM_USAGE_FILE=str(root / 'native_llm_usage.json'))
    command = [*launch['command'], '--recover-rollout']
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    with (root / 'run.log').open('ab', buffering=0) as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=stream, creationflags=flags)
    launch.update(previous_pid=source_pid, pid=process.pid, command=command,
                  resumed_at=now, recovery='successful_frozen_rollout_cache_then_failed_only_retries')
    with (root / 'awake.log').open('ab', buffering=0) as stream:
        awake = subprocess.Popen([str(background_python()), str(ROOT / 'scripts/keep_experiment_awake.py'),
            '--pid', str(process.pid), '--script', 'run_multiwoz_aligned_pilot.py'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, creationflags=flags)
    launch['awake_helper_pid'] = awake.pid
    for path in [root / 'launch.json', ROOT / 'outputs/latest_multiwoz_native_run.json']:
        write_json(path, launch)
    with (root / 'monitor.log').open('ab', buffering=0) as stream:
        monitor = subprocess.Popen([str(background_python()), '-u', str(ROOT / 'scripts/monitor_multiwoz_native.py'),
            '--root', str(root), '--watch', '--interval', '30'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, creationflags=flags)
    launch['monitor_pid'] = monitor.pid
    for path in [root / 'launch.json', ROOT / 'outputs/latest_multiwoz_native_run.json']:
        write_json(path, launch)
    supervisor.update(status='recovery_runner_started', new_pid=process.pid, failed_turns=len(failures))
    write_json(root / 'rollout_recovery_supervisor.json', supervisor)
    print(json.dumps(supervisor), flush=True)


if __name__ == '__main__':
    main()
