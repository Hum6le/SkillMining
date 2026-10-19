"""Stop only a named SGD batch and its descendants; default is preview only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import psutil

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = {'launch_sgd_server_batch.py', 'run_full_sgd_experiments.py',
           'run_sgd_domain_eval.py', 'run_sgd_trace2skill_pilot.py'}


def resolved_argument(value, cwd):
    path = Path(value)
    return (path if path.is_absolute() else Path(cwd)/path).resolve()


def belongs_to_batch(command, cwd, target, project_root=ROOT):
    """Match actual argv path arguments, never command substrings or PID alone."""
    script = next((arg for arg in command if Path(arg).name in SCRIPTS), None)
    if script is None or resolved_argument(script, cwd).parent != (project_root/'scripts').resolve():
        return False
    for option in ['--output-dir', '--resume-run']:
        if option in command:
            index = command.index(option)
            if index+1 < len(command):
                output = resolved_argument(command[index+1], cwd)
                if output == target or target in output.parents:
                    return True
    return False


def matching_processes(target):
    result = []
    for process in psutil.process_iter(['pid', 'cmdline', 'cwd']):
        try:
            if process.pid != psutil.Process().pid and belongs_to_batch(
                    process.info['cmdline'] or [], process.info['cwd'] or ROOT, target):
                result.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return result


def identity(process):
    return process.pid, process.create_time()


def stop_batch(target, execute=False):
    target = Path(target).resolve()
    if target == ROOT or target == ROOT/'outputs' or ROOT/'outputs' not in target.parents:
        raise ValueError('Specify one concrete batch directory beneath this project outputs directory')
    if not any((target/name).is_file() for name in ['batch_plan.json', 'launches.json']):
        raise FileNotFoundError(f'No SGD batch metadata in {target}; verify --output-dir')
    roots = matching_processes(target)
    result = {'batch': str(target), 'execute': execute, 'matched_roots': [p.pid for p in roots],
              'processes': [], 'data_deleted': False}
    frozen = []
    processes = {}
    try:
        # Suspend every matching launcher/coordinator/domain runner before
        # enumerating descendants, preventing fresh children during shutdown.
        if execute:
            for process in roots:
                try:
                    process.suspend()
                    frozen.append((process, identity(process)))
                except psutil.NoSuchProcess:
                    continue
        for process in roots:
            try:
                for owned in [process, *process.children(recursive=True)]:
                    processes[identity(owned)] = owned
            except psutil.NoSuchProcess:
                continue
        result['processes'] = [{'pid': pid, 'created': created} for pid, created in processes]
        if execute:
            for expected, process in processes.items():
                try:
                    if process.is_running() and identity(process) == expected:
                        process.kill()
                except psutil.NoSuchProcess:
                    continue
            _, alive = psutil.wait_procs(list(processes.values()), timeout=10)
            remaining = []
            for process in alive:
                try:
                    if process.is_running() and identity(process) in processes:
                        remaining.append(process.pid)
                except psutil.NoSuchProcess:
                    pass
            remaining.extend(p.pid for p in matching_processes(target))
            result['remaining_pids'] = sorted(set(remaining))
            result['status'] = 'stopped' if not remaining else 'incomplete_stop'
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            (target/f'stop_{stamp}.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        else:
            result['status'] = 'preview_only'
        return result
    finally:
        # If permission or inspection fails before shutdown completes, avoid
        # leaving an accidentally suspended process hidden from the caller.
        for process, expected in frozen:
            try:
                if process.is_running() and identity(process) == expected:
                    process.resume()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Actually stop; otherwise inspect only')
    args = parser.parse_args()
    report = stop_batch(args.output_dir, args.execute)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get('remaining_pids'):
        sys.exit(1)
