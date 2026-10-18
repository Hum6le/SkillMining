#!/usr/bin/env python3
"""Launch a fresh hotel 10% native-policy run invisibly on this Windows host."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.launch_multiwoz_hotel_pilot import background_python, credential_env, DEFAULT_ENV


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('Workers must be positive')
    now = datetime.now(timezone(timedelta(hours=8)))
    root = (args.output_dir or ROOT / 'outputs' / f'multiwoz_hotel_native_{now:%Y%m%d_%H%M%S}').resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Fresh run requires an empty directory')
    root.mkdir(parents=True, exist_ok=True)
    env, model = credential_env(DEFAULT_ENV)
    env.update(PYTHONUNBUFFERED='1', SKILLMINING_ENABLE_THINKING='false',
               SKILLMINING_LLM_USAGE_FILE=str(root / 'native_llm_usage.json'))
    # Content-addressed observations are verified again by the runtime cache.
    copied = 0
    for source in [ROOT / 'outputs/multiwoz_slot_diagnostic_20261018_113434/observations',
                   ROOT / 'outputs/multiwoz_slot_pair_probe_20261018_123208/observations']:
        for path in source.glob('*.json'):
            value = json.loads(path.read_text(encoding='utf-8'))
            if value.get('signature') != path.stem or 'observation' not in value:
                continue
            target = root / 'observations' / path.name
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
                copied += 1
    command = [str(background_python()), '-u', str(ROOT / 'scripts/run_multiwoz_aligned_pilot.py'),
        '--method', 'ours', '--ours-slot-protocol', 'native', '--model', model,
        '--output-dir', str(root), '--workers', str(args.workers), '--fraction', '0.1', '--seed', '42', '--resume']
    with (root / 'preflight.log').open('wb') as stream:
        subprocess.run([sys.executable, *command[1:], '--dry-run'], cwd=ROOT, env=env,
                       stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, check=True)
    manifest = json.loads((root / 'data_manifest.json').read_text(encoding='utf-8'))
    assert not manifest['smoke_limits'] and manifest['available_train_turns'] == 1818 and manifest['eval_turns'] == 228
    paths = [*ROOT.glob('eval_tod/*.py'), *ROOT.glob('eval_tod/abcd/*.py'),
             *ROOT.glob('skill_mining/*.py'), ROOT / 'llm.py', ROOT / 'scripts/run_multiwoz_aligned_pilot.py',
             ROOT / 'scripts/audit_multiwoz_aligned_run.py', ROOT / 'scripts/launch_multiwoz_native.py',
             ROOT / 'scripts/monitor_multiwoz_native.py', ROOT / 'scripts/llm_usage_utils.py',
             ROOT / 'scripts/run_multiwoz_matched_eval.py']
    hashes = {}
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        target = root / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'code_snapshot.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    with (root / 'run.log').open('ab', buffering=0) as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=stream, creationflags=flags)
    launch = {'pid': process.pid, 'output_root': str(root), 'command': command, 'workers': args.workers,
        'model': model, 'thinking': False, 'started_at': now.isoformat(), 'status': 'running',
        'reused_observations': copied, 'train_turns': 1818, 'validation_turns': 228,
        'training': 'fresh TraceCover, free-text cards, standard frozen rollout/diagnosis/sequential reflection',
        'restored': ['native free-text slot policies', 'original snippet selection with four initialization cases',
                     'complete action cards', 'whole-library agent reference tools'],
        'retained': ['native slot_policy reflection upsert', 'validated candidate transport',
                     'common DB provider and fixed-head binder', 'TRAIN binary/empty-value normalization'],
        'log': str(root / 'run.log')}
    awake = [str(background_python()), str(ROOT / 'scripts/keep_experiment_awake.py'), '--pid', str(process.pid),
             '--script', 'run_multiwoz_aligned_pilot.py']
    with (root / 'awake.log').open('ab', buffering=0) as stream:
        helper = subprocess.Popen(awake, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, creationflags=flags)
    launch['awake_helper_pid'] = helper.pid
    for path in [root / 'launch.json', ROOT / 'outputs/latest_multiwoz_native_run.json']:
        path.write_text(json.dumps(launch, ensure_ascii=False, indent=2), encoding='utf-8')
    with (root / 'monitor.log').open('ab', buffering=0) as stream:
        monitor = subprocess.Popen([str(background_python()), '-u', str(ROOT / 'scripts/monitor_multiwoz_native.py'),
            '--root', str(root), '--watch', '--interval', '30'], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=stream, creationflags=flags)
    launch['monitor_pid'] = monitor.pid
    for path in [root / 'launch.json', ROOT / 'outputs/latest_multiwoz_native_run.json']:
        path.write_text(json.dumps(launch, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(launch, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
