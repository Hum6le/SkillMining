"""Persist progress and compare a fresh SGD Ours run with saved diagnostics."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.multiwoz_aligned import atomic_json
from eval_tod.sgd_adapter import score_policies, slot_signature


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A writer may be midway through its last line.
    return result


def snapshot(root):
    import psutil
    launch = read(root/'launch.json')
    directory = root/'ours'
    phases = list((directory/'parallel_cases').glob('*'))
    def done(prefix):
        return sum(len(list(p.glob('worker*/case_*_usage.json'))) for p in phases if p.name.startswith(prefix))
    summary = read(directory/'summary.json')
    diagnoses = list(directory.glob('diagnosis_[0-9][0-9][0-9][0-9].json'))
    reflections = list(directory.glob('reflection_[0-9][0-9][0-9][0-9].json'))
    train_done, eval_done = done('train_rollout_'), done('testing_')
    if summary:
        phase = summary['execution_status']
    elif eval_done:
        phase = 'validation'
    elif reflections:
        phase = 'online_reflection'
    elif diagnoses:
        phase = 'online_diagnosis'
    elif (directory/'online_schedule.json').exists():
        phase = 'train_rollout'
    else:
        phase = 'offline_compile'
    result = {'updated_at': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'phase': phase,
        'process_alive': psutil.pid_exists(launch.get('pid', -1)), 'workers': launch.get('workers'),
        'train_completed': train_done, 'train_total': 507, 'validation_completed': eval_done,
        'validation_total': 115, 'diagnoses_completed': len(diagnoses),
        'reflection_groups_committed': len(reflections),
        'error_records': len(rows(directory/'case_errors.jsonl')),
        'available_memory_GB': round(psutil.virtual_memory().available/2**30, 2),
        'result': {key: summary[key] for key in ['action_set_exact', 'slot_exact_given_action', 'turn_joint_ast'] if key in summary}}
    if launch.get('mode', '').startswith('recover_execution_failed_TRAIN'):
        recovery = read(root/'recovery_progress.json')
        manifest = read(root/'recovery_manifest.json')
        initial = len(manifest.get('initial_failed_cases', []))
        recovered_count = recovery.get('recovered', read(root/'train_recovery_audit.json').get('failed_cases_recovered', 0))
        result.update(phase=recovery.get('phase', 'recover_train_failures'),
            retry_workers=launch.get('retry_workers'), train_cases_preserved=507-initial,
            train_completed=max(train_done, recovery.get('train_complete', 507-initial+recovered_count)),
            recovered_cases=recovered_count,
            known_failed_cases=recovery.get('known_failed_cases', initial),
            retry_worker_cases_processed=sum(1 for _ in (root/'retry_cases/parallel_cases').glob('*/worker*/case_*_usage.json')),
            recovery=recovery)
        evaluation = root/'evaluation'
        if evaluation.exists():
            result['validation_completed'] = sum(1 for _ in (evaluation/'parallel_cases').glob('testing_*/worker*/case_*_usage.json'))
        if summary:
            result['phase'] = summary['execution_status']
    return result


def finalize(root):
    directory = root/'ours'
    summary = read(directory/'summary.json')
    old_root = ROOT/'outputs/sgd_aligned_failure_recovery_20261018_204325'
    previous = rows(old_root/'ours/predictions.jsonl')
    current = rows(directory/'predictions.jsonl')
    identity = lambda rs: [(r['dialogue_id'], r['turn_index'], r['gold']) for r in rs]
    if len(current) != 115 or identity(current) != identity(previous):
        raise ValueError('Fresh evaluation changed the fixed 115 cases/order/gold')
    training = read(directory/'ours_training.json')
    if training.get('scheduled_turns') != 507:
        raise ValueError('Training schedule did not retain all 507 TRAIN policy turns')
    pairs = [(a, b) for a, b in zip(previous, current) if all(
        score_policies([r['gold']], [r['prediction']])['action_set_exact'] for r in [a, b])]
    pair = {'both_actions_correct': len(pairs),
        'previous_payload_correct': sum(slot_signature(a['gold']) == slot_signature(a['prediction']) for a, b in pairs),
        'new_payload_correct': sum(slot_signature(b['gold']) == slot_signature(b['prediction']) for a, b in pairs)}
    history = read(old_root/'comparison.json')['methods']
    result = {'new_ours': summary, 'previous_diagnostic': history, 'same_115_ids_order_gold': True,
        'paired_correct_actions': pair, 'training': training,
        'comparison_scope': 'Fresh same-runtime Ours TRAIN plus validation; historical AWM/Trace2Skill resources were not retrained.'}
    atomic_json(root/'comparison.json', result)
    text = ['# SGD Homes: fresh field-supervised Ours', '', result['comparison_scope'], '',
        '| Run | Action exact | Action F1 | Payload exact given action | Joint |', '|---|---:|---:|---:|---:|']
    for label, row in [('Previous Ours diagnostic', history['ours']), ('New Ours training', summary),
                       ('Historical AWM diagnostic', history['awm']), ('Historical Trace2Skill diagnostic', history['trace2skill'])]:
        text.append('| '+' | '.join([label, *[f'{100*row[k]:.2f}%' for k in
            ['action_set_exact', 'action_micro_f1', 'slot_exact_given_action', 'turn_joint_ast']]])+' |')
    text += ['', 'Paired correct-action subset: '+json.dumps(pair),
             '', 'Execution status: '+summary['execution_status']]
    (root/'comparison.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        result = snapshot(args.root)
        atomic_json(args.root/'progress.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if (args.root/'ours/summary.json').exists():
            if not read(args.root/'launch.json').get('mode', '').startswith('recover_execution_failed_TRAIN'):
                finalize(args.root)
            return
        if not result['process_alive']:
            result['phase'] = 'process_exited_without_summary'
            atomic_json(args.root/'progress.json', result)
            return
        if not args.watch:
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
