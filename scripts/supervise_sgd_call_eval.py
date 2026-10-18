#!/usr/bin/env python3
"""Reuse completed predictions and hand off Ours at its final training checkpoint."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.sgd_adapter import (iter_family_dialogues, evaluation_turn_indices,
    gold_policy, score_policies, score_api_calls)
from eval_tod.multiwoz_aligned import atomic_json
from scripts.launch_sgd_homes_pilot import start
from scripts.launch_multiwoz_hotel_pilot import background_python, credential_env, DEFAULT_ENV


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def prepare(root):
    original = read(root / 'plan.json')
    splits = ROOT / 'data/eval/sgd/splits_10pct_seed42'
    rows = {split: list(iter_family_dialogues(splits, 'Homes', split)) for split in ['train', 'validation']}
    targets = [[d['dialogue_id'], index] for d in rows['validation']
               for index in evaluation_turn_indices(d, 'call')]
    manifest = {**original['data_manifest'], 'evaluation_targets': 'call',
        'validation_policy_turns': original['data_manifest'].get('validation_policy_turns', original['data_manifest']['validation_turns']),
        'validation_turns': len(targets), 'evaluation_case_ids': targets,
        'train_call_turns': sum(len(evaluation_turn_indices(d, 'call')) for d in rows['train']),
        'training_scope_note': 'This run retains its completed 507-policy-turn training. CALL-only retraining was not performed.'}
    assert len(targets) == 32 and manifest['train_call_turns'] == 158
    plan = {**original, 'data_manifest': manifest}
    atomic_json(root / 'call_only_plan.json', plan)
    return plan, {(d['dialogue_id'], i): gold_policy(d, i) for d in rows['validation']
                  for i in evaluation_turn_indices(d, 'call')}


def summarize(root, plan, targets, jobs):
    methods = {}
    for method in ['awm', 'trace2skill', 'ours']:
        directory = root / method
        summary = read(directory / 'summary.json')
        if not summary:
            continue
        rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        selected = [row for row in rows if (row['dialogue_id'], row['turn_index']) in targets]
        ids = [(row['dialogue_id'], row['turn_index']) for row in selected]
        if ids != list(targets):
            raise ValueError(f'{method}: CALL targets are missing, duplicated, or reordered')
        assert all(row['gold'] == targets[(row['dialogue_id'], row['turn_index'])] for row in selected)
        truth, predictions = [row['gold'] for row in selected], [row['prediction'] for row in selected]
        derived = {**summary, **score_policies(truth, predictions),
            'evaluation_targets': 'call', 'evaluation_case_ids': [list(item) for item in ids],
            'backend_call_metrics': score_api_calls(truth, predictions),
            'source_prediction_turns': len(rows),
            'source_summary': str(directory / 'summary.json'),
            'training_scope_note': plan['data_manifest']['training_scope_note']}
        atomic_json(directory / 'call_only_summary.json', derived)
        methods[method] = derived
    result = {'data_manifest': plan['data_manifest'], 'methods': methods, 'jobs': jobs,
              'all_completed': len(methods) == 3, 'evaluation_targets': 'call'}
    atomic_json(root / 'call_only_comparison.json', result)
    lines = ['# SGD Homes 1/10：CALL turn 评测', '',
        '按数据标注固定选择 validation 的 32 个 CALL turn；此前对话仍作为输入上下文。',
        '本轮三方法训练均保留原来的 507 policy turn；158 是其中 CALL turn 数，未重新训练。', '',
        '## 仅后端 API 的指标', '',
        'Action = service + API method；slots = 命名参数。此视图不要求预测 dialogue acts。', '',
        '| 方法 | API action exact | 参数 exact（action 正确时） | API joint AST |',
        '|---|---:|---:|---:|']
    for method in ['awm', 'trace2skill', 'ours']:
        backend = methods.get(method, {}).get('backend_call_metrics', {})
        values = [f'{100 * backend[key]:.2f}%' if key in backend else '运行中'
                  for key in ['api_action_exact', 'api_slot_exact_given_action', 'api_joint_ast']]
        label = method
        if methods.get(method, {}).get('execution_status') == 'completed_with_errors':
            label += '（训练有报错）'
        lines.append('| ' + ' | '.join([label, *values]) + ' |')
    lines += ['', '## 原 policy 指标在相同 32 个 turn 上的结果', '',
        '这些指标仍要求 CALL 和所有 dialogue acts 同时正确，和上面的 API 指标定义不同。', '',
        '| 方法 | Policy action set exact | Policy joint AST |', '|---|---:|---:|']
    for method in ['awm', 'trace2skill', 'ours']:
        summary = methods.get(method, {})
        values = [f'{100 * summary[key]:.2f}%' if key in summary else '运行中'
                  for key in ['action_set_exact', 'turn_joint_ast']]
        lines.append('| ' + ' | '.join([method, *values]) + ' |')
    lines += ['', '两组已完成 baseline 直接复用原始预测，没有新增模型请求。',
        '未按模型输出筛选样本：在 gold CALL turn 上漏选 CALL 仍计错。',
        '115-turn 的旧报告保留，属于之前的全 policy-turn 口径。', '']
    for method, summary in methods.items():
        if summary.get('case_errors', {}).get('total'):
            lines += [f"{method}：训练错误 {summary['case_errors']['total']} 条，"
                f"阶段 {summary['case_errors'].get('by_phase', {})}；当前数值是带错误 checkpoint 的结果。", '']
    (root / 'call_only_comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    if len(methods) == 3 and read(root / 'evaluation_scope.json').get('evaluation_targets') != 'policy':
        atomic_json(root / 'comparison.json', result)
        shutil.copy2(root / 'call_only_comparison.md', root / 'comparison.md')


def frozen_eval_command(command, directory):
    command = list(command)
    if '--eval-targets' in command:
        command[command.index('--eval-targets') + 1] = 'call'
    else:
        command += ['--eval-targets', 'call']
    command += ['--ours-resource-dir', str(directory)]
    return command


def supervise(root, plan, targets):
    import os
    import psutil
    jobs = read(root / 'jobs.json')['jobs']
    launch = read(root / 'launch.json')
    if (root / 'ours/summary.json').exists():
        summarize(root, plan, targets, jobs)
        atomic_json(root / 'call_eval_handoff.json', {'status': 'completed_using_existing_predictions',
            'evaluation_targets': len(targets), 'new_model_calls': 0})
        return
    old_supervisor = psutil.Process(launch['pid'])
    cmd = old_supervisor.cmdline()
    assert any(Path(arg).name == 'launch_sgd_homes_pilot.py' for arg in cmd)
    assert Path(cmd[cmd.index('--supervise') + 1]).resolve() == root
    source = psutil.Process(jobs['ours']['pid'])
    cmd = source.cmdline()
    assert any(Path(arg).name == 'run_sgd_domain_eval.py' for arg in cmd)
    assert Path(cmd[cmd.index('--output-dir') + 1]).resolve() == root / 'ours'
    state = {'pid': os.getpid(), 'source_pid': source.pid, 'status': 'waiting_for_final_training_checkpoint',
             'evaluation_targets': 32, 'training_turns_retained': 507}
    atomic_json(root / 'call_eval_handoff.json', state)
    # Terminate only the completed baselines' scheduler. Its running Ours child
    # remains alive; this watcher owns the next stage and its progress files.
    launch.update(previous_supervisor_pid=launch['pid'], pid=os.getpid(),
                  evaluation_targets='call', validation_turns=32)
    atomic_json(root / 'launch.json', launch)
    atomic_json(ROOT / 'outputs/latest_sgd_homes_run.json', launch)
    old_supervisor.terminate()
    old_supervisor.wait(timeout=5)
    assert source.is_running()
    helper = start([str(background_python()), str(ROOT / 'scripts/keep_experiment_awake.py'),
        '--pid', str(os.getpid()), '--script', Path(__file__).name], root / 'awake.log')
    launch['awake_helper_pid'] = helper.pid
    atomic_json(root / 'launch.json', launch)
    atomic_json(ROOT / 'outputs/latest_sgd_homes_run.json', launch)
    directory = root / 'ours'
    while not (directory / 'ours_training.json').exists() and not (directory / 'summary.json').exists():
        if not source.is_running():
            raise RuntimeError('Ours exited before its complete training checkpoint')
        time.sleep(.5)
    if not (directory / 'summary.json').exists():
        checkpoint = read(directory / 'sgd_skill_state.json')
        assert checkpoint['stage'] in {'refined', 'refined_with_errors'}
        env, _ = credential_env(DEFAULT_ENV)
        env.update(PYTHONUNBUFFERED='1', SKILLMINING_ENABLE_THINKING='false')
        source.suspend()
        processes = [*source.children(recursive=True), source]
        for process in processes:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=5)
        for process in alive:
            process.kill()
        partial = directory / 'predictions.jsonl'
        if partial.exists():
            partial.replace(directory / 'pre_handoff_policy_predictions.jsonl')
        interrupted = {'time': datetime.now(timezone.utc).isoformat(),
            'source_pid': source.pid, 'reason': 'Complete trained resources retained; evaluate only the 32 labeled CALL targets',
            'inflight_evaluation_requests_may_have_been_billed': True}
        with (directory / 'supervisor_interruptions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(interrupted) + '\n')
        from scripts.run_multiwoz_matched_eval import recover_logged_usage
        recover_logged_usage(directory)
        command = frozen_eval_command(plan['commands']['ours'], directory)
        process = start(command, root / 'ours.log', env)
        jobs['ours'].update(pid=process.pid, previous_pid=source.pid, status='running',
            evaluation_targets='call', validation_turns=32)
        state.update(status='call_only_evaluation_running', new_pid=process.pid, command=command)
        atomic_json(root / 'call_eval_handoff.json', state)
        atomic_json(root / 'jobs.json', {'jobs': jobs, 'time': datetime.now(timezone.utc).isoformat()})
        returncode = process.wait()
        summary = read(directory / 'summary.json')
        if returncode or summary.get('counts', {}).get('turns') != 32:
            raise RuntimeError(f'CALL-only evaluation failed: exit={returncode}')
    summary = read(directory / 'summary.json')
    jobs['ours'].update(status=summary['execution_status'], finished_at=datetime.now(timezone.utc).isoformat(),
                        evaluation_targets='call', validation_turns=32)
    atomic_json(root / 'jobs.json', {'jobs': jobs, 'time': datetime.now(timezone.utc).isoformat()})
    summarize(root, plan, targets, jobs)
    state.update(status='completed')
    atomic_json(root / 'call_eval_handoff.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan, targets = prepare(root)
    summarize(root, plan, targets, read(root / 'jobs.json')['jobs'])
    if args.watch:
        try:
            supervise(root, plan, targets)
        except Exception as exc:
            atomic_json(root / 'call_eval_handoff_failed.json', {'error': str(exc), 'type': type(exc).__name__})
            raise
    print(json.dumps({'report': str(root / 'call_only_comparison.md'), 'evaluation_turns': len(targets)}))


if __name__ == '__main__':
    main()
