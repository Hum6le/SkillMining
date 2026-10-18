#!/usr/bin/env python3
"""Restore the full policy-turn comparison from unchanged saved predictions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.sgd_adapter import score_policies
from eval_tod.multiwoz_aligned import atomic_json


def rescore(root, ours_run=None):
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    expected = plan['data_manifest'].get('validation_policy_turns', plan['data_manifest']['validation_turns'])
    methods, identities = {}, {}
    for method in ['awm', 'trace2skill', 'ours']:
        directory = ours_run if method == 'ours' and ours_run else root / method
        source = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        assert len(rows) == expected
        identities[method] = [(row['dialogue_id'], row['turn_index'], row['gold']) for row in rows]
        scored = score_policies([row['gold'] for row in rows], [row['prediction'] for row in rows])
        assert scored['counts'] == source['counts']
        counts = scored['counts']
        methods[method] = {**scored, 'execution_status': source['execution_status'],
            'case_errors': source['case_errors'], 'resource_training_errors': source.get('resource_training_errors'),
            'source_run': str(directory), 'source_prediction_sha256': hashlib.sha256((directory / 'predictions.jsonl').read_bytes()).hexdigest(),
            'action_micro_precision': counts['action_tp'] / counts['action_pred'] if counts['action_pred'] else 0,
            'action_micro_recall': counts['action_tp'] / counts['action_gold'] if counts['action_gold'] else 0,
            'llm_usage': source['llm_usage']}
    assert identities['awm'] == identities['trace2skill'] == identities['ours']
    result = {'time': datetime.now(timezone.utc).isoformat(), 'evaluation_targets': 'policy',
        'data_manifest': {**plan['data_manifest'], 'evaluation_targets': 'policy', 'validation_turns': expected},
        'methods': methods, 'jobs': json.loads((root / 'jobs.json').read_text(encoding='utf-8'))['jobs'],
        'all_completed': True, 'new_model_calls_for_rescoring': 0,
        'checks': {'all_policy_turns_included': True, 'same_case_order_and_gold': True,
                   'summary_matches_raw_prediction_counts': True}}
    lines = ['# SGD Homes 1/10：全部 policy turn', '',
        f'训练 507 个 policy turn；validation 全部 {expected} 个 policy turn（含 CALL 和非 CALL）。',
        'Action 是无序的 CALL 与 dialogue-act head 集合；Joint 要求该轮全部 action 和 payload 正确。', '',
        '| 方法 | Action acc | Action micro F1 | Slot exact（action 正确时） | Joint AST | 状态 |',
        '|---|---:|---:|---:|---:|---|']
    for method, row in methods.items():
        values = [f'{row[key] * 100:.2f}%' for key in ['action_set_exact', 'action_micro_f1', 'slot_exact_given_action', 'turn_joint_ast']]
        lines.append('| ' + ' | '.join([method, *values, row['execution_status']]) + ' |')
    lines += ['', 'Slot exact 的分母是 action set 完全正确的 turn，并非所有 turn。',
        '这是同一全 policy 指标的比较；CALL-only API 指标保留为辅助诊断。', '']
    for method, row in methods.items():
        errors = row.get('resource_training_errors') or row['case_errors']
        if errors.get('total'):
            lines += [f"{method}：训练错误 {errors['total']} 条，{errors.get('by_phase', {})}。", '']
    if ours_run:
        lines += ['Ours 使用地址映射修复并补跑失败 reflection 后的 checkpoint；baseline 复用原有预测。',
                  f'Ours 新评测目录：`{ours_run}`。', '']
    text = '\n'.join(lines)
    suffix = 'repaired' if ours_run else 'original'
    atomic_json(root / f'comparison_policy_{suffix}.json', result)
    (root / f'comparison_policy_{suffix}.md').write_text(text, encoding='utf-8')
    atomic_json(root / 'comparison.json', result)
    (root / 'comparison.md').write_text(text, encoding='utf-8')
    atomic_json(root / 'evaluation_scope.json', {'evaluation_targets': 'policy', 'validation_turns': expected,
        'note': 'All policy turns are primary. CALL-only remains an auxiliary diagnostic.', 'time': result['time']})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ours-run', type=Path)
    args = parser.parse_args()
    result = rescore(args.root.resolve(), args.ours_run.resolve() if args.ours_run else None)
    print(json.dumps({method: {key: row[key] for key in ['action_set_exact', 'action_micro_f1',
        'slot_exact_given_action', 'turn_joint_ast', 'execution_status']} for method, row in result['methods'].items()}))
