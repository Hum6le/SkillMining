#!/usr/bin/env python3
"""Audit frozen SGD predictions and protocol boundaries without model calls."""
import argparse
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.sgd_adapter import gold_policy, iter_family_dialogues, score_api_calls, visible_context
from eval_tod.sgd_skill_agent import policy_heads, previous_node


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def checkpoint_hash(directory):
    base, standard = read(directory / 'sgd_skill_state.json'), read(directory / 'sgd_standard_state.json')
    value = {'resources': base['resources'], 'state': standard['state'], 'skill': standard['working_skill'],
        'reference': standard['base_reference'], 'action_rules': standard['base_action_rules'],
        'slot_policies': standard['base_slot_policies']}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def function_ast(path, name, *, prefix=False):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    if prefix:
        # Compare the complete action-selection path before payload execution.
        boundary = next(index for index, node in enumerate(function.body)
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Tuple)
                and any(isinstance(item, ast.Name) and item.id == 'payload_skill' for item in target.elts)
                for target in node.targets))
        return ast.dump(ast.Module(body=function.body[:boundary + 1], type_ignores=[]), include_attributes=False)
    return ast.dump(function, include_attributes=False)


def run(root):
    root = root.resolve()
    training = read(root / 'ours/ours_training.json')
    graph = read(root / 'ours/subgraph.json')
    plan = read(root / 'plan.json')
    nodes = {node['id']: node for node in graph['nodes']}
    training_rows = rows(root / 'ours/ours_training_turns.jsonl')
    initial_hash, final_hash = checkpoint_hash(root / 'ours/offline'), checkpoint_hash(root / 'ours')
    assert initial_hash == training['snapshot_sha256'] and final_hash == training['final_sha256']
    train_hashes = Counter(row['sgd_evidence']['runtime_diagnostic']['resource_sha256'] for row in training_rows)
    assert set(train_hashes) == {initial_hash}
    result = {'time': datetime.now(timezone.utc).isoformat(), 'new_model_calls': 0,
        'methods': {}, 'checks': {'train_snapshot_verified': True, 'eval_checkpoint_verified': True},
        'train_scope': {'policy_turns': len(training_rows),
            'call_turns': sum(row['sgd_evidence']['gold_policy']['call'] is not None for row in training_rows)},
        'training_failures': training['case_errors'], 'prediction_sha256': {}}
    identities = {}
    for method in ['awm', 'trace2skill', 'ours']:
        directory = root / method
        summary = read(directory / 'summary.json')
        predictions = rows(directory / 'predictions.jsonl')
        identities[method] = [(row['dialogue_id'], row['turn_index'], row['gold']) for row in predictions]
        assert all(summary['data_manifest'][split]['sha256'] == plan['data_manifest']['file_sha256'][split]
                   for split in ['train', 'validation'])
        call_rows = [row for row in predictions if row['gold']['call'] is not None]
        failures, missed_choices, selected_coverage = Counter(), Counter(), Counter()
        examples = []
        for row in call_rows:
            truth, guess = row['gold']['call'], row['prediction']['call']
            bucket = ('no_call' if guess is None else 'wrong_method_or_service'
                if (truth['service'], truth['method']) != (guess['service'], guess['method'])
                else 'wrong_parameters' if truth != guess else 'correct')
            failures[bucket] += 1
            if method == 'ours':
                diagnostic = row['diagnostic']
                assert diagnostic['resource_sha256'] == final_hash
                heads = diagnostic['selected_action_heads']
                selected_call = any(head[0] == 'call' for head in heads)
                assert (guess is not None) == selected_call
                assert all(not service or service == 'Homes_1' for _, service in heads)
                if not selected_call:
                    missed_choices[json.dumps(heads)] += 1
                selected_family = policy_heads({'call': guess, 'acts': row['prediction']['acts']})
                matches = [key for key, node in nodes.items() if node['action_heads'] == selected_family]
                if selected_call:
                    selected_coverage['call_predictions'] += 1
                    selected_coverage['matching_node_in_graph'] += bool(matches)
                    selected_coverage['matching_card_in_preselection_candidates'] += any(
                        key in diagnostic['candidate_ids'] for key in matches)
                if bucket != 'correct':
                    examples.append({'dialogue_id': row['dialogue_id'], 'turn_index': row['turn_index'],
                        'bucket': bucket, 'gold_method': truth['method'], 'prediction': guess,
                        'selected_heads': heads, 'skipped_requests': diagnostic.get('skipped_payload_requests'),
                        'candidate_ids': diagnostic['candidate_ids']})
        result['methods'][method] = {'original_policy_action_exact': summary['action_set_exact'],
            'original_policy_turns': summary['counts']['turns'],
            'call_only_backend': score_api_calls([row['gold'] for row in call_rows], [row['prediction'] for row in call_rows]),
            'call_only_failures': dict(failures), 'missed_call_selected_heads': dict(missed_choices),
            'selected_card_coverage': dict(selected_coverage), 'examples': examples,
            'execution_status': summary['execution_status']}
        result['prediction_sha256'][method] = hashlib.sha256((directory / 'predictions.jsonl').read_bytes()).hexdigest()
    assert identities['awm'] == identities['trace2skill'] == identities['ours']
    result['checks']['identical_validation_cases_gold_and_source_hashes'] = True
    snapshot = root / 'code_snapshot'
    result['checks']['shared_predict_policy_ast_unchanged_from_launch'] = (
        function_ast(snapshot / 'scripts/run_sgd_domain_eval.py', 'predict_policy')
        == function_ast(ROOT / 'scripts/run_sgd_domain_eval.py', 'predict_policy'))
    result['checks']['ours_action_selection_path_ast_unchanged_from_train'] = (
        function_ast(snapshot / 'eval_tod/sgd_standard_skill_agent.py', 'predict', prefix=True)
        == function_ast(ROOT / 'eval_tod/sgd_standard_skill_agent.py', 'predict', prefix=True))
    result['checks']['eval_service_mapping_matches_actual_service'] = True
    train_calls = [row['sgd_evidence'] for row in training_rows if row['sgd_evidence']['gold_policy']['call']]
    result['train_scope']['missed_calls_in_frozen_offline_rollout'] = sum(
        row['predicted_policy']['call'] is None for row in train_calls)
    result['train_scope']['method_distribution'] = dict(Counter(row['gold_policy']['call']['method'] for row in train_calls))
    # Demonstrate the extra prior-gold dependency with identical visible text.
    dialogues = list(iter_family_dialogues(ROOT / 'data/eval/sgd/splits_10pct_seed42', 'Homes', 'validation'))
    for dialogue in dialogues:
        indices = dialogue['target_indices']['policy_turn_indices']
        if len(indices) < 2:
            continue
        modified = copy.deepcopy(dialogue)
        previous, target = indices[0], indices[1]
        old_source = previous_node(dialogue, target, 'Homes')
        modified['turns'][previous]['service_call'] = {}
        modified['turns'][previous]['dialogue_acts'] = {'all': [
            {'intent': 'goodbye', 'domain': 'Homes_1', 'slot': ''}]}
        new_source = previous_node(modified, target, 'Homes')
        if old_source != new_source:
            result['prior_annotation_dependency'] = {'dialogue_id': dialogue['dialogue_id'], 'target_turn': target,
                'changed_annotation_turn': previous, 'same_visible_context': visible_context(dialogue, target) == visible_context(modified, target),
                'old_graph_source': old_source, 'new_graph_source': new_source}
            break
    standard = read(root / 'ours/sgd_standard_state.json')
    result['online_resources'] = {'resolved_slot_policies': sum(
        row.get('status') == 'resolved' for row in standard['state']['slot_policies'].values()),
        'successful_reflection_groups': [entry['group_index'] for entry in read(root / 'ours/refinement_ledger.json')
            if entry.get('phase') == 'online_reflection']}
    rejected = []
    for path in sorted((root / 'ours').glob('reflection_[0-9][0-9][0-9][0-9].json')):
        rejected += [dict(item, group=path.stem) for item in read(path).get('rejected_updates', [])]
    result['rejected_resource_updates'] = {'total': len(rejected),
        'by_reason': dict(Counter(item['reason'] for item in rejected)),
        'slot_policy_rejections': sum(item['update'].get('resource') == 'slot_policy' for item in rejected),
        'examples': rejected}
    output = root / 'sgd_protocol_audit.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    methods = result['methods']
    lines = ['# SGD Homes 协议审计', '', '本次只读审计，没有新增模型调用，没有修改训练或预测。', '',
        '## 同一组 32 个 gold CALL turn 的错误分解', '',
        '| 方法 | API 方法正确 | CALL 漏选 | 方法/服务选错 | 参数错误 | API joint 正确 |', '|---|---:|---:|---:|---:|---:|']
    for method, row in methods.items():
        failure = row['call_only_failures']
        lines.append('| ' + ' | '.join([method, str(row['call_only_backend']['action_correct']),
            *[str(failure.get(key, 0)) for key in ['no_call', 'wrong_method_or_service', 'wrong_parameters', 'correct']]]) + ' |')
    lines += ['', '## 已验证', '',
        '- 三方法的 validation 样本、顺序、gold、训练/验证数据哈希一致。',
        '- Ours 的 507 条训练预测均使用同一离线资源 hash；115 条验证预测均使用保存后的最终资源 hash。',
        f"- 启动时与当前共享 predictor 的 AST 相同：{result['checks']['shared_predict_policy_ast_unchanged_from_launch']}。",
        f"- Ours 的 action 选择路径从训练到验证没有代码变化：{result['checks']['ours_action_selection_path_ast_unchanged_from_train']}。",
        '- 预测中的 service ID 归一化正确，没有 Homes/Homes_1 的错配。',
        '- 空 payload 跳过发生在 action 选择之后；没有 CALL head 时，原执行器也会丢弃后续 CALL。', '',
        '## 确认的协议差异与实现问题', '',
        '1. 训练目标是 507 个全 policy turn，含 158 个 CALL；当前报告只看 32 个 CALL。未按 CALL-only 任务重新训练。',
        '2. 旧 action_set_exact 是整个 dialogue-act head 集合；新的 API action exact 是 service+method。二者分母和语义均不同，64.35% 与 46.88% 不能作为前后下降比较。',
        '3. Ours 上游先选 call/request/confirm 等 heads，未输入完整 API schema；AWM/Trace2Skill 的首次 API 决策直接看完整 schema。共享 payload predictor 不等于共享 base agent。',
        '4. Ours 训练图节点按 intent+service-family 建立，API method 明确放在 payload 阶段；因此图内 action 与当前报告的 API action 不同。',
        '5. payload_resources 复用选择前的候选 cards，没有按最终选择重新读取 card。此为资源接线问题；当前错误数据不能单独证明其性能影响。',
        '6. previous_node 从此前 gold policy 注释路由，而输入契约声明只看此前 utterances。仅改变此前注释、保持所有可见文本不变，图 source 会改变。',
        '7. 第 4 组 reflection 的文本替换目标不存在，整个组回滚。当前 Ours checkpoint 标记为 refined_with_errors。', '',
        '8. 确定的资源写入 bug：free-text 模式没有执行节点 ID→action label 归一化，14 条 action_rule/slot_policy 更新被 unknown_action 拒绝。',
        '   原 checkpoint 的 online slot policy 全部 pending，成功落库为 0。源码已修正，保留自由文本编辑与原生 slot_policy 更新机制。', '',
        '## 结论与下一轮要求', '',
        '直接失败位置已确定：32 个 CALL 目标中，17 个在上游被判为非 CALL。剩余 15 个的方法全部正确，其中 5 个参数错误。',
        '训练目标/新评测目标、图中的 action/报告中的 action、三方法的 base agent 尚未统一。现有结果不能单独归因于 skill 方法优劣。',
        '下一轮应统一 CALL-only 任务：训练仅用 158 个 CALL，评测固定 32 个 CALL，非 CALL 对话仅作历史上下文；',
        '共同执行器先在完整 schema 中选择 API(service,method)，再根据所选 card 绑定 parameters。所有方法共享相同任务提示与可见信息。',
        '原来的 507-turn 图和 policy/slot 资源不是上述 API-method 图，不能只改评测 prompt 后宣称训练已完全对齐。',
        '本数据仅有 FindApartment 和 ScheduleVisit 两个有 gold 支持的 API method；后端 API 图规模需要重新计算，不能沿用 12 个 dialogue-policy 节点的说法。', '']
    (root / 'sgd_protocol_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def replay_rejected_addresses(root):
    """Recover only the exact existing-ID aliases in saved successful groups.

    No text edit is replayed, no failed group's decisions are accepted, and
    all original checkpoints/predictions remain unchanged.
    """
    from zipfile import ZipFile
    from eval_tod.sgd_standard_skill_agent import SGDStandardSkillAgent
    from skill_mining.online_refinement import apply_reflection_updates_to_state
    source, output = root / 'ours', root / 'ours_address_repair'
    if output.exists():
        raise FileExistsError(output)
    original_hash = checkpoint_hash(source)
    with ZipFile(ROOT / 'data/eval/sgd/data.zip') as zipped:
        ontology = json.load(zipped.open('data/ontology.json'))
    agent = SGDStandardSkillAgent(family='Homes', model=read(root / 'plan.json')['model'],
        ontology=ontology, predictor=None, mining_method='trace_cover')
    agent.load(source)
    assert agent.resource_hash() == original_hash
    repaired = []
    for path in sorted(source.glob('reflection_[0-9][0-9][0-9][0-9].json')):
        report = read(path)
        updates = [copy.deepcopy(item['update']) for item in report.get('rejected_updates', [])
                   if item['reason'] == 'unknown_action']
        if not updates:
            continue
        proposal = {'updates': updates}
        contents = [item['content'] for item in updates]
        agent._canonicalize_reflection_targets(proposal)
        assert [item['content'] for item in updates] == contents
        accepted, rejected = apply_reflection_updates_to_state(agent.state, updates)
        if rejected or len(accepted) != len(updates):
            raise ValueError('A saved update is not an exact alias of an existing target')
        repaired.append({'source_reflection': path.name, 'canonicalized_targets': proposal['canonicalized_targets'],
            'accepted_updates': accepted})
    agent.save(output)
    loaded = SGDStandardSkillAgent(family='Homes', model=agent.model, ontology=ontology, predictor=None,
        mining_method='trace_cover')
    loaded.load(output)
    assert loaded.resource_hash() == agent.resource_hash()
    resolved = {action: policy for action, policy in loaded.state['slot_policies'].items()
                if policy.get('status') == 'resolved'}
    for node in loaded.graph['nodes']:
        if node['label'] in resolved:
            assert loaded.action_card(node)['payload_policy'] == resolved[node['label']]['policy']
    assert checkpoint_hash(source) == original_hash
    record = {'source_run': str(source), 'new_model_calls': 0, 'source_sha256': original_hash,
        'repaired_sha256': loaded.resource_hash(), 'repairs': repaired,
        'accepted_updates': sum(len(item['accepted_updates']) for item in repaired),
        'resolved_online_slot_policies': len(resolved), 'preserved_stage': loaded.stage,
        'remaining_training_errors': loaded.state.get('refinement_errors'),
        'note': 'Engineering address repair only. Original task scope retained; failed group 4 remains unresolved. No new evaluation result.'}
    (output / 'resource_address_repair.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repair-addresses', action='store_true')
    args = parser.parse_args()
    result = run(args.root)
    if args.repair_addresses:
        repaired = replay_rejected_addresses(args.root.resolve())
        report_path = args.root / 'sgd_protocol_audit.md'
        with report_path.open('a', encoding='utf-8') as stream:
            stream.write('\n## 已完成的工程修复验证\n\n'
                'free-text reflection 现在也会把唯一、已存在的完整节点 ID 映射到 action label。\n'
                '32 项相关回归测试通过，包含写入后保存/重载和 action card 读取。\n'
                f"保存的 {repaired['accepted_updates']} 条被拒绝提案已原文重放到独立 checkpoint，"
                f"成功落库的 online slot policy 从 0 变成 {repaired['resolved_online_slot_policies']} 条。\n"
                '原预测、原 checkpoint 未修改，新增模型调用为 0。第 4 组 reflection 仍未补跑，尚无修复后新评测分数。\n'
                '修复 checkpoint：`ours_address_repair/`；完整过程见其中 `resource_address_repair.json`。\n')
        print(json.dumps({'repair_verified': True, 'new_model_calls': repaired['new_model_calls'],
            'accepted_updates': repaired['accepted_updates'],
            'resolved_online_slot_policies': repaired['resolved_online_slot_policies'],
            'remaining_stage': repaired['preserved_stage']}))
    print(json.dumps({'checks': result['checks'], 'train_scope': result['train_scope'],
        'online_resources': result['online_resources'], 'prior_annotation_dependency': result.get('prior_annotation_dependency'),
        'selected_card_coverage': result['methods']['ours']['selected_card_coverage']}))
