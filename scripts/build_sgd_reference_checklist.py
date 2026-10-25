"""Build a source-grounded SGD knowledge reference, independent of mined artifacts.

This compiles public contracts and definitions, not a complete gold process model.
No model calls, dialogue reading, or skill-dependent reference selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


def build(source_dir: Path, service: str) -> tuple[list[dict], dict]:
    schema_path = source_dir / 'train_schema.json'
    schemas = json.loads(schema_path.read_text(encoding='utf-8'))
    index, schema = next((i, s) for i, s in enumerate(schemas) if s['service_name'] == service)
    readme = (source_dir / 'README.md').read_text(encoding='utf-8')
    challenge = (source_dir / 'dstc8.md').read_text(encoding='utf-8')
    rules = []

    def add(key, group, kind, statement, file, locator, **extra):
        rules.append({'id': key, 'service': service, 'group': group, 'kind': kind,
            'statement': statement, 'authority': 'official_public_documentation',
            'source_file': f'sources/{file}', 'source_locator': locator, **extra})

    pointer = f'/{index}'
    methods = [i['name'] for i in schema['intents']]
    add('API.ALLOWED', 'api_contract', 'allowed_method_set',
        f'{service} 的可调用 intent/API 集合为 {methods}。', 'train_schema.json', pointer + '/intents',
        allowed_methods=methods)
    for i, intent in enumerate(schema['intents']):
        name = intent['name']
        prefix = f'API.{name}'
        loc = pointer + f'/intents/{i}'
        add(prefix + '.PURPOSE', 'api_contract', 'intent_semantics',
            f'{name} 的任务含义：{intent["description"]}。', 'train_schema.json', loc + '/description')
        add(prefix + '.TRANSACTION', 'api_contract', 'transaction_type',
            f'{name} 的 is_transactional={intent["is_transactional"]}。', 'train_schema.json', loc + '/is_transactional')
        for j, slot in enumerate(intent['required_slots']):
            add(prefix + f'.REQUIRED.{slot}', 'api_contract', 'call_precondition',
                f'调用 {name} 前必须绑定 {slot}。', 'train_schema.json', loc + f'/required_slots/{j}',
                method=name, slot=slot)
        for slot, default in sorted(intent['optional_slots'].items()):
            add(prefix + f'.OPTIONAL.{slot}', 'api_contract', 'optional_input_default',
                f'{name} 允许可选约束 {slot}；默认 {default}，用户可覆盖默认值。',
                'train_schema.json', loc + f'/optional_slots/{slot}', method=name, slot=slot, default=default)
        allowed = intent['required_slots'] + list(intent['optional_slots'])
        add(prefix + '.PARAMETERS', 'api_contract', 'allowed_parameter_set',
            f'{name} 的参数名来自其 required/optional 集合 {allowed}。', 'train_schema.json', loc,
            method=name, allowed_parameters=allowed)
        for j, slot in enumerate(intent['result_slots']):
            add(prefix + f'.RESULT.{slot}', 'result_bindings', 'available_result_field',
                f'{name} 的结果接口包含 {slot} 字段；不表示每轮必须输出该字段。',
                'train_schema.json', loc + f'/result_slots/{j}', method=name, slot=slot)
    for i, slot in enumerate(schema['slots']):
        name = slot['name']
        loc = pointer + f'/slots/{i}'
        add(f'SLOT.{name}.MEANING', 'slot_semantics', 'field_semantics',
            f'{name} 的字段含义：{slot["description"]}。', 'train_schema.json', loc + '/description', slot=name)
        if slot['is_categorical']:
            add(f'SLOT.{name}.VALUES', 'slot_semantics', 'categorical_value_set',
                f'{name} 的普通类别值集合为 {slot["possible_values"]}；dontcare 默认另按可选参数规则处理。',
                'train_schema.json', loc + '/possible_values', slot=name, values=slot['possible_values'])

    # These are definitions, not invented universal triggers for the acts.
    system = [
        ('INFORM', '向用户告知一个字段的取值。', 'slot 和 values 必须非空。'),
        ('REQUEST', '向用户询问一个字段的取值。', '必须带 slot；values 可以为空，有值时表示提供给用户的例子。'),
        ('CONFIRM', '在事务性调用前向用户确认一个字段值；该定义不规定所有字段和全部执行路径的固定确认步骤。', None),
        ('OFFER', '向用户提出候选字段值。', '带 slot 及该字段被提出的 values 列表。'),
        ('NOTIFY_SUCCESS', '向用户报告请求成功。', 'slot 和 values 为空。'),
        ('NOTIFY_FAILURE', '向用户报告请求失败。', 'slot 和 values 为空。'),
        ('INFORM_COUNT', '向用户告知满足请求的结果数量；存在该动作不代表每次搜索后必须使用。', '原始格式 slot=count，values 含一个结果数量。'),
        ('OFFER_INTENT', '向用户提议执行一个属于当前服务的 intent。', '原始格式 slot=intent，values 含一个 intent 名称，不是房源名称。'),
        ('REQ_MORE', '系统询问用户是否还需要其它帮助，不是用户索取更多房源。', 'slot 和 values 为空。'),
        ('GOODBYE', '结束对话。', 'slot 和 values 为空。'),
    ]
    start = readme.index('List of possible system acts:')
    end = readme.index('List of possible user acts:', start)
    for act, meaning, payload in system:
        match = re.search(r'\*\s+\*\*' + act + r'\*\*\s+-', readme[start:end])
        assert match, act
        line = readme[:start + match.start()].count('\n') + 1
        add(f'ACT.SYSTEM.{act}.MEANING', 'system_act_semantics', 'system_act_definition', meaning,
            'README.md', f'line:{line}', act=act)
        if payload:
            add(f'ACT.SYSTEM.{act}.PAYLOAD', 'system_act_payload', 'raw_act_payload_contract',
                payload, 'README.md', f'line:{line}', act=act,
                representation_note='Judge meaning after conversion to local benchmark serialization; do not compare raw field positions directly.')
    user = [
        ('INFORM_INTENT', '用户表达希望完成一个任务。'),
        ('NEGATE_INTENT', '用户拒绝系统提议的 intent。'),
        ('AFFIRM_INTENT', '用户接受系统提议的 intent。'),
        ('INFORM', '用户向系统提供字段值。'),
        ('REQUEST', '用户向系统询问字段值，也可询问是否为指定值。'),
        ('AFFIRM', '用户同意系统的提议。'),
        ('NEGATE', '用户否定系统的提议。'),
        ('SELECT', '用户接受被提供的结果，可选择全部被提出的值或指定字段值。'),
        ('REQUEST_ALTS', '用户要求被提供的结果以外的其它结果；与系统 REQ_MORE 区分。'),
        ('THANK_YOU', '用户向系统表示感谢。'),
        ('GOODBYE', '用户结束对话。'),
    ]
    for act, meaning in user:
        match = re.search(r'\*\s+\*\*' + act + r'\*\*\s+-', readme[end:])
        assert match, act
        line = readme[:end + match.start()].count('\n') + 1
        add(f'ACT.USER.{act}.MEANING', 'user_act_semantics', 'user_act_definition', meaning,
            'README.md', f'line:{line}', act=act)
    marker = '**parameters** - A dictionary mapping slot name'
    assert marker in readme
    add('REP.CALL.CANONICAL', 'representation', 'canonical_api_values',
        'service_call.parameters 使用规范化取值；对话动作的自然语言表述与该规范化值不必字面相同。',
        'README.md', 'line:' + str(readme[:readme.index(marker)].count('\n') + 1))
    marker = '**The list of all possible entities is not accessible.**'
    assert marker in challenge
    add('REP.ENTITY_ACCESS', 'representation', 'backend_access_constraint',
        '官方不向系统公开所有实体的完整列表；不能把所有实体/属性预先已知作为执行前提。',
        'dstc8.md', 'line:' + str(challenge[:challenge.index(marker)].count('\n') + 1))
    line = next(i for i, text in enumerate(challenge.splitlines(), 1) if '2019 has been treated' in text)
    add('REP.RELATIVE_DATE', 'representation', 'relative_date_anchor',
        '原始 SGD 对话中的相对日期以 2019-03-01 为 today 的时间基准。', 'dstc8.md', f'line:{line}')
    assert len({r['id'] for r in rules}) == len(rules)
    meta = {'service': service, 'reference_items': len(rules), 'by_group': dict(Counter(r['group'] for r in rules)),
        'source_type': 'Public official schema, README definitions, and challenge documentation',
        'reference_status': 'development checklist, not an official exhaustive guideline benchmark',
        'skill_artifacts_read': False, 'dialogues_read': False, 'llm_calls': 0,
        'sources': [{'file': name, 'sha256': hashlib.sha256((source_dir / name).read_bytes()).hexdigest()}
                    for name in ('train_schema.json', 'README.md', 'dstc8.md')]}
    return rules, meta


def write_reference(source_dir: Path, output_dir: Path, service: str):
    rules, meta = build(source_dir, service)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'reference.json').write_text(json.dumps({'metadata': meta, 'items': rules}, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output_dir / 'coverage_annotation.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['reference_id', 'group', 'reference_statement', 'method', 'coverage_status', 'source_file', 'source_lines', 'supporting_clause', 'conflict_clause', 'rationale']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rule in rules:
            for method in ('awm', 'trace2skill', 'ours'):
                writer.writerow({'reference_id': rule['id'], 'group': rule['group'], 'reference_statement': rule['statement'],
                                 'method': method, 'coverage_status': 'pending'})
    lines = ['# Homes_1 官方知识参考清单（开发版）', '',
        f'共 {len(rules)} 个固定参考项。来源只含官方 train/schema.json、README、dstc8.md；构建脚本不读取待评 skill 或任何对话。', '',
        '这是作者整理的 official-document-derived reference checklist，不是官方发布的完整 Homes SOP 或官方评测基准。', '',
        '| 类别 | 参考项数 |', '|---|---:|']
    lines += [f'| {group} | {count} |' for group, count in meta['by_group'].items()]
    lines += ['', '## 统计协议', '',
        '- 对三个方法使用同一套 reference ID 作分母，不以各方法抽取项数作分母。同一参考项命中多处也只计一次。',
        '- coverage_status：covered / partial / absent / contradicted / unresolved_conflict / unverifiable。只有完整支持适用条件和约束、且无同作用域的未解决冲突，才计 strict covered。',
        '- 词名出现、复制一个具体实例、泛称“遵循 schema”均不自动算显式知识覆盖；必须给出能支持该参考项的原文约束及位置。',
        '- 同一语句可覆盖多个参考项；无需强行使各方法产物的句子数相同。',
        '- Strict coverage = covered 参考项数 / 固定参考项数。另报 partial、contradicted、unresolved_conflict、unverifiable，均不按完整覆盖计。',
        '- 优先分组报告覆盖率；result_bindings 等字段数量较多，不宜让未加权总分掩盖动作/流程知识。需要总分时同时给 macro（各组均值）和 micro（全部参考项均值），预先固定组权重。',
        '- api_contract、slot_semantics、result_bindings 是接口层知识；system_act_semantics、user_act_semantics 是行为含义；system_act_payload、representation 是表示契约。不能把三者混成“流程发现”贡献。',
        '- 只有 CONFIRM 定义和 API precondition 等具有局部过程约束；这些公开资料没有给出每个分支都必须遵循的完整 Homes 流程。',
        '- 系统 OFFER_INTENT 在原始 SGD 和本地 ConvLab 转换版字段位置不同，先规范化表示；不能把 slot=ScheduleVisit 直接判错。',
        '- 最终产物覆盖率与相对初始资源的新增覆盖分开。若 schema 已供各方法使用，接口知识覆盖不能单独证明挖掘能力。',
        '- 不从已看过的方法输出反推参考项；当前作者已查看该场景的部分产物，因此本开发版也不声称盲测或预注册。正式使用前冻结清单和判定准则。',
        '- TRAIN 归纳的经验规律可做单独扩展集，必须有支持/反例及来源标记；不能改名为官方强制 guideline。', '',
        '## 尚未完成', '', 'coverage_annotation.csv 是三方法逐项判定模板，全部 pending。本目录没有生成覆盖率或伪造实验分数。', '',
        '## 参考项', '']
    for r in rules:
        lines += [f'### {r["id"]}', '', r['statement'], '', f'来源：`{r["source_file"]}`，`{r["source_locator"]}`。类别：`{r["group"]}`。', '']
    (output_dir / 'REFERENCE.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/sgd_homes_reference_20261025_v1'))
    parser.add_argument('--service', default='Homes_1')
    args = parser.parse_args()
    write_reference(args.output_dir / 'sources', args.output_dir, args.service)
