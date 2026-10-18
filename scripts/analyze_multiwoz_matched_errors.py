#!/usr/bin/env python3
"""Read-only error audit of frozen predictions; no model calls or rescoring in place."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.sgd_adapter import action_counter, score_policies, slot_signature
from eval_tod.multiwoz_matched_executor import load_kb, query_records

METHODS = ('awm', 'trace2skill', 'ours')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def key(act):
    return act['intent'], '' if act.get('domain') == 'general' else act.get('domain', ''), act.get('slot') or ''


def shape(policy):
    return Counter(key(a) for a in policy['acts'])


def action_ok(row):
    return not row['prediction'].get('_execution_error') and action_counter(row['gold']) == action_counter(row['prediction'])


def joint_ok(row):
    return action_ok(row) and slot_signature(row['gold']) == slot_signature(row['prediction'])


def normalize(policy, *, binary=(), empty=False):
    policy = deepcopy(policy)
    for act in policy['acts']:
        if key(act) in binary or (empty and act.get('value', None) in (None, '')):
            act.pop('value', None)
    return policy


def category(row):
    if not action_ok(row):
        return 'action_wrong'
    if joint_ok(row):
        return 'exact'
    gold, pred = shape(row['gold']), shape(row['prediction'])
    missing, extra = gold - pred, pred - gold
    if missing and extra:
        return 'missing_and_extra_slots'
    if missing:
        return 'missing_slots_only'
    if extra:
        return 'extra_slots_only'
    return 'value_only'


def run(root):
    manifest = read(root / 'manifest.json')
    original = read(Path(manifest['source_root']) / 'ours_retry_11_12/data_manifest.json')
    source = Path(original['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_sha256']
    raw = read(source)
    dialogues = {d['dialogue_id']: d for d in raw}
    train_ids = set(manifest['selected_ids']['train'])
    kinds = defaultdict(Counter)
    train_bundles = defaultdict(list)
    for d in raw:
        if d['dialogue_id'] not in train_ids:
            continue
        for turn in d['turns']:
            if turn['speaker'] != 'system':
                continue
            acts = []
            for kind, group in turn['dialogue_acts'].items():
                for act in group:
                    kinds[key(act)][kind] += 1
                    acts.append(act)
            if acts:
                heads = tuple(sorted({key(a)[:2] for a in acts}))
                train_bundles[heads].append(tuple(sorted(key(a) for a in acts)))
    binary = {k for k, v in kinds.items() if set(v) == {'binary'}}
    rows = {m: [json.loads(line) for line in (root / m / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
            for m in METHODS}
    ids = [(r['dialogue_id'], r['turn_index']) for r in rows['ours']]
    assert len(ids) == len(set(ids)) == manifest['validation_turns']
    cards = read(root / 'corrected_action_cards.json')['cards']
    observations = [read(root / 'observations' / f'case_{i:04d}.json')['observation'] for i in range(len(ids))]
    result = {'root': str(root), 'protocol': manifest['protocol'], 'analysis_only': True,
              'train_only_binary_keys': [list(k) for k in sorted(binary)], 'methods': {}, 'paired': {},
              'ours_payload_errors': [], 'ours_action_errors': []}
    hashes = {m: hashlib.sha256((root / m / 'predictions.jsonl').read_bytes()).hexdigest() for m in METHODS}
    for method, records in rows.items():
        assert [(r['dialogue_id'], r['turn_index']) for r in records] == ids
        assert [r['gold'] for r in records] == [r['gold'] for r in rows['ours']]
        gold, pred = [r['gold'] for r in records], [r['prediction'] for r in records]
        primary = score_policies(gold, pred)
        audits = {}
        for name, kwargs in [('empty_omission', {'empty': True}), ('train_binary_value_omission', {'binary': binary})]:
            ng, np = [normalize(p, **kwargs) for p in gold], [normalize(p, **kwargs) for p in pred]
            audits[name] = {'scores': score_policies(ng, np), 'recovered_case_indices': [
                i for i, (g, p) in enumerate(zip(ng, np)) if action_ok(records[i])
                and not joint_ok(records[i]) and slot_signature(g) == slot_signature(p)]}
        by_values = {}
        for name, predicate in [('gold_has_value', lambda p: any('value' in a for a in p['acts'])),
                                ('gold_has_no_values', lambda p: all('value' not in a for a in p['acts'])),
                                ('gold_all_slotless', lambda p: all(not a['slot'] for a in p['acts']))]:
            indices = [i for i, r in enumerate(records) if predicate(r['gold'])]
            by_values[name] = {'total': len(indices), 'action_correct': sum(action_ok(records[i]) for i in indices),
                               'joint_correct': sum(joint_ok(records[i]) for i in indices)}
        binary_errors = []
        for i, r in enumerate(records):
            raw_turn = dialogues[r['dialogue_id']]['turns'][r['turn_index']]
            gold_binary = {key(a) for a in raw_turn['dialogue_acts'].get('binary', [])}
            bad = [a for a in r['prediction']['acts'] if key(a) in gold_binary and 'value' in a]
            if bad:
                binary_errors.append({'case_index': i, 'action_correct': action_ok(r), 'acts': bad})
        result['methods'][method] = {'primary': primary, 'categories': dict(Counter(map(category, records))),
            'normalization_audits': audits, 'by_gold_value_presence': by_values, 'binary_value_mismatches': binary_errors,
            'explicit_empty_values': {name: sum(a.get('value', None) == '' for p in policies for a in p['acts'])
                                      for name, policies in [('gold', gold), ('prediction', pred)]}}
    correct_sets = {m: {i for i, r in enumerate(records) if action_ok(r)} for m, records in rows.items()}
    for baseline in METHODS[:2]:
        common = sorted(correct_sets['ours'] & correct_sets[baseline])
        result['paired'][baseline] = {'common_action_correct': len(common), 'case_indices': common,
            'ours_joint_correct': sum(joint_ok(rows['ours'][i]) for i in common),
            'baseline_joint_correct': sum(joint_ok(rows[baseline][i]) for i in common),
            'ours_only_correct': [i for i in common if joint_ok(rows['ours'][i]) and not joint_ok(rows[baseline][i])],
            'baseline_only_correct': [i for i in common if joint_ok(rows[baseline][i]) and not joint_ok(rows['ours'][i])]}
    train_ambiguity = []
    for cid, card in cards.items():
        heads = tuple(sorted(tuple(h) for h in card['heads']))
        samples = train_bundles[heads]
        hits = [i for i, r in enumerate(rows['ours']) if cid in r['diagnostic']['selected_card_ids'] and action_ok(r)]
        if hits:
            train_ambiguity.append({'card_id': cid, 'heads': card['heads'], 'train_turns': len(samples),
                'distinct_slot_multisets': len(set(samples)), 'train_context_examples_seen_by_card_completion': min(4, len(samples)),
                'action_correct_eval_turns': len(hits), 'payload_errors': sum(not joint_ok(rows['ours'][i]) for i in hits)})
    result['train_card_ambiguity'] = sorted(train_ambiguity, key=lambda x: -x['payload_errors'])
    action_categories, missed_heads, extra_heads = Counter(), Counter(), Counter()
    global_only = 0
    for i, r in enumerate(rows['ours']):
        if action_ok(r):
            continue
        gold_heads, pred_heads = set(action_counter(r['gold'])), set(action_counter(r['prediction']))
        missing, extra = gold_heads - pred_heads, pred_heads - gold_heads
        kind = 'missing_and_extra' if missing and extra else 'missing_only' if missing else 'extra_only'
        action_categories[kind] += 1
        missed_heads.update(missing)
        extra_heads.update(extra)
        only_general = {h for h in gold_heads if h[1]} == {h for h in pred_heads if h[1]}
        global_only += only_general
        result['ours_action_errors'].append({'case_index': i, 'dialogue_id': r['dialogue_id'], 'turn_index': r['turn_index'],
            'category': kind, 'missing_heads': sorted(missing), 'extra_heads': sorted(extra), 'only_general_heads_differ': only_general})
    result['ours_action_error_summary'] = {'categories': dict(action_categories), 'only_general_heads_differ': global_only,
        'missing_heads': [{'head': k, 'count': n} for k, n in missed_heads.most_common()],
        'extra_heads': [{'head': k, 'count': n} for k, n in extra_heads.most_common()]}
    result['ours_action_correct_with_exact_card'] = sum(bool(r['diagnostic']['selected_card_ids']) for r in rows['ours'] if action_ok(r))
    for i, r in enumerate(rows['ours']):
        if not action_ok(r) or joint_ok(r):
            continue
        d = dialogues[r['dialogue_id']]
        t = r['turn_index']
        context = '\n'.join(f"{turn['speaker'].upper()}: {turn['utterance']}" for turn in d['turns'][:t])[-12000:]
        missing, extra = shape(r['gold']) - shape(r['prediction']), shape(r['prediction']) - shape(r['gold'])
        refs = [a for a in r['gold']['acts'] if a['slot'] == 'ref' and 'value' in a]
        exposed = context + '\n' + json.dumps(observations[i]['observations'], ensure_ascii=False)
        result['ours_payload_errors'].append({'case_index': i, 'dialogue_id': r['dialogue_id'], 'turn_index': t,
            'category': category(r), 'missing_slot_occurrences': list(missing.elements()), 'extra_slot_occurrences': list(extra.elements()),
            'gold': r['gold'], 'predictions': {m: rows[m][i]['prediction'] for m in METHODS},
            'baseline_action_correct': {m: action_ok(rows[m][i]) for m in METHODS[:2]},
            'baseline_joint_correct': {m: joint_ok(rows[m][i]) for m in METHODS[:2]},
            'selected_cards': [cards[cid] for cid in r['diagnostic']['selected_card_ids']],
            'prior_context': context, 'heldout_target_utterance_FOR_ANALYSIS_ONLY': d['turns'][t]['utterance'],
            'raw_annotation_FOR_ANALYSIS_ONLY': d['turns'][t]['dialogue_acts'],
            'shared_observation': observations[i], 'gold_ref_absent_from_visible_input': [a for a in refs if a['value'].casefold() not in exposed.casefold()]})
    kb = load_kb(str(source.parent))
    truncation_evidence = []
    for i, r in enumerate(rows['ours']):
        if not action_ok(r) or joint_ok(r):
            continue
        for obs in observations[i]['observations']:
            if not obs['records_truncated']:
                continue
            candidates = query_records(kb, obs['domain'], obs['constraints'])
            for act in r['gold']['acts']:
                if act['domain'] != obs['domain'] or 'value' not in act:
                    continue
                value = act['value'].casefold()
                all_matches = [n for n, rec in enumerate(candidates) if str(rec.get(act['slot'], '')).casefold() == value]
                if all_matches and min(all_matches) >= len(obs['records']):
                    truncation_evidence.append({'case_index': i, 'act': act, 'first_matching_record_index': min(all_matches),
                                               'returned_records': len(obs['records']), 'matching_records': len(candidates)})
    result['exact_gold_value_available_in_query_but_truncated'] = truncation_evidence
    result['source_prediction_sha256'] = hashes
    assert all(hashlib.sha256((root / m / 'predictions.jsonl').read_bytes()).hexdigest() == hashes[m] for m in METHODS)
    (root / 'error_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(root, result)
    print(json.dumps({'categories': {m: v['categories'] for m, v in result['methods'].items()},
                      'paired': result['paired'], 'train_card_ambiguity': result['train_card_ambiguity'][:6],
                      'truncation': truncation_evidence}, ensure_ascii=False, indent=2))


def write_report(root, result):
    labels = {'awm': 'AWM', 'trace2skill': 'Trace2Skill', 'ours': 'Ours'}
    lines = ['# MultiWOZ 最新匹配评测：空值与错误 case 审计', '',
        '分析对象：228 个 validation agent turn；冻结训练资源、共同只读 DB、先选 action 再填 slot。未重训练或重新调用模型；原始预测与主指标保持原样。', '',
        '## 1. 空值对齐结论', '',
        '- 三组 gold 与 prediction 的显式 `value=""` 均为 0。空字符串/缺省键规范化不会改变主指标。',
        '- 但 binary act 还有遗漏：原始标注的 `inform(hotel,internet)` / `parking` 本来没有 value，生成器与部分卡片会补 `yes`。这与空字符串是两种问题。',
        '- 以下规范化审计只删除 TRAIN 中始终为 binary 的 `(intent,domain,slot)` 的 value，保留 slot、act 数量与 action；不读取验证标签制定规则。', '',
        '| 方法 | Action 正确 | 原 Joint 正确 | binary 规范化后正确（离线诊断） | 原条件 Slot exact |',
        '|---|---:|---:|---:|---:|']
    for method, v in result['methods'].items():
        c = v['primary']['counts']; corrected = v['normalization_audits']['train_binary_value_omission']['scores']['counts']['joint_exact']
        lines.append(f"| {labels[method]} | {c['action_exact']}/228 | {c['joint_exact']}/228 | {corrected}/228 | {100*v['primary']['slot_exact_given_action']:.2f}% |")
    lines += ['', 'binary 差异只使 AWM、Trace2Skill 各少算 1 个完整正确 turn；Ours 没有因此单独损失一个完整正确 turn。因此它需要修，但不能解释 Ours 的主要结果。', '',
        '## 2. 相同样本上的 payload 比较', '',
        '| 比较 | 双方 Action 都正确的相同 turn | Ours Slot exact | 对方 Slot exact |', '|---|---:|---:|---:|']
    for method, pair in result['paired'].items():
        n = pair['common_action_correct']
        lines.append(f"| Ours / {labels[method]} | {n} | {pair['ours_joint_correct']}/{n} | {pair['baseline_joint_correct']}/{n} |")
    lines += ['', '原条件指标的分母分别是 65、57、71 个不同 turn，不能直接解释成 Ours 在相同样本上填 slot 更差。这些共同正确子集也有选择偏差，样本很少；它们不是 gold-action oracle 实验。', '',
        '| 方法 | 含 value 的 gold turn：Action 正确 / Joint 正确 | gold 全部不带 value：Action 正确 / Joint 正确 |',
        '|---|---:|---:|']
    for method, v in result['methods'].items():
        a = v['by_gold_value_presence']['gold_has_value']; b = v['by_gold_value_presence']['gold_has_no_values']
        lines.append(f"| {labels[method]} | {a['action_correct']} / {a['joint_correct']} | {b['action_correct']} / {b['joint_correct']} |")
    lines += ['', 'Ours 正确识别了更多需要输出值的回合，同时这些回合更难做到整套 slot/value 与唯一参考标注逐字一致。', '',
        f"## 3. Ours：Action 正确后的 {len(result['ours_payload_errors'])} 个 payload 错误", '',
        '| 互斥类别 | turn 数 |', '|---|---:|---:|']
    for name, cat in [('仅多填 slot', 'extra_slots_only'), ('仅漏填 slot', 'missing_slots_only'),
                      ('同时多填与漏填', 'missing_and_extra_slots'), ('slot 名称和重数一致，仅 value/是否带值不同', 'value_only')]:
        lines.append(f"| {name} | {result['methods']['ours']['categories'].get(cat, 0)} |")
    lines += ['', '以上前三类只检查 `(intent,domain,slot)` 的多重集；这些回合仍可能同时存在值错误。8 个 value-only 中，6 个是未观测的预订号，另 2 个是邮编与 `choice`。', '',
        '## 4. 代表 case', '']
    notes = {
        16: ('多填字段与卡片条件不足', '参考只含 day、leave at 和 slotless offerbook。Ours 填了 8 个 train 属性。选中卡片把全部 train 字段写成平铺的可用清单，未说明本轮应选哪几个；共同合同虽要求不要输出所有 DB 字段，实际输出仍展开了记录。'),
        155: ('request 粒度没有学到参考策略', '参考同时询问 day 与 leave at，Ours 只问 day；这里所有 act 都没有 value，完全排除空值因素。AWM 此回合完全正确。不过一次问一个还是两个条件也涉及对话策略的多解性。'),
        211: ('相同地址被标成两个 span', '参考把完整地址拆成两条同名 address act；Ours 把完整 DB 地址放在一条 act 中。语义地址正确，但严格多重集不相等。这属于标注表示差异，不能直接当作事实错误。'),
        37: ('参考预订号不可由当前输入确定', '参考 ref 不在先前对话或只读 DB 里。Ours 输出了虚构 ref；这是生成错误，同时现有工具协议也无法生成该参考预订号。卡片仍要求从“booking result”取 ref，但执行器没有 booking 工具。'),
        33: ('计数与自然语言表述不一致', '共同 DB 命中 5 条，Ours choice=5；参考 choice=several。不能把逐字不等直接解释成错误计数。'),
        156: ('DB 截断使参考实体不可见', '满足条件有 15 班，执行器按原 DB 顺序只返回前 5 班。参考 TR6539/20:08 在本次返回中不可见；Ours 选择另一个满足约束的班次，同时多填了 slot。'),
        4: ('实体别名导致查询零结果后臆造', '先前对话提到 Cherry Hinton Water Play，随后系统说位置在 Cherry Hinton Hall。planner 用 Hall 当实体名查出 0 条；Ours 仍给出错误邮编。这里是查询实体消歧和无证据时的输出约束问题。'),
    }
    errors = {e['case_index']: e for e in result['ours_payload_errors']}
    for i, (title, note) in notes.items():
        e = errors[i]
        lines += [f"### case_{i:04d}：{title}", '', f"`{e['dialogue_id']}`，原 turn index={e['turn_index']}。{note}", '',
                  '**Gold（仅用于分析）**', '', '```json', json.dumps(e['gold']['acts'], ensure_ascii=False), '```', '',
                  '**Ours**', '', '```json', json.dumps(e['predictions']['ours']['acts'], ensure_ascii=False), '```', '']
    lines += ['## 5. 训练与卡片证据', '',
        f"Ours 所有 {result['ours_action_correct_with_exact_card']} 个 action 正确回合都加载了所选完整 bundle 的精确 action card。当前这些错误没有发现之前那种未重新读取选中 card 的问题。", '',
        '当前节点是完整 `(intent,domain)` 集合，slot 不进入节点标识。同一 `inform(hotel)` 节点覆盖查星级、查电话、报搜索结果等不同子任务。卡片补全只见每节点最先出现的最多 4 个带上下文样本，其余证据是逐字段频数，丢失了 slot 共现、先后提问和上下文条件。', '',
        '| Action heads | TRAIN turn | 不同 slot 多重集 | 补全看到的上下文例子 | 本轮 Action 正确 / payload 错误 |',
        '|---|---:|---:|---:|---:|']
    for c in result['train_card_ambiguity'][:7]:
        heads = ', '.join(f'{h[0]}:{h[1]}' for h in c['heads'])
        lines.append(f"| {heads} | {c['train_turns']} | {c['distinct_slot_multisets']} | {c['train_context_examples_seen_by_card_completion']} | {c['action_correct_eval_turns']} / {c['payload_errors']} |")
    action = result['ours_action_error_summary']
    lines += ['', '所有这些证据都只能支持定位问题，不能证明单一原因。当前 116 张卡片“完整”仅表示字段齐全，不代表条件、事实来源及标注类型已经语义正确；冻结训练资源也尚未按新的 DB 协议重新 refine。', '',
        '## 6. Action 错误补充', '',
        f"另有 {len(result['ours_action_errors'])} 个 action 不完全正确的回合：只多选 {action['categories'].get('extra_only', 0)}、只漏选 {action['categories'].get('missing_only', 0)}、两者都有 {action['categories'].get('missing_and_extra', 0)}。其中 {action['only_general_heads_differ']} 个只差 general acts；领域 action 集合相同。", '',
        '最常见的漏选是 reqmore（38 次），最常见的多选是 request(hotel)（42 次）。部分参考的礼貌语和询问顺序存在多解性，仍须与实际多选/漏选领域动作区分。完整 case 索引见 JSON 的 ours_action_errors。', '',
        '## 7. 修复状态与建议顺序', '',
        '1. 已在后续公共匹配执行器补充 TRAIN-only binary 类型规范化：保留有 slot 无 value 的 act，仅删除 binary 的 value。混合标注类型不会整体删值。协议升至 v2_train_binary，旧 v1 缓存不会被静默复用。未付费重跑或替换本轮旧分数。相关 12 项测试通过，提取的 51 个 value-free act 键与独立原始 TRAIN 标注审计完全一致。',
        '2. 使用标注类型分别处理 categorical、non-categorical span 和 binary；整轮精确匹配之外，另报 slot 名称、值规范化与不可观测 ref 分项。正式主指标需固定协议并三组共同重评，不能只改 Ours 分数。',
        '3. 卡片编译选择覆盖不同 slot 组合的训练例子，保留组合和条件；避免频数平铺成所有字段清单。',
        '4. 先解决无 booking observation 和 DB 返回候选截断的问题，再在相同可观测输入上重新 refine；从训练集的错误定位样本开始验证，不能拿这里的验证 case 写专用规则。', '',
        '这只是一个固定 seed、32 个对话的描述性错误分析，同一对话的多个 turn 并不独立；未作显著性或泛化优势结论。', '',
        '完整 payload 错误的 prior context、三组预测、共同 DB、选中卡片和原始标注见 [error_analysis.json](error_analysis.json)。所有读取的预测文件 SHA256 记录在 JSON 中；分析脚本验证原文件未改变。', '']
    (root / 'error_analysis.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/multiwoz_hotel_matched_20261018_035410')
    run(parser.parse_args().root)
