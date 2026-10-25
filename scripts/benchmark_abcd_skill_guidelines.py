"""Portable ABCD guideline-to-skill coverage: reference, evaluate, adjudicate.

Reference and dry-run use only stdlib. Live evaluation optionally reuses the
company llm.py; local relay is an explicit opt-in. No training or agent rollout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skill_mining.guideline_coverage import (VERSION, FINE_TITLES, build_reference,
    coverage_messages, validate_assessments, summarize, digest, at_pointer)
from scripts.extract_skill_knowledge import Gate, dump, read, read_lines, sha


def reference(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    flows = list(FINE_TITLES) if args.all else [x.strip() for x in args.subflows.split(',') if x.strip()]
    raw = {name: read(args.data_dir / name) for name in ('guidelines.json', 'ontology.json', 'kb.json')}
    value = build_reference(raw['guidelines.json'], raw['ontology.json'], raw['kb.json'], flows)
    dump(args.output_dir / 'reference.json', value)
    sources = args.output_dir / 'sources'
    sources.mkdir()
    for name in raw:
        (sources / name).write_bytes((args.data_dir / name).read_bytes())
    with (args.output_dir / 'reference_review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['reference_id', 'flow', 'fine_task', 'kind', 'button', 'eligibility', 'reference_issue', 'clauses', 'condition_quotes']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for unit in value['units']:
            writer.writerow({**unit, 'clauses': json.dumps(unit['clauses'], ensure_ascii=False),
                             'condition_quotes': json.dumps(unit['condition_quotes'], ensure_ascii=False)})
    summary = {'subflows': flows, 'reference_units': len(value['units']),
        'eligible': sum(u['eligibility'] == 'eligible' for u in value['units']),
        'conditional_action': sum(u['eligibility'] == 'eligible' and u['kind'] == 'conditional_action' for u in value['units']),
        'reference_review_required': sum(u['eligibility'] != 'eligible' for u in value['units']), 'llm_calls': 0}
    dump(args.output_dir / 'reference_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_inputs(reference_path, source_dir, rules_dir, origins):
    reference_data = read(reference_path)
    if reference_data['protocol'] != VERSION:
        raise ValueError('Wrong reference protocol')
    official = {key: read(reference_path.parent / 'sources' / filename) for key, filename in
                [('guidelines', 'guidelines.json'), ('ontology', 'ontology.json'), ('kb', 'kb.json')]}
    if {key: digest(value) for key, value in official.items()} != reference_data['source_hashes']:
        raise ValueError('Official reference sources changed')
    reference_ids = [u['reference_id'] for u in reference_data['units']]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError('Duplicate official reference ID')
    for unit in reference_data['units']:
        if unit['flow'] not in reference_data['subflows'] or unit['fine_task'] not in FINE_TITLES[unit['flow']] or FINE_TITLES[unit['flow']][unit['fine_task']] != unit['fine_title']:
            raise ValueError('Official fine-task scope changed')
        if unit['eligibility'] not in {'eligible', 'reference_review_required'}:
            raise ValueError('Unknown reference eligibility')
        if unit['clauses'] != [ref['quote'] for ref in unit['source_refs']]:
            raise ValueError('Reference clauses do not match official source quotes')
        for ref in unit['source_refs']:
            if ref['file'] != 'guidelines.json' or at_pointer(official['guidelines'], ref['pointer']) != ref['quote']:
                raise ValueError('Official reference citation changed')
    inventory = read(source_dir / 'inventory.json')
    for filename, field in [('documents.jsonl', 'documents_sha256'), ('chunks.jsonl', 'chunks_sha256')]:
        if sha(source_dir / filename) != inventory[field]:
            raise ValueError('Frozen inventory changed: ' + filename)
    extraction = read(rules_dir / 'summary.json')
    if not extraction['complete']:
        details = extraction.get('failed_chunk_details', [])
        raise ValueError('Extraction is incomplete; resume extraction before coverage evaluation. '
                         f"Successful={extraction.get('successful_chunks')}, failed={extraction.get('failed_chunks')}. "
                         + json.dumps(details[:5], ensure_ascii=False))
    if extraction['source_inventory_sha256'] != sha(source_dir / 'inventory.json'):
        raise ValueError('Rules belong to a different source inventory')
    docs = {d['document_id']: d for d in read_lines(source_dir / 'documents.jsonl')}
    methods = sorted(inventory['counts'])
    flows = reference_data['subflows']
    if not methods or not flows:
        raise ValueError('Empty method/flow selection')
    for method in methods:
        for flow in flows:
            selected = [d for d in docs.values() if d['method'] == method and d['family'] == flow and d['snapshot'] == 'final']
            if not selected:
                raise ValueError(f'Missing final resources for {method}/{flow}; do not silently drop this method/flow')
    allowed_origins = set(origins.split(','))
    if not allowed_origins <= {'procedural_rule', 'executor_contract'}:
        raise ValueError('Only explicit procedural rules/contracts can be evaluated; examples are not knowledge')
    candidates = {(m, f): [] for m in methods for f in flows}
    seen = set()
    for rule in read_lines(rules_dir / 'rules.jsonl'):
        if rule['rule_id'] in seen:
            raise ValueError('Duplicate rule ID')
        seen.add(rule['rule_id'])
        for cite in rule['citations']:
            doc = docs[cite['document_id']]
            if doc['method'] != rule['method'] or doc['family'] != rule['family'] or doc['snapshot'] != rule['snapshot']:
                raise ValueError('Cross-method/flow/snapshot citation')
            if cite['file_sha256'] != doc['file_sha256'] or cite['text_sha256'] != doc['text_sha256']:
                raise ValueError('Source identity mismatch')
            lo, hi = cite['start_char'], cite['end_char']
            if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= len(doc['text']) or doc['text'][lo:hi] != cite['quote']:
                raise ValueError('Modified or invalid literal rule citation')
        key = rule['method'], rule['family']
        if key in candidates and rule['snapshot'] == 'final' and rule['origin'] in allowed_origins:
            if not rule['citations']:
                raise ValueError('Rule without source evidence')
            candidates[key].append(rule)
    return reference_data, methods, candidates


def export_results(out, reference_data, methods, candidates, batch_paths, *, reviewed=False, reviewer=None):
    rows = []
    for path in batch_paths:
        saved = read(path)
        if saved['status'] != 'complete':
            continue
        units = [u for u in reference_data['units'] if u['reference_id'] in saved['reference_ids']]
        checked = validate_assessments(saved['raw'], units, candidates[saved['method'], saved['flow']])
        for row in checked:
            rows.append({**row, 'method': saved['method'], 'flow': saved['flow'],
                         'review_status': 'reviewed' if reviewed else 'model_proposal_pending_review'})
    summary = summarize(reference_data, methods, rows)
    summary['label_status'] = 'reviewed' if reviewed else 'model_proposal_pending_review'
    summary['reviewer'] = reviewer
    summary['interpretation'] = 'Coverage of explicit extracted rules against fixed official-text units; not task success or exhaustive artifact recall.'
    dump(out / ('reviewed_assessments.json' if reviewed else 'proposed_assessments.json'), rows)
    dump(out / ('reviewed_summary.json' if reviewed else 'provisional_summary.json'), summary)
    return rows, summary


def review_template(out, reference_data, methods, rows):
    indexed = {(r['method'], r['reference_id']): r for r in rows}
    prior = {}
    path = out / 'coverage_review.csv'
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for entry in csv.DictReader(stream):
                key = entry['method'], entry['reference_id']
                if key in prior:
                    raise ValueError('Duplicate row in existing review spreadsheet')
                prior[key] = entry
    with (out / 'coverage_review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['method', 'reference_id', 'flow', 'fine_task', 'kind', 'reference_clauses', 'proposed_status', 'proposed_reason',
                  'reviewed_status', 'reviewed_reason', 'support_rule_ids', 'conflict_rule_ids', 'condition_rule_ids']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in methods:
            for unit in reference_data['units']:
                if unit['eligibility'] != 'eligible':
                    continue
                row = indexed.get((method, unit['reference_id']), {})
                entry = {'method': method, 'reference_id': unit['reference_id'], 'flow': unit['flow'], 'fine_task': unit['fine_task'],
                    'kind': unit['kind'], 'reference_clauses': json.dumps(unit['clauses'], ensure_ascii=False),
                    'proposed_status': row.get('status', 'unassessed'), 'proposed_reason': row.get('reason', ''),
                    **{field: json.dumps(row.get(field, []), ensure_ascii=False) for field in ('support_rule_ids', 'conflict_rule_ids', 'condition_rule_ids')}}
                old = prior.get((method, unit['reference_id']), {})
                if old.get('reviewed_status') or old.get('reviewed_reason'):
                    for field in ('reviewed_status', 'reviewed_reason', 'support_rule_ids', 'conflict_rule_ids', 'condition_rule_ids'):
                        entry[field] = old.get(field, '')
                writer.writerow(entry)


def evaluate(args):
    ref, methods, candidates = load_inputs(args.reference, args.source_dir, args.rules_dir, args.origins)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'version': VERSION, 'reference_sha256': sha(args.reference), 'documents_sha256': sha(args.source_dir / 'documents.jsonl'),
        'rules_sha256': sha(args.rules_dir / 'rules.jsonl'), 'origins': args.origins, 'provider': args.provider,
        'model': args.model, 'workflow_id': args.workflow_id, 'batch_size': args.batch_size,
        'script_sha256': sha(Path(__file__)), 'core_sha256': sha(ROOT / 'skill_mining/guideline_coverage.py')}
    if (out / 'protocol.json').exists() and read(out / 'protocol.json') != protocol:
        raise ValueError('Coverage protocol/input changed; use a new output directory')
    dump(out / 'protocol.json', protocol)
    dump(out / 'reference_snapshot.json', ref)
    dump(out / 'candidate_snapshot.json', [{'method': m, 'flow': f, 'rules': r} for (m, f), r in candidates.items()])
    tasks = []
    for method in methods:
        for flow in ref['subflows']:
            units = [u for u in ref['units'] if u['flow'] == flow and u['eligibility'] == 'eligible']
            for i in range(0, len(units), args.batch_size):
                tasks.append((method, flow, i, units[i:i+args.batch_size]))
    manifest = {'methods': methods, 'subflows': ref['subflows'], 'total_batches': len(tasks),
        'candidate_counts': {m: {f: len(candidates[m, f]) for f in ref['subflows']} for m in methods},
        'expected_assessments': len(methods) * sum(u['eligibility'] == 'eligible' for u in ref['units']),
        'excluded_reference_units': [u['reference_id'] for u in ref['units'] if u['eligibility'] != 'eligible']}
    dump(out / 'input_manifest.json', manifest)
    dump(out / 'input_freeze.json', {name: sha(out / name) for name in
        ['protocol.json', 'reference_snapshot.json', 'candidate_snapshot.json', 'input_manifest.json']})
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.provider == 'existing' and args.workflow_id:
        os.environ['SKILLMINING_LLM_BACKEND'] = 'workflow'
    import llm
    if args.provider == 'shubiaobiao':
        from eval_tod.sgd_local_config import configure, DEFAULT_ENV
        configure(args.env_file or DEFAULT_ENV, args.model)
    workflow_config = None
    if args.workflow_id:
        if args.provider != 'existing' or not callable(getattr(llm, 'get_llm_config', None)):
            raise ValueError('workflow-id requires existing company workflow-aware llm.py')
        from config import LLM_CONFIG
        workflow_config = {**LLM_CONFIG, 'provider': 'workflow', 'workflow_id': args.workflow_id}
    gate = Gate(args.qps)
    def one(task):
        method, flow, i, units = task
        path = out / 'batches' / f'{method}_{flow}_{i:04d}.json'
        rules = candidates[method, flow]
        if path.exists() and read(path).get('status') == 'complete':
            old = read(path)
            validate_assessments(old['raw'], units, rules)
            return old
        base = coverage_messages(units, rules)
        if len(json.dumps(base, ensure_ascii=False)) > args.max_prompt_chars:
            raise ValueError('Prompt exceeds explicit input budget; reduce batch-size or inspect rules, never silently truncate')
        attempts = read(path).get('attempts', []) if path.exists() else []
        prompt = base
        for _ in range(args.retries):
            raw = ''
            try:
                if rules:
                    gate.wait()
                    kwargs = {'model': args.model, 'temperature': 0., 'call_tag': 'abcd_guideline_coverage'}
                    if workflow_config is not None:
                        kwargs['config'] = workflow_config
                    raw = llm.chat(prompt, **kwargs)
                else:
                    raw = json.dumps({'assessments': [{'reference_id': u['reference_id'], 'status': 'absent',
                        'reason': 'Complete extraction produced no eligible explicit rules for this method/flow.',
                        'support_rule_ids': [], 'conflict_rule_ids': [], 'condition_rule_ids': []} for u in units]})
                checked = validate_assessments(raw, units, rules)
                saved = {'status': 'complete', 'method': method, 'flow': flow, 'reference_ids': [u['reference_id'] for u in units],
                         'raw': raw, 'assessments': checked, 'attempts': attempts}
                dump(path, saved)
                return saved
            except Exception as exc:
                attempts.append({'error': str(exc), 'raw': raw})
                prompt = base + [{'role': 'assistant', 'content': str(raw or '')},
                    {'role': 'user', 'content': 'Repair validation error, returning every requested row: ' + str(exc)}]
                time.sleep(1)
        saved = {'status': 'failed', 'method': method, 'flow': flow, 'reference_ids': [u['reference_id'] for u in units], 'attempts': attempts}
        dump(path, saved)
        return saved
    completed = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(one, task) for task in (tasks[:args.max_batches] if args.max_batches else tasks)]
            for future in as_completed(futures):
                saved = future.result()
                completed.append(saved)
                dump(out / 'progress.json', {'completed_batches_this_run': len(completed), 'total_batches': len(tasks),
                    'failed_this_run': sum(s['status'] != 'complete' for s in completed)})
                print(f"{saved['method']}/{saved['flow']} {len(completed)}/{len(tasks)} {saved['status']}", flush=True)
    finally:
        tracker = getattr(llm, 'get_usage_summary', None)
        if tracker:
            dump(out / f'llm_usage_{time.time_ns()}.json', tracker())
        paths = [out / 'batches' / f'{m}_{f}_{i:04d}.json' for m, f, i, _ in tasks]
        rows, summary = export_results(out, ref, methods, candidates, [p for p in paths if p.exists()])
        # Refresh proposals after resume, retaining already reviewed decisions.
        review_template(out, ref, methods, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def adjudicate(args):
    proposal = args.proposal_dir
    for name, expected in read(proposal / 'input_freeze.json').items():
        if sha(proposal / name) != expected:
            raise ValueError('Frozen proposal snapshot changed: ' + name)
    ref = read(proposal / 'reference_snapshot.json')
    manifest = read(proposal / 'input_manifest.json')
    candidates = {(x['method'], x['flow']): x['rules'] for x in read(proposal / 'candidate_snapshot.json')}
    refmap = {u['reference_id']: u for u in ref['units'] if u['eligibility'] == 'eligible'}
    rows, seen = [], set()
    with args.decisions.open(encoding='utf-8-sig', newline='') as stream:
        for entry in csv.DictReader(stream):
            key = entry['method'], entry['reference_id']
            if key in seen or entry['method'] not in manifest['methods'] or entry['reference_id'] not in refmap:
                raise ValueError('Duplicate or unexpected review row: ' + str(key))
            seen.add(key)
            if not entry['reviewed_status'].strip():
                continue
            unit = refmap[entry['reference_id']]
            row = {'reference_id': entry['reference_id'], 'status': entry['reviewed_status'].strip(), 'reason': entry['reviewed_reason']}
            for field in ('support_rule_ids', 'conflict_rule_ids', 'condition_rule_ids'):
                row[field] = json.loads(entry[field] or '[]')
            checked = validate_assessments(json.dumps({'assessments': [row]}), [unit], candidates[entry['method'], unit['flow']])[0]
            rows.append({**checked, 'method': entry['method'], 'flow': unit['flow'], 'review_status': 'reviewed'})
    result = summarize(ref, manifest['methods'], rows)
    if not result['complete']:
        raise ValueError(f"Review incomplete: {result['assessed']}/{result['expected_assessments']} decisions; no final score written")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result.update(label_status='reviewed', reviewer=args.reviewer, independent_human_review='not_asserted',
                  decisions_sha256=sha(args.decisions), proposal_protocol_sha256=sha(proposal / 'protocol.json'))
    dump(args.output_dir / 'reviewed_assessments.json', rows)
    dump(args.output_dir / 'reviewed_summary.json', result)
    dump(args.output_dir / 'reference_snapshot.json', ref)
    with (args.output_dir / 'table.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'conditional_covered', 'conditional_n', 'conditional_coverage', 'all_covered', 'all_n', 'all_coverage', 'conflicting'])
        for method, own in result['methods'].items():
            c = own['conditional_action']
            writer.writerow([method, c['covered'], c['reference_units'], c['strict_coverage'], own['covered'], own['reference_units'], own['strict_coverage'], own['conflicting']])
    report = ['# ABCD guideline 条件知识覆盖（复核后）', '',
        '固定官方文字参考单元；不是 graph edge recall、系统动作定义覆盖或端到端成功率。', '',
        '| 方法 | 条件单元覆盖 | 条件严格覆盖率 | 全部单元覆盖 | 全部严格覆盖率 | 冲突项 |', '|---|---:|---:|---:|---:|---:|']
    tex = [r'\begin{tabular}{lrrrrr}', r'\toprule', r'Method & Cond. count & Cond. (\%) & All count & All (\%) & Conflicts \\', r'\midrule']
    for method, own in result['methods'].items():
        c = own['conditional_action']
        pct = lambda value: f'{value*100:.2f}' if value is not None else 'N/A'
        conditional_count = f"{c['covered']}/{c['reference_units']}"
        all_count = f"{own['covered']}/{own['reference_units']}"
        report.append(f"| {method} | {conditional_count} | {pct(c['strict_coverage'])}% | {all_count} | {pct(own['strict_coverage'])}% | {own['conflicting']} |")
        safe_method = method.replace('_', r'\_')
        tex.append(f"{safe_method} & {conditional_count} & {pct(c['strict_coverage'])} & {all_count} & {pct(own['strict_coverage'])} & {own['conflicting']}" + r' \\')
    report += ['', f"复核人标签：{args.reviewer}。该标签不自动代表独立人工 gold。", '',
        'partial、absent、contradicted、unresolved_conflict、unverifiable 不按完整覆盖计。',
        '全部状态、fine-task/flow macro 和重复原文组诊断见 reviewed_summary.json；逐项源证据见 reviewed_assessments.json。',
        '含糊/冲突的官方参考排除项见 reference_snapshot.json。抽取遗漏仍可能降低本指标，不将其声称为完整产物召回率。', '']
    (args.output_dir / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    tex += [r'\bottomrule', r'\end{tabular}']
    (args.output_dir / 'table.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('reference')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--subflows', '--subflow', dest='subflows', help='ABCD coarse flows, e.g. shipping_issue')
    g.add_argument('--all', action='store_true')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/eval/abcd/data')
    p.add_argument('--output-dir', type=Path, required=True)
    p.set_defaults(func=reference)
    p = sub.add_parser('evaluate')
    for name in ['reference', 'source-dir', 'rules-dir', 'output-dir']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--provider', choices=['existing', 'shubiaobiao'], default='existing')
    p.add_argument('--env-file', type=Path)
    p.add_argument('--workflow-id')
    p.add_argument('--model', default='qwen3.6-flash')
    p.add_argument('--origins', default='procedural_rule')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--qps', type=float, default=1.8)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--max-prompt-chars', type=int, default=400000)
    p.add_argument('--max-batches', type=int)
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=evaluate)
    p = sub.add_parser('adjudicate')
    p.add_argument('--proposal-dir', type=Path, required=True)
    p.add_argument('--decisions', type=Path, required=True)
    p.add_argument('--reviewer', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.set_defaults(func=adjudicate)
    args = parser.parse_args()
    if args.command == 'evaluate' and (min(args.workers, args.batch_size, args.retries, args.max_prompt_chars) < 1 or args.qps <= 0 or args.max_batches is not None and args.max_batches < 1):
        parser.error('Positive workers/batch-size/retries/prompt budget/QPS/max-batches required')
    args.func(args)


if __name__ == '__main__':
    main()
