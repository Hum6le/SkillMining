"""Dialogue-level Trace2Skill evidence, matching the local ABCD learning loop.

Task labels remain SGD named payloads. The analysis unit and mixed evidence
policy match ABCD: one failed record per imperfect dialogue, and one success
record per dialogue with at least one correct turn. A dialogue may supply both.
"""
from collections import OrderedDict
import json
from pathlib import Path

from eval_tod.sgd_llm import sgd_chat_with_retry
from eval_tod.sgd_skill_agent import _json_object

PROTOCOL = 'sgd_trace2skill_abcd_dialogue_analysis_v2'
ANALYSIS_BATCH_SIZE = 8
PROMPT_CHAR_BUDGET = 400_000


def build_analysis_jobs(rows, batch_number, skill, raw_dir, batch_size=ANALYSIS_BATCH_SIZE):
    """Group TRAIN rows by dialogue before batching; never pool their lessons."""
    if batch_size < 1:
        raise ValueError('Analysis batch size must be positive')
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(row['dialogue_id'], []).append(row)
    cases = []
    for dialogue_id, turns in grouped.items():
        turns = sorted(turns, key=lambda row: row['turn_index'])
        cases.append({'instance_id': dialogue_id, 'turns': turns})
    jobs = []
    for kind in ['error', 'success']:
        selected = [case for case in cases if any(
            not row['joint_correct'] if kind == 'error' else row['joint_correct']
            for row in case['turns'])]
        pending = []
        for case in selected:
            candidate = pending + [case]
            if pending and (len(candidate) > batch_size or
                    len(build_analysis_prompt(kind, candidate, skill, '')) > PROMPT_CHAR_BUDGET):
                jobs.append((kind, f'{batch_number:04d}-{len(jobs):04d}', pending, skill, raw_dir))
                pending = []
            pending.append(case)
        if pending:
            jobs.append((kind, f'{batch_number:04d}-{len(jobs):04d}', pending, skill, raw_dir))
    return jobs


def evidence_for(case, *, failures_only=False):
    return [{key: row[key] for key in (
        'sample_id', 'turn_index', 'gold', 'prediction', 'joint_correct',
        'api_observation', 'observation_visible') if key in row}
        for row in case['turns'] if not failures_only or not row['joint_correct']]


def build_analysis_prompt(kind, cases, skill, task_contract):
    common = (
        'Analyze SGD TRAIN dialogues independently. Return exactly one analysis per '
        'supplied instance_id; never merge different dialogues into one record. '
        'Gold labels and correctness are TRAIN supervision only. Separate action-head '
        'choice from named slot/value binding. Generalize value sources, conditions, '
        'missing-value behavior and serialization; never memorize instance values. '
        'The field joint_correct is strict policy correctness, not dialogue success.\n'
    )
    if kind == 'error':
        instructions = (
            'Diagnose each dialogue with at least one failed turn. Use failed turns '
            'as error evidence and correct turns only as context. Return JSON '
            '{"analyses":[{"instance_id":"exact supplied ID",'
            '"failure_cause":{"title":"...","description":"...","content":"..."},'
            '"failure_memory":{"title":"...","description":"...","content":"..."}}]}. '
            'Both items must contain concrete reusable guidance grounded in that '
            'dialogue. Do not claim that proposed corrections have been replayed.\n'
        )
    else:
        instructions = (
            'Each dialogue has at least one correct turn, but may contain errors. '
            'Distill correct turns as positive evidence and incorrect turns only as '
            'contrast. Do not assume the whole dialogue succeeded. Return JSON '
            '{"analyses":[{"instance_id":"exact supplied ID","items":['
            '{"title":"...","description":"...","content":"..."}]}]}. '
            'Return at most three success items per dialogue; an empty list is allowed.\n'
        )
    visible_cases = []
    for case in cases:
        visible_cases.append({'instance_id': case['instance_id'], 'turns': [
            {key: row[key] for key in (
                'sample_id', 'turn_index', 'context', 'gold', 'prediction',
                'joint_correct', 'api_observation', 'observation_visible') if key in row}
            for row in case['turns']]})
    return (common + instructions + task_contract + '\nCurrent complete skill:\n' + skill
            + '\nAnnotated TRAIN dialogues:\n' + json.dumps(visible_cases, ensure_ascii=False))


def analyze_dialogue_job(job, model, *, task_contract):
    kind, index, cases, skill, raw_dir = job
    prompt = build_analysis_prompt(kind, cases, skill, task_contract)
    if len(prompt) > PROMPT_CHAR_BUDGET:
        raise ValueError('Trace2Skill dialogue analysis exceeds the prompt character budget')
    expected = {case['instance_id'] for case in cases}
    messages = [{'role': 'system', 'content': 'Return the requested JSON only. ' + task_contract},
                {'role': 'user', 'content': prompt}]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    responses = []
    for attempt in range(3):
        raw = sgd_chat_with_retry(messages, model=model, temperature=0.2,
            call_tag=f'trace2skill_sgd_{kind}_analysis' + ('_repair' if attempt else ''))
        responses.append(raw)
        (raw_dir / f'{kind}_{index}.json').write_text(json.dumps({
            'analysis_protocol': PROTOCOL, 'instance_ids': sorted(expected),
            'responses': responses}, ensure_ascii=False, indent=2), encoding='utf-8')
        try:
            payload = _json_object(raw)
            entries = payload.get('analyses')
            if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
                raise ValueError('Expected an analyses array of objects')
            ids = [entry.get('instance_id') for entry in entries]
            if len(ids) != len(expected) or any(not isinstance(i, str) for i in ids) or set(ids) != expected:
                raise ValueError('Return each supplied instance_id exactly once, with no other IDs')
            parsed = {}
            for entry in entries:
                if kind == 'error':
                    items = []
                    for field in ['failure_cause', 'failure_memory']:
                        value = entry.get(field)
                        if not isinstance(value, dict) or not isinstance(value.get('content'), str) or not value['content'].strip():
                            raise ValueError(f'{field} needs a nonempty content string')
                        items.append(dict(value, type=field))
                else:
                    values = entry.get('items')
                    if not isinstance(values, list) or len(values) > 3:
                        raise ValueError('Each success case needs an items array with at most three entries')
                    items = []
                    for value in values:
                        if not isinstance(value, dict) or not isinstance(value.get('content'), str) or not value['content'].strip():
                            raise ValueError('Success items need nonempty content strings')
                        items.append(dict(value, type='success_memory'))
                parsed[entry['instance_id']] = items
            return [{'instance_id': f'{kind}-{case["instance_id"]}',
                     'source_file': 'training_turns.jsonl', 'items': parsed[case['instance_id']],
                     'analysis_protocol': PROTOCOL,
                     'ast_evidence': evidence_for(case, failures_only=(kind == 'error'))}
                    for case in cases]
        except (ValueError, TypeError, KeyError) as exc:
            if attempt == 2:
                raise
            messages.extend([{'role': 'assistant', 'content': raw}, {'role': 'user', 'content':
                'Repair the JSON envelope using the same evidence. ' + str(exc)}])
    raise AssertionError('Unreachable')
