#!/usr/bin/env python3
"""SGD adapter reusing the existing Trace2Skill analysis and MAP/REDUCE/APPLY."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_multiwoz_trace2skill_pilot import _predict, _evolve, _write_json
from eval_tod.sgd_trace2skill_analysis import (PROTOCOL as ANALYSIS_PROTOCOL,
    build_analysis_jobs, analyze_dialogue_job)
from scripts.run_sgd_domain_eval import predict_policy
from eval_tod.sgd_adapter import (gold_policy, score_policies, visible_context, iter_family_dialogues,
    evaluation_turn_indices, score_api_calls)
from eval_tod.sgd_agent import SGDAWMAgent
from eval_tod.sgd_errors import error_summary, record_error
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase, native_snapshot, record_attempt


SEED_SKILL = """---
name: sgd_trace2skill
description: Training-derived SGD backend API and named dialogue-act policy
---

# SGD system-turn policy

Predict the next system turn as an optional CALL(service, method, parameters)
and an unordered set of dialogue acts. Only prior utterances and the available
service schema are visible. The current system utterance, current gold acts,
belief-state labels and future turns are hidden.

## API decision and payload

Choose whether an API call is needed. Bind its actual service ID, method and
named parameters from the visible context and schema. API observations are
provided by the evaluator only after an exactly correct predicted call.

## Dialogue acts and named slots

Choose context-appropriate action heads, then their applicable named slots and
values. Multiple acts may share a head. Use only the visible context or the API
observation actually returned. Never copy training instance values as defaults.
Use an empty domain for service-free acts. Never impose an order on same-turn acts.
"""


def run(args):
    output = args.output_dir.resolve()
    # The full coordinator creates these two bookkeeping files before invoking
    # a domain runner. Actual experiment artifacts must still be protected.
    if output.exists() and any(p.name not in {'invocation.json', 'run.log'} for p in output.iterdir()):
        raise FileExistsError(output)
    train = list(iter_family_dialogues(args.splits_dir, args.domain, 'train'))
    validation = list(iter_family_dialogues(args.splits_dir, args.domain, args.split))
    if getattr(args, 'max_train_dialogues', None):
        train = train[:args.max_train_dialogues]
    if getattr(args, 'max_dialogues', None):
        validation = validation[:args.max_dialogues]
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol': 'sgd_train_seen_domain_policy_turns_v1', 'method': 'trace2skill',
        'adapter': 'sgd_trace2skill_shared_predictor_map_reduce_apply_v2',
        'analysis_protocol': ANALYSIS_PROTOCOL,
        'evolution_settings': {'map_batch_size': 8, 'merge_batch_size': 5,
            'max_merge_levels': 5, 'max_skill_lines': 500, 'patch_pipeline': 'json',
            'success_analysis_unit': 'dialogue_with_any_correct_turn',
            'failure_analysis_unit': 'dialogue_with_any_failed_turn',
            'analysis_batch_dialogues': 8, 'self_verifier': False},
        'model': args.model, 'domain': args.domain, 'split': args.split,
        'workers': args.workers, 'batch_dialogues': args.batch_dialogues,
        'train_dialogues': len(train), 'validation_dialogues': len(validation),
        'train_turns': sum(len(d['target_indices']['policy_turn_indices']) for d in train),
        'validation_turns': sum(len(evaluation_turn_indices(d, getattr(args, 'eval_targets', 'policy'))) for d in validation),
        'evaluation_targets': getattr(args, 'eval_targets', 'policy'),
        'train_dialogue_ids': [d['dialogue_id'] for d in train],
        'validation_dialogue_ids': [d['dialogue_id'] for d in validation],
        'data_manifest': {s: {'sha256': hashlib.sha256((args.splits_dir / args.domain / (s + '.jsonl.gz')).read_bytes()).hexdigest()}
                          for s in ['train', args.split]}}
    _write_json(output / 'data_manifest.json', manifest)
    if getattr(args, 'dry_run', False):
        return manifest
    from llm import resolve_config
    resolve_config(model=args.model)
    configure_usage(output)
    with ZipFile(args.archive) as zipped:
        ontology = json.load(zipped.open('data/ontology.json'))
    active_predictor = predict_policy
    aligned = getattr(args, 'execution_protocol', 'legacy') == 'aligned'
    if aligned:
        from eval_tod.sgd_aligned import predict_policy as active_predictor, training_contract, TASK, PROTOCOL
        ontology['_sgd_training_contract'] = training_contract(train)
        manifest['execution_protocol'] = PROTOCOL
        _write_json(output / 'data_manifest.json', manifest)
    learning_contract = (
        'The following task contract describes INFERENCE visibility. During this '
        'TRAIN-only analysis/evolution, the attached gold policy labels and '
        'prediction comparisons are permitted supervision. Never use evaluation '
        'labels or turn-specific TRAIN values as reusable defaults.\n'
        + (TASK if aligned else SGDAWMAgent.induction_task_contract))
    frozen = getattr(args, 'skill_resource_dir', None)
    skill_dir = frozen.resolve() if frozen else output / 'evolved_skill'
    if frozen:
        if not (skill_dir/'SKILL.md').is_file():
            raise FileNotFoundError(skill_dir/'SKILL.md')
        manifest['frozen_skill_resources'] = str(skill_dir)
        _write_json(output/'data_manifest.json', manifest)
    else:
        skill_dir.mkdir()
        (skill_dir / 'SKILL.md').write_text(
            SEED_SKILL + ('\n## Shared executor contract\n\n' + TASK if aligned else ''), encoding='utf-8')
    batches = [] if frozen else [train[i:i + args.batch_dialogues] for i in range(0, len(train), args.batch_dialogues)]
    with (output / 'training_turns.jsonl').open('w', encoding='utf-8') as stream:
        for batch_number, batch in enumerate(batches, 1):
            set_phase('train_rollout')
            cases = [(d, i) for d in batch for i in d['target_indices']['policy_turn_indices']]
            print(f'Trace2Skill SGD batch {batch_number}/{len(batches)}: {len(cases)} turns', flush=True)
            predictions = ordered_cases(lambda case: _predict(case, args.model, ontology, skill_dir, active_predictor),
                cases, workers=args.workers, directory=output, phase=f'train_rollout_batch_{batch_number:04d}')
            rows = []
            for outcome, (dialogue, index) in zip(predictions, cases):
                gold = gold_policy(dialogue, index)
                try:
                    prediction, diagnostic = require_value(outcome)
                    correct = bool(score_policies([gold], [prediction])['turn_joint_ast'])
                except Exception as exc:
                    record_error(output, 'train_rollout', exc, dialogue_id=dialogue['dialogue_id'], turn_index=index)
                    prediction, diagnostic, correct = {'call': None, 'acts': [], '_execution_error': True}, {}, False
                row = {'sample_id': f"{dialogue['dialogue_id']}:{index}", 'batch': batch_number,
                    'dialogue_id': dialogue['dialogue_id'], 'turn_index': index,
                    'context': visible_context(dialogue, index)[-12000:], 'gold': gold,
                    'prediction': prediction, 'joint_correct': correct,
                    'api_observation': diagnostic.get('api_observation'),
                    'observation_visible': diagnostic.get('observation_visible', False)}
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
            set_phase('trace2skill_analysis')
            skill = (skill_dir / 'SKILL.md').read_text(encoding='utf-8')
            jobs = build_analysis_jobs(rows, batch_number, skill, output / 'analysis_raw')
            records = []
            analyses = ordered_cases(lambda job: analyze_dialogue_job(job, args.model,
                task_contract=learning_contract), jobs, workers=args.workers,
                directory=output, phase=f'analysis_batch_{batch_number:04d}')
            for outcome, job in zip(analyses, jobs):
                try:
                    records.extend(require_value(outcome))
                except Exception as exc:
                    record_error(output, 'trace2skill_analysis', exc, batch=batch_number, job=job[1])
            batch_dir = output / 'batches' / f'batch_{batch_number:04d}'
            batch_dir.mkdir(parents=True)
            _write_json(batch_dir / 'analysis_records.json', records)
            set_phase('trace2skill_evolution')
            before = native_snapshot()
            evolution_ok = False
            try:
                evolved = _evolve(records, skill_dir, batch_dir, args.model, args.workers,
                    dataset_name='sgd', task_contract=learning_contract)
                evolution_ok = True
            except Exception as exc:
                record_error(output, 'trace2skill_evolution', exc, batch=batch_number)
                evolved = {'status': 'error', 'error': repr(exc)}
            finally:
                # The official evolver calls llm.chat from several threads.
                # Measure that whole isolated parent-process phase once, so
                # overlapping before/after snapshots cannot double count calls.
                if native_snapshot().get('calls', 0) > before.get('calls', 0):
                    record_attempt(before, '', '', tag='trace2skill_sgd_evolution', success=evolution_ok)
            _write_json(batch_dir / 'summary.json', {'turns': len(rows),
                'joint_correct': sum(r['joint_correct'] for r in rows),
                'analysis_records': len(records), 'evolution': evolved})
            print(f"Trace2Skill SGD batch {batch_number}: evolution={evolved['status']}", flush=True)
    set_phase('testing')
    cases = [(d, i) for d in validation for i in evaluation_turn_indices(d, getattr(args, 'eval_targets', 'policy'))]
    skill_hash = hashlib.sha256((skill_dir / 'SKILL.md').read_bytes()).hexdigest()
    predictions = ordered_cases(lambda case: _predict(case, args.model, ontology, skill_dir, active_predictor),
        cases, workers=args.workers, directory=output, phase='testing')
    gold_rows, pred_rows = [], []
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for outcome, (dialogue, index) in zip(predictions, cases):
            gold = gold_policy(dialogue, index)
            try:
                prediction, diagnostic = require_value(outcome)
                score_policies([gold], [prediction])
            except Exception as exc:
                record_error(output, 'testing', exc, dialogue_id=dialogue['dialogue_id'], turn_index=index)
                prediction, diagnostic = {'call': None, 'acts': [], '_execution_error': True}, {'status': 'error'}
            gold_rows.append(gold)
            pred_rows.append(prediction)
            stream.write(json.dumps({'dialogue_id': dialogue['dialogue_id'], 'turn_index': index,
                'gold': gold, 'prediction': prediction, 'diagnostic': diagnostic}, ensure_ascii=False) + '\n')
            stream.flush()
    assert hashlib.sha256((skill_dir / 'SKILL.md').read_bytes()).hexdigest() == skill_hash
    errors = error_summary(output)
    summary = {**manifest, **score_policies(gold_rows, pred_rows), 'case_errors': errors,
        'llm_usage': persist_usage(), 'execution_status': errors['status'], 'skill_sha256': skill_hash}
    from eval_tod.sgd_adapter import score_slot_items
    summary['slot_item_metrics'] = score_slot_items(gold_rows, pred_rows)
    if getattr(args, 'eval_targets', 'policy') == 'call':
        summary['backend_call_metrics'] = score_api_calls(gold_rows, pred_rows)
    _write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', default='Homes')
    parser.add_argument('--splits-dir', type=Path, default=ROOT / 'data/eval/sgd/splits_10pct_seed42')
    parser.add_argument('--archive', type=Path, default=ROOT / 'data/eval/sgd/data.zip')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='qwen3.6-flash')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--eval-targets', choices=['policy', 'call'], default='policy')
    parser.add_argument('--execution-protocol', choices=['legacy', 'aligned'], default='legacy')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--workflow-ids', default='', help='Request pool IDs in threaded workflow mode')
    parser.add_argument('--batch-dialogues', type=int, default=25)
    parser.add_argument('--skill-resource-dir', type=Path, help='Frozen evolved skill; skip TRAIN evolution')
    parser.add_argument('--max-train-dialogues', type=int)
    parser.add_argument('--max-dialogues', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.batch_dialogues < 1:
        parser.error('workers and batch-dialogues must be positive')
    if any(value is not None and value < 1 for value in [args.max_train_dialogues, args.max_dialogues]):
        parser.error('Dialogue limits must be positive')
    result = run(args)
    print(json.dumps({key: result.get(key) for key in ['method', 'domain', 'train_turns',
        'validation_turns', 'action_set_exact', 'slot_exact_given_action',
        'turn_joint_ast', 'execution_status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
