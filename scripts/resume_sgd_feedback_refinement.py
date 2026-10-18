"""Resume complete saved TRAIN supervision, reusing valid diagnosis batches."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.multiwoz_aligned import atomic_json
from eval_tod.sgd_adapter import iter_family_dialogues, gold_policy, score_policies, slot_signature, visible_context
from eval_tod.sgd_aligned import SGDAlignedSkillAgent, predict_policy, training_contract
from eval_tod.sgd_skill_agent import previous_node, policy_token
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase
from eval_tod.sgd_errors import error_summary
from skill_mining.online_refinement import build_post_rollout_batches, group_batch_reports, diagnose_rollout_batch
from scripts.recover_sgd_training_failures import read, rows, identity
from scripts.launch_sgd_homes_pilot import start
from scripts.launch_multiwoz_hotel_pilot import background_python


def make_record(dialogue, index, prediction, diagnostic):
    gold = gold_policy(dialogue, index)
    scores = score_policies([gold], [prediction])
    action_ok = bool(scores['action_set_exact'])
    slot_ok = slot_signature(gold) == slot_signature(prediction)
    source = previous_node(dialogue, index, 'Homes')
    target, predicted = policy_token(gold), policy_token(prediction)
    sid = f"{dialogue['dialogue_id']}:{index}"
    context = visible_context(dialogue, index)[-1800:]
    evidence = {'sample_id': sid, 'conversation_id': dialogue['dialogue_id'], 'turn_index': index,
        'source_action': source, 'gold_target': target, 'gold_policy': gold,
        'predicted_policy': prediction, 'action_correct': action_ok, 'slot_exact_correct': slot_ok,
        'slot_error_bucket': 'strict_correct' if slot_ok else 'named_payload_mismatch',
        'context': context, 'runtime_diagnostic': diagnostic}
    prefix = [policy_token(gold_policy(dialogue, i)) for i in dialogue['target_indices']['policy_turn_indices'] if i < index]
    return {'sample': {'sample_id': sid, 'conversation_id': dialogue['dialogue_id'],
        'source_action': source, 'target_action': target, 'prefix_action_sequence': prefix},
        'result': {'predicted_action': predicted, 'context': context}, 'sgd_evidence': evidence}


def complete_records(recovery):
    manifest = read(recovery/'recovery_manifest.json')
    source = Path(manifest['source_root'])
    train = list(iter_family_dialogues(ROOT/'data/eval/sgd/splits_10pct_seed42', 'Homes', 'train'))
    cases = {f"{d['dialogue_id']}:{i}": (d, i) for d in train for i in d['target_indices']['policy_turn_indices']}
    originals = rows(source/'ours/ours_training_turns.jsonl')
    assert [identity(r) for r in originals] == list(cases) and len(cases) == 507
    retry = {r['sample_id']: r for r in rows(recovery/'retry_predictions.jsonl') if r.get('status') != 'error'}
    result = []
    for row in originals:
        key = identity(row)
        if row.get('status') == 'error':
            replacement = retry[key]
            row = make_record(*cases[key], replacement['prediction'], replacement['diagnostic'])
        evidence = row['sgd_evidence']
        assert evidence['gold_policy'] == gold_policy(*cases[key])
        assert evidence['runtime_diagnostic']['resource_sha256'] == manifest['frozen_sha256']
        result.append(row)
    by_id = {identity(r): r for r in result}
    prior = [r for r in rows(recovery/'ours/ours_training_turns.jsonl') if r.get('status') != 'error']
    assert all(row == by_id[identity(row)] for row in prior)
    return train, result, source, manifest


def batch_key(sample_ids):
    return tuple(sample_ids)


def run(args):
    recovery, out = args.source_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    directory = out/'ours'
    directory.mkdir(exist_ok=True)
    progress = out/'recovery_progress.json'
    train, records, source, manifest = complete_records(recovery)
    with ZipFile(ROOT/'data/eval/sgd/data.zip') as archive:
        ontology = json.load(archive.open('data/ontology.json'))
    ontology['_sgd_training_contract'] = training_contract(train)
    agent = SGDAlignedSkillAgent(family='Homes', model=args.model, ontology=ontology,
        predictor=predict_policy, mining_method='trace_cover')
    agent.load(source/'ours/offline')
    assert agent.resource_hash() == manifest['frozen_sha256']
    initial_hash = agent.resource_hash()
    agent.runtime_diagnostics_dir = directory
    agent.parallel_workers = args.workers
    (directory/'ours_training_turns.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    atomic_json(directory/'online_schedule.json', {'mode': 'standard', 'schedule_unit': 'policy_turn',
        'num_selected_samples': 507, 'batch_size': 8, 'sample_ids': [identity(r) for r in records]})
    agent.stage = 'refining'
    agent._accumulate(records)
    agent.save(directory)
    batches = list(enumerate(build_post_rollout_batches(records, max_batch_size=8), 1))
    cached = {}
    for path in sorted((recovery/'ours').glob('diagnosis_[0-9][0-9][0-9][0-9].json')):
        report = read(path)
        if report.get('status') == 'ok':
            cached[batch_key(report['sample_ids'])] = (report, path)
    reports, pending = {}, []
    for index, batch in batches:
        key = batch_key([identity(r) for r in batch])
        if key in cached:
            report, path = cached[key]
            report = deepcopy(report)
            report.update(batch_id=str(index), cached_source_report=str(path),
                cache_scope='Same frozen resources and exact ordered TRAIN batch; graph support counts may differ after restoring one previously lost record.')
            reports[index] = report
            atomic_json(directory/f'diagnosis_{index:04d}.json', report)
        else:
            pending.append((index, batch))
    atomic_json(out/'resume_audit.json', {'complete_train_turns': 507, 'new_rollout_llm_calls': 0,
        'source_recovery': str(recovery), 'frozen_sha256': initial_hash,
        'valid_diagnoses_reused': len(reports), 'diagnoses_to_run': len(pending),
        'all_prior_valid_rollout_records_unchanged': True})
    configure_usage(directory)
    for attempt in range(1, 4):
        if not pending:
            break
        atomic_json(progress, {'phase': 'recover_diagnosis', 'train_complete': 507,
            'recovered': 41, 'known_failed_cases': 41, 'diagnoses_complete': len(reports),
            'diagnoses_total': len(batches), 'pending': len(pending), 'attempt': attempt})
        set_phase('online_diagnosis')
        def diagnose(item):
            index, batch = item
            return diagnose_rollout_batch(str(index), batch, agent.state, agent.diagnosis_skill_text(),
                args.model, reference=agent.base_reference, action_rules=agent.base_action_rules,
                slot_policies=agent.base_slot_policies, record_projector=agent._project_record,
                prompt_template=agent.diagnosis_prompt, chat_fn=agent._chat, **agent.diagnosis_options(batch))
        outcomes = ordered_cases(diagnose, pending, workers=min(args.diagnosis_workers, len(pending)),
                                 directory=directory, phase=f'online_diagnosis_recovery_{attempt}')
        again = []
        for item, outcome in zip(pending, outcomes):
            index, batch = item
            try:
                report = require_value(outcome)
                if report.get('status') != 'ok':
                    raise RuntimeError(report.get('error', 'invalid report'))
                reports[index] = report
                atomic_json(directory/f'diagnosis_{index:04d}.json', report)
            except Exception as exc:
                again.append(item)
                atomic_json(out/f'diagnosis_{index:04d}_attempt_{attempt}_error.json', {'error': str(exc)})
            atomic_json(progress, {'phase': 'recover_diagnosis', 'train_complete': 507,
                'recovered': 41, 'known_failed_cases': 41, 'diagnoses_complete': len(reports),
                'diagnoses_total': len(batches), 'attempt': attempt})
        pending = again
    if pending:
        raise RuntimeError(f'Unresolved diagnosis batches: {[i for i, _ in pending]}')
    groups = group_batch_reports([reports[i] for i, _ in batches], max_reports=16)
    atomic_json(progress, {'phase': 'online_reflection', 'train_complete': 507,
        'recovered': 41, 'diagnoses_complete': len(reports), 'reflection_groups': len(groups)})
    agent.reflect_groups(groups, directory)
    errors = error_summary(directory)
    agent.state['refinement_errors'] = errors
    agent.stage = 'refined_with_errors' if errors['total'] else 'refined'
    agent.save(directory)
    training = {'protocol': agent.protocol, 'schedule_unit': 'policy_turn', 'rollout_turns': 507,
        'scheduled_turns': 507, 'failed_rollout_turns': 0, 'reports': len(reports),
        'reflection_groups': len(groups), 'case_errors': errors, 'snapshot_sha256': initial_hash,
        'final_sha256': agent.resource_hash(), 'new_rollout_llm_calls': 0,
        'valid_diagnoses_reused': read(out/'resume_audit.json')['valid_diagnoses_reused'],
        'selected_dialogue_ids': [d['dialogue_id'] for d in train]}
    atomic_json(directory/'ours_training.json', training)
    atomic_json(out/'training_usage.json', persist_usage())
    command = [str(background_python()), '-u', str(ROOT/'scripts/run_sgd_domain_eval.py'),
        '--domain', 'Homes', '--method', 'ours', '--execution-protocol', 'aligned', '--split', 'validation',
        '--eval-targets', 'policy', '--splits-dir', str(ROOT/'data/eval/sgd/splits_10pct_seed42'),
        '--archive', str(ROOT/'data/eval/sgd/data.zip'), '--model', args.model, '--workers', str(args.workers),
        '--ours-resource-dir', str(directory), '--output-dir', str(out/'evaluation')]
    child = start(command, out/'evaluation.log')
    atomic_json(out/'evaluation_launch.json', {'pid': child.pid, 'command': command})
    while child.poll() is None:
        atomic_json(progress, {'phase': 'validation', 'train_complete': 507, 'recovered': 41,
            'completed': len(rows(out/'evaluation/predictions.jsonl')), 'total': 115})
        time.sleep(10)
    if child.returncode:
        raise RuntimeError(f'Evaluation exited with {child.returncode}')
    evaluated = out/'evaluation'
    if read(evaluated/'summary.json')['case_errors']['total']:
        # The existing failure-only recovery preserves every valid prediction.
        staging = out/'evaluation_recovery_source'
        old = ROOT/'outputs/sgd_aligned_failure_recovery_20261018_204325'
        for method in ['awm', 'trace2skill', 'ours']:
            target = staging/method
            target.mkdir(parents=True, exist_ok=True)
            origin = evaluated if method == 'ours' else old/method
            for name in ['predictions.jsonl', 'case_errors.jsonl']:
                if (origin/name).exists(): shutil.copy2(origin/name, target/name)
        atomic_json(staging/'recovery_manifest.json', {'source_run': str(old)})
        for retry in range(1, 3):
            target = out/f'validation_recovery_{retry}'
            cmd = [str(background_python()), '-u', str(ROOT/'scripts/recover_sgd_aligned_failures.py'),
                '--source-root', str(staging), '--output-dir', str(target), '--ours-resource-dir', str(directory),
                '--workers', '4', '--model', args.model]
            process = start(cmd, out/f'validation_recovery_{retry}.log')
            while process.poll() is None:
                atomic_json(progress, {'phase': 'validation_failure_recovery', 'train_complete': 507,
                    'recovered': 41, 'pass': retry})
                time.sleep(10)
            if process.returncode: raise RuntimeError('Validation recovery failed')
            evaluated = target/'ours'
            if read(evaluated/'summary.json')['case_errors']['total'] == 0: break
            staging = target
    for name in ['predictions.jsonl', 'summary.json']:
        shutil.copy2(evaluated/name, directory/name)
    from scripts.monitor_sgd_slot_feedback import finalize
    finalize(out)
    atomic_json(progress, {'phase': 'completed', 'train_complete': 507, 'recovered': 41,
        'validation_complete': 115, 'execution_status': read(directory/'summary.json')['execution_status']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='qwen3.6-flash')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--diagnosis-workers', type=int, default=4)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir/'run_failed.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise
