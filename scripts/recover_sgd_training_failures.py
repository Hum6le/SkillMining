"""Recover failed frozen TRAIN cases now, then refine on the complete cached rollout."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
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
from eval_tod.sgd_adapter import iter_family_dialogues, gold_policy, score_policies
from eval_tod.sgd_aligned import SGDAlignedSkillAgent, predict_policy, training_contract
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase
from scripts.launch_sgd_homes_pilot import start
from scripts.launch_multiwoz_hotel_pilot import background_python


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    if not path.exists():
        return []
    output = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return output


def identity(row):
    return row.get('sample_id') or row['sample']['sample_id']


def stop_source_after_durable_rollout(source, expected, progress):
    """Stop only the explicitly identified experiment after all TRAIN rows persist."""
    import psutil
    launch = read(source/'launch.json')
    while True:
        saved = rows(source/'ours/ours_training_turns.jsonl')
        ids = [identity(r) for r in saved]
        if len(ids) == len(expected) and ids == expected:
            break
        if not psutil.pid_exists(launch['pid']):
            raise RuntimeError('Original process exited before a complete durable TRAIN rollout')
        atomic_json(progress, {'phase': 'wait_for_original_rollout', 'saved': len(saved), 'total': len(expected)})
        time.sleep(5)
    try:
        process = psutil.Process(launch['pid'])
        command = process.cmdline()
        if not any('run_sgd_domain_eval.py' in a for a in command) or str(source/'ours') not in command:
            raise RuntimeError('Source PID no longer belongs to the exact experiment')
        children = process.children(recursive=True)
        # All 507 source rows are durably saved. Dependent analysis against the
        # incomplete rollout is superseded by the complete cached rollout below.
        for child in children:
            child.terminate()
        process.terminate()
        _, alive = psutil.wait_procs(children+[process], timeout=5)
        for child in alive:
            child.kill()
    except psutil.NoSuchProcess:
        pass
    return saved


def run(args):
    source, out = args.source_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    progress = out/'recovery_progress.json'
    splits = ROOT/'data/eval/sgd/splits_10pct_seed42'
    train = list(iter_family_dialogues(splits, 'Homes', 'train'))
    cases = {f"{d['dialogue_id']}:{i}": (d, i) for d in train for i in d['target_indices']['policy_turn_indices']}
    assert len(cases) == 507
    with ZipFile(ROOT/'data/eval/sgd/data.zip') as archive:
        ontology = json.load(archive.open('data/ontology.json'))
    ontology['_sgd_training_contract'] = training_contract(train)
    agent = SGDAlignedSkillAgent(family='Homes', model=args.model, ontology=ontology,
                                predictor=predict_policy, mining_method='trace_cover')
    agent.load(source/'ours/offline')
    frozen_hash = agent.resource_hash()
    configure_usage(out/'recovery_usage')
    recovered = {}
    seen_failures = set()
    original_snapshot = rows(source/'ours/ours_training_turns.jsonl')
    initial_failures = [identity(r) for r in original_snapshot if r.get('status') == 'error']
    atomic_json(out/'recovery_manifest.json', {'source_root': str(source), 'checkpoint': str(source/'ours/offline'),
        'frozen_sha256': frozen_hash, 'initial_failed_cases': initial_failures,
        'selection': 'Only execution failures; valid predictions reused unchanged',
        'train_turns': 507, 'validation_turns': 115, 'retry_workers': args.retry_workers,
        'refine_workers': args.workers, 'whole_training_rollout_repeated_with_llm': False})
    def recover(ids, pass_number):
        if not ids:
            return
        set_phase('train_recovery')
        outcomes = ordered_cases(lambda item: agent.predict(*item), [cases[k] for k in ids],
            workers=min(args.retry_workers, len(ids)), directory=out/'retry_cases', phase=f'train_recovery_{pass_number}')
        with (out/'retry_predictions.jsonl').open('a', encoding='utf-8') as stream:
            for number, (key, outcome) in enumerate(zip(ids, outcomes), 1):
                try:
                    pred, diagnostic = require_value(outcome)
                    score_policies([gold_policy(*cases[key])], [pred])
                    if diagnostic['resource_sha256'] != frozen_hash:
                        raise ValueError('Retry used changed frozen resources')
                    recovered[key] = (pred, diagnostic)
                    row = {'sample_id': key, 'pass': pass_number, 'prediction': pred, 'diagnostic': diagnostic}
                except Exception as exc:
                    row = {'sample_id': key, 'pass': pass_number, 'status': 'error', 'error': str(exc)}
                stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
                atomic_json(progress, {'phase': 'recover_train_failures', 'pass': pass_number,
                    'attempted_this_pass': number, 'pass_total': len(ids), 'recovered': len(recovered),
                    'known_failed_cases': len(seen_failures), 'retry_workers': args.retry_workers})
        atomic_json(out/'recovery_usage.json', persist_usage())
    seen_failures.update(initial_failures)
    recover(initial_failures, 1)
    saved = stop_source_after_durable_rollout(source, list(cases), progress)
    source_sha = hashlib.sha256((source/'ours/ours_training_turns.jsonl').read_bytes()).hexdigest()
    cache = {}
    for row in saved:
        key = identity(row)
        if row.get('status') == 'error':
            seen_failures.add(key)
            continue
        evidence = row['sgd_evidence']
        if evidence['gold_policy'] != gold_policy(*cases[key]):
            raise ValueError('Saved TRAIN gold differs from fixed split')
        if evidence['runtime_diagnostic']['resource_sha256'] != frozen_hash:
            raise ValueError('Saved TRAIN resources differ from recovery checkpoint')
        cache[key] = (evidence['predicted_policy'], evidence['runtime_diagnostic'])
    for pass_number in range(2, 5):
        pending = [key for key in cases if key in seen_failures and key not in recovered]
        if not pending:
            break
        recover(pending, pass_number)
    unresolved = sorted(seen_failures-recovered.keys())
    if unresolved:
        raise RuntimeError(f'Recovery has unresolved TRAIN cases: {unresolved}; complete refinement was not started')
    cache.update(recovered)
    assert set(cache) == set(cases) and agent.resource_hash() == frozen_hash
    atomic_json(out/'train_recovery_audit.json', {'source_rollout_sha256': source_sha,
        'successful_cases_reused': 507-len(seen_failures), 'failed_cases_recovered': len(recovered),
        'total_complete': len(cache), 'unresolved': [], 'frozen_resource_sha256': frozen_hash})
    # Native refine still builds the complete fixed-order record set, clusters,
    # diagnoses and reflects. The cached prediction function makes zero new
    # rollout model calls for either previously valid or already recovered rows.
    native_predict = agent.predict
    def cached_predict(dialogue, index):
        return deepcopy(cache[f"{dialogue['dialogue_id']}:{index}"])
    agent.predict = cached_predict
    agent.parallel_workers = args.workers
    agent.runtime_diagnostics_dir = out/'ours'
    agent.save(out/'ours')
    atomic_json(progress, {'phase': 'complete_standard_refinement', 'train_complete': 507,
                           'reused': 507-len(recovered), 'recovered': len(recovered)})
    configure_usage(out/'ours')
    summary = agent.refine(train, out/'ours', batch_size=8, analysis_batch_size=8)
    agent.predict = native_predict
    agent.save(out/'ours')
    atomic_json(out/'training_usage.json', persist_usage())
    if summary['failed_rollout_turns']:
        raise RuntimeError('Cached complete rollout unexpectedly lost TRAIN cases')
    command = [str(background_python()), '-u', str(ROOT/'scripts/run_sgd_domain_eval.py'),
        '--domain', 'Homes', '--method', 'ours', '--execution-protocol', 'aligned',
        '--split', 'validation', '--eval-targets', 'policy', '--splits-dir', str(splits),
        '--archive', str(ROOT/'data/eval/sgd/data.zip'), '--model', args.model,
        '--workers', str(args.workers), '--ours-resource-dir', str(out/'ours'),
        '--output-dir', str(out/'evaluation')]
    child = start(command, out/'evaluation.log')
    atomic_json(out/'evaluation_launch.json', {'pid': child.pid, 'command': command})
    while child.poll() is None:
        atomic_json(progress, {'phase': 'validation', 'train_complete': 507,
            'completed': len(rows(out/'evaluation/predictions.jsonl')), 'total': 115})
        time.sleep(10)
    if child.returncode:
        raise RuntimeError(f'Evaluation exited with {child.returncode}')
    for filename in ['predictions.jsonl', 'summary.json']:
        shutil.copy2(out/'evaluation'/filename, out/'ours'/filename)
    from scripts.monitor_sgd_slot_feedback import finalize
    finalize(out)
    atomic_json(progress, {'phase': 'completed', 'train_complete': 507, 'validation_complete': 115,
                           'execution_status': read(out/'ours/summary.json')['execution_status']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='qwen3.6-flash')
    parser.add_argument('--retry-workers', type=int, default=4)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir/'run_failed.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise
