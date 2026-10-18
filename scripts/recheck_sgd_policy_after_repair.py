#!/usr/bin/env python3
"""Recover the rolled-back SGD group and reevaluate all policy turns."""
import argparse
from pathlib import Path
import json
import shutil
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.multiwoz_aligned import atomic_json
from eval_tod.sgd_standard_skill_agent import SGDStandardSkillAgent
from eval_tod.sgd_errors import error_summary
from eval_tod.sgd_usage import configure_usage, persist_usage
from scripts.replay_multiwoz_reflection import restore_failed_groups
from scripts.run_sgd_domain_eval import predict_policy
from scripts.launch_sgd_homes_pilot import start
from scripts.launch_multiwoz_hotel_pilot import background_python
from scripts.rescore_sgd_policy_results import rescore
from skill_mining.online_refinement import group_batch_reports


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(args):
    source_root, output = args.source_root.resolve(), args.output_dir.resolve()
    source, repaired = source_root / 'ours', output / 'repaired_resources'
    repaired.mkdir(parents=True, exist_ok=False)
    plan = read(source_root / 'plan.json')
    with ZipFile(ROOT / 'data/eval/sgd/data.zip') as archive:
        ontology = json.load(archive.open('data/ontology.json'))
    agent = SGDStandardSkillAgent(family='Homes', model=plan['model'], ontology=ontology,
        predictor=predict_policy, mining_method='trace_cover')
    agent.load(source)
    reports = [read(path) for path in sorted(source.glob('diagnosis_[0-9][0-9][0-9][0-9].json'))]
    groups = group_batch_reports(reports, max_reports=16)
    failed = [json.loads(line) for line in (source / 'case_errors.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert all(row['phase'] == 'online_reflection' and row['rolled_back'] for row in failed)
    indices = [row['group_index'] for row in failed]
    selected = restore_failed_groups(agent, source, repaired, groups, indices)
    address = source_root / 'ours_address_repair'
    address_record = read(address / 'resource_address_repair.json')
    assert address_record['source_sha256'] == agent.resource_hash()
    agent.load(address)
    assert address_record['repaired_sha256'] == agent.resource_hash()
    agent.parallel_workers = args.workers
    agent.runtime_diagnostics_dir = repaired
    agent.reflection_prompt += (
        '\nRecovery serialization: express semantic changes using the native resource updates '
        '(action_rule, slot_policy, reference, transition_guard). Return skill_operations as an empty array '
        'for this repair, so stale text-splice targets cannot invalidate the whole group. '
        'Keep resource bodies as flexible natural-language prose. Address existing nodes/edges only; '
        'unsupported topology belongs in reference. Preserve supported earlier behavior.\n')
    configure_usage(repaired)
    atomic_json(output / 'progress.json', {'phase': 'repair_reflection', 'groups': indices})
    agent.stage = 'refining'
    agent.reflect_groups(selected, repaired, group_indices=indices)
    errors = error_summary(repaired)
    agent.state['refinement_errors'] = errors
    agent.stage = 'refined_with_errors' if errors['total'] else 'refined'
    agent.save(repaired)
    original = read(source / 'ours_training.json')
    training = {**original, 'final_sha256': agent.resource_hash(), 'case_errors': errors,
        'source_run': str(source), 'address_updates_reused': address_record['accepted_updates'],
        'retry_groups': indices, 'new_rollout_prediction_calls': 0,
        'reflection_application_order': [entry['group_index'] for entry in agent.ledger if entry.get('phase') == 'online_reflection']}
    atomic_json(repaired / 'ours_training.json', training)
    atomic_json(repaired / 'resource_address_repair.json', address_record)
    atomic_json(output / 'repair_usage.json', persist_usage())
    if errors['total']:
        raise RuntimeError('Failed reflection remains unresolved; repaired checkpoint is marked with errors')
    command = [str(background_python()), '-u', str(ROOT / 'scripts/run_sgd_domain_eval.py'),
        '--domain', 'Homes', '--method', 'ours', '--split', 'validation', '--eval-targets', 'policy',
        '--splits-dir', str(ROOT / 'data/eval/sgd/splits_10pct_seed42'),
        '--archive', str(ROOT / 'data/eval/sgd/data.zip'), '--model', plan['model'],
        '--workers', str(args.workers), '--ours-resource-dir', str(repaired),
        '--output-dir', str(output / 'evaluation')]
    atomic_json(output / 'progress.json', {'phase': 'evaluation', 'total': 115, 'command': command})
    process = start(command, output / 'evaluation.log')
    atomic_json(output / 'evaluation_launch.json', {'pid': process.pid, 'command': command})
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f'Full policy evaluation exited with {returncode}')
    result = rescore(source_root, output / 'evaluation')
    atomic_json(output / 'comparison.json', result)
    shutil.copy2(source_root / 'comparison.md', output / 'comparison.md')
    atomic_json(output / 'progress.json', {'phase': 'completed', 'completed': 115, 'total': 115})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir / 'run_failed.json', {'error': str(exc), 'type': type(exc).__name__})
        raise
