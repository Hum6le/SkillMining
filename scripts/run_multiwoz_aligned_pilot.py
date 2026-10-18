#!/usr/bin/env python3
"""Fresh MultiWOZ training/evaluation under one persisted execution contract."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_tod.multiwoz_aligned import (
    PROTOCOL, OFFLINE_SLOT_PROTOCOL, INITIAL_SNIPPETS_PER_NODE,
    AlignedBaselinePredictor, MultiWOZAlignedAWMAgent, MultiWOZAlignedSkillAgent,
    ObservationCache, RuntimeLimits, atomic_json,
)
from eval_tod.multiwoz_policy import load_hotel_pilot, ontology_from_train
from eval_tod.multiwoz_slot_policy import ConditionalMultiWOZSkillAgent
from eval_tod.multiwoz_aligned import CONDITIONAL_SLOT_PROTOCOL
from eval_tod.multiwoz_native import NativeMultiWOZSkillAgent, OFFLINE_PROTOCOL as NATIVE_OFFLINE_PROTOCOL
from eval_tod.sgd_adapter import gold_policy, score_policies
from eval_tod.sgd_errors import record_error
from eval_tod.sgd_errors import error_summary
from eval_tod.multiwoz_matched_executor import PROTOCOL as EXECUTOR_PROTOCOL, digest
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase
from scripts.run_multiwoz_matched_eval import recover_logged_usage


def progress(output, phase, completed=0, total=0):
    atomic_json(output / 'progress.json', {'phase': phase, 'completed': completed, 'total': total,
                                          'time': datetime.now().isoformat()})
    print(f'{phase}: {completed}/{total}', flush=True)


def run(args):
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Use a fresh output directory; a legacy/frozen reevaluation is not aligned training')
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and (output / 'run_failed.json').exists():
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        (output / 'run_failed.json').replace(output / f'previous_run_failed_{stamp}.json')
    if args.resume and any(output.rglob('llm_requests.jsonl')):
        recover_logged_usage(output)
    configure_usage(output)
    case_workers = getattr(args, 'case_workers', None) or args.workers
    data = load_hotel_pilot(args.source, seed=args.seed, fraction=args.fraction)
    train = data['train'][:args.max_train_dialogues] if args.max_train_dialogues else data['train']
    eval_cases = [(d, t) for d in data[args.split] for t in d['target_indices']['policy_turn_indices']]
    if args.max_eval_cases:
        eval_cases = eval_cases[:args.max_eval_cases]
    ontology = ontology_from_train(train)
    cache = ObservationCache(args.observation_cache or output / 'observations', args.source.parent, args.model)
    limits = RuntimeLimits(reference_max_rounds=args.reference_max_rounds,
                           reference_context_chars=args.reference_context_chars)
    mode = getattr(args, 'ours_slot_protocol', 'native') if args.method == 'ours' else 'legacy'
    agent_class = {'native': NativeMultiWOZSkillAgent, 'conditional': ConditionalMultiWOZSkillAgent,
                   'legacy': MultiWOZAlignedSkillAgent}[mode]
    offline_protocol = {'native': NATIVE_OFFLINE_PROTOCOL, 'conditional': CONDITIONAL_SLOT_PROTOCOL,
                        'legacy': OFFLINE_SLOT_PROTOCOL}[mode]
    manifest = {'protocol': agent_class.protocol,
        'method': args.method, 'model': args.model, 'seed': args.seed,
        'fraction': args.fraction, 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'training_mode': 'fresh_offline_then_standard_frozen_rollout_and_sequential_reflection',
        'train_dialogue_ids': [d['dialogue_id'] for d in train], 'train_dialogues': len(train),
        'available_train_turns': sum(len(d['target_indices']['policy_turn_indices']) for d in train),
        'max_batches': args.max_batches, 'split': args.split,
        'eval_case_ids': [[d['dialogue_id'], t] for d, t in eval_cases], 'eval_turns': len(eval_cases),
        'workers': args.workers, 'smoke_limits': bool(args.max_train_dialogues or args.max_batches or args.max_eval_cases)}
    manifest['execution_contract'] = {'executor_protocol': EXECUTOR_PROTOCOL, 'model': args.model,
        'db_sha256': cache.db_sha256, 'ontology_sha256': digest(ontology), 'retrieval': asdict(limits)}
    if args.method == 'ours':
        manifest['offline_resources'] = {'protocol': offline_protocol,
            'initial_snippets_per_node': args.card_examples}
    if (output / 'data_manifest.json').exists():
        old = json.loads((output / 'data_manifest.json').read_text(encoding='utf-8'))
        if old != manifest:
            raise ValueError('Resume input/selection/runtime limits changed')
    atomic_json(output / 'data_manifest.json', manifest)
    atomic_json(output / 'train_ontology.json', ontology)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
        return manifest
    if (output / 'summary.json').exists():
        return json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    # Prepare once before any method's learning or evaluation. All resource
    # compilation examples and rollouts consume the same context-only cache.
    all_cases = [(d, t) for d in train for t in d['target_indices']['policy_turn_indices']] + eval_cases
    progress(output, 'shared_db_planning', 0, len(all_cases))
    pending, db_completed = all_cases, 0
    for attempt in range(1, 4):
        failed = []
        cache_workers = min(case_workers, 4) if getattr(args, 'recover_rollout', False) else case_workers
        outcomes = ordered_cases(lambda case: cache.get(*case), pending, workers=cache_workers,
                                 directory=output, phase='shared_db_planning')
        for outcome, case in zip(outcomes, pending):
            try:
                require_value(outcome)
                db_completed += 1
            except Exception as exc:
                failed.append(case)
                with (output / 'db_retry_history.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'attempt': attempt, 'dialogue_id': case[0]['dialogue_id'],
                        'turn_index': case[1], 'error': str(exc)}, ensure_ascii=False) + '\n')
            progress(output, 'shared_db_planning', db_completed, len(all_cases))
        if not failed:
            break
        pending = failed
    if failed:
        raise RuntimeError(f'{len(failed)} DB planning cases failed bounded retries; successful observations retained for --resume')
    cache.allow_generate = False
    if args.method == 'ours':
        agent = agent_class(family='hotel', model=args.model, ontology=ontology,
            predictor=None, observations=cache, limits=limits, mining_method='trace_cover', seed=args.seed)
        agent.parallel_workers = case_workers
        recover_rollout = getattr(args, 'recover_rollout', False)
        if (output / 'ours_training_turns.jsonl').exists() and not recover_rollout:
            raise ValueError('Use a fresh run or explicit reflection recovery for a partially refined checkpoint')
        progress(output, 'offline_compile')
        if recover_rollout:
            from eval_tod.multiwoz_rollout_recovery import prepare_rollout_recovery
            agent.load(output / 'offline')
            replay, recovery = prepare_rollout_recovery(agent, train, output,
                restart_diagnosis=getattr(args, 'restart_diagnosis', False))
            atomic_json(output / 'execution_layout.json', {'original_workers': args.workers,
                'case_workers': case_workers, 'cache_verification_workers': min(case_workers, 4),
                'restart_diagnosis': getattr(args, 'restart_diagnosis', False),
                'reason': 'Resume frozen predictions with an explicit resource-safe execution layout.'})
            agent.predict = replay
            print(f"Frozen rollout recovery: {recovery['successful_cached_turns']} cached; "
                  f"{len(recovery['retry_sample_ids'])} failed cases to retry", flush=True)
        else:
            agent.mine(train, output / 'offline', cases_per_node=args.card_examples)
        offline_contract = agent.runtime_contract()
        progress(output, 'standard_train_rollout_and_reflection')
        training = agent.refine(train, output, batch_size=8, analysis_batch_size=8, max_batches=args.max_batches)
        if recover_rollout:
            del agent.predict
        if training['case_errors']['total']:
            raise RuntimeError('TRAIN/reflection failed; refusing to label this a completed aligned run')
        if agent.runtime_contract() != offline_contract:
            raise AssertionError('Offline, TRAIN rollout and final runtime contract changed')
        agent.save(output)
        # Evaluate through the same persisted artifact/loader used by future
        # invocations, not an in-memory card completion in the evaluator.
        reloaded = agent_class(family='hotel', model=args.model, ontology=ontology,
            predictor=None, observations=cache, limits=limits, mining_method='trace_cover', seed=args.seed)
        reloaded.load(output)
        assert reloaded.resource_hash() == agent.resource_hash()
        agent = reloaded
    else:
        agent = MultiWOZAlignedAWMAgent(family='hotel', model=args.model, ontology=ontology,
            predictor=AlignedBaselinePredictor(cache, limits))
        agent.parallel_workers = case_workers
        for batch, start in enumerate(range(0, len(train), 20), 1):
            progress(output, 'awm_training', start, len(train))
            agent.train_batch(train[start:start + 20], batch_index=batch, trace_path=output / 'awm_training_turns.jsonl')
            agent.save(output)
            if args.max_batches and batch >= args.max_batches:
                break
        training = {'protocol': PROTOCOL, 'training_mode': 'awm_frozen_batch_rollout_then_induction'}
        if error_summary(output)['total']:
            raise RuntimeError('AWM TRAIN failed; refusing a completed aligned comparison')
        frozen = MultiWOZAlignedAWMAgent(family='hotel', model=args.model, ontology=ontology,
            predictor=AlignedBaselinePredictor(cache, limits))
        frozen.load(output)
        agent = frozen
    set_phase('testing')
    progress(output, 'evaluation', 0, len(eval_cases))
    outcomes = ordered_cases(lambda case: agent.predict(*case), eval_cases, workers=case_workers,
                             directory=output, phase='testing')
    records = []
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for n, (outcome, (dialogue, turn)) in enumerate(zip(outcomes, eval_cases), 1):
            try:
                prediction, diagnostic = require_value(outcome)
            except Exception as exc:
                failure = record_error(output, 'testing', exc, dialogue_id=dialogue['dialogue_id'], turn_index=turn)
                prediction = {'call': None, 'acts': [], '_execution_error': True}
                diagnostic = {'status': 'error', 'error': failure['error']}
            row = {'dialogue_id': dialogue['dialogue_id'], 'turn_index': turn,
                   'gold': gold_policy(dialogue, turn), 'prediction': prediction, 'diagnostic': diagnostic}
            records.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            progress(output, 'evaluation', n, len(eval_cases))
    # A compile error can end iteration before the parent merges later worker
    # usage, even though their append-only request logs are already durable.
    recover_logged_usage(output)
    configure_usage(output)
    summary = {**manifest, 'training': training,
        **score_policies([r['gold'] for r in records], [r['prediction'] for r in records]), 'llm_usage': persist_usage()}
    atomic_json(output / 'summary.json', summary)
    if mode == 'native':
        from scripts.audit_multiwoz_aligned_run import audit
        atomic_json(output / 'protocol_audit.json', audit(output))
    progress(output, 'completed', len(records), len(eval_cases))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['ours', 'awm'], default='ours')
    parser.add_argument('--source', type=Path, default=ROOT / 'data/eval/multiwoz21/data/data/dialogues.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--observation-cache', type=Path)
    parser.add_argument('--model', default='qwen3.6-flash')
    parser.add_argument('--fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--ours-slot-protocol', choices=['native', 'legacy', 'conditional'], default='native',
                        help='Native free-text policies by default; legacy/conditional reproduce fixed-emit diagnostics')
    parser.add_argument('--card-examples', type=int, default=INITIAL_SNIPPETS_PER_NODE,
                        help='Diverse TRAIN snippets per node for initial slot mining and reference seeding (default: 4)')
    parser.add_argument('--reference-max-rounds', type=int, default=4)
    parser.add_argument('--reference-context-chars', type=int, default=32768)
    parser.add_argument('--max-train-dialogues', type=int)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--max-eval-cases', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume completed DB/compile caches with identical inputs')
    parser.add_argument('--recover-rollout', action='store_true',
                        help='Reuse a complete frozen rollout and retry only failed cases before diagnosis')
    parser.add_argument('--restart-diagnosis', action='store_true',
                        help='Replay cached predictions and restart diagnosis before any reflection has committed')
    parser.add_argument('--case-workers', type=int,
                        help='Override execution concurrency on recovery without changing the original data manifest')
    args = parser.parse_args()
    if args.restart_diagnosis and not args.recover_rollout:
        parser.error('--restart-diagnosis requires --recover-rollout')
    if args.case_workers is not None and args.case_workers < 1:
        parser.error('--case-workers must be positive')
    if args.recover_rollout and (not args.resume or args.method != 'ours' or args.max_batches):
        parser.error('--recover-rollout requires --resume, method ours, and the full scheduled TRAIN set')
    if args.method == 'ours' and args.ours_slot_protocol in {'native', 'conditional'} and args.card_examples != 4:
        parser.error('Native/conditional slot protocols use four initialization snippets per node')
    if any(v is not None and v < 1 for v in [args.workers, args.card_examples, args.reference_max_rounds,
                                            args.reference_context_chars, args.max_train_dialogues,
                                            args.max_batches, args.max_eval_cases]):
        parser.error('Limits must be positive')
    try:
        run(args)
    except Exception as exc:
        if args.output_dir.exists():
            atomic_json(args.output_dir / 'run_failed.json', {'error': str(exc), 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()
