#!/usr/bin/env python3
"""Local 1/10 SGD ReAct / Reflexion / ExpeL / Skill-DisCo, frozen validation."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_tod.multiwoz_aligned import atomic_json
from eval_tod.sgd_local_config import DEFAULT_ENV
from eval_tod.sgd_extra_config import configure
from eval_tod.sgd_adapter import iter_family_dialogues, score_policies, metrics_from_counts
from eval_tod.sgd_aligned import training_contract, PROTOCOL, digest
from eval_tod.sgd_local_parallel import local_cases
from eval_tod.sgd_parallel import require_value
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase
from eval_tod.sgd_extra_baselines import (predict_row, predict_dialogue, reflexion_dialogue,
    reflection_knowledge, induce_expel, Reflection, ReflectionStore, ExpeLRuleStore,
    EXPEL_PROTOCOL, RULE_UPDATE_PROTOCOL)

ADAPTER = 'sgd_extra_baselines_train_only_frozen_validation_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def resume_manifest(manifest, previous, out):
    """Worker count is scheduling metadata; retain the original hash on resume."""
    if previous is None:
        return manifest
    semantic = lambda row: {k:v for k,v in row.items() if k != 'workers'}
    if semantic(previous) != semantic(manifest):
        raise ValueError(f'Resume protocol/data mismatch: {out}')
    return previous


def families(root):
    index = read(root/'INDEX.json')
    return [family for family, row in index['families'].items()
            if row.get('train', {}).get('policy_turns', 0)
            and row.get('validation', {}).get('policy_turns', 0)]


def journal_usage(out):
    """Rebuild once from unique request journals, including interrupted cases."""
    from scripts.llm_usage_utils import merge_usage_summaries
    by_phase = {}
    for path in out.rglob('llm_requests.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            phase = row['phase']
            by_phase[phase] = merge_usage_summaries(by_phase.get(phase, {'total':{}}),
                {'total':row['total'], 'by_call_tag':{row['call_tag']:row['total']}})
    empty = {'total':{}}
    result = {'generation':merge_usage_summaries(empty,*(u for p,u in by_phase.items() if p!='testing')),
        'testing':merge_usage_summaries(empty,by_phase.get('testing',empty)),
        'total':merge_usage_summaries(empty,*by_phase.values()), 'by_phase':by_phase,
        'accounting_complete':True, 'source':'all_unique_request_journals',
        'note':'Includes retries and interrupted attempts. Provider counts where returned; '
               'failed-request tokens without provider usage are estimates, not billed usage.'}
    atomic_json(out/'llm_usage.json',result)
    return result


def run_family(args, family, config):
    out = args.output_dir/family
    out.mkdir(parents=True, exist_ok=True)
    train = list(iter_family_dialogues(args.splits_dir, family, 'train'))
    evaluation = list(iter_family_dialogues(args.splits_dir, family, 'validation'))
    if args.max_train_dialogues:
        train = train[:args.max_train_dialogues]
    if args.max_eval_dialogues:
        evaluation = evaluation[:args.max_eval_dialogues]
    if {d['dialogue_id'] for d in train} & {d['dialogue_id'] for d in evaluation}:
        raise ValueError('TRAIN/validation dialogue overlap')
    with ZipFile(args.archive) as zipped:
        ontology_bytes = zipped.read('data/ontology.json')
    manifest = {'adapter': ADAPTER, 'execution_protocol': PROTOCOL, 'method': args.method,
        'family': family, 'model': config['model'], 'provider': config['provider'],
        'base_url': config['base_url'], 'enable_thinking': config['enable_thinking'],
        'workers': args.workers, 'evaluation_targets': 'policy', 'split': 'validation',
        'max_trials': args.max_trials, 'batch_dialogues': args.batch_dialogues,
        'max_rules': args.max_rules, 'reflection_limit': 3,
        'train_ids': [d['dialogue_id'] for d in train],
        'validation_ids': [d['dialogue_id'] for d in evaluation],
        'train_turns': sum(len(d['target_indices']['policy_turn_indices']) for d in train),
        'validation_turns': sum(len(d['target_indices']['policy_turn_indices']) for d in evaluation),
        'data_sha256': {s: hashlib.sha256((args.splits_dir/family/(s+'.jsonl.gz')).read_bytes()).hexdigest()
                        for s in ['train', 'validation']},
        'ontology_sha256': hashlib.sha256(ontology_bytes).hexdigest(),
        'max_eval_turns': args.max_eval_turns,
        'checkpoint_selection': 'final_training_artifact_no_validation_selection',
        'skill_disco_runtime': 'compiled_predicted_prefix_heads_shared_payload_binder',
        'skill_disco_verification': '20pct_whole_dialogue_holdout_within_TRAIN_only'}
    if args.method == 'skill_disco':
        from skill_disco.sgd_adapter import RUNTIME
        manifest['skill_disco_runtime'] = RUNTIME
    if args.method == 'expel':
        manifest['expel_protocol'] = EXPEL_PROTOCOL
        manifest['rule_update_protocol'] = RULE_UPDATE_PROTOCOL
    if config['provider'] == 'workflow':
        manifest['workflow_pool_method'] = config['workflow_pool_method']
        manifest['workflow_settings'] = config['workflow_settings']
    previous = read(out/'data_manifest.json')
    manifest = resume_manifest(manifest, previous, out)
    atomic_json(out/'data_manifest.json', manifest)
    complete = read(out/'summary.json', {})
    if complete.get('evaluation_complete'):
        if complete.get('manifest_sha256') != digest(manifest):
            raise ValueError('Completed summary has a different manifest')
        print(f'SKIP verified {args.method}/{family}', flush=True)
        return complete
    if args.dry_run:
        return manifest
    # The data manifest keeps its initial worker count for immutable result
    # hashes. Record actual resumed concurrency separately for auditability.
    with (out/'execution_history.jsonl').open('a', encoding='utf-8') as history:
        history.write(json.dumps({'started_at': now(), 'workers': args.workers,
                                  'manifest_initial_workers': manifest['workers']})+'\n')
    configure_usage(out)
    ontology = json.loads(ontology_bytes)
    ontology['_sgd_training_contract'] = training_contract(train)
    started = time.time()
    def progress(phase, completed, total, **extra):
        state = {'method': args.method, 'family': family, 'phase': phase,
            'completed': completed, 'total': total, 'updated_at': now(),
            'elapsed_seconds': time.time()-started, 'workers': args.workers, **extra}
        atomic_json(out/'progress.json', state)
        atomic_json(args.output_dir/'progress.json', state)
        if completed % 20 == 0 or completed == total:
            print(f"{args.method}/{family} {phase}: {completed}/{total}", flush=True)
    resources = read(out/'resources.json')
    if resources is None:
        if args.method == 'react':
            resources = {'type': 'react', 'training': 'none', 'knowledge': ''}
        elif args.method == 'reflexion':
            target_dir = out/'training_dialogues'
            target_dir.mkdir(exist_ok=True)
            pending = [d for d in train if not (target_dir/(d['dialogue_id']+'.json')).exists()]
            progress('reflexion_training', len(train)-len(pending), len(train))
            def job(dialogue):
                result = reflexion_dialogue(dialogue, config['model'], ontology,
                    max_trials=args.max_trials,
                    checkpoint=target_dir/(dialogue['dialogue_id']+'.partial.json'))
                atomic_json(target_dir/(dialogue['dialogue_id']+'.json'), result)
                return dialogue['dialogue_id']
            for result in local_cases(job, pending, workers=args.workers, directory=out,
                    phase='reflexion_training', progress=lambda n, elapsed:
                    progress('reflexion_training', len(train)-len(pending)+n, len(train))):
                require_value(result)
            notes = [note for d in train for note in read(target_dir/(d['dialogue_id']+'.json'))['reflections']]
            resources = {'type': 'reflexion', 'reflections': notes,
                'max_trials': args.max_trials, 'reflection_limit': 3}
        elif args.method == 'expel':
            checkpoint = read(out/'expel_training_state.json', {})
            store = ExpeLRuleStore(max_rules=args.max_rules)
            if checkpoint:
                from expel_adapter.abcd import _Rule
                store.rules = [_Rule(**r) for r in checkpoint['rules']]
            batches = [train[i:i+args.batch_dialogues] for i in range(0, len(train), args.batch_dialogues)]
            for number, batch in enumerate(batches, 1):
                if number <= checkpoint.get('completed_batches', 0):
                    continue
                knowledge = 'ExpeL training-derived insights:\n'+store.text if store.rules else ''
                progress('expel_rollout', 0, len(batch), batch=number, batches=len(batches))
                rows = [require_value(r) for r in local_cases(
                    lambda d: predict_dialogue(d, config['model'], ontology, knowledge), batch,
                    workers=args.workers, directory=out, phase='expel_rollout',
                    progress=lambda n, elapsed: progress('expel_rollout', n, len(batch),
                                                         batch=number, batches=len(batches)))]
                set_phase('expel_rule_induction')
                induction = induce_expel(store, batch, rows, model=config['model'], ontology=ontology)
                batch_path = out/'training_batches'/f'{number:04d}.json'
                batch_path.parent.mkdir(exist_ok=True)
                atomic_json(batch_path, {'turns': rows, 'induction': induction})
                checkpoint = {'completed_batches': number, 'rules': [asdict(r) for r in store.rules]}
                atomic_json(out/'expel_training_state.json', checkpoint)
                progress('expel_rule_induction', number, len(batches), rules=len(store.rules))
            resources = {'type': 'expel', 'rules': [asdict(r) for r in store.rules], 'knowledge': store.text,
                         'learning_protocol': EXPEL_PROTOCOL, 'rule_update_protocol': RULE_UPDATE_PROTOCOL}
        else:
            from skill_disco.sgd_adapter import mine
            progress('skill_disco_discovery', 0, len(train))
            artifact = mine(train, out, model=config['model'], workers=args.workers,
                min_support=1 if args.smoke else 2,
                progress=lambda n, elapsed: progress('skill_disco_discovery', n,
                    len(train)-max(1, min(len(train)-1, round(len(train)*.2)))))
            atomic_json(out/'generation_artifact.json', artifact)
            resources = {'type': 'skill_disco', 'artifact': artifact}
        atomic_json(out/'resources.json', resources)
    frozen_hash = digest(resources)
    store = ReflectionStore([Reflection(**r) for r in resources.get('reflections', [])])
    prior_rows = []
    prediction_file = out/'predictions.jsonl'
    if prediction_file.exists():
        prior_rows = [json.loads(line) for line in prediction_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    existing = {r['sample_id']: r for r in prior_rows}
    if len(existing) != len(prior_rows):
        raise ValueError('Duplicate evaluation sample checkpoint')
    expected_cases = [(d, i) for d in evaluation for i in d['target_indices']['policy_turn_indices']]
    if args.max_eval_turns:
        expected_cases = expected_cases[:args.max_eval_turns]
    expected_ids = {f"{d['dialogue_id']}:{i}" for d, i in expected_cases}
    if not existing.keys() <= expected_ids:
        raise ValueError('Evaluation checkpoint includes unexpected sample IDs')
    def evaluate(case):
        dialogue, i = case
        knowledge = (reflection_knowledge(store, dialogue, i, limit=3) if args.method=='reflexion'
                     else resources.get('knowledge', ''))
        return predict_row(dialogue, i, config['model'], ontology, knowledge)
    if args.method == 'skill_disco':
        from skill_disco.sgd_adapter import predict as disco_predict
        def evaluate_dialogue(dialogue):
            result = []
            for i in dialogue['target_indices']['policy_turn_indices']:
                sid = f"{dialogue['dialogue_id']}:{i}"
                if sid not in expected_ids:
                    continue
                if sid in existing:
                    result.append(existing[sid]); continue
                try:
                    pred, diagnostic = disco_predict(dialogue, i, model=config['model'],
                        ontology=ontology, artifact=resources['artifact'], previous=result)
                    from eval_tod.sgd_adapter import gold_policy
                    gold = gold_policy(dialogue, i)
                    row = {'sample_id': sid, 'dialogue_id': dialogue['dialogue_id'],
                        'turn_index': i, 'gold': gold, 'prediction': pred, 'diagnostic': diagnostic,
                        'joint_correct': bool(score_policies([gold], [pred])['turn_joint_ast']), 'error': None}
                except Exception as exc:
                    from eval_tod.sgd_adapter import gold_policy
                    row = {'sample_id': sid, 'dialogue_id': dialogue['dialogue_id'], 'turn_index': i,
                        'gold': gold_policy(dialogue, i), 'prediction': {'call': None, 'acts': [], '_execution_error': True},
                        'error': repr(exc), 'joint_correct': False, 'diagnostic': {}}
                result.append(row)
                case_path = out/'evaluation_dialogues'/f'{dialogue["dialogue_id"]}.json'
                case_path.parent.mkdir(exist_ok=True)
                atomic_json(case_path, result)
            return result
        # Recover complete/partial dialogue checkpoints without rerunning paid calls.
        for file in (out/'evaluation_dialogues').glob('*.json') if (out/'evaluation_dialogues').exists() else []:
            for row in read(file):
                if row['sample_id'] in expected_ids:
                    existing.setdefault(row['sample_id'], row)
        jobs = [d for d in evaluation if any(f"{d['dialogue_id']}:{i}" in expected_ids-existing.keys()
                for i in d['target_indices']['policy_turn_indices'])]
        function = evaluate_dialogue
    else:
        jobs = [(d, i) for d, i in expected_cases if f"{d['dialogue_id']}:{i}" not in existing]
        function = evaluate
    progress('testing', len(existing), len(expected_cases))
    with prediction_file.open('a', encoding='utf-8') as stream:
        for outcome in local_cases(function, jobs, workers=args.workers, directory=out, phase='testing'):
            result = require_value(outcome)
            for row in result if isinstance(result, list) else [result]:
                if row['sample_id'] in existing:
                    continue
                row['frozen_resource_sha256'] = frozen_hash
                stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
                existing[row['sample_id']] = row
            progress('testing', len(existing), len(expected_cases))
    if existing.keys() != expected_ids:
        raise RuntimeError('Evaluation incomplete')
    if digest(read(out/'resources.json')) != frozen_hash:
        raise RuntimeError('Resource changed during frozen evaluation')
    rows = [existing[f"{d['dialogue_id']}:{i}"] for d, i in expected_cases]
    # Dialogue checkpoints may have survived a crash before the parent journal
    # was appended. Materialize the exact complete set once, including them.
    for row in rows:
        if row.get('frozen_resource_sha256', frozen_hash) != frozen_hash:
            raise RuntimeError('Recovered prediction belongs to a different resource')
        row['frozen_resource_sha256'] = frozen_hash
    temporary = prediction_file.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    temporary.replace(prediction_file)
    metrics = score_policies([r['gold'] for r in rows], [r['prediction'] for r in rows])
    summary = {'method': args.method, 'domain': family, 'model': config['model'],
        'evaluation_complete': True, 'manifest_sha256': digest(manifest),
        'resource_sha256': frozen_hash, 'metrics': metrics, **metrics,
        'case_errors': sum(bool(r['error']) for r in rows),
        'execution_status': 'completed_with_case_errors' if any(r['error'] for r in rows) else 'completed',
        'generation': {'reflections': len(resources.get('reflections', [])),
            'rules': len(resources.get('rules', [])),
            'verified_skills': len(resources.get('artifact', {}).get('verified_contracts', [])),
            'discovery_errors': len(resources.get('artifact', {}).get('discovery_errors', []))},
        'llm_usage': journal_usage(out), 'finished_at': now()}
    atomic_json(out/'summary.json', summary)
    progress('complete', len(rows), len(rows), case_errors=summary['case_errors'])
    return summary


def run(args):
    config = configure(args)
    chosen = args.families.split(',') if args.families else families(args.splits_dir)
    if any(f not in families(args.splits_dir) for f in chosen):
        raise ValueError('Only the historical nine seen-schema validation families are allowed')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir/'run_config.json', {**config, 'method': args.method,
        'families': chosen, 'workers': args.workers, 'smoke': args.smoke,
        'splits_dir': str(args.splits_dir.resolve()), 'started_at': now()})
    summaries = []
    try:
        for family in chosen:
            summaries.append(run_family(args, family, config))
            if args.dry_run:
                continue
            from collections import Counter
            counts = Counter()
            for s in summaries:
                counts.update(s['metrics']['counts'])
            aggregate = {'method': args.method, 'model': config['model'],
                'completed_families': [s['domain'] for s in summaries], 'expected_families': chosen,
                'evaluation_complete': len(summaries)==len(chosen), 'smoke': args.smoke,
                **metrics_from_counts(counts), 'case_errors': sum(s['case_errors'] for s in summaries),
                'families': {s['domain']: s['metrics'] for s in summaries}}
            from scripts.llm_usage_utils import merge_usage_summaries
            aggregate['llm_usage'] = {phase: merge_usage_summaries({'total': {}},
                *(s['llm_usage'].get(phase, {}) for s in summaries))
                for phase in ['generation', 'testing', 'total']}
            aggregate['generation'] = {s['domain']: s['generation'] for s in summaries}
            atomic_json(args.output_dir/'aggregate.json', aggregate)
            if getattr(args, 'comparison_root', None):
                from scripts.launch_sgd_extra_baselines import collect
                collect(args.comparison_root)
        return summaries
    except Exception as exc:
        atomic_json(args.output_dir/'fatal_error.json', {'error': repr(exc), 'time': now()})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', required=True, choices=['react','reflexion','expel','skill_disco'])
    p.add_argument('--splits-dir', type=Path, default=ROOT/'data/eval/sgd/protocol_10pct_seed42_seen_schema_v1')
    p.add_argument('--archive', type=Path, default=ROOT/'data/eval/sgd/data.zip')
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--env-file', type=Path, default=DEFAULT_ENV)
    p.add_argument('--comparison-root', type=Path,
                   help='Refresh the existing combined result table after each completed family')
    p.add_argument('--model', default=None)
    p.add_argument('--provider', choices=['shubiaobiao','workflow'], default='shubiaobiao')
    p.add_argument('--workflow-pools', type=Path, default=ROOT/'configs/sgd_workflows.json')
    p.add_argument('--workflow-pool-method', default='awm',
                   help='Reuse an existing ID pool in configs/sgd_workflows.json')
    p.add_argument('--families', default='')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-trials', type=int, default=2)
    p.add_argument('--batch-dialogues', type=int, default=20)
    p.add_argument('--max-rules', type=int, default=20)
    p.add_argument('--max-train-dialogues', type=int)
    p.add_argument('--max-eval-dialogues', type=int)
    p.add_argument('--max-eval-turns', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    max_workers = (10 if args.provider == 'workflow' else
                   12 if args.method == 'expel' else 8 if args.method == 'reflexion' else 4)
    if args.workers < 1 or args.workers > max_workers or args.max_trials < 1 or args.batch_dialogues < 1:
        p.error(f'Use 1..{max_workers} workers for this method; trials/batch size must be positive')
    if any(v is not None and v < 1 for v in [args.max_train_dialogues,args.max_eval_dialogues,args.max_eval_turns]):
        p.error('Smoke limits must be positive')
    if not args.smoke and any(v is not None for v in [args.max_train_dialogues,args.max_eval_dialogues,args.max_eval_turns]):
        p.error('Truncation requires --smoke and cannot be reported as a full run')
    run(args)


if __name__ == '__main__':
    main()
