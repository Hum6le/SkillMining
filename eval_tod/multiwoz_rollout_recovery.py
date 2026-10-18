"""Recover failed frozen TRAIN cases without charging successful predictions again."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from eval_tod.multiwoz_matched_executor import digest, write_json
from eval_tod.sgd_adapter import gold_policy, visible_context


def prepare_rollout_recovery(agent, train, output, *, restart_diagnosis=False):
    """Validate the complete frozen rollout before archiving the old error log."""
    output = Path(output)
    if ((output / 'ours_training.json').exists() or any(output.glob('reflection_*.json'))
            or (not restart_diagnosis and any(output.glob('diagnosis_*.json')))):
        raise ValueError('Rollout recovery must run before diagnosis/reflection has committed results')
    cases = [(d, i) for d in train for i in d['target_indices']['policy_turn_indices']]
    source = output / 'ours_training_turns.jsonl'
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != len(cases):
        raise ValueError('Recovery requires the complete scheduled rollout, including explicit error rows')
    resource_hash = agent.resource_hash()
    failed, successes = [], []
    for row, (dialogue, index) in zip(rows, cases):
        expected = f"{dialogue['dialogue_id']}:{index}"
        if row.get('status') == 'error':
            if row.get('sample_id') != expected:
                raise ValueError('Failed rollout identity/order changed')
            failed.append(expected)
            continue
        evidence = row['sgd_evidence']
        if (row['sample']['sample_id'] != expected or evidence['sample_id'] != expected
                or evidence['gold_policy'] != gold_policy(dialogue, index)
                or evidence['context'] != visible_context(dialogue, index)[-1800:]):
            raise ValueError('Cached rollout identity, gold, or visible context changed')
        diagnostic = evidence['runtime_diagnostic']
        if diagnostic['resource_sha256'] != resource_hash:
            raise ValueError('Cached prediction does not use the loaded frozen offline resources')
        successes.append((expected, evidence['predicted_policy'], diagnostic))
    if not failed and not restart_diagnosis:
        raise ValueError('No failed rollout case needs recovery')
    errors_path = output / 'case_errors.jsonl'
    errors = [json.loads(line) for line in errors_path.read_text(encoding='utf-8').splitlines() if line.strip()] if errors_path.exists() else []
    allowed_phases = {'train_rollout', 'online_diagnosis'} if restart_diagnosis else {'train_rollout'}
    if any(row['phase'] not in allowed_phases for row in errors):
        raise ValueError('Recovery cannot clear errors from other phases')
    if any(f"{row['dialogue_id']}:{row['turn_index']}" not in failed for row in errors if row['phase'] == 'train_rollout'):
        raise ValueError('Historical error identity does not match a failed rollout row')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    history = output / 'rollout_recovery_history' / stamp
    cache = history / 'successful_predictions'
    cache.mkdir(parents=True)
    shutil.copy2(source, history / 'original_rollout.jsonl')
    for sample_id, prediction, diagnostic in successes:
        write_json(cache / (digest(sample_id) + '.json'), {
            'sample_id': sample_id, 'prediction': prediction, 'diagnostic': diagnostic})
    manifest = {'source_sha256': digest(rows), 'snapshot_sha256': resource_hash,
        'successful_cached_turns': len(successes), 'retry_sample_ids': failed,
        'scheduled_turns': len(cases), 'history': str(history),
        'restart_diagnosis_before_any_reflection': restart_diagnosis,
        'policy': 'Reuse identical successful frozen predictions; rerun only failed cases; then perform the original standard diagnosis/reflection once.'}
    write_json(history / 'recovery.json', manifest)
    write_json(output / 'rollout_recovery.json', manifest)
    if restart_diagnosis:
        for path in output.glob('diagnosis_*.json'):
            path.replace(history / path.name)
    if errors_path.exists():
        # Retain historical failures, but exclude resolved attempts from the
        # current error_summary, which recursively finds case_errors.jsonl.
        errors_path.replace(history / 'historical_errors.jsonl')
    return FrozenRolloutReplay(agent.predict, cache, frozenset(failed)), manifest


class FrozenRolloutReplay:
    def __init__(self, predictor, cache, retry_ids, attempts=2):
        self.predictor, self.cache = predictor, Path(cache)
        self.retry_ids, self.attempts = retry_ids, attempts

    def __call__(self, dialogue, index):
        sample_id = f"{dialogue['dialogue_id']}:{index}"
        path = self.cache / (digest(sample_id) + '.json')
        if path.exists():
            row = json.loads(path.read_text(encoding='utf-8'))
            if row['sample_id'] != sample_id:
                raise ValueError('Cached sample identity mismatch')
            return row['prediction'], row['diagnostic']
        if sample_id not in self.retry_ids:
            raise ValueError('Missing successful cached prediction; refusing a paid full rerun')
        for attempt in range(self.attempts):
            try:
                prediction, diagnostic = self.predictor(dialogue, index)
                write_json(path, {'sample_id': sample_id, 'prediction': prediction, 'diagnostic': diagnostic})
                return prediction, diagnostic
            except (ValueError, TypeError, KeyError):
                if attempt + 1 == self.attempts:
                    raise
