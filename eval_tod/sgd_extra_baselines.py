"""SGD task bindings for the existing ABCD ReAct, Reflexion and ExpeL protocols."""
from copy import deepcopy
from dataclasses import asdict
import json
import re

from expel_adapter.abcd import ExpeLRuleStore, _parse_operations, RULE_UPDATE_PROTOCOL
from reflexion_adapter.abcd import Reflection, ReflectionStore
from eval_tod.sgd_adapter import gold_policy, score_policies, visible_context
from eval_tod.sgd_aligned import predict_policy, TASK
from eval_tod.sgd_llm import sgd_chat_with_retry
from eval_tod.sgd_parallel import require_value

EXPEL_PROTOCOL = 'abcd_aligned_train_gold_policy_pairs_frozen_validation_v2'


def query(dialogue):
    return ' '.join(t.get('utterance', '') for t in dialogue['turns']
                    if t['speaker'].lower() == 'user')[:1000]


def redact(text, dialogue):
    values = set()
    for index in dialogue['target_indices']['policy_turn_indices']:
        policy = gold_policy(dialogue, index)
        values.update(a['value'] for a in policy['acts'] if 'value' in a)
        if policy['call']:
            values.update(policy['call']['parameters'].values())
    for value in sorted(values, key=len, reverse=True):
        if len(value) >= 2 or value.isdigit():
            text = re.sub(r'(?<!\w)'+re.escape(value)+r'(?!\w)', '[training_value]',
                          text, flags=re.I)
    return text


def reflection_knowledge(store, dialogue, index, *, same=False, limit=3):
    selected = store.select(visible_context(dialogue, index),
        conversation_id=dialogue['dialogue_id'], same_conversation=same, limit=limit)
    if not selected:
        return ''
    return ('Training self-reflections (apply only when relevant; never copy instance values):\n'
            + '\n'.join(f'{n}. {row.text[:800]}' for n, row in enumerate(selected, 1)))


def predict_row(dialogue, index, model, ontology, knowledge=''):
    try:
        prediction, diagnostic = predict_policy(dialogue, index, model=model,
            ontology=ontology, skill_text=knowledge)
        error = None
    except Exception as exc:
        prediction = {'call': None, 'acts': [], '_execution_error': True}
        diagnostic = {}
        error = f'{type(exc).__name__}: {exc}'
    # Gold is accessed only after the actor returns; none enters its prompt.
    gold = gold_policy(dialogue, index)
    return {'sample_id': f"{dialogue['dialogue_id']}:{index}",
            'dialogue_id': dialogue['dialogue_id'], 'turn_index': index,
            'gold': gold, 'prediction': prediction, 'diagnostic': diagnostic,
            'joint_correct': bool(score_policies([gold], [prediction])['turn_joint_ast']),
            'error': error}


def predict_dialogue(dialogue, model, ontology, knowledge=''):
    return [predict_row(dialogue, i, model, ontology, knowledge)
            for i in dialogue['target_indices']['policy_turn_indices']]


def reflexion_dialogue(dialogue, model, ontology, *, max_trials=2, limit=3, checkpoint=None):
    """ABCD semantics: retry each TRAIN episode with its own raw reflections."""
    from eval_tod.multiwoz_aligned import atomic_json
    state = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint and checkpoint.exists() else {}
    store = ReflectionStore([Reflection(**r) for r in state.get('reflections', [])])
    trials = state.get('trials', [])
    if trials and trials[-1]['succeeded']:
        return {'trials':trials,'reflections':[asdict(r) for r in store.reflections]}
    def save(active):
        if checkpoint:
            atomic_json(checkpoint, {'trials': trials, 'reflections': [asdict(r) for r in store.reflections],
                                     'active_rows': active})
    for trial in range(len(trials)+1, max_trials+1):
        rows = state.pop('active_rows', [])
        done = {r['turn_index'] for r in rows}
        for i in dialogue['target_indices']['policy_turn_indices']:
            if i in done:
                continue
            rows.append(predict_row(dialogue, i, model, ontology,
                    reflection_knowledge(store, dialogue, i, same=True, limit=limit)))
            save(rows)
        metrics = score_policies([r['gold'] for r in rows], [r['prediction'] for r in rows])
        success = metrics['turn_joint_ast'] == 1
        errors = [r for r in rows if not r['joint_correct'] and not r['error']]
        note = ''
        if not success and errors:
            feedback = [{'turn_index': r['turn_index'],
                'dialogue_prefix': visible_context(dialogue, r['turn_index'])[-600:],
                'predicted_policy': r['prediction'], 'expected_policy': r['gold']}
                for r in errors[:4]]
            prompt = ('You are the self-reflection component of an SGD agent. '
                'Write one actionable plain-text lesson, at most 120 words, for the next '
                'training attempt. Explain action-set or named payload errors using the '
                'TRAIN feedback. Do not copy instance values, IDs or dialogue IDs. '
                'Do not invent environment outcomes.\n'+TASK
                +f'\nTrial: {trial}\nCustomer request: {query(dialogue)[:600]}'
                +'\nPrevious reflections: '+json.dumps([r.text for r in store.reflections[-limit:]])
                +'\nTraining feedback: '+json.dumps(feedback, ensure_ascii=False))
            note = redact(sgd_chat_with_retry(prompt, model=model, temperature=0,
                call_tag='sgd_reflexion_self_reflection').strip(), dialogue)[:1200]
            store.reflections.append(Reflection(dialogue['dialogue_id'], trial,
                query(dialogue), note, metrics['turn_joint_ast']))
        trials.append({'conversation_id': dialogue['dialogue_id'], 'trial_index': trial,
            'metrics': metrics, 'succeeded': success, 'reflection': note, 'turns': rows})
        save([])
        if success:
            break
    return {'trials': trials, 'reflections': [asdict(r) for r in store.reflections]}


def induce_expel(store, dialogues, all_rows, *, model, ontology=None):
    """Match local ABCD TRAIN evidence: predictions, gold targets and episode outcome.

    Gold is visible only to this offline inducer, after actor rollouts. No dense
    correctness flags are included. Validation actors receive frozen rules only.
    This is the project's supervised ExpeL adaptation, not an assertion that
    the original ExpeL benchmark supplies expert answers.
    """
    evidence = []
    for dialogue, rows in zip(dialogues, all_rows):
        # Transport errors are not behavioral evidence. They remain counted in
        # rollout statistics and are listed as exclusions in the checkpoint.
        if any(r['error'] for r in rows):
            continue
        evidence.append({'outcome': 'SUCCESS' if all(r['joint_correct'] for r in rows) else 'FAILURE',
            'trajectory': [{'turn_index': r['turn_index'], 'speaker': 'system',
                            'visible_prefix': visible_context(dialogue, r['turn_index']),
                            'predicted_policy': r['prediction'], 'gold_policy': r['gold'],
                            'reference_response': dialogue['turns'][r['turn_index']].get('utterance', '')}
                           for r in rows]})
    if not evidence:
        return {'operations': [], 'raw_output': '', 'skipped': 'no valid episode evidence'}
    schema = {}
    if ontology is not None:
        from eval_tod.sgd_aligned import schema_for_dialogue
        schema = schema_for_dialogue(ontology, sorted({s for d in dialogues for s in d['services']}))
    prompt = ('You analyze SGD SYSTEM agent experiences. Compare TRAIN episodes jointly. '
        'SUCCESS means the entire episode has exact action sets and named payloads. '
        'As in the local ABCD adapter, offline evidence includes predicted and gold policies '
        'and the TRAIN reference response, plus episode outcome; no per-turn correctness flags. '
        'Even a FAILURE episode can contain correct decisions: compare each prediction with '
        'its TRAIN gold target and preserve supported behavior. Gold targets are authoritative '
        'for observed action semantics, named slots and value presence. The reference response '
        'is TRAIN evidence, never an input available to an inference actor. '
        'All targets are SYSTEM acts, not labels for the user\'s utterance. '
        'Do not relabel an act merely by a keyword in the user request. '
        'Extract conditional, transferable insights grounded in observable context. '
        'Never assume an unobserved gold action; never copy names, IDs or concrete values. '
        'Use exact methods and slot names from TRAIN gold and the supplied service schemas. '
        'Keep API parameter formatting separate from dialogue-act formatting; preserve whether '
        'the value field is absent, empty, or populated. Avoid universal rules from isolated cases. '
        'Use at most four operations, one per line, plain text only: ADD: <actual rule>; '
        'AGREE 1: <reason>; EDIT 1: <entire new rule text>; REMOVE 1: <reason>. '
        'Numbers reference the existing list BEFORE this update. When the list is empty, '
        'only ADD is allowed. EDIT contains the replacement rule itself, without '
        '\"Replace rule N with\", old-to-new arrows, or edit instructions. '
        'If no supported update exists, return NO_CHANGES.\n'+TASK
        +'\nLegal TRAIN service schemas:\n'+json.dumps(schema, ensure_ascii=False)
        +'\nExisting rules:\n'+(store.text or '(none)')
        +'\nJoint episode evidence:\n'+json.dumps(evidence[:20], ensure_ascii=False))
    messages = [{'role':'system','content':'Return only the requested ExpeL rule operations or NO_CHANGES.'},
                {'role':'user','content':prompt}]
    for attempt in range(3):
        raw = sgd_chat_with_retry(messages, model=model, temperature=0,
            call_tag='sgd_expel_rule_induction' if attempt == 0 else 'sgd_expel_rule_induction_repair')
        operations = _parse_operations(raw)
        invalid = [op for op, _ in operations if op != 'ADD' and
                   (len(op.split()) != 2 or not op.split()[1].isdigit()
                    or not 1 <= int(op.split()[1]) <= len(store.rules))]
        if not invalid and (operations or raw.strip() == 'NO_CHANGES'):
            break
        if attempt == 2:
            raise ValueError('ExpeL rule update must use valid existing rule numbers or NO_CHANGES')
        messages.extend([{'role':'assistant','content':raw}, {'role':'user','content':
            f'Invalid operation numbering or format. There are {len(store.rules)} existing rules. '
            'Use only ADD if there are none; otherwise valid 1-based references. Return operations or NO_CHANGES.'}])
    store.apply(operations)
    return {'raw_output': raw, 'operations': operations, 'rules': store.text,
            'episodes_used': len(evidence), 'episodes_excluded_transport': len(dialogues)-len(evidence),
            'episode_successes':sum(e['outcome']=='SUCCESS' for e in evidence),
            'protocol':EXPEL_PROTOCOL, 'rule_updates':RULE_UPDATE_PROTOCOL,
            'supervision':'TRAIN predicted/gold policy pairs and reference responses; episode outcome; no dense correctness flags'}
