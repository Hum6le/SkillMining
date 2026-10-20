"""Native SGD policy-turn IR for the five-stage ABCD Skill-DisCo implementation.

Each primitive represents one unordered system policy. Its three IR arguments
serialize the API method, named API parameters and named dialogue acts; this
transport order does not impose order on the SGD labels. The original stage
implementations, contract parser, source validator and isolated replay worker
are reused. No validation dialogue is used for mining or verification.
"""
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from skill_disco.abcd_trace import NormalizedABCDTrace, NormalizedActionStep, NormalizedDialogueEvent
from skill_disco.semantic_abstraction import annotate_trace_semantics
from skill_disco.operation_extraction import extract_trace_operations, semantic_operation_from_dict
from skill_disco.consolidation import group_operation_batch, consolidate_groups
from skill_disco.skill_specification import specify_skill_contract, skill_contract_from_dict
from skill_disco.synthesis import synthesize_and_verify
from skill_disco.abcd_verification import verify_abcd_replay, _templates
from skill_disco.compiled_pipeline import split_induction_and_verification
from skill_disco.pseudocode import render_skill_library
from skill_disco.name_resolution import make_verified_names_unique
from skill_disco.callable_runtime import CompiledSkillLibrary
from eval_tod.sgd_adapter import gold_policy, visible_context
from eval_tod.sgd_aligned import TASK, input_text, predict_policy, request_stage_json
from eval_tod.sgd_llm import sgd_chat_with_retry
from eval_tod.sgd_local_parallel import local_cases
from eval_tod.sgd_parallel import require_value
from eval_tod.multiwoz_aligned import atomic_json
from eval_tod.sgd_usage import set_phase
from eval_tod.sgd_standard_skill_agent import normalize_action_heads


RUNTIME = 'compiled_schema_scoped_prefix_heads_shared_payload_binder_v2'


def legal_projection_heads(heads, services):
    # A compiled primitive names an exact schema version. Never remap it to
    # another service merely because both services share a domain family.
    if any(not isinstance(h, (list, tuple)) or len(h) != 2
           or (h[1] and h[1] not in services) for h in heads):
        raise ValueError('Compiled policy references an unavailable service schema')
    return normalize_action_heads({'action_heads': [list(h) for h in heads]}, services)


def applicable_artifact(artifact, services):
    """Scope compiled tools to the public service IDs of the current task."""
    result = deepcopy(artifact)
    compatible = []
    for item in result.get('compiled_skills', []):
        if item.get('status') != 'verified':
            continue
        contract = skill_contract_from_dict(item['contract'])
        templates = _templates(contract)
        try:
            if not templates:
                continue
            for name, _ in templates:
                legal_projection_heads(artifact['policy_vocabulary'][name], services)
        except (ValueError, TypeError, KeyError):
            continue
        compatible.append(item)
    result['compiled_skills'] = compatible
    result['verified_contracts'] = [deepcopy(item['contract']) for item in compatible
                                    if item.get('implementation')]
    result['skill_library'] = render_skill_library(
        [skill_contract_from_dict(item['contract']) for item in compatible])
    return result


def heads(policy):
    result = {(a['intent'], a['domain']) for a in policy['acts']}
    if policy['call']:
        result.add(('call', policy['call']['service']))
    return sorted(result)


def label(policy):
    return 'policy__'+'__'.join(re.sub(r'\W', '_', i+'_'+s) for i, s in heads(policy))


def slots(policy):
    call = policy['call']
    return [json.dumps(call['method'] if call else None, ensure_ascii=False),
            json.dumps(call['parameters'] if call else {}, sort_keys=True, ensure_ascii=False),
            json.dumps(policy['acts'], sort_keys=True, ensure_ascii=False)]


def normalize(dialogue):
    targets = set(dialogue['target_indices']['policy_turn_indices'])
    events, steps = [], []
    for index, turn in enumerate(dialogue['turns']):
        text = turn.get('utterance', '')
        if index in targets:
            policy = gold_policy(dialogue, index)
            names = [f'step{len(steps)}_api_method', f'step{len(steps)}_api_parameters',
                     f'step{len(steps)}_dialogue_acts']
            parameterized = label(policy)+'('+', '.join(names)+')'
            # The post-action utterance is TRAIN evidence; runtime never sees it.
            steps.append(NormalizedActionStep(index, label(policy), slots(policy), names,
                parameterized, visible_context(dialogue, index).splitlines(), text))
            event_type, parameterized_text = 'backend_action', parameterized
        else:
            event_type = 'customer_observation' if turn['speaker'].lower() == 'user' else 'agent_response'
            parameterized_text = text
        events.append(NormalizedDialogueEvent(index, turn['speaker'], event_type, text, parameterized_text))
    return NormalizedABCDTrace(dialogue['dialogue_id'], 'SGD', 'SGD', events, steps)


def chat(prompt, **kwargs):
    note = ('This is SGD, not ABCD. A backend action in the shared IR means ONE SGD '
        'policy decision, including call and non-call turns. The IR argument order '
        'is serialization only; SGD acts and named parameters are UNORDERED. '
        'api_method is a JSON string/null, api_parameters a JSON-object string, '
        'dialogue_acts a JSON-array string. Preserve absent versus empty value. '
        'Do not infer a fixed pipeline from an isolated example.\n'+TASK+'\n')
    return sgd_chat_with_retry(note+prompt, call_tag='sgd_skill_disco_induction', **kwargs)


def stage(function, *args, **kwargs):
    last = None
    for _ in range(3):
        try:
            return function(*args, **kwargs)
        except (ValueError, RuntimeError, KeyError) as exc:
            last = exc
    raise last


def replay_cases(contract, dialogues, vocabulary, max_cases=12):
    templates = _templates(contract)
    if len(templates) < 2:
        return []
    required = {p.name for p in contract.parameters if p.required}
    declared = {p.name for p in contract.parameters}
    cases = []
    for dialogue in dialogues:
        trace = normalize(dialogue)
        for offset in range(len(trace.steps)-len(templates)+1):
            steps = trace.steps[offset:offset+len(templates)]
            bindings = {}
            valid = True
            for step_, (name, parameters) in zip(steps, templates):
                if step_.action_name != name or len(step_.slot_values) != len(parameters):
                    valid = False; break
                for parameter, value in zip(parameters, step_.slot_values):
                    if parameter not in declared or (parameter in bindings and bindings[parameter] != value):
                        valid = False; break
                    bindings[parameter] = value
            if valid and required <= bindings.keys():
                cases.append({'conversation_id': trace.conversation_id, 'bindings': bindings,
                    'available_actions': vocabulary, 'actions': [
                        {'name': s.action_name, 'slots': s.slot_values, 'observation': s.observation}
                        for s in steps]})
                break
        if len(cases) >= max_cases:
            break
    return cases


def mine(dialogues, directory, *, model, workers=4, min_support=2, progress=None):
    directory = Path(directory)
    split_input = [{'convo_id': d['dialogue_id'], 'dialogue': d} for d in dialogues]
    induction, heldout = split_induction_and_verification(split_input, fraction=.2)
    induction = [d['dialogue'] for d in induction]
    heldout = [d['dialogue'] for d in heldout]
    vocabulary = {}
    for d in dialogues:
        for i in d['target_indices']['policy_turn_indices']:
            policy = gold_policy(d, i)
            vocabulary[label(policy)] = [list(h) for h in heads(policy)]
    def discover(d):
        path = directory/'discovery'/f"{hashlib.sha256(d['dialogue_id'].encode()).hexdigest()}.json"
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        trace = normalize(d)
        annotations, raw = stage(annotate_trace_semantics, trace, chat, model=model)
        operations, rejected, extracted = stage(extract_trace_operations, trace, annotations, chat, model=model)
        artifact = {'conversation_id': trace.conversation_id, 'normalized_trace': trace.to_dict(),
            'semantic_raw': raw, 'semantic_annotations': [asdict(a) for a in annotations],
            'operation_raw': extracted, 'operations': [o.to_dict() for o in operations], 'rejected': rejected}
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, artifact)
        return artifact
    traces, discovery_errors = [], []
    for result in local_cases(discover, induction, workers=workers, directory=directory,
                              phase='skill_disco_discovery', progress=progress):
        if result['error']:
            discovery_errors.append(result['error'])
        else:
            traces.append(require_value(result))
    operations = [semantic_operation_from_dict(o) for t in traces for o in t['operations']]
    by_id = {o.operation_id: o for o in operations}
    groups = []
    group_jobs = list(enumerate([operations[i:i+20] for i in range(0, len(operations), 20)]))
    for result in local_cases(lambda job: stage(group_operation_batch, job[1], chat,
            batch_index=job[0], model=model), group_jobs, workers=workers,
            directory=directory, phase='skill_disco_grouping'):
        groups.extend(require_value(result)[0])
    set_phase('skill_disco_consolidation')
    clusters, consolidation_raw = (stage(consolidate_groups, groups, by_id,
        len(traces), chat, model=model) if groups else ([], ''))
    eligible = [c for c in clusters if len(c.supporting_conversations) >= min_support]
    def compile_cluster(cluster):
        examples = [by_id[i] for i in cluster.operation_ids]
        contract, raw = stage(specify_skill_contract, cluster, examples, chat, model=model)
        cases = replay_cases(contract, heldout, sorted(vocabulary))
        if not cases:
            return {'status': 'discarded', 'contract': contract.to_dict(),
                    'implementation': None, 'reason': 'no matching TRAIN-heldout replay cases'}
        outcome = synthesize_and_verify(contract, examples, chat,
            lambda source, c: verify_abcd_replay(source, c, cases),
            environment_note='SGD recorded TRAIN policy replay. env.step(policy_label, '
                '[api_method_json, api_parameters_json, dialogue_acts_json]) returns '
                '(observation, available_actions). Copy each parameter string unchanged; '
                'do not parse JSON. Return success/observation/available_actions/process_trace.',
            model=model, max_attempts=3)
        outcome['verification_cases'] = len(cases)
        outcome['specification_raw'] = raw
        return outcome
    compiled = [require_value(r) for r in local_cases(compile_cluster, eligible,
        workers=workers, directory=directory, phase='skill_disco_compile_verify')]
    verified = [skill_contract_from_dict(r['contract']) for r in compiled if r['status']=='verified']
    artifact = {'method': 'skill_disco_sgd_five_stage_shared_binder_v1',
        'split': {'induction_ids': [d['dialogue_id'] for d in induction],
                  'verification_ids': [d['dialogue_id'] for d in heldout]},
        'stages': ['semantic_abstraction', 'operation_extraction', 'consolidation',
                   'skill_specification', 'synthesis_and_train_heldout_replay'],
        'traces': traces, 'discovery_errors': discovery_errors,
        'groups': [g.to_dict() for g in groups], 'clusters': [c.to_dict() for c in clusters],
        'consolidation_raw': consolidation_raw, 'contracts': [r['contract'] for r in compiled],
        'compiled_skills': compiled, 'verified_contracts': [c.to_dict() for c in verified],
        'policy_vocabulary': vocabulary, 'skill_library': render_skill_library(verified)}
    make_verified_names_unique(artifact)
    return artifact


def predict(dialogue, index, *, model, ontology, artifact, previous=()):
    artifact = applicable_artifact(artifact, dialogue['services'])
    library = CompiledSkillLibrary(deepcopy(artifact))
    specs = library.tool_specs()
    selected = None
    diagnostic = {}
    if specs:
        prompt = (TASK+'\nChoose a verified compiled skill only if its arguments can be '
            'grounded from the visible prefix. Otherwise return null. Return JSON '
            '{"skill_name":null|string,"arguments":{}}.\nSkills:\n'
            +json.dumps(specs, ensure_ascii=False)+input_text(dialogue, index, ontology))
        def validate(p):
            name, arguments = p.get('skill_name'), p.get('arguments', {})
            if name is None:
                return None
            if name not in {s['name'] for s in specs} or not isinstance(arguments, dict):
                raise ValueError('Choose a known skill and an argument object, or null')
            return {'name': name, 'arguments': arguments}
        selected, diagnostic = request_stage_json(prompt, model, 'sgd_skill_disco_selection',
            validate, instruction='Choose a compiled skill, not a raw policy. Return only the requested JSON.')
    projected = None
    if selected:
        item = next(r for r in artifact['compiled_skills']
                    if r['status']=='verified' and r['contract']['skill_name']==selected['name'])
        previous_actions = [{'name': label(r['prediction']), 'slots': slots(r['prediction']),
            'observation': dialogue['turns'][r['turn_index']].get('utterance', '')}
            for r in previous if r['turn_index'] < index and not r['error']]
        payload = {'source': item['implementation'], 'skill_name': selected['name'],
                   'arguments': selected['arguments'], 'previous': previous_actions,
                   'vocabulary': artifact['policy_vocabulary']}
        try:
            worker = Path(__file__).with_name('sgd_prefix_worker.py')
            proc = subprocess.run([sys.executable, '-I', str(worker)],
                input=json.dumps(payload), text=True, encoding='utf-8', capture_output=True,
                timeout=6, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            projected = json.loads(proc.stdout).get('projection') if proc.returncode == 0 else None
            if projected is not None:
                projected['heads'] = legal_projection_heads(projected['heads'], dialogue['services'])
        except (ValueError, subprocess.TimeoutExpired):
            projected = None
    guidance = artifact['skill_library']
    if projected:
        guidance += '\nCompiled skill proposed payload (advisory; re-ground using current visible inputs):\n'+json.dumps(projected)
    prediction, meta = predict_policy(dialogue, index, model=model, ontology=ontology,
        skill_text=guidance, selected_action_heads=projected['heads'] if projected else None)
    meta['skill_disco'] = {'selection': selected, 'selection_diagnostic': diagnostic,
        'projection': projected, 'fallback': projected is None,
        'runtime': RUNTIME}
    return prediction, meta
