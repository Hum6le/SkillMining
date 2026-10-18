"""Shared SGD policy executor with causal logged tool history and native cards.

This is an offline, teacher-forced policy benchmark, not an interactive SGD
simulator. Only an exact predicted current CALL can unlock its logged result.
All methods see the same prior logged API events, schema and TRAIN format
statistics. Current/future response text and act annotations are never inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from eval_tod.sgd_adapter import (canonical_acts, canonical_call, gold_policy,
    visible_context)
from eval_tod.sgd_skill_agent import _family, policy_heads, policy_token, _json_object
from eval_tod.sgd_standard_skill_agent import SGDStandardSkillAgent, normalize_action_heads
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_llm import selected_payload_heads, sgd_chat_with_retry
from eval_tod.multiwoz_matched_executor import request_json
from eval_tod.multiwoz_aligned import AgentReferenceSession, RuntimeLimits, atomic_json
from eval_tod.sgd_slot_diagnostics import payload_diff, compact_api_sources, select_train_exemplars

PROTOCOL = 'sgd_all_policy_shared_heads_payload_logged_history_v3'
TASK = """Predict one SGD system policy: optional CALL(service, method, parameters)
plus an UNORDERED collection of dialogue acts. Heads are (intent, actual service
ID); one head can have several named slots. First select heads, then bind only
their payloads. API method is part of the CALL payload in this benchmark.
Use prior utterances, the shared prior logged API observations, service schema
and TRAIN-derived resources. Current/future response text and gold acts are hidden.
Prior API history is a teacher-forced logged prefix; it is not a predicted rollout.
Current API results are revealed only after an exactly matching predicted CALL.
Select one contextually appropriate entity; do not emit every database row or
every schema field. Missing, extra and duplicate acts affect the strict score.
Follow TRAIN value-presence and formatting conventions; absent value, empty
string and the literal string 'null' are distinct. Do not manufacture unknown
values or copy entity values from training examples. Keep alternative supported
slot patterns conditional on context, not a union checklist.
"""
LEARNING = TASK + """
Slot policy and reference bodies are flexible natural-language procedures.
Use native slot_policy upserts on exact graph node IDs/labels. Explain applicable
slots, multiplicity, value source and serialization, including conditional cases.
Decide whether extra TRAIN examples or exceptions belong in reference. The agent
may search/read/drop reference before and after head selection. No validation or
test labels may enter resources. Unsupported graph topology remains reference.
TRAIN slot_diff facts and raw paired act JSON are authoritative over a report's
prose summary. Distinguish the slot identifier from its optional value field.
Absent value, value="", JSON null and the string "null" are NOT equivalent.
Learn the observed representation explicitly; do not infer that every named
slot requires a value. Keep dialogue-act formatting separate from CALL parameter
formatting. Use only methods in the provided legal service schema.
"""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prior_api_history(dialogue, turn_index):
    """Observed prefix only. Retain the latest result for each service/method."""
    latest = {}
    for index, turn in enumerate(dialogue['turns'][:turn_index]):
        for service, call in (turn.get('service_call') or {}).items():
            if service not in dialogue['services']:
                continue
            latest[service, call['method']] = {
                'turn_index': index, 'call': canonical_call({'service': service, **call}),
                'result': deepcopy((turn.get('db_results') or {}).get(service))}
    return sorted(latest.values(), key=lambda row: row['turn_index'])


def training_contract(dialogues):
    """Format statistics only: never a union of mandatory output slots."""
    presence, formats = defaultdict(Counter), defaultdict(Counter)
    for dialogue in dialogues:
        for index in dialogue['target_indices']['policy_turn_indices']:
            for act in gold_policy(dialogue, index)['acts']:
                presence[act['intent']]['with_value' if 'value' in act else 'without_value'] += 1
                if 'value' in act:
                    value = act['value']
                    if re.fullmatch(r'\$[\d,]+(?:\.\d+)?', value):
                        kind = 'currency_dollar_and_grouping' if ',' in value else 'currency_dollar'
                    elif re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                        kind = 'ISO_date'
                    elif value in {'True', 'False'}:
                        kind = 'capitalized_boolean'
                    else:
                        continue
                    formats[act['slot']][kind] += 1
    return {'source': 'selected_TRAIN_policy_labels_only',
            'value_presence_counts': {k: dict(v) for k, v in sorted(presence.items())},
            'act_value_format_counts': {k: dict(v) for k, v in sorted(formats.items())},
            'note': 'Observed conventions, not rules to emit every slot. API parameter and act formats may differ.'}


def schema_for_dialogue(ontology, services):
    return {service: deepcopy(ontology['domains'][service]) for service in services}


def input_text(dialogue, index, ontology):
    return ('\nAvailable complete service schema:\n'
        + json.dumps(schema_for_dialogue(ontology, dialogue['services']), ensure_ascii=False)
        + '\nTRAIN serialization statistics:\n'
        + json.dumps(ontology.get('_sgd_training_contract', {}), ensure_ascii=False)
        + '\nPrior logged API observations (prefix only):\n'
        + json.dumps(prior_api_history(dialogue, index), ensure_ascii=False)
        + '\nDialogue so far:\n' + visible_context(dialogue, index)[-12000:])


def request_stage_json(prompt, model, tag, validator, *, instruction, attempts=3):
    """Keep the stage's output contract above advisory method knowledge.

    Resource prose can describe business actions, but cannot replace this
    request's tool-control schema. Invalid outputs remain explicit failures;
    never translate a business CALL into an invented reference finish action.
    """
    failures = []
    messages = [{'role': 'system', 'content': instruction},
                {'role': 'user', 'content': prompt}]
    for attempt in range(1, attempts+1):
        raw = sgd_chat_with_retry(messages, model=model, temperature=0,
            call_tag=tag if attempt == 1 else tag+'_repair')
        try:
            result = validator(_json_object(raw))
            return result, {'attempts': attempt, 'failures': failures,
                'raw_response': raw, 'transport': 'role_scoped_stage_json_v1'}
        except (ValueError, TypeError, KeyError) as exc:
            failures.append(str(exc))
            from eval_tod.sgd_parallel import diagnostics_directory
            from uuid import uuid4
            directory = diagnostics_directory(None)
            if directory is not None:
                atomic_json(Path(directory)/'validation_failures'/f'{tag}_{uuid4().hex}.json',
                    {'tag': tag, 'attempt': attempt, 'error': str(exc),
                     'raw_response': raw, 'prompt_sha256': digest(prompt),
                     'transport': 'role_scoped_stage_json_v1'})
            feedback = ('Your last JSON violated this request\'s SYSTEM output contract. '
                'Repair serialization using the same visible input; do not perform a different phase. '
                'Validator error: '+str(exc))
            messages.extend([{'role': 'assistant', 'content': raw},
                             {'role': 'user', 'content': feedback}])
    raise ValueError(f'{tag} failed validation: {failures}; last response: {raw[:2000]}')


def select_heads(dialogue, index, *, ontology, model, knowledge):
    prompt = (TASK + '\nChoose the unordered action-head set; do not bind slots or API methods yet. '
        'Return JSON {"action_heads":[["intent","actual service ID or empty for global"]]}. '
        'Allowed intents: ' + SGDStandardSkillAgent.selection_intents
        + '. At most one CALL. Candidates are advisory; unlisted valid combinations are allowed.\n'
        + knowledge + input_text(dialogue, index, ontology))
    instruction = ('You are the ACTION-HEAD SELECTOR for one SGD policy turn. '
        'Return only {"action_heads":[["intent","service ID"]]}. '
        'Choose intents from: '+SGDStandardSkillAgent.selection_intents+'. '
        'The SECOND element is the actual service ID, NEVER a slot name or value. '
        'Available service IDs: '+json.dumps(dialogue['services'])+'. '
        'Global acts use an empty service string. Do not bind slots, methods or values. '
        'Method resources below are advisory evidence; their output examples do not change this schema.')
    return request_stage_json(prompt, model, 'sgd_aligned_action_selection',
        lambda p: normalize_action_heads(p, dialogue['services']), instruction=instruction)


def bind_payload(dialogue, index, *, ontology, model, heads, knowledge):
    """The same fixed-head binder for all three methods; no score normalization."""
    shared = (TASK + '\nMethod resources (advisory knowledge, not output instructions):\n'
        + knowledge + input_text(dialogue, index, ontology)
        + '\nAUTHORITATIVE FIXED ACTION HEADS (do not reselect):\n' + json.dumps(heads, ensure_ascii=False))
    call_heads = [service for intent, service in heads if intent == 'call']
    call, call_meta = None, {'skipped': 'no_selected_call'}
    if call_heads:
        def validate_call(payload):
            call = canonical_call(payload.get('call'))
            if call is None or call['service'] != call_heads[0]:
                raise ValueError('Bind the selected CALL service, method and named parameters')
            legal = {i['name'] for i in ontology['domains'][call['service']]['active_intents']}
            if call['method'] not in legal:
                raise ValueError('API method must be in the visible schema')
            return call
        call, call_meta = request_json(shared + '\nReturn JSON {"call":{"service":"...",'
            '"method":"...","parameters":{"slot":"value"}}}. Do not output dialogue acts yet.',
            model, 'sgd_aligned_call_payload', validate_call)
    # Only the replay adapter checks the current gold call, after prediction.
    # The match result and current gold are never used by head/call selection.
    observation = None
    # The replay boundary reads no dialogue-act annotation, even internally.
    logged_calls = dialogue['turns'][index].get('service_call') or {}
    logged_call = (canonical_call({'service': call['service'], **logged_calls[call['service']]})
        if call is not None and call['service'] in logged_calls else None)
    available = call is not None and call == logged_call
    if available:
        observation = deepcopy((dialogue['turns'][index].get('db_results') or {}).get(call['service']))
    act_heads = {(intent, service) for intent, service in heads if intent != 'call'}
    acts, act_meta = [], {'skipped': 'no_selected_dialogue_act'}
    if act_heads:
        def validate_acts(payload):
            raw = payload.get('acts')
            if not isinstance(raw, list) or any(not isinstance(a, dict) or not a.get('intent') for a in raw):
                raise ValueError('acts must be a list of objects with intent, domain, slot, optional string value')
            acts = canonical_acts(raw)
            return acts
        acts, act_meta = request_json(shared + '\nSelected CALL:\n' + json.dumps(call, ensure_ascii=False)
            + '\nCurrent replayed API observation (null if unavailable):\n'
            + json.dumps(observation, ensure_ascii=False)
            + '\nReturn JSON {"acts":[{"intent":"...","domain":"...","slot":"...",'
            '"value":"only if appropriate"}]}. Bind the fixed dialogue-act heads; do not reselect them.',
            model, 'sgd_aligned_acts_payload', validate_acts)
        outside = [a for a in acts if (a['intent'], a['domain']) not in act_heads]
        acts = [a for a in acts if (a['intent'], a['domain']) in act_heads]
        observed = {(a['intent'], a['domain']) for a in acts}
        unbound = sorted(act_heads - observed)
        # The executor owns the already-selected heads. An omitted binding is
        # an empty payload, never a second action decision. It remains strictly
        # wrong when a named slot/value was required. No gold is consulted.
        acts.extend({'intent': intent, 'domain': service, 'slot': ''} for intent, service in unbound)
        acts = canonical_acts(acts)
        act_meta.update(discarded_outside_head_acts=outside, unbound_selected_heads=unbound)
    return {'call': call, 'acts': acts}, {'executor_protocol': PROTOCOL,
        'call_payload': call_meta, 'act_payload': act_meta,
        'observation_visible': available, 'api_observation': observation,
        'prior_api_history_sha256': digest(prior_api_history(dialogue, index)),
        'selected_action_heads': heads}


def predict_policy(dialogue, turn_index, *, model, ontology, frequency_graph=None,
                   workflow_text='', exemplar_text='', skill_text='', reference_text='',
                   selected_action_heads=None):
    knowledge = '\n\n'.join(text for text in [workflow_text, exemplar_text, skill_text, reference_text] if text)
    selection = None
    heads = selected_action_heads
    if heads is None:
        heads, selection = select_heads(dialogue, turn_index, ontology=ontology, model=model, knowledge=knowledge)
    with selected_payload_heads(heads):
        prediction, diagnostic = bind_payload(dialogue, turn_index, ontology=ontology,
            model=model, heads=heads, knowledge=knowledge)
    diagnostic['action_selection'] = selection
    return prediction, diagnostic


class SGDReferenceSession(AgentReferenceSession):
    """Reuse the existing read/search/drop implementation with the SGD task."""
    def catalog_text(self, stage):
        return ('\nReference catalog (IDs/titles only; content requires search/read):\n'
            + json.dumps([{'section_id': row['section_id'], 'title': row['title']}
                          for row in self.sections.values()], ensure_ascii=False))

    def validate(self, payload):
        payload = deepcopy(payload)
        if 'reference_action' in payload:
            if 'action' in payload and payload['action'] != payload['reference_action']:
                raise ValueError('Use one reference_action; conflicting action fields are invalid')
            payload['action'] = payload.pop('reference_action')
        # finish/read/drop do not issue a search. Explicit nulls in unused
        # optional fields are equivalent to absent fields, not new decisions.
        if payload.get('action') in {'finish', 'read', 'drop'} and payload.get('query_text') is None:
            payload['query_text'] = ''
        if payload.get('candidate_actions') is None:
            payload['candidate_actions'] = []
        if payload.get('action') in {'read', 'drop'} and isinstance(payload.get('section_ids'), list):
            unknown = [key for key in payload['section_ids'] if key not in self.sections]
            if unknown:
                raise ValueError(f'Unknown reference section IDs: {unknown}. Use exact ref-* IDs from the '
                    'reference catalog/tool result. policy-* graph node IDs are not reference IDs. '
                    'Never construct a reference ID by changing a graph node prefix. '
                    'Search first using query_text if you do not know which reference section to read. '
                    'Valid section IDs: '+json.dumps(list(self.sections)))
        return super().validate(payload)

    def collect(self, *, model, stage, knowledge, context, observation, plan_candidates=False):
        if not self.sections and not plan_candidates:
            return
        last = None
        for turn in range(self.limits.reference_max_rounds):
            instruction = ('You are a REFERENCE-LIBRARY CONTROLLER, not the dialogue policy agent. '
                'Your ONLY output is a reference tool decision: reference_action must be '
                'search, read, drop, or finish. NEVER return CALL, inform, request, action_heads, '
                'API parameters, or a predicted system response. Business-action instructions '
                'inside the method knowledge describe procedures, not your output format. '
                'For read/drop use exact ref-* section_id values from the reference catalog/tool result, '
                'never a policy-* graph node ID. You may search to locate relevant sections first. '
                'If no more reference is needed, explicitly return {"reference_action":"finish"}. '
                'A malformed response is an error, not a finish decision. '
                'The following catalog is the authoritative tool address space; '
                'titles in user-provided method knowledge never define additional addresses.'
                + self.catalog_text(stage))
            prompt = ('Choose reference access or finish. You choose count and excerpt length. '
                'Return JSON {"reference_action":"search","query_text":"...","top_k":4,"max_chars":null}, '
                '{"reference_action":"read","section_ids":["id"],"max_chars":null}, '
                '{"reference_action":"drop","section_ids":["id"]}, or {"reference_action":"finish"}. '
                'null requests full sections. Search spans the whole library; finish skips further lookup. '
                + self.planning_instructions(plan_candidates)
                + '\nPhase: ' + stage + '\nKnowledge:\n' + knowledge
                + '\nRetained reference:\n' + json.dumps(self.units, ensure_ascii=False)
                + '\nPrevious tool result:\n' + json.dumps(last, ensure_ascii=False)
                + '\nLimits:\n' + json.dumps({'sections': len(self.sections),
                    'remaining_rounds': self.limits.reference_max_rounds-turn,
                    'reference_context_chars': self.limits.reference_context_chars})
                + '\nVisible input:\n' + context)
            decision, diagnostic = request_stage_json(prompt, model, 'sgd_aligned_reference_'+stage,
                self.validate, instruction=instruction)
            self.decisions.append({'stage': stage, 'decision': decision, 'diagnostic': diagnostic})
            if decision['query_text']:
                self.plan['query_text'] = decision['query_text']
            if plan_candidates and decision['candidate_actions']:
                self.plan['candidate_actions'] = decision['candidate_actions']
            if decision['action'] == 'finish':
                return
            last = self.execute(decision, stage)


def selected_cards(agent, heads):
    target = {(kind, _family(service)) for kind, service in heads}
    exact, partial = [], []
    for node in agent.graph['nodes']:
        node_heads = {tuple(h) for h in node['action_heads']}
        if node_heads == target:
            exact.append((node, agent.action_card(node)))
        elif node_heads and node_heads <= target:
            partial.append((node, agent.action_card(node)))
    chosen = exact or partial
    return [dict(card, node_id=node['id']) for node, card in chosen], bool(exact)


class SGDAlignedSkillAgent(SGDStandardSkillAgent):
    """Keep native standard TRAIN scheduling, diagnosis and slot-policy upserts."""
    # Keep the resource schema version. Runtime contract is saved separately;
    # reusing a legacy checkpoint must be explicitly labeled as a diagnostic.
    task_prompt = TASK
    diagnosis_prompt = SGDStandardSkillAgent.diagnosis_prompt + '\n' + LEARNING
    reflection_prompt = SGDStandardSkillAgent.reflection_prompt + '\n' + LEARNING

    def runtime_contract(self):
        return {'executor_protocol': PROTOCOL, 'input': 'prior_utterances_plus_logged_API_prefix',
            'runtime_gold_policy_history': False, 'current_api': 'exact_call_replay',
            'selected_card_lookup': 'after_final_heads',
            'ontology_sha256': digest({k: v for k, v in self.ontology.items() if not k.startswith('_sgd_')}),
            'training_contract_sha256': digest(self.ontology.get('_sgd_training_contract', {}))}

    def resource_hash(self):
        return digest([super().resource_hash(), self.runtime_contract()])

    def load(self, directory):
        marker = Path(directory) / 'aligned_runtime_contract.json'
        if marker.exists() and json.loads(marker.read_text(encoding='utf-8')) != self.runtime_contract():
            raise ValueError('Aligned TRAIN/evaluation runtime contract mismatch')
        super().load(Path(directory))

    def _project_record(self, record):
        # Full raw model/reference responses remain in saved diagnostics, not
        # repeated inside every diagnosis/reflection prompt.
        evidence = deepcopy(record['sgd_evidence'])
        diagnostic = evidence.get('runtime_diagnostic', {})
        evidence['runtime_diagnostic'] = {key: diagnostic.get(key) for key in [
            'executor_protocol', 'source_node', 'selected_action_heads',
            'selected_card_ids', 'selected_card_exact', 'observation_visible',
            'prior_api_history_sha256']}
        evidence['runtime_diagnostic']['reference_sections'] = [
            {'section_id': unit['section_id'], 'title': unit.get('title', '')}
            for unit in diagnostic.get('retrieved_reference', [])]
        selected = set(diagnostic.get('selected_card_ids', []))
        evidence['selected_action_cards'] = [self.action_card(node)
            for node in self.graph['nodes'] if node['id'] in selected]
        evidence['slot_diff'] = payload_diff(evidence['gold_policy'], evidence['predicted_policy'],
                                             action_correct=evidence['action_correct'])
        evidence['visible_api_sources'] = deepcopy(diagnostic.get('visible_api_sources', []))
        return evidence

    def training_feedback_context(self, labels):
        relevant = [node for node in self.graph['nodes'] if node['label'] in labels or node['id'] in labels]
        return {'scope': 'TRAIN supervision only; never a runtime prediction input',
            'field_contract': 'Absent value, empty string, JSON null and literal null are distinct. Slot identifiers need not carry values. Preserve conditional alternatives, not the union of all slots.',
            'legal_service_schema': {s: deepcopy(v) for s, v in self.ontology['domains'].items() if _family(s) == self.family},
            'TRAIN_serialization_statistics': deepcopy(self.ontology.get('_sgd_training_contract', {})),
            'current_action_cards': [dict(self.action_card(node), node_id=node['id']) for node in relevant]}

    def diagnosis_options(self, records):
        labels = {record['sgd_evidence']['gold_target'] for record in records}
        from skill_mining.online_refinement import _resource_lookup_sections
        sections = _resource_lookup_sections('reference', self.base_reference, ' '.join(sorted(labels)), 2)
        return {'trajectory_exemplar_selector': select_train_exemplars,
                'available_resources_override': sections + [{'resource': 'TRAIN_field_evidence_contract',
                    'content': self.training_feedback_context(labels)}]}

    def reflection_options(self, reports):
        labels = {row.get('gold_target') for report in reports for row in report.get('trajectory_exemplars', [])}
        for report in reports:
            for update in report.get('candidate_updates', []):
                labels.add(update.get('action'))
        return {'include_trajectory_exemplars': True, 'evidence_context': self.training_feedback_context(labels)}

    def runtime_skill_text(self):
        if self.working_skill:
            # Standard reflection may edit main skill prose as well as native
            # slot resources. Preserve those accepted text edits at runtime.
            # This is the backbone artifact, not stored model/ReAct traces.
            return self.working_skill
        rows = ['# SGD backbone skill', TASK, '## Action bundles']
        rows.extend(n['label'] + ': ' + json.dumps(n['action_heads']) for n in self.graph.get('nodes', []))
        rows += ['## Backbone and accepted guards']
        for edge in self.state.get('edges', {}).values():
            if edge.get('visibility') == 'skill':
                rows.append(f"{edge['source_action']} -> {edge['target_action']}: {edge.get('guard', '')}")
        return '\n'.join(rows)

    def reference_text(self):
        rows = [self.base_reference]
        for key, edge in self.state.get('edges', {}).items():
            if edge.get('visibility') != 'skill':
                rows.append(f"## {key}\n{edge['source_action']} -> {edge['target_action']}: {edge.get('guard', '')}")
        rows.extend(f"## Note {i}\n{note.get('content', '')}" for i, note in enumerate(self.state.get('reference_notes', [])))
        return '\n\n'.join(rows)

    def predict(self, dialogue, turn_index):
        # No previous_node(): routing candidates come only from visible inputs.
        context = input_text(dialogue, turn_index, self.ontology)
        skill = self.runtime_skill_text()
        limits = RuntimeLimits(candidate_limit=self.candidate_limit)
        session = SGDReferenceSession(self.reference_text(), limits)
        session.collect(model=self.model, stage='action', knowledge=skill,
                        context=context, observation=None, plan_candidates=True)
        aliases = {alias: n for n in self.graph['nodes'] for alias in (n['id'], n['label'])}
        tokens = set(re.findall(r'\w+', (session.plan['query_text'] or context).lower()))
        ranked = sorted(self.graph['nodes'], key=lambda n: (
            -len(tokens & set(re.findall(r'\w+', self.action_card(n)['rule'].lower()))), n['id']))
        ids = list(dict.fromkeys([aliases[a]['id'] for a in session.plan['candidate_actions'] if a in aliases]
                                 + [n['id'] for n in ranked]))[:self.candidate_limit]
        by_id = {n['id']: n for n in self.graph['nodes']}
        candidate_cards = [self.action_card(by_id[k]) for k in ids]
        knowledge = skill + '\nCandidate cards:\n' + json.dumps(candidate_cards, ensure_ascii=False)
        knowledge += '\nAgent-selected reference:\n' + json.dumps(session.units, ensure_ascii=False)
        heads, selection = select_heads(dialogue, turn_index, ontology=self.ontology,
                                        model=self.model, knowledge=knowledge)
        cards, exact = selected_cards(self, heads)
        payload = 'Complete selected Action Cards:\n' + json.dumps(cards, ensure_ascii=False)
        if not exact:
            payload += '\nUnseen bundle: partial cards apply only to their listed heads; do not substitute a different bundle.'
        session.collect(model=self.model, stage='payload', knowledge=payload+'\nFixed heads:\n'+json.dumps(heads),
                        context=context, observation=None)
        payload += '\nAgent-selected reference:\n' + json.dumps(session.units, ensure_ascii=False)
        with selected_payload_heads(heads):
            prediction, diagnostic = bind_payload(dialogue, turn_index, ontology=self.ontology,
                model=self.model, heads=heads, knowledge=payload)
        diagnostic.update(source_node='context_only_no_gold_policy_history', candidate_ids=ids,
            selected_card_ids=[c['node_id'] for c in cards], selected_card_exact=exact,
            resource_sha256=self.resource_hash(), action_selection=selection,
            retrieved_reference=deepcopy(session.units), reference_plan=deepcopy(session.plan),
            reference_plan_diagnostic=session.diagnostic())
        events = prior_api_history(dialogue, turn_index)
        if diagnostic.get('observation_visible'):
            events.append({'turn_index': turn_index, 'call': prediction['call'],
                           'result': diagnostic.get('api_observation'), 'source': 'current_exact_call_replay'})
        diagnostic['visible_api_sources'] = compact_api_sources(events, context)
        return prediction, diagnostic

    def mine(self, dialogues, directory, *, cases_per_node=4, compile_batch_size=4, **kwargs):
        """TraceCover + required per-node prose cards, validated before use."""
        directory = Path(directory)
        super().mine(dialogues, directory, cases_per_node=cases_per_node, compile_rules=False)
        cases, counts, shapes = defaultdict(list), Counter(), defaultdict(Counter)
        for dialogue in dialogues:
            for index in dialogue['target_indices']['policy_turn_indices']:
                gold = gold_policy(dialogue, index)
                key = f'{self.family}:{policy_token(gold)}'
                shape = json.dumps([(a['intent'], a['slot'], 'value' in a) for a in gold['acts']])
                counts[key] += 1
                shapes[key][shape] += 1
                observation = (dialogue['turns'][index].get('db_results') or {}).get(gold['call']['service']) if gold['call'] else None
                cases[key].append({'sample_id': f"{dialogue['dialogue_id']}:{index}",
                    'context': visible_context(dialogue, index)[-1800:],
                    'prior_api_history': prior_api_history(dialogue, index), 'gold': gold,
                    'api_observation_after_gold_call': observation, '_shape': shape})
        profiles = []
        for node in self.graph['nodes']:
            candidates, selected, used = cases[node['id']], [], set()
            for row in candidates:
                if row['_shape'] not in used and len(selected) < cases_per_node:
                    used.add(row['_shape']); selected.append(row)
            selected_ids = {row['sample_id'] for row in selected}
            selected.extend(row for row in candidates if row['sample_id'] not in selected_ids)
            snippets = [{k: v for k, v in row.items() if k != '_shape'} for row in selected[:cases_per_node]]
            profiles.append({'node_id': node['id'], 'heads': node['action_heads'],
                'training_turns': counts[node['id']], 'initial_snippets': snippets,
                'observed_slot_shapes': dict(shapes[node['id']])})
        atomic_json(directory / 'train_card_profiles.json', profiles)
        def compile_job(profile):
            key = profile['node_id']
            def validate(payload):
                if payload.get('node_id') != key:
                    raise ValueError(f'node_id must be exactly {key}; slot names and API methods are not resource keys')
                for field in ['action_rule', 'slot_policy']:
                    if not isinstance(payload.get(field), str) or not payload[field].strip():
                        raise ValueError(f'Each node needs nonempty {field}; generic fallback is not compilation')
                if payload['slot_policy'].strip() == 'Use named slots; ground values in prior context or visible API observation.':
                    raise ValueError('Compile a node-specific TRAIN slot policy instead of returning the uncompiled fallback')
                if not isinstance(payload.get('reference', ''), str):
                    raise ValueError('reference must be prose text')
                return payload
            prompt = (LEARNING + '\nCompile one TRAIN card as JSON {"node_id":"exact ID",'
                '"action_rule":"flexible prose","slot_policy":"flexible prose",'
                '"reference":"optional residual procedures or compact examples"}. '
                'The observed shapes describe alternatives, not mandatory slots. '
                'Learn value presence and formatting, grounding, entity choice, conditional omissions/extras. '
                'Return both required prose fields on this exact node; do not use method/slot names as keys.\n'
                + json.dumps(profile, ensure_ascii=False)
                + '\nComplete TRAIN service schema:\n'
                + json.dumps({s: self.ontology['domains'][s] for s in sorted({s for d in dialogues for s in d['services']})}, ensure_ascii=False)
                + '\nTRAIN serialization statistics:\n'
                + json.dumps(self.ontology.get('_sgd_training_contract', {}), ensure_ascii=False))
            card, diagnostic = request_json(prompt, self.model, 'sgd_aligned_offline_card', validate)
            return card, diagnostic
        outcomes = ordered_cases(compile_job, profiles, workers=self.parallel_workers,
            directory=directory, phase='offline_compile_cards')
        references, compiled = [], []
        for profile in profiles:
            for number, snippet in enumerate(profile['initial_snippets'], 1):
                # Compact supervised TRAIN evidence, never a full ReAct trace.
                shown = {k: snippet[k] for k in ['sample_id', 'context', 'gold']}
                references.append(f"## {profile['node_id']} TRAIN snippet {number}\n"
                    + 'TRAIN example only; never copy instance values as defaults.\n'
                    + json.dumps(shown, ensure_ascii=False))
        for outcome in outcomes:
            card, diagnostic = require_value(outcome)  # Never silently retain fallback policies.
            key = card['node_id']
            self.resources['action_rules'][key] = card['action_rule']
            self.resources['slot_policies'][key] = card['slot_policy']
            self.ledger.append({'phase': 'offline_compile_card', 'node_id': key, 'applied': True,
                                'diagnostic': diagnostic})
            compiled.append(key)
            if card.get('reference', '').strip():
                references.append(f"## {key}\n{card['reference']}")
        assert set(compiled) == {n['id'] for n in self.graph['nodes']}
        self.base_action_rules = '\n\n'.join(f'## {k}\n{v}' for k, v in self.resources['action_rules'].items())
        self.base_slot_policies = '\n\n'.join(f'## {k}\n{v}' for k, v in self.resources['slot_policies'].items())
        self.base_reference = '\n\n'.join(references)
        self.working_skill = self._seed_skill()
        self.save(directory)
        atomic_json(directory / 'offline_slot_audit.json', {'verified': True,
            'compiled_node_ids': compiled, 'initial_snippets_per_node': cases_per_node,
            'actual_snippets': {p['node_id']: len(p['initial_snippets']) for p in profiles},
            'nonempty_native_slot_policies': len(compiled), 'runtime_protocol': PROTOCOL})

    def save(self, directory):
        super().save(directory)
        atomic_json(Path(directory) / 'aligned_runtime_contract.json', self.runtime_contract())
