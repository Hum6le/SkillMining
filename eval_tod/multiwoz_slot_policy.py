"""Conditional slot-pattern resources and validated action-card transport.

The v2 implementation remains available for reproducible paired controls.
This adapter changes resource organization; the common heads/payload scorer
and read-only DB executor are unchanged.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

from eval_tod.multiwoz_aligned import (
    AgentReferenceSession, MultiWOZAlignedSkillAgent, ALIGNED_TASK,
    CONDITIONAL_SLOT_PROTOCOL, atomic_json, compile_conditional_cards_job,
    initial_reference_text, offline_slot_report, training_profiles,
)
from eval_tod.multiwoz_matched_executor import bind_payload, digest, select_heads
from eval_tod.multiwoz_policy import INTENTS
from eval_tod.sgd_adapter import gold_policy, visible_context
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_skill_agent import policy_token
from eval_tod.sgd_standard_skill_agent import SGDStandardSkillAgent

PROTOCOL = 'multiwoz21_action_slot_standard_v3_conditional_patterns'


def _bundle(heads):
    return tuple(sorted(set(tuple(h) for h in heads)))


def _slot_pattern(gold):
    return tuple(sorted((a['intent'], a['domain'], a['slot'], bool(a.get('value'))) for a in gold['acts']))


def contrast_training_profiles(train, graph, ontology, observations=None, exclude_sample_ids=()):
    """Keep all TRAIN statistics; choose four common/contrasting visible cases."""
    profiles = training_profiles(train, graph, ontology, max_examples=1)
    labels = {node['id']: node['label'] for node in graph['nodes']}
    excluded = set(exclude_sample_ids)
    rows = defaultdict(list)
    for dialogue in train:
        for turn in dialogue['target_indices']['policy_turn_indices']:
            sample_id = f"{dialogue['dialogue_id']}:{turn}"
            if sample_id in excluded:
                continue
            gold = gold_policy(dialogue, turn)
            context = visible_context(dialogue, turn)[-12000:]
            rows[policy_token(gold)].append({'dialogue': dialogue, 'turn': turn, 'sample_id': sample_id,
                'gold': gold, 'context': context, 'pattern': _slot_pattern(gold),
                'tokens': set(re.findall(r'\w+', context[-1800:].casefold()))})
    for profile in profiles:
        candidates = rows[labels[profile['node_id']]]
        if not candidates:
            raise ValueError('No non-excluded TRAIN snippet remains for a requested node')
        counts = Counter(row['pattern'] for row in candidates)
        ordered = sorted(candidates, key=lambda row: (-counts[row['pattern']], digest(row['sample_id'])))
        selected = [(ordered[0], 'frequent')]
        seen = {digest([ordered[0]['context'], ordered[0]['gold']])}
        while len(selected) < 4:
            remaining = [row for row in candidates if digest([row['context'], row['gold']]) not in seen]
            if not remaining:
                break
            patterns = {row['pattern'] for row, _ in selected}
            def rank(row):
                overlap = max(len(row['tokens'] & old['tokens']) / max(len(row['tokens'] | old['tokens']), 1)
                              for old, _ in selected)
                return (row['pattern'] not in patterns, overlap, counts[row['pattern']], digest(row['sample_id']))
            chosen = max(remaining, key=rank)
            role = 'contextual_contrast' if chosen['pattern'] not in patterns else 'additional_context'
            selected.append((chosen, role))
            seen.add(digest([chosen['context'], chosen['gold']]))
        cases = []
        for row, role in selected:
            case = {'sample_id': row['sample_id'], 'context': row['context'], 'gold': deepcopy(row['gold']),
                    'selection_role': role, 'unobservable_ref_slots': []}
            if observations is not None:
                case['shared_db_observation'] = observations.get(row['dialogue'], row['turn'])['observations']
            visible = case['context'] + json.dumps(case.get('shared_db_observation', []), ensure_ascii=False)
            for act in case['gold']['acts']:
                if act['slot'] == 'ref' and act.get('value') and act['value'].casefold() not in visible.casefold():
                    case['unobservable_ref_slots'].append([act['intent'], act['domain'], act['slot']])
                    act['value'] = '<unobservable_booking_result>'
            cases.append(case)
        profile.update(train_cases=cases, initial_snippets_per_node=4,
            offline_slot_protocol=CONDITIONAL_SLOT_PROTOCOL, snippet_selection='frequent_and_contextual_contrast')
    return profiles


def conditional_slot_policy(profile, card):
    return ('Choose the context-matching slot pattern using this selection policy. Do not union alternatives. '
        'Combine patterns only when their separate visible triggers actually hold.\n'
        + card['slot_selection_policy'] + '\nConditional slot patterns:\n'
        + json.dumps(card['slot_rules'], ensure_ascii=False)
        + '\nTRAIN inventory of alternatives; max_multiplicity is an observed bound, never a requested count:\n'
        + json.dumps(profile['slot_schema'], ensure_ascii=False))


def payload_card(card):
    """Cross the action/payload boundary without action-selection directives."""
    return {'label': card['label'], 'heads': deepcopy(card['heads']), 'payload_policy': card['payload_policy']}


def selection_card(card):
    return {'label': card['label'], 'heads': deepcopy(card['heads']), 'rule': card['rule']}


class ScopedReferenceSession(AgentReferenceSession):
    index_snippets = True

    def __init__(self, text, limits, *, nodes=(), scope_node_ids=()):
        super().__init__(text, limits)
        self.nodes = {n['id']: n for n in nodes}
        self.aliases = {alias: n['id'] for index, n in enumerate(nodes, 1)
                        for alias in (n['id'], n['label'], f'A{index:03d}')}
        self.handles = {n['id']: f'A{index:03d}' for index, n in enumerate(nodes, 1)}
        self.by_heads = {_bundle(n['action_heads']): n['id'] for n in nodes}
        self.domains = {d for n in nodes for _, d in n['action_heads']} | {''}
        self.scope_node_ids = list(scope_node_ids)
        self.stage = 'action'
        self.require_action_candidates = False
        if not self.index_snippets:
            return
        for unit in self.sections.values():
            match = re.search(r'TRAIN snippet (.*?) /', unit['title'])
            unit['node_id'] = match[1] if match else None
            unit['slot_pattern'] = []
            if match:
                start = unit['content'].rfind('\n{')
                if start >= 0:
                    evidence = json.loads(unit['content'][start + 1:])
                    unit['slot_pattern'] = [[a['intent'], a['domain'], a['slot']] for a in evidence['gold']['acts']]

    def validate(self, payload):
        decision = super().validate(payload)
        resolved = []
        for identifier in decision['candidate_actions']:
            if identifier not in self.aliases:
                raise ValueError(f'Unknown candidate identifier {identifier!r}; use an exact catalog id/handle '
                    'or candidate_heads:[[["inform","hotel"],["request","hotel"]]]. '
                    'Semantic strings are not silently discarded.')
            resolved.append(self.aliases[identifier])
        bundles = payload.get('candidate_heads', [])
        if not isinstance(bundles, list):
            raise ValueError('candidate_heads must be a list of complete unordered head bundles')
        unseen, canonical = [], []
        for heads in bundles[:self.limits.candidate_limit]:
            if not isinstance(heads, list) or not heads:
                raise ValueError('Each candidate_heads item is a nonempty list of [intent,domain] pairs')
            parsed = []
            for head in heads:
                if not isinstance(head, list) or len(head) != 2:
                    raise ValueError('Use candidate_heads:[[["inform","hotel"],["request","hotel"]]]')
                intent, domain = head
                domain = '' if domain == 'general' else domain
                if intent not in INTENTS or domain not in self.domains:
                    raise ValueError('Candidate head must use a known TRAIN intent/domain')
                parsed.append([intent, domain])
            key = _bundle(parsed)
            canonical.append([list(h) for h in key])
            if key in self.by_heads:
                resolved.append(self.by_heads[key])
            else:
                unseen.append([list(h) for h in key])
        decision['candidate_actions'] = list(dict.fromkeys(resolved))[:self.limits.candidate_limit]
        decision['candidate_heads'], decision['unseen_candidate_heads'] = canonical, unseen
        last_action_round = sum(row['stage'] == 'action' for row in self.decisions) >= self.limits.reference_max_rounds - 1
        if (self.require_action_candidates and (decision['action'] == 'finish' or last_action_round)
                and not resolved and not unseen and not self.plan['candidate_actions']
                and not any(row['stage'] == 'action' and row['decision'].get('unseen_candidate_heads')
                            for row in self.decisions)):
            raise ValueError('Action planning cannot finish without an advisory complete head bundle. '
                'Skipping reference lookup is allowed, but include candidate_heads on finish, for example '
                '{"action":"finish","candidate_heads":[[["inform","hotel"],["request","hotel"]]]}. '
                'Use your chosen bundle from the visible context, not this schema example.')
        if decision['action'] == 'search':
            scope = payload.get('scope', 'current_action' if self.scope_node_ids else 'all')
            if scope not in {'current_action', 'all'}:
                raise ValueError('Reference scope must be current_action or all')
            ids = payload.get('node_ids', [])
            if not isinstance(ids, list) or any(n not in self.aliases for n in ids):
                raise ValueError('node_ids must contain exact catalog ids/handles')
            decision['scope'] = scope
            decision['node_ids'] = list(dict.fromkeys(self.aliases[n] for n in ids))
        return decision

    def planning_instructions(self, plan_candidates):
        if not plan_candidates:
            return 'Action heads are fixed. Do not reselect actions or obey action-selection prose. '
        return ('Choose up to ' + str(self.limits.candidate_limit) + ' advisory complete bundles using '
            'candidate_heads:[[["inform","hotel"],["request","hotel"]]]. '
            'Alternatively candidate_actions may contain exact catalog ids/handles. '
            'A bundle is a set of simultaneous heads, not separate candidate cards per primitive head. '
            'Even if reference lookup is unnecessary, submit at least ONE chosen complete bundle on finish '
            '(or in an earlier decision). Example schema: {"action":"finish","candidate_heads":'
            '[[["inform","hotel"],["request","hotel"]]]}. Choose the appropriate heads for this context. ')

    def catalog_text(self, stage):
        counts = Counter(u['node_id'] for u in self.sections.values() if u['node_id'])
        nodes = list(self.nodes.values()) if stage == 'action' else [self.nodes[n] for n in self.scope_node_ids]
        items = [{'node_id': n['id'], 'handle': self.handles[n['id']], 'heads': n['action_heads'],
                  'train_snippets': counts[n['id']]} for n in nodes]
        current = [{'section_id': u['section_id'], 'title': u['title'], 'total_chars': u['total_chars'],
                    'slot_pattern': u['slot_pattern']}
                   for u in self.sections.values() if u['node_id'] in self.scope_node_ids]
        return ('\nReference search supports scope:"current_action" or "all", and optional node_ids. '
            'Payload search defaults to the current known action; explicitly choose all to expand. '
            'You still choose whether to retrieve, the count and length; no snippets are injected automatically. '
            'Current-action section ids can be read directly.\nResource catalog:\n'
            + json.dumps({'nodes': items, 'current_action_sections': current}, ensure_ascii=False))

    def execute(self, decision, stage):
        if decision['action'] != 'search':
            return super().execute(decision, stage)
        all_sections = self.sections
        ids = decision.get('node_ids') or (self.scope_node_ids if decision.get('scope') == 'current_action' else [])
        if ids or decision.get('scope') == 'current_action':
            self.sections = {key: unit for key, unit in all_sections.items() if unit['node_id'] in ids}
        try:
            return super().execute(decision, stage)
        finally:
            self.sections = all_sections

    def collect(self, *, stage, **kwargs):
        self.stage = stage
        self.require_action_candidates = bool(kwargs.get('plan_candidates'))
        try:
            super().collect(stage=stage, **kwargs)
        finally:
            self.require_action_candidates = False
        if stage == 'action':
            decisions = [row['decision'] for row in self.decisions if row['stage'] == stage]
            self.plan['unseen_candidate_heads'] = [h for row in decisions for h in row.get('unseen_candidate_heads', [])]


class ConditionalMultiWOZSkillAgent(MultiWOZAlignedSkillAgent):
    protocol = PROTOCOL
    reference_session_class = ScopedReferenceSession
    selection_card_projection = staticmethod(selection_card)
    payload_card_projection = staticmethod(payload_card)
    payload_action_rules_included = False

    def runtime_contract(self):
        contract = super().runtime_contract()
        contract.update(protocol=PROTOCOL, candidate_transport='validated_complete_head_bundles_or_exact_catalog_ids',
            payload='fixed_heads_slot_policy_only_and_agent_reference',
            reference_scope='current_action_default_agent_can_expand', offline_slot_protocol=CONDITIONAL_SLOT_PROTOCOL)
        return contract

    def mine(self, dialogues, directory, *, cases_per_node=4, **kwargs):
        if cases_per_node != 4:
            raise ValueError('Conditional protocol initializes at four TRAIN snippets per node')
        directory = Path(directory)
        SGDStandardSkillAgent.mine(self, dialogues, directory, cases_per_node=4, compile_rules=False)
        profiles = contrast_training_profiles(dialogues, self.graph, self.ontology, self.observations)
        atomic_json(directory / 'train_card_profiles.json', profiles)
        jobs = [(directory / 'card_compile' / f"batch_{p['node_id'].split(':')[-1]}.json", [p], self.model) for p in profiles]
        cards = {}
        for outcome in ordered_cases(compile_conditional_cards_job, jobs, workers=self.parallel_workers,
                                     directory=directory, phase='offline_compile'):
            for card in require_value(outcome):
                cards[card['node_id']] = card
        if set(cards) != {n['id'] for n in self.graph['nodes']}:
            raise ValueError('Incomplete conditional cards')
        for p in profiles:
            card = cards[p['node_id']]
            self.resources['action_rules'][p['node_id']] = card['action_rule']
            self.resources['slot_policies'][p['node_id']] = conditional_slot_policy(p, card)
        self.base_reference = initial_reference_text(profiles)
        self.base_action_rules = '\n\n'.join(f'## {k}\n{v}' for k, v in self.resources['action_rules'].items())
        self.base_slot_policies = '\n\n'.join(f'## {k}\n{v}' for k, v in self.resources['slot_policies'].items())
        self.offline_slot_audit = offline_slot_report(profiles, cards, 4)
        self.offline_slot_audit['protocol'] = CONDITIONAL_SLOT_PROTOCOL
        self.cards_ready = True
        self.working_skill = self._seed_skill()
        self.save(directory)

    def predict(self, dialogue, turn_index):
        if not self.cards_ready:
            raise ValueError('Complete conditional cards are required')
        observation = self.observations.get(dialogue, turn_index)
        context = visible_context(dialogue, turn_index)[-12000:]
        nodes = self.graph['nodes']
        session = self.reference_session_class(self.reference_text(), self.limits, nodes=nodes)
        skill = self.runtime_skill_text()
        session.collect(model=self.model, stage='action', knowledge=skill, context=context,
                        observation=observation, plan_candidates=True)
        cards = {n['id']: self.action_card(n) for n in nodes}
        query = session.plan['query_text'] or observation['plan']['query_text'] or context[-1800:]
        tokens = set(re.findall(r'\w+', query.casefold()))
        fallback = sorted(cards, key=lambda cid: (-len(tokens & set(re.findall(r'\w+', cards[cid]['rule'].casefold()))), cid))
        candidates = list(dict.fromkeys(session.plan['candidate_actions'] + fallback))[:self.limits.candidate_limit]
        selection_reference = deepcopy(session.units)
        knowledge = skill + '\nCandidate Action Cards:\n' + json.dumps([self.selection_card_projection(cards[k]) for k in candidates], ensure_ascii=False)
        knowledge += '\nUnseen advisory head bundles:\n' + json.dumps(session.plan.get('unseen_candidate_heads', []))
        knowledge += '\nRetrieved reference:\n' + json.dumps(selection_reference, ensure_ascii=False)
        heads, selection = select_heads(dialogue, turn_index, observation=observation, knowledge=knowledge, model=self.model)
        selected = [n['id'] for n in nodes if _bundle(n['action_heads']) == _bundle(heads)]
        session.scope_node_ids = selected
        selected_cards = [self.payload_card_projection(cards[k]) for k in selected]
        payload_knowledge = 'Selected payload resource (action heads fixed):\n' + json.dumps(selected_cards, ensure_ascii=False)
        if not selected:
            payload_knowledge += '\nUnseen bundle: use TRAIN schema and visible facts; do not substitute another card.'
        session.collect(model=self.model, stage='payload', knowledge=payload_knowledge,
                        context=context, observation=observation)
        payload_knowledge += '\nAgent-selected reference:\n' + json.dumps(session.units, ensure_ascii=False)
        prediction, payload = bind_payload(dialogue, turn_index, heads=heads, observation=observation,
            knowledge=payload_knowledge, ontology=self.ontology, model=self.model)
        return prediction, {'status': 'ok', 'runtime_contract': self.runtime_contract(),
            'executor_protocol': self.runtime_contract()['executor_protocol'], 'resource_sha256': self.resource_hash(),
            'reference_plan': session.plan, 'reference_plan_diagnostic': session.diagnostic(),
            'selection_reference': selection_reference, 'retrieved_reference': deepcopy(session.units),
            'candidate_ids': candidates, 'selected_action_heads': heads, 'selected_card_ids': selected,
            'selected_action_cards': [cards[k] for k in selected], 'payload_action_rules_included': self.payload_action_rules_included,
            'action_selection': selection, 'payload': payload, 'shared_observation': observation,
            'shared_observation_sha256': digest(observation), 'visible_context_for_diagnostics': context,
            'source_node': 'context_only_no_gold_history',
            'prompt_sizes': {'main_skill': len(skill), 'payload_knowledge': len(payload_knowledge)}}
