"""One MultiWOZ runtime for offline resources, TRAIN rollout, refine and eval.

The historical pilot and frozen-resource reevaluation remain reproducible in
their original modules. New runs use this explicit protocol and fresh resources.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from uuid import uuid4

from eval_tod.multiwoz_policy import MultiWOZStandardSkillAgent, MultiWOZAWMAgent, TASK
from eval_tod.multiwoz_matched_executor import (
    CONTRACT, PROTOCOL as EXECUTOR_PROTOCOL, bind_payload, digest, plan_observation,
    request_json, select_heads,
)
from eval_tod.sgd_adapter import gold_policy, visible_context
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_skill_agent import policy_token
from eval_tod.sgd_skill_agent import _json_object

PROTOCOL = 'multiwoz21_action_slot_standard_aligned_v2_agent_reference'
OFFLINE_SLOT_PROTOCOL = 'multiwoz21_train_slot_resources_v3_initial_four'
CONDITIONAL_SLOT_PROTOCOL = 'multiwoz21_train_conditional_slot_groups_v4_four'
INITIAL_SNIPPETS_PER_NODE = 4
ALIGNED_TASK = TASK.replace('Use only prior dialogue utterances at prediction time;',
    'Use prior dialogue utterances and the shared read-only DB observation at prediction time;') + '\n' + CONTRACT
LEARNING_CONTRACT = """
TRAIN rollouts and final evaluation use the SAME context-only DB planner,
resource lookup, action selection and fixed-head payload executor. The DB is
read-only and cannot book or produce a reference. Do not learn an instruction
to invent or memorize unobservable refs, entity choices or future surface forms.
The scorer remains strict; a missing observation is an evidence limitation.
Action rules choose the head bundle. Slot policies choose the applicable slots,
their multiplicity, and their value source CONDITIONAL on visible context.
Do not turn a union of alternative slots into a mandatory output checklist.
Binary acts keep their slot and omit value. Non-categorical annotation may have
several spans for one slot. Separate genuine omissions from representation or
unobservable-reference errors. Preserve supported successes.
Write reusable slot procedures into slot_policy on the exact node. Decide
whether additional residual branches, exceptions or compact TRAIN examples
would help in reference, and write those reference updates when justified.
The runtime agent decides whether to retrieve, what to query, how much to read,
and whether to look up more after selecting an action. Reference can support
slot inference. Do not use training instance values as runtime defaults.
"""
COMMON_LEARNING_CONTRACT = LEARNING_CONTRACT.split('Write reusable slot procedures into slot_policy', 1)[0] + """
Where reference storage is available, decide whether extra procedural notes or
compact TRAIN examples would be useful and add them when supported. Runtime
reference search/read/drop/finish decisions belong to the agent.
"""


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # A concurrent Windows reader can temporarily deny replacement.
            # Keep the complete temp file and retry only the filesystem write.
            if attempt == 9:
                raise
            time.sleep(min(.02 * 2 ** attempt, .2))


@dataclass(frozen=True)
class RuntimeLimits:
    candidate_limit: int = 3
    reference_max_rounds: int = 4
    reference_context_chars: int = 32768

    def __post_init__(self):
        if min(self.candidate_limit, self.reference_max_rounds) < 1 or self.reference_context_chars < 2:
            raise ValueError('Invalid runtime retrieval limits')


@dataclass
class ObservationCache:
    directory: Path
    kb_dir: Path
    model: str
    allow_generate: bool = True

    def __post_init__(self):
        self.directory, self.kb_dir = Path(self.directory), Path(self.kb_dir)
        self.db_sha256 = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(self.kb_dir.glob('*_db.json'))})

    def identity(self, dialogue, turn):
        # No target annotations, past gold heads, belief state or future text.
        return digest([EXECUTOR_PROTOCOL, self.model, self.db_sha256, dialogue['dialogue_id'],
                       turn, dialogue['services'], visible_context(dialogue, turn)[-12000:]])

    def get(self, dialogue, turn):
        signature = self.identity(dialogue, turn)
        path = self.directory / f'{signature}.json'
        if path.exists():
            cached = json.loads(path.read_text(encoding='utf-8'))
            if cached['signature'] != signature or cached['executor_protocol'] != EXECUTOR_PROTOCOL:
                raise ValueError('Observation cache protocol mismatch')
            return cached['observation']
        if not self.allow_generate:
            raise ValueError('Shared observation was not prepared before rollout')
        observation = plan_observation(dialogue, turn, kb_dir=self.kb_dir, model=self.model)
        atomic_json(path, {'signature': signature, 'executor_protocol': EXECUTOR_PROTOCOL, 'observation': observation})
        return observation


def reference_sections(text):
    """Index full sections; requested content is not pre-truncated to snippets."""
    headings = list(re.finditer(r'(?m)^#{1,4}\s+(.+)$', text))
    raw = [(m.group(1).strip(), text[m.start():headings[i + 1].start() if i + 1 < len(headings) else len(text)].strip())
           for i, m in enumerate(headings)] if headings else [('reference', text.strip())]
    sections, seen = {}, set()
    for title, content in raw:
        canonical = re.sub(r'\s+', ' ', content).strip()
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        identifier = 'ref-' + digest(canonical)[:16]
        sections[identifier] = {'section_id': identifier, 'title': title, 'content': content, 'total_chars': len(content)}
    return sections


class AgentReferenceSession:
    """Agent-controlled search/read/drop/finish before and after action choice."""

    def __init__(self, text, limits):
        self.sections = reference_sections(text)
        self.limits = limits
        self.units = []
        self.calls = []
        self.decisions = []
        self.plan = {'query_text': '', 'candidate_actions': []}

    def validate(self, payload):
        action = payload.get('action')
        if action not in {'search', 'read', 'drop', 'finish'}:
            raise ValueError('Reference action must be search/read/drop/finish; finish explicitly skips or stops lookup')
        decision = {'action': action}
        candidates = payload.get('candidate_actions', [])
        if not isinstance(candidates, list) or any(not isinstance(a, str) for a in candidates):
            raise ValueError('candidate_actions must be known node-label strings')
        decision['candidate_actions'] = candidates[:self.limits.candidate_limit]
        query = payload.get('query_text', '')
        if not isinstance(query, str):
            raise ValueError('query_text must be a string')
        decision['query_text'] = query.strip()[:300]
        if action == 'search':
            if not decision['query_text']:
                raise ValueError('Search needs an explicit query_text')
            top_k = payload.get('top_k')
            if type(top_k) is not int or top_k < 1:
                raise ValueError('Choose a positive integer top_k for search; use finish for zero lookups')
            decision['top_k'] = top_k
        if action in {'read', 'drop'}:
            identifiers = payload.get('section_ids')
            if (not isinstance(identifiers, list) or not identifiers or any(
                    not isinstance(key, str) or key not in self.sections for key in identifiers)):
                raise ValueError('Use section_ids returned by this reference library')
            decision['section_ids'] = list(dict.fromkeys(identifiers))
        if action in {'search', 'read'}:
            maximum = payload.get('max_chars')
            if maximum is not None and (type(maximum) is not int or maximum < 0):
                raise ValueError('max_chars is null for full sections or a nonnegative per-section character count')
            decision['max_chars'] = maximum
        return decision

    def _merge(self, section, maximum):
        old = next((u for u in self.units if u['section_id'] == section['section_id']), None)
        desired = section['content'] if maximum is None else section['content'][:maximum]
        if old and len(old['content']) >= len(desired):
            return
        others = [u for u in self.units if u['section_id'] != section['section_id']]

        def unit(length):
            return {**section, 'content': desired[:length], 'truncated': length < section['total_chars']}

        low, high, best = 0, len(desired), None
        while low <= high:
            middle = (low + high) // 2
            candidate = unit(middle)
            if len(json.dumps(others + [candidate], ensure_ascii=False)) <= self.limits.reference_context_chars:
                best, low = candidate, middle + 1
            else:
                high = middle - 1
        if best is not None and (old is None or len(best['content']) > len(old['content'])):
            self.units = others + [best]

    def execute(self, decision, stage):
        action = decision['action']
        if action == 'search':
            tokens = set(re.findall(r'\w+', decision['query_text'].casefold()))
            sections = sorted(self.sections.values(), key=lambda u: (
                -len(tokens & set(re.findall(r'\w+', u['content'].casefold()))), u['section_id']))[:decision['top_k']]
            for section in sections:
                self._merge(section, decision['max_chars'])
        elif action == 'read':
            sections = [self.sections[key] for key in decision['section_ids']]
            for section in sections:
                self._merge(section, decision['max_chars'])
        elif action == 'drop':
            sections = [self.sections[key] for key in decision['section_ids']]
            self.units = [u for u in self.units if u['section_id'] not in decision['section_ids']]
        else:
            raise ValueError('Finish is a decision, not a tool execution')
        visible = {u['section_id']: u for u in self.units}
        result = {'sections': [{'section_id': u['section_id'], 'title': u['title'], 'total_chars': u['total_chars'],
                   'included_chars': len(visible.get(u['section_id'], {}).get('content', '')),
                   'truncated': visible.get(u['section_id'], {}).get('truncated', True)} for u in sections],
                  'reference_context_chars': len(json.dumps(self.units, ensure_ascii=False)),
                  'context_capacity_chars': self.limits.reference_context_chars}
        self.calls.append({'stage': stage, 'request': deepcopy(decision), 'result': result,
                           'retained_section_ids': [u['section_id'] for u in self.units]})
        return result

    def planning_instructions(self, plan_candidates):
        return ('Include candidate_actions with up to ' + str(self.limits.candidate_limit)
                + ' known node labels for action comparison; candidates are advisory. ' if plan_candidates else
                'Action heads have already been fixed; do not reselect them. ')

    def catalog_text(self, stage):
        return ''

    def collect(self, *, model, stage, knowledge, context, observation, plan_candidates=False):
        if not self.sections and not plan_candidates:
            return
        last_result = None
        for turn in range(self.limits.reference_max_rounds):
            prompt = (ALIGNED_TASK + '\nChoose whether reference would help this ' + stage + ' stage. '
                'You control lookup, count and length. Return one JSON tool decision or finish. '
                'Schemas: {"action":"search","query_text":"your query","top_k":5,"max_chars":null}, '
                '{"action":"read","section_ids":["returned id"],"max_chars":null}, '
                '{"action":"drop","section_ids":["irrelevant id"]}, or {"action":"finish"}. '
                'null reads full sections; max_chars optionally requests a per-section excerpt; 0 lists metadata. '
                'Search covers the entire library. Read can expand an earlier excerpt. Drop releases context. '
                'Use finish immediately if lookup is unnecessary, no useful reference exists, or evidence is sufficient. '
                'You may retrieve several sections and issue follow-up searches. Treat reference examples as procedures, '
                'never as defaults for current entity values. '
                + self.planning_instructions(plan_candidates)
                + self.catalog_text(stage)
                + '\nReference library/remaining decision rounds/context capacity: '
                + json.dumps({'sections': len(self.sections), 'remaining_rounds': self.limits.reference_max_rounds - turn,
                              'context_chars': self.limits.reference_context_chars})
                + '\nCurrent method knowledge:\n' + knowledge
                + '\nCurrent retained reference:\n' + json.dumps(self.units, ensure_ascii=False)
                + '\nPrevious tool result:\n' + json.dumps(last_result, ensure_ascii=False)
                + '\nShared DB observation:\n' + json.dumps(observation['observations'], ensure_ascii=False)
                + '\nPrior dialogue:\n' + context)
            tag = 'multiwoz_resource_plan' if stage == 'action' else 'multiwoz_payload_reference'
            decision, diagnostic = request_json(prompt, model, tag, self.validate)
            self.decisions.append({'stage': stage, 'decision': decision, 'diagnostic': diagnostic})
            if decision['query_text']:
                self.plan['query_text'] = decision['query_text']
            if plan_candidates and decision['candidate_actions']:
                self.plan['candidate_actions'] = decision['candidate_actions']
            if decision['action'] == 'finish':
                return
            last_result = self.execute(decision, stage)

    def diagnostic(self):
        return {'mode': 'agent', 'calls': deepcopy(self.calls), 'decisions': deepcopy(self.decisions),
                'limits': asdict(self.limits), 'retained_section_ids': [u['section_id'] for u in self.units]}


def _act_key(act):
    return act['intent'], act['domain'], act['slot']


def training_profiles(train, graph, ontology, max_examples=INITIAL_SNIPPETS_PER_NODE, observations=None):
    """Summarize ALL TRAIN slots; initialize from four diverse snippets per node."""
    if type(max_examples) is not int or max_examples < 1:
        raise ValueError('Initial snippets per node must be a positive integer')
    by_token = defaultdict(list)
    for dialogue in train:
        for turn in dialogue['target_indices']['policy_turn_indices']:
            gold = gold_policy(dialogue, turn)
            types = defaultdict(set)
            for annotation in dialogue['turns'][turn].get('act_annotation_types', []):
                types[_act_key(annotation)].add(annotation['type'])
            pattern = tuple(sorted((*_act_key(a), '|'.join(sorted(types[_act_key(a)])) or 'unknown')
                                   for a in gold['acts']))
            context = visible_context(dialogue, turn)[-12000:]
            by_token[policy_token(gold)].append({'context': context, 'gold': gold, 'pattern': pattern,
                                                'types': types, 'dialogue': dialogue, 'turn_index': turn})
    profiles = []
    for node in graph['nodes']:
        examples = by_token[node['label']]
        if not examples:
            raise ValueError('A graph node has no TRAIN evidence')
        patterns = Counter(row['pattern'] for row in examples)
        selected, covered, seen_snippets = [], set(), set()
        # Duplicate utterance/label evidence should not spend the four slots.
        unique = {}
        for i, row in enumerate(examples):
            unique.setdefault(digest([row['context'], row['gold']]), i)
        remaining = list(unique.values())
        while remaining and len(selected) < max_examples:
            def rank(i):
                pattern = examples[i]['pattern']
                features = {('pattern', pattern)} | {('slot', act) for act in pattern}
                return (len(features - covered), patterns[pattern], -i)
            chosen = max(remaining, key=rank)
            row = examples[chosen]
            example = {'sample_id': f"{row['dialogue']['dialogue_id']}:{row['turn_index']}",
                       'context': row['context'], 'gold': deepcopy(row['gold'])}
            if observations is not None:
                example['shared_db_observation'] = observations.get(row['dialogue'], row['turn_index'])['observations']
            visible = example['context'] + json.dumps(example.get('shared_db_observation', []), ensure_ascii=False)
            example['unobservable_ref_slots'] = []
            for act in example['gold']['acts']:
                if act['slot'] == 'ref' and act.get('value') and act['value'].casefold() not in visible.casefold():
                    example['unobservable_ref_slots'].append(list(_act_key(act)))
                    act['value'] = '<unobservable_booking_result>'
            remaining.remove(chosen)
            signature = digest([example['context'], example['gold']])
            if signature in seen_snippets:
                continue
            seen_snippets.add(signature)
            selected.append(example)
            covered |= {('pattern', row['pattern'])} | {('slot', act) for act in row['pattern']}
        heads = {tuple(h) for h in node['action_heads']}
        schema = {}
        for row in examples:
            counts = Counter(_act_key(a) for a in row['gold']['acts'])
            for key, count in counts.items():
                item = schema.setdefault(key, {'intent': key[0], 'domain': key[1], 'slot': key[2],
                    'train_turns_present': 0, 'max_multiplicity': 0, 'annotation_types': set(), 'value_modes': set()})
                item['train_turns_present'] += 1
                item['max_multiplicity'] = max(item['max_multiplicity'], count)
                item['annotation_types'].update(row['types'].get(key, {'unknown'}))
            for act in row['gold']['acts']:
                schema[_act_key(act)]['value_modes'].add('valued' if act.get('value') else 'value_free')
        for item in schema.values():
            item['annotation_types'] = sorted(item['annotation_types'])
            item['value_modes'] = sorted(item['value_modes'])
            item['optional'] = item['train_turns_present'] < len(examples)
        profiles.append({'node_id': node['id'], 'heads': node['action_heads'],
            'offline_slot_protocol': OFFLINE_SLOT_PROTOCOL, 'initial_snippets_per_node': max_examples,
            'train_turns': len(examples), 'distinct_slot_patterns': len(patterns),
            'slot_patterns': [{'acts': p, 'count': n} for p, n in patterns.most_common()],
            'slot_schema': [schema[key] for key in sorted(schema)],
            'train_cases': selected,
            'allowed_slots': {d: ontology['domains'].get(d, {}).get('slots', []) for _, d in heads if d},
            'value_free_acts': [a for a in ontology.get('value_free_acts', []) if tuple(a[:2]) in heads]})
    return profiles


def slot_rule_coverage(profile, card):
    """Check schema coverage, not whether prose conditions are semantically right."""
    expected = {(*_act_key(item), mode) for item in profile.get('slot_schema', []) for mode in item['value_modes']}
    observed = {(*_act_key(act), 'value_free' if act['value_source'] == 'binary' else 'valued')
                for rule in card['slot_rules'] for act in rule['emit']}
    missing = sorted(expected - observed)
    unsupported = sorted(observed - expected) if expected else []
    return {'expected_slot_shapes': len(expected), 'covered_slot_shapes': len(expected & observed),
            'missing_slot_shapes': missing, 'unsupported_slot_shapes': unsupported,
            'verified': not missing and not unsupported}


def initial_reference_text(profiles):
    rows = []
    for profile in profiles:
        for index, example in enumerate(profile['train_cases'], 1):
            rows.append(f"## TRAIN snippet {profile['node_id']} / {index}\n"
                'Illustrative TRAIN evidence only. Infer conditions and value sources; '
                'never copy example entity values into the current prediction. '
                'The booking placeholder means no observable reference value exists.\n'
                + json.dumps({'node_id': profile['node_id'], 'heads': profile['heads'], **example}, ensure_ascii=False))
    return '\n\n'.join(rows)


def initial_slot_policy(profile, card):
    return ('Select only applicable conditional slot rules; their emit lists are alternatives, not a union.\n'
        + json.dumps(card['slot_rules'], ensure_ascii=False)
        + '\nTRAIN slot schema (inventory of alternatives, not mandatory output; '
          'max_multiplicity is an observed bound, not a required count):\n'
        + json.dumps(profile['slot_schema'], ensure_ascii=False))


def offline_slot_report(profiles, completed, initial_snippets):
    coverage = {p['node_id']: {**slot_rule_coverage(p, completed[p['node_id']]),
        'initial_snippets': len(p['train_cases']),
        'sample_ids': [case['sample_id'] for case in p['train_cases']],
        'train_turns': p['train_turns'], 'distinct_slot_patterns': p['distinct_slot_patterns']}
        for p in profiles}
    return {'protocol': OFFLINE_SLOT_PROTOCOL, 'initial_snippets_per_node': initial_snippets,
        'profile_sha256': digest(profiles), 'compiled_cards_sha256': digest(completed),
        'coverage_kind': 'TRAIN slot/value schema; condition semantics require held-out evaluation',
        'nodes': coverage, 'verified': all(item['verified'] for item in coverage.values())}


def compile_cards_job(job):
    return _compile_cards_job(job, conditional_groups=False)


def compile_conditional_cards_job(job):
    return _compile_cards_job(job, conditional_groups=True)


def _compile_cards_job(job, *, conditional_groups):
    path, profiles, model = job
    path = Path(path)
    offline_protocol = CONDITIONAL_SLOT_PROTOCOL if conditional_groups else OFFLINE_SLOT_PROTOCOL
    signature = digest([PROTOCOL, offline_protocol, EXECUTOR_PROTOCOL, profiles, model])
    saved = None
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['signature'] != signature:
            raise ValueError('TRAIN card compilation cache mismatch')
    expected = {p['node_id']: p for p in profiles}

    def validate(payload):
        cards = payload.get('cards')
        if not isinstance(cards, list) or len(cards) != len(expected) or {
                c.get('node_id') for c in cards if isinstance(c, dict)} != set(expected):
            returned = [c.get('node_id') if isinstance(c, dict) else '<non-object>' for c in cards] if isinstance(cards, list) else None
            raise ValueError('Return exactly one complete card per exact requested node_id. '
                + json.dumps({'expected_node_ids': sorted(expected), 'returned_node_ids': returned}, ensure_ascii=False)
                + '. Combine ALL slot_rules for a node inside ONE card object. Do not create a separate card '
                  'for each slot, rule, TRAIN example, alternative pattern, or primitive head. '
                  'Keep one generalized action_rule and put alternative conditions in that card slot_rules array.')
        for card in cards:
            profile = expected[card['node_id']]
            heads = {tuple(h) for h in profile['heads']}
            if not isinstance(card.get('action_rule'), str) or len(card['action_rule'].strip()) < 20:
                raise ValueError('Specific action_rule is required')
            if conditional_groups and (not isinstance(card.get('slot_selection_policy'), str)
                    or len(card['slot_selection_policy'].strip()) < 20):
                raise ValueError('slot_selection_policy must explain how prior context chooses a complete slot subset, '
                                 'including when to omit optional fields and how many missing fields to request')
            rules = card.get('slot_rules')
            if not isinstance(rules, list) or not rules:
                raise ValueError('Require nonempty conditional slot_rules, not a union slot checklist')
            observed, binary_source_errors = set(), []
            for rule in rules:
                if (not isinstance(rule, dict) or not isinstance(rule.get('when'), str) or not rule['when'].strip()
                        or not isinstance(rule.get('emit'), list) or not rule['emit']):
                    raise ValueError('Each slot rule needs a visible-context condition and nonempty emit list. '
                        'Put omission-only advice in slot_selection_policy; do not add emit:[] rules. '
                        'A slotless act still needs a full intent/domain/slot/value_source emit object with slot:"".')
                if not conditional_groups and len(rule['emit']) != 1:
                    raise ValueError('Give each slot/value shape its own observable when condition and one emit entry. '
                        'Do not couple independently optional slots such as phone/address/postcode. '
                        'Several rules may apply together when several facts are requested; describe repeated spans in when.')
                for act in rule['emit']:
                    if not isinstance(act, dict) or set(act) != {'intent', 'domain', 'slot', 'value_source'}:
                        raise ValueError('Emit entries use intent/domain/slot/value_source, never literal values')
                    head = act['intent'], act['domain']
                    if head not in heads or not isinstance(act['slot'], str):
                        raise ValueError('Rule emitted a head outside this exact action card')
                    if act['slot'] and act['slot'] not in profile['allowed_slots'].get(act['domain'], []):
                        raise ValueError('Rule emitted an unknown TRAIN slot')
                    # A value-free TRAIN binary act has no source value. This
                    # is representation normalization, not a guessed rule.
                    if list(_act_key(act)) in profile['value_free_acts'] and act['value_source'] in ('', None):
                        act['value_source'] = 'binary'
                    if act['value_source'] not in {'binary', 'context', 'db', 'context_or_db', 'unavailable'}:
                        raise ValueError(f"Invalid value source for {_act_key(act)}: {act['value_source']!r}; "
                                         'use binary/context/db/context_or_db/unavailable')
                    if list(_act_key(act)) in profile['value_free_acts'] and act['value_source'] != 'binary':
                        binary_source_errors.append({'act': list(_act_key(act)), 'supplied_source': act['value_source'],
                                                     'required_source': 'binary'})
                    observed.add(head)
            if binary_source_errors:
                raise ValueError('TRAIN binary acts have no value. Fix ALL these exact entries in one response: '
                                 + json.dumps(binary_source_errors, ensure_ascii=False))
            if observed != heads:
                raise ValueError('Conditional rules must account for every head of the card. '
                    + json.dumps({'expected_heads': sorted(heads), 'missing_heads': sorted(heads - observed)})
                    + '. Empty/global domain is ""; slotless acts use slot:"" and value_source:"binary".')
            coverage = slot_rule_coverage(profile, card)
            if not coverage['verified']:
                examples = [{'intent': shape[0], 'domain': shape[1], 'slot': shape[2],
                    'value_source': 'binary' if shape[3] == 'value_free' else 'context_or_db'}
                    for shape in coverage['missing_slot_shapes']]
                raise ValueError('Conditional slot rules must cover every TRAIN slot/value shape for this node; '
                    'coverage is over alternative rules, not one mandatory emission. '
                    + json.dumps(coverage, ensure_ascii=False)
                    + ' Missing emit shapes (supply your own TRAIN-supported when conditions): '
                    + json.dumps(examples, ensure_ascii=False)
                    + '. Empty slot is a literal ""; value_free inform acts also require binary.')
        return cards

    if saved is not None:
        return validate({'cards': saved['cards']})

    if conditional_groups:
        compilation_task = (
            '\nCompile exact action cards from TRAIN only. Return JSON {"cards":[{"node_id":"COPY EXACT ID",'
            '"action_rule":"when this exact head bundle applies; no slot requirements",'
            '"slot_selection_policy":"how visible context chooses the appropriate complete slot pattern",'
            '"slot_rules":[{"when":"observable prior-context condition",'
            '"emit":[{"intent":"inform","domain":"hotel","slot":"name","value_source":"context_or_db"}]}]}]}. '
            'Exactly ONE card per node; keep all conditional patterns in its slot_rules array. '
            'All requested heads must occur across rules, including slotless general acts with domain:"". '
            'Every rule needs at least one emit. Put omission-only advice in slot_selection_policy. '
            'A TRAIN empty slot remains slot:""; value_free inform acts use binary too. '
            'A rule may emit SEVERAL jointly chosen slots. Learn coherent subsets from the TRAIN co-occurrence table, '
            'not a union of all possible slots and not independent always-on field switches. '
            'Use prior user requests, information already supplied, the previous visible question and the latest reply, '
            'and observable search/clarification/booking phase to distinguish patterns. '
            'A field existing in DB is a value source, not a reason to say it. Missing required information does not '
            'automatically mean requesting every missing field in one turn: learn the TRAIN request batching policy. '
            'The four initialization cases include common and contrasting contexts. Explain their different subsets '
            'and suppressions; use the FULL co-occurrence statistics for alternatives outside those four cases. '
            'Do not use circular triggers such as "the system intends to output price" or assume the hidden next '
            'system utterance. Describe the observable fact which makes that price relevant NOW. '
            'Keep routing/action selection out of slot_selection_policy, and slot mandates out of action_rule. '
            'Cover every slot/value shape across alternatives without emitting them together. '
            'Repeated spans are context/representation dependent; max_multiplicity is not a requested repeat count. '
            'Emit entries have exactly intent/domain/slot/value_source, no literal values. '
            'value_source is binary/context/db/context_or_db/unavailable; TRAIN value_free acts require binary. '
            'No training IDs, example values, invented booking results, fixed search thresholds, '
            'or unsupported domain slot names in the learned policy. '
            'When TRAIN does not identify a unique human response policy, state the supported ambiguity '
            'and a context-grounded preference instead of inventing an obligatory rule.\nTRAIN profiles:\n')
    else:
        compilation_task = (
        '\nCompile exact action cards from TRAIN only. Return JSON {"cards":[{"node_id":"COPY EXACT ID",'
        '"action_rule":"when this exact bundle applies","slot_rules":[{"when":"observable condition",'
        '"emit":[{"intent":"inform","domain":"hotel","slot":"name","value_source":"context_or_db"}]}]}]}. '
        'Use concise generalized conditional rules per card, enough to cover ALL slot_schema entries, '
        'including entries not demonstrated in the four initialization snippets. '
        'Each rule has exactly ONE emit entry and a slot-specific when grounded in PRIOR user utterances '
        'and the current read-only DB observation. Several rules can apply together when several facts are needed. '
        'Each requested NODE gets exactly ONE card object containing ALL its slot_rules. '
        'Never create one card per slot/rule/pattern/head, and never repeat a node_id in the cards array. '
        'Do not group independently optional fields under a shared emit list. '
        'For example, a phone request triggers phone only, not address and postcode. '
        'A field merely existing in DB is not sufficient reason to emit it. '
        'Describe when to repeat a slot for multiple spans in that slot rule; no fixed repeat count. '
        'Avoid circular conditions such as "when the system outputs this slot" or conditions using future utterances. '
        'Each emit object has exactly those four fields. '
        'Empty slot is allowed. value_source is binary/context/db/context_or_db/unavailable. '
        'No instance values or dialogue IDs. Mention omissions, repeated same-slot spans and how to select '
        'the applicable subset in the condition text. Distinguish answering requested details, summarizing '
        'search constraints and confirming bookings. Invent no booking observations. '
        'The full co-occurrence table lists ALTERNATIVES; never emit every variant together. '
        'Across alternative rules cover every slot and every observed value mode. Use binary for value_free acts, '
        'and context/db/context_or_db/unavailable for valued acts. Do not add slots unsupported for this node. '
        'Optional slots need observable conditions. max_multiplicity is a bound observed in TRAIN, '
        'not a required repeat count. An unobservable booking placeholder is not a value source; use unavailable. '
        'Rule text is guidance for an executor, not code. Every requested card is mandatory.\nTRAIN profiles:\n')
    prompt = ALIGNED_TASK + '\n' + LEARNING_CONTRACT + compilation_task + json.dumps(profiles, ensure_ascii=False)
    # A failed validation attempt already consumed a model call. A narrow
    # representation fix can reuse that exact prompt's logged response after
    # the same full validator accepts it; never synthesize missing card rules.
    for failure_path in sorted(path.parent.parent.glob('parallel_cases/**/validation_failures/multiwoz_aligned_card_compile_*.json')):
        failure = json.loads(failure_path.read_text(encoding='utf-8'))
        if failure.get('prompt_sha256') != digest(prompt):
            continue
        try:
            cards = validate(_json_object(failure['raw_response']))
        except (ValueError, KeyError, TypeError):
            continue
        atomic_json(path, {'signature': signature, 'protocol': PROTOCOL, 'cards': cards,
            'diagnostic': {'attempts': 0, 'format_recovery': 'train_binary_value_source_omission',
                           'recovered_response_path': str(failure_path), 'raw_response': failure['raw_response']}})
        return cards
    cards, diagnostic = request_json(prompt, model, 'multiwoz_aligned_card_compile', validate)
    atomic_json(path, {'signature': signature, 'protocol': PROTOCOL, 'cards': cards, 'diagnostic': diagnostic})
    return cards


class MultiWOZAlignedSkillAgent(MultiWOZStandardSkillAgent):
    protocol = PROTOCOL
    task_prompt = ALIGNED_TASK
    diagnosis_prompt = MultiWOZStandardSkillAgent.diagnosis_prompt.replace(TASK, ALIGNED_TASK) + '\n' + LEARNING_CONTRACT
    # Deliberately derive from the native template without the legacy no-DB text.
    from eval_tod.sgd_standard_skill_agent import REFLECTION_PROMPT as _REFLECTION, _TASK
    reflection_prompt = _REFLECTION.replace(_TASK, ALIGNED_TASK) + '\n' + LEARNING_CONTRACT

    def __init__(self, *, observations, limits=None, **kwargs):
        self.observations = observations
        self.limits = limits or RuntimeLimits()
        self.cards_ready = False
        self.offline_slot_audit = None
        kwargs['candidate_limit'] = self.limits.candidate_limit
        kwargs['reference_max_chars'] = self.limits.reference_context_chars
        super().__init__(**kwargs)
        self.structured_reflection = True

    def runtime_contract(self):
        return {'protocol': PROTOCOL, 'executor_protocol': EXECUTOR_PROTOCOL, 'model': self.model,
                'db_sha256': self.observations.db_sha256, 'ontology_sha256': digest(self.ontology),
                'retrieval': asdict(self.limits), 'input': 'prior_utterances_and_context_planned_readonly_db',
                'runtime_gold_action_history': False, 'reference_policy': 'agent_search_read_drop_finish_before_and_after_action',
                'payload': 'fixed_heads_exact_current_card_and_agent_selected_reference',
                'score': 'original_strict_unordered_act_multiset'}

    def resource_hash(self):
        return digest([super().resource_hash(), self.runtime_contract()])

    def mine(self, dialogues, directory, *, cases_per_node=INITIAL_SNIPPETS_PER_NODE, compile_batch_size=4, **kwargs):
        # Reuse TraceCover and native standard initialization; replace only the
        # permissive generic compiler with complete conditional TRAIN cards.
        directory = Path(directory)
        super().mine(dialogues, directory, cases_per_node=cases_per_node, compile_rules=False)
        profiles = training_profiles(dialogues, self.graph, self.ontology, cases_per_node, self.observations)
        atomic_json(directory / 'train_card_profiles.json', profiles)
        groups = [profiles[i:i + compile_batch_size] for i in range(0, len(profiles), compile_batch_size)]
        jobs, index = [], 0
        while groups:
            group = groups.pop(0)
            if len(json.dumps(group, ensure_ascii=False)) > 48000 and len(group) > 1:
                middle = len(group) // 2
                groups[:0] = [group[:middle], group[middle:]]
                continue
            index += 1
            jobs.append((directory / 'card_compile' / f'batch_{index:04d}.json', group, self.model))
        outcomes = ordered_cases(compile_cards_job, jobs, workers=self.parallel_workers,
                                 directory=directory, phase='offline_compile')
        completed = {}
        for outcome in outcomes:
            for card in require_value(outcome):
                completed[card['node_id']] = card
        if set(completed) != {n['id'] for n in self.graph['nodes']}:
            raise ValueError('Incomplete action cards: refusing TRAIN rollout/evaluation')
        self.offline_slot_audit = offline_slot_report(profiles, completed, cases_per_node)
        if not self.offline_slot_audit['verified']:
            raise ValueError('Incomplete offline slot resources: refusing TRAIN rollout/evaluation')
        by_id = {p['node_id']: p for p in profiles}
        for key, card in completed.items():
            self.resources['action_rules'][key] = card['action_rule']
            self.resources['slot_policies'][key] = initial_slot_policy(by_id[key], card)
        self.base_action_rules = '\n\n'.join(f'## {k}\n{v}' for k, v in self.resources['action_rules'].items())
        self.base_slot_policies = '\n\n'.join(f'## {k}\n{v}' for k, v in self.resources['slot_policies'].items())
        self.base_reference = initial_reference_text(profiles)
        self.cards_ready = True
        self.working_skill = self._seed_skill()
        self.save(directory)

    def runtime_skill_text(self):
        if not self.graph or not self.state:
            return ALIGNED_TASK
        # State is authoritative. Render each topology/guard once; Action Card
        # text belongs in candidate/selected cards, never repeated in main skill.
        rows = ['# MultiWOZ backbone skill', ALIGNED_TASK, '## Nodes (unordered heads)']
        for node in self.graph['nodes']:
            rows.append(node['label'] + ': ' + ', '.join(f'{i}:{d or "general"}' for i, d in node['action_heads']))
        rows.append('## Backbone and accepted routing guards')
        for edge in self.state.get('edges', {}).values():
            if edge.get('visibility') == 'skill':
                rows.append(f"{edge['source_action']} -> {edge['target_action']}: {edge.get('guard', '')}")
        return '\n'.join(rows)

    def reference_text(self):
        rows = [self.base_reference] if self.base_reference else []
        for edge_id, edge in self.state.get('edges', {}).items():
            if edge.get('visibility') != 'skill':
                guard = edge.get('guard') or self.resources['transition_guards'].get(f"{edge['source']} => {edge['target']}", '')
                rows.append(f"## {edge_id}\n{edge['source_action']} -> {edge['target_action']}: {guard}")
        for index, note in enumerate(self.state.get('reference_notes', [])):
            rows.append(f"## Note {index}: {note.get('edge_id', '')}\n{note.get('content', '')}")
        return '\n\n'.join(rows)

    def predict(self, dialogue, turn_index):
        if not self.cards_ready:
            raise ValueError('Complete offline cards are required before prediction')
        observation = self.observations.get(dialogue, turn_index)
        context = visible_context(dialogue, turn_index)[-12000:]
        skill = self.runtime_skill_text()
        session = AgentReferenceSession(self.reference_text(), self.limits)
        session.collect(model=self.model, stage='action', knowledge=skill, context=context,
                        observation=observation, plan_candidates=True)
        plan = deepcopy(session.plan)
        query = plan['query_text'] or observation['plan']['query_text'] or context[-1800:]
        aliases = {alias: node['id'] for node in self.graph['nodes'] for alias in (node['id'], node['label'])}
        cards = {node['id']: self.action_card(node) for node in self.graph['nodes']}
        tokens = set(re.findall(r'\w+', query.casefold()))
        fallback = sorted(cards, key=lambda cid: (-len(tokens & set(re.findall(r'\w+', cards[cid]['rule'].casefold()))), cid))
        candidates = list(dict.fromkeys([aliases[a] for a in plan['candidate_actions'] if a in aliases] + fallback))[:self.limits.candidate_limit]
        reference_text = json.dumps(session.units, ensure_ascii=False)
        knowledge = skill + '\nCandidate Action Cards:\n' + json.dumps([cards[c] for c in candidates], ensure_ascii=False)
        knowledge += '\nRetrieved reference:\n' + reference_text
        heads, selection = select_heads(dialogue, turn_index, observation=observation, knowledge=knowledge, model=self.model)
        chosen = {tuple(h) for h in heads}
        selected = [node['id'] for node in self.graph['nodes'] if {tuple(h) for h in node['action_heads']} == chosen]
        payload_knowledge = 'Selected complete Action Card:\n' + json.dumps([cards[c] for c in selected], ensure_ascii=False)
        if not selected:
            payload_knowledge += '\nUnseen bundle: use TRAIN schema and visible evidence; do not substitute an unrelated card.'
        selection_reference = deepcopy(session.units)
        session.collect(model=self.model, stage='payload', knowledge=payload_knowledge + '\nFixed heads:\n' + json.dumps(heads),
                        context=context, observation=observation)
        reference_text = json.dumps(session.units, ensure_ascii=False)
        payload_knowledge += '\nAgent-selected reference:\n' + reference_text
        prediction, payload = bind_payload(dialogue, turn_index, heads=heads, observation=observation,
            knowledge=payload_knowledge, ontology=self.ontology, model=self.model)
        return prediction, {'status': 'ok', 'executor_protocol': EXECUTOR_PROTOCOL, 'runtime_contract': self.runtime_contract(),
            'resource_sha256': self.resource_hash(), 'source_node': 'context_only_no_gold_history',
            'reference_plan': session.plan, 'reference_plan_diagnostic': session.diagnostic(),
            'selection_reference': selection_reference, 'retrieved_reference': deepcopy(session.units),
            'candidate_ids': candidates, 'selected_action_heads': heads, 'selected_card_ids': selected,
            'selected_action_cards': [cards[c] for c in selected], 'action_selection': selection, 'payload': payload,
            'shared_observation': observation, 'shared_observation_sha256': digest(observation),
            'visible_context_for_diagnostics': context,
            'prompt_sizes': {'main_skill': len(skill), 'payload_knowledge': len(payload_knowledge), 'reference': len(reference_text)}}

    def diagnosis_skill_text(self):
        return self.runtime_skill_text()

    def diagnosis_options(self, records):
        labels = {r['sgd_evidence']['gold_target'] for r in records} | {
            policy_token(r['sgd_evidence']['predicted_policy']) for r in records}
        available = []
        for node in self.graph['nodes']:
            if node['label'] in labels:
                card = self.action_card(node)
                available.extend([
                    {'resource': 'action_rule', 'title': node['label'], 'content': card['rule']},
                    {'resource': 'slot_policy', 'title': node['label'], 'content': card['payload_policy']}])
        return {'available_resources_override': available}

    def _native_resource_updates(self, updates):
        """Unpack explicit composite-card serialization into its typed stores.

        The model has already supplied both rules. This does not infer missing
        content or move arbitrary prose based on keyword heuristics.
        """
        aliases = {alias: n for n in self.graph['nodes'] for alias in (n['id'], n['label'])}
        normalized, changes = [], []
        for original in updates:
            update = deepcopy(original)
            content = update.get('content') if isinstance(update, dict) else None
            body = None
            if isinstance(content, str) and content.lstrip().startswith('{'):
                try:
                    body = json.loads(content)
                except json.JSONDecodeError:
                    pass
            if (isinstance(body, dict) and 'rule' in body and 'payload_policy' in body
                    and update.get('resource') in {'action_card', 'action_rule', 'action_node', 'slot_policy'}):
                if update.get('op', 'upsert') != 'upsert':
                    raise ValueError('Composite card deletion cannot delete a slot policy')
                node = aliases.get(update.get('action') or body.get('label'))
                if node is None:
                    raise ValueError('Composite card target must name an existing exact action node')
                if body.get('label') and aliases.get(body['label']) != node:
                    raise ValueError('Composite card label conflicts with resource action target')
                if body.get('heads') is not None and {tuple(h) for h in body['heads']} != {
                        tuple(h) for h in node['action_heads']}:
                    raise ValueError('Composite card may not change node action heads')
                for field, resource in [('rule', 'action_rule'), ('payload_policy', 'slot_policy')]:
                    if not isinstance(body[field], str) or not body[field].strip():
                        raise ValueError('Composite card needs nonempty rule and payload_policy strings')
                    normalized.append({**update, 'resource': resource, 'action': node['label'], 'content': body[field]})
                changes.append({'action': node['label'], 'from_resource': update['resource'],
                    'to_resources': ['action_rule', 'slot_policy'], 'source_content_sha256': digest(content)})
            else:
                normalized.append(update)
        return normalized, changes

    def normalize_diagnosis(self, report):
        normalized, changes = self._native_resource_updates(report.get('candidate_updates', []))
        report['candidate_updates'] = normalized
        report['resource_serialization_normalizations'] = changes
        return report

    def _validate_structured_reflection(self, reflection):
        updates, changes = self._native_resource_updates(reflection.get('updates', []))
        reflection['updates'] = updates
        reflection['resource_serialization_normalizations'] = changes
        super()._validate_structured_reflection(reflection)

    def _project_record(self, record):
        evidence = record['sgd_evidence']
        diagnostic = evidence['runtime_diagnostic']
        result = {k: deepcopy(evidence[k]) for k in ('sample_id', 'conversation_id', 'turn_index', 'source_action',
            'gold_target', 'gold_policy', 'predicted_policy', 'action_correct', 'slot_exact_correct', 'slot_error_bucket', 'context')}
        result['shared_db_observation'] = diagnostic['shared_observation']['observations']
        result['reference_query'] = diagnostic['reference_plan']['query_text']
        result['retrieved_reference'] = diagnostic['retrieved_reference']
        result['selection_reference'] = deepcopy(diagnostic.get('selection_reference', diagnostic['retrieved_reference']))
        lookup = diagnostic.get('reference_plan_diagnostic', {})
        result['reference_lookup'] = {'calls': deepcopy(lookup.get('calls', [])),
            'decisions': [{k: deepcopy(d.get(k)) for k in ('stage', 'decision')} for d in lookup.get('decisions', [])]}
        result['selected_card_ids'] = diagnostic['selected_card_ids']
        result['executor_protocol'] = diagnostic['executor_protocol']
        # Do not teach memorized future booking IDs to the optimizer.
        visible = diagnostic.get('visible_context_for_diagnostics', result['context']) + json.dumps(
            result['shared_db_observation'], ensure_ascii=False)
        unavailable = []
        for act in result['gold_policy']['acts']:
            if act['slot'] == 'ref' and act.get('value') and act['value'].casefold() not in visible.casefold():
                unavailable.append([act['intent'], act['domain'], act['slot']])
                act['value'] = '<unobservable_booking_result>'
        for pred in result['predicted_policy']['acts']:
            if pred['slot'] == 'ref' and pred.get('value') and pred['value'].casefold() not in visible.casefold():
                pred['value'] = '<unobserved_generated_ref>'
        result['unobservable_gold_ref_slots'] = unavailable
        return result

    def save(self, directory):
        super().save(directory)
        if self.state:
            (directory / 'reference.md').write_text(self.reference_text(), encoding='utf-8')
            atomic_json(directory / 'runtime_contract.json', {**self.runtime_contract(), 'cards_ready': self.cards_ready})
            if self.offline_slot_audit is not None:
                atomic_json(directory / 'offline_slot_audit.json', self.offline_slot_audit)
            if self.cards_ready:
                atomic_json(directory / 'action_cards.json', {node['id']: self.action_card(node) for node in self.graph['nodes']})

    def load(self, directory):
        saved = json.loads((directory / 'runtime_contract.json').read_text(encoding='utf-8'))
        if {k: v for k, v in saved.items() if k != 'cards_ready'} != self.runtime_contract():
            raise ValueError('Train/eval runtime contract mismatch; legacy resources cannot masquerade as aligned training')
        super().load(directory)
        audit_path = directory / 'offline_slot_audit.json'
        self.offline_slot_audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else None
        if self.offline_slot_audit is not None and not self.offline_slot_audit['verified']:
            raise ValueError('Offline slot resource coverage was not verified')
        self.cards_ready = bool(saved['cards_ready'])
        if self.cards_ready:
            for node in self.graph['nodes']:
                rule = self.action_card(node)['rule']
                if rule.lstrip().startswith('{'):
                    try:
                        composite = json.loads(rule)
                    except json.JSONDecodeError:
                        composite = None
                    if isinstance(composite, dict) and 'payload_policy' in composite:
                        raise ValueError('Composite card stored inside action_rule; replay typed updates before evaluation')


@dataclass
class AlignedBaselinePredictor:
    """The same heads/payload executor, also usable inside baseline TRAIN rollout."""
    observations: ObservationCache
    limits: RuntimeLimits = RuntimeLimits()

    def __call__(self, dialogue, turn_index, *, model, ontology, workflow_text='', exemplar_text='',
                 skill_text='', reference_text='', **kwargs):
        observation = self.observations.get(dialogue, turn_index)
        context = visible_context(dialogue, turn_index)[-12000:]
        session = AgentReferenceSession(reference_text, self.limits)
        base_knowledge = '\n'.join((workflow_text, exemplar_text, skill_text))
        session.collect(model=model, stage='action', knowledge=base_knowledge, context=context, observation=observation)
        selection_reference = deepcopy(session.units)
        knowledge = base_knowledge + '\nAgent-selected reference:\n' + json.dumps(session.units, ensure_ascii=False)
        heads, selection = select_heads(dialogue, turn_index, observation=observation, knowledge=knowledge, model=model)
        session.collect(model=model, stage='payload', knowledge=base_knowledge + '\nFixed heads:\n' + json.dumps(heads),
                        context=context, observation=observation)
        knowledge = base_knowledge + '\nAgent-selected reference:\n' + json.dumps(session.units, ensure_ascii=False)
        prediction, payload = bind_payload(dialogue, turn_index, heads=heads, observation=observation,
            knowledge=knowledge, ontology=ontology, model=model)
        return prediction, {'status': 'ok', 'executor_protocol': EXECUTOR_PROTOCOL, 'selected_action_heads': heads,
            'action_selection': selection, 'payload': payload, 'shared_observation': observation,
            'shared_observation_sha256': digest(observation), 'selection_reference': selection_reference,
            'retrieved_reference': deepcopy(session.units), 'reference_plan': session.plan,
            'reference_plan_diagnostic': session.diagnostic()}


class MultiWOZAlignedAWMAgent(MultiWOZAWMAgent):
    induction_task_contract = ALIGNED_TASK + '\n' + COMMON_LEARNING_CONTRACT
