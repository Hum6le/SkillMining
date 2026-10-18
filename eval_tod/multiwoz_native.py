"""Native prose slot resources with the existing standard update mechanism.

Legacy fixed-emit protocols stay reproducible. Only this fresh-run adapter
restores unrestricted policy bodies, complete cards and global reference search.
"""
from copy import deepcopy
import json
from pathlib import Path

from eval_tod.multiwoz_aligned import (
    ALIGNED_TASK, LEARNING_CONTRACT, AgentReferenceSession, MultiWOZAlignedSkillAgent,
    atomic_json, initial_reference_text, training_profiles,
)
from eval_tod.multiwoz_matched_executor import digest, request_json, PROTOCOL as EXECUTOR_PROTOCOL
from eval_tod.multiwoz_slot_policy import ConditionalMultiWOZSkillAgent, ScopedReferenceSession
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_standard_skill_agent import SGDStandardSkillAgent
from eval_tod.sgd_skill_agent import _json_object

PROTOCOL = 'multiwoz21_native_text_slot_standard_v1'
OFFLINE_PROTOCOL = 'multiwoz21_train_native_text_policy_init4_v1'


def parse_native_response(raw):
    try:
        return _json_object(raw)
    except json.JSONDecodeError as exc:
        raise ValueError('Invalid resource JSON serialization. Every text body must be a JSON string. '
            'Escape interior double quotes with a backslash, or use single quotes within the prose. '
            'Keep policy wording/meaning intact; return the complete updates object. '
            f'Parser location: line {exc.lineno}, column {exc.colno}: {exc.msg}') from exc


class GlobalReferenceSession(ScopedReferenceSession):
    """Retain validated action transport, restore whole-library reference tools."""
    index_snippets = False

    def catalog_text(self, stage):
        if stage != 'action':
            return ''
        return '\nAction identifier catalog:\n' + json.dumps([
            {'node_id': n['id'], 'handle': self.handles[n['id']], 'heads': n['action_heads']}
            for n in self.nodes.values()], ensure_ascii=False)

    def planning_instructions(self, plan_candidates):
        if not plan_candidates:
            return 'Action heads are fixed; choose whether any whole-library reference would help payload inference. '
        return super().planning_instructions(True)

    def validate(self, payload):
        result = super().validate(payload)
        if result['action'] == 'search':
            result['scope'], result['node_ids'] = 'all', []
        return result

    def execute(self, decision, stage):
        return AgentReferenceSession.execute(self, decision, stage)


def compile_native_resources_job(job):
    """Use native resource/key/text updates; the policy body has no DSL."""
    path, profiles, model = job
    path = Path(path)
    signature = digest([PROTOCOL, OFFLINE_PROTOCOL, EXECUTOR_PROTOCOL, profiles, model])
    nodes = {p['node_id'] for p in profiles}
    address_normalizations = []

    def validate(payload):
        updates = payload.get('updates')
        if not isinstance(updates, list):
            raise ValueError('Return updates as native resource/key/text objects')
        address_normalizations.clear()
        seen, grouped = set(), {}
        for update in updates:
            if (not isinstance(update, dict) or update.get('resource') not in {'action_rules', 'slot_policies', 'reference'}
                    or not isinstance(update.get('key'), str) or not isinstance(update.get('text'), str)
                    or not update['text'].strip()):
                raise ValueError('Use native resource/key/text objects with nonempty prose; '
                    'requested parent node IDs: ' + json.dumps(sorted(nodes)))
            supplied = update['key']
            parents = [node for node in nodes if supplied == node or supplied.startswith(node + ':')]
            if len(parents) != 1:
                raise ValueError('Resource key must address one of these existing parent nodes: '
                    + json.dumps(sorted(nodes)) + '. Put optional subsection names inside text.')
            parent = parents[0]
            key = update['resource'], supplied
            if key in seen:
                raise ValueError('One resource body per node; organize its sections within that text')
            seen.add(key)
            body = update['text']
            if supplied != parent:
                section = supplied[len(parent) + 1:]
                if not section.strip():
                    raise ValueError('Subsection address requires a nonempty section name')
                body = f'### {section}\n{body}'
                address_normalizations.append({'resource': update['resource'], 'from': supplied,
                    'to': parent, 'section': section, 'text_sha256': digest(update['text'])})
            grouped.setdefault((update['resource'], parent), []).append(body)
        required = {(role, node) for role in ('action_rules', 'slot_policies') for node in nodes}
        if required - set(grouped):
            raise ValueError('Missing native resources: ' + json.dumps(sorted(required - set(grouped))))
        return [{'resource': role, 'key': node, 'text': '\n\n'.join(parts)}
                for (role, node), parts in grouped.items()]

    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['signature'] != signature:
            raise ValueError('Native resource compilation input changed')
        return validate(saved)
    prompt = (ALIGNED_TASK + '\n' + LEARNING_CONTRACT
        + '\nCompile native action_rules and slot_policies from TRAIN only, before agent rollout. '
        'Return JSON {"updates":[{"resource":"action_rules|slot_policies|reference",'
        '"key":"exact requested node_id","text":"final reusable natural-language resource"}]}. '
        'Provide action_rules and slot_policies for every node. reference is optional: decide whether '
        'additional procedural branches, evidence or exceptions are useful. '
        'Organize each policy freely using paragraphs, headings, steps, joint reasoning, alternatives '
        'and omissions as appropriate. There is no required when/emit/value_source format, no rule '
        'per slot requirement, and no requirement to enumerate every observed slot in policy prose. '
        'The runtime schema separately defines legal outputs. TRAIN inventory and co-occurrence '
        'statistics describe alternatives; they are not an obligatory output union. '
        'Learn from prior questions/replies and successes about what to answer/request now, what to '
        'omit, when to combine fields, how many spans, and where visible values come from. '
        'Use exactly the supplied visible information boundary; current/future system text and '
        'belief annotations are unavailable. No invented booking observations or placeholder refs. '
        'Give finalized concise procedures, not drafting commentary, self-corrections or sample-by-sample '
        'analysis. Keep instance values and sample IDs out of reusable policy text. Unsupported '
        'distinctions should remain explicitly uncertain instead of becoming mandatory rules. '
        'These are free prose resources that standard reflection can later revise.\nTRAIN profiles:\n'
        + json.dumps(profiles, ensure_ascii=False))
    # Revalidate a paid response with the identical TRAIN prompt before making
    # another call. Only explicit subsection addresses are normalized; every
    # prose body and its order are retained, without inventing rules or nodes.
    root = path.parent.parent
    manifest = root.parent / 'data_manifest.json'
    same_run_model = manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')).get('model') == model
    for failure_path in sorted(root.glob('parallel_cases/**/validation_failures/multiwoz_native_resource_compile_*.json')):
        failure = json.loads(failure_path.read_text(encoding='utf-8'))
        if failure.get('prompt_sha256') != digest(prompt) or not same_run_model:
            continue
        try:
            updates = validate(_json_object(failure['raw_response']))
        except (ValueError, TypeError, KeyError):
            continue
        diagnostic = {'attempts': 0, 'format_recovery': 'explicit_subsections_joined_without_text_changes',
            'recovered_response_path': str(failure_path), 'raw_response': failure['raw_response'],
            'address_normalizations': deepcopy(address_normalizations)}
        atomic_json(path, {'signature': signature, 'protocol': OFFLINE_PROTOCOL, 'updates': updates, 'diagnostic': diagnostic})
        return updates
    updates, diagnostic = request_json(prompt, model, 'multiwoz_native_resource_compile', validate,
                                       parser=parse_native_response)
    diagnostic['address_normalizations'] = deepcopy(address_normalizations)
    atomic_json(path, {'signature': signature, 'protocol': OFFLINE_PROTOCOL,
                       'updates': updates, 'diagnostic': diagnostic})
    return updates


def native_resources_report(profiles, resources, snippets):
    expected = {p['node_id'] for p in profiles}
    nodes = {p['node_id']: {'initial_snippets': len(p['train_cases']),
        'sample_ids': [c['sample_id'] for c in p['train_cases']], 'train_turns': p['train_turns'],
        'distinct_slot_patterns': p['distinct_slot_patterns'], 'inventory_slot_shapes': sum(len(s['value_modes']) for s in p['slot_schema']),
        'action_rule_sha256': digest(resources['action_rules'][p['node_id']]),
        'slot_policy_sha256': digest(resources['slot_policies'][p['node_id']])} for p in profiles}
    return {'protocol': OFFLINE_PROTOCOL, 'initial_snippets_per_node': snippets,
        'profile_sha256': digest(profiles), 'compiled_resources_sha256': digest(resources),
        'coverage_kind': 'Native resource completeness/provenance; policy semantics are not certified by schema coverage',
        'nodes': nodes, 'verified': all(expected == set(resources[role]) for role in ('action_rules', 'slot_policies'))}


def native_reference(profiles, resources):
    notes = [f"## Slot reference {key}\n{text}" for key, text in resources.get('reference', {}).items()]
    return '\n\n'.join([initial_reference_text(profiles), *notes])


class NativeMultiWOZSkillAgent(ConditionalMultiWOZSkillAgent):
    protocol = PROTOCOL
    reference_session_class = GlobalReferenceSession
    selection_card_projection = staticmethod(deepcopy)
    payload_card_projection = staticmethod(deepcopy)
    payload_action_rules_included = True

    def runtime_contract(self):
        contract = MultiWOZAlignedSkillAgent.runtime_contract(self)
        contract.update(protocol=PROTOCOL, offline_slot_protocol=OFFLINE_PROTOCOL,
            candidate_transport='validated_complete_head_bundles_or_exact_catalog_ids',
            payload='fixed_heads_complete_selected_card_and_agent_reference', payload_action_rules_included=True,
            reference_scope='whole_library_agent_controlled', slot_policy_format='native_free_text',
            slot_policy_updates='existing_standard_diagnosis_reflection_native_upsert')
        return contract

    def mine(self, dialogues, directory, *, cases_per_node=4, **kwargs):
        if cases_per_node != 4:
            raise ValueError('This run retains four initial TRAIN snippets per node')
        directory = Path(directory)
        SGDStandardSkillAgent.mine(self, dialogues, directory, cases_per_node=4, compile_rules=False)
        # Restore the pre-probe snippet selection, not the contrast-selection intervention.
        profiles = training_profiles(dialogues, self.graph, self.ontology, 4, self.observations)
        for profile in profiles:
            profile['offline_slot_protocol'] = OFFLINE_PROTOCOL
        atomic_json(directory / 'train_card_profiles.json', profiles)
        jobs = [(directory / 'card_compile' / f"batch_{p['node_id'].split(':')[-1]}.json", [p], self.model)
                for p in profiles]
        compiled = {'action_rules': {}, 'slot_policies': {}, 'reference': {}}
        pending = jobs
        for attempt in range(1, 4):
            failed = []
            outcomes = ordered_cases(compile_native_resources_job, pending, workers=self.parallel_workers,
                                     directory=directory, phase='offline_compile')
            for outcome, job in zip(outcomes, pending):
                try:
                    for update in require_value(outcome):
                        compiled[update['resource']][update['key']] = update['text']
                except Exception as exc:
                    failed.append(job)
                    with (directory / 'compile_retry_history.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps({'attempt': attempt, 'path': str(job[0]), 'error': str(exc)}) + '\n')
                atomic_json(directory / 'compile_progress.json', {'completed': len(compiled['slot_policies']),
                    'total': len(profiles), 'attempt': attempt})
            if not failed:
                break
            pending = failed
        if failed:
            raise RuntimeError(f'{len(failed)} native resource jobs still failed; successes are cached for --resume')
        self.offline_slot_audit = native_resources_report(profiles, compiled, 4)
        if not self.offline_slot_audit['verified']:
            raise ValueError('Missing native action/slot resource')
        atomic_json(directory / 'compiled_native_resources.json', compiled)
        self.resources['action_rules'].update(compiled['action_rules'])
        self.resources['slot_policies'].update(compiled['slot_policies'])
        self.base_action_rules = '\n\n'.join(f'## {k}\n{v}' for k, v in compiled['action_rules'].items())
        self.base_slot_policies = '\n\n'.join(f'## {k}\n{v}' for k, v in compiled['slot_policies'].items())
        self.base_reference = native_reference(profiles, compiled)
        self.cards_ready = True
        self.working_skill = self._seed_skill()
        self.save(directory)
