"""SGD family projections, causal policy targets, and structural scoring."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Iterator


def iter_family_dialogues(root: Path, family: str, split: str) -> Iterator[dict]:
    path = root / family / f"{split}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def visible_context(dialogue: dict, turn_index: int) -> str:
    """Expose utterances before the target turn; never gold annotations."""
    return "\n".join(
        f"{turn['speaker'].upper()}: {turn.get('utterance', '')}"
        for turn in dialogue["turns"][:turn_index]
    )


def canonical_call(value: object) -> dict | None:
    if not isinstance(value, dict) or not value.get("service") or not value.get("method"):
        return None
    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    return {
        "service": str(value["service"]),
        "method": str(value["method"]),
        "parameters": {str(key): str(item) for key, item in sorted(parameters.items())},
    }


def canonical_acts(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    result = []
    for act in value:
        if not isinstance(act, dict) or not act.get("intent"):
            continue
        item = {
            "intent": str(act["intent"]).lower(),
            "domain": str(act.get("domain") or ""),
            "slot": str(act.get("slot") or ""),
        }
        if "value" in act:
            item["value"] = str(act["value"])
        result.append(item)
    return sorted(result, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))


def gold_policy(dialogue: dict, turn_index: int) -> dict:
    turn = dialogue["turns"][turn_index]
    family_services = set(dialogue["services"])
    calls = [
        canonical_call({"service": service, **call})
        for service, call in (turn.get("service_call") or {}).items()
        if service in family_services
    ]
    if len(calls) > 1:
        raise ValueError("SGD target turn has multiple family API calls")
    include_global = len({s.rsplit("_", 1)[0] for s in dialogue["all_services"]}) == 1
    acts = [
        act
        for group in (turn.get("dialogue_acts") or {}).values()
        for act in group
        if act.get("domain") in family_services or (include_global and not act.get("domain"))
    ]
    return {"call": calls[0] if calls else None, "acts": canonical_acts(acts)}


def evaluation_turn_indices(dialogue: dict, scope: str = 'policy') -> list[int]:
    """Select labeled benchmark targets, independently of model predictions.

    Prior non-CALL turns remain visible context. CALL-only selection never
    tells the predictor the gold API method, parameters, or current response.
    """
    if scope not in {'policy', 'call'}:
        raise ValueError('Evaluation scope must be policy or call')
    indices = dialogue['target_indices']['policy_turn_indices']
    return [index for index in indices
            if scope == 'policy' or gold_policy(dialogue, index)['call'] is not None]


def score_api_calls(gold: list[dict], predicted: list[dict]) -> dict:
    """Backend-only view: action=(service, method), slots=named parameters."""
    if len(gold) != len(predicted) or any(row['call'] is None for row in gold):
        raise ValueError('Backend CALL scoring requires paired gold CALL targets')
    action_correct = joint_correct = 0
    for truth, guess in zip(gold, predicted):
        expected, actual = truth['call'], guess['call']
        action_ok = (not guess.get('_execution_error') and actual is not None
            and (actual['service'], actual['method']) == (expected['service'], expected['method']))
        action_correct += int(bool(action_ok))
        joint_correct += int(bool(action_ok and actual == expected))
    return {'turns': len(gold), 'action_correct': action_correct, 'joint_correct': joint_correct,
        'api_action_exact': action_correct / len(gold) if gold else 0.0,
        'api_slot_exact_given_action': joint_correct / action_correct if action_correct else 0.0,
        'api_joint_ast': joint_correct / len(gold) if gold else 0.0,
        'action_definition': 'actual service ID plus API method',
        'slot_definition': 'complete named API parameter map; dialogue acts excluded'}


def action_counter(policy: dict) -> Counter:
    heads = Counter({head: 1 for head in {
        (act["intent"], act["domain"]) for act in policy["acts"]
    }})
    if policy["call"] is not None:
        heads[("call", policy["call"]["service"])] += 1
    return heads


def slot_signature(policy: dict) -> tuple:
    call = policy["call"]
    call_payload = None if call is None else (
        call["service"], call["method"], tuple(sorted(call["parameters"].items()))
    )
    acts = tuple(sorted(
        (act["intent"], act["domain"], act["slot"], "value" in act, act.get("value", ""))
        for act in policy["acts"]
    ))
    return call_payload, acts


def score_policies(gold: list[dict], predicted: list[dict]) -> dict:
    if len(gold) != len(predicted):
        raise ValueError("Gold and prediction counts differ")
    counts = Counter()
    for truth, guess in zip(gold, predicted):
        failed = bool(guess.get("_execution_error"))
        true_actions, pred_actions = action_counter(truth), action_counter(guess)
        common = sum((true_actions & pred_actions).values())
        counts["action_tp"] += common
        counts["action_gold"] += sum(true_actions.values())
        counts["action_pred"] += sum(pred_actions.values())
        action_ok = not failed and true_actions == pred_actions
        joint_ok = action_ok and slot_signature(truth) == slot_signature(guess)
        counts["turns"] += 1
        counts["execution_failed_turns"] += int(failed)
        counts["action_exact"] += int(action_ok)
        counts["joint_exact"] += int(joint_ok)
        counts["slot_exact_given_action"] += int(joint_ok)
        counts["slot_denominator"] += int(action_ok)
        true_call, pred_call = truth["call"], guess["call"]
        counts["call_tp"] += int(true_call is not None and pred_call is not None)
        counts["call_gold"] += int(true_call is not None)
        counts["call_pred"] += int(pred_call is not None)
        if true_call is not None:
            counts["api_joint_correct"] += int(not failed and true_call == pred_call)
    return metrics_from_counts(counts)


def metrics_from_counts(counts: Counter | dict) -> dict:
    counts = Counter(counts)
    def ratio(numerator: str, denominator: str) -> float:
        return counts[numerator] / counts[denominator] if counts[denominator] else 0.0
    def f1(prefix: str) -> float:
        denominator = counts[f"{prefix}_gold"] + counts[f"{prefix}_pred"]
        return 2 * counts[f"{prefix}_tp"] / denominator if denominator else 0.0
    return {
        "counts": dict(counts),
        "action_set_exact": ratio("action_exact", "turns"),
        "action_micro_f1": f1("action"),
        "slot_exact_given_action": ratio("slot_exact_given_action", "slot_denominator"),
        "turn_joint_ast": ratio("joint_exact", "turns"),
        "call_gate_f1": f1("call"),
        "api_joint_ast": ratio("api_joint_correct", "call_gold"),
    }


def score_slot_items(gold: list[dict], predicted: list[dict]) -> dict:
    """Additional named-slot diagnostics; leave strict turn scoring unchanged.

    Each act's named slot/value is one item, including value-field presence.
    Each API parameter is one item. Slotless acts and the API method are not
    slot items. Multiplicity matters. Report both all-turn and action-correct
    subsets, since conditional accuracy alone has a changing denominator.
    """
    if len(gold) != len(predicted):
        raise ValueError('Gold and prediction counts differ')

    def items(policy):
        result = Counter((('act', a['intent'], a['domain'], a['slot']),
                          'value' in a, a.get('value', ''))
                         for a in policy['acts'] if a['slot'])
        call = policy['call']
        if call:
            result.update((('parameter', call['service'], call['method'], key), True, value)
                          for key, value in call['parameters'].items())
        return result

    def metrics(pairs):
        tp = ng = np = matched_names = 0
        for truth, guess in pairs:
            expected, actual = items(truth), items(guess)
            if guess.get('_execution_error'):
                actual = Counter()
            tp += sum((expected & actual).values())
            ng += sum(expected.values())
            np += sum(actual.values())
            en, an = Counter(), Counter()
            for (name, _, _), number in expected.items(): en[name] += number
            for (name, _, _), number in actual.items(): an[name] += number
            matched_names += sum((en & an).values())
        return {'tp': tp, 'gold': ng, 'predicted': np, 'matched_slot_names': matched_names,
            'precision': tp / np if np else 0., 'recall': tp / ng if ng else 0.,
            'micro_f1': 2 * tp / (ng + np) if ng + np else 0.,
            'slot_name_micro_f1': 2 * matched_names / (ng + np) if ng + np else 0.,
            'value_exact_given_matched_slot_name': tp / matched_names if matched_names else 0.}
    paired = list(zip(gold, predicted))
    correct = [(g, p) for g, p in paired if not p.get('_execution_error')
               and action_counter(g) == action_counter(p)]
    return {'definition': 'Named act slots plus API parameters; multiset; exact strings and value presence',
            'all_turns': metrics(paired), 'action_correct_turns': len(correct),
            'given_correct_actions': metrics(correct)}
