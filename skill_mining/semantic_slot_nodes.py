"""Selective action abstraction: split semantic enums, never raw entity values.

This is an opt-in graph-mining representation, not new backend API actions.
The explicit action/position enum policy is a prior, not learned supervision.
Unknown/free-text values map to the original action (no per-value nodes).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import math
import re

from eval_tod.abcd.action_schema import canonical_action_name, load_action_schema


ENUMS = {
    "update-order": {0: (
        "change order", "change address", "change item", "change time", "change date", "change price",
        "cancel order", "waive fee", "give credit", "by mail", "in store", "drop off center",
    )},
    "update-account": {0: ("add service", "remove service", "extend subscription", "renew subscription")},
    "shipping-status": {0: ("order received", "in transit", "out for delivery", "delivered")},
    "membership": {0: ("gold", "silver", "bronze", "guest")},
}
DEFAULT_ACTIONS = tuple(ENUMS)


def normalize_category(value):
    # Normalize formatting, not semantics: no fuzzy matching, substrings,
    # names/IDs inferred from frequency, or value-dependent new categories.
    return " ".join(re.sub(r"[_-]+", " ", str(value).strip().casefold()).split())


def make_policy(actions=DEFAULT_ACTIONS):
    if len(set(actions)) != len(actions) or set(actions) - set(ENUMS):
        raise ValueError("Semantic slot actions must be a unique subset of " + ",".join(ENUMS))
    return {"protocol": "selective_semantic_slot_nodes_v1", "position_base": 0,
            "actions": {a: {str(p): list(values) for p, values in ENUMS[a].items()} for a in actions},
            "unknown_value": "original_action", "policy_source": "explicit finite semantic-enum prior; not learned",
            "normalization": "case/whitespace/hyphen/underscore only; no fuzzy matching"}


def identity(action, slots, policy):
    """Return one stable internal identity and its reversible API-action metadata."""
    action, suffix = canonical_action_name(action, load_action_schema()["actions"])
    values = suffix + list(slots or [])
    signature = []
    for position, allowed in sorted(policy["actions"].get(action, {}).items(), key=lambda pair: int(pair[0])):
        index = int(position)
        category = normalize_category(values[index]) if index < len(values) else ""
        if category in allowed:
            signature.append({"position": index, "category": category})
    token = action + "".join(f"-mode-s{item['position']}-{item['category'].replace(' ', '-')}" for item in signature)
    return token, {"action": action, "semantic_slots": signature,
                   "display_label": action + "".join(f" [arg{item['position']+1}={item['category']}]" for item in signature)}


def encode_conversations(conversations, policy):
    """Copy targets for the miner only; preserve all slots and original dialogue text."""
    encoded, registry = copy.deepcopy(conversations), {}
    split_events = 0
    for conversation in encoded:
        for turn in conversation.get("delexed", []):
            targets = turn.get("targets") or []
            if len(targets) < 3 or targets[1] != "take_action" or not targets[2]:
                continue
            slots = targets[3] if len(targets) > 3 and isinstance(targets[3], list) else []
            token, metadata = identity(targets[2], slots, policy)
            # Legacy action:slot suffix is already copied into explicit slots,
            # so the downstream miner must not drop it during canonicalization.
            _, suffix = canonical_action_name(targets[2], load_action_schema()["actions"])
            if suffix:
                while len(targets) < 4:
                    targets.append([])
                targets[3] = suffix + slots
            targets[2] = token
            registry[token] = metadata
            split_events += bool(metadata["semantic_slots"])
    return encoded, registry, split_events


def mine_with_node_identity(miner, flow, conversations, *, node_identity="action", policy=None, **kwargs):
    """Wrap any existing graph miner; the default path is exactly unchanged.

    ``miner`` can return a subgraph directly or the conventional mining bundle.
    Other than its node identities, the wrapped miner/optimization is reused.
    """
    if node_identity == "action":
        return miner(flow, conversations, **kwargs)
    if node_identity != "semantic-slot":
        raise ValueError("node_identity must be action or semantic-slot")
    policy = policy or make_policy()
    encoded, registry, count = encode_conversations(conversations, policy)
    result = miner(flow, encoded, **kwargs)
    graph = result["subgraph"] if "subgraph" in result else result
    for node in graph["nodes"]:
        token = str(node["id"]).split(":", 1)[-1]
        metadata = registry[token]
        node.update(action=metadata["action"], semantic_slots=metadata["semantic_slots"],
                    label=metadata["display_label"])
    graph["node_identity"] = {"mode": node_identity, "policy": policy, "registry": registry,
                              "typed_action_events": count,
                              "split_actions": sorted({row["action"] for row in registry.values() if row["semantic_slots"]}),
                              "extra_nodes": len(registry) - len({row["action"] for row in registry.values()}),
                              "consumer_contract": "Internal nodes, NOT backend action names. Decode node.action; slots remain original. Graph-only branch; legacy runtime is not automatically adapted."}
    return result


def _steps(flow, conversation, policy):
    result = []
    for turn in conversation.get("delexed", []):
        targets = turn.get("targets") or []
        if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
            slots = targets[3] if len(targets) > 3 and isinstance(targets[3], list) else []
            token, metadata = identity(targets[2], slots, policy)
            result.append((f"{flow}:{token}", metadata["action"]))
    return result


def evaluate_common_action_targets(flow, graph, train, test, policy):
    """Identical next-API-action task/ranking/branch subset for both representations.

    At a transition, only the source (a PAST action and its observed slots) is
    used to select a node. The next action/slots never select the source state.
    Rank canonical next actions by summed retained training support. For an
    unseen source role, back off to all trained roles of the same API action.
    """
    registry = graph.get("node_identity", {}).get("registry", {})
    typed = graph.get("node_identity", {}).get("mode") == "semantic-slot"
    def base(node):
        token = str(node).split(":", 1)[-1]
        return registry.get(token, {}).get("action", token)
    full_targets = defaultdict(set)
    for conv in train:
        seq = _steps(flow, conv, policy)
        for left, right in zip(seq, seq[1:]):
            full_targets[left[1]].add(right[1])
    branching = {action for action, targets in full_targets.items() if len(targets) >= 2}
    local = graph.get("local_transitions", {})
    action_nodes = defaultdict(list)
    for node in graph.get("nodes", []):
        action_nodes[base(node["id"])].append(node["id"])
    counts = Counter()
    for conv in test:
        seq = _steps(flow, conv, policy)
        for left, right in zip(seq, seq[1:]):
            source = left[0] if typed else f"{flow}:{left[1]}"
            if source in action_nodes.get(left[1], []):
                sources = [source]
            else:
                sources = action_nodes.get(left[1], [])
                counts["source_backoffs"] += 1
            scores = Counter()
            for source_id in sources:
                for edge in local.get(source_id, []):
                    scores[base(edge["target"])] += max(int(edge.get("support", 0)), 1)
            candidates = sorted(scores, key=lambda action: (-scores[action], action))
            rank = candidates.index(right[1]) + 1 if right[1] in scores else 0
            counts["transitions"] += 1
            counts["hits"] += bool(rank)
            counts["top1_hits"] += rank == 1
            counts["reciprocal_rank_sum"] += 1 / rank if rank else 0
            counts["candidate_sum"] += len(candidates)
            if left[1] in branching:
                counts["branch_transitions"] += 1
                counts["branch_hits"] += bool(rank)
                counts["branch_reciprocal_rank_sum"] += 1 / rank if rank else 0
                counts["branch_candidate_sum"] += len(candidates)
    # Entropy of canonical successors conditional on the ORIGINAL vs TYPED
    # source. All full observed edges are used (not pruned to inflate purity).
    by_source = defaultdict(Counter)
    for edge in graph.get("edges", []):
        by_source[edge["source"]][base(edge["target"])] += int(edge.get("support", 0))
    entropy_sum = sum(-sum(n * math.log2(n / sum(dist.values())) for n in dist.values() if n)
                      for dist in by_source.values() if sum(dist.values()))
    train_transitions = sum(sum(dist.values()) for dist in by_source.values())
    return {**dict(counts), "training_transitions": train_transitions, "training_entropy_sum": entropy_sum,
            "training_conditional_entropy": entropy_sum / train_transitions if train_transitions else 0,
            **common_rates(counts)}


def common_rates(counts):
    n, b = counts.get("transitions", 0), counts.get("branch_transitions", 0)
    ratio = lambda key, denominator: counts.get(key, 0) / denominator if denominator else None
    return {"next_action_recall": ratio("hits", n), "next_action_top1": ratio("top1_hits", n),
            "next_action_mrr": ratio("reciprocal_rank_sum", n),
            "average_action_candidates": ratio("candidate_sum", n),
            "branch_recall": ratio("branch_hits", b), "branch_mrr": ratio("branch_reciprocal_rank_sum", b),
            "branch_average_candidates": ratio("branch_candidate_sum", b),
            "source_backoff_rate": ratio("source_backoffs", n)}
