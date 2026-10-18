"""Frozen, guideline-derived structural probes, not a business-compliance oracle.

Positive probes are nominal KB adjacencies plus explicitly sourced shortcuts.
Negative probes reverse an order occurring in the nominal reference routes,
only when the opposite order never occurs in any route of that coarse flow.
Unmentioned pairs are unknown, not negative. Retries and cross-task composition
can legitimately reverse an order: this metric is a DEVELOPMENT skeleton proxy.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json


PROTOCOL = "abcd_sop_direction_probes_v2"
UNORDERED = {
    "credit_card": ("Invalid Credit Card", "Choose these in any order."),
    "shopping_cart": ("Cart Not Updating", "Choose actions in any order you want."),
    "search_results": ("Search Not Working", "Shuffle up the order in which you choose these actions."),
    "slow_speed": ("Website Too Slow", "Pick these actions in any order you wish."),
}
SHORTCUTS = (
    ("purchase_dispute", "Promo Code Invalid", "Ask the Oracle", "ask-the-oracle", "promo-code",
     "If the customer is right (Oracle says Yes), skip to the Promo Code step below"),
    ("purchase_dispute", "Promo Code Out of Date", "Ask the Oracle", "ask-the-oracle", "promo-code",
     "If the customer is right (Oracle says Yes), skip to the Promo Code step below"),
    ("order_issue", "Status Mystery Fee", "Ask the Oracle", "ask-the-oracle", "update-order",
     "If this was a company mistake (Oracle says Yes), skip the next step"),
    ("order_issue", "Status Delivery Time", "Ask the Oracle", "ask-the-oracle", "update-order",
     "If this was a company mistake (Oracle says Yes), skip the next step"),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def build_reference(kb, ontology, guidelines, *, include_shortcuts=True):
    """Only reference resources are inputs; neither train nor test influences probes."""
    normalize = lambda name: name.lower().replace("-", " ").replace(" ", "_")
    guideline_flows = {normalize(name): value for name, value in guidelines.items()}
    result = {}
    for flow, fine_names in ontology["intents"]["subflows"].items():
        routes, positives, precedences = {}, defaultdict(list), set()
        for fine in fine_names:
            if fine not in kb:
                raise ValueError(f"Reference missing KB route: {flow}/{fine}")
            sequence = list(kb[fine])
            if not sequence or any(not isinstance(a, str) or not a for a in sequence):
                raise ValueError(f"Invalid KB sequence: {fine}")
            routes[fine] = sequence
            for index, (source, target) in enumerate(zip(sequence, sequence[1:])):
                positives[source, target].append({"source_file": "kb.json", "fine": fine, "step": index})
            if flow == "troubleshoot_site" and fine in UNORDERED:
                title, quote = UNORDERED[fine]
                instructions = guideline_flows[flow]["subflows"][title]["instructions"]
                if not any(quote in text for text in instructions):
                    raise ValueError(f"Unordered-rule provenance changed: {flow}/{title}")
                for source in sequence:
                    for target in sequence:
                        if source != target:
                            positives[source, target].append({"source_file": "guidelines.json",
                                "fine_title": title, "unordered_quote": quote})
                            precedences.add((source, target))
            else:
                precedences.update((a, b) for i, a in enumerate(sequence) for b in sequence[i+1:] if a != b)
        if include_shortcuts:
            for coarse, fine_title, button, source, target, quote in SHORTCUTS:
                if coarse != flow:
                    continue
                steps = guideline_flows[flow]["subflows"][fine_title]["actions"]
                if not any(step["button"] == button and quote in step.get("subtext", []) for step in steps):
                    raise ValueError(f"Shortcut provenance changed: {flow}/{fine_title}")
                positives[source, target].append({"source_file": "guidelines.json", "fine_title": fine_title,
                                                   "button": button, "condition_quote": quote})
        negatives = {(b, a) for a, b in precedences
                     if (b, a) not in precedences and (b, a) not in positives}
        if not positives or set(positives) & negatives:
            raise ValueError(f"Unusable/disagreeing reference probes: {flow}")
        result[flow] = {"routes": routes,
            "positive": [{"source": a, "target": b, "evidence": evidence}
                         for (a, b), evidence in sorted(positives.items())],
            "reverse_order": [{"source": a, "target": b,
                               "forward_order_witnesses": [fine for fine, seq in routes.items()
                                   if b in seq and a in seq and seq.index(b) < seq.index(a)]}
                              for a, b in sorted(negatives)]}
    return {"protocol": PROTOCOL, "source_hashes": {"kb": digest(kb), "ontology": digest(ontology),
            "guidelines": digest(guidelines)}, "include_shortcuts": include_shortcuts,
            "limits": "Nominal-order structural proxy, NOT certified illegal transitions. No guards, retries or call/return oracle.",
            "flows": result}


def retained_pairs(graph):
    """Evaluate executable local transitions, never accidentally the full DFG."""
    actions = {str(node["id"]): node.get("action", str(node["id"]).split(":", 1)[-1])
               for node in graph.get("nodes", [])}
    return {(actions.get(str(edge["source"]), str(edge["source"]).split(":", 1)[-1]),
             actions.get(str(edge["target"]), str(edge["target"]).split(":", 1)[-1]))
            for edges in graph.get("local_transitions", {}).values() for edge in edges
            if str(edge["source"]) != "<START>"}


def hmean(left, right):
    return 2 * left * right / (left + right) if left + right else 0.0


def score_reference(graph, reference):
    pairs = retained_pairs(graph)
    positives = {(row["source"], row["target"]) for row in reference["positive"]}
    negatives = {(row["source"], row["target"]) for row in reference["reverse_order"]}
    tp, fp = len(pairs & positives), len(pairs & negatives)
    recall = tp / len(positives)
    reject = 1 - fp / len(negatives) if negatives else None
    scored = positives | negatives
    routes = list(reference["routes"].values())
    complete = sum(all((a, b) in pairs for a, b in zip(seq, seq[1:])) for seq in routes)
    # A conventional F1 on the explicitly probed universe is also diagnostic;
    # the primary harmonic R/specificity is invariant to positive/negative ratio.
    return {"sop_discrimination": hmean(recall, reject) if reject is not None else recall,
            "score_mode": "balanced_recall_rejection" if negatives else "recall_only_no_order_constraints",
            "sop_recall": recall,
            "reverse_order_rejection": reject, "probe_precision": tp / (tp + fp) if tp + fp else 0.0,
            "positive_probes": len(positives), "reverse_probes": len(negatives),
            "positive_hits": tp, "reverse_order_admitted": fp,
            "nominal_route_acceptance": complete / len(routes),
            "nominal_routes": len(routes), "complete_nominal_routes": complete,
            "scored_retained_edges": len(pairs & scored), "unknown_retained_edges": len(pairs - scored),
            "edge_judgment_coverage": len(pairs & scored) / len(pairs) if pairs else None,
            "reference_extra_edges": len(positives - pairs),
            "missing_reference_edges": [list(edge) for edge in sorted(positives - pairs)],
            "admitted_reverse_edges": [list(edge) for edge in sorted(pairs & negatives)]}


def aggregate_reference(rows):
    """Domain weights are train action-turn counts; do not use test to select graphs."""
    weight = sum(row["train_action_turns"] for row in rows)
    keys = ("sop_discrimination", "sop_recall", "reverse_order_rejection", "nominal_route_acceptance")
    averages = {}
    for key in keys:
        valid = [row for row in rows if row[key] is not None]
        denominator = sum(row["train_action_turns"] for row in valid)
        averages[key] = sum(row[key] * row["train_action_turns"] for row in valid) / denominator if denominator else None
    return {**averages, "weight_unit": "train_action_turns", "weight": weight,
            "num_flows": len(rows),
            "macro_sop_discrimination": sum(row["sop_discrimination"] for row in rows) / len(rows) if rows else None,
            "positive_probes": sum(row["positive_probes"] for row in rows),
            "reverse_probes": sum(row["reverse_probes"] for row in rows),
            "flows_with_order_negatives": sum(bool(row["reverse_probes"]) for row in rows),
            "recall_only_weight_fraction": sum(row["train_action_turns"] for row in rows if not row["reverse_probes"]) / weight if weight else None,
            "positive_hits": sum(row["positive_hits"] for row in rows),
            "reverse_order_admitted": sum(row["reverse_order_admitted"] for row in rows)}
