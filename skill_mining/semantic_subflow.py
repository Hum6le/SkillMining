"""Session-aware latent subflow discovery and LLM semantic grounding."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval_tod.abcd.action_schema import canonical_action_name, load_action_schema


def _actions(conv: dict[str, Any]) -> list[str]:
    result = []
    action_schema = load_action_schema()
    for turn in conv.get("delexed") or []:
        targets = turn.get("targets") or []
        if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
            action, _ = canonical_action_name(targets[2], action_schema.get("actions"))
            if action:
                result.append(action)
    return result


def _signature(conv: dict[str, Any]) -> set[str]:
    actions = _actions(conv)
    # A session signature needs both action-presence and transition evidence.
    # Keeping only edges makes a session disappear after a shared routing node
    # is removed, even when its remaining actions still distinguish it.
    nodes = {f"node:{action}" for action in actions}
    edges = {f"{source}=>{target}" for source, target in zip(actions, actions[1:])}
    return nodes | edges


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(len(left | right), 1)


def _edge_nodes(edge: str) -> tuple[str, str]:
    if "=>" not in edge:
        node = edge.removeprefix("node:")
        return node, node
    source, target = edge.split("=>", 1)
    return source, target


def _residual_components(edges: set[str]) -> list[set[str]]:
    """Return weak components of a directed residual transition graph."""
    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source, target = _edge_nodes(edge)
        if source == target:
            neighbors.setdefault(source, set())
            continue
        neighbors[source].add(target)
        neighbors[target].add(source)
    components = []
    unseen = set(neighbors)
    while unseen:
        seed = min(unseen)
        queue = [seed]
        component = set()
        unseen.remove(seed)
        while queue:
            node = queue.pop()
            component.add(node)
            for neighbor in neighbors[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _partition_after_node_removal(
    records: list[dict[str, Any]], removed_nodes: set[str], max_skills: int,
    min_sessions: int,
) -> dict[str, Any] | None:
    """Cluster sessions by residual action and transition signatures."""
    residual_by_session = {}
    all_edges: set[str] = set()
    for row in records:
        sid = str(row["conversation"].get("convo_id", "?"))
        residual = {
            edge for edge in row["signature"]
            if not any(node in removed_nodes for node in _edge_nodes(edge))
        }
        residual_by_session[sid] = residual
        all_edges |= residual
    if not all_edges:
        return None

    # Build overlapping candidate groups from residual graph features. A
    # feature is either an action-presence token or a transition edge; its
    # signature is the set of features that recur in sessions carrying it.
    feature_support = Counter(feature for signature in residual_by_session.values() for feature in signature)
    candidates = []
    for seed, seed_support in feature_support.most_common():
        seed_members = [
            row for row in records
            if seed in residual_by_session[str(row["conversation"].get("convo_id", "?"))]
        ]
        if len(seed_members) < min_sessions:
            continue
        prevalence = Counter(
            edge for row in seed_members
            for edge in residual_by_session[str(row["conversation"].get("convo_id", "?"))]
        )
        threshold = max(2, int(0.2 * len(seed_members)))
        group_features = {feature for feature, count in prevalence.items() if count >= threshold}
        group_features.add(seed)
        if not group_features:
            continue
        if any(_jaccard(group_features, group["features"]) > 0.92 for group in candidates):
            continue
        candidates.append({
            "seed": seed, "seed_support": seed_support,
            "features": group_features, "seed_members": seed_members,
        })
        if len(candidates) >= max_skills * 3:
            break
    if len(candidates) < 2:
        return None

    # Select the most useful candidate motifs by support while preserving
    # overlap. A session is assigned to the best-covered motif for downstream
    # mining, but top-2 coverage/margin remains in the artifact.
    assignments: dict[str, dict[str, Any]] = {}
    group_members: list[list[dict[str, Any]]] = [[] for _ in candidates]
    for row in records:
        sid = str(row["conversation"].get("convo_id", "?"))
        signature = residual_by_session[sid]
        if not signature:
            continue
        scores = [len(signature & group["features"]) / len(signature) for group in candidates]
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        best_index, second_index = order[0], order[1] if len(order) > 1 else None
        if scores[best_index] <= 0:
            continue
        assignments[sid] = {
            "group_index": best_index,
            "coverage": round(scores[best_index], 6),
            "margin": round(scores[best_index] - (scores[second_index] if second_index is not None else 0.0), 6),
            "candidate_coverages": [round(score, 6) for score in scores],
        }
        group_members[best_index].append(row)

    retained = []
    for index, group in enumerate(candidates):
        members = group_members[index]
        if len(members) < min_sessions:
            continue
        session_edges = [residual_by_session[str(row["conversation"].get("convo_id", "?"))] for row in members]
        pair_scores = [_jaccard(left, right) for i, left in enumerate(session_edges) for right in session_edges[i + 1:]]
        retained.append({
            **group, "members": members, "support_sessions": len(members),
            "support_ratio": len(members) / max(len(records), 1),
            "cohesion": sum(pair_scores) / max(len(pair_scores), 1),
            "source_index": index,
        })
    retained.sort(key=lambda group: (-group["support_sessions"], group["seed"]))
    retained = retained[:max_skills]
    if len(retained) < 2:
        return None

    retained_indices = {group["source_index"] for group in retained}
    final_assignments = {sid: value for sid, value in assignments.items() if value["group_index"] in retained_indices}

    # Score every retained group independently before aggregating. This avoids
    # mixing a strong group's support with another group's cohesion via a
    # product of global averages.
    group_objectives = []
    group_overlaps = []
    for index, group in enumerate(retained):
        overlaps = [
            _jaccard(group["features"], other["features"])
            for other_index, other in enumerate(retained)
            if other_index != index
        ]
        mean_group_overlap = sum(overlaps) / max(len(overlaps), 1)
        group_overlaps.append(mean_group_overlap)
        group_objectives.append(
            group["support_ratio"] / (1.0 + mean_group_overlap)
        )
    mean_overlap = sum(group_overlaps) / max(len(group_overlaps), 1)
    mean_support = sum(group["support_ratio"] for group in retained) / len(retained)
    mean_cohesion = sum(group["cohesion"] for group in retained) / len(retained)
    objective = sum(group_objectives)
    return {
        "groups": retained,
        "session_assignments": final_assignments,
        "metrics": {
            "mean_support": mean_support,
            "mean_overlap": mean_overlap,
            "mean_cohesion": mean_cohesion,
            "objective": objective,
            "group_objectives": group_objectives,
            "group_mean_overlaps": group_overlaps,
            "assigned_sessions": len(final_assignments),
        },
    }


def discover_semantic_subflows(
    subflow: str, conversations: list[dict[str, Any]], max_skills: int = 4,
    min_sessions: int = 20, similarity_threshold: float = 0.35,
) -> dict[str, Any]:
    """Iteratively remove high-support shared nodes until split quality declines.

    The removal graph is only used to reveal session boundaries. The selected
    sessions retain their complete original trajectories for downstream skill
    mining, so removed nodes remain available as shared interfaces.
    """
    records = [{"conversation": c, "signature": _signature(c)} for c in conversations]
    records = [r for r in records if r["signature"]]
    removed_nodes: set[str] = set()
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    tolerance = 1e-6

    all_nodes = sorted({node for row in records for edge in row["signature"] for node in _edge_nodes(edge)})
    while len(removed_nodes) < max(len(all_nodes) - 1, 0):
        supports = Counter(
            node for row in records
            for node in {n for edge in row["signature"] for n in _edge_nodes(edge)}
            if node not in removed_nodes
        )
        if not supports:
            break
        # A frequent leaf/chain node may not expose a partition at all. Skip
        # such non-separating candidates and select the highest-support node
        # whose removal actually yields at least two supported components.
        node = ""
        support = 0
        candidate_removed: set[str] | None = None
        candidate = None
        for candidate_node, candidate_support in sorted(
            supports.items(), key=lambda item: (-item[1], item[0])
        ):
            proposed_removed = removed_nodes | {candidate_node}
            proposed = _partition_after_node_removal(
                records, proposed_removed, max_skills, min_sessions,
            )
            if proposed is not None:
                node = candidate_node
                support = candidate_support
                candidate_removed = proposed_removed
                candidate = proposed
                break
        if candidate is None or candidate_removed is None:
            break
        candidate_objective = candidate["metrics"]["objective"] if candidate else 0.0
        previous_objective = best["metrics"]["objective"] if best else 0.0
        accepted = candidate is not None and candidate_objective > previous_objective + tolerance
        history.append({
            "iteration": len(history) + 1, "removed_node": node,
            "node_session_support": support,
            "objective": candidate_objective,
            "previous_objective": previous_objective,
            "accepted": accepted,
            "metrics": candidate.get("metrics", {}) if candidate else {},
        })
        if not accepted:
            break
        removed_nodes = candidate_removed
        best = candidate

    if best is None:
        # No valid split is preferable to inventing unsupported latent skills.
        # Keep one fallback group so the surrounding pipeline remains runnable.
        best = {
            "groups": [{"nodes": set(all_nodes), "features": set().union(*(r["signature"] for r in records)),
                        "members": records, "support_sessions": len(records),
                        "support_ratio": 1.0, "cohesion": 1.0, "source_index": 0}],
            "session_assignments": {str(r["conversation"].get("convo_id", "?")): {"group_index": 0, "coverage": 1.0, "margin": 1.0} for r in records},
            "metrics": {"mean_support": 1.0, "mean_overlap": 0.0, "mean_cohesion": 1.0,
                        "objective": 0.0, "assigned_sessions": len(records)},
        }

    skills = []
    session_assignments = {}
    original_group_nodes: list[set[str]] = []
    original_group_edges: list[set[str]] = []
    for index, group in enumerate(best["groups"]):
        skill_id = f"skill_{index:02d}"
        members = group["members"]
        residual_nodes = sorted(
            feature.removeprefix("node:") for feature in group["features"]
            if feature.startswith("node:")
        )
        residual_edges = sorted(
            feature for feature in group["features"] if "=>" in feature
        )
        node_counts = Counter(a for r in members for a in _actions(r["conversation"]))
        edge_counts = Counter(e for r in members for e in r["signature"])
        original_group_nodes.append(set(node_counts))
        original_group_edges.append(set(edge_counts))
        examples = []
        for row in sorted(members, key=lambda r: -len(r["signature"]))[:4]:
            conv = row["conversation"]
            utterances = [str(t.get("text", "")) for t in conv.get("original", [])
                          if isinstance(t, dict) and t.get("speaker") == "customer" and t.get("text")]
            examples.append({"session_id": str(conv.get("convo_id", "?")),
                             "actions": _actions(conv), "customer_utterances": utterances[:3]})
        skills.append({
            "skill_id": skill_id, "seed_transition": group.get("seed", "residual_motif"),
            "support_sessions": group["support_sessions"],
            "support_ratio": round(group["support_ratio"], 6),
            "cohesion": round(group["cohesion"], 6),
            "residual_nodes": residual_nodes,
            "residual_edges": residual_edges,
            "nodes": [{"label": a, "support": n} for a, n in node_counts.most_common()],
            "edges": [{"transition": e, "support": n} for e, n in edge_counts.most_common()],
            "examples": examples,
        })
        for row in members:
            sid = str(row["conversation"].get("convo_id", "?"))
            details = best["session_assignments"].get(sid, {})
            session_assignments[sid] = {"skill_id": skill_id, **details}
    shared_nodes = set(removed_nodes)
    shared_edges: set[str] = set()
    for index, nodes in enumerate(original_group_nodes):
        for other_nodes in original_group_nodes[index + 1:]:
            shared_nodes |= nodes & other_nodes
    for index, edges in enumerate(original_group_edges):
        for other_edges in original_group_edges[index + 1:]:
            shared_edges |= edges & other_edges
    return {
        "protocol": "iterative_shared_node_splitting_v1", "subflow": subflow,
        "num_sessions": len(records), "skills": skills,
        "session_assignments": session_assignments,
        "removed_partition_nodes": sorted(removed_nodes),
        "shared_interface_nodes": sorted(shared_nodes),
        "shared_interface_edges": sorted(shared_edges),
        "objective_history": history,
        "final_metrics": best["metrics"],
        "similarity_threshold": similarity_threshold, "min_sessions": min_sessions,
    }


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _coarse_flow_prior(subflow: str) -> tuple[str, str]:
    """Return the official ABCD coarse-flow name and its short description."""
    guideline_path = Path(__file__).resolve().parents[1] / "data" / "eval" / "abcd" / "data" / "guidelines.json"
    if not guideline_path.exists():
        return subflow, "No official coarse-flow description is available."
    guidelines = json.loads(guideline_path.read_text(encoding="utf-8"))
    normalized = re.sub(r"[^a-z0-9]+", "_", subflow.lower()).strip("_")
    for name, details in guidelines.items():
        key = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
        if key == normalized:
            return str(name), str((details or {}).get("description", ""))
    return subflow, "No official coarse-flow description is available."


def _full_dialogue_text(conversation: dict[str, Any]) -> str:
    """Format the complete observable dialogue for semantic routing induction."""
    speaker_names = {"customer": "Customer", "agent": "Agent", "action": "System"}
    lines = []
    for turn in conversation.get("original") or conversation.get("delexed") or []:
        if isinstance(turn, dict):
            raw_speaker = turn.get("speaker", "")
            raw_text = turn.get("text", "")
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            # ABCD's original dialogue format is [speaker, utterance].
            raw_speaker, raw_text = turn[0], turn[1]
        else:
            continue
        text = str(raw_text).strip()
        if text:
            speaker_key = str(raw_speaker).strip().lower()
            speaker = speaker_names.get(speaker_key, str(raw_speaker or "Unknown"))
            lines.append(f"[{speaker}] {text}")
    return "\n".join(lines) or "[No dialogue text available]"


def _extract_card_list(raw: str) -> list[dict[str, Any]]:
    parsed = _extract_json(raw)
    if not parsed:
        return []
    cards = parsed.get("cards", [])
    return [item for item in cards if isinstance(item, dict)] if isinstance(cards, list) else []


def ground_skill_cards(
    discovery: dict[str, Any], conversations: list[dict[str, Any]],
    model: str = "deepseek-chat", prompt_path: Path | None = None,
) -> dict[str, Any]:
    """Jointly induce all routing cards from representative full dialogues."""
    by_id = {str(conv.get("convo_id", "?")): conv for conv in conversations}
    assignments = discovery.get("session_assignments", {})
    flow_name, flow_description = _coarse_flow_prior(str(discovery.get("subflow", "")))
    groups = []
    for skill in discovery.get("skills", []):
        skill_id = str(skill["skill_id"])
        ranked_members = []
        for session_id, assignment in assignments.items():
            if not isinstance(assignment, dict) or assignment.get("skill_id") != skill_id:
                continue
            conversation = by_id.get(str(session_id))
            if conversation is not None:
                ranked_members.append((float(assignment.get("coverage", 0.0)), str(session_id), conversation))
        ranked_members.sort(key=lambda item: (-item[0], item[1]))
        examples = [
            {
                "session_id": session_id,
                "group_coverage": round(coverage, 6),
                "dialogue": _full_dialogue_text(conversation),
            }
            for coverage, session_id, conversation in ranked_members[:4]
        ]
        groups.append({
            "skill_id": skill_id,
            "support_sessions": skill["support_sessions"],
            "residual_nodes": skill.get("residual_nodes", []),
            "residual_edges": skill.get("residual_edges", []),
            "representative_dialogues": examples,
        })

    prompt = (
        "You are inducing a mutually distinguishable routing taxonomy for latent dialogue skills.\n\n"
        f"<coarse_scenario>\nName: {flow_name}\nPrior: {flow_description}\n</coarse_scenario>\n\n"
        "The groups below were discovered from action-graph structure inside this single coarse scenario. "
        "They are not ground-truth labels. Read ALL groups jointly and infer the user intent/sub-scenario represented by each group. "
        "Your cards must make boundaries between groups explicit: distinguish groups with similar actions using the customer goal, "
        "request wording, and outcome. Do not use dataset labels, hidden state, or private values. Do not describe a group only by "
        "shared tool actions. Each card must be a useful routing specification, not a one-line summary: write a 2-4 sentence "
        "description, list several observable customer cues, state what this skill covers and does not cover, and explicitly compare "
        "it with the other supplied groups.\n\n"
        "Return JSON only with exactly one card per supplied skill_id: {\"cards\":[{\"skill_id\":\"...\","
        "\"name\":\"short intent name\",\"summary\":\"2-4 sentence routing description\","
        "\"customer_goals\":[\"underlying user goal\"],\"positive_evidence\":[\"observable cue\"],"
        "\"negative_evidence\":[\"observable cue that rules this skill out\"],"
        "\"distinguish_from\":[{\"skill_id\":\"other skill id\",\"rule\":\"how to choose between them\"}],"
        "\"typical_outcome\":\"what the user expects to achieve\","
        "\"boundary_uncertainty\":\"remaining ambiguity, if any\"}]}. "
        "Return exactly one card for every supplied skill_id, preserving its skill_id.\n\n"
        "<discovered_groups>\n" + json.dumps(groups, ensure_ascii=False, indent=2) + "\n</discovered_groups>"
    )
    if prompt_path is not None:
        prompt_path.write_text(prompt, encoding="utf-8")

    parsed_by_id: dict[str, dict[str, Any]] = {}
    try:
        from llm import chat
        raw = chat([
            {"role": "system", "content": "Return detailed, evidence-grounded JSON only. The cards will be shown directly to another model for skill routing."},
            {"role": "user", "content": prompt},
        ], model=model, temperature=0.0)
        parsed_by_id = {str(card.get("skill_id")): card for card in _extract_card_list(raw)}
    except Exception:
        pass

    cards = []
    for skill in discovery.get("skills", []):
        parsed = parsed_by_id.get(str(skill["skill_id"]), {})
        cards.append({"skill_id": skill["skill_id"],
                      "name": parsed.get("name", skill["skill_id"]),
                      "summary": parsed.get("summary", f"Workflow centered on {skill['seed_transition']}"),
                      "customer_goals": parsed.get("customer_goals", []),
                      "positive_evidence": parsed.get("positive_evidence", [skill["seed_transition"]]),
                      "negative_evidence": parsed.get("negative_evidence", []),
                      "avoid_when": parsed.get("avoid_when", []),
                      "distinguish_from": parsed.get("distinguish_from", []),
                      "typical_outcome": parsed.get("typical_outcome", "unknown"),
                      "boundary_uncertainty": parsed.get("boundary_uncertainty", "unknown"),
                      "support_sessions": skill["support_sessions"],
                      "distinctive_edges": skill["edges"][:8]})
    discovery["skill_cards"] = cards
    return discovery


def format_skill_cards(discovery: dict[str, Any]) -> str:
    lines = ["<available_skills>", "Choose only from these evidence-grounded cards."]
    for card in discovery.get("skill_cards", []):
        distinctions = card.get("distinguish_from", [])
        distinction_text = "; ".join(
            f"{item.get('skill_id', 'other')}: {item.get('rule', '')}"
            for item in distinctions if isinstance(item, dict)
        ) or "not specified"
        lines.extend([f"\n[{card['skill_id']}] {card['name']}",
                      f"Routing description: {card['summary']}",
                      "Customer goals: " + ("; ".join(map(str, card.get("customer_goals", []))) or "not specified"),
                      "Positive evidence: " + ("; ".join(map(str, card.get("positive_evidence", []))) or "not specified"),
                      "Negative evidence: " + ("; ".join(map(str, card.get("negative_evidence", []))) or "not specified"),
                      "Do not use this skill when: " + ("; ".join(map(str, card.get("avoid_when", []))) or "not specified"),
                      "Distinguish from other skills: " + distinction_text,
                      "Typical outcome: " + str(card.get("typical_outcome", "unknown")),
                      "Graph evidence: " + ", ".join(e["transition"] for e in card.get("distinctive_edges", []))])
    lines.append("</available_skills>")
    return "\n".join(lines)
