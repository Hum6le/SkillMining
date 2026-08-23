"""Session-aware latent subflow discovery and LLM semantic grounding."""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


def _actions(conv: dict[str, Any]) -> list[str]:
    result = []
    for turn in conv.get("delexed") or []:
        targets = turn.get("targets") or []
        if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
            result.append(str(targets[2]).strip())
    return result


def _signature(conv: dict[str, Any]) -> set[str]:
    actions = _actions(conv)
    edges = {f"{a}=>{b}" for a, b in zip(actions, actions[1:])}
    return edges or {f"node:{a}" for a in actions}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(len(left | right), 1)


def discover_semantic_subflows(
    subflow: str, conversations: list[dict[str, Any]], max_skills: int = 4,
    min_sessions: int = 20, similarity_threshold: float = 0.35,
) -> dict[str, Any]:
    """Find session-supported, potentially overlapping transition regions."""
    records = [{"conversation": c, "signature": _signature(c)} for c in conversations]
    records = [r for r in records if r["signature"]]
    edge_support = Counter(e for row in records for e in row["signature"])
    clusters: list[dict[str, Any]] = []
    for seed, _ in edge_support.most_common():
        if len(clusters) >= max_skills:
            break
        members = [r for r in records if seed in r["signature"]]
        if len(members) < min_sessions:
            continue
        edges = set().union(*(r["signature"] for r in members))
        if any(_jaccard(edges, c["edges"]) > 0.92 for c in clusters):
            continue
        clusters.append({"seed": seed, "members": members, "edges": edges})
    if not clusters:
        clusters = [{"seed": "all", "members": records, "edges": set().union(*(r["signature"] for r in records))}]

    assignments = {}
    for row in records:
        sid = str(row["conversation"].get("convo_id", "?"))
        _, index = max(((_jaccard(row["signature"], c["edges"]), i) for i, c in enumerate(clusters)), default=(0.0, 0))
        assignments[sid] = f"skill_{index:02d}"

    skills = []
    for index, cluster in enumerate(clusters):
        members = cluster["members"]
        node_counts = Counter(a for r in members for a in _actions(r["conversation"]))
        edge_counts = Counter(e for r in members for e in r["signature"])
        examples = []
        for row in sorted(members, key=lambda r: -len(r["signature"]))[:4]:
            conv = row["conversation"]
            utterances = [str(t.get("text", "")) for t in conv.get("original", [])
                          if isinstance(t, dict) and t.get("speaker") == "customer" and t.get("text")]
            examples.append({"session_id": str(conv.get("convo_id", "?")),
                             "actions": _actions(conv), "customer_utterances": utterances[:3]})
        skills.append({
            "skill_id": f"skill_{index:02d}", "seed_transition": cluster["seed"],
            "support_sessions": len(members),
            "nodes": [{"label": a, "support": n} for a, n in node_counts.most_common()],
            "edges": [{"transition": e, "support": n} for e, n in edge_counts.most_common()],
            "examples": examples,
        })
    return {"protocol": "session_transition_motif_discovery_v1", "subflow": subflow,
            "num_sessions": len(records), "skills": skills,
            "session_assignments": assignments, "similarity_threshold": similarity_threshold,
            "min_sessions": min_sessions}


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def ground_skill_cards(discovery: dict[str, Any], model: str = "deepseek-chat") -> dict[str, Any]:
    """Use the LLM for names and semantic boundaries, with deterministic fallback."""
    from llm import chat
    cards = []
    for skill in discovery.get("skills", []):
        prompt = ("Ground this router card only in the supplied graph and dialogue evidence. "
                  "Do not invent hidden state or hard-code values. Return JSON with keys "
                  "name, summary, positive_evidence, avoid_when, boundary_uncertainty.\n" +
                  json.dumps(skill, ensure_ascii=False)[:12000])
        try:
            raw = chat([{"role": "system", "content": "Return concise evidence-grounded JSON only."},
                        {"role": "user", "content": prompt}], model=model, temperature=0.0)
            parsed = _extract_json(raw) or {}
        except Exception:
            parsed = {}
        cards.append({"skill_id": skill["skill_id"],
                      "name": parsed.get("name", skill["skill_id"]),
                      "summary": parsed.get("summary", f"Workflow centered on {skill['seed_transition']}"),
                      "positive_evidence": parsed.get("positive_evidence", [skill["seed_transition"]]),
                      "avoid_when": parsed.get("avoid_when", []),
                      "boundary_uncertainty": parsed.get("boundary_uncertainty", "unknown"),
                      "support_sessions": skill["support_sessions"],
                      "distinctive_edges": skill["edges"][:8]})
    discovery["skill_cards"] = cards
    return discovery


def format_skill_cards(discovery: dict[str, Any]) -> str:
    lines = ["<available_skills>", "Choose only from these evidence-grounded cards."]
    for card in discovery.get("skill_cards", []):
        lines.extend([f"\n[{card['skill_id']}] {card['name']}",
                      f"Summary: {card['summary']}",
                      "Positive evidence: " + ", ".join(map(str, card.get("positive_evidence", []))),
                      "Avoid when: " + (", ".join(map(str, card.get("avoid_when", []))) or "not specified"),
                      "Distinctive transitions: " + ", ".join(e["transition"] for e in card.get("distinctive_edges", []))])
    lines.append("</available_skills>")
    return "\n".join(lines)
