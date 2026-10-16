"""One frozen resource pool for the two flat K1 access conditions.

No LLM calls or semantic rewriting. Parse once, merge cards once, and use the
same renderer for all-open context and native retrieval. Ungrouped preambles
and unassigned skill policies are shared always-visible constraints.
"""
from __future__ import annotations

import re

from eval_tod.abcd.resource_rendering import render_action_card, render_reference_unit
from .k1_flatten import RESOURCE_NAMES, digest, graph_flat_progressive_resources

VERSION = "k1-shared-resource-pool-v1"


def unkeyed_resource_blocks(text: str, name: str) -> list[str]:
    """Conservatively retain text outside explicit native resource sections.

    Top-level headings between cards can introduce global constraints too;
    they must not accidentally belong only to the previous action's card.
    Native card/reference bodies themselves remain unchanged in the pool.
    """
    blocks, pending = [], []
    keyed = False
    fence = ""

    def flush():
        if "".join(pending).strip():
            blocks.append("".join(pending).strip())
        pending.clear()

    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            if not fence:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = ""
            if not keyed:
                pending.append(line)
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line) if not fence else None
        if heading:
            level, title = len(heading[1]), heading[2]
            if name == "reference.md":
                starts_key = level == 2
                starts_global = level == 1
            else:
                starts_key = level == 4 and bool(re.fullmatch(r"`[^`]+`", title))
                starts_global = level <= 4 and not starts_key
            if starts_key:
                flush()
                keyed = True
            elif starts_global:
                keyed = False
        if not keyed:
            pending.append(line)
    flush()
    return blocks


def build_resource_pool(original: dict[str, str], flat: dict) -> dict:
    from eval_tod.abcd.agent import _parse_action_policy_sections, _parse_reference_sections

    progressive = graph_flat_progressive_resources(original, flat)
    cards: dict[str, dict[str, str]] = {}
    # Same source order and deduplication as the previous native card reader.
    for role, name in (("action_rule", "action_rules.md"), ("slot_policy", "slot_policies.md")):
        for text in (original[name], progressive["inline_policies"][name]):
            for row in _parse_action_policy_sections(text):
                card = cards.setdefault(row["action"], {})
                previous = card.get(role, "")
                if row["body"] not in previous:
                    card[role] = (previous + "\n\n" + row["body"]).strip()

    parsed_references = _parse_reference_sections(original["reference.md"])
    references, seen = [], set()
    for section in parsed_references:
        identity = digest(section)
        if identity in seen:
            continue  # Only exact duplicate units, never merge different evidence.
        seen.add(identity)
        section["body_hash"] = digest(section["body"])
        section["unit_id"] = "R-" + digest(section)[:20]
        references.append(section)
    card_ids = {action: "A-" + digest({"action": action, "card": card})[:20]
                for action, card in cards.items()}

    graph = progressive["resources"]["skill.md"]
    globals_ = []
    # Native parsers do not expose unkeyed constraints independently. Keep
    # them verbatim and identically visible in both access conditions.
    for name in ("action_rules.md", "slot_policies.md", "reference.md"):
        for text in unkeyed_resource_blocks(original[name], name):
            globals_.append({"source": name, "scope": "unkeyed resource constraints", "text": text})

    rendered_cards = {action: render_action_card(action, card) for action, card in cards.items()}
    rendered_references = [render_reference_unit(r.get("transition_title") or r["title"], r["body"])
                           for r in references]
    existing_text = "\n".join([graph, *rendered_cards.values(), *rendered_references,
                                *(r["text"] for r in globals_)])
    # Conservative fallback for skill-local action/slot prose that cannot be
    # assigned to the native inline-card parser. Never make it full-only.
    for node in flat["records"]["actions"]:
        for role in ("action_rule", "slot_policy"):
            for record in node[role]:
                if record["origin"]["file"] == "skill.md" and record["text"] not in existing_text:
                    globals_.append({"source": "skill.md", "scope": record.get("scope", ""),
                                     "text": record["text"]})
                    existing_text += "\n" + record["text"]
    for record in globals_:
        record["unit_id"] = "G-" + digest(record)[:20]
    global_text = "\n\n".join(f"## SHARED_GLOBAL {r['unit_id']}\nSource: {r['source']}\n"
                               f"Scope: {r['scope']}\n{r['text']}" for r in globals_)
    main_context = graph.rstrip() + ("\n\n" + global_text if global_text else "") + "\n"
    cards_text = "\n".join(rendered_cards[action] for action in sorted(cards))
    references_text = "\n".join(rendered_references)
    full_context = (main_context.rstrip() + "\n\n<all_action_cards>\n" + cards_text
                    + "\n</all_action_cards>\n\n<all_reference_sections>\n" + references_text
                    + "\n</all_reference_sections>\n")
    canonical = {"version": VERSION, "graph_context": graph, "global_rules": globals_,
                 "action_cards": cards, "reference_sections": references}
    units = [{"unit_id": card_ids[a], "kind": "action_card", "key": a, "text_hash": digest(rendered_cards[a])}
             for a in sorted(cards)]
    units += [{"unit_id": r["unit_id"], "kind": "reference", "key": r.get("transition_title") or r["title"],
               "text_hash": digest(text)} for r, text in zip(references, rendered_references)]
    units += [{"unit_id": r["unit_id"], "kind": "global", "key": r["source"], "text_hash": digest(r["text"])}
              for r in globals_]
    manifest = {"version": VERSION, "pool_hash": digest(canonical), "units": units,
                "source_hashes": {name: digest(original[name]) for name in RESOURCE_NAMES},
                "graph_hash": digest(graph), "shared_main_context_hash": digest(main_context),
                "full_context_hash": digest(full_context),
                "counts": {"action_cards": len(cards), "reference_sections": len(references), "global_rules": len(globals_)},
                "exact_duplicate_reference_units_removed": len(parsed_references) - len(references),
                "knowledge_equivalence": "Identical parsed units and rendered unit bodies; only exposure differs. This is not a proof that arbitrary source Markdown was semantically interpreted correctly."}
    return {"canonical": canonical, "manifest": manifest, "action_cards": cards,
            "action_card_ids": card_ids, "reference_sections": references, "main_context": main_context,
            "global_context": global_text, "cards_text": cards_text, "references_text": references_text,
            "full_context": full_context, "progressive": progressive}
