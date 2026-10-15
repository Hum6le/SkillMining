"""AWM file-edit interface using the same exact-text applicator as ours.

No graph diagnosis, reflection or routing semantics are used here.
"""
from __future__ import annotations
import json
import re
from difflib import SequenceMatcher

EDITABLE_FILES = ("skill.md", "action_rules.md", "slot_policies.md")


class AWMEditError(ValueError):
    def __init__(self, message, repair_context=None):
        super().__init__(message)
        self.repair_context = repair_context or {}


def anchor_repair_context(source, anchor, positions, occurrence):
    """Suggest literal current-file excerpts; NEVER use similarity for writing."""
    candidates = []
    if positions:
        for number, position in enumerate(positions[:6], 1):
            candidates.append({"occurrence": number,
                "surrounding_text": source[max(0, position - 180):position + min(len(anchor), 400) + 180]})
    else:
        lines = [(m.start(), m.group(0)) for m in re.finditer(r"(?m)^.+$", source) if m.group(0).strip()]
        ranked = sorted(lines, key=lambda item: SequenceMatcher(None, anchor[:1000], item[1][:1000], autojunk=False).ratio(), reverse=True)
        seen = set()
        for position, line in ranked:
            snippet = line[:600]
            if snippet in seen:
                continue
            seen.add(snippet)
            candidates.append({"match_text": snippet, "exact_match_count": source.count(snippet),
                               "line": source[:position].count("\n") + 1})
            if len(candidates) == 4:
                break
    return {"exact_match_count": len(positions), "requested_occurrence": occurrence,
            "anchor_preview": anchor[:300], "literal_candidates": candidates,
            "instruction": "These are current-file excerpts for explicit reselection, not applied replacements. Copy exact text, preserve intent, and provide occurrence when repeated."}

TOOL_INSTRUCTIONS = '''
## File Editing Tool: apply_text_edits (output override)
Do NOT output a complete rewritten workflow or a dump of resources.
In this single induction, propose a concise edit plan for the current files.
Writable files: skill.md (Existing Workflow), action_rules.md, slot_policies.md
(Current Resources). reference.md, action schema and adapter-owned instructions
are READ-ONLY. Use only the current training feedback, not test examples.

Tool arguments: return exactly one JSON object with an "operations" array.
Each operation must specify:
- file: exactly one writable filename, never an absolute path.
- section (recommended metadata): heading containing the edit, e.g. "## Routing
  Policies" or "#### `send-link`". The exact anchor/occurrence determines the
  location; the tool infers the actual heading. A section hint is not an anchor.
- op: replace, insert_before, insert_after, or delete.
- match_text: a nonempty EXACT excerpt copied from that file. Choose a small
  unique anchor; if repeated, include occurrence (1-based in the current file).
  Never paraphrase it or copy from the batch dialogue/examples. Prefer a small
  verbatim paragraph/line, not a regenerated whole section. On error, inspect
  exact_match_count and literal_candidates and explicitly select a valid anchor.
- new_text: replacement/insertion text; MUST be empty for delete.
- rationale (recommended metadata): what evidence supports this specific change.
Missing section/rationale does not invalidate an otherwise exact safe edit.
Edits execute in array order against the CURRENT modified file, not line numbers.
Keep match_text/new_text local, preserve unrelated rules and canonical action
headings, and keep workflow and binding cards consistent. Do not hard-code
private values from examples. Do not create duplicate action sections.
The tool validates every operation on a copy first. Any failed edit rejects the
WHOLE plan: no partial writes. A later retry receives the concrete error.
No changes: {"operations":[]}.
Example:
{"operations":[{"file":"slot_policies.md","section":"#### `send-link`",
"op":"replace","match_text":"Use the most recent email.",
"new_text":"Use the email validated for the current account.",
"rationale":"The sampled failures reused an unverified address."}]}
'''


def apply_awm_edit_plan(raw, files, valid_actions):
    """Parse/validate/apply atomically in memory; caller checkpoints the bundle."""
    from skill_mining.online_refinement import apply_dynamic_skill_operations
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    plan = json.loads(text)
    if not isinstance(plan, dict) or set(plan) != {"operations"} or not isinstance(plan["operations"], list):
        raise ValueError('Expected {"operations":[...]} edit plan only')
    current = {name: files[name] for name in EDITABLE_FILES}
    outcomes = []
    for index, operation in enumerate(plan["operations"], 1):
        try:
            if not isinstance(operation, dict) or set(operation) - {
                "file", "section", "op", "match_text", "new_text", "rationale", "occurrence"}:
                raise ValueError("unsupported edit fields")
            for key in ("file", "op", "match_text", "new_text"):
                if not isinstance(operation.get(key), str):
                    raise ValueError(f"missing/non-text {key}")
            name = operation["file"]
            if name not in EDITABLE_FILES:
                raise ValueError("file is not writable")
            section_hint = operation.get("section")
            anchor = operation["match_text"]
            if not anchor:
                raise ValueError("match_text must be nonempty")
            source = current[name]
            occurrences = [m.start() for m in re.finditer(re.escape(anchor), source)]
            occurrence = operation.get("occurrence", 1)
            if not occurrences:
                raise AWMEditError("anchor missing: 0 exact matches; copy literal current-file text",
                    anchor_repair_context(source, anchor, occurrences, operation.get("occurrence")))
            if len(occurrences) > 1 and "occurrence" not in operation:
                raise AWMEditError(f"anchor ambiguous: {len(occurrences)} exact matches; supply a 1-based occurrence or a longer unique excerpt",
                    anchor_repair_context(source, anchor, occurrences, None))
            if type(occurrence) is not int or not 1 <= occurrence <= len(occurrences):
                raise AWMEditError(f"invalid occurrence: {occurrence!r}; expected 1..{len(occurrences)}",
                    anchor_repair_context(source, anchor, occurrences, occurrence))
            position = occurrences[occurrence - 1]
            # Headings are descriptive, not a second fragile addressing scheme.
            # Resolve the hierarchy from the EXACT selected anchor, never fuzzy
            # match model-authored section names or guess a missing text anchor.
            heading_path = []
            for heading in re.finditer(r"(?m)^(#{1,6})[ \t]+[^\n]+", source):
                if heading.start() > position:
                    break
                depth = len(heading[1])
                while heading_path and heading_path[-1][0] >= depth:
                    heading_path.pop()
                heading_path.append((depth, heading[0].strip()))
            section = heading_path[-1][1] if heading_path else "<document>"
            warnings = []
            if not isinstance(section_hint, str) or not section_hint.strip():
                warnings.append("section_inferred")
            elif section_hint != "<document>" and not any(
                section_hint.strip() in {h, re.sub(r"^#{1,6}\s+", "", h)} for _, h in heading_path
            ):
                warnings.append("section_hint_mismatch_exact_anchor_used")
            normalized = dict(operation)
            if not isinstance(operation.get("rationale"), str) or not operation["rationale"].strip():
                normalized["rationale"] = ""
                warnings.append("rationale_missing")
            updated, records = apply_dynamic_skill_operations(source, [normalized])
            if records[0].get("error") or not records[0].get("applied"):
                raise ValueError(str(records[0].get("error") or "edit was not applied"))
            if name != "skill.md":
                before = re.findall(r"(?m)^####\s+`([^`]+)`\s*$", source)
                after = re.findall(r"(?m)^####\s+`([^`]+)`\s*$", updated)
                for action in set(after):
                    if action not in valid_actions and action not in before:
                        raise ValueError("new card heading is not a canonical action")
                    if after.count(action) > max(1, before.count(action)):
                        raise ValueError("edit creates duplicate action cards")
            current[name] = updated
            outcomes.append({**records[0], "file": name, "section": section,
                             "section_hint": section_hint, "section_path": [h for _, h in heading_path],
                             "warnings": warnings})
        except Exception as exc:
            context = {**getattr(exc, "repair_context", {}), "edit_index": index,
                       "file": operation.get("file", "?") if isinstance(operation, dict) else "?",
                       "validated_prefix_edits": index - 1,
                       "note": "Earlier operations were simulated only; the whole rejected plan was NOT committed."}
            raise AWMEditError(f"Edit {index} ({context['file']}): {exc}", context) from exc
    if not current["skill.md"].strip():
        raise ValueError("Edit plan erased the entire workflow")
    return current, outcomes
