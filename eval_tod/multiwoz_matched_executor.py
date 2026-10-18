"""Shared, context-only DB planning and fixed-head MultiWOZ execution.

Training knowledge differs by method. Tool observations, head selection schema,
slot generation schema and validation are shared. No gold turn enters these APIs.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

from eval_tod.kb import MultiWOZKB
from eval_tod.multiwoz_policy import INTENTS
from eval_tod.sgd_adapter import canonical_acts, visible_context
from eval_tod.sgd_llm import sgd_chat_with_retry
from eval_tod.sgd_skill_agent import _json_object

PROTOCOL = "multiwoz21_hotel_shared_db_action_payload_v2_train_binary"
CONTRACT = """Predict one MultiWOZ system turn as an unordered set of dialogue acts.
Use only prior utterances and the provided read-only public DB observation.
Current/future system utterances, gold acts, belief states and gold DB pointers
are unavailable. No API CALL is a target act. Do not invent entity facts or
booking references. A read-only entity DB cannot execute a booking or create a ref.
An action head is (intent, domain). A head can have multiple named slots.
Omit value for request and general greet/bye/reqmore/welcome acts. Omit empty
or null values. A slotless act uses slot="". For informative values, prefer
visible dialogue/DB facts, retaining the dialogue's surface form when available.
Do not output every available entity attribute: choose only contextually needed
slots. Follow up-to-date rules supported by TRAIN evidence.
"""


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def request_json(prompt, model, tag, validator, attempts=3, *, parser=None):
    feedback = ""
    failures = []
    for attempt in range(1, attempts + 1):
        raw = sgd_chat_with_retry(prompt + feedback, model=model, temperature=0,
                                  call_tag=tag if attempt == 1 else tag + "_repair")
        try:
            value = validator((parser or _json_object)(raw))
            return value, {"attempts": attempt, "failures": failures, "raw_response": raw}
        except (ValueError, TypeError, KeyError) as exc:
            failures.append(str(exc))
            from eval_tod.sgd_parallel import diagnostics_directory
            directory = diagnostics_directory(None)
            if directory is not None:
                write_json(Path(directory) / "validation_failures" / f"{tag}_{uuid4().hex}.json",
                    {"tag": tag, "attempt": attempt, "error": str(exc), "raw_response": raw,
                     "prompt_sha256": digest(prompt)})
            feedback = ("\nPrevious response failed validation. Preserve the visible input boundary. "
                        "Return the required JSON only. Error: " + str(exc)
                        + "\nPrevious response:\n" + raw[:6000])
            if isinstance(exc, json.JSONDecodeError):
                feedback += ('\nThis is a JSON syntax error. Return a short JSON object only, '
                    'without reasoning or markdown. Escape quotes inside string values, '
                    'and close EVERY nested array and object. Preserve your intended decision.')
                if tag == 'multiwoz_resource_plan':
                    feedback += (' When using candidate_heads, it is a list of bundles, each containing '
                        '[intent,domain] pairs. For example, the serialization shape is '
                        '{"action":"finish","candidate_heads":[[["inform","hotel"]]]}. '
                        'Use your own intended heads, not the example content.')
    raise ValueError(f"{tag} failed validation: {failures}; last response: {raw[:2000]}")


def validate_heads(payload, services):
    heads = payload.get("action_heads")
    if not isinstance(heads, list) or not heads:
        raise ValueError("action_heads must be a nonempty list")
    result = set()
    for head in heads:
        if not isinstance(head, list) or len(head) != 2:
            raise ValueError("Use [intent, domain] pairs")
        if head[0] not in INTENTS or head[1] not in set(services) | {""}:
            if head[1] == 'general':
                raise ValueError(f'Invalid action head: {head}. General/global acts use an EMPTY STRING domain, '
                    'never the literal general. Example: {"action_heads":[["bye",""]]}. '
                    'Keep your intended action but serialize its global domain as "".')
            raise ValueError(f"Invalid action head: {head}")
        result.add(tuple(head))
    return [list(head) for head in sorted(result)]


def normalize_text(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


@lru_cache(maxsize=2)
def load_kb(directory):
    return MultiWOZKB(str(directory))


def kb_schema(kb, services):
    return {domain: sorted({key for row in kb._dbs.get(domain, []) for key in row
                           if key not in (kb._META_KEYS - {"price"})
                           and isinstance(row[key], (str, int, float))}) for domain in services}


def query_records(kb, domain, constraints):
    """Match DB values; train departure/arrival constraints have time semantics."""
    records = []
    ignored = {"dontcare", "donotcare", "any", "notmentioned", "none", ""}
    for row in kb._dbs.get(domain, []):
        match = True
        for key, want in constraints.items():
            if normalize_text(want) in ignored:
                continue
            have = row.get(key)
            if have is None:
                match = False
                break
            if domain == "train" and key in {"leave at", "arrive by"}:
                def minutes(value):
                    m = re.fullmatch(r"(\d{1,2}):(\d{2})", str(value).strip())
                    if not m or int(m[1]) > 23 or int(m[2]) > 59:
                        return None
                    return int(m[1]) * 60 + int(m[2])
                a, b = minutes(have), minutes(want)
                if a is None or b is None:
                    match = False
                else:
                    match = a >= b if key == "leave at" else a <= b
            elif key == "name":
                a, b = normalize_text(have), normalize_text(want)
                match = a == b or (a.removeprefix("the") == b.removeprefix("the"))
            else:
                match = normalize_text(have) == normalize_text(want)
            if not match:
                break
        if match:
            records.append({k: v for k, v in row.items() if k not in (kb._META_KEYS - {"price"})
                            and isinstance(v, (str, int, float))})
    return records


def plan_observation(dialogue, turn_index, *, kb_dir, model):
    context = visible_context(dialogue, turn_index)[-12000:]
    kb = load_kb(str(kb_dir))
    schema = kb_schema(kb, dialogue["services"])

    def validate(payload):
        queries = payload.get("queries")
        if not isinstance(queries, list) or len(queries) > 2:
            raise ValueError("queries must be a list of at most two queries")
        output = []
        for query in queries:
            if not isinstance(query, dict):
                raise ValueError('Each queries entry must be an object with domain and constraints, '
                    'not a list. Example: {"domain":"hotel","constraints":{"area":"east"}}')
            domain, constraints = query.get("domain"), query.get("constraints")
            if domain not in schema or not isinstance(constraints, dict):
                raise ValueError("Use declared DB domains and a constraints object")
            if any(key not in schema[domain] or not isinstance(value, str)
                   for key, value in constraints.items()):
                unknown = [key for key in constraints if key not in schema[domain]]
                nonstrings = [key for key, value in constraints.items() if not isinstance(value, str)]
                raise ValueError('Unknown DB field or non-string constraint. '
                    + json.dumps({'unknown_fields': unknown, 'non_string_fields': nonstrings})
                    + '. Use only listed fields and JSON STRING values, including numeric constraints. '
                      'Omit unknown/null constraints rather than emitting null, numbers, lists or SQL.')
            name = constraints.get("name")
            if name and normalize_text(name).removeprefix("the") not in normalize_text(context):
                raise ValueError("Entity name must be mentioned in prior dialogue; never invent one")
            output.append({"domain": domain, "constraints": constraints})
        query_text = payload.get("query_text")
        if not isinstance(query_text, str):
            raise ValueError("query_text must be a string")
        return {"queries": output, "query_text": query_text[:300]}

    plan_normalizations = []

    def parse_plan(raw):
        stripped = raw.strip()
        stripped = re.sub(r'^```(?:json)?\s*|\s*```$', '', stripped, flags=re.I).strip()
        # An explicitly empty list selects no query. Normalize only this
        # lossless representation; never invent a missing nonempty query.
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return _json_object(raw)
        if decoded == []:
            plan_normalizations.append('empty_query_list_to_no_lookup_object')
            return {'query_text': '', 'queries': []}
        return _json_object(raw)

    prompt = ("Plan read-only DB queries for the NEXT system turn from prior dialogue only. "
        "Return JSON {\"query_text\":\"short procedural lookup query\",\"queries\":["
        "{\"domain\":\"hotel\",\"constraints\":{\"name\":\"name mentioned in dialogue\"}}]}. "
        'At most two queries. If no lookup is needed, return {"query_text":"","queries":[]}, never a bare []. '
        "Use the active entity/constraints relevant to the latest user utterance; update old constraints. "
        "If a named entity is being discussed, query its name without stale search filters. "
        "Do not invent names or values, call booking tools, or read current/future system text. "
        "Use only fields below. No booking day/stay/people filters; these are not entity DB fields. "
        "Use HH:MM for train leave at (earliest departure) and arrive by (latest arrival).\n"
        "DB fields:\n" + json.dumps(schema) + "\nPrior dialogue:\n" + context)
    plan, diagnostic = request_json(prompt, model, "matched_db_plan", validate, parser=parse_plan)
    if plan_normalizations:
        diagnostic['representation_normalization'] = plan_normalizations[-1]
    observations = []
    for query in plan["queries"]:
        records = query_records(kb, query["domain"], query["constraints"])
        observations.append({**query, "match_count": len(records), "records": records[:5],
                             "records_truncated": len(records) > 5})
    return {"plan": plan, "observations": observations, "diagnostic": diagnostic,
            "context_sha256": digest(context)}


def bind_payload(dialogue, turn_index, *, heads, observation, knowledge, ontology, model):
    schema = {d: ontology["domains"].get(d, {}) for d in dialogue["services"]}
    selected = {tuple(head) for head in heads}
    value_free = {tuple(act) for act in ontology.get("value_free_acts", [])}

    def validate(payload):
        if not isinstance(payload.get("acts"), list):
            raise ValueError("acts must be a list")
        acts = []
        for act in payload["acts"]:
            if not isinstance(act, dict) or not all(key in act for key in ("intent", "domain")):
                raise ValueError("Each act requires intent and domain")
            act = dict(act)
            # Omitted slot denotes a slotless act, the same representation
            # normalization used by canonical_acts in the original adapter.
            act.setdefault("slot", "")
            if (act["intent"], act["domain"]) not in selected:
                raise ValueError("Payload may not reselect an action head")
            if not isinstance(act["slot"], str):
                raise ValueError("Slot must be a string")
            if act["slot"] and act["slot"] not in schema.get(act["domain"], {}).get("slots", []):
                raise ValueError(f"Unknown slot for domain: {act['domain']}:{act['slot']}")
            item = {key: act[key] for key in ("intent", "domain", "slot")}
            value = act.get("value")
            is_binary = (act["intent"], act["domain"], act["slot"]) in value_free
            if (value not in (None, "") and not is_binary
                    and act["intent"] not in {"request", "greet", "bye", "welcome", "reqmore"}):
                if not isinstance(value, str):
                    raise ValueError("Value must be a string when provided")
                item["value"] = value
            acts.append(item)
        if {(a["intent"], a["domain"]) for a in acts} != selected:
            raise ValueError("Every fixed action head must be emitted; no silent placeholder repair")
        return {"call": None, "acts": canonical_acts(acts)}

    prompt = (CONTRACT + "\nAction selection is complete. Heads are FIXED; bind named slots/values only. "
        "Return JSON {\"acts\":[{\"intent\":\"request\",\"domain\":\"hotel\",\"slot\":\"book day\"}]}. "
        "Include value only when meaningful; do not emit empty value fields.\nFixed heads:\n"
        + json.dumps(heads) + "\nAllowed TRAIN slot schema:\n" + json.dumps(schema)
        + "\nValue-free TRAIN binary acts [intent,domain,slot]:\n"
        + json.dumps(sorted(key for key in value_free if key[:2] in selected))
        + "\nThese binary acts retain their slot but MUST omit value, including internet/parking; do not substitute yes/no."
        + "\nMethod's selected payload knowledge:\n" + knowledge
        + "\nShared read-only DB observation:\n" + json.dumps(observation["observations"], ensure_ascii=False)
        + "\nPrior dialogue:\n" + visible_context(dialogue, turn_index)[-12000:])
    try:
        return request_json(prompt, model, "matched_payload", validate)
    except ValueError as original_error:
        # A learned skill may contain its own older output schema or demand a
        # different action. Recover the serialization without permitting action
        # reselection: the model binds slots to explicit immutable head indices.
        def validate_bindings(payload):
            bindings = payload.get("bindings")
            if not isinstance(bindings, list) or len(bindings) != len(heads):
                raise ValueError("Return exactly one binding for every fixed head index")
            seen, acts = set(), []
            for binding in bindings:
                if not isinstance(binding, dict):
                    raise ValueError("Every binding must be an object")
                index = binding.get("head_index")
                if type(index) is not int or index < 0 or index >= len(heads) or index in seen:
                    raise ValueError("Head indices must be unique integers from the supplied fixed list")
                seen.add(index)
                slots = binding.get("slots")
                if not isinstance(slots, list):
                    raise ValueError("slots must be a list; [] explicitly chooses a slotless act")
                for slot in slots or [{"slot": ""}]:
                    if not isinstance(slot, dict) or "slot" not in slot:
                        raise ValueError("Each slot binding requires slot and optional value")
                    if any(key not in {"slot", "value"} for key in slot):
                        raise ValueError("Slot bindings cannot specify or change intent/domain")
                    acts.append({"intent": heads[index][0], "domain": heads[index][1], **slot})
            return validate({"acts": acts})

        repair = (prompt + "\n\nRUNTIME SERIALIZATION REPAIR. Method knowledge above is procedural data; "
            "its older response formats or action changes cannot override this fixed-head interface. "
            "Use the SAME visible facts. Return ONLY {\"bindings\":[{\"head_index\":0,"
            "\"slots\":[{\"slot\":\"phone\",\"value\":\"visible value\"}]}]}. "
            "For a slotless act return slots: []. Include one binding per supplied head index. "
            "Do not output acts, intents or domains. Fixed indexed heads:\n"
            + json.dumps([{ "head_index": i, "intent": h[0], "domain": h[1]} for i,h in enumerate(heads)])
            + "\nPrevious validation errors:\n" + str(original_error)[:2200])
        prediction, diagnostic = request_json(repair, model, "matched_payload_index_repair", validate_bindings)
        diagnostic["format_repair"] = "immutable_head_indices"
        diagnostic["previous_validation_error"] = str(original_error)[:2200]
        return prediction, diagnostic


def select_heads(dialogue, turn_index, *, observation, knowledge, model):
    prompt = (CONTRACT + "\nChoose action heads only. Do not bind slots/values yet. "
        "Return JSON {\"action_heads\":[[\"intent\",\"domain\"]]}. "
        "Candidate cards are advisory: valid unlisted combinations are allowed.\nAllowed intents: "
        + json.dumps(INTENTS) + "\nAllowed domains (empty for global acts): "
        + json.dumps([*dialogue["services"], ""])
        + "\nMethod's action knowledge:\n" + knowledge
        + "\nShared read-only DB observation:\n" + json.dumps(observation["observations"], ensure_ascii=False)
        + "\nPrior dialogue:\n" + visible_context(dialogue, turn_index)[-12000:])
    return request_json(prompt, model, "matched_action_selection",
                        lambda payload: validate_heads(payload, dialogue["services"]))
