"""Loss-audited, deterministic projection of *visible* K1 resources.

No induction, rollout, or state replay. State supplies node-name aliases only:
unexposed guards/evidence in skill_dag_state.json must never become extra
baseline knowledge. Ambiguous prose is retained as scoped global text.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re

VERSION = "k1-flat-visible-v1"
RESOURCE_NAMES = ("skill.md", "reference.md", "action_rules.md", "slot_policies.md")
FLAT_INSTRUCTION = (
    "Records below are an unordered decision graph, not a mandatory execution sequence. "
    "EDGE records describe possible transitions; apply their conditions and exceptions. "
    "ACTION records describe action choice and ordered slot binding. GLOBAL_RULE records "
    "retain constraints that cannot safely be assigned to one edge. Compare all applicable "
    "rules against the current dialogue. Evidence examples never supply current slot values. "
    "Record order and IDs imply no priority."
)


def digest(value) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def flatten_resources(resources: dict[str, str], state: dict | None = None) -> dict:
    """Return flat records, renderings, and a line-by-line disposition ledger.

    Structural DSL/tree/table lines are converted to plain edge endpoints. Text
    is copied, never paraphrased. Only standalone markup/access-only lines are
    removed. Multi-target/unknown rules remain visible with their source scope.
    """
    missing = set(RESOURCE_NAMES) - resources.keys()
    if missing or any(not isinstance(resources[name], str) for name in RESOURCE_NAMES):
        raise ValueError(f"Expected four textual K1 resources; missing={sorted(missing)}")
    aliases = {str(n["id"]): str(n.get("label") or n["id"])
               for n in (state or {}).get("nodes", [])}
    actions: set[str] = set()  # State aliases are not permission to expose hidden nodes.
    for text in resources.values():
        actions.update(re.findall(r"(?m)^####\s+`([^`]+)`\s*$", text))
        for node, encoded in re.findall(r'^\s*(\S+)\s+("(?:[^"\\]|\\.)*")\s+slots=\d+\.\.\d+\s*$', text, re.M):
            aliases[node] = json.loads(encoded)
            actions.add(aliases[node])
    aliases.update({action: action for action in actions})

    def label(value):
        value = str(value).strip().strip("`\".,; ")
        return aliases.get(value, value)

    nodes = {action: {"action": action, "action_rule": [], "slot_policy": []} for action in actions}
    edges: dict[tuple[str, str], dict] = {}
    globals_: list[dict] = []
    ledger, warnings = [], []

    def edge(source, target):
        key = label(source), label(target)
        return edges.setdefault(key, {"source": key[0], "target": key[1], "rules": []})

    def disposition(name, number, status, target, reason=""):
        ledger.append({"file": name, "line": number, "disposition": status,
                       "target": target, "reason": reason})

    for name in RESOURCE_NAMES:
        lines = resources[name].splitlines()
        headings: list[tuple[int, str]] = []
        source = target = action = ""
        pending: list[tuple[int, str]] = []
        fenced = False
        dsl_section = ""
        tree_stack: list[tuple[int, str]] = []

        def flush():
            nonlocal pending
            if not pending:
                return
            body = "\n".join(line for _, line in pending).strip()
            if not body:
                pending = []
                return
            scope = " / ".join(h for _, h in headings if not re.search(r"backbone|primary route", h, re.I))
            entry = {"text": body, "scope": scope,
                     "origin": {"file": name, "lines": [i for i, _ in pending]}}
            lower = scope.lower()
            if action:
                role = "slot_policy" if "slot polic" in lower or name == "slot_policies.md" else "action_rule"
                nodes.setdefault(action, {"action": action, "action_rule": [], "slot_policy": []})[role].append(entry)
                dest = f"action:{action}:{role}"
            else:
                pair = re.search(r"(?:From|After)\s+`([^`]+)`.*?(?:transition|branch|continue)\s+to\s+`([^`]+)`", body, re.I)
                arrow = re.match(r"(?:[-*+]\s*)?`([^`]+)`\s*(?:->|→)\s*`([^`]+)`(?:\s*[:;]|\s+when\b)", body, re.I)
                rule_source, rule_target = source, target
                if pair:
                    rule_source, rule_target = label(pair[1]), label(pair[2])
                elif arrow:
                    rule_source, rule_target = label(arrow[1]), label(arrow[2])
                elif rule_source and not rule_target:
                    continuation = re.match(r"(?:[-*+]\s*)?(?:Continue|Transition|Proceed)\s+to\s+`([^`]+)`", body, re.I)
                    if continuation:
                        rule_target = label(continuation[1])
                    else:
                        mentioned = {label(v) for v in re.findall(r"`([^`]+)`", body)} & actions
                        candidates = mentioned - {rule_source}
                        if len(candidates) == 1:
                            rule_target = next(iter(candidates))
                        elif not candidates and re.search(r"repeat|retry", body, re.I) and rule_source in mentioned:
                            rule_target = rule_source
                if rule_source and rule_target:
                    edge(rule_source, rule_target)["rules"].append(entry)
                    dest = f"edge:{rule_source}=>{rule_target}"
                else:
                    entry.update(source_scope=rule_source or None)
                    globals_.append(entry)
                    dest = f"global:{len(globals_)}"
                    if "routing" in lower or "transition" in lower:
                        warnings.append({**entry["origin"], "reason": "ambiguous routing retained as scoped global rule"})
            for number, _ in pending:
                disposition(name, number, "preserved", dest)
            pending = []

        for number, raw in enumerate(lines, 1):
            line = raw.strip()
            if not line:
                # A blank inside a code/evidence block belongs to that block.
                if fenced:
                    pending.append((number, raw))
                else:
                    flush()
                    disposition(name, number, "formatting", "", "blank separator")
                continue
            comment = re.fullmatch(r"<!--\s*(.*?)\s*-->", line)
            if comment and not fenced:
                flush()
                marker = re.match(r"ROUTE_SOURCE:\s*(.+)", comment[1])
                route_edge = re.match(r"ROUTE_EDGE:\s*(.+?)=>(.+)", comment[1])
                if marker:
                    source, target = label(marker[1]), ""
                elif route_edge:
                    source, target = label(route_edge[1]), label(route_edge[2])
                if marker or route_edge or re.fullmatch(r"[A-Z_]+(?:START|END)", comment[1]):
                    disposition(name, number, "formatting", "", "writer anchor; scope carried into record")
                    continue
                # Unknown comments may contain policy, so do not drop them.
            heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", raw) if not fenced else None
            if heading:
                flush()
                level, title = len(heading[1]), heading[2]
                headings = [(n, h) for n, h in headings if n < level] + [(level, title)]
                source = target = action = ""
                dsl_section = ""
                tree_stack = []
                route = re.fullmatch(r"(?:Routing after|After)\s+`([^`]+)`", title, re.I)
                transition = re.fullmatch(r"`?([^`]+?)`?\s*(?:->|→)\s*`?([^`]+?)`?", title)
                card = re.fullmatch(r"`([^`]+)`", title)
                if route:
                    source = label(route[1])
                elif transition:
                    source, target = label(transition[1]), label(transition[2])
                    edge(source, target)
                elif card and card[1] in actions:
                    action = card[1]
                elif name == "reference.md" and title in actions:
                    source = title
                disposition(name, number, "formatting", "", "heading retained as record scope/identity")
                continue
            section = " / ".join(h for _, h in headings).lower()
            layout = bool(re.search(r"backbone (?:dsl|tree|edges)|primary route", section))
            if layout:
                flush()
                if line.startswith("```") or line == "BACKBONE v1" or re.fullmatch(r"root\s+\S+", line):
                    disposition(name, number, "layout_removed", "", "DSL wrapper/root declaration")
                    continue
                if line in {"nodes {", "edges {", "}"}:
                    dsl_section = line.split()[0] if line != "}" else ""
                    disposition(name, number, "layout_removed", "", "DSL grouping")
                    continue
                node = re.fullmatch(r'(\S+)\s+("(?:[^"\\]|\\.)*")\s+slots=(\d+)\.\.(\d+)', line)
                if node:
                    act = label(node[1])
                    nodes.setdefault(act, {"action": act, "action_rule": [], "slot_policy": []})["action_rule"].append(
                        {"text": f"Ordered slot count: {node[3]}..{node[4]}.", "scope": "slot contract",
                         "origin": {"file": name, "lines": [number]}})
                    disposition(name, number, "converted", f"action:{act}", "DSL node contract")
                    continue
                pairs = re.findall(r"`([^`]+)`\s*(?:->|→)\s*`([^`]+)`", line)
                plain = re.fullmatch(r"(\S+)\s+->\s+(\S+)", line)
                if not pairs and plain:
                    pairs = [plain.groups()]
                chain = re.fullmatch(r"(?:[-*+]\s+|\d+[.)]\s+)?(`[^`]+`(?:\s*(?:->|→)\s*`[^`]+`)+)\s*\.?", line)
                if chain:
                    names = re.findall(r"`([^`]+)`", chain[1])
                    pairs = list(zip(names, names[1:]))
                if pairs and (plain or re.fullmatch(
                    r"-?\s*`[^`]+`\s*(?:->|→)\s*`[^`]+`(?:;\s*(?:support|score)=[^;]+)*\.?", line) or chain):
                    for a, b in pairs:
                        record = edge(a, b)
                        metadata = re.search(r";\s*((?:support|score)=.*)$", line)
                        if metadata:
                            record["rules"].append({"text": "Observed edge statistics: " + metadata[1],
                                "scope": "edge statistics", "origin": {"file": name, "lines": [number]}})
                    disposition(name, number, "converted", [f"edge:{label(a)}=>{label(b)}" for a, b in pairs], "plain edge endpoints; no placement/priority")
                    continue
                tree = re.match(r"^([ |│]*)(?:\|--|`--|├──|└──)\s*`([^`]+)`\s*$", raw)
                if tree:
                    depth = len(tree[1]) // 4
                    while tree_stack and tree_stack[-1][0] >= depth:
                        tree_stack.pop()
                    parent = tree_stack[-1][1] if tree_stack else "ROOT"
                    child = label(tree[2])
                    edge(parent, child)
                    tree_stack.append((depth, child))
                    disposition(name, number, "converted", f"edge:{parent}=>{child}", "tree converted to plain edge")
                    continue
                if line in {"ROOT", "(empty backbone)"}:
                    disposition(name, number, "layout_removed", "", "tree wrapper")
                    continue
                # Unrecognized layout prose is kept, not guessed away.
            if re.fullmatch(r"-\s*Placement:\s*(?:backbone|residual)\s*", line, re.I):
                flush()
                disposition(name, number, "layout_removed", "", "backbone/residual designation")
                continue
            position = re.fullmatch(r"-\s*Position:\s*(.+)", line, re.I) if action else None
            if position:
                converted = []
                valid = True
                for part in position[1].rstrip(".").split(";"):
                    match = re.fullmatch(r"\s*(?:backbone\s+)?(parent[s]?|child of|children)\s+(.+?)\s*", part, re.I)
                    if not match or re.sub(r"`[^`]+`|[\s,.]|\band\b|\bnone\b|\(none\)", "", match[2], flags=re.I):
                        valid = False
                        break
                    for value in re.findall(r"`([^`]+)`", match[2]):
                        converted.append((label(value), action) if match[1].lower() in {"parent", "parents", "child of"}
                                         else (action, label(value)))
                if valid:
                    flush()
                    for a, b in converted:
                        edge(a, b)
                    disposition(name, number, "converted", [f"edge:{a}=>{b}" for a, b in converted],
                                "static card placement converted to plain edge endpoints")
                    continue
            # Remove only standalone access directives, not mixed routing prose.
            if re.fullmatch(r"-?\s*(?:see|consult|retrieve|read)\s+`?(?:reference|action_rules|slot_policies)\.md`?\.?", line, re.I):
                flush()
                disposition(name, number, "access_removed", "", "pure file-access directive replaced by flat resource protocol")
                continue
            if line == "- Routing details are written in the Routing Policies section.":
                flush()
                disposition(name, number, "access_removed", "", "pure section-access instruction")
                continue
            if line.startswith("```"):
                if not fenced:
                    flush()
                pending.append((number, raw))
                fenced = not fenced
                if not fenced:
                    flush()
                continue
            if not fenced and re.match(r"^(?:[-*+] |\d+[.)] )", raw):
                flush()
            pending.append((number, raw))
        flush()
        if fenced:
            warnings.append({"file": name, "reason": "source has an unclosed non-layout code fence; retained verbatim"})

    # These warnings are for human review, not a license to erase mixed prose.
    for entries in ([r for e in edges.values() for r in e["rules"]], globals_,
                    [r for n in nodes.values() for role in ("action_rule", "slot_policy") for r in n[role]]):
        for record in entries:
            if re.search(r"\bbackbone\b|\bresidual\b|(?:retrieve|consult|read).*\.md", record["text"], re.I):
                warnings.append({**record["origin"], "reason": "mixed layout/access wording retained to avoid changing policy; review before formal evaluation"})

    edge_records = []
    for i, ((a, b), record) in enumerate(sorted(edges.items()), 1):
        edge_records.append({"id": f"E{i:04}", **record})
        for act in (a, b):
            if act in actions:
                nodes.setdefault(act, {"action": act, "action_rule": [], "slot_policy": []})
    node_records = [{"id": f"A{i:04}", **node} for i, (_, node) in enumerate(sorted(nodes.items()), 1)]
    global_records = [{"id": f"G{i:04}", **record} for i, record in enumerate(globals_, 1)]
    records = {"actions": node_records, "edges": edge_records, "global_rules": global_records}

    def prose(entries):
        return "\n\n".join((f"Scope: {entry['scope']}\n" if entry.get("scope") else "") + entry["text"] for entry in entries)

    edge_blocks = []
    for record in edge_records:
        body = prose(record["rules"]) or "Observed possible transition; no explicit trigger is present in the visible K1 resources."
        edge_blocks.append(f"## {record['source']} -> {record['target']}\nRecord: {record['id']}\n{body}")
    global_blocks = [f"## GLOBAL_RULE {r['id']}\nScope: {r['scope']}\n"
                     + (f"Source scope: {r['source_scope']}\n" if r.get("source_scope") else "") + r["text"]
                     for r in global_records]
    action_rules = "# Flat action rules\n\n" + "\n\n".join(
        f"#### `{n['action']}`\n{prose(n['action_rule'])}" for n in node_records if n["action_rule"])
    slots = "# Flat slot policies\n\n" + "\n\n".join(
        f"#### `{n['action']}`\n{prose(n['slot_policy'])}" for n in node_records if n["slot_policy"])
    shared_globals = "\n\n".join(global_blocks)
    reference = "# Flat transition records\n\n" + "\n\n".join(edge_blocks)
    flat_skill = ("# Flat decision graph\n\n" + FLAT_INSTRUCTION + "\n\nAvailable actions: "
                  + ", ".join(f"`{n['action']}`" for n in node_records) + "\n\n" + shared_globals)
    full_text = "\n\n".join((flat_skill, reference, action_rules, slots)).rstrip() + "\n"
    expected = {(name, i) for name in RESOURCE_NAMES for i in range(1, len(resources[name].splitlines()) + 1)}
    actual = [(item["file"], item["line"]) for item in ledger]
    if set(actual) != expected or len(actual) != len(expected):
        raise AssertionError("Flattening ledger must account for every source line exactly once")
    counts = defaultdict(int)
    for item in ledger:
        counts[item["disposition"]] += 1
    return {"version": VERSION, "records": records, "full_context": full_text,
            "resources": {"skill.md": flat_skill.rstrip() + "\n", "reference.md": reference.rstrip() + "\n",
                          "action_rules.md": action_rules.rstrip() + "\n", "slot_policies.md": slots.rstrip() + "\n"},
            "audit": {"version": VERSION, "source_hashes": {n: digest(resources[n]) for n in RESOURCE_NAMES},
                      "records_hash": digest(records), "line_accounting_complete": True,
                      "line_counts": dict(counts), "ledger": sorted(ledger, key=lambda r: (r["file"], r["line"])),
                      "warnings": warnings,
                      "limitations": "Line accounting is not a proof of semantic equivalence. Ambiguous prose is retained; inspect scoped globals before formal evaluation. State-only policies and raw training graph edges are never exposed."}}
