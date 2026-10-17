"""Frozen visible-K1 organization controls. No induction or hidden-state edges."""
from __future__ import annotations

from collections import defaultdict
import json
import random
import re

from .k1_flatten import digest

VERSION = "k1-organization-v1"
BASE_ARMS = ("flat_react", "source_grouped", "ours_organized")
INSTRUCTION = (
    "The graph supplies advisory procedural evidence, not unconditional permissions or a mandatory "
    "execution sequence. Check every applicable condition against the current dialogue. "
    "Missing inline conditions require supporting reference evidence, not invented triggers. "
    "Use the same native resource tools for action cards, ordered slot policies and references. "
    "Examples never supply current instance values. Any hierarchy is a display organization; "
    "it does not remove other transitions or override their conditions."
)


def random_arm(seed):
    return f"random_hierarchy_seed_{seed}"


def visible_backbone(skill, edges, actions):
    """Read published DSL/tree/table only; never recover a backbone from state.

    A backbone may cover fewer actions than the resource pool. Those other
    actions stay visible, with all their transitions, outside the display tree.
    """
    import networkx as nx

    selected, roots = set(), set()
    aliases = {}
    # Same published node-label convention as the existing flattener.
    for key, encoded in re.findall(r'^\s*(\S+)\s+("(?:[^"\\]|\\.)*")\s+slots=\d+\.\.\d+\s*$', skill, re.M):
        aliases[key] = json.loads(encoded)
    dsl = re.search(r"(?m)^BACKBONE v1\s*$", skill)
    if dsl:
        tail = skill[dsl.start():]
        stop = re.search(r"(?m)^```|^#{1,3}\s", tail)
        block = tail[:stop.start()] if stop else tail
        root_lines = re.findall(r"(?m)^root[ \t]+(\S+)[ \t]*$", block)
        edge_block = re.search(r"(?ms)^edges[ \t]*\{[ \t]*\n(.*?)^\}[ \t]*$", block)
        if len(root_lines) != 1 or edge_block is None:
            raise ValueError("Malformed published Backbone DSL: require one root and edges block")
        published_root = aliases.get(root_lines[0], root_lines[0])
        roots.add(published_root)
        aliases["ROOT"] = published_root  # Native human-readable table uses ROOT.
        for raw in edge_block[1].splitlines():
            if not raw.strip():
                continue
            pair = re.fullmatch(r"\s*(\S+)\s+->\s+(\S+)\s*", raw)
            if not pair:
                raise ValueError(f"Malformed published Backbone DSL edge: {raw}")
            selected.add((aliases.get(pair[1], pair[1]), aliases.get(pair[2], pair[2])))

    active, stack = False, []
    for raw in skill.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", raw)
        if heading and len(heading[1]) <= 3:
            active = bool(re.search(r"backbone (?:tree|edges)|primary route", heading[2], re.I))
            stack = []
            continue
        if not active:
            continue
        if raw.strip() == "ROOT":
            roots.add(next(iter(roots)) if len(roots) == 1 else "ROOT")
        branch = re.match(r"^([ |│]*)(?:\|--|`--|├──|└──)\s*`([^`]+)`\s*$", raw)
        if branch:
            depth, child = len(branch[1]) // 4, branch[2]
            while stack and stack[-1][0] >= depth:
                stack.pop()
            parent = stack[-1][1] if stack else next(iter(roots)) if len(roots) == 1 else "ROOT"
            if not stack and parent not in actions:
                roots.add(parent)
            selected.add((parent, child))
            stack.append((depth, child))
        # Published edge tables and explicit quoted primary routes.
        chain = re.fullmatch(r"\s*(?:[-*+]\s+|\d+[.)]\s+)?(`[^`]+`(?:\s*(?:->|→)\s*`[^`]+`)+)\s*\.?", raw)
        head = re.match(r"\s*(?:[-*+]\s+|\d+[.)]\s+)?`([^`]+)`\s*(?:->|→)\s*`([^`]+)`", raw)
        names = re.findall(r"`([^`]+)`", chain[1]) if chain else [head[1], head[2]] if head else []
        if names:
            for a, b in zip(names, names[1:]):
                selected.add((aliases.get(a, a), aliases.get(b, b)))

    if not selected:
        raise ValueError("No explicit published backbone DSL/tree/edge table; hierarchy arms unavailable")
    available = {(e["source"], e["target"]) for e in edges}
    if selected - available:
        raise ValueError(f"Published backbone endpoints absent from visible pool: {sorted(selected - available)}")
    inferred = {a for a, _ in selected} - {b for _, b in selected}
    roots |= inferred
    if len(roots) != 1:
        raise ValueError(f"Published backbone must have one unambiguous root: {sorted(roots)}")
    root = next(iter(roots))
    graph = nx.DiGraph(sorted(selected))
    if not nx.is_arborescence(graph) or graph.in_degree(root) != 0:
        raise ValueError("Published backbone is not a rooted tree; do not silently repair hierarchy")
    if set(graph.nodes) - {root} - set(actions):
        raise ValueError("Published backbone contains an action absent from the common pool")
    return {"root": root, "nodes": sorted(set(graph.nodes) - {root}),
            "edges": [list(pair) for pair in sorted(selected)], "source": "published skill.md only"}


def random_backbone(layout, edges, seed):
    """One fixed random-priority arborescence; not uniform tree sampling."""
    import networkx as nx
    root = layout["root"]
    nodes = set(layout["nodes"]) | {root}
    original = {tuple(pair) for pair in layout["edges"]}
    # Only already-published root attachments are eligible. Never add shortcuts.
    allowed_roots = {pair for pair in original if pair[0] == root}
    candidates = {(e["source"], e["target"]) for e in edges
                  if e["source"] in nodes and e["target"] in nodes and e["source"] != e["target"]
                  and e["target"] != root and (e["source"] != root or (e["source"], e["target"]) in allowed_roots)}
    rng = random.Random(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(nodes))
    for a, b in sorted(candidates):
        graph.add_edge(a, b, weight=rng.random())
    try:
        tree = nx.maximum_spanning_arborescence(graph, attr="weight")
    except nx.NetworkXException as exc:
        raise ValueError("Visible edges cannot form a legal spanning display tree") from exc
    sampled = set(tree.edges())
    if tree.in_degree(root) != 0 or set(tree.nodes) != nodes:
        raise ValueError("Random display tree has wrong root or missing nodes")
    changed = sampled != original
    return {**layout, "edges": [list(p) for p in sorted(sampled)]}, {
        "seed": seed, "algorithm": "random-priority maximum rooted arborescence",
        "uniform_sampling": False, "changed": changed,
        "removed_original_tree_edges": len(original - sampled),
        "tree_edge_change_fraction": len(original - sampled) / len(original) if original else 0,
        "changed_business_tree_edges": len({p for p in original - sampled if p[0] != root}),
    }


def edge_unit(edge):
    # This exact text is included ONCE in every arm, never reassigned/reworded.
    parts = [f"<edge id={json.dumps(edge['id'])} source={json.dumps(edge['source'])} target={json.dumps(edge['target'])}>"]
    for rule in edge["rules"]:
        if rule.get("scope"):
            parts.append("Scope: " + rule["scope"])
        parts.append(rule["text"])
    if not edge["rules"]:
        parts.append("No inline trigger was published; consult reference evidence if needed.")
    return "\n".join(parts + ["</edge>"])


def _source_order(layout):
    children = defaultdict(list)
    for a, b in layout["edges"]:
        children[a].append(b)
    order, pending = [], [layout["root"]]
    while pending:
        source = pending.pop(0)
        order.append(source)
        pending.extend(sorted(children[source]))
    return order


def _display_tree(layout, by_pair):
    children = defaultdict(list)
    for a, b in layout["edges"]:
        children[a].append(b)
    lines, pending = [], [(layout["root"], "", layout["root"])]
    # A validated tree has no repeated nodes. Iterative rendering avoids
    # recursion limits on long published workflows.
    while pending:
        source, prefix, line = pending.pop()
        lines.append(line)
        successors = sorted(children[source])
        tasks = []
        for index, child in enumerate(successors):
            last = index == len(successors) - 1
            tasks.append((child, prefix + ("    " if last else "|   "),
                          prefix + ("`-- " if last else "|-- ") + f"`{child}` [{by_pair[(source, child)]}]"))
        for child, child_prefix, line in reversed(tasks):
            pending.append((child, child_prefix, line))
    return "\n".join(lines)


def render_view(edges, actions, common_globals, style, layout=None):
    parts = ["# Frozen decision graph", INSTRUCTION,
             "Available actions: " + ", ".join(f"`{a}`" for a in actions)]
    if style == "flat_react":
        parts.append("## Independent transition records")
        parts.extend(edge_unit(e) for e in edges)
    else:
        groups = defaultdict(list)
        for e in edges:
            groups[e["source"]].append(e)
        if layout:
            parts.append("## Display backbone (advisory, not an execution constraint)")
            order = _source_order(layout)
            # IDs refer to the very same edge bodies printed later.
            by_pair = {(e["source"], e["target"]): e["id"] for e in edges}
            parts.append("```text\n" + _display_tree(layout, by_pair) + "\n```")
            sources = [a for a in order if a in groups] + sorted(set(groups) - set(order))
            tree = {tuple(p) for p in layout["edges"]}
        else:
            sources, tree = sorted(groups), set()
        for source in sources:
            outgoing = groups[source]
            parts.extend([f"## Decision set after `{source}`", "Alternatives: " + ", ".join(f"`{e['target']}`" for e in outgoing)])
            if layout:
                primary = [e for e in outgoing if (e["source"], e["target"]) in tree]
                others = [e for e in outgoing if (e["source"], e["target"]) not in tree]
                for label, subset in (("Backbone transitions", primary), ("Other transitions (all retained)", others)):
                    if subset:
                        parts.append("### " + label)
                        parts.extend(edge_unit(e) for e in subset)
            else:
                parts.extend(edge_unit(e) for e in outgoing)
    parts.append(common_globals)
    return "\n\n".join(parts).rstrip() + "\n"


def build_variants(original, flat, seeds=(0, 1, 2)):
    pool = flat["shared_pool"]
    projection = pool["progressive"]["audit"]
    edges = projection["inline_edge_rules"]
    actions = sorted(n["action"] for n in flat["records"]["actions"])
    common = []
    for r in projection["inline_globals"]:
        common.append(f"## GLOBAL_RULE {r['id']}\nScope: {r['scope']}\n"
                      + (f"Source scope: {r['source_scope']}\n" if r.get("source_scope") else "") + r["text"])
    common.append(pool["global_context"])
    common_globals = "\n\n".join(common)
    knowledge = {"actions": actions, "edge_units": [{"id": e["id"], "text_hash": digest(edge_unit(e))} for e in edges],
                 "globals_hash": digest(common_globals), "resource_pool_hash": pool["manifest"]["pool_hash"]}
    layout, layout_error = None, None
    try:
        layout = visible_backbone(original["skill.md"], edges, actions)
    except ValueError as exc:
        layout_error = str(exc)
    variants = {}
    for arm in (*BASE_ARMS, *(random_arm(s) for s in seeds)):
        chosen, error, randomization = None, None, None
        if arm == "ours_organized" or arm.startswith("random_hierarchy_seed_"):
            error = layout_error
            chosen = layout
            if not error and arm.startswith("random_hierarchy_seed_"):
                seed = int(arm.rsplit("_", 1)[1])
                try:
                    chosen, randomization = random_backbone(layout, edges, seed)
                except ValueError as exc:
                    error = str(exc)
        workflow = None if error else render_view(edges, actions, common_globals, arm, chosen)
        if randomization is not None and not randomization["changed"]:
            # No alternate tree: exactly match the controlled Ours workflow.
            workflow = render_view(edges, actions, common_globals, "ours_organized", layout)
        metadata = {"version": VERSION, "arm": arm, "available": error is None, "error": error,
                    "knowledge_hash": digest(knowledge), "edge_units": knowledge["edge_units"],
                    "edge_count": len(edges), "action_count": len(actions),
                    "business_edge_count": sum(e["source"] in actions and e["target"] in actions for e in edges),
                    "branching_source_count": sum(sum(e["source"] == a for e in edges) > 1 for a in actions),
                    "layout": chosen, "randomization": randomization,
                    "workflow_hash": digest(workflow) if workflow is not None else None,
                    "workflow_chars": len(workflow) if workflow is not None else None,
                    "limitation": "Controlled rendering, not exact production Ours. Original card/reference prose may retain indirect structural hints. No token-matching or semantic-equivalence claim."}
        variants[arm] = {"workflow": workflow, "metadata": metadata}
    return variants
