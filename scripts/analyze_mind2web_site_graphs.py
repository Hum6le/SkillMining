#!/usr/bin/env python3
"""Count train-only Mind2Web action graphs per website from official AWM exemplars.

The exemplar file is a compact rendering of the original training trajectories.
Its assistant messages contain both the executable operation and the original
human-readable action representation. No test data or HTML download is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXEMPLARS = (
    ROOT / "third_party/agent-workflow-memory/mind2web/data/memory/exemplars.json"
)
ACTION_RE = re.compile(r"^Action: `([^`]*)` \((.*)\)$")
ELEMENT_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")


def compact(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def decode_example(messages: list[dict]) -> tuple[str, str, list[tuple[str, str, str, str]]]:
    if not messages or len(messages) % 2:
        raise ValueError("An exemplar must contain alternating user/action pairs")
    for i, message in enumerate(messages):
        if message["role"] != ("user" if i % 2 == 0 else "assistant"):
            raise ValueError("Exemplar roles do not alternate")
    specifier = messages[0]["specifier"]
    website = re.search(r"(?m)^Website: (.+)$", specifier)
    task = re.search(r"(?m)^Task: (.+)$", specifier)
    if not website or not task:
        raise ValueError("Missing website or task in exemplar specifier")
    actions = []
    for message in messages[1::2]:
        match = ACTION_RE.fullmatch(message["content"])
        if not match:
            raise ValueError(f"Unexpected action format: {message['content'][:120]}")
        operation = match.group(1).split(" ", 1)[0].upper()
        if operation not in {"CLICK", "TYPE", "SELECT"}:
            raise ValueError(f"Unexpected executable operation: {operation}")
        representation = match.group(2)
        element, sep, _ = representation.rpartition(" -> ")
        match_element = ELEMENT_RE.fullmatch(element)
        if not sep or not match_element:
            raise ValueError(f"Unexpected action representation: {representation}")
        tag = compact(match_element.group(1))
        descriptor = compact(match_element.group(2))
        actions.append((operation, tag, descriptor, representation))
    return website.group(1).strip(), task.group(1).strip(), actions


def graph_stats(examples: list[tuple[str, list[tuple[str, str, str, str]]]]) -> dict:
    stats = {
        "tasks": len(examples),
        "action_occurrences": 0,
        "transition_occurrences": 0,
        "blank_descriptors": 0,
    }
    for name, key in (
        ("op", lambda a: (a[0],)),
        ("op_tag", lambda a: (a[0], a[1])),
        ("surface", lambda a: (a[0], a[1], a[2])),
    ):
        nodes, edges = set(), set()
        edge_task_support = Counter()
        edge_occurrences = Counter()
        for _, actions in examples:
            sequence = [key(action) for action in actions]
            nodes.update(sequence)
            task_edges = list(zip(sequence, sequence[1:]))
            edges.update(task_edges)
            edge_task_support.update(set(task_edges))
            edge_occurrences.update(task_edges)
        stats[f"{name}_nodes"] = len(nodes)
        stats[f"{name}_edges"] = len(edges)
        stats[f"{name}_edges_in_2plus_tasks"] = sum(
            support >= 2 for support in edge_task_support.values()
        )
        stats[f"{name}_transitions_on_reused_edges"] = sum(
            count for edge, count in edge_occurrences.items()
            if edge_task_support[edge] >= 2
        )
    for _, actions in examples:
        stats["action_occurrences"] += len(actions)
        stats["transition_occurrences"] += max(0, len(actions) - 1)
        stats["blank_descriptors"] += sum(not action[2] for action in actions)
    stats["actions_per_task"] = round(stats["action_occurrences"] / len(examples), 2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exemplars", type=Path, default=DEFAULT_EXEMPLARS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-json", type=Path,
                        help="Optional original Mind2Web train shard for exact sequence validation")
    args = parser.parse_args()
    raw = json.loads(args.exemplars.read_text(encoding="utf-8"))
    by_site = defaultdict(list)
    by_task = {}
    for messages in raw:
        website, task, actions = decode_example(messages)
        if (website, task) in by_task:
            raise ValueError(f"Repeated website/task pair: {website}: {task}")
        by_task[(website, task)] = actions
        by_site[website].append((task, actions))
    if args.validate_json:
        shard = json.loads(args.validate_json.read_text(encoding="utf-8"))
        for record in shard:
            key = (record["website"], record["confirmed_task"].strip())
            expected = record["action_reprs"]
            if key not in by_task or [a[3] for a in by_task[key]] != expected:
                raise ValueError(f"Exemplar and original train shard disagree: {key}")
        print(f"Validated {len(shard)} original training trajectories exactly")
    rows = [{"website": site, **graph_stats(examples)}
            for site, examples in by_site.items()]
    rows.sort(key=lambda row: (-row["tasks"], row["website"]))
    totals = {
        "source": str(args.exemplars.resolve()),
        "training_tasks": len(raw),
        "training_websites": len(rows),
        "training_actions": sum(row["action_occurrences"] for row in rows),
        "within_task_transitions": sum(row["transition_occurrences"] for row in rows),
        "blank_descriptor_actions": sum(row["blank_descriptors"] for row in rows),
        "median_site_tasks": statistics.median(row["tasks"] for row in rows),
        "median_site_surface_nodes": statistics.median(row["surface_nodes"] for row in rows),
        "median_site_surface_edges": statistics.median(row["surface_edges"] for row in rows),
        "median_site_op_tag_nodes": statistics.median(row["op_tag_nodes"] for row in rows),
        "median_site_op_tag_edges": statistics.median(row["op_tag_edges"] for row in rows),
        "median_site_surface_edges_in_2plus_tasks": statistics.median(
            row["surface_edges_in_2plus_tasks"] for row in rows
        ),
        "median_site_op_tag_edges_in_2plus_tasks": statistics.median(
            row["op_tag_edges_in_2plus_tasks"] for row in rows
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sites.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(totals, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Mind2Web training graph size by website", "",
        "Source: official AWM `mind2web/data/memory/exemplars.json` (train examples only).",
        "Each website has a separate observed action graph. Edges connect adjacent actions",
        "within a task; no artificial start/end edges are counted.", "",
        "`op`: CLICK/TYPE/SELECT; `op_tag`: executable operation plus bracketed",
        "element type; `surface`: operation, element type, and lowercased visible",
        "descriptor. Operation values are excluded. Surface labels are a lexical proxy,",
        "not a validated semantic role ontology; blank descriptors merge distinct elements.", "",
        f"Tasks: {totals['training_tasks']}; websites: {totals['training_websites']}; "
        f"action occurrences: {totals['training_actions']}; "
        f"within-task transitions: {totals['within_task_transitions']}.",
        f"Blank element descriptors: {totals['blank_descriptor_actions']} "
        f"({totals['blank_descriptor_actions'] / totals['training_actions']:.1%} of actions).", "",
        "`reused edges` means the same edge occurs in at least two different training tasks.", "",
        "| Website | Tasks | Actions | Transitions | Op+tag nodes | Op+tag edges | Op+tag reused | Surface nodes | Surface edges | Surface reused |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| {website} | {tasks} | {action_occurrences} | "
                     "{transition_occurrences} | {op_tag_nodes} | {op_tag_edges} | "
                     "{op_tag_edges_in_2plus_tasks} | {surface_nodes} | {surface_edges} | "
                     "{surface_edges_in_2plus_tasks} |".format(**row))
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    print("Top websites:")
    for row in rows[:15]:
        print(f"  {row['website']:20s} tasks={row['tasks']:3d} "
              f"actions={row['action_occurrences']:3d} "
              f"op_tag={row['op_tag_nodes']:2d}/{row['op_tag_edges']:3d} "
              f"surface={row['surface_nodes']:3d}/{row['surface_edges']:3d}")


if __name__ == "__main__":
    main()
