#!/usr/bin/env python3
"""Measure prospective MultiWOZ action-bundle graphs without LLM calls.

One node is an unordered set of (intent, domain) heads on a labeled system
turn. Slot names/values stay in the payload. This is an analysis projection,
not an implemented MultiWOZ skill agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skill_mining.backbone_workflow_mining import mine_backbone_workflow_trace_cover


def action_heads(turn: dict) -> tuple[tuple[str, str], ...]:
    acts = turn.get("dialogue_acts") or {}
    return tuple(sorted({
        (str(act["intent"]).lower(), str(act.get("domain") or ""))
        for group in acts.values() for act in group if act.get("intent")
    }))


def measure(dialogues: list[dict], name: str) -> dict:
    projected = []
    labeled_turns = unlabeled_turns = 0
    unique_heads: set[tuple[str, str]] = set()
    for dialogue in dialogues:
        delexed = []
        for turn in dialogue["turns"]:
            speaker = turn["speaker"]
            if speaker != "system":
                delexed.append({"speaker": "customer", "text": turn.get("utterance", ""), "targets": []})
                continue
            heads = action_heads(turn)
            if not heads:
                unlabeled_turns += 1
                delexed.append({"speaker": "agent", "text": turn.get("utterance", ""), "targets": []})
                continue
            labeled_turns += 1
            unique_heads.update(heads)
            value = json.dumps(heads, ensure_ascii=False, separators=(",", ":"))
            token = "policy-" + hashlib.sha256(value.encode()).hexdigest()[:16]
            delexed.append({"speaker": "action", "text": "",
                            "targets": [None, "take_action", token, []]})
        projected.append({"convo_id": dialogue["dialogue_id"], "delexed": delexed})
    row = {"scenario": name, "dialogues": len(dialogues), "labeled_system_turns": labeled_turns,
           "unlabeled_system_turns": unlabeled_turns, "atomic_action_heads": len(unique_heads)}
    if labeled_turns:
        graph = mine_backbone_workflow_trace_cover(
            name, projected, min_branch_support=2, trace_coverage_target=.8,
            corpus_fitness_target=.95, routing_complexity_weight=2.,
        )["subgraph"]
        row.update(nodes=len(graph["nodes"]), edges=len(graph["edges"]),
                   backbone_edges=len(graph["backbone"]["edges"]),
                   residual_edges=len(graph["residual_edges"]))
    else:
        row.update(nodes=0, edges=0, backbone_edges=0, residual_edges=0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path,
                        default=Path("data/eval/multiwoz21/splits"))
    parser.add_argument("--official-source", type=Path,
                        help="Use the source's official train/validation/test labels")
    parser.add_argument("--fraction", type=float, default=.1,
                        help="Fraction of each official training scenario to mine")
    parser.add_argument("--scenario", action="append", help="Repeat to restrict scenarios")
    parser.add_argument("--min-train-dialogues", type=int, default=20)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1 or not math.isfinite(args.fraction):
        parser.error("--fraction must be finite and in (0, 1]")
    if args.official_source:
        source = json.loads(args.official_source.read_text(encoding="utf-8"))
        grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for dialogue in source:
            name = "_".join(sorted(domain for domain in dialogue["domains"]
                                   if domain != "general")) or "general"
            grouped[name][dialogue["data_split"]].append(dialogue)
        scenarios = args.scenario or sorted(grouped)
        for name in scenarios:
            splits = grouped[name]
            train = splits["train"]
            quota = math.ceil(len(train) * args.fraction)
            selected = sorted(train, key=lambda item: (
                hashlib.sha256(f"42\0{name}\0{item['dialogue_id']}".encode()).digest(),
                item["dialogue_id"],
            ))[:quota]
            if len(selected) < args.min_train_dialogues or not splits["validation"] or not splits["test"]:
                continue
            row = measure(selected, name)
            row.update(official_train_dialogues=len(train),
                       validation_dialogues=len(splits["validation"]),
                       test_dialogues=len(splits["test"]),
                       train_sample_fraction=args.fraction)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        return
    summary = json.loads((args.splits_dir / "split_summary.json").read_text(encoding="utf-8"))
    scenarios = args.scenario or [name for name, row in summary.items()
                                  if name != "_totals" and row["sampled_1_10"]["train"] >= args.min_train_dialogues
                                  and row["sampled_1_10"]["val"] and row["sampled_1_10"]["test"]]
    for name in scenarios:
        path = args.splits_dir / f"scenario_{name}--train.json"
        dialogues = json.loads(path.read_text(encoding="utf-8"))
        row = measure(dialogues, name)
        row["validation_dialogues"] = summary[name]["sampled_1_10"]["val"]
        row["test_dialogues"] = summary[name]["sampled_1_10"]["test"]
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
