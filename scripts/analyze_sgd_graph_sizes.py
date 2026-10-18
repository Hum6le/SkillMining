#!/usr/bin/env python3
"""Measure SGD graphs with the same trace-cover miner used by the SGD runner.

No LLM requests or skill compilation are performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_tod.sgd_adapter import gold_policy, iter_family_dialogues
from eval_tod.sgd_skill_agent import policy_token
from skill_mining.backbone_workflow_mining import mine_backbone_workflow_trace_cover


def measure(root: Path, family: str) -> dict:
    projected = []
    turns = 0
    for dialogue in iter_family_dialogues(root, family, "train"):
        delexed = [{"speaker": "customer" if turn["speaker"] == "user" else "agent",
                    "text": turn.get("utterance", ""), "targets": []}
                   for turn in dialogue["turns"]]
        for index in dialogue["target_indices"]["policy_turn_indices"]:
            token = policy_token(gold_policy(dialogue, index))
            delexed[index] = {"speaker": "action", "text": "",
                              "targets": [None, "take_action", token, []]}
            turns += 1
        projected.append({"convo_id": dialogue["dialogue_id"], "delexed": delexed})
    if not turns:
        return {"family": family, "train_dialogues": len(projected), "train_turns": 0,
                "nodes": 0, "edges": 0, "backbone_edges": 0, "residual_edges": 0}
    graph = mine_backbone_workflow_trace_cover(
        family, projected, min_branch_support=2, trace_coverage_target=.8,
        corpus_fitness_target=.95, routing_complexity_weight=2.,
    )["subgraph"]
    return {"family": family, "train_dialogues": len(projected), "train_turns": turns,
            "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
            "backbone_edges": len(graph["backbone"]["edges"]),
            "residual_edges": len(graph["residual_edges"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path,
                        default=Path("data/eval/sgd/splits_10pct_seed42"))
    parser.add_argument("--family", action="append", help="Repeat for selected families")
    args = parser.parse_args()
    index = json.loads((args.splits_dir / "INDEX.json").read_text(encoding="utf-8"))
    families = args.family or index["train_families"]
    for family in families:
        if family not in index["train_families"]:
            parser.error(f"Unknown train family: {family}")
        row = measure(args.splits_dir, family)
        row["validation_turns"] = index["families"][family]["validation"].get("policy_turns", 0)
        row["test_turns"] = index["families"][family]["test"].get("policy_turns", 0)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
