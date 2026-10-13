#!/usr/bin/env python3
"""CPU-only A3 projection check on real ABCD data. No LLM requests.

Induction prose and action/slot cards are placeholders for format testing only;
the actual graph and evaluation traces are real. This is not a performance run.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_a3_abcd import _json, build_variants, graph_metrics
from skill_mining.backbone_workflow_mining import mine_backbone_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subflow", default="account_access")
    parser.add_argument("--max-train", type=int, default=20)
    parser.add_argument("--max-test", type=int, default=5)
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = ROOT / "data/eval/abcd/splits" / args.subflow
    train = json.loads((source / "train.json").read_text(encoding="utf-8"))[:args.max_train]
    test = json.loads((source / "test.json").read_text(encoding="utf-8"))[:args.max_test]
    graph = mine_backbone_workflow(args.subflow, train)["subgraph"]
    rules = {}
    for edge in graph["edges"]:
        rules.setdefault(edge["source"], []).append({"target": edge["target"],
            "status": "underspecified", "condition": "MOCK condition: format sanity only."})
    graph["transition_induction"] = {"rules_by_source": rules}
    seeds = tuple(int(item) for item in args.random_seeds.split(","))
    variants = build_variants(args.subflow, graph, "# Action Rules\nMOCK cards.", "# Slot Policies\nMOCK policies.", random_seeds=seeds)
    output = Path(args.output_dir)
    results = {}
    for arm, variant in variants.items():
        for name, text in variant["resources"].items():
            directory = output / arm
            directory.mkdir(parents=True, exist_ok=True)
            (directory / name).write_text(text, encoding="utf-8")
        results[arm] = graph_metrics(args.subflow, variant, test)
    _json(output / "sanity_result.json", {"real_graph": True, "mock_induction": True,
          "llm_calls": 0, "performance_claim": False,
          "randomization": {job: variant["randomization"] for job, variant in variants.items() if variant["randomization"]},
          "results": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
