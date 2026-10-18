#!/usr/bin/env python3
"""Compare strict scoring with empty value fields treated as omitted values."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.sgd_adapter import score_policies


def audit(directory: Path) -> dict:
    path = directory / "predictions.jsonl"
    phase = "validation"
    if not path.exists():
        path = directory / "ours_training_turns.jsonl"
        phase = "train_rollout_partial"
    gold, predicted = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "sgd_evidence" in row:
            row = {"gold": row["sgd_evidence"]["gold_policy"],
                   "prediction": row["sgd_evidence"]["predicted_policy"]}
        if "gold" in row and "prediction" in row:
            gold.append(row["gold"])
            predicted.append(row["prediction"])

    def normalized(policy):
        policy = copy.deepcopy(policy)
        for act in policy["acts"]:
            if act.get("value") == "":
                act.pop("value")
        return policy

    result = {"source": str(path.resolve()), "phase": phase, "turns": len(gold),
              "normalization": "Treat value=empty string as omitted value; all other fields stay exact.",
              "strict": score_policies(gold, predicted),
              "empty_values_omitted": score_policies([normalized(p) for p in gold],
                                                       [normalized(p) for p in predicted])}
    (directory / "empty_value_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = audit(parser.parse_args().directory)
    print(json.dumps({"phase": result["phase"], "turns": result["turns"],
                      "strict_joint": result["strict"]["turn_joint_ast"],
                      "empty_values_omitted_joint": result["empty_values_omitted"]["turn_joint_ast"]}))
