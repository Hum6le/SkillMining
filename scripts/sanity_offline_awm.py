#!/usr/bin/env python3
"""Real ABCD data + native AWM induction/memory/metrics with mocked predictions.

Zero model requests. Oracle action predictions and mocked induction are NOT
evidence of method performance; this script only checks the adapter contract.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval_tod.abcd.agent import ABCDAgent
from scripts.abcd_offline_awm import run_offline_awm, _save
from scripts.abcd_runtime_utils import RUNTIME_DEFAULTS


class OracleMockAgent(ABCDAgent):
    def _chat(self, *args, **kwargs):
        return json.dumps({"workflow": "### MOCK workflow pattern\n**When**: A known request is observed.\n**Do**: Use current dialogue values.",
                           "action_card_updates": []})

    def predict_all_turns(self, conversation, *, turn_index=None, **kwargs):
        rows = []
        for index, turn in enumerate(conversation["delexed"]):
            targets = turn["targets"]
            if targets[1] != "take_action" or (turn_index is not None and index != turn_index):
                continue
            rows.append({"convo_id": str(conversation["convo_id"]), "turn_index": index,
                         "target_type": "action", "predicted_action": targets[2],
                         "predicted_slots": [str(value) for value in targets[3]], "prediction": "Oracle mock action",
                         "context": "Mock prediction on a real dialogue prefix."})
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subflow", default="shipping_issue")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    flow_dir = Path(args.output_dir) / args.subflow
    if flow_dir.exists():
        parser.error("Use a fresh sanity output directory")
    offline = flow_dir / "offline_artifact"
    offline.mkdir(parents=True)
    for name in ("skill.md", "reference.md", "action_rules.md", "slot_policies.md"):
        (offline / name).write_text("MOCK OFFLINE RESOURCE: " + name, encoding="utf-8")
    _save(offline / "mining_usage.json", {"total": {"calls": 0, "total_tokens": 0, "usage_available": True}})
    run_args = argparse.Namespace(model="mock", max_train=2, max_test=2, max_batches=2, batch_size=8,
                                  resume=False, skip_utterance_eval=True, **RUNTIME_DEFAULTS)
    with patch("llm.resolve_config", return_value={"model": "mock", "api_key": "", "base_url": ""}), \
            patch("eval_tod.abcd.agent.ABCDAgent", OracleMockAgent):
        run_offline_awm(run_args, args.subflow, flow_dir, ["MOCK1", "MOCK2"])
    state = json.loads((flow_dir / "offline_awm" / "awm_state.json").read_text(encoding="utf-8"))
    _save(Path(args.output_dir) / "sanity_manifest.json", {
        "real_dialogues": True, "oracle_predictions": True, "mocked_induction": True,
        "mocked_offline_resources": True, "native_memory_and_metrics": True,
        "llm_calls": 0, "performance_claim": False, "completed_batches": state["completed_batches"],
        "exemplars": len(state["exemplars"])})


if __name__ == "__main__":
    main()
