#!/usr/bin/env python3
"""Rebuild one pilot turn's prompts without calling the model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_tod.multiwoz_policy as policy
from eval_tod.multiwoz_policy import MultiWOZStandardSkillAgent, load_hotel_pilot, ontology_from_train
from eval_tod.sgd_adapter import visible_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--report-name", default="prompt_audit.json")
    args = parser.parse_args()
    output = args.root.resolve()
    manifest = json.loads((output / "ours" / "data_manifest.json").read_text(encoding="utf-8"))
    dialogues = load_hotel_pilot(Path(manifest["source"]), seed=manifest["seed"],
                                  fraction=manifest["fraction"])["train"]
    dialogue = dialogues[0]
    turn_index = dialogue["target_indices"]["policy_turn_indices"][0]
    agent = MultiWOZStandardSkillAgent(family="hotel", model=manifest["model"],
                                       ontology=ontology_from_train(dialogues),
                                       predictor=policy.predict_policy,
                                       mining_method="trace_cover", seed=manifest["seed"])
    checkpoint = args.checkpoint_dir or output / "ours" / "offline"
    saved = json.loads((checkpoint / "sgd_standard_state.json").read_text(encoding="utf-8"))
    agent.protocol = saved["protocol"]
    agent.load(checkpoint)
    prompts: dict[str, str] = {}

    def fake_request(prompt, _dialogue, _turn_index, *, stage, **_kwargs):
        prompts[stage] = prompt
        if stage == "reference_plan":
            return {"query_text": "", "candidate_actions": []}, {}
        if stage == "action_selection":
            return [["inform", "hotel"]], {}
        raise AssertionError(stage)

    def fake_chat(prompt, **_kwargs):
        prompts["act_payload"] = prompt
        return '{"acts":[{"intent":"inform","domain":"hotel","slot":""}]}'

    agent._request_validated_json = fake_request
    original_chat = policy.sgd_chat_with_retry
    policy.sgd_chat_with_retry = fake_chat
    try:
        agent.predict(dialogue, turn_index)
    finally:
        policy.sgd_chat_with_retry = original_chat

    context = visible_context(dialogue, turn_index)[-12000:]
    skill = agent.runtime_resources(dialogue, turn_index)["skill_text"]
    markers = ("react_trace", "Thought:", "Observation:", "Action Input:",
               "Current system utterance:", "Gold acts:")
    result = {"dialogue_id": dialogue["dialogue_id"], "turn_index": turn_index,
              "prior_utterance_chars": len(context),
              "working_skill_chars": len(agent.working_skill),
              "runtime_skill_chars": len(skill),
              "prompts": {stage: {"chars": len(prompt),
                                  "runtime_skill_occurrences": prompt.count(skill),
                                  "working_skill_occurrences": prompt.count(agent.working_skill),
                                  "markers": {marker: prompt.count(marker) for marker in markers
                                              if marker in prompt}}
                          for stage, prompt in prompts.items()}}
    target = output / args.report_name
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
