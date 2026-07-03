#!/usr/bin/env python3
"""Quick smoke test for ABCDAgent — run a few dialogues and print results."""

import os
import sys
from pathlib import Path

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval_tod.abcd.data import load_abcd_data
from eval_tod.abcd.agent import ABCDAgent
from eval_tod import evaluate_all
from awm import MemoryStore, WorkflowStore

# ── Config ────────────────────────────────────────────────────
N_DIALOGUES = 3
SPLIT = "test"
MODEL = "deepseek-chat"

print(f"Loading {N_DIALOGUES} ABCD {SPLIT} dialogues...")
convs = load_abcd_data(SPLIT)[:N_DIALOGUES]

print(f"Running ABCDAgent (model={MODEL})...")
print()

agent = ABCDAgent(model=MODEL, workflow=WorkflowStore(), memory=MemoryStore())
preds = agent.generate_predictions(convs)

for i, (conv, pred) in enumerate(zip(convs, preds)):
    scenario = conv["scenario"]
    flow = scenario.get("flow", "?")
    subflow = scenario.get("subflow", "?")

    # Get the last agent utterance as reference
    agent_utts = [
        t for t in conv["delexed"]
        if t.get("speaker") == "agent"
    ]
    ref_text = agent_utts[-1].get("text", "") if agent_utts else "(no agent turns)"
    gen_text = pred.response_text or "(empty)"

    print(f"[{i+1}/{N_DIALOGUES}] convo={conv['convo_id']}  {flow}/{subflow}")
    print(f"  turns: {len(conv['delexed'])} (agent: {len(agent_utts)})")
    print(f"  ref:  {ref_text[:150]}")
    print(f"  gen:  {gen_text[:150]}")
    print()

# Evaluate
print("=" * 50)
print("Evaluation...")
result = evaluate_all(convs, preds, dataset_name="abcd")
print(result["summary"])
print("Done.")
