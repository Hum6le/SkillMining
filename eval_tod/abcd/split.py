#!/usr/bin/env python3
"""ABCD intent-based data splitting and turn-level data extraction.

Provides:
  - ``split_by_subflow()``: split conversations per subflow → train/test
  - ``extract_all_agent_turns()``: flatten dialogues into per-turn samples
  - ``build_turn_samples()``: full pipeline: split → extract turns
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnSample:
    """A single prediction target: one agent turn within a dialogue."""

    convo_id: str
    turn_index: int          # 0-based index in delexed turns
    subflow: str             # scenario.subflow
    flow: str                # scenario.flow
    dialogue_idx: int        # which dialogue this belongs to (for grouping)
    turn_count: int          # total turns in this dialogue
    agent_turn_num: int      # which agent turn this is (1st, 2nd, ...)
    total_agent_turns: int   # total agent turns in this dialogue

    context: str             # all turns BEFORE this agent turn
    reference: str           # ground-truth agent utterance
    scenario: dict[str, Any] = field(default_factory=dict)  # full scenario info


def split_by_subflow(
    conversations: list[dict],
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split ABCD conversations by subflow for stratified train/test.

    Each subflow gets its own train/test split, ensuring every intent
    appears in both splits.
    """
    rng = random.Random(seed)

    # Group by subflow
    by_subflow: dict[str, list[dict]] = defaultdict(list)
    for conv in conversations:
        subflow = str(conv.get("scenario", {}).get("subflow", "unknown"))
        by_subflow[subflow].append(conv)

    train: list[dict] = []
    test: list[dict] = []

    for subflow, convs in sorted(by_subflow.items()):
        rng.shuffle(convs)
        n_train = max(1, int(len(convs) * train_frac))
        train.extend(convs[:n_train])
        test.extend(convs[n_train:])

    # Shuffle within each split
    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


def extract_all_agent_turns(
    conversations: list[dict],
) -> list[TurnSample]:
    """Extract ALL agent turns from a list of conversations as TurnSamples.

    Each agent turn becomes one prediction target with its preceding context.
    """
    samples: list[TurnSample] = []

    for conv_idx, conv in enumerate(conversations):
        convo_id = str(conv.get("convo_id", conv_idx))
        scenario = conv.get("scenario", {})
        subflow = str(scenario.get("subflow", "unknown"))
        flow = str(scenario.get("flow", "unknown"))
        delexed = conv.get("delexed", [])

        # Find all agent turns
        agent_turn_indices = [
            i for i, t in enumerate(delexed)
            if t.get("speaker") == "agent" and t.get("text", "").strip()
        ]
        total_agent = len(agent_turn_indices)

        for agent_num, turn_idx in enumerate(agent_turn_indices, 1):
            # Context: all turns BEFORE this agent turn
            context_lines: list[str] = []
            for i in range(turn_idx):
                t = delexed[i]
                spk = t.get("speaker", "unknown")
                txt = t.get("text", "").strip()
                if not txt:
                    continue
                label = {"agent": "Agent", "customer": "Customer", "action": "System"}.get(spk, spk)
                context_lines.append(f"[{label}] {txt}")

            # Reference: this agent turn's text
            reference = str(delexed[turn_idx].get("text", "")).strip()

            if not context_lines or not reference:
                continue

            samples.append(TurnSample(
                convo_id=convo_id,
                turn_index=turn_idx,
                subflow=subflow,
                flow=flow,
                dialogue_idx=conv_idx,
                turn_count=len(delexed),
                agent_turn_num=agent_num,
                total_agent_turns=total_agent,
                context="\n".join(context_lines),
                reference=reference,
                scenario=scenario,
            ))

    return samples


def build_turn_samples(
    conversations: list[dict],
    train_frac: float = 0.8,
    seed: int = 42,
    max_train: int | None = None,
    max_test: int | None = None,
) -> tuple[list[TurnSample], list[TurnSample]]:
    """Full pipeline: split by subflow → extract all agent turns.

    Returns:
        (train_samples, test_samples)
    """
    train_convs, test_convs = split_by_subflow(conversations, train_frac, seed)

    if max_train:
        train_convs = train_convs[:max_train]
    if max_test:
        test_convs = test_convs[:max_test]

    train_samples = extract_all_agent_turns(train_convs)
    test_samples = extract_all_agent_turns(test_convs)

    return train_samples, test_samples


def summarise_split(train_samples: list[TurnSample], test_samples: list[TurnSample]) -> str:
    """Print a summary of the split."""
    train_by_subflow = defaultdict(int)
    test_by_subflow = defaultdict(int)
    for s in train_samples:
        train_by_subflow[s.subflow] += 1
    for s in test_samples:
        test_by_subflow[s.subflow] += 1

    lines = [
        f"Train: {len(train_samples)} turns from {len({s.dialogue_idx for s in train_samples})} dialogues",
        f"Test:  {len(test_samples)} turns from {len({s.dialogue_idx for s in test_samples})} dialogues",
        "",
        f"{'Subflow':35s} {'Train':>6s} {'Test':>6s}",
        "-" * 50,
    ]
    all_subflows = sorted(set(train_by_subflow) | set(test_by_subflow))
    for sf in all_subflows:
        lines.append(f"{sf:35s} {train_by_subflow[sf]:6d} {test_by_subflow[sf]:6d}")
    return "\n".join(lines)
