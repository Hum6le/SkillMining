"""Dataclass definitions for ToD evaluation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Goal:
    """User goal / task objective for a dialogue."""

    description: str
    inform: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # domain -> {slot_name: value}  (value may contain | for alternatives)
    request: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # domain -> {slot_name: ""}  (empty dict means "all info")


@dataclass
class Turn:
    """A single dialogue turn."""

    speaker: str  # "user" or "system"
    utterance: str
    utt_idx: int
    dialogue_acts: Dict[str, Any] = field(default_factory=dict)
    state: Optional[Dict[str, Any]] = None  # belief state (user turns)
    booked: Optional[Dict[str, Any]] = None  # booking info (system turns)


@dataclass
class Dialogue:
    """A complete task-oriented dialogue."""

    dataset: str
    data_split: str
    dialogue_id: str
    original_id: str
    domains: List[str]
    goal: Goal
    turns: List[Turn]


@dataclass
class Prediction:
    """Agent prediction for a single dialogue."""

    dialogue_id: str
    inform_slots: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # domain -> {slot_name: predicted_value}

    request_slots: Dict[str, List[str]] = field(default_factory=dict)
    # domain -> [list of slot names the agent requested]

    booking: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # domain -> {"reference": "ABC123", "book_day": "...", ...}

    response_text: str = ""
    # natural language response to the user


@dataclass
class DialogueMetrics:
    """Per-dialogue evaluation metrics."""

    dialogue_id: str
    info_rate: float
    success: bool
    inform_correct: int
    inform_total: int
    request_correct: int
    request_total: int
    booking_passed: Optional[bool] = None  # None if no booking required
    domains_evaluated: List[str] = field(default_factory=list)


@dataclass
class AggregateMetrics:
    """Aggregate evaluation metrics across all dialogues."""

    num_dialogues: int
    info_rate: float  # aggregated: sum(correct) / sum(total)
    mean_info_rate: float  # mean of per-dialogue info_rates
    success_rate: float
    num_success: int
    num_fail: int
    llm_judge_scores: Dict[str, float] = field(default_factory=dict)
    per_domain_metrics: Dict[str, "AggregateMetrics"] = field(default_factory=dict)
