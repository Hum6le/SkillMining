"""ABCD evaluation schemas.

Mirrors the three turn types in the ABCD dataset:

- ``utterance``: agent selects from 100 utterance candidates
- ``action``: system takes an action (e.g. pull up account)
- ``customer``: customer responds (no prediction target)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ABCDTurnPrediction:
    """Prediction for a single ABCD turn.

    Attributes:
        turn_index: 0-based index within the conversation.
        turn_type: One of ``"utterance"``, ``"action"``, ``"customer"``.
        predicted_utterance_id: For ``utterance`` turns, the chosen
            utterance id (0-99).  ``None`` for other turn types.
        predicted_action: For ``action`` turns, the predicted next
            action name (e.g. ``"pull-up-account"``).  ``None`` otherwise.
        predicted_slots: For ``action`` turns, the predicted slot
            values (list of strings).  ``None`` otherwise.
    """

    turn_index: int
    turn_type: str
    predicted_utterance_id: int | None = None
    predicted_action: str | None = None
    predicted_slots: list[str] | None = None


@dataclass
class ABCDPrediction:
    """Full prediction for one ABCD conversation.

    Attributes:
        conversation_id: The ``convo_id`` from the dataset.
        turns: Ordered list of per-turn predictions.
    """

    conversation_id: str
    turns: list[ABCDTurnPrediction] = field(default_factory=list)


# ── Ground-truth extraction helpers ──────────────────────────────

@dataclass
class ABCDGroundTruth:
    """Extracted ground truth for one conversation turn.

    All fields are derived from the ``targets`` list in each delexed turn.
    """

    turn_index: int
    speaker: str           # "agent" | "action" | "customer"
    turn_type: str         # "utterance" | "action" | "customer"
    utterance_id: int | None       # ground-truth utterance id (utterance turns)
    action_name: str | None        # ground-truth next action (action turns)
    slot_values: list[str] | None  # ground-truth slot values (action turns)
    text: str = ""                 # the turn text
    candidates: list[int] = field(default_factory=list)  # utterance turn candidates
