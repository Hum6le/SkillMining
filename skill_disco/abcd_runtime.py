"""Causal per-turn invocation of verified skills on offline ABCD dialogues.

ABCD has no live backend. A compiled skill is replayed over *past* observed
actions; its first unobserved env.step call becomes the current action
prediction. The frozen target action and all future turns remain inaccessible.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from eval_tod.abcd.action_schema import canonicalize_prediction, load_action_schema
from eval_tod.abcd.agent import _get_original_turn
from llm import chat

from .callable_runtime import CompiledSkillLibrary
from .runtime import create_skill_disco_abcd_agent


class PendingAction(Exception):
    def __init__(self, action: str, slots: list[str]):
        super().__init__(action)
        self.action = action
        self.slots = slots


class PrefixReplayEnvironment:
    """Replay known actions, then stop before the first unknown transition."""

    def __init__(self, previous_actions: list[dict[str, Any]], available_actions: list[str]):
        self.previous_actions = previous_actions
        self.available_actions = available_actions
        self.position = 0

    def step(self, action: str, slots: list[str] | None = None):
        actual_slots = [str(value) for value in (slots or [])]
        if self.position == len(self.previous_actions):
            raise PendingAction(str(action), actual_slots)
        expected = self.previous_actions[self.position]
        if str(action) != expected["action"] or actual_slots != expected["slots"]:
            raise ValueError("compiled skill does not match the observed action prefix")
        self.position += 1
        return expected["observation"], self.available_actions


def _json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else text
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        candidate = candidate[start:end + 1] if start >= 0 and end > start else ""
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("skill selection must be a JSON object")
    return payload


def _visible_context(conversation: dict[str, Any], turn_index: int) -> str:
    lines = []
    for index, turn in enumerate(conversation.get("delexed", [])[:turn_index]):
        speaker, text = _get_original_turn(conversation, index, turn)
        if text:
            lines.append(f"[{speaker}] {text}")
    return "\n".join(lines)


def _prior_actions(
    conversation: dict[str, Any], turn_index: int,
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events = []
    for prediction in predictions:
        index = int(prediction["turn_index"])
        if index >= turn_index or prediction.get("target_type") != "action":
            continue
        action = str(prediction.get("predicted_action", ""))
        if not action:
            continue
        _, observation = _get_original_turn(
            conversation, index, conversation["delexed"][index],
        )
        events.append({
            "turn_index": index,
            "action": action,
            "slots": [str(value) for value in (prediction.get("predicted_slots") or [])],
            "observation": observation,
        })
    return events


class CompiledSkillDiscoABCDAgent:
    """ABCD evaluator interface with verified skill calls and base fallback."""

    def __init__(
        self, library: CompiledSkillLibrary, skill_guidance: str, *,
        model: str = "deepseek-chat", response_logger=None,
    ):
        self.library = library
        self.base = create_skill_disco_abcd_agent(
            skill_guidance, model=model, response_logger=response_logger,
        )
        self.model = model
        self.response_logger = response_logger
        self.action_schema = load_action_schema()
        self.available_actions = sorted(self.action_schema["actions"])

    def _select_skill(self, context: str) -> tuple[str, dict[str, Any]] | None:
        specs = self.library.tool_specs()
        if not specs:
            return None
        prompt = (
            "You may call one verified procedural skill for the next ABCD backend "
            "action. Use only facts visible in the dialogue prefix. Do not guess "
            "missing customer values. If no skill applies, choose null. Return "
            "only JSON: {\"skill_name\": null|string, \"arguments\": {}}.\n\n"
            "Verified skills:\n" + json.dumps(specs, ensure_ascii=False)
            + "\n\nDialogue prefix:\n" + context
        )
        try:
            raw = chat(
                [{"role": "user", "content": prompt}], model=self.model,
                temperature=0.0, response_logger=self.response_logger,
                call_tag="skill_disco_selection",
            )
            choice = _json_object(raw)
        except Exception:
            return None
        name = choice.get("skill_name")
        arguments = choice.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        if name not in {item["name"] for item in specs}:
            return None
        return name, arguments

    def _project_skill(
        self, name: str, arguments: dict[str, Any],
        conversation: dict[str, Any], turn_index: int,
        prior_predictions: list[dict[str, Any]],
    ) -> tuple[str, list[str], dict[str, Any]] | None:
        previous = _prior_actions(conversation, turn_index, prior_predictions)
        # Try every possible start in the visible action history. Prefer the
        # longest exact prefix, then the most recent start when none matches.
        candidates = []
        for offset in range(len(previous) + 1):
            env = PrefixReplayEnvironment(previous[offset:], self.available_actions)
            try:
                self.library.invoke(name, arguments, env)
            except PendingAction as pending:
                if env.position != len(previous[offset:]):
                    continue
                action, slots, validation = canonicalize_prediction(
                    pending.action, pending.slots, self.action_schema,
                )
                if action and validation["valid"]:
                    candidates.append((env.position, offset, action, slots, validation))
            except Exception:
                continue
        if not candidates:
            return None
        _, start, action, slots, validation = max(candidates, key=lambda row: (row[0], row[1]))
        return action, slots, {"skill_name": name, "arguments": arguments,
                               "matched_prior_actions": len(previous) - start,
                               "action_schema_validation": validation}

    def predict_all_turns(
        self, conversation: dict[str, Any], verbose: bool = False,
        predict_actions: bool = False,
    ) -> list[dict[str, Any]]:
        if not predict_actions:
            return self.base.predict_all_turns(conversation, verbose=verbose,
                                               predict_actions=False)
        turns = conversation.get("delexed", [])
        target_indices = [
            index for index, turn in enumerate(turns)
            if (turn.get("speaker") == "agent" and str(turn.get("text", "")).strip())
            or (len(turn.get("targets", [])) >= 3 and turn["targets"][1] == "take_action")
        ]
        results = []
        for turn_index in target_indices:
            targets = turns[turn_index].get("targets", [])
            is_action = len(targets) >= 3 and targets[1] == "take_action"
            projected = None
            if is_action:
                context = _visible_context(conversation, turn_index)
                selected = self._select_skill(context)
                if selected:
                    projected = self._project_skill(
                        selected[0], selected[1], conversation, turn_index, results,
                    )
            if projected is None:
                results.extend(self.base.predict_all_turns(
                    conversation, verbose=verbose, predict_actions=True,
                    turn_index=turn_index,
                ))
                continue
            action, slots, metadata = projected
            results.append({
                "convo_id": str(conversation.get("convo_id", "?")),
                "turn_index": turn_index,
                "target_type": "action",
                "context": context,
                "context_view": "original",
                "prediction": "",
                "predicted_action": action,
                "predicted_slots": slots,
                "skill_invocation": metadata,
            })
        return results

    def generate_all_turn_predictions(
        self, conversations: list[dict[str, Any]], verbose: bool = True,
        predict_actions: bool = False,
    ) -> list[dict[str, Any]]:
        results = []
        for index, conversation in enumerate(conversations):
            if verbose:
                print(f"  [{index + 1}/{len(conversations)}] convo={conversation.get('convo_id', '?')}")
            results.extend(self.predict_all_turns(
                conversation, verbose=verbose, predict_actions=predict_actions,
            ))
            if index < len(conversations) - 1:
                time.sleep(self.base.delay)
        return results


def summarize_skill_invocations(turn_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Count actual callable predictions; ABCD has no episode turn-savings metric."""
    action_rows = [row for row in turn_results if row.get("target_type") == "action"]
    invoked = [row for row in action_rows if isinstance(row.get("skill_invocation"), dict)]
    by_skill: dict[str, int] = {}
    for row in invoked:
        name = str(row["skill_invocation"].get("skill_name", ""))
        by_skill[name] = by_skill.get(name, 0) + 1
    return {
        "action_target_turns": len(action_rows),
        "skill_invoked_turns": len(invoked),
        "base_fallback_turns": len(action_rows) - len(invoked),
        "skill_invocation_rate": len(invoked) / len(action_rows) if action_rows else 0.0,
        "by_skill": by_skill,
        "matched_prior_actions": sum(
            int(row["skill_invocation"].get("matched_prior_actions", 0)) for row in invoked
        ),
    }
