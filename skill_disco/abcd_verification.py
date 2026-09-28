"""Held-out ABCD trace replay for Stage-5 skill verification.

ABCD supplies recorded backend actions, not a live transition environment.
Replay is therefore an exact action/slot/postcondition proxy; it cannot prove
counterfactual task success for branches absent from the recorded trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from eval_tod.abcd.action_schema import load_action_schema

from .abcd_trace import normalize_abcd_conversation
from .skill_specification import SkillContract
from .synthesis import VerificationResult


_ACTION = re.compile(r"^([^()]+)\((.*)\)$")


def _templates(contract: SkillContract) -> list[tuple[str, list[str]]]:
    templates = []
    for text in contract.canonical_action_sequence:
        match = _ACTION.fullmatch(text.strip())
        if match is None:
            return []
        slots = [part.strip() for part in match.group(2).split(",") if part.strip()]
        templates.append((match.group(1).strip(), slots))
    return templates


def build_replay_cases(
    contract: SkillContract,
    heldout_conversations: list[dict[str, Any]],
    *,
    max_cases: int = 12,
) -> list[dict[str, Any]]:
    """Find exact multi-action instances without sending held-out values to synthesis."""
    templates = _templates(contract)
    if len(templates) < 2:
        return []
    parameter_names = {parameter.name for parameter in contract.parameters}
    # The paper's environments expose admissible actions. ABCD has no such
    # per-state API, so expose its fixed public action vocabulary instead of
    # leaking the next gold action from a replay fixture.
    available_actions = sorted(load_action_schema()["actions"])
    cases = []
    for conversation in heldout_conversations:
        trace = normalize_abcd_conversation(conversation)
        for offset in range(len(trace.steps) - len(templates) + 1):
            steps = trace.steps[offset:offset + len(templates)]
            bindings: dict[str, str] = {}
            valid = True
            for step, (name, slot_names) in zip(steps, templates):
                if step.action_name != name or len(step.slot_values) != len(slot_names):
                    valid = False
                    break
                for parameter, value in zip(slot_names, step.slot_values):
                    if parameter not in parameter_names:
                        valid = False
                        break
                    if parameter in bindings and bindings[parameter] != value:
                        valid = False
                        break
                    bindings[parameter] = value
                if not valid:
                    break
            if not valid or any(
                parameter.required and parameter.name not in bindings
                for parameter in contract.parameters
            ):
                continue
            cases.append({
                "conversation_id": trace.conversation_id,
                "bindings": bindings,
                "available_actions": available_actions,
                "actions": [{
                    "name": step.action_name,
                    "slots": step.slot_values,
                    "observation": step.observation,
                } for step in steps],
            })
            break
        if len(cases) >= max_cases:
            break
    return cases


def verify_abcd_replay(
    source: str,
    contract: SkillContract,
    cases: list[dict[str, Any]],
    *,
    timeout_seconds: float = 5.0,
) -> VerificationResult:
    """Execute each case in a bounded worker and check the recorded transition."""
    if not cases:
        return VerificationResult(False, False, False, 0, 0, 0,
                                  "No held-out conversation matches this skill's action signature")
    worker = Path(__file__).with_name("replay_worker.py")
    failures = []
    passed = 0
    for case in cases:
        payload = {"source": source, "skill_name": contract.skill_name,
                   "case": case, "parameters": [item.to_dict() for item in contract.parameters]}
        try:
            process = subprocess.run(
                [sys.executable, "-I", str(worker)], input=json.dumps(payload),
                text=True, capture_output=True, timeout=timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{case['conversation_id']}: execution timed out")
            continue
        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError:
            failures.append(f"{case['conversation_id']}: worker failed: {process.stderr[:200]}")
            continue
        if process.returncode == 0 and result.get("passed") is True:
            passed += 1
        else:
            failures.append(f"{case['conversation_id']}: {result.get('feedback', 'replay mismatch')}")
    all_passed = passed == len(cases)
    return VerificationResult(
        passed=all_passed,
        runtime_correct=all_passed,
        postconditions_met=all_passed,
        actions_saved=min(len(case["actions"]) - 1 for case in cases) if all_passed else 0,
        cases_passed=passed,
        cases_total=len(cases),
        feedback="; ".join(failures[:5]) if failures else "Exact held-out action, slot, and observation replay passed",
    )
