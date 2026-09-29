"""Five-stage Skill-DisCo pipeline with held-out ABCD replay verification."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from .abcd_verification import build_replay_cases, verify_abcd_replay
from .operation_extraction import semantic_operation_from_dict
from .name_resolution import make_verified_names_unique
from .pipeline import _retrying_chat, run_offline_pseudocode_pipeline
from .pseudocode import render_skill_library
from .skill_specification import skill_contract_from_dict
from .synthesis import synthesize_and_verify


def split_induction_and_verification(
    conversations: list[dict[str, Any]], *, fraction: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out complete conversations before any discovery model call."""
    if len(conversations) < 2 or not 0 < fraction < 1:
        raise ValueError("at least two conversations and a fraction in (0, 1) are required")
    identifiers = [str(item.get("convo_id", "")) for item in conversations]
    if not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("conversation IDs must be nonempty and unique")
    ranked = sorted(
        range(len(conversations)),
        key=lambda index: hashlib.sha256(identifiers[index].encode("utf-8")).digest(),
    )
    heldout_indices = set(ranked[:max(1, min(len(ranked) - 1, round(len(ranked) * fraction)))])
    induction = [item for index, item in enumerate(conversations) if index not in heldout_indices]
    verification = [item for index, item in enumerate(conversations) if index in heldout_indices]
    return induction, verification


def run_compiled_abcd_pipeline(
    conversations: list[dict[str, Any]],
    chat_fn: Callable[..., str],
    *,
    model: str = "deepseek-chat",
    grouping_batch_size: int = 20,
    min_support: int = 2,
    verification_fraction: float = 0.2,
    verification_cases: int = 12,
    max_synthesis_attempts: int = 3,
) -> dict[str, Any]:
    """Distill on induction conversations, then compile and replay on held-out ones."""
    induction, verification = split_induction_and_verification(
        conversations, fraction=verification_fraction,
    )
    artifact = run_offline_pseudocode_pipeline(
        induction, chat_fn, model=model,
        grouping_batch_size=grouping_batch_size, min_support=min_support,
    )
    operations = {
        item["operation_id"]: semantic_operation_from_dict(item)
        for trace in artifact["traces"]
        for item in trace.get("operations", [])
    }
    compiled = []
    verified_contracts = []
    resilient_chat = _retrying_chat(chat_fn)
    for raw_contract in artifact["contracts"]:
        contract = skill_contract_from_dict(raw_contract)
        examples = [operations[item] for item in contract.source_operation_ids if item in operations]
        cases = build_replay_cases(contract, verification, max_cases=verification_cases)
        if not cases:
            compiled.append({"status": "discarded", "contract": raw_contract,
                             "implementation": None, "attempts": [],
                             "reason": "no matching held-out replay cases"})
            continue
        outcome = synthesize_and_verify(
            contract, examples, resilient_chat,
            lambda source, skill: verify_abcd_replay(source, skill, cases),
            environment_note=(
                "ABCD offline action replay: call env.step(action_name: str, "
                "ordered_slot_values: list[str]); it returns "
                "(observation: str, available_actions: list[str]). "
                "Use exact action names from canonical_action_sequence. "
                "The test fixture will reject incorrect action names, slot values, "
                "missing steps, or extra steps."
            ),
            model=model, max_attempts=max_synthesis_attempts,
        )
        outcome["verification_cases"] = len(cases)
        compiled.append(outcome)
        if outcome["status"] == "verified":
            verified_contracts.append(contract)
    artifact.update({
        "method": "skill-disco-abcd-five-stage-replay",
        "stages": artifact["stages"] + ["synthesis_and_heldout_replay_verification"],
        "split": {
            "induction_conversations": len(induction),
            "verification_conversations": len(verification),
            "verification_fraction": verification_fraction,
            "induction_ids": [str(item["convo_id"]) for item in induction],
            "verification_ids": [str(item["convo_id"]) for item in verification],
        },
        "compiled_skills": compiled,
        "verified_contracts": [item.to_dict() for item in verified_contracts],
        "skill_library": render_skill_library(verified_contracts),
    })
    make_verified_names_unique(artifact)
    return artifact
