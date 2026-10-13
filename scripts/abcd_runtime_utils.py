"""One ABCD runtime contract for A2 direct arms and online-refine agents.

Only knowledge resources differ between arms; the base agent, candidate action
selection, retrieval limits and grounding path must not drift independently.
"""
from __future__ import annotations

RUNTIME_DEFAULTS = {
    "reference_top_k": 3,
    "reference_max_chars": 1800,
    "disable_competitive_action_cards": False,
    "action_selection_candidate_limit": 3,
}


def abcd_runtime_options(args) -> dict:
    return {
        "model": args.model,
        "max_turns": 1,
        "workflow_max_chars": None,
        "exemplar_max_chars": 3000,
        "reference_top_k": getattr(args, "reference_top_k", RUNTIME_DEFAULTS["reference_top_k"]),
        "reference_max_chars": getattr(args, "reference_max_chars", RUNTIME_DEFAULTS["reference_max_chars"]),
        "competitive_action_cards": not getattr(args, "disable_competitive_action_cards", RUNTIME_DEFAULTS["disable_competitive_action_cards"]),
        "action_selection_candidate_limit": getattr(args, "action_selection_candidate_limit", RUNTIME_DEFAULTS["action_selection_candidate_limit"]),
        "expose_scenario_labels": False,
    }


def build_abcd_runtime_agent(args, *, workflow, reference_text="", action_rules_text="",
                             slot_policies_text="", response_logger=None, workflow_id=None, memory=None):
    from awm import MemoryStore
    from eval_tod.abcd.agent import ABCDAgent

    return ABCDAgent(
        **abcd_runtime_options(args), workflow=workflow, memory=memory if memory is not None else MemoryStore(),
        reference_text=reference_text, action_rules_text=action_rules_text,
        slot_policies_text=slot_policies_text, response_logger=response_logger,
        workflow_id=workflow_id,
    )
