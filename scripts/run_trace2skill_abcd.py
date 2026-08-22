#!/usr/bin/env python3
r"""ABCD Trace2Skill-style pipeline driven by AST.

This is a separate pipeline from the original Trace2Skill MultiWOZ flow.
Each run is restricted to exactly one ABCD subflow; global results are
aggregated across independent subflow runs by a separate script.
It keeps the same high-level loop:

1. Run a seed agent with a skill/workflow prompt
2. Evaluate on AST
3. Analyze failed conversations
4. Evolve the skill with Trace2Skill's ParallelSkillEvolver
5. Re-evaluate on AST

The key difference is that failure detection and optimization are driven by
ABCD Action State Tracking (AST) instead of MultiWOZ IR/Success.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

_TRACE2SKILL = _PROJECT_ROOT / "Trace2Skill"
if str(_TRACE2SKILL) in sys.path:
    sys.path.remove(str(_TRACE2SKILL))
sys.path.insert(0, str(_TRACE2SKILL))

from eval_tod.abcd.agent import ABCDAgent, compute_ast_from_turn_results
from eval_tod.abcd.data import load_abcd_data
from eval_tod.abcd.metrics import evaluate_abcd
from eval_tod.abcd.agent import turn_results_to_abcd_predictions
from eval_tod.abcd.data import extract_ground_truth
from eval_tod.cli import evaluate_text_records
from eval_tod.response_logger import ResponseLogger
from llm import chat

ABCD_DIR = "data/eval/abcd/data"
DEFAULT_SKILL_PATH = "eval_tod/skills/abcd_trace2skill/SKILL.md"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_LLM_QPS = 3.0
_REPORT_ITEM_PATTERN = re.compile(
    r"^#\s+(Failure Cause Item|Failure Memory Item)\s+(\d+)\s*\n",
    re.MULTILINE,
)


class _RateLimitedChat:
    """Thread-safe wrapper around llm.chat with a fixed minimum call interval."""

    def __init__(self, chat_fn, qps: float):
        self._chat_fn = chat_fn
        self._min_interval = 1.0 / qps if qps and qps > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def __call__(self, *args, **kwargs):
        if self._min_interval > 0:
            with self._lock:
                now = time.monotonic()
                wait = self._next_allowed - now
                if wait > 0:
                    time.sleep(wait)
                    now = time.monotonic()
                self._next_allowed = now + self._min_interval
        return self._chat_fn(*args, **kwargs)


class _RetryingChat:
    """Thread-safe-ish retry wrapper for transient LLM transport failures."""

    def __init__(self, chat_fn, max_retries: int, base_delay: float):
        self._chat_fn = chat_fn
        self._max_retries = max(0, max_retries)
        self._base_delay = max(0.0, base_delay)

    def __call__(self, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._chat_fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    break
                delay = self._base_delay * (2 ** attempt)
                logging.getLogger("abcd_trace2skill").warning(
                    "LLM call failed (%s); retry %d/%d after %.1fs",
                    exc,
                    attempt + 1,
                    self._max_retries,
                    delay,
                )
                if delay > 0:
                    time.sleep(delay)
        raise last_exc  # type: ignore[misc]


def _install_llm_wrappers(qps: float, max_retries: int, retry_base_delay: float):
    """Patch llm.chat so project calls share rate limits and retry policy."""
    import llm

    global chat
    original_chat = llm.chat
    wrapped_chat = _RateLimitedChat(original_chat, qps)
    if max_retries > 0:
        wrapped_chat = _RetryingChat(wrapped_chat, max_retries, retry_base_delay)
    llm.chat = wrapped_chat
    chat = wrapped_chat
    return original_chat


@dataclass
class PipelineOutputs:
    seed_eval: dict[str, Any] | None
    evolved_eval: dict[str, Any]
    output_dir: Path
    evolved_skill_path: Path


class _ChatClientAdapter:
    """Minimal adapter so Trace2Skill evolver can use the project's chat() API."""

    def __init__(self, model: str, response_logger: ResponseLogger | None = None):
        self.model = model
        self.response_logger = response_logger

    def chat(self, messages, settings=None) -> str:
        temperature = getattr(settings, "temperature", 0.3) if settings is not None else 0.3
        return chat(
            messages,
            model=self.model,
            temperature=temperature,
            response_logger=self.response_logger,
        )


def _load_conversations(args) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load train/test conversations from either explicit files or named splits."""
    if bool(args.train_file) != bool(args.test_file):
        raise ValueError("--train-file and --test-file must be provided together")

    if args.train_file and args.test_file:
        train_path = Path(args.train_file).resolve()
        test_path = Path(args.test_file).resolve()
        train_convs = json.loads(train_path.read_text(encoding="utf-8"))
        test_convs = json.loads(test_path.read_text(encoding="utf-8"))
        source_info = {
            "mode": "files",
            "train_file": str(train_path),
            "test_file": str(test_path),
            "data_path": None,
            "train_split": None,
            "test_split": None,
        }
        return train_convs, test_convs, source_info

    train_convs = load_abcd_data(args.train_split, args.data_path)
    test_convs = load_abcd_data(args.test_split, args.data_path)
    source_info = {
        "mode": "splits",
        "train_file": None,
        "test_file": None,
        "data_path": args.data_path,
        "train_split": args.train_split,
        "test_split": args.test_split,
    }
    return train_convs, test_convs, source_info


def _default_skill_text() -> str:
    return """---
name: abcd_trace2skill
description: Basic ABCD customer-service skill focused on choosing correct actions and slots
---

# ABCD Action-Slot Dialogue Skill

You are a customer service agent for retail support conversations.

## Primary objective

For each agent turn, first infer the correct backend action and required slot values,
then produce a short natural-language response that matches that action.

## Action discipline

- Predict the backend action before writing the response.
- Use slot values exactly when they are explicitly available in the dialogue context.
- Do not invent slot values that were not established by the customer or system state.
- If the action needs no slots, output no slots.
- If no backend action is needed, use `none`.

## Slot policy

For every ordered action argument, determine whether its value comes from the
latest customer utterance, earlier dialogue state, or stable scenario facts.
Use a value only after it is available for the active request. If a required
value is missing or unverified, ask/verify it or defer the dependent action;
never fabricate it. Reuse an earlier value only when it still refers to the
same customer, request, or entity, and always preserve the action's slot order.

## Response discipline

- The response must be consistent with the chosen action.
- Keep responses concise and helpful.
- Avoid promising actions that are not reflected in the backend action choice.

## Common failure patterns to avoid

- Correct response text but wrong backend action
- Correct action name but missing or misordered slot values
- Taking action too early before verification
- Using stale customer information from earlier turns
"""


def _load_skill_text(skill_path: Path) -> str:
    skill_path = skill_path.resolve()
    if not skill_path.exists():
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(_default_skill_text(), encoding="utf-8")
    return skill_path.read_text(encoding="utf-8")


def _chunk_list(items: list[Any], batch_size: int) -> list[list[Any]]:
    if not items:
        return []
    if batch_size <= 0 or batch_size >= len(items):
        return [items]
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _build_agent(
    model: str,
    workflow_text: str,
    response_logger: ResponseLogger,
    reference_text: str = "",
    expose_scenario_labels: bool = True,
) -> ABCDAgent:
    from awm import MemoryStore, WorkflowStore

    workflow = WorkflowStore()
    if workflow_text.strip():
        workflow.update(workflow_text)
    return ABCDAgent(
        model=model,
        workflow=workflow,
        memory=MemoryStore(),
        reference_text=reference_text,
        expose_scenario_labels=expose_scenario_labels,
        response_logger=response_logger,
    )


def _evaluate_turn_results(
    conversations: list[dict[str, Any]],
    turn_results: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    text_turns = [
        r for r in turn_results if r.get("target_type", "utterance") == "utterance"
    ]
    preds = [r["prediction"] for r in text_turns]
    refs = [r["reference"] for r in text_turns]
    text_eval = evaluate_text_records(preds, refs)

    abcd_preds = turn_results_to_abcd_predictions(turn_results, conversations)
    all_gt = [extract_ground_truth(conv) for conv in conversations]
    abcd_eval = evaluate_abcd(all_gt, abcd_preds)
    parsed_actions = sum(1 for r in turn_results if r.get("predicted_action"))
    direct_actions = sum(
        1 for r in turn_results
        if r.get("target_type") == "action" and r.get("predicted_action")
    )
    log.info(
        "%s diagnostics: all_targets=%d utterance_targets=%d parsed_actions=%d "
        "direct_action_predictions=%d gt_action_turns=%d",
        label, len(turn_results), len(text_turns), parsed_actions,
        direct_actions, abcd_eval.ast.total_action_turns,
    )
    if abcd_eval.ast.total_action_turns and direct_actions == 0:
        log.warning(
            "%s produced zero direct action predictions although the set "
            "contains %d action turns. Check raw LLM outputs and parser.",
            label, abcd_eval.ast.total_action_turns,
        )
    records = [
        {
            "conversation_id": pred.conversation_id,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "turn_type": t.turn_type,
                    "predicted_action": t.predicted_action,
                    "predicted_slots": t.predicted_slots,
                    "predicted_utterance_id": t.predicted_utterance_id,
                }
                for t in pred.turns
            ],
        }
        for pred in abcd_preds
    ]

    return {
        "label": label,
        "num_conversations": len(conversations),
        "num_turns": len(turn_results),
        "text": {
            "bert_f1": round(text_eval["bert_f1"], 4),
            "bleu_1": round(text_eval["bleu_1"], 1),
            "bleu_4": round(text_eval["bleu_4"], 1),
            "rouge_1": round(text_eval["rouge_1"], 4),
            "rouge_2": round(text_eval["rouge_2"], 4),
            "rouge_l": round(text_eval["rouge_l"], 4),
            "meteor": round(text_eval["meteor"], 4),
        },
        "ast_cds": {
            "ast_joint": round(abcd_eval.ast.joint_accuracy, 4),
            "ast_action_name": round(abcd_eval.ast.action_name_accuracy, 4),
            "ast_slot_value": round(abcd_eval.ast.slot_value_accuracy, 4),
            "cds_overall": round(abcd_eval.cds.overall_cds, 4),
            "num_action_turns": abcd_eval.ast.total_action_turns,
        },
        "num_all_targets": len(turn_results),
        "num_text_targets": len(text_turns),
        "num_parsed_action_predictions": parsed_actions,
        "num_direct_action_predictions": direct_actions,
        "abcd_predictions": records,
        "summary": (
            f"AST={abcd_eval.ast.joint_accuracy:.4f} "
            f"Action={abcd_eval.ast.action_name_accuracy:.4f} "
            f"Slot={abcd_eval.ast.slot_value_accuracy:.4f} "
            f"CDS={abcd_eval.cds.overall_cds:.4f} "
            f"BERT-F1={text_eval['bert_f1']:.4f} "
            f"BLEU-4={text_eval['bleu_4']:.1f} "
            f"ROUGE-L={text_eval['rouge_l']:.4f} "
            f"METEOR={text_eval['meteor']:.4f}"
        ),
    }


def _build_ast_mismatch_report(
    conversation: dict[str, Any],
    turn_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Build a verified AST mismatch report using the same mapping as evaluation."""
    convo_id = str(conversation.get("convo_id", "?"))
    predictions = turn_results_to_abcd_predictions(turn_results, [conversation])
    pred = predictions[0] if predictions else None
    pred_by_idx = {p.turn_index: p for p in (pred.turns if pred else [])}
    truths = extract_ground_truth(conversation)

    agent_rows = sorted(turn_results, key=lambda row: row["turn_index"])
    mismatches: list[dict[str, Any]] = []
    report_lines = [
        f"Conversation ID: {convo_id}",
        "Each row below is scored exactly as AST scores it: action name and slot values must both match.",
        "",
    ]

    for gt in truths:
        if gt.turn_type != "action":
            continue

        mapped_agent = None
        for row in agent_rows:
            if row["turn_index"] < gt.turn_index:
                mapped_agent = row
            else:
                break

        turn_pred = pred_by_idx.get(gt.turn_index)
        predicted_action = turn_pred.predicted_action if turn_pred else None
        predicted_slots = turn_pred.predicted_slots if turn_pred else None
        gold_slots = gt.slot_values or []
        pred_slots = predicted_slots or []
        action_ok = predicted_action == gt.action_name
        slots_ok = pred_slots == gold_slots

        if action_ok and slots_ok:
            continue

        item = {
            "action_turn_index": gt.turn_index,
            "source_agent_turn_index": mapped_agent.get("turn_index") if mapped_agent else None,
            "source_agent_context": mapped_agent.get("context", "") if mapped_agent else "",
            "source_agent_response": mapped_agent.get("prediction", "") if mapped_agent else "",
            "reference_agent_response": mapped_agent.get("reference", "") if mapped_agent else "",
            "predicted_action": predicted_action,
            "predicted_slots": pred_slots,
            "gold_action": gt.action_name,
            "gold_slots": gold_slots,
            "action_match": action_ok,
            "slots_match": slots_ok,
        }
        mismatches.append(item)

        report_lines.extend([
            f"## Mismatch at action turn {gt.turn_index}",
            f"- Source agent turn: {item['source_agent_turn_index']}",
            f"- Predicted action: {predicted_action}",
            f"- Gold action: {gt.action_name}",
            f"- Predicted slots: {pred_slots}",
            f"- Gold slots: {gold_slots}",
            f"- Action match: {action_ok}",
            f"- Slot match: {slots_ok}",
            "- Source agent context:",
            item["source_agent_context"][:1600],
            f"- Source agent response: {item['source_agent_response']}",
            f"- Reference agent response: {item['reference_agent_response']}",
            "",
        ])

    return mismatches, "\n".join(report_lines)


def _verify_corrected_actions(
    mismatches: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Check whether proposed corrections exactly fix the AST mismatches."""
    correction_by_turn = {
        int(c.get("turn_index")): c
        for c in corrections
        if str(c.get("turn_index", "")).isdigit()
    }
    feedback: list[str] = []
    all_ok = True

    for mismatch in mismatches:
        turn_index = int(mismatch["action_turn_index"])
        correction = correction_by_turn.get(turn_index)
        if correction is None:
            all_ok = False
            feedback.append(f"turn {turn_index}: missing correction")
            continue

        corrected_action = correction.get("corrected_action")
        corrected_slots = correction.get("corrected_slots") or []
        if not isinstance(corrected_slots, list):
            corrected_slots = [str(corrected_slots)]

        action_ok = corrected_action == mismatch["gold_action"]
        slots_ok = corrected_slots == mismatch["gold_slots"]
        if not action_ok or not slots_ok:
            all_ok = False
            feedback.append(
                "turn {turn}: expected action={gold_action!r}, slots={gold_slots!r}; "
                "got action={action!r}, slots={slots!r}".format(
                    turn=turn_index,
                    gold_action=mismatch["gold_action"],
                    gold_slots=mismatch["gold_slots"],
                    action=corrected_action,
                    slots=corrected_slots,
                )
            )

    return all_ok, "\n".join(feedback) if feedback else "All proposed corrections exactly match AST gold labels."


def _extract_corrections(report: str) -> list[dict[str, Any]]:
    """Extract the corrections JSON block from a verified analysis report."""
    import re

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", report, re.DOTALL)
    payload = fenced.group(1) if fenced else ""
    if not payload:
        start = report.find("{")
        end = report.rfind("}")
        payload = report[start:end + 1] if start != -1 and end != -1 and end > start else ""
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    corrections = data.get("corrections", [])
    return corrections if isinstance(corrections, list) else []


_ABCD_VERIFIED_ANALYSIS_SYSTEM = """You are an expert failure-analysis agent for ABCD task-oriented dialogue action tracking.

Your job is to diagnose why an agent failed AST (Action State Tracking), propose exact corrected action/slot labels for every mismatched action turn, and then write reusable skill lessons.

AST is strict: a turn is correct only when the action name exactly matches and the slot value list exactly matches in order.

You will receive a verified mismatch report. Do not guess beyond that report. Ground every cause in the provided context, predicted labels, and gold labels.

For every reusable skill lesson involving slots, explicitly identify the slot
policy failure: value source (latest customer utterance, prior dialogue state,
or scenario fact), availability/reuse condition, missing-value behavior, and
required order. Generalize the policy; never put example-specific customer
values into the lesson.

First output a JSON block with exact corrections:
```json
{
  "corrections": [
    {
      "turn_index": 0,
      "corrected_action": "action-name",
      "corrected_slots": ["slot1", "slot2"],
      "reason": "short evidence-based reason"
    }
  ]
}
```

Then output the analysis in this exact markdown format:

# Failure Cause Item 1
## Title
<One-line summary>
## Description
<What went wrong, citing the mismatched turn and evidence>
## Content
<What the agent should do instead, stated as actionable skill guidance>
## Relation to Skill
<Concrete skill update suggestion>

# Failure Memory Item 1
## Title
<Reusable pattern>
## Description
<When this pattern occurs>
## Content
<Concrete behavior to remember>
## Skill Reflection
<Where/how the skill should encode this lesson>

Output at least one Failure Cause Item and one Failure Memory Item."""


def _has_parseable_failure_items(report: str) -> bool:
    """Return True when Trace2Skill's parser will find at least one item."""
    return bool(_REPORT_ITEM_PATTERN.search(report))


def _build_parseable_report_suffix(
    case: dict[str, Any],
    corrections: list[dict[str, Any]],
    reason: str,
) -> str:
    """Create parser-compatible sections when the LLM report format drifts."""
    first = (case.get("ast_mismatches") or [{}])[0]
    first_correction = corrections[0] if corrections else {}
    turn_index = first.get("action_turn_index", "?")
    predicted_action = first.get("predicted_action")
    predicted_slots = first.get("predicted_slots", [])
    gold_action = first.get("gold_action") or first_correction.get("corrected_action", "")
    gold_slots = first.get("gold_slots") or first_correction.get("corrected_slots", [])
    return "\n".join([
        "",
        "<!-- Parser-compatible fallback appended by run_trace2skill_abcd.py.",
        f"Reason: {reason}",
        "-->",
        "",
        "# Failure Cause Item 1",
        "## Title",
        "Incorrect backend action or slot label for an ABCD action turn",
        "## Description",
        (
            f"At action turn {turn_index}, the model predicted action {predicted_action!r} "
            f"with slots {predicted_slots!r}, but AST verification requires action "
            f"{gold_action!r} with ordered slots {gold_slots!r}."
        ),
        "## Content",
        (
            "Before writing the natural-language response, identify the exact backend "
            "action label and ordered slot values required by the current subflow state. "
            "Do not rely on a plausible response alone, because AST scores the backend "
            "action and slot list directly."
        ),
        "## Relation to Skill",
        (
            "Add explicit action-slot tracking guidance: for each agent turn, decide the "
            "next backend action first, copy slot values exactly from the dialogue or "
            "scenario context, preserve slot order, and then make the response consistent "
            "with that action. State whether each value is newly collected, safely reused, "
            "or missing and therefore requires verification or deferral."
        ),
        "",
        "# Failure Memory Item 1",
        "## Title",
        "Verify action and ordered slots before response generation",
        "## Description",
        "ABCD failures can occur even when the response sounds reasonable if the backend label is wrong.",
        "## Content",
        (
            "For action-bearing turns, treat the backend action and ordered slot list as "
            "the primary prediction target. Generate the response only after the exact "
            "action-slot pair has been selected. Track each value's source, availability, "
            "and safe-reuse condition rather than guessing from a similar example."
        ),
        "## Skill Reflection",
        (
            "The skill should include a dedicated AST discipline section that prioritizes "
            "exact action labels, exact slot values, and order-sensitive slot matching."
        ),
    ])


def _build_verified_analysis_prompt(case: dict[str, Any], feedback: str | None = None) -> str:
    prompt = "\n".join([
        "## Failed ABCD Dialogue",
        f"Dialogue ID: {case.get('dialogue_id', 'N/A')}",
        f"Subflow: {case.get('domains', [])}",
        "",
        "## Scenario",
        case.get("goal_description", "")[:2000],
        "",
        "## AST Summary",
        f"Dialogue AST: {case.get('info_rate', 'N/A')}",
        f"Joint correct: {case.get('inform_correct', '?')}/{case.get('inform_total', '?')}",
        "",
        "## Verified AST Mismatch Report",
        case.get("ast_mismatch_report", "")[:9000],
        "",
        "## Full Trajectory",
        case.get("trajectory", "")[:5000],
    ])
    if feedback:
        prompt += "\n\n## Local Verification Feedback\n" + feedback
        prompt += "\nRevise the corrections JSON so every corrected action and corrected_slots exactly matches the verified AST labels."
    return prompt


def _fallback_verified_report(case: dict[str, Any], feedback: str) -> str:
    first = (case.get("ast_mismatches") or [{}])[0]
    return "\n".join([
        "```json",
        json.dumps({"corrections": []}, indent=2),
        "```",
        "",
        "# Failure Cause Item 1",
        "## Title",
        "AST mismatch analysis was not locally verified",
        "## Description",
        f"The analyzer did not produce corrections that passed local AST verification. {feedback}",
        "## Content",
        (
            "For each action turn, compare the predicted action name and ordered slot list "
            "against the verified AST labels before updating the skill."
        ),
        "## Relation to Skill",
        (
            "Add explicit guidance that ABCD actions must be selected as exact backend action "
            "labels with ordered slot values, not inferred from response text alone."
        ),
        "",
        "# Failure Memory Item 1",
        "## Title",
        "Verify action and slot labels before response wording",
        "## Description",
        "ABCD AST failures often come from plausible responses paired with incorrect backend labels.",
        "## Content",
        (
            f"On action turn {first.get('action_turn_index', '?')}, first determine the exact "
            "backend action and ordered slots, then write the response to match that decision."
        ),
        "## Skill Reflection",
        "The skill should prioritize exact action-slot tracking before natural-language response generation.",
    ])


def _save_verified_report(
    output_dir: Path,
    case: dict[str, Any],
    report: str,
    verified: bool,
    parseable: bool,
    parse_repaired: bool,
) -> str:
    did = case.get("dialogue_id", "unknown").replace("/", "_").replace("\\", "_")
    case_dir = output_dir / did
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    (case_dir / "verification.json").write_text(
        json.dumps({
            "verified": verified,
            "parseable": parseable,
            "parse_repaired": parse_repaired,
            "failure_log_path": case.get("failure_log_path"),
            "num_mismatches": len(case.get("ast_mismatches", [])),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if verified and parseable:
        (case_dir / "evaluate_passed.flag").write_text("passed\n", encoding="utf-8")
    return str(case_dir)


def _run_verified_abcd_error_analysis(
    failed_cases: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    response_logger: ResponseLogger,
    *,
    max_rounds: int = 2,
) -> list[str]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[str] = []

    for idx, case in enumerate(failed_cases, start=1):
        print(f"  Verified ABCD analysis {idx}/{len(failed_cases)}: {case.get('dialogue_id', '?')}")
        report = ""
        verified = False
        parseable = False
        parse_repaired = False
        feedback: str | None = None
        corrections: list[dict[str, Any]] = []

        for _ in range(max_rounds + 1):
            prompt = _build_verified_analysis_prompt(case, feedback)
            report = chat(
                [
                    {"role": "system", "content": _ABCD_VERIFIED_ANALYSIS_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.2,
                response_logger=response_logger,
            ).strip()
            corrections = _extract_corrections(report)
            verified, feedback_text = _verify_corrected_actions(
                case.get("ast_mismatches", []),
                corrections,
            )
            parseable = _has_parseable_failure_items(report)
            if verified and parseable:
                break
            feedback_parts = [feedback_text]
            if verified and not parseable:
                feedback_parts.append(
                    "The corrections are locally verified, but the markdown analysis "
                    "is not parseable. Use top-level headings exactly like "
                    "`# Failure Cause Item 1` and `# Failure Memory Item 1`, followed by "
                    "`## Title`, `## Description`, and `## Content` sections."
                )
            feedback = "\n".join(part for part in feedback_parts if part)

        if not report:
            report = _fallback_verified_report(case, feedback or "No LLM report was returned.")
            parseable = _has_parseable_failure_items(report)
        elif not verified:
            report = report + "\n\n<!-- Local AST verification failed:\n" + (feedback or "") + "\n-->\n"
            parseable = _has_parseable_failure_items(report)
        elif not parseable:
            report = report + _build_parseable_report_suffix(
                case,
                corrections,
                "LLM output did not match Trace2Skill parser headings.",
            )
            parse_repaired = True
            parseable = _has_parseable_failure_items(report)

        output_paths.append(_save_verified_report(
            output_dir,
            case,
            report,
            verified,
            parseable,
            parse_repaired,
        ))

    return output_paths


def _build_ast_failure_cases(
    conversations: list[dict[str, Any]],
    turn_results: list[dict[str, Any]],
    ast_scores: list[dict[str, Any]],
    *,
    log_dir: Path,
    hide_scenario_labels: bool = False,
) -> list[dict[str, Any]]:
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    by_convo: dict[str, list[dict[str, Any]]] = {}
    for row in turn_results:
        by_convo.setdefault(str(row["convo_id"]), []).append(row)

    failed_cases: list[dict[str, Any]] = []
    for conv, ast in zip(conversations, ast_scores):
        if ast.get("ast_score", 0.0) >= 1.0:
            continue

        convo_id = str(conv.get("convo_id", "?"))
        turns = sorted(by_convo.get(convo_id, []), key=lambda x: x["turn_index"])
        ast_mismatches, ast_mismatch_report = _build_ast_mismatch_report(conv, turns)
        trajectory_lines = []
        for row in turns:
            trajectory_lines.append(f"[Context upto turn {row['turn_index']}]")
            trajectory_lines.append(row.get("context", ""))
            trajectory_lines.append(f"[Predicted action] {row.get('predicted_action', '')}")
            trajectory_lines.append(f"[Predicted slots] {row.get('predicted_slots', [])}")
            trajectory_lines.append(f"[Predicted response] {row.get('prediction', '')}")
            trajectory_lines.append(f"[Reference response] {row.get('reference', '')}")
            trajectory_lines.append("")

        safe_id = convo_id.replace("/", "_").replace("\\", "_")
        log_path = log_dir / f"{safe_id}.json"
        log_payload = {
            "convo_id": convo_id,
            "ast_score": ast.get("ast_score", 0.0),
            "ast_mismatches": ast_mismatches,
            "turn_results": turns,
        }
        log_path.write_text(
            json.dumps(log_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        scenario = conv.get("scenario", {})
        failed_cases.append({
            "dialogue_id": f"abcd-{convo_id}",
            "failure_log_path": str(log_path),
            "ast_mismatches": ast_mismatches,
            "ast_mismatch_report": ast_mismatch_report,
            "domains": (
                [str(scenario.get("subflow", "unknown"))]
                if not hide_scenario_labels else ["mixed_abcd"]
            ),
            "goal_description": (
                json.dumps(scenario, ensure_ascii=False)
                if not hide_scenario_labels
                else "Mixed ABCD customer-service dialogue; infer the task from the trajectory and verified action labels."
            ),
            "info_rate": ast.get("ast_score", 0.0),
            "success": False,
            "inform_correct": ast.get("action_correct", 0),
            "inform_total": ast.get("action_total", 0),
            "request_correct": ast.get("action_correct", 0),
            "request_total": ast.get("action_total", 0),
            "booking_passed": None,
            "inform_slots": {},
            "request_slots": {},
            "booking": {},
            "goal_inform": {},
            "goal_request": {},
            "has_booking": False,
            "trajectory": "\n".join(trajectory_lines),
        })

    return failed_cases


def _run_error_analysis(
    failed_cases: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    response_logger: ResponseLogger,
) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _run_verified_abcd_error_analysis(
        failed_cases,
        output_dir,
        model,
        response_logger,
    )

    parsed_path = (output_dir.parent / f"{output_dir.name}_parsed.json").resolve()
    subprocess.run(
        [
            sys.executable,
            str(_TRACE2SKILL / "analysis" / "parse_error_analysis_outputs.py"),
            "--input_dir", str(output_dir),
            "--output", str(parsed_path),
        ],
        cwd=str(_TRACE2SKILL),
        check=True,
    )
    records = json.loads(parsed_path.read_text(encoding="utf-8"))
    if not records:
        report_count = len(list(output_dir.glob("*/analysis_report.md")))
        passed_count = len(list(output_dir.glob("*/evaluate_passed.flag")))
        failed_parse_count = len(list((output_dir / "failed_to_parse").glob("*/analysis_report.md")))
        debug = {
            "error": "Parsed zero error-analysis records",
            "output_dir": str(output_dir),
            "parsed_path": str(parsed_path),
            "analysis_reports": report_count,
            "evaluate_passed_flags": passed_count,
            "failed_to_parse_reports": failed_parse_count,
            "hint": (
                "The parser only keeps reports that have evaluate_passed.flag and "
                "parser-compatible '# Failure Cause Item N' / '# Failure Memory Item N' sections."
            ),
        }
        debug_path = output_dir.parent / f"{output_dir.name}_parse_debug.json"
        debug_path.write_text(json.dumps(debug, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(
            "Parsed zero error-analysis records. "
            f"reports={report_count}, passed_flags={passed_count}, "
            f"failed_to_parse={failed_parse_count}. Debug: {debug_path}"
        )
    return parsed_path


def _run_skill_evolution(
    records_path: Path,
    skill_path: Path,
    output_dir: Path,
    model: str,
    response_logger: ResponseLogger,
) -> list[str]:
    from skill_evolver.parallel_evolving_agent import ParallelSkillEvolver

    records_path = records_path.resolve()
    skill_path = skill_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not records:
        return []

    evolver = ParallelSkillEvolver(
        client=_ChatClientAdapter(model=model, response_logger=response_logger),
        skill_dir=str(skill_path.parent),
        batch_size=1,
        merge_batch_size=5,
        max_workers=3,
        max_merge_levels=5,
        temperature=0.3,
        verbose=True,
        dry_run=False,
        prompt_variant="generic",
        output_dir=output_dir,
        parse_failure_dir=output_dir.parent / "parse_failures",
        max_skill_lines=500,
        skip_translation=False,
        patch_pipeline="json",
    )
    result = evolver.run(records, input_mode="records")
    return result.get("changelog", [])


def run_pipeline(args) -> PipelineOutputs:
    from scripts.llm_usage_utils import reset_usage, get_usage, write_usage
    reset_usage()
    model = args.model
    _install_llm_wrappers(args.llm_qps, args.llm_max_retries, args.llm_retry_base_delay)

    train_convs, test_convs, source_info = _load_conversations(args)
    subflow = args.subflow.strip()
    train_convs = [
        conv for conv in train_convs
        if str(conv.get("scenario", {}).get("subflow", "")) == subflow
    ]
    test_convs = [
        conv for conv in test_convs
        if str(conv.get("scenario", {}).get("subflow", "")) == subflow
    ]
    if not train_convs or not test_convs:
        raise ValueError(
            f"Subflow {subflow!r} has no train/test conversations in the ABCD split"
        )
    if args.max_train:
        train_convs = train_convs[:args.max_train]
    if args.max_test:
        test_convs = test_convs[:args.max_test]

    resume_dir = getattr(args, "resume_dir", None)
    if resume_dir:
        out_dir = Path(resume_dir).resolve()
        if not out_dir.exists():
            raise FileNotFoundError(f"--resume-dir does not exist: {out_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = (Path(args.output_dir) / f"abcd_trace2skill_{timestamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(out_dir / "run.log", mode="a" if resume_dir else "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("abcd_trace2skill")
    if resume_dir:
        log.info("Resuming existing run from %s", out_dir)

    response_logger = ResponseLogger(out_dir / "llm_responses")

    previous_summary: dict[str, Any] = {}
    summary_path = out_dir / "summary.json"
    if resume_dir and summary_path.exists():
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    current_resume_config = {
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_file": str(Path(args.train_file).resolve()) if args.train_file else None,
        "test_file": str(Path(args.test_file).resolve()) if args.test_file else None,
        "subflow": subflow,
        "max_train": args.max_train,
        "max_test": args.max_test,
        "evolution_batch_size": args.evolution_batch_size,
        "max_evolution_batches": args.max_evolution_batches,
    }
    if resume_dir and previous_summary.get("config"):
        previous_config = previous_summary["config"]
        mismatches = []
        for key, value in current_resume_config.items():
            if key in previous_config and previous_config[key] != value:
                mismatches.append(
                    f"{key}: previous={previous_config[key]!r}, current={value!r}"
                )
        if mismatches:
            raise ValueError(
                "Resume configuration does not match the existing run:\n"
                + "\n".join(f"- {item}" for item in mismatches)
            )

    seed_skill_path = Path(args.skill_path).resolve()
    seed_skill_text = _load_skill_text(seed_skill_path)
    from eval_tod.reference_lookup import load_trace2skill_references
    seed_reference_text = load_trace2skill_references(seed_skill_path)
    evolved_skill_dir = out_dir / "evolved_skill"
    evolved_skill_dir.mkdir(parents=True, exist_ok=True)
    evolved_skill_path = evolved_skill_dir / "SKILL.md"
    if resume_dir and evolved_skill_path.exists():
        log.info("Using existing evolved skill: %s", evolved_skill_path)
    else:
        evolved_skill_path.write_text(seed_skill_text, encoding="utf-8")
    seed_references_dir = seed_skill_path.parent / "references"
    if seed_references_dir.exists() and seed_references_dir.is_dir() and not resume_dir:
        shutil.copytree(seed_references_dir, evolved_skill_dir / "references", dirs_exist_ok=True)
    seed_reference_file = seed_skill_path.parent / "reference.md"
    evolved_reference_file = evolved_skill_dir / "reference.md"
    if seed_reference_file.exists() and not evolved_reference_file.exists():
        shutil.copy2(seed_reference_file, evolved_reference_file)

    if source_info["mode"] == "files":
        log.info(
            "Loaded %d train and %d test conversations from files",
            len(train_convs),
            len(test_convs),
        )
        log.info("Train file: %s", source_info["train_file"])
        log.info("Test file: %s", source_info["test_file"])
    else:
        log.info(
            "Loaded %d train and %d test conversations from splits (%s/%s)",
            len(train_convs),
            len(test_convs),
            source_info["train_split"],
            source_info["test_split"],
        )
    log.info("LLM rate limit: %.2f QPS", args.llm_qps)
    log.info(
        "LLM retries: max_retries=%d, base_delay=%.1fs",
        args.llm_max_retries,
        args.llm_retry_base_delay,
    )

    # Stage 1: seed run on training set to mine failures
    seed_train_turns_path = out_dir / "seed_train_turns.json"
    seed_train_eval_path = out_dir / "seed_train_eval.json"
    if resume_dir and seed_train_turns_path.exists() and seed_train_eval_path.exists():
        log.info("Stage 1: reusing existing seed train outputs")
        seed_train_turns = json.loads(seed_train_turns_path.read_text(encoding="utf-8"))
        train_ast_scores = compute_ast_from_turn_results(train_convs, seed_train_turns)
        train_eval = json.loads(seed_train_eval_path.read_text(encoding="utf-8"))
    else:
        log.info("Stage 1: seed run on training set")
        seed_train_agent = _build_agent(
            model, seed_skill_text, response_logger, seed_reference_text,
            expose_scenario_labels=False,
        )
        seed_train_turns = seed_train_agent.generate_all_turn_predictions(
            train_convs,
            predict_actions=True,
        )
        seed_train_turns_path.write_text(
            json.dumps(seed_train_turns, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        train_ast_scores = compute_ast_from_turn_results(train_convs, seed_train_turns)
        train_eval = _evaluate_turn_results(train_convs, seed_train_turns, "seed_train")
        seed_train_eval_path.write_text(
            json.dumps(train_eval, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Seed train: %s", train_eval["summary"])

    seed_failed_cases = _build_ast_failure_cases(
        train_convs,
        seed_train_turns,
        train_ast_scores,
        log_dir=out_dir / "seed_failure_logs",
        hide_scenario_labels=False,
    )
    log.info("Seed AST failures on train: %d / %d", len(seed_failed_cases), len(train_convs))

    # Stage 2: iterative batch evolution.  Unlike the internal MAP batches in
    # ParallelSkillEvolver, these outer batches update the skill on disk after
    # each batch.  The next batch therefore reads and patches the already
    # evolved skill, matching the official Trace2Skill training loop.
    train_batches = _chunk_list(train_convs, args.evolution_batch_size)
    if args.max_evolution_batches:
        train_batches = train_batches[:args.max_evolution_batches]
    log.info(
        "Stage 2: iterative batch evolution (%d batches, batch_size=%s)",
        len(train_batches),
        args.evolution_batch_size if args.evolution_batch_size > 0 else "all",
    )

    existing_batch_history: list[dict[str, Any]] = []
    batch_history_path = out_dir / "batch_history.json"
    if resume_dir and batch_history_path.exists():
        existing_batch_history = json.loads(batch_history_path.read_text(encoding="utf-8"))
    elif resume_dir and previous_summary.get("batch_history"):
        existing_batch_history = previous_summary.get("batch_history", [])
    if resume_dir:
        by_batch = {
            str(row.get("batch")): row
            for row in existing_batch_history
            if row.get("batch")
        }
        for batch_summary_path in sorted((out_dir / "train_batches").glob("batch_*/batch_summary.json")):
            row = json.loads(batch_summary_path.read_text(encoding="utf-8"))
            if row.get("batch"):
                by_batch[str(row["batch"])] = row
        existing_batch_history = [by_batch[key] for key in sorted(by_batch)]

    completed_batches = {
        str(row.get("batch")): row
        for row in existing_batch_history
        if row.get("batch") and row.get("status", "completed") != "error"
    }
    changelog: list[str] = list(previous_summary.get("changelog", [])) if resume_dir else []
    if resume_dir and not changelog:
        for row in existing_batch_history:
            label = str(row.get("batch", "batch_unknown"))
            changelog.extend(f"{label}: {entry}" for entry in row.get("changelog", []))
    batch_history: list[dict[str, Any]] = [
        row for row in existing_batch_history if row.get("status", "completed") != "error"
    ]
    total_failed_cases = sum(int(row.get("failed_cases", 0)) for row in batch_history)

    for batch_idx, batch_convs in enumerate(train_batches, start=1):
        label = f"batch_{batch_idx:04d}"
        batch_dir = out_dir / "train_batches" / label
        batch_dir.mkdir(parents=True, exist_ok=True)

        if label in completed_batches:
            log.info("Stage 2.%d: %s already completed; skipping", batch_idx, label)
            continue

        current_skill_text = evolved_skill_path.read_text(encoding="utf-8")
        current_reference_text = load_trace2skill_references(evolved_skill_path)
        pre_skill_lines = len(current_skill_text.splitlines())
        log.info(
            "Stage 2.%d: %s with %d conversations (pre-skill lines=%d)",
            batch_idx,
            label,
            len(batch_convs),
            pre_skill_lines,
        )

        batch_agent = _build_agent(
            model,
            current_skill_text,
            response_logger,
            current_reference_text,
            expose_scenario_labels=False,
        )
        batch_turns = batch_agent.generate_all_turn_predictions(
            batch_convs,
            predict_actions=True,
        )
        (batch_dir / "turns.json").write_text(
            json.dumps(batch_turns, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        batch_ast_scores = compute_ast_from_turn_results(batch_convs, batch_turns)
        batch_eval = _evaluate_turn_results(batch_convs, batch_turns, label)
        (batch_dir / "eval.json").write_text(
            json.dumps(batch_eval, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("%s eval: %s", label, batch_eval["summary"])

        batch_failed_cases = _build_ast_failure_cases(
            batch_convs,
            batch_turns,
            batch_ast_scores,
            log_dir=batch_dir / "failure_logs",
            hide_scenario_labels=False,
        )
        total_failed_cases += len(batch_failed_cases)
        log.info(
            "%s AST failures: %d / %d",
            label,
            len(batch_failed_cases),
            len(batch_convs),
        )

        batch_changelog: list[str] = []
        parsed_path: Path | None = None
        if batch_failed_cases:
            try:
                error_dir = out_dir / "error_analysis" / label
                parsed_path = _run_error_analysis(
                    batch_failed_cases,
                    error_dir,
                    model,
                    response_logger,
                )
                log.info("%s parsed error analysis -> %s", label, parsed_path)

                batch_changelog = _run_skill_evolution(
                    parsed_path,
                    evolved_skill_path,
                    out_dir / "intermediates" / label,
                    model,
                    response_logger,
                )
                changelog.extend(f"{label}: {entry}" for entry in batch_changelog)
                log.info("%s applied %d evolution changes", label, len(batch_changelog))
            except Exception as exc:
                error_record = {
                    "batch": label,
                    "status": "error",
                    "num_conversations": len(batch_convs),
                    "failed_cases": len(batch_failed_cases),
                    "eval": batch_eval,
                    "parsed_error_analysis": str(parsed_path) if parsed_path else None,
                    "changelog": batch_changelog,
                    "pre_skill_lines": pre_skill_lines,
                    "post_skill_lines": len(evolved_skill_path.read_text(encoding="utf-8").splitlines()),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                (batch_dir / "batch_summary.json").write_text(
                    json.dumps(error_record, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                if args.continue_on_batch_error:
                    batch_history.append(error_record)
                    log.exception("%s failed; continuing because --continue-on-batch-error is set", label)
                    continue
                raise
        else:
            log.info("%s has no AST failures; skill unchanged", label)

        post_skill_lines = len(evolved_skill_path.read_text(encoding="utf-8").splitlines())
        batch_record = {
            "batch": label,
            "status": "completed",
            "num_conversations": len(batch_convs),
            "failed_cases": len(batch_failed_cases),
            "eval": batch_eval,
            "parsed_error_analysis": str(parsed_path) if parsed_path else None,
            "changelog": batch_changelog,
            "pre_skill_lines": pre_skill_lines,
            "post_skill_lines": post_skill_lines,
        }
        batch_history.append(batch_record)
        (batch_dir / "batch_summary.json").write_text(
            json.dumps(batch_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (out_dir / "batch_history.json").write_text(
        json.dumps(batch_history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(
        "Iterative evolution complete: %d failed cases across %d batches, %d changelog entries",
        total_failed_cases,
        len(train_batches),
        len(changelog),
    )

    seed_test_eval = None
    if args.skip_seed_test:
        log.info("Stage 4: skipping seed evaluation on test (--skip-seed-test)")
    else:
        log.info("Stage 4: seed evaluation on test")
        seed_test_agent = _build_agent(
            model, seed_skill_text, response_logger, seed_reference_text,
            expose_scenario_labels=False,
        )
        seed_test_turns = seed_test_agent.generate_all_turn_predictions(
            test_convs,
            predict_actions=True,
        )
        seed_test_eval = _evaluate_turn_results(test_convs, seed_test_turns, "seed_test")
        (out_dir / "seed_test_turns.json").write_text(
            json.dumps(seed_test_turns, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / "seed_test_eval.json").write_text(
            json.dumps(seed_test_eval, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / "seed_test_abcd_predictions.json").write_text(
            json.dumps(seed_test_eval["abcd_predictions"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Seed test: %s", seed_test_eval["summary"])

    # Stage 5: evolved test evaluation
    log.info("Stage 5: evolved evaluation on test")
    evolved_skill_text = evolved_skill_path.read_text(encoding="utf-8")
    evolved_reference_text = load_trace2skill_references(evolved_skill_path)
    evolved_test_agent = _build_agent(
        model,
        evolved_skill_text,
        response_logger,
        evolved_reference_text,
        expose_scenario_labels=False,
    )
    evolved_test_turns = evolved_test_agent.generate_all_turn_predictions(
        test_convs,
        predict_actions=True,
    )
    evolved_test_eval = _evaluate_turn_results(test_convs, evolved_test_turns, "evolved_test")
    (out_dir / "evolved_test_turns.json").write_text(
        json.dumps(evolved_test_turns, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "evolved_test_eval.json").write_text(
        json.dumps(evolved_test_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "evolved_test_abcd_predictions.json").write_text(
        json.dumps(evolved_test_eval["abcd_predictions"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Evolved test: %s", evolved_test_eval["summary"])

    summary = {
        "config": {
            "data_path": source_info["data_path"],
            "train_split": source_info["train_split"],
            "test_split": source_info["test_split"],
            "train_file": source_info["train_file"],
            "test_file": source_info["test_file"],
            "max_train": args.max_train,
            "max_test": args.max_test,
            "subflow": subflow,
            "model": model,
            "llm_qps": args.llm_qps,
            "llm_max_retries": args.llm_max_retries,
            "llm_retry_base_delay": args.llm_retry_base_delay,
            "skill_path": str(seed_skill_path),
            "seed_reference_chars": len(seed_reference_text),
            "skip_seed_test": args.skip_seed_test,
            "evolution_batch_size": args.evolution_batch_size,
            "max_evolution_batches": args.max_evolution_batches,
            "resume_dir": str(out_dir) if resume_dir else None,
            "continue_on_batch_error": args.continue_on_batch_error,
        },
        "seed_train": train_eval,
        "seed_test": seed_test_eval,
        "evolved_test": evolved_test_eval,
        "evolved_reference_chars": len(evolved_reference_text),
        "seed_failed_train_cases": len(seed_failed_cases),
        "iterative_failed_train_cases": total_failed_cases,
        "batch_history": batch_history,
        "changelog": changelog,
    }
    summary["llm_usage"] = get_usage()
    write_usage(out_dir / "llm_usage.json")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return PipelineOutputs(
        seed_eval=seed_test_eval,
        evolved_eval=evolved_test_eval,
        output_dir=out_dir,
        evolved_skill_path=evolved_skill_path,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ABCD Trace2Skill-style pipeline driven by AST",
    )
    parser.add_argument("--data-path", default=ABCD_DIR)
    parser.add_argument(
        "--train-file",
        default=None,
        help="Pre-split train conversations JSON file",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help="Pre-split test conversations JSON file",
    )
    parser.add_argument("--train-split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--test-split", default="test", choices=["train", "dev", "test"])
    parser.add_argument(
        "--subflow",
        required=True,
        help="Run exactly one ABCD subflow; repeat this command for each subflow",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument(
        "--evolution-batch-size",
        type=int,
        default=25,
        help=(
            "Outer training batch size for iterative skill evolution. "
            "Each batch patches the skill on disk before the next batch runs; "
            "set <=0 to evolve once over all training conversations."
        ),
    )
    parser.add_argument(
        "--max-evolution-batches",
        type=int,
        default=None,
        help="Optional cap on outer evolution batches for debugging",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--llm-qps",
        type=float,
        default=DEFAULT_LLM_QPS,
        help="Maximum LLM requests per second in this process; set <=0 to disable",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=3,
        help="Retry each failed LLM call up to this many times before surfacing the error",
    )
    parser.add_argument(
        "--llm-retry-base-delay",
        type=float,
        default=2.0,
        help="Base delay in seconds for exponential LLM retry backoff",
    )
    parser.add_argument("--skill-path", default=DEFAULT_SKILL_PATH)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--resume-dir",
        default=None,
        help=(
            "Resume an existing output directory. Reuses evolved_skill/SKILL.md, "
            "seed train outputs, and skips completed train_batches/*/batch_summary.json entries."
        ),
    )
    parser.add_argument(
        "--continue-on-batch-error",
        action="store_true",
        help="Record a failed training batch and continue with the next batch",
    )
    parser.add_argument(
        "--skip-seed-test",
        action="store_true",
        help="Skip seed baseline evaluation on the test set",
    )
    args = parser.parse_args()

    result = run_pipeline(args)
    evolved_ast = result.evolved_eval["ast_cds"]["ast_joint"]
    print("\n" + "=" * 60)
    print("ABCD TRACE2SKILL PIPELINE COMPLETE")
    print(f"Output:      {result.output_dir}")
    print(f"Skill:       {result.evolved_skill_path}")
    if result.seed_eval is not None:
        seed_ast = result.seed_eval["ast_cds"]["ast_joint"]
        print(f"Seed AST:    {seed_ast:.4f}")
        print(f"Delta AST:   {evolved_ast - seed_ast:+.4f}")
    else:
        print("Seed AST:    skipped")
    print(f"Evolved AST: {evolved_ast:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
