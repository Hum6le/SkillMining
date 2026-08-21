#!/usr/bin/env python3
"""AWM training and evaluation for one ABCD subflow.

Usage:
    python scripts/run_awm_abcd.py --subflow recover_username

What it does:
    1. Load one ABCD subflow from train/dev/test splits
    2. Offline mode: compile one frozen workflow and memory from train data
    3. Online mode: batch rollout → induce workflow → update memory
    4. Final evaluation on test set (all metrics)
    5. Save all outputs to outputs/awm_abcd_{timestamp}/
"""

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from eval_tod.cli import evaluate_abcd_bundle

# ── Config ────────────────────────────────────────────────────
ABCD_DIR = "data/eval/abcd/data"
BATCH_SIZE = 20             # dialogues per batch
MAX_BATCHES = None          # None = all batches
CHECKPOINT_EVERY = 10       # save workflow/memory checkpoint every N batches
MODEL = "deepseek-chat"
SEED = 42

# ── Setup ─────────────────────────────────────────────────────
_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# The full parallel launcher sets this to a unique method/subflow directory.
# Retain the timestamped default for direct, standalone invocations.
OUT_DIR = Path(os.environ.get("ABCD_OUTPUT_DIR", f"outputs/awm_abcd_{_TIMESTAMP}"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_TRACE_PATH = OUT_DIR / "training_turns.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "run.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _build_batches(items: list, size: int, max_batches: int | None = None) -> list[list]:
    """Split items into fixed-size batches."""
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    if max_batches:
        batches = batches[:max_batches]
    return batches


def _exemplars_to_reference_text(memory, hide_labels: bool) -> str:
    """Expose accumulated AWM exemplars to the shared reference tool."""
    sections = []
    for index, exemplar in enumerate(memory.exemplars, start=1):
        lines = [f"## AWM Exemplar {index}"]
        if not hide_labels:
            domains = ", ".join(str(x) for x in exemplar.get("domains", []))
            if domains:
                lines.append(f"Domains: {domains}")
            if exemplar.get("goal"):
                lines.append(f"Goal: {exemplar['goal']}")
        if exemplar.get("trajectory"):
            lines.append("Trajectory:")
            lines.append(str(exemplar["trajectory"]))
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def main():
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="AWM training on ABCD")
    _parser.add_argument("--resume-from", type=str, default=None,
                         help="Resume from a checkpoint directory (e.g. outputs/awm_abcd_xxx/checkpoints/batch_0050)")
    _parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate a saved AWM run/checkpoint",
    )
    _parser.add_argument(
        "--eval-from",
        type=str,
        default=None,
        help="Completed AWM output directory or checkpoint to evaluate",
    )
    _parser.add_argument(
        "--subflow",
        required=True,
        help="Run exactly one ABCD subflow; repeat this command for each subflow",
    )
    _parser.add_argument("--max-train", type=int, default=None)
    _parser.add_argument("--max-dev", type=int, default=None)
    _parser.add_argument("--max-test", type=int, default=None)
    _parser.add_argument(
        "--induction-mode",
        choices=("offline", "online"),
        default="offline",
        help=(
            "offline: compile one frozen workflow from the full train split; "
            "online: preserve legacy batch rollout and incremental updates"
        ),
    )
    _parser.add_argument(
        "--reference-path",
        default=None,
        help="Optional reference.md or Trace2Skill skill directory to load",
    )
    _parser.add_argument(
        "--workflow-max-chars", type=int, default=8000,
        help="Maximum workflow characters injected into one inference prompt",
    )
    _parser.add_argument(
        "--exemplar-max-chars", type=int, default=3000,
        help="Maximum retrieved exemplar characters injected into one inference prompt",
    )
    _args, _unknown = _parser.parse_known_args()

    if _args.eval_only and not _args.eval_from:
        _parser.error("--eval-only requires --eval-from")
    if _args.eval_only and _args.resume_from:
        _parser.error("Use --eval-from with --eval-only, not --resume-from")
    if _args.resume_from and _args.induction_mode != "online":
        _parser.error("--resume-from is supported only with --induction-mode online")

    from eval_tod.abcd.data import load_abcd_data
    from eval_tod.abcd.agent import ABCDAgent
    from eval_tod.response_logger import ResponseLogger
    from eval_tod.reference_lookup import load_skill_mining_reference, load_trace2skill_references
    from awm import MemoryStore, WorkflowStore

    # ── Load data ─────────────────────────────────────────────
    log.info("Loading ABCD dataset...")
    subflow = _args.subflow.strip()
    # Use the same deterministic per-subflow session split as Graph Mining
    # and Trace2Skill.  The flattened turn targets are built later by
    # ABCDAgent.generate_all_turn_predictions(); these files remain at the
    # conversation/session level so workflow induction sees full dialogues.
    split_dir = Path("data/eval/abcd/splits") / subflow
    split_train = split_dir / "train.json"
    split_test = split_dir / "test.json"
    if split_train.exists() and split_test.exists():
        train_convs = json.loads(split_train.read_text(encoding="utf-8"))
        test_convs = json.loads(split_test.read_text(encoding="utf-8"))
        # AWM does not currently use dev turns for model selection, but keep
        # the field populated for a stable summary and future validation.
        dev_convs = []
        log.info("Using shared subflow split files: %s", split_dir)
    else:
        log.warning("Shared split files not found; falling back to source splits")
        all_train = load_abcd_data("train", ABCD_DIR)
        all_dev = load_abcd_data("dev", ABCD_DIR)
        all_test = load_abcd_data("test", ABCD_DIR)
        keep = lambda conv: str(conv.get("scenario", {}).get("subflow", "")) == subflow
        train_convs = [conv for conv in all_train if keep(conv)]
        dev_convs = [conv for conv in all_dev if keep(conv)]
        test_convs = [conv for conv in all_test if keep(conv)]
    if not train_convs or not test_convs:
        raise ValueError(
            f"Subflow {subflow!r} has no train/test conversations in the ABCD split"
        )
    if _args.max_train:
        train_convs = train_convs[:_args.max_train]
    if _args.max_dev:
        dev_convs = dev_convs[:_args.max_dev]
    if _args.max_test:
        test_convs = test_convs[:_args.max_test]
    log.info(f"Train: {len(train_convs)}, Dev: {len(dev_convs)}, Test: {len(test_convs)}")

    # ── Build batches ─────────────────────────────────────────
    batches = _build_batches(train_convs, BATCH_SIZE, MAX_BATCHES)
    log.info(f"Batches: {len(batches)} (batch_size={BATCH_SIZE})")
    run_config = {
        "subflow": subflow,
        "induction_mode": _args.induction_mode,
        "max_train": _args.max_train,
        "max_dev": _args.max_dev,
        "max_test": _args.max_test,
        "batch_size": BATCH_SIZE,
        "max_batches": MAX_BATCHES,
    }

    # ── Init agent + memory ───────────────────────────────────
    logger = ResponseLogger(str(OUT_DIR / "llm_responses"))
    workflow = WorkflowStore()
    memory = MemoryStore()
    external_reference_text = ""
    if _args.reference_path:
        external_reference_text = load_skill_mining_reference(_args.reference_path)
        if not external_reference_text:
            external_reference_text = load_trace2skill_references(_args.reference_path)
        log.info("Loaded external reference: %d chars", len(external_reference_text))
    reference_text = external_reference_text

    eval_source_dir = None
    if _args.eval_only:
        eval_source_dir = Path(_args.eval_from).resolve()
        if not eval_source_dir.exists():
            raise FileNotFoundError(f"Evaluation source not found: {eval_source_dir}")
        saved_workflow = eval_source_dir / "awm_workflow.txt"
        saved_memory = eval_source_dir / "awm_exemplars.json"
        if not saved_workflow.exists():
            saved_workflow = eval_source_dir / "workflow.txt"
        if not saved_memory.exists():
            saved_memory = eval_source_dir / "exemplars.json"
        if saved_workflow.exists():
            workflow.load(str(saved_workflow))
        if saved_memory.exists():
            memory.load(str(saved_memory))
        saved_reference = eval_source_dir / "awm_reference.md"
        if saved_reference.exists():
            reference_text = saved_reference.read_text(encoding="utf-8")
        log.info(
            "Eval-only state loaded: workflow=%d lines, exemplars=%d, reference=%d chars",
            len(workflow), len(memory), len(reference_text),
        )

    # ── Resume from checkpoint ────────────────────────────────
    start_batch = 1
    if _args.resume_from:
        ckpt_dir = Path(_args.resume_from)
        if not ckpt_dir.exists():
            log.error(f"Checkpoint not found: {ckpt_dir}")
            sys.exit(1)

        # Load state
        state_path = ckpt_dir / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            start_batch = state.get("next_batch", 1)
            previous_config = state.get("config", {})
            mismatches = [
                f"{key}: previous={previous_config[key]!r}, current={value!r}"
                for key, value in run_config.items()
                if key in previous_config and previous_config[key] != value
            ]
            if mismatches:
                raise ValueError(
                    "Checkpoint configuration does not match the current run:\n"
                    + "\n".join(f"- {item}" for item in mismatches)
                )
        else:
            # Infer from directory name: batch_0050 → 50
            m = re.search(r"batch_(\d+)", ckpt_dir.name)
            start_batch = int(m.group(1)) + 1 if m else 1

        # Load workflow + memory
        wf_path = ckpt_dir / "workflow.txt"
        if wf_path.exists():
            workflow.load(str(wf_path))
        mem_path = ckpt_dir / "exemplars.json"
        if mem_path.exists():
            memory.load(str(mem_path))

        log.info(f"Resumed from {ckpt_dir}: batch {start_batch}/{len(batches)}, "
                 f"workflow={len(workflow)} lines, memory={len(memory)} exemplars")

    agent = ABCDAgent(
        model=MODEL, workflow=workflow, memory=memory,
        reference_text=reference_text,
        workflow_max_chars=_args.workflow_max_chars,
        exemplar_max_chars=_args.exemplar_max_chars,
        expose_scenario_labels=False,
        response_logger=logger,
    )

    # ── Batch training loop ───────────────────────────────────

    if not _args.eval_only and _args.induction_mode == "offline":
        from eval_tod.schemas import Prediction
        from eval_tod.abcd.agent import _build_abcd_turn_trajectory

        # Compile once from the complete, fixed train corpus. No train-time
        # rollouts, AST feedback, or intermediate workflow updates occur.
        demonstration_turns = []
        offline_predictions = []
        offline_metrics = []
        for conv in train_convs:
            convo_id = str(conv.get("convo_id", "?"))
            gold_rows = _build_abcd_turn_trajectory(conv, [])
            rows = [
                {
                    "convo_id": convo_id,
                    "turn_index": row["turn_index"],
                    "target_type": "action",
                    "predicted_action": row["gold_action"] or "",
                    "predicted_slots": row["gold_slots"],
                    "prediction": "",
                }
                for row in gold_rows
                if row["turn_type"] == "action"
            ]
            demonstration_turns.extend(rows)
            offline_predictions.append(Prediction(
                dialogue_id=f"abcd-{convo_id}",
                inform_slots={}, request_slots={}, booking={}, response_text="",
            ))
            offline_metrics.append({"action_total": 0, "action_correct": 0})

        log.info(
            "Offline induction: compiling a frozen workflow from %d train dialogues and %d gold action turns",
            len(train_convs), len(demonstration_turns),
        )
        agent.induce(
            train_convs, offline_predictions, offline_metrics,
            turn_results=demonstration_turns, offline_demonstrations=True,
        )

        # Freeze all train demonstrations as evidence. Unlike online AWM,
        # exemplars are not selected by the current agent's rollout accuracy.
        for conv in train_convs:
            convo_id = str(conv.get("convo_id", "?"))
            rows = [row for row in demonstration_turns if row["convo_id"] == convo_id]
            scenario = conv.get("scenario", {})
            structured = _build_abcd_turn_trajectory(conv, rows)
            memory.add_dict({
                "dialogue_id": f"abcd-{convo_id}",
                "domains": [scenario.get("flow", "?"), scenario.get("subflow", "?")],
                "goal": f"{scenario.get('flow', '?')}/{scenario.get('subflow', '?')}",
                "trajectory": json.dumps(structured, ensure_ascii=False)[:4000],
                "trajectory_turns": structured,
            })

        reference_text = "\n\n".join(
            part for part in [
                external_reference_text,
                _exemplars_to_reference_text(memory, hide_labels=False),
            ] if part
        )
        agent.set_reference_text(reference_text)
        (OUT_DIR / "offline_induction_manifest.json").write_text(
            json.dumps({
                "protocol": "offline_frozen_train_demonstrations",
                "train_dialogues": len(train_convs),
                "gold_action_turns": len(demonstration_turns),
                "workflow_lines": len(workflow),
                "memory_exemplars": len(memory),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    training_batches = [] if (_args.eval_only or _args.induction_mode == "offline") else batches
    for batch_idx, batch in enumerate(training_batches, start=1):
        if batch_idx < start_batch:
            continue  # skip already processed batches

        log.info(f"{'─'*40}")
        log.info(f"Batch {batch_idx}/{len(batches)}: {len(batch)} dialogues")

        # 1. Run agent (turn-level) + compute real AST
        from eval_tod.abcd.agent import compute_ast_from_turn_results
        from eval_tod.schemas import Prediction
        turn_results = agent.generate_all_turn_predictions(
            batch, predict_actions=True, verbose=False)
        with TRAINING_TRACE_PATH.open("a", encoding="utf-8") as trace_file:
            for row in turn_results:
                trace_file.write(json.dumps({
                    "batch_index": batch_idx,
                    **row,
                }, ensure_ascii=False) + "\n")
        log.info("  Saved %d training turn traces", len(turn_results))
        eval_dicts = compute_ast_from_turn_results(batch, turn_results)
        # Use last-turn predictions for induce
        preds = []
        for conv in batch:
            cid = str(conv.get("convo_id", "?"))
            conv_turns = [r for r in turn_results if r["convo_id"] == cid]
            last = conv_turns[-1]["prediction"] if conv_turns else ""
            preds.append(Prediction(
                dialogue_id=f"abcd-{cid}",
                inform_slots={}, request_slots={}, booking={},
                response_text=last,
            ))
        updated_workflow = agent.induce(
            batch,
            preds,
            eval_dicts,
            turn_results=turn_results,
        )
        agent.update_memory(
            batch,
            preds,
            eval_dicts,
            turn_results=turn_results,
        )
        eligible_exemplars = sum(
            1 for metrics in eval_dicts
            if int(metrics.get("action_total", 0) or 0) > 0
            and int(metrics.get("action_correct", 0) or 0)
            == int(metrics.get("action_total", 0) or 0)
        )
        log.info(
            "  Batch %d AST exemplar eligibility: %d/%d; memory_total=%d",
            batch_idx, eligible_exemplars, len(eval_dicts), len(memory),
        )
        reference_text = "\n\n".join(
            part for part in [
                external_reference_text,
                _exemplars_to_reference_text(
                    memory,
                    hide_labels=False,
                ),
            ] if part
        )
        agent.set_reference_text(reference_text)
        log.info(
            "Batch %d state: workflow_lines=%d, exemplars=%d, reference_chars=%d, induced=%s",
            batch_idx,
            len(workflow),
            len(memory),
            len(reference_text),
            bool(updated_workflow.strip()),
        )

        # 3. Checkpoint (with state for resume)
        if batch_idx % CHECKPOINT_EVERY == 0:
            ckpt_dir = OUT_DIR / "checkpoints" / f"batch_{batch_idx:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save_workflow(str(ckpt_dir / "workflow.txt"))
            agent.save_memory(str(ckpt_dir / "exemplars.json"))
            (ckpt_dir / "state.json").write_text(
                json.dumps({
                    "next_batch": batch_idx + 1,
                    "total_batches": len(batches),
                    "config": run_config,
                }),
                encoding="utf-8")
            log.info(f"  Checkpoint saved: {ckpt_dir}")

    # ── Final test evaluation ──────────────────────────────────
    log.info("=" * 50)
    log.info("Final test evaluation")
    test_agent = ABCDAgent(
        model=MODEL, workflow=workflow, memory=memory,
        reference_text=reference_text,
        workflow_max_chars=_args.workflow_max_chars,
        exemplar_max_chars=_args.exemplar_max_chars,
        expose_scenario_labels=False,
        response_logger=logger,
    )

    # Save turn-level predictions with actions (for AST/CDS + error analysis)
    log.info("  Generating turn-level predictions with actions...")
    test_turns = test_agent.generate_all_turn_predictions(
        test_convs, predict_actions=True)
    (OUT_DIR / "test_turn_predictions.json").write_text(
        json.dumps(test_turns, indent=2, ensure_ascii=False),
        encoding="utf-8")
    log.info(f"  Saved {len(test_turns)} turn predictions")

    # Derive dialogue-level text predictions from the same full turn run.
    # ``predict_and_save`` only evaluates the final agent utterance and would
    # launch a second, inconsistent generation pass.  Keep that final-turn
    # artifact for compatibility, but make the full turn file above the
    # source of truth for this evaluation.
    text_by_convo = {}
    for row in test_turns:
        if row.get("target_type", "utterance") != "utterance":
            continue
        text_by_convo[str(row["convo_id"])] = row.get("prediction", "")
    text_records = [
        {
            "dialogue_id": f"abcd-{conv.get('convo_id', '?')}",
            "response_text": text_by_convo.get(str(conv.get("convo_id", "?")), ""),
        }
        for conv in test_convs
    ]
    (OUT_DIR / "test_final_preds.json").write_text(
        json.dumps(text_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "test_turn_text_predictions.json").write_text(
        json.dumps(
            [r for r in test_turns if r.get("target_type", "utterance") == "utterance"],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ``test_turns`` is a flat list of per-agent-turn records, while the
    # shared evaluator consumes one ABCD prediction object per dialogue.
    # Convert before serializing so action turns are aligned consistently with
    # the training-time ``compute_ast_from_turn_results`` path.
    from eval_tod.abcd.agent import turn_results_to_abcd_predictions

    grouped_abcd_preds = turn_results_to_abcd_predictions(test_turns, test_convs)
    abcd_records = [
        {
            "conversation_id": pred.conversation_id,
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "turn_type": turn.turn_type,
                    "predicted_utterance_id": turn.predicted_utterance_id,
                    "predicted_action": turn.predicted_action,
                    "predicted_slots": turn.predicted_slots,
                }
                for turn in pred.turns
            ],
        }
        for pred in grouped_abcd_preds
    ]
    (OUT_DIR / "test_abcd_predictions.json").write_text(
        json.dumps(abcd_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    test_result = evaluate_abcd_bundle(
        test_convs,
        text_records=text_records,
        abcd_records=abcd_records,
        text_prediction_key="response_text",
    )
    log.info(f"Final test: {test_result['summary']}")

    # ── Save everything ───────────────────────────────────────
    agent.save_workflow(str(OUT_DIR / "awm_workflow.txt"))
    agent.save_memory(str(OUT_DIR / "awm_exemplars.json"))
    (OUT_DIR / "awm_reference.md").write_text(reference_text, encoding="utf-8")

    summary = {
        "config": {
            "batch_size": BATCH_SIZE, "max_batches": MAX_BATCHES,
            "checkpoint_every": CHECKPOINT_EVERY,
            "model": MODEL, "seed": SEED,
            "dataset": "abcd", "eval_mode": "single_subflow",
            "subflow": subflow,
            "induction_mode": _args.induction_mode,
            "max_train": _args.max_train, "max_dev": _args.max_dev,
            "max_test": _args.max_test,
            "workflow_max_chars": _args.workflow_max_chars,
            "exemplar_max_chars": _args.exemplar_max_chars,
            "reference_path": str(Path(_args.reference_path).resolve()) if _args.reference_path else None,
            "eval_only": bool(_args.eval_only),
            "eval_from": str(eval_source_dir) if eval_source_dir else None,
        },
        "data": {
            "train": len(train_convs), "dev": len(dev_convs),
            "test": len(test_convs), "batches": len(batches),
        },
        "final_test": test_result,
        "workflow_lines": len(workflow),
        "memory_exemplars": len(memory),
        "reference_chars": len(reference_text),
        "resource_policy": {
            "reference_lookup": "llm_planned_retrieve_reference",
            "exemplar_lookup": "runtime_domain_overlap_top_k",
            "exemplar_lookup_is_react_tool": False,
        },
        "llm_calls_logged": logger.count,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("=" * 50)
    log.info(f"DONE. Output: {OUT_DIR}")
    log.info(f"Final test: {test_result.get('summary', '')}")
    log.info(f"LLM calls logged: {logger.count}")
    return summary


if __name__ == "__main__":
    main()
