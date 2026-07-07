#!/usr/bin/env python3
r"""ABCD 全指标评估：Skill Mining → Metadata → Agent 自选 Skill → 全指标评估。

评估指标：
  - Text: BERTScore, BLEU (1/4), ROUGE (1/2/L)
  - ABCD: AST (Action Name + Slot + Joint), CDS (Cascading Dialogue Success)
  - Skill Selection Accuracy: agent 选的 skill vs ground-truth subflow

用法：
  # 从 mined_skills.json 开始（跳过 mining）
  python scripts/run_abcd_eval.py \
    --skills outputs/abcd_hg_xxx/mined_skills.json \
    --split test --max-sessions 100

  # 完整流程：mining → metadata → eval
  python scripts/run_abcd_eval.py \
    --split test --max-sessions 100 --do-mining
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

_SKILL_DIR = _PROJECT_ROOT / "skill_mining"
if str(_SKILL_DIR) in sys.path:
    sys.path.remove(str(_SKILL_DIR))
sys.path.insert(0, str(_SKILL_DIR))

from eval_tod.abcd.data import load_abcd_data
from eval_tod.abcd.agent_skill import SkillSelectingAgent, compute_selection_accuracy
from skill_mining.abcd_session_hg import (
    SessionHypergraph,
    greedy_vertex_cover,
    abcd_to_operator_results,
    group_by_subflow,
    mine_per_subflow,
)

ABCD_DIR = "data/eval/abcd/data"
MODEL = "deepseek-chat"
RHO = 0.8
MAX_VERTICES = 30
MIN_SESSIONS = 2

_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = Path(f"outputs/abcd_eval_{_TIMESTAMP}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "eval.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ABCD Full Evaluation: Skill Cards + Self-Selecting Agent + All Metrics"
    )
    parser.add_argument("--skills", type=str, default=None,
                        help="mined_skills.json path (skip mining if provided)")
    parser.add_argument("--split", default="test",
                        choices=["train", "dev", "test"])
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Limit conversations for evaluation")
    parser.add_argument("--do-mining", action="store_true",
                        help="Run skill mining before evaluation (needs train data for mining, "
                             "then evaluates on --split)")
    parser.add_argument("--train-max-sessions", type=int, default=500,
                        help="Max train sessions for skill mining")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Pre-computed skill_metadata.json (skip metadata generation)")
    parser.add_argument("--rho", type=float, default=RHO)
    parser.add_argument("--max-vertices", type=int, default=MAX_VERTICES)
    parser.add_argument("--min-sessions", type=int, default=MIN_SESSIONS)
    parser.add_argument("--no-metadata", action="store_true",
                        help="Skip LLM metadata generation, use raw vertex sets as skill cards")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stop after skill card generation (no agent inference)")
    args = parser.parse_args()

    # ── 1. Get skills ──────────────────────────────────────────
    intent_skills: Dict[str, dict] = {}

    if args.do_mining:
        log.info(f"Loading ABCD train for skill mining...")
        train_convs = load_abcd_data("train", ABCD_DIR)
        if args.train_max_sessions:
            train_convs = train_convs[:args.train_max_sessions]
        log.info(f"  Train: {len(train_convs)} conversations")

        log.info(f"Mining per-subflow skills (rho={args.rho}, max_vertices={args.max_vertices})...")
        intent_groups = group_by_subflow(train_convs)
        intent_skills = mine_per_subflow(
            train_convs,
            rho=args.rho,
            max_vertices=args.max_vertices,
            min_sessions=args.min_sessions,
        )
        log.info(f"  {len(intent_skills)} subflows with valid skills")

        # Save mined skills
        with open(OUT_DIR / "mined_skills.json", "w", encoding="utf-8") as f:
            json.dump({"intent_skills": intent_skills}, f, indent=2, ensure_ascii=False)

    elif args.skills:
        log.info(f"Loading skills from {args.skills}")
        data = json.loads(Path(args.skills).read_text(encoding="utf-8"))
        intent_skills = data.get("intent_skills", data.get("per_intent", data))
        log.info(f"  {len(intent_skills)} subflows")

    else:
        log.error("Need --skills or --do-mining")
        sys.exit(1)

    # ── 2. Skill metadata ──────────────────────────────────────
    skill_cards_prompt: str = ""
    skill_metadata: Dict[str, dict] = {}

    if args.metadata:
        log.info(f"Loading skill metadata from {args.metadata}")
        skill_metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        from skill_mining.skill_metadata import format_skill_cards_prompt
        skill_cards_prompt = format_skill_cards_prompt(skill_metadata)
    elif not args.no_metadata and intent_skills:
        log.info("Generating skill metadata via LLM...")
        from skill_mining.skill_metadata import generate_skill_metadata, format_skill_cards_prompt
        skill_metadata = generate_skill_metadata(intent_skills)
        log.info(f"  {len(skill_metadata)} skill cards generated")

        # Save
        with open(OUT_DIR / "skill_metadata.json", "w", encoding="utf-8") as f:
            json.dump(skill_metadata, f, indent=2, ensure_ascii=False)

        skill_cards_prompt = format_skill_cards_prompt(skill_metadata)
        (OUT_DIR / "skill_cards_prompt.txt").write_text(skill_cards_prompt, encoding="utf-8")
    else:
        # Raw vertex sets as skill cards (no LLM metadata)
        log.info("Using raw vertex sets as skill cards (no LLM metadata)")
        cards = ["## Available Skills\n"]
        for name, info in sorted(intent_skills.items()):
            ops = info.get("selected_vertices", [])
            ops_clean = []
            for op in ops:
                parts = op.split(":", 1)
                ops_clean.append(parts[1] if len(parts) == 2 else op)
            cards.append(
                f"### [{name}]\n"
                f"- **When to use**: customer has a `{name}` request\n"
                f"- **Actions**: {' → '.join(ops_clean[:10])}\n"
            )
        skill_cards_prompt = "\n".join(cards)
        (OUT_DIR / "skill_cards_prompt.txt").write_text(skill_cards_prompt, encoding="utf-8")

    log.info(f"Skill cards prompt: {len(skill_cards_prompt.splitlines())} lines")

    if args.dry_run:
        log.info("Dry run — stopping after skill card generation.")
        log.info(f"Output: {OUT_DIR}")
        return

    # ── 3. Load eval data ─────────────────────────────────────
    log.info(f"Loading ABCD {args.split} for evaluation...")
    eval_convs = load_abcd_data(args.split, ABCD_DIR)
    if args.max_sessions:
        eval_convs = eval_convs[:args.max_sessions]
    log.info(f"  {len(eval_convs)} conversations")

    # ── 4. Run agent ───────────────────────────────────────────
    log.info("Creating SkillSelectingAgent...")
    agent = SkillSelectingAgent(model=MODEL)
    agent.set_skill_cards(skill_cards_prompt, skill_metadata)

    log.info(f"Generating predictions (both formats) for {len(eval_convs)} dialogues...")
    text_preds, abcd_preds = agent.generate_all_predictions(eval_convs)

    # Save predictions
    with open(OUT_DIR / "text_predictions.json", "w", encoding="utf-8") as f:
        json.dump([{
            "dialogue_id": p.dialogue_id,
            "response_text": p.response_text,
        } for p in text_preds], f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "abcd_predictions.json", "w", encoding="utf-8") as f:
        json.dump([{
            "conversation_id": p.conversation_id,
            "turns": [{
                "turn_index": t.turn_index,
                "turn_type": t.turn_type,
                "predicted_action": t.predicted_action,
                "predicted_slots": t.predicted_slots,
            } for t in p.turns],
        } for p in abcd_preds], f, indent=2, ensure_ascii=False)

    # ── 5. Full evaluation ─────────────────────────────────────
    log.info("=" * 50)
    log.info("Running full evaluation...")

    # 5a. Text metrics (ROUGE / BLEU / BERT)
    from eval_tod import evaluate_all
    log.info("\n--- Text Metrics ---")
    result = evaluate_all(eval_convs, text_preds, dataset_name="abcd")

    text_metrics = result.get("text", {})
    if text_metrics and "error" not in text_metrics:
        log.info(f"  BERT-F1:  {text_metrics.get('bert_f1', 'N/A'):.4f}")
        log.info(f"  BLEU-1:   {text_metrics.get('bleu_1', 'N/A'):.1f}")
        log.info(f"  BLEU-4:   {text_metrics.get('bleu_4', 'N/A'):.1f}")
        log.info(f"  ROUGE-1:  {text_metrics.get('rouge_1', 'N/A'):.4f}")
        log.info(f"  ROUGE-2:  {text_metrics.get('rouge_2', 'N/A'):.4f}")
        log.info(f"  ROUGE-L:  {text_metrics.get('rouge_l', 'N/A'):.4f}")

    # 5b. AST / CDS
    from eval_tod.abcd.data import extract_ground_truth
    from eval_tod.abcd.metrics import evaluate_abcd
    log.info("\n--- AST / CDS ---")
    all_gt = [extract_ground_truth(conv) for conv in eval_convs]
    abcd_result = evaluate_abcd(all_gt, abcd_preds)
    log.info(f"  AST Joint:       {abcd_result.ast.joint_accuracy:.4f}")
    log.info(f"  AST Action Name: {abcd_result.ast.action_name_accuracy:.4f}")
    log.info(f"  AST Slot Value:  {abcd_result.ast.slot_value_accuracy:.4f}")
    log.info(f"  CDS Overall:     {abcd_result.cds.overall_cds:.4f}")
    log.info(f"  Action turns:    {abcd_result.ast.total_action_turns}")

    # 5c. Skill selection accuracy
    log.info("\n--- Skill Selection Accuracy ---")
    sel_result = compute_selection_accuracy(agent.selection_log)
    log.info(f"  Accuracy: {sel_result['accuracy']:.4f} ({sel_result['correct']}/{sel_result['total']})")

    # Per-subflow breakdown
    log.info("\n  Per-subflow selection accuracy:")
    for sf, stats in sorted(sel_result["per_subflow"].items(),
                             key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / max(stats["total"], 1)
        log.info(f"    {sf:35s}  {acc:.0%} ({stats['correct']}/{stats['total']})")

    # ── 6. Save all results ────────────────────────────────────
    final = {
        "config": {
            "split": args.split,
            "max_sessions": args.max_sessions,
            "rho": args.rho,
            "max_vertices": args.max_vertices,
            "model": MODEL,
            "num_skills": len(intent_skills),
            "has_metadata": len(skill_metadata) > 0,
        },
        "text_metrics": text_metrics,
        "ast_cds": {
            "ast_joint": abcd_result.ast.joint_accuracy,
            "ast_action_name": abcd_result.ast.action_name_accuracy,
            "ast_slot_value": abcd_result.ast.slot_value_accuracy,
            "cds_overall": abcd_result.cds.overall_cds,
            "num_action_turns": abcd_result.ast.total_action_turns,
        },
        "skill_selection": sel_result,
        "selection_log": agent.selection_log,
    }
    with open(OUT_DIR / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'=' * 50}")
    log.info(f"DONE. Output: {OUT_DIR}")
    log.info(f"  eval_results.json          — all metrics")
    log.info(f"  text_predictions.json      — NL responses")
    log.info(f"  abcd_predictions.json      — action predictions (AST/CDS)")
    log.info(f"  skill_cards_prompt.txt     — skill cards shown to agent")
    log.info(f"\nSummary:")
    log.info(f"  BERT-F1:  {text_metrics.get('bert_f1', 'N/A'):.4f}")
    log.info(f"  BLEU-4:   {text_metrics.get('bleu_4', 'N/A'):.1f}")
    log.info(f"  ROUGE-L:  {text_metrics.get('rouge_l', 'N/A'):.4f}")
    log.info(f"  AST:      {abcd_result.ast.joint_accuracy:.4f}")
    log.info(f"  CDS:      {abcd_result.cds.overall_cds:.4f}")
    log.info(f"  Skill_Acc:{sel_result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
