#!/usr/bin/env python3
r"""单个 Subflow 的 Skill Mining → 直接评估（无 AWM 训练）。

流程：
  1. 加载该 subflow 的 train/test 数据
  2. Skill Mining：session HG → vertex cover → subgraph → skill.md
  3. Seed Baseline：空 workflow，逐轮预测 + 评估
  4. Mined Skill：加载 skill.md 作为 workflow，逐轮预测 + 评估
  5. 对比 seed vs mined

用法：
  # 单个 subflow
  python scripts/run_subflow_eval.py --subflow recover_password

  # 所有 subflow（逐个跑）
  python scripts/run_subflow_eval.py --all --min-sessions 50

  # 仅跑 seed baseline（跳过 mining）
  python scripts/run_subflow_eval.py --subflow recover_password --skip-mining \
    --skill-path outputs/skills/recover_password/skill.md
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))
_SKILL_DIR = _PROJECT_ROOT / "skill_mining"
if str(_SKILL_DIR) in sys.path:
    sys.path.remove(str(_SKILL_DIR))
sys.path.insert(0, str(_SKILL_DIR))

from eval_tod.abcd.data import load_abcd_data
from eval_tod.abcd.split import extract_all_agent_turns
from eval_tod.text_eval import evaluate_responses

SPLITS_DIR = Path("data/eval/abcd/splits")
MODEL = "deepseek-chat"

_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = Path(f"outputs/subflow_eval_{_TIMESTAMP}")
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
# Core
# ═══════════════════════════════════════════════════════════════

def load_subflow_data(subflow: str) -> tuple[list, list]:
    """Load train + test conversations for one subflow."""
    train_path = SPLITS_DIR / subflow / "train.json"
    test_path = SPLITS_DIR / subflow / "test.json"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Split data not found for {subflow}. "
                                f"Run: python scripts/split_abcd_by_intent.py")
    train = json.loads(train_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    return train, test


def mine_subflow_skill(subflow: str, train_convs: list) -> dict:
    """Mine skill from subflow training data: HG → vertex cover → subgraph."""
    from skill_mining.abcd_session_hg import (
        SessionHypergraph, greedy_vertex_cover, abcd_to_operator_results,
    )
    from skill_mining.subgraph_mining import (
        build_dag_from_operator_results, extract_subgraph,
        find_main_pathway, find_branch_points,
    )

    # Convert to operators → HG → vertex cover
    op_results = abcd_to_operator_results(train_convs)
    hg = SessionHypergraph.from_operator_results(op_results)
    selected, _ = greedy_vertex_cover(hg, rho=0.8, max_vertices=30)

    coverage = sum(1 for e in hg.hyperedges
                   if len(selected & e.vertices) >= 0.8 * e.size)
    n_edges = len(hg.hyperedges)

    # Build DAG → subgraph
    dag = build_dag_from_operator_results(op_results)
    sub = extract_subgraph(dag, selected)

    # Pathways + branches
    pathways = find_main_pathway(sub)
    branches = find_branch_points(sub)

    skill_info = {
        "selected_vertices": sorted(selected),
        "num_selected": len(selected),
        "coverage_pct": round(100 * coverage / max(n_edges, 1), 1),
        "num_sessions": len(train_convs),
    }

    subgraph = {
        **sub, "pathways": pathways, "branch_points": branches,
        "n_sessions": len(train_convs),
        "coverage_pct": skill_info["coverage_pct"],
    }

    # Generate reference.md (operator → dialogue snippets)
    from skill_mining.skill_writer import _find_operator_snippets, build_reference_md
    operators = skill_info["selected_vertices"]
    op_snippets = _find_operator_snippets(train_convs, subflow, operators)
    reference_md = build_reference_md(subflow, op_snippets, max_snippets_per_op=5)

    return {"skill_info": skill_info, "subgraph": subgraph,
            "operator_results": op_results, "reference_md": reference_md}


def build_workflow_from_skill(subflow: str, skill_info: dict, subgraph: dict) -> str:
    """Build workflow text from mined skill (use skill.md format)."""
    from skill_mining.skill_writer import build_skill_md_from_subgraph
    return build_skill_md_from_subgraph(subflow, subgraph, {}, use_llm=False)


def evaluate_agent_on_subflow(
    agent, test_convs: list, label: str,
) -> dict:
    """Run turn-level predictions + evaluation."""
    turn_results = agent.generate_all_turn_predictions(
        test_convs, predict_actions=True, verbose=False)
    preds = [r["prediction"] for r in turn_results]
    refs = [r["reference"] for r in turn_results]
    text_result = evaluate_responses(preds, refs)

    # AST from turn results
    from eval_tod.abcd.agent import compute_ast_from_turn_results
    ast_scores = compute_ast_from_turn_results(test_convs, turn_results)
    ast_mean = sum(s["ast_score"] for s in ast_scores) / max(len(ast_scores), 1)

    return {
        "label": label,
        "n_turns": len(preds),
        "text": {
            "bert_f1": round(text_result.bert_f1, 4),
            "bleu_4": round(text_result.bleu_4, 1),
            "rouge_l": round(text_result.rouge_l, 4),
        },
        "ast_mean": round(ast_mean, 4),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Subflow Skill Mining → Direct Evaluation")
    parser.add_argument("--subflow", default=None,
                        help="Single subflow name (e.g. recover_password)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all subflows in splits/")
    parser.add_argument("--min-sessions", type=int, default=50,
                        help="Min training sessions for --all")
    parser.add_argument("--skip-mining", action="store_true",
                        help="Skip mining, use pre-built skill.md")
    parser.add_argument("--skill-path", default=None,
                        help="Path to pre-built skill.md (for --skip-mining)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    # Determine subflows to process
    if args.all:
        if not SPLITS_DIR.exists():
            log.error("No splits found. Run: python scripts/split_abcd_by_intent.py")
            sys.exit(1)
        subflows = sorted(
            d.name for d in SPLITS_DIR.iterdir()
            if d.is_dir() and (d / "train.json").exists()
        )
        # Filter by min_sessions
        filtered = []
        for sf in subflows:
            train = json.loads((SPLITS_DIR / sf / "train.json").read_text(encoding="utf-8"))
            if len(train) >= args.min_sessions:
                filtered.append(sf)
        subflows = filtered
        log.info(f"Processing {len(subflows)} subflows (min_sessions={args.min_sessions})")
    elif args.subflow:
        subflows = [args.subflow]
    else:
        log.error("Need --subflow or --all")
        sys.exit(1)

    all_results = {}

    for subflow in subflows:
        log.info(f"\n{'='*50}")
        log.info(f"Subflow: {subflow}")
        log.info(f"{'='*50}")

        # ── 1. Load data ──────────────────────────────────────
        train_convs, test_convs = load_subflow_data(subflow)
        if args.max_train:
            train_convs = train_convs[:args.max_train]
        if args.max_test:
            test_convs = test_convs[:args.max_test]
        log.info(f"  Train: {len(train_convs)}, Test: {len(test_convs)}")

        # ── 2. Mine skill ─────────────────────────────────────
        if args.skip_mining:
            skill_text = ""
            if args.skill_path:
                skill_text = Path(args.skill_path).read_text(encoding="utf-8")
            skill_info = {"selected_vertices": [], "coverage_pct": 0, "num_sessions": 0}
        else:
            mined = mine_subflow_skill(subflow, train_convs)
            skill_info = mined["skill_info"]
            skill_text = build_workflow_from_skill(subflow, skill_info, mined["subgraph"])

            # Save
            sf_out = OUT_DIR / subflow
            sf_out.mkdir(parents=True, exist_ok=True)
            (sf_out / "skill.md").write_text(skill_text, encoding="utf-8")
            (sf_out / "reference.md").write_text(
                mined.get("reference_md", ""), encoding="utf-8")
            (sf_out / "subgraph.json").write_text(
                json.dumps(mined["subgraph"], indent=2, ensure_ascii=False),
                encoding="utf-8")
            n_snippets = sum(1 for v in mined.get("reference_md", "").split("\n")
                            if v.startswith("```text"))
            log.info(f"  Mined: {skill_info['num_selected']} vertices, "
                     f"{skill_info['coverage_pct']:.0f}% coverage, "
                     f"{n_snippets} reference snippets")

        # ── 3. Seed Baseline ──────────────────────────────────
        from eval_tod.abcd.agent import ABCDAgent
        from awm import WorkflowStore, MemoryStore

        log.info("  Seed baseline...")
        seed_agent = ABCDAgent(
            model=args.model, workflow=WorkflowStore(), memory=MemoryStore())
        seed_result = evaluate_agent_on_subflow(seed_agent, test_convs, "seed")
        log.info(f"    BERT={seed_result['text']['bert_f1']:.4f}  "
                 f"BLEU-4={seed_result['text']['bleu_4']:.1f}  AST={seed_result['ast_mean']:.4f}")

        # ── 4. Mined Skill ────────────────────────────────────
        log.info("  Mined skill evaluation...")
        wf = WorkflowStore()
        if skill_text:
            wf.update(skill_text)
        mined_agent = ABCDAgent(
            model=args.model, workflow=wf, memory=MemoryStore())
        mined_result = evaluate_agent_on_subflow(mined_agent, test_convs, "mined")
        log.info(f"    BERT={mined_result['text']['bert_f1']:.4f}  "
                 f"BLEU-4={mined_result['text']['bleu_4']:.1f}  AST={mined_result['ast_mean']:.4f}")

        # ── 5. Delta ──────────────────────────────────────────
        delta_bert = mined_result['text']['bert_f1'] - seed_result['text']['bert_f1']
        delta_ast = mined_result['ast_mean'] - seed_result['ast_mean']
        log.info(f"  Δ BERT={delta_bert:+.4f}  Δ AST={delta_ast:+.4f}")

        all_results[subflow] = {
            "train_sessions": len(train_convs),
            "test_sessions": len(test_convs),
            "skill_vertices": skill_info.get("num_selected", 0),
            "coverage_pct": skill_info.get("coverage_pct", 0),
            "seed": seed_result,
            "mined": mined_result,
            "delta": {"bert_f1": round(delta_bert, 4), "ast": round(delta_ast, 4)},
        }

    # ── 6. Summary ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"SUMMARY")
    print(f"{'='*55}")
    print(f"{'Subflow':35s} {'ΔBERT':>8s} {'ΔAST':>8s} {'Seed':>8s} {'Mined':>8s}")
    print("-" * 72)
    for sf, r in sorted(all_results.items(), key=lambda x: -x[1]["delta"]["bert_f1"]):
        d = r["delta"]
        s = r["seed"]["text"]["bert_f1"]
        m = r["mined"]["text"]["bert_f1"]
        print(f"{sf:35s} {d['bert_f1']:+.4f} {d['ast']:+.4f} {s:.4f} {m:.4f}")

    (OUT_DIR / "summary.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
