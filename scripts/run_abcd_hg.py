#!/usr/bin/env python3
r"""ABCD 全流程：意图划分 → 超图 → Vertex Cover → Seed Workflow → 训练+评估。

与 ``run_awm_abcd.py`` 的区别：
  - 不用空白 WorkflowStore 冷启动
  - 从训练数据的 session hypergraph 中挖掘 per-intent 算子集作为 seed workflow
  - 不跑 seed baseline（空 skill 评估无意义）

用法：
  python scripts/run_abcd_hg.py                        # 默认：subflow 分组 + 全量训练
  python scripts/run_abcd_hg.py --use-intent-classify  # LLM 意图分类分组
  python scripts/run_abcd_hg.py --max-train 500        # 限制训练数据量（快速验证）
  python scripts/run_abcd_hg.py --skip-training         # 仅做 skill mining，不训练
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Path setup ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

_SKILL_DIR = _PROJECT_ROOT / "skill_mining"
if str(_SKILL_DIR) in sys.path:
    sys.path.remove(str(_SKILL_DIR))
sys.path.insert(0, str(_SKILL_DIR))

from eval_tod.abcd.data import load_abcd_data
from eval_tod.cli import evaluate_abcd_bundle
from skill_mining.abcd_session_hg import (
    SessionHypergraph,
    greedy_vertex_cover,
    abcd_to_operator_results,
)

# ── Config ────────────────────────────────────────────────────
ABCD_DIR = "data/eval/abcd/data"
BATCH_SIZE = 20
MAX_BATCHES = None
VAL_EVERY = 5
CHECKPOINT_EVERY = 10
MODEL = "deepseek-chat"
SEED = 42

# Hypergraph params
RHO = 0.8
MAX_VERTICES_PER_INTENT = 1000
MIN_SESSIONS_PER_INTENT = 20  # 意图至少要有 N 个 session 才参与

# ── Setup ─────────────────────────────────────────────────────
_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = Path(f"outputs/abcd_hg_{_TIMESTAMP}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


# ═══════════════════════════════════════════════════════════════
# Phase 1: Skill Mining — 意图划分 → 超图 → Vertex Cover
# ═══════════════════════════════════════════════════════════════

def group_by_subflow(conversations: list[dict]) -> Dict[str, list[dict]]:
    """用 ABCD 自带的 subflow 标签分组。"""
    groups: Dict[str, list[dict]] = defaultdict(list)
    for conv in conversations:
        subflow = str(conv.get("scenario", {}).get("subflow", "unknown"))
        groups[subflow].append(conv)
    return dict(groups)


def mine_intent_skills(
    conversations: list[dict],
    intent_map: Dict[str, list[dict]],
    rho: float = RHO,
    max_vertices: int = MAX_VERTICES_PER_INTENT,
    min_sessions: int = MIN_SESSIONS_PER_INTENT,
) -> Dict[str, dict]:
    """对每个意图分组：建超图 → vertex cover → 返回算子集。

    Returns:
        {intent_name: {selected_vertices, coverage_pct, num_sessions, stats}}
    """
    results: Dict[str, dict] = {}

    for intent, convs in sorted(intent_map.items()):
        if len(convs) < min_sessions:
            log.info(f"  Skip '{intent}': {len(convs)} sessions < {min_sessions}")
            continue

        # 转 operator + 建超图
        op_results = abcd_to_operator_results(convs)
        if not op_results:
            log.info(f"  Skip '{intent}': no operators extracted")
            continue

        hg = SessionHypergraph.from_operator_results(op_results)
        stats = hg.stats()

        # Greedy vertex cover
        selected, _ = greedy_vertex_cover(hg, rho=rho, max_vertices=max_vertices)

        coverage = sum(
            1 for e in hg.hyperedges
            if len(selected & e.vertices) >= math.ceil(rho * e.size)
        )
        total_e = len(hg.hyperedges)

        results[intent] = {
            "selected_vertices": sorted(selected),
            "num_selected": len(selected),
            "coverage": coverage,
            "total_hyperedges": total_e,
            "coverage_pct": round(100 * coverage / max(total_e, 1), 1),
            "num_sessions": len(convs),
            "num_vertices_total": stats["num_vertices"],
            "avg_hyperedge_size": round(stats["avg_hyperedge_size"], 1),
        }

        log.info(
            f"  '{intent}': {len(convs)} sessions, "
            f"{stats['num_vertices']} ops, "
            f"{len(selected)} selected → {results[intent]['coverage_pct']:.0f}% coverage"
        )

    return results


def vertex_sets_to_per_intent_workflows(
    intent_skills: Dict[str, dict],
) -> Dict[str, str]:
    """将每个 intent 的算子集独立转换为 workflow text。

    Returns:
        {intent_name: workflow_text_for_this_intent_only}
    """
    if not intent_skills:
        return {}

    workflows: Dict[str, str] = {}
    for intent, info in intent_skills.items():
        vertices = info["selected_vertices"]
        if not vertices:
            continue

        ops_readable = []
        for v in vertices:
            parts = v.split(":", 1)
            if len(parts) == 2:
                _, op = parts
                ops_readable.append(f"`{op}`")
            else:
                ops_readable.append(f"`{v}`")

        workflows[intent] = (
            f"## Intent Workflow: {intent}\n"
            f"This conversation is about **{intent}**. "
            f"The following action patterns were mined from {info['num_sessions']} training examples "
            f"({info['coverage_pct']:.0f}% coverage).\n\n"
            f"**Key Actions** ({info['num_selected']}): {', '.join(ops_readable)}\n\n"
            f"**Strategy**: Follow the action sequence above. "
            f"Start with the most relevant action and proceed step by step.\n"
        )

    return workflows


def vertex_sets_to_merged_workflow(intent_skills: Dict[str, dict]) -> str:
    """所有 intent 的 workflow 合并成一个文本（训练时用）。"""
    per_intent = vertex_sets_to_per_intent_workflows(intent_skills)
    if not per_intent:
        return ""

    blocks = [
        "## Pre-Mined Intent Workflows (from Session Hypergraph + Vertex Cover)",
        "The following patterns were extracted from training data. "
        "Use the relevant one for the current conversation.\n",
    ]
    for intent, text in sorted(per_intent.items()):
        blocks.append(text)
    return "\n".join(blocks)


# ═══════════════════════════════════════════════════════════════
# Phase 2: Agent 训练 + 评估
# ═══════════════════════════════════════════════════════════════

def _build_batches(items: list, size: int, max_batches: int | None = None) -> list[list]:
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    if max_batches:
        batches = batches[:max_batches]
    return batches


def _get_subflow(conv: dict) -> str:
    """Extract subflow label from an ABCD conversation."""
    return str(conv.get("scenario", {}).get("subflow", "unknown"))


def _prediction_records(predictions: list[Any]) -> list[dict[str, str]]:
    """Convert dialogue-level Prediction objects to CLI-friendly records."""
    return [
        {
            "dialogue_id": pred.dialogue_id,
            "response_text": pred.response_text,
        }
        for pred in predictions
    ]


def _evaluate_with_per_intent_workflows(
    dialogues: list[dict],
    per_intent_workflows: Dict[str, str],
    trained_workflow,
    trained_memory,
    logger,
    label: str = "eval",
) -> tuple[list, dict]:
    """Per-intent 评估：按 subflow 分组，每组用对应的 workflow 推理。

    每个对话只注入它所属 subflow 的 skill，而非所有 skill 混在一起。
    """
    from eval_tod.abcd.agent import ABCDAgent
    from awm import WorkflowStore

    # 按 subflow 分组
    groups: Dict[str, list[dict]] = defaultdict(list)
    for conv in dialogues:
        groups[_get_subflow(conv)].append(conv)

    all_preds: list[Any] = []
    total = len(dialogues)

    for subflow, group in sorted(groups.items()):
        # 为这个 subflow 构建专属 agent
        wf = WorkflowStore()
        # 先注入该 subflow 的 seed skill
        if subflow in per_intent_workflows:
            wf.update(per_intent_workflows[subflow])
        # 再注入训练中归纳的通用 workflow
        if trained_workflow and trained_workflow.text:
            wf.update(trained_workflow.text)

        agent = ABCDAgent(
            model=MODEL, workflow=wf, memory=trained_memory,
            response_logger=logger,
        )
        preds = agent.generate_predictions(group)
        all_preds.extend(preds)

        log.info(f"  [{label}] {subflow}: {len(group)} dialogues "
                 f"(seed_skill={'✓' if subflow in per_intent_workflows else '✗'})")

    # 按原顺序重排 predictions
    result = evaluate_abcd_bundle(
        dialogues,
        text_records=_prediction_records(all_preds),
        text_prediction_key="response_text",
    )
    return all_preds, result


def run_training(
    train_convs: list[dict],
    dev_convs: list[dict],
    test_convs: list[dict],
    per_intent_workflows: Dict[str, str] | None = None,
    merged_seed_workflow: str = "",
    batch_size: int = BATCH_SIZE,
    max_batches: int | None = MAX_BATCHES,
):
    from scripts.llm_usage_utils import reset_usage, get_usage, write_usage, split_usage_summary
    reset_usage()
    """训练 + 评估。

    训练时用合并的 workflow（batch 内混合多种 subflow）。
    验证/测试时按 subflow 分组，每组只注入对应的 intent-specific skill。
    """
    from eval_tod.abcd.agent import ABCDAgent
    from eval_tod.response_logger import ResponseLogger
    from awm import MemoryStore, WorkflowStore

    if per_intent_workflows is None:
        per_intent_workflows = {}

    logger = ResponseLogger(str(OUT_DIR / "llm_responses"))
    memory = MemoryStore()

    # 训练时用合并 workflow（batch 内混合多种 subflow，agent 自行从 batch 中归纳）
    train_workflow = WorkflowStore()
    if merged_seed_workflow:
        train_workflow.update(merged_seed_workflow)
        log.info(f"Train seed workflow: {len(train_workflow)} lines")
    else:
        log.info("No seed workflow for training (blank start)")

    agent = ABCDAgent(
        model=MODEL, workflow=train_workflow, memory=memory,
        response_logger=logger,
    )

    # ── Batch training loop ───────────────────────────────────
    batches = _build_batches(train_convs, batch_size, max_batches)
    log.info(f"Batches: {len(batches)} (batch_size={batch_size})")
    log.info(f"Per-intent workflows available: {len(per_intent_workflows)} subflows")

    batch_metrics: list[dict] = []
    val_history: list[dict] = []

    for batch_idx, batch in enumerate(batches, start=1):
        log.info(f"{'─'*40}")
        log.info(f"Batch {batch_idx}/{len(batches)}: {len(batch)} dialogues")

        # 1. Run agent (训练时用合并 workflow)
        preds = agent.generate_predictions(batch)

        # 2. Evaluate
        result = evaluate_abcd_bundle(
            batch,
            text_records=_prediction_records(preds),
            text_prediction_key="response_text",
        )
        batch_metrics.append({"batch": batch_idx, "summary": result.get("summary", "")})
        log.info(f"  {result['summary']}")

        # 3. Build per-dialogue metrics
        text_metrics = result.get("text", {})
        per_sample = text_metrics.get("per_sample", []) if isinstance(text_metrics, dict) else []
        eval_dicts = [
            {"bert_f1": s.get("bert_f1", 0)} if isinstance(s, dict) else {"bert_f1": 0}
            for s in per_sample
        ]
        while len(eval_dicts) < len(batch):
            eval_dicts.append({"bert_f1": 0})

        # 4. Induce workflow from this batch
        agent.induce(batch, preds, eval_dicts)

        # 5. Update memory
        agent.update_memory(batch, preds, eval_dicts)

        # 6. Checkpoint
        if batch_idx % CHECKPOINT_EVERY == 0:
            ckpt_dir = OUT_DIR / "checkpoints" / f"batch_{batch_idx:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save_workflow(str(ckpt_dir / "workflow.txt"))
            agent.save_memory(str(ckpt_dir / "exemplars.json"))
            log.info(f"  Checkpoint saved: {ckpt_dir}")

        # 7. Validation — per-intent 评估
        if batch_idx % VAL_EVERY == 0:
            _, val_result = _evaluate_with_per_intent_workflows(
                dev_convs, per_intent_workflows,
                trained_workflow=agent.workflow,
                trained_memory=memory,
                logger=logger,
                label=f"val_batch_{batch_idx}",
            )
            val_history.append({
                "label": f"batch_{batch_idx}",
                **(val_result.get("text") or {}),
                **(val_result.get("ast_cds") or {}),
            })
            log.info(f"  Val (per-intent): {val_result['summary']}")

    # ── Final test — per-intent 评估 ──────────────────────────
    log.info("=" * 50)
    log.info("Final test evaluation (per-intent workflow selection)")
    generation_usage = get_usage()
    reset_usage()
    test_preds, test_result = _evaluate_with_per_intent_workflows(
        test_convs, per_intent_workflows,
        trained_workflow=agent.workflow,
        trained_memory=memory,
        logger=logger,
        label="test",
    )

    # Save test preds
    with open(OUT_DIR / "test_final_preds.json", "w", encoding="utf-8") as f:
        json.dump([{
            "dialogue_id": p.dialogue_id,
            "response_text": p.response_text,
        } for p in test_preds], f, indent=2, ensure_ascii=False)

    log.info(f"Final test (per-intent): {test_result['summary']}")

    # ── Save ───────────────────────────────────────────────────
    agent.save_workflow(str(OUT_DIR / "awm_workflow.txt"))
    agent.save_memory(str(OUT_DIR / "awm_exemplars.json"))
    llm_usage = split_usage_summary(generation_usage, get_usage())
    (OUT_DIR / "llm_usage.json").write_text(
        json.dumps(llm_usage, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "batch_metrics": batch_metrics,
        "val_history": val_history,
        "test_result": test_result,
        "final_workflow_lines": len(agent.workflow),
        "final_memory_exemplars": len(memory),
        "llm_calls_logged": logger.count,
        "llm_usage": llm_usage,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ABCD: Hypergraph Skill Mining → Training → Evaluation"
    )
    parser.add_argument("--max-train", type=int, default=None,
                        help="限制训练数据量")
    parser.add_argument("--max-dev", type=int, default=None,
                        help="限制验证数据量")
    parser.add_argument("--max-test", type=int, default=None,
                        help="限制测试数据量")

    # Intent grouping
    parser.add_argument("--use-builtin-intent", action="store_true", default=True,
                        help="使用 ABCD 自带的 subflow 标签作为意图分组（默认开启）")
    parser.add_argument("--no-builtin-intent", action="store_true",
                        help="关闭 builtin intent 分组，退化为全局超图（无意图划分）")
    parser.add_argument("--use-intent-classify", action="store_true",
                        help="使用 LLM 意图分类替代 subflow 分组")
    parser.add_argument("--intent-map", type=str, default=None,
                        help="已有的 intent_session_map.json 路径")

    # Training
    parser.add_argument("--skip-training", action="store_true",
                        help="仅做 skill mining，不训练")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=None)

    # Hypergraph params
    parser.add_argument("--rho", type=float, default=RHO,
                        help="Vertex cover 覆盖率阈值")
    parser.add_argument("--max-vertices", type=int, default=MAX_VERTICES_PER_INTENT,
                        help="每个意图最大选中顶点数")
    parser.add_argument("--min-sessions", type=int, default=MIN_SESSIONS_PER_INTENT,
                        help="最少 session 数阈值")
    args = parser.parse_args()

    # Resolve intent mode
    use_builtin = args.use_builtin_intent and not args.no_builtin_intent

    # ── 1. Load data ──────────────────────────────────────────
    log.info("Loading ABCD dataset...")
    train_convs = load_abcd_data("train", ABCD_DIR)
    dev_convs = load_abcd_data("dev", ABCD_DIR)
    test_convs = load_abcd_data("test", ABCD_DIR)

    if args.max_train:
        train_convs = train_convs[:args.max_train]
    if args.max_dev:
        dev_convs = dev_convs[:args.max_dev]
    if args.max_test:
        test_convs = test_convs[:args.max_test]

    log.info(f"Train: {len(train_convs)}, Dev: {len(dev_convs)}, Test: {len(test_convs)}")

    # ── 2. Skill Mining — 意图分组 ────────────────────────────
    log.info("=" * 50)
    log.info("Phase 1: Skill Mining — Intent Grouping + Hypergraph + Vertex Cover")

    intent_groups: Dict[str, list[dict]] = {}
    intent_mode = "none"

    if args.intent_map:
        intent_mode = "intent_map"
        log.info(f"Loading intent map from {args.intent_map}...")
        intent_map_raw = json.loads(Path(args.intent_map).read_text(encoding="utf-8"))
        conv_index = {str(c.get("convo_id", "")): c for c in train_convs}
        for intent, cids in intent_map_raw.items():
            convs = [conv_index[c] for c in cids if c in conv_index]
            if convs:
                intent_groups[intent] = convs
        log.info(f"  {len(intent_groups)} intents from map")

    elif args.use_intent_classify:
        intent_mode = "llm_classify"
        log.info("Running LLM intent classification...")
        from skill_mining.abcd_intent_classify import (
            format_abcd_dialogue_with_actions,
            process_batch,
            deduplicate_intents_with_llm,
            build_intent_map,
            load_intent_memory,
            save_intent_memory,
        )
        intent_out = _SKILL_DIR / "output" / "abcd_intent"
        intent_out.mkdir(parents=True, exist_ok=True)
        memory_path = intent_out / "intent_memory.json"

        dialogues = []
        for conv in train_convs:
            cid = str(conv.get("convo_id", "?"))
            text = format_abcd_dialogue_with_actions(conv)
            sf = str(conv.get("scenario", {}).get("subflow", ""))
            if text.strip():
                dialogues.append((cid, text, sf))

        intent_lib = load_intent_memory(memory_path)
        results, intent_lib = process_batch(
            dialogues, intent_lib, allow_new=True,
            stage_name="Stage 1: Extract + New Intents",
            output_dir=intent_out, memory_path=memory_path,
        )
        if len(intent_lib) > 1:
            intent_lib = deduplicate_intents_with_llm(intent_lib, intent_out)
            save_intent_memory(memory_path, intent_lib)
        results, intent_lib = process_batch(
            dialogues, intent_lib, allow_new=False,
            stage_name="Stage 2: Closed-World Classify",
            output_dir=intent_out, memory_path=memory_path,
        )

        intent_map_raw = build_intent_map(results)
        conv_index = {str(c.get("convo_id", "")): c for c in train_convs}
        for intent, cids in intent_map_raw.items():
            convs = [conv_index[c] for c in cids if c in conv_index]
            if convs:
                intent_groups[intent] = convs
        log.info(f"  {len(intent_groups)} intents from LLM classification")

    elif use_builtin:
        intent_mode = "builtin_subflow"
        log.info("Grouping by ABCD built-in subflow labels...")
        intent_groups = group_by_subflow(train_convs)
        log.info(f"  {len(intent_groups)} subflow groups")

    else:
        intent_mode = "none"
        log.info("No intent grouping — will use empty seed workflow")

    # ── 3. Skill Mining — 超图 + Vertex Cover ─────────────────
    intent_skills: Dict[str, dict] = {}

    if intent_groups:
        log.info(f"Mining intent skills (rho={args.rho}, max_vertices={args.max_vertices}, "
                 f"min_sessions={args.min_sessions})...")
        intent_skills = mine_intent_skills(
            train_convs, intent_groups,
            rho=args.rho, max_vertices=args.max_vertices,
            min_sessions=args.min_sessions,
        )
        log.info(f"Valid intents with skills: {len(intent_skills)}")
    else:
        log.info("No intent groups — skipping per-intent HG mining")

    # ── 4. 保存 skill mining 结果 ─────────────────────────────
    if intent_skills:
        skill_mining_output = {
            "config": {
                "rho": args.rho,
                "max_vertices": args.max_vertices,
                "min_sessions": args.min_sessions,
                "intent_mode": intent_mode,
                "num_intent_groups": len(intent_groups),
                "num_valid_intents": len(intent_skills),
            },
            "intent_skills": intent_skills,
        }
        skill_path = OUT_DIR / "mined_skills.json"
        skill_path.write_text(
            json.dumps(skill_mining_output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(f"Mined skills saved → {skill_path}")

    # ── 5. 转换为 per-intent workflow texts ────────────────────
    per_intent_workflows = vertex_sets_to_per_intent_workflows(intent_skills)

    # 保存 per-intent workflows
    wf_dir = OUT_DIR / "per_intent_workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    for intent, wf_text in per_intent_workflows.items():
        safe_name = intent.replace("/", "_").replace("\\", "_").replace(":", "_")[:50]
        (wf_dir / f"{safe_name}.txt").write_text(wf_text, encoding="utf-8")

    # 也保存一个合并版本（训练时参考）
    merged_wf = vertex_sets_to_merged_workflow(intent_skills)
    (OUT_DIR / "seed_workflow_merged.txt").write_text(merged_wf, encoding="utf-8")

    log.info(f"Per-intent workflows: {len(per_intent_workflows)} subflows → {wf_dir}")
    log.info(f"Merged workflow: {len(merged_wf.splitlines())} lines → {OUT_DIR / 'seed_workflow_merged.txt'}")

    # ── 6. Agent 训练 + 评估 ───────────────────────────────────
    if args.skip_training:
        log.info("Skipping training (--skip-training). Done.")
        return

    log.info("=" * 50)
    log.info(f"Phase 2: Agent Training + Evaluation "
             f"(intent_mode={intent_mode}, per_intent_skills={len(per_intent_workflows)})")
    log.info("Training: merged workflow  |  Eval: per-intent workflow selection")

    training_results = run_training(
        train_convs, dev_convs, test_convs,
        per_intent_workflows=per_intent_workflows,
        merged_seed_workflow=merged_wf,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    # ── 7. Final summary ───────────────────────────────────────
    summary = {
        "config": {
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "val_every": VAL_EVERY,
            "model": MODEL,
            "intent_mode": intent_mode,
            "rho": args.rho,
            "max_vertices_per_intent": args.max_vertices,
            "min_sessions_per_intent": args.min_sessions,
            "dataset": "abcd",
            "method": "hypergraph_vertex_cover_seeded",
        },
        "data": {
            "train": len(train_convs),
            "dev": len(dev_convs),
            "test": len(test_convs),
        },
        "skill_mining": {
            "intent_mode": intent_mode,
            "num_intent_groups": len(intent_groups),
            "num_valid_intents": len(intent_skills),
            "total_selected_operators": sum(
                len(v["selected_vertices"]) for v in intent_skills.values()
            ),
        },
        "training": {
            "num_batches": len(training_results["batch_metrics"]),
            "final_workflow_lines": training_results["final_workflow_lines"],
            "final_memory_exemplars": training_results["final_memory_exemplars"],
            "llm_calls_logged": training_results["llm_calls_logged"],
            "test_summary": training_results["test_result"].get("summary", ""),
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("=" * 50)
    log.info(f"DONE. Output: {OUT_DIR}")
    log.info(f"Intent mode: {intent_mode}, valid intents: {len(intent_skills)}")
    log.info(f"Final test: {training_results['test_result'].get('summary', '')}")


if __name__ == "__main__":
    main()
