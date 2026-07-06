#!/usr/bin/env python3
r"""ABCD Per-Intent Session Hypergraph → Vertex Cover。

加载意图分类结果，为每个意图独立构建 session hypergraph，
运行 greedy vertex cover，并对比全局 vs per-intent 的覆盖效果。

用法：
  # 先跑意图分类得到 intent_session_map.json
  python skill_mining/abcd_intent_classify.py --split test --max-sessions 100

  # 再跑 per-intent 超图分析
  python skill_mining/abcd_per_intent_hg.py \
    --intent-map skill_mining/output/abcd_intent/intent_session_map.json \
    --split test --rho 0.8

  # 如果不指定 --intent-map，使用 subflow 作为意图（快速测试）
  python skill_mining/abcd_per_intent_hg.py --split test --use-subflow
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ── Path setup ────────────────────────────────────────────────
_SKILL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SKILL_DIR.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SKILL_DIR) in sys.path:
    sys.path.remove(str(_SKILL_DIR))
sys.path.insert(0, str(_SKILL_DIR))

from eval_tod.abcd.data import load_abcd_data
from abcd_session_hg import (
    SessionHypergraph,
    greedy_vertex_cover,
    abcd_to_operator_results,
)

_OUTPUT_DIR = _SKILL_DIR / "output" / "abcd_intent"


# ── Per-Intent 超图构建 ─────────────────────────────────────────

def build_per_intent_hypergraphs(
    conversations: list[dict],
    intent_map: Dict[str, List[str]],
    min_sessions: int = 2,
) -> Dict[str, SessionHypergraph]:
    """为每个意图构建独立的 session hypergraph。

    Args:
        conversations: 全部 ABCD conversations
        intent_map: intent → [convo_id] 映射
        min_sessions: 最少 session 数，低于此值的意图被跳过

    Returns:
        intent → SessionHypergraph 字典
    """
    # 建立 convo_id → conv 的快速索引
    convo_index: Dict[str, dict] = {}
    for conv in conversations:
        cid = str(conv.get("convo_id", "?"))
        convo_index[cid] = conv

    per_intent_hg: Dict[str, SessionHypergraph] = {}

    for intent, convo_ids in sorted(intent_map.items()):
        if len(convo_ids) < min_sessions:
            print(f"  跳过 '{intent}': 仅 {len(convo_ids)} 个 session（< {min_sessions}）")
            continue

        # 筛选该 intent 的 conversations
        intent_convs = [convo_index[cid] for cid in convo_ids if cid in convo_index]
        if len(intent_convs) < min_sessions:
            print(f"  跳过 '{intent}': 匹配到 {len(intent_convs)} 个 conv（< {min_sessions}）")
            continue

        # 转换 operator + 建超图
        results = abcd_to_operator_results(intent_convs)
        if not results:
            print(f"  跳过 '{intent}': 无可提取的 operator")
            continue

        hg = SessionHypergraph.from_operator_results(results)
        per_intent_hg[intent] = hg
        print(f"  '{intent}': {len(results)} sessions, "
              f"{hg.stats()['num_vertices']} vertices, "
              f"{hg.stats()['num_hyperedges']} hyperedges")

    return per_intent_hg


# ── Per-Intent Vertex Cover ─────────────────────────────────────

def run_per_intent_vertex_cover(
    per_intent_hg: Dict[str, SessionHypergraph],
    rho: float = 0.8,
    max_vertices: int = 30,
) -> Dict[str, dict]:
    """对每个 intent 的超图运行 greedy vertex cover。

    Returns:
        intent → {selected_vertices, coverage, stats, ...} 字典
    """
    results: Dict[str, dict] = {}

    for intent, hg in sorted(per_intent_hg.items()):
        selected, history = greedy_vertex_cover(hg, rho=rho, max_vertices=max_vertices)
        stats = hg.stats()

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
            "num_vertices_total": stats["num_vertices"],
            "avg_hyperedge_size": round(stats["avg_hyperedge_size"], 1),
            "rho": rho,
            "max_vertices": max_vertices,
        }

        print(f"  '{intent}': {len(selected)} vertices → "
              f"{coverage}/{total_e} coverage "
              f"({results[intent]['coverage_pct']:.1f}%)")

    return results


# ── 桥接算子识别 ────────────────────────────────────────────────

def find_bridge_operators(
    per_intent_results: Dict[str, dict],
    min_intents: int = 2,
) -> List[dict]:
    """识别跨多个 intent 的桥接算子（共享算子）。

    Args:
        per_intent_results: per-intent vertex cover 结果
        min_intents: 至少出现在 N 个 intent 的 vertex set 中才算桥接

    Returns:
        排序后的桥接算子列表 [{operator, intents, num_intents}]
    """
    # operator → 出现该算子的 intent 列表
    op_intents: Dict[str, List[str]] = defaultdict(list)

    for intent, result in per_intent_results.items():
        for op in result["selected_vertices"]:
            op_intents[op].append(intent)

    bridges = [
        {
            "operator": op,
            "intents": intents,
            "num_intents": len(intents),
        }
        for op, intents in op_intents.items()
        if len(intents) >= min_intents
    ]
    bridges.sort(key=lambda x: -x["num_intents"])

    return bridges


# ── 全局 vs Per-Intent 覆盖对比 ─────────────────────────────────

def compare_global_vs_per_intent(
    conversations: list[dict],
    per_intent_hg: Dict[str, SessionHypergraph],
    per_intent_results: Dict[str, dict],
    rho: float = 0.8,
    max_vertices: int = 30,
) -> dict:
    """对比全局 vertex cover 和 per-intent vertex cover 的覆盖效果。"""
    # 全局超图
    print("\n构建全局超图...")
    global_results = abcd_to_operator_results(conversations)
    global_hg = SessionHypergraph.from_operator_results(global_results)
    global_selected, _ = greedy_vertex_cover(global_hg, rho=rho, max_vertices=max_vertices)
    global_coverage = sum(
        1 for e in global_hg.hyperedges
        if len(global_selected & e.vertices) >= math.ceil(rho * e.size)
    )

    print(f"  全局: {len(global_selected)} vertices → "
          f"{global_coverage}/{len(global_hg.hyperedges)} coverage "
          f"({100 * global_coverage / max(len(global_hg.hyperedges), 1):.1f}%)")

    # Per-intent 汇总
    total_intent_vertices = set()
    total_intent_edges_covered = 0
    total_intent_edges = 0
    for intent, result in per_intent_results.items():
        total_intent_vertices.update(result["selected_vertices"])
        total_intent_edges_covered += result["coverage"]
        total_intent_edges += result["total_hyperedges"]

    print(f"  Per-intent 合并: {len(total_intent_vertices)} unique vertices → "
          f"{total_intent_edges_covered}/{total_intent_edges} coverage "
          f"({100 * total_intent_edges_covered / max(total_intent_edges, 1):.1f}%)")

    # 独有算子分析
    global_only = global_selected - total_intent_vertices
    intent_only = total_intent_vertices - global_selected
    shared = global_selected & total_intent_vertices

    print(f"  全局独有: {len(global_only)} | Intent独有: {len(intent_only)} | 共有: {len(shared)}")

    return {
        "global": {
            "num_vertices": len(global_selected),
            "selected_vertices": sorted(global_selected),
            "coverage": global_coverage,
            "total_hyperedges": len(global_hg.hyperedges),
            "coverage_pct": round(
                100 * global_coverage / max(len(global_hg.hyperedges), 1), 1
            ),
        },
        "per_intent_merged": {
            "num_unique_vertices": len(total_intent_vertices),
            "total_edges_covered": total_intent_edges_covered,
            "total_edges": total_intent_edges,
            "coverage_pct": round(
                100 * total_intent_edges_covered / max(total_intent_edges, 1), 1
            ),
        },
        "set_analysis": {
            "global_only": sorted(global_only),
            "intent_only": sorted(intent_only),
            "shared": sorted(shared),
            "num_global_only": len(global_only),
            "num_intent_only": len(intent_only),
            "num_shared": len(shared),
        },
    }


# ── Subflow 作为意图（fallback）───────────────────────────────────

def build_subflow_intent_map(conversations: list[dict]) -> Dict[str, List[str]]:
    """用 ABCD 自带的 subflow 标签作为意图，构建 intent map。"""
    intent_map: Dict[str, List[str]] = defaultdict(list)
    for conv in conversations:
        subflow = str(conv.get("scenario", {}).get("subflow", "unknown"))
        cid = str(conv.get("convo_id", "?"))
        intent_map[subflow].append(cid)
    return dict(intent_map)


# ── 主流程 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ABCD Per-Intent Session Hypergraph + Vertex Cover"
    )
    parser.add_argument("--intent-map", default=None,
                        help="intent_session_map.json 路径（意图分类结果）")
    parser.add_argument("--split", default="test",
                        choices=["train", "dev", "test"],
                        help="ABCD 数据分片")
    parser.add_argument("--use-subflow", action="store_true",
                        help="使用 ABCD 自带的 subflow 作为意图（不需要 --intent-map）")
    parser.add_argument("--rho", type=float, default=0.8,
                        help="Vertex cover 覆盖阈值")
    parser.add_argument("--max-vertices", type=int, default=30,
                        help="每个意图最大选中顶点数")
    parser.add_argument("--min-sessions", type=int, default=2,
                        help="最少 session 数，低于此值的意图跳过 per-intent 分析")
    parser.add_argument("--min-bridge-intents", type=int, default=2,
                        help="桥接算子至少出现在 N 个 intent 中")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="限制 ABCD 对话总数")
    parser.add_argument("--output-dir", default=None,
                        help="自定义输出目录")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载 ABCD ──────────────────────────────────────────
    print(f"加载 ABCD {args.split} split...")
    conversations = load_abcd_data(args.split)
    if args.max_sessions:
        conversations = conversations[:args.max_sessions]
    total_convos = len(conversations)
    print(f"  {total_convos} conversations")

    # ── 2. 获取 intent map ────────────────────────────────────
    if args.use_subflow:
        print("\n使用 subflow 作为意图标签...")
        intent_map = build_subflow_intent_map(conversations)
    elif args.intent_map:
        print(f"\n加载意图分类结果: {args.intent_map}")
        intent_map = json.loads(Path(args.intent_map).read_text(encoding="utf-8"))
        # 过滤 — 只保留当前 split 中存在的 convo_id
        valid_ids = {str(c.get("convo_id", "")) for c in conversations}
        filtered_map = {}
        for intent, cids in intent_map.items():
            valid_cids = [c for c in cids if c in valid_ids]
            if valid_cids:
                filtered_map[intent] = valid_cids
        intent_map = filtered_map
    else:
        print("错误: 需要 --intent-map 或 --use-subflow")
        sys.exit(1)

    print(f"  意图数: {len(intent_map)}")
    total_mapped_sessions = sum(len(v) for v in intent_map.values())
    print(f"  覆盖 sessions: {total_mapped_sessions}/{total_convos}")

    # ── 3. Per-intent 超图 ────────────────────────────────────
    print(f"\n构建 Per-Intent 超图（min_sessions={args.min_sessions}）...")
    per_intent_hg = build_per_intent_hypergraphs(
        conversations, intent_map, min_sessions=args.min_sessions,
    )
    print(f"  有效意图: {len(per_intent_hg)}")

    # ── 4. Per-intent Vertex Cover ────────────────────────────
    print(f"\nPer-Intent Vertex Cover (rho={args.rho}, max_vertices={args.max_vertices})...")
    per_intent_results = run_per_intent_vertex_cover(
        per_intent_hg, rho=args.rho, max_vertices=args.max_vertices,
    )

    # ── 5. 桥接算子 ──────────────────────────────────────────
    print(f"\n桥接算子分析 (min_intents={args.min_bridge_intents})...")
    bridges = find_bridge_operators(per_intent_results, min_intents=args.min_bridge_intents)
    print(f"  桥接算子: {len(bridges)}")
    for b in bridges[:15]:
        print(f"    {b['operator']:45s} 出现在 {b['num_intents']} 个 intent: {', '.join(b['intents'][:5])}")

    # ── 6. 全局 vs Per-Intent 对比 ────────────────────────────
    print(f"\n全局 vs Per-Intent 覆盖对比...")
    comparison = compare_global_vs_per_intent(
        conversations, per_intent_hg, per_intent_results,
        rho=args.rho, max_vertices=args.max_vertices,
    )

    # ── 7. 输出 ────────────────────────────────────────────────
    # Per-intent 结果
    per_intent_output = {
        "config": {
            "rho": args.rho,
            "max_vertices": args.max_vertices,
            "min_sessions": args.min_sessions,
            "split": args.split,
            "total_conversations": total_convos,
            "num_intents": len(intent_map),
            "num_valid_intents": len(per_intent_hg),
        },
        "per_intent": per_intent_results,
    }
    per_intent_path = out_dir / "per_intent_vertex_subsets.json"
    per_intent_path.write_text(
        json.dumps(per_intent_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nPer-intent 结果 -> {per_intent_path}")

    # 桥接算子
    bridges_path = out_dir / "bridge_operators.json"
    bridges_path.write_text(
        json.dumps({
            "min_intents": args.min_bridge_intents,
            "total_bridge_operators": len(bridges),
            "bridge_operators": bridges,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"桥接算子 -> {bridges_path}")

    # 对比
    comparison_path = out_dir / "intent_coverage_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"覆盖对比 -> {comparison_path}")

    # ── 8. 汇总 ────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print("Done.")
    print(f"{'=' * 50}")
    print(f"  per_intent_vertex_subsets.json   — 每个意图的 vertex set")
    print(f"  bridge_operators.json            — 跨意图桥接算子")
    print(f"  intent_coverage_comparison.json  — 全局 vs per-intent 对比")


if __name__ == "__main__":
    main()
