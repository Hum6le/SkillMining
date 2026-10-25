"""ABCD nominal standard-route retention from frozen TRAIN-only graph choices.

No selection, tuning, LLM calls, or skill execution. Unordered tasks are kept
separate. Officially documented conditional shortcuts are auxiliary variants.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import permutations
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skill_mining.sop_reference_benchmark import build_reference, UNORDERED, SHORTCUTS, digest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_score(sequence, selected, observed):
    required = list(zip(sequence, sequence[1:]))
    if not required:
        raise ValueError("Route must contain a transition")
    missing = [edge for edge in required if edge not in selected]
    return {"required_transitions": len(required), "missing_transitions": len(missing),
        "coverage": 1 - len(missing) / len(required), "complete": not missing,
        "train_observable": all(edge in observed for edge in required),
        "missing_train_unseen": sum(edge not in observed for edge in missing),
        "missing_train_observed": sum(edge in observed for edge in missing),
        "missing_edges": [list(e) for e in missing]}


def shortcut_sequence(sequence, source, target):
    i, j = sequence.index(source), sequence.index(target)
    if j <= i + 1:
        raise ValueError("Documented shortcut must skip at least one nominal action")
    return sequence[:i + 1] + sequence[j:]


def unordered_score(actions, selected):
    """Order flexibility diagnostic, not a demand to execute every option."""
    if len(actions) != len(set(actions)):
        raise ValueError("Unordered option set must have distinct actions")
    accepted = []
    total = 0
    for seq in permutations(actions):
        total += 1
        if all(edge in selected for edge in zip(seq, seq[1:])):
            accepted.append(list(seq))
    return {"option_actions": len(actions), "permutations": total, "accepted_permutations": len(accepted),
            "accepted_fraction": len(accepted) / total, "at_least_one_order": bool(accepted),
            "all_orders": len(accepted) == total, "example_accepted_orders": accepted[:3]}


def run(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    plan_path = args.graph_bench_dir / "frozen_rankings.json"
    freeze = read(args.graph_bench_dir / "selection_freeze.json")
    assert sha(plan_path) == freeze["sha256"], "Frozen graph choices changed"
    plans = read(plan_path)
    # These exact graph choices predate this route evaluation and are not changed.
    resource_dir = ROOT / "data/eval/abcd/data"
    reference = build_reference(*(read(resource_dir / name) for name in ("kb.json", "ontology.json", "guidelines.json")))
    dump(out / "reference.json", reference)
    protocol = {"dataset": "ABCD 55 fine tasks in ten coarse flows",
        "primary": "task-macro nominal route retention: all consecutive nominal KB relations must be present",
        "unordered": "four explicitly unordered tasks excluded from ordered denominator; separately enumerate option-action permutations as order-flexibility diagnostic, not required executions",
        "shortcuts": "four explicitly sourced conditional skips scored separately; nominal route stays the primary route",
        "observable_ceiling": "TRAIN full DFG is ceiling only for observed-edge selection methods; PM generalizations may exceed it",
        "train_observable_subset": "auxiliary: only routes whose every required edge is in full TRAIN DFG; keep full 51-task result primary",
        "graph_input_sha256": sha(plan_path), "graph_input": str(plan_path),
        "graph_provenance_sha256": sha(args.graph_bench_dir / "provenance.json"),
        "reference_sha256": digest(reference), "llm_calls": 0,
        "scope": "canonical adjacency structural retention, no branch-guard, slot, Petri token execution or agent-success claim",
        "task_weight": "each named task equal weight; shared route shapes counted separately, unique sequence count also reported",
        "development_note": "existing guideline reference already used in development; no new blind test or independent task-sample significance claim",
        "source_sha256": sha(Path(__file__))}
    dump(out / "protocol.json", protocol)
    (out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    methods = list(next(iter(plans.values()))["ranked"])
    ordered, unordered, variants, graph_sizes = [], [], [], {m: 0 for m in methods}
    distinct_routes = set()
    for flow, plan in plans.items():
        observed = {(u.split(":", 1)[1], v.split(":", 1)[1]) for u, v in plan["full_train_pairs"]}
        for method in methods:
            selected = {(u.split(":", 1)[1], v.split(":", 1)[1])
                        for u, vs in plan["ranked"][method].items() for v in vs}
            graph_sizes[method] += len(selected)
            for fine, sequence in reference["flows"][flow]["routes"].items():
                common = {"flow": flow, "fine": fine, "method": method}
                if flow == "troubleshoot_site" and fine in UNORDERED:
                    unordered.append({**common, **unordered_score(sequence, selected)})
                    continue
                distinct_routes.add(tuple(sequence))
                ordered.append({**common, "sequence": sequence, **route_score(sequence, selected, observed)})
                for coarse, title, _, source, target, quote in SHORTCUTS:
                    normalized = title.lower().replace(" ", "_")
                    if coarse == flow and fine == normalized:
                        alternative = shortcut_sequence(sequence, source, target)
                        variants.append({**common, "sequence": alternative, "condition_quote": quote,
                                         **route_score(alternative, selected, observed)})
    summary = []
    for method in methods:
        tasks = [r for r in ordered if r["method"] == method]
        available = [r for r in tasks if r["train_observable"]]
        options = [r for r in unordered if r["method"] == method]
        shortcuts = [r for r in variants if r["method"] == method]
        assert len(tasks) == 51 and len(options) == 4 and len(shortcuts) == 4
        # Cross-check completion independently via the missing counts.
        assert all(r["complete"] == (r["missing_transitions"] == 0) for r in tasks + shortcuts)
        summary.append({"method": method, "edges": graph_sizes[method], "ordered_tasks": len(tasks),
            "complete_tasks": sum(r["complete"] for r in tasks), "complete_rate": float(np.mean([r["complete"] for r in tasks])),
            "mean_missing_transitions": float(np.mean([r["missing_transitions"] for r in tasks])),
            "mean_route_coverage": float(np.mean([r["coverage"] for r in tasks])),
            "train_observable_tasks": len(available), "complete_observable_tasks": sum(r["complete"] for r in available),
            "observable_complete_rate": float(np.mean([r["complete"] for r in available])),
            "missing_train_unseen_total": sum(r["missing_train_unseen"] for r in tasks),
            "missing_train_observed_total": sum(r["missing_train_observed"] for r in tasks),
            "shortcut_variants_complete": sum(r["complete"] for r in shortcuts),
            "unordered_mean_order_fraction": float(np.mean([r["accepted_fraction"] for r in options]))})
    differences = []
    for native, baseline in (("tracecover_original", "frequency_k385"),
                             ("tracecover_support_matched", "frequency_k438"), ("tracecover_log", "frequency_k752")):
        base = {(r["flow"], r["fine"]): r for r in ordered if r["method"] == baseline}
        for row in (r for r in ordered if r["method"] == native):
            other = base[row["flow"], row["fine"]]
            if row["complete"] != other["complete"]:
                differences.append({"native": native, "baseline": baseline, "flow": row["flow"], "fine": row["fine"],
                    "native_complete": row["complete"], "baseline_complete": other["complete"],
                    "native_missing": row["missing_edges"], "baseline_missing": other["missing_edges"]})
    dump(out / "results.json", {"protocol": protocol, "summary": summary, "ordered_routes": ordered,
        "unordered_tasks": unordered, "conditional_shortcut_variants": variants, "matched_completion_differences": differences})
    for filename, rows in (("summary.csv", summary), ("ordered_routes.csv", ordered),
                            ("unordered_tasks.csv", unordered), ("shortcut_variants.csv", variants)):
        with (out / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    # Validate all rows again using an independent adjacency representation.
    for row in ordered + variants:
        ranked = plans[row["flow"]]["ranked"][row["method"]]
        seq = [row["flow"] + ":" + a for a in row["sequence"]]
        misses = [i for i in range(len(seq) - 1) if seq[i+1] not in ranked.get(seq[i], [])]
        assert len(misses) == row["missing_transitions"]
    assert sha(plan_path) == freeze["sha256"]
    ceiling = next(r for r in summary if r["method"] == "full_dfg")
    dump(out / "audit.json", {"frozen_graph_sha256_unchanged": True, "ordered_tasks": 51,
        "distinct_ordered_sequences": len(distinct_routes), "unordered_tasks": 4, "shortcut_variants": 4,
        "train_observable_routes": ceiling["complete_tasks"], "independent_route_recounts": len(ordered + variants), "llm_calls": 0})
    report = ["# ABCD 标准路线完整保留", "", "CPU 图级计算，LLM 调用为 0。", "",
        "主表分母为 51 个有名义顺序的任务，每个任务等权；4 个任意顺序任务单列。这里检验的是名义相邻关系全部保留，不代表条件、slot 或 agent 执行正确。", "",
        "| 方法 | 关系数 | 完整名义路线 / 51 | 完整率 | 每任务平均缺失转移 | TRAIN 可观测路线完整率 | 条件跳步路线 / 4 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        report.append(f"| {row['method']} | {row['edges']} | {row['complete_tasks']}/51 | {100*row['complete_rate']:.2f}% | {row['mean_missing_transitions']:.3f} | {row['complete_observable_tasks']}/{row['train_observable_tasks']} | {row['shortcut_variants_complete']}/4 |")
    report += ["", f"完整 TRAIN DFG 能保留 {ceiling['complete_tasks']}/51 条名义路线；这是只选 observed 边方法的上限，PM 泛化关系可以超过该上限。", "",
        f"51 个任务实际有 {len(distinct_routes)} 种不同 action 序列，任务之间共享大量关系；不把它们视为独立随机样本做显著性宣称。", "",
        "## 任意顺序任务", "", "对每个任务的选项动作枚举无重复排列，计算可容纳多少种排列。它仅反映顺序灵活性，不要求真实对话执行所有选项，也不代表 Petri 整条可执行性。", "",
        "| 方法 | 四任务平均可容纳排列比例 |", "|---|---:|"]
    for row in summary:
        report.append(f"| {row['method']} | {100*row['unordered_mean_order_fraction']:.2f}% |")
    report += ["", "## 等边数完整路线差异", "", "以下列出全部有差异的任务，不只挑对 TraceCover 有利的案例。", ""]
    for row in differences:
        report.append(f"- {row['native']} vs {row['baseline']}: {row['flow']}/{row['fine']}; native={row['native_complete']}, frequency={row['baseline_complete']}; native missing={row['native_missing']}; frequency missing={row['baseline_missing']}")
    report += ["", "## 协议限定", "", *["- " + key + ": " + str(value) for key, value in protocol.items()], "",
        "各方法使用原生图大小；另列三个 exact-size 频次对照。原版支持度候选门槛不同，不能将其对照描述为仅改变选择目标。",
        "Inductive/Heuristics 关系来自已完成的官方 PM4Py 投影，数量不是原生模型弧数。所有泛化关系均保留。",
        "四条跳步路线只在原文明确条件下成立，本计算只检查关系存在，不判断条件正确性；名义路线与条件路线不混合分母。", ""]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"summary": summary, "audit": read(out / "audit.json")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-bench-dir", type=Path, default=ROOT / "outputs/abcd_graph_ranking_20261025_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with patch("llm.chat", side_effect=AssertionError("LLM forbidden in route evaluation")):
        run(args)
