"""Render the frozen SGD graph experiment, preserving raw curve values."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_spine_tradeoff import STYLE, METHODS

LABELS = {"production_tracecover": "当前 SGD 原版 TraceCover",
          "paper_log_lift": "论文 log/lift 权重版", "support_mean_matched": "support 权重敏感性版"}


def branch_cost(catalog, indices):
    """Training source-mass-weighted outgoing branch-pair count, as in miner."""
    source_mass = Counter()
    outgoing = Counter()
    for edge in catalog["edges"]:
        source_mass[edge["source"]] += edge["support"]
    total = sum(source_mass.values())
    for i in indices:
        outgoing[catalog["edges"][i]["source"]] += 1
    return sum(source_mass[node] / total * degree * (degree - 1) / 2 for node, degree in outgoing.items())


def main(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    source = args.input_dir
    data = json.loads((source / "results.json").read_text(encoding="utf-8"))
    audit = json.loads((source / "evaluation_audit.json").read_text(encoding="utf-8"))
    catalogs = json.loads((source / "catalogs.json").read_text(encoding="utf-8"))
    plans = [json.loads(line) for line in (source / "selections.jsonl").read_text(encoding="utf-8").splitlines()]
    costs = defaultdict(list)
    for plan in plans:
        if plan["point"] == 0:
            costs[plan["variant"], plan["method"], plan["seed"]].append(branch_cost(catalogs[plan["flow"]], plan["indices"]))
    summary = {}
    for variant in LABELS:
        summary[variant] = {}
        for method in METHODS:
            values = [np.mean(c) for (v, m, _), c in costs.items() if v == variant and m == method]
            summary[variant][method] = float(np.mean(values))
    (source / "posthoc_branching.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.rcParams.update(STYLE)
    files = []
    # Shared axes allow comparison across weighting variants and splits.
    for variant in ("paper_log_lift", "support_mean_matched"):
        for split in ("heldout", "train"):
            fig, ax = plt.subplots(figsize=(3.55, 2.85))
            for method, (label, color, marker, line) in METHODS.items():
                rows = sorted([r for r in data["curves"] if r["variant"] == variant and r["method"] == method], key=lambda r: r["point"])
                x = np.array([r["edge_ratio"] * 100 for r in rows])
                y = np.array([r[split + "_coverage"] * 100 for r in rows])
                ax.plot(x, y, color=color, marker=marker, linestyle=line, label=label, markersize=3.5, linewidth=1.5)
                if method == "random":
                    std = np.array([r[split + "_coverage_seed_std"] * 100 for r in rows])
                    ax.fill_between(x, y - std, y + std, color=color, alpha=.16, linewidth=0)
            ax.set(xlim=(45, 101), ylim=(45, 101), xlabel="Observed transition types retained (%)", ylabel="Mean session coverage (%)")
            ax.grid(axis="y", color="#dddddd", linewidth=.5)
            ax.legend(loc="lower right", frameon=False)
            for ext in ("pdf", "png"):
                path = out / (variant + "_" + split + "." + ext)
                fig.savefig(path)
                files.append(path)
            plt.close(fig)
    names = {"tracecover": "TraceCover", "frequency": "频次 + 可达性修复", "weight": "边权 + 可达性修复", "random": "随机优先级 + 可达性修复"}
    report = ["# SGD spine 压缩—覆盖实验", "", "2026-10-25；CPU 图级实验，LLM 调用为 0。", "",
        "使用现有 1/5 SGD seen-schema 划分的九个场景：" + ", ".join(data["protocol"]["families"]) + "。",
        "TRAIN 共 16,591 个 policy target，validation 共 2,480 个；call 与 non-call 均保留，同一轮 action heads 合并为节点，slot/API method 不进入节点身份。",
        "主指标是未饱和的逐场景会话投影平均覆盖率；留出集未见转移仍计入分母。"]
    for split, counts in audit["session_counts"].items():
        report.append(f"- {split}: {counts['projections']} 条有转移的会话投影，来自 {counts['unique_original_dialogues']} 条不同原始对话。")
    report += ["", "训练阶段先冻结所有选择，再读取 validation 对话；所有选择均独立逐条重算覆盖率、边数与入口可达性。", ""]
    for variant, label in LABELS.items():
        report += ["## " + label, "", "| 方法 | 边数 / 369 | TRAIN 覆盖 | validation 覆盖 | validation seed 标准差 |", "|---|---:|---:|---:|---:|"]
        for method in METHODS:
            row = next(r for r in data["curves"] if r["variant"] == variant and r["point"] == 0 and r["method"] == method)
            report.append(f"| {names[method]} | {row['edges']} | {100*row['train_coverage']:.2f}% | {100*row['heldout_coverage']:.2f}% | {100*row['heldout_coverage_seed_std']:.2f} pp |")
        report += ["", "| TraceCover 减基线（validation） | 差值 pp | 95% 原始对话 cluster bootstrap 区间 pp |", "|---|---:|---:|"]
        for row in data["endpoint_bootstrap"]:
            if row["variant"] == variant and row["split"] == "heldout":
                report.append(f"| {names[row['baseline']]} | {row['native_minus_baseline_pp']:+.2f} | [{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] |")
        freq = summary[variant]["frequency"]
        native = summary[variant]["tracecover"]
        report += ["", f"探索性结构诊断：TRAIN source-mass 加权分支对数（九场景算术平均）TraceCover={native:.3f}，频次={freq:.3f}，降低 {100*(1-native/freq):.1f}%。这与优化目标相关，不是独立的语义正确性证据。", ""]
    report += ["## 解释与边界", "",
        "三个版本在等规模终点均未超过频次基线。论文权重版 291/369 条边保留 96.22% validation 会话证据，但频次基线为 96.71%；因此不能声称等规模 coverage 优势。",
        "当前原版 174/369 条边保留 88.08% 原始覆盖；其饱和 F 为 95.52%，二者口径不同。原版为本地默认 miner 重算，不是服务器最终 skill 产物。",
        "support 版本相对频次的差距较小，bootstrap 区间包含 0；它是明确标记的敏感性实验，不可替代论文公式版本。",
        "可支持的表述：较小转移子图保留了大部分会话证据；同规模覆盖并未优于频次选择。结构收益需结合分支复杂度等单独论证。",
        "SGD 没有与 ABCD guideline 同口径的官方流程转移 gold，本实验不把 schema/API 清单当流程真值。",
        "同一原始对话可能出现在多个 family。主指标保持 family projection 平均，bootstrap 按原始 dialogue_id 聚类，保留全部投影；区间条件于固定图与固定随机 seed 集合。",
        "基线是排序后做可达性修复的启发式，不是全局约束最优解；随机是随机优先级加修复，不是可行图均匀抽样。",
        "曲线使用原生训练搜索同边数最后快照，未进行固定预算独立再优化；可非单调。预算按 family 固定，不比较全局最优预算分配。",
        "validation 已参与此前开发，本分析不是全新盲测；不涉及额外 agent 训练/评测，不修改论文或生产算法。", ""]
    (source / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    caption = ("SGD 20% seen-schema compression--coverage trade-off across nine independently mined families. "
        "All methods share the same per-family TRAIN-derived edge budget, action inventory and observed entries; "
        "repair edges count toward the budget. Coverage is the unsaturated mean over family dialogue projections "
        "with at least one transition; unseen validation pairs remain in the denominator. Native points are TRAIN "
        "search checkpoints, not independently optimized fixed-budget solutions. Random bands show seed standard "
        "deviation over 20 priority permutations with repair, not confidence intervals. Validation is existing "
        "development data. Production TraceCover is reported separately as an endpoint in REPORT.md.")
    (out / "README.md").write_text(caption + "\n\nData: " + str(source.resolve()) + "\n\nReproduce: python scripts/plot_sgd_spine_tradeoff.py --input-dir " + str(source) + " --output-dir NEW_DIRECTORY\n", encoding="utf-8")
    (out / "plot_source.py").write_bytes(Path(__file__).read_bytes())
    manifest = {"results_sha256": hashlib.sha256((source / "results.json").read_bytes()).hexdigest(),
        "figures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved figures and report: " + str(source / "REPORT.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
