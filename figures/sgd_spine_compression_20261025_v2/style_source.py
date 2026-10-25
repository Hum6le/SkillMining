"""Reproducible publication plots and endpoint report from recorded CSV/JSON."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix", "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": .04}
METHODS = {"tracecover": ("TraceCover checkpoints", "#0072B2", "o", "-"),
           "frequency": ("Frequency + repair", "#D55E00", "s", "--"),
           "weight": (r"Edge weight $w_e$ + repair", "#009E73", "^", "-."),
           "random": ("Random priority + repair", "#777777", "x", ":")}
VARIANTS = {"paper_log_lift": "Paper log/lift weighting", "support_mean_matched": "Mean-matched support weighting"}


def main(args):
    data = json.loads((args.input_dir / "results.json").read_text(encoding="utf-8"))
    curves = data["curves"]
    seed_count = len(data["protocol"]["random_priority_seeds"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    includes, files = [], []
    for variant, long_name in VARIANTS.items():
        for split in ("heldout", "train"):
            fig, ax = plt.subplots(figsize=(3.55, 2.85))
            group = [r for r in curves if r["variant"] == variant]
            for method, (label, color, marker, line) in METHODS.items():
                rows = sorted([r for r in group if r["method"] == method], key=lambda r: r["point"])
                x = np.array([100 * r["edge_ratio"] for r in rows])
                y = np.array([100 * r[split + "_coverage"] for r in rows])
                ax.plot(x, y, label=label, color=color, marker=marker, linestyle=line,
                        linewidth=1.5, markersize=3.5)
                if method == "random":
                    std = np.array([100 * r[split + "_coverage_seed_std"] for r in rows])
                    ax.fill_between(x, y - std, y + std, color=color, alpha=.16, linewidth=0)
            # Raw values are shown as recorded: no monotone envelope or smoothing.
            xmin = min(100 * r["edge_ratio"] for r in group)
            ymin = min(100 * (r[split + "_coverage"] - r[split + "_coverage_seed_std"]) for r in group)
            ax.set_xlim(10 * np.floor(xmin / 10), 101)
            ax.set_ylim(max(0, 10 * np.floor(ymin / 10)), 101)
            ax.set_xlabel("Observed transition types retained (%)")
            ax.set_ylabel("Mean session coverage (%)")
            ax.grid(axis="y", color="#dddddd", linewidth=.5, alpha=.7)
            ax.legend(loc="lower right", frameon=False)
            name = variant + "_" + split
            for extension in ("pdf", "png"):
                path = args.output_dir / (name + "." + extension)
                fig.savefig(path)
                files.append(str(path))
            plt.close(fig)
            caption = (f"Spine compression--coverage trade-off with {long_name.lower()} on ABCD "
                + ("held-out development dialogues" if split == "heldout" else "training dialogues")
                + ". All methods use identical TRAIN-derived edge budgets within each of ten flows, "
                "fixed action inventories and observed entries, and entry reachability. "
                "The x-axis is the sum of retained edge counts divided by the sum of observed counts. "
                "Coverage averages unsaturated retained transition fractions across dialogues with at least one transition; "
                "unseen held-out transitions remain in the denominator. TraceCover points are recorded native training-search "
                "checkpoints, not independently optimized fixed-budget solutions. Baseline repair edges count toward the budget. "
                f"Random shading denotes one standard deviation across {seed_count} random-priority seeds, not a confidence interval. "
                "The held-out split is historical development replay, not a new blind test.")
            includes.append("\\begin{figure}[t]\n  \\centering\n  \\includegraphics[width=\\linewidth]{figures/spine_compression_20261025/"
                + name + ".pdf}\n  \\caption{" + caption + "}\n  \\label{fig:spine-" + name.replace("_", "-") + "}\n\\end{figure}\n")
    (args.output_dir / "latex_includes.tex").write_text("\n".join(includes), encoding="utf-8")
    # Report exact endpoints and paired differences; no claims selected by outcome.
    names = {k: v[0].replace("$", "") for k, v in METHODS.items()}
    report = ["# Spine 压缩—覆盖：等规模对照", "", "2026-10-25；CPU-only，LLM 调用 0。",
        "", "主纵轴为未饱和的逐会话转移覆盖率，所有方法同一 flow/点保留完全相同的边数，并满足相同入口可达要求。",
        "正文公式对应 paper_log_lift；438 边 support_mean_matched 为明确标注的敏感性版本。", ""]
    for variant, long_name in VARIANTS.items():
        report += ["## " + long_name, "", "以下为原生最终图的等规模终点；曲线覆盖该终点至完整观测图。", "",
            "| 方法 | 边数 | TRAIN 原始覆盖 | 留出原始覆盖 | 随机种子标准差（留出，pp） | Guideline 正关系微观召回（118 探针） |",
            "|---|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            row = next(r for r in curves if r["variant"] == variant and r["point"] == 0 and r["method"] == method)
            report.append(f"| {names[method]} | {row['edges']} | {100*row['train_coverage']:.2f}% | {100*row['heldout_coverage']:.2f}% | "
                          f"{100*row['heldout_coverage_seed_std']:.2f} | {100*row['guideline_micro_recall']:.2f}% |")
        report += ["", "终点配对差值（TraceCover 减基线；固定图、对话 bootstrap）：", "",
                   "| 基线 | 留出差值 pp | 95% bootstrap 区间 pp |", "|---|---:|---:|"]
        for row in data["endpoint_bootstrap"]:
            if row["variant"] == variant and row["split"] == "heldout":
                report.append(f"| {names[row['baseline']]} | {row['native_minus_baseline_pp']:+.2f} | "
                              f"[{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] |")
        report.append("")
    report += ["## 协议边界", "",
        "- 所有图在本次读取留出对话和 guideline 探针文件前已冻结，选择计划 SHA 在评测结束时再次校验。阶段一的历史 per_flow 缓存含评估字段，但只使用其 TRAIN hash 校验来源，没有据历史分数选点。",
        "- Baseline 是 top-k 优先级 + 可达性补边 + 逆优先级安全删边回到 exact k；不是全局最优约束求解器。",
        f"- Random 是均匀随机边优先级加修复，不是对可行子图均匀抽样；本轮 {seed_count} 个固定 seeds，详情见 protocol.json。",
        "- 原生曲线使用最后一个同规模 TRAIN 搜索快照，不按留出成绩取包络；非单调点原样保留。",
        "- 预算按原生 TRAIN 轨迹逐 flow 分配；只比较该固定分配，不声称全局预算分配最优。",
        "- 只评价动作相邻关系保留，不代表完整流程可执行、条件解释正确或 agent 成功率。",
        "- saturated F 的 95% 约束不能当成 95% 原始平均覆盖。",
        "- 历史开发 replay 已参与其它开发；本次属于开发分析。bootstrap 只反映固定图的对话抽样变化，不反映训练/算法随机性；随机基线区间条件于固定 seed 集合，不包含 seed Monte Carlo 误差。",
        "- 不据该结果自动修改算法、主实验或论文结论。", ""]
    (args.input_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (args.output_dir / "README.md").write_text("# Spine compression figures\n\nData: " + str(args.input_dir.resolve())
        + "\n\nReproduce: python scripts/plot_spine_tradeoff.py --input-dir " + str(args.input_dir)
        + " --output-dir " + str(args.output_dir) + "\n\nCaptions: latex_includes.tex. Main paper-aligned weighting: paper_log_lift. "
        "support_mean_matched is a sensitivity. Random band is seed standard deviation.\n", encoding="utf-8")
    (args.output_dir / "plot_source.py").write_bytes(Path(__file__).read_bytes())
    manifest = {"input_results_sha256": hashlib.sha256((args.input_dir / "results.json").read_bytes()).hexdigest(),
                "files": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved curves, PDF/PNG figures and " + str(args.input_dir / "REPORT.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
