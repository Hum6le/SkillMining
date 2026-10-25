"""Reproduce a reported-cost Pareto preview from the ABCD main LaTeX table.

No model calls. Missing costs are excluded, never imputed. The current main
method is one point, not a curve assembled from its historical revisions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def number(cell):
    clean = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell).strip()
    if clean == "-":
        return None
    if not re.fullmatch(r"\d+(?:\.\d+)?", clean):
        raise ValueError(f"Unrecognized main-table numeric cell: {cell!r}")
    return float(clean)


def extract_rows(source):
    label = source.index(r"\label{tab:main-results}")
    start = source.rfind(r"\begin{table*}", 0, label)
    block = source[start:source.index(r"\end{table*}", label)]
    if r"Calls\\($10^3$)" not in block or r"Tokens\\($10^7$)" not in block:
        raise ValueError("Main-table cost units changed; review the extractor rather than guessing")
    pattern = re.compile(
        r"^\s*(ReAct|Reflexion|AWM \(online\)|ExpeL|A-MEM|Trace2Skill|SKILL-DISCO|SkillMiner [^&\n]+?)"
        r"\s*&\s*(.*?)\\\\[ \t]*(?:\n|$)", re.M | re.S)
    rows = []
    for match in pattern.finditer(block):
        method = match[1].strip()
        fields = [number(cell) for cell in match[2].split("&")]
        if len(fields) != 10:
            raise ValueError(f"Expected ABCD+SGD 10 numeric fields for {method}")
        latest = method == "SkillMiner v2.3 (online V2)"
        historical = method.startswith("SkillMiner") and not latest
        missing = [key for key, value in zip(("SR", "calls", "tokens"), (fields[0], fields[3], fields[4])) if value is None]
        reason = "Historical method revision, not an independent baseline" if historical else (
            "Missing reported " + ", ".join(missing) if missing else "")
        rows.append({"method": method, "label": "SkillMiner" if latest else method,
                     "sr_percent": fields[0], "action_percent": fields[1], "slot_percent": fields[2],
                     "mining_calls": round(fields[3] * 1000) if fields[3] is not None else None,
                     "mining_tokens": round(fields[4] * 10**7) if fields[4] is not None else None,
                     "included": not reason, "exclusion_reason": reason})
    if len(rows) != 10 or len({r["method"] for r in rows}) != 10:
        raise ValueError("Main-table method set changed; review extraction coverage")
    return rows


def frontier(rows, key):
    return sorted([p for p in rows if not any(
        q[key] <= p[key] and q["sr_percent"] >= p["sr_percent"]
        and (q[key] < p[key] or q["sr_percent"] > p["sr_percent"])
        for q in rows)], key=lambda p: p[key])


def plot(rows, output, performance_x=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix"})
    styles = {"Reflexion": ("#666666", "D"), "AWM (online)": ("#E69F00", "o"),
              "Trace2Skill": ("#CC79A7", "^"), "SKILL-DISCO": ("#0072B2", "s"),
              "SkillMiner": ("#D55E00", "*")}
    offsets = {
        "mining_tokens": {"Reflexion": (6, -13), "AWM (online)": (-6, 7), "Trace2Skill": (-7, -13),
                          "SKILL-DISCO": (7, 8), "SkillMiner": (-7, 8)},
        "mining_calls": {"Reflexion": (-7, -13), "AWM (online)": (7, 7), "Trace2Skill": (-7, 7),
                         "SKILL-DISCO": (7, 8), "SkillMiner": (7, 7)}}
    if performance_x:
        offsets = {
            "mining_tokens": {"Reflexion": (7, 7), "AWM (online)": (7, -11), "Trace2Skill": (-7, 7),
                              "SKILL-DISCO": (7, 7), "SkillMiner": (-7, 8)},
            "mining_calls": {"Reflexion": (7, -11), "AWM (online)": (7, -11), "Trace2Skill": (-7, 7),
                             "SKILL-DISCO": (7, 7), "SkillMiner": (-7, -12)}}
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0), constrained_layout=True)
    for ax, key, scale, limit, xlabel in (
        (axes[0], "mining_tokens", 10**8, 9.5, r"Reported mining tokens ($\times 10^8$)"),
        (axes[1], "mining_calls", 1000, 260, r"Reported mining calls ($\times 10^3$)")):
        best = frontier(rows, key)
        best_cost, best_performance = [r[key] / scale for r in best], [r["sr_percent"] for r in best]
        ax.step(best_performance if performance_x else best_cost,
                best_cost if performance_x else best_performance,
                where="pre" if performance_x else "post", color="#999999", linestyle="--", linewidth=1.1,
                label="Observed frontier", zorder=1)
        for row in rows:
            label = row["label"]
            color, marker = styles[label]
            dx, dy = offsets[key][label]
            x, y = ((row["sr_percent"], row[key] / scale) if performance_x else
                    (row[key] / scale, row["sr_percent"]))
            ax.scatter(x, y, color=color, marker=marker,
                       s=115 if label == "SkillMiner" else 38, edgecolors="black", linewidths=.45, zorder=3)
            ax.annotate(label, (x, y), xytext=(dx, dy),
                        textcoords="offset points", ha="left" if dx > 0 else "right",
                        va="bottom" if dy > 0 else "top", fontsize=8.2,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": .15},
                        fontweight="bold" if label == "SkillMiner" else "normal")
        if performance_x:
            ax.set(xlim=(0, 58), ylim=(0, limit), xlabel="Joint AST (%)", ylabel=xlabel)
            ax.set_xticks(range(0, 60, 10))
            ax.legend(loc="upper left", frameon=False)
        else:
            ax.set(xlim=(0, limit), ylim=(0, 58), xlabel=xlabel, ylabel="Joint AST (%)")
            ax.set_yticks(range(0, 61, 10))
            ax.set_ylim(0, 58)
            ax.legend(loc="lower right", frameon=False)
    stem = "main_results_pareto_performance_x" if performance_x else "main_results_pareto"
    for fmt in ("pdf", "png"):
        fig.savefig(output / f"{stem}.{fmt}", dpi=300, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2] / "paper/overleaf_sync/main.tex")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--x-axis", choices=("cost", "performance"), default="cost")
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    rows = extract_rows(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "main_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Figures read an explicit exported data artifact, not numeric literals.
    with csv_path.open(encoding="utf-8", newline="") as f:
        included = [{**r, "sr_percent": float(r["sr_percent"]),
                     "mining_calls": int(r["mining_calls"]), "mining_tokens": int(r["mining_tokens"])}
                    for r in csv.DictReader(f) if r["included"] == "True"]
    summary = {"source": str(args.source.resolve()), "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "dataset": "ABCD main-results aggregate, not purchase_dispute A3", "included": included,
               "excluded": [r for r in rows if not r["included"]],
               "frontiers": {key: [r["label"] for r in frontier(included, key)] for key in ("mining_tokens", "mining_calls")},
               "limitations": ["Reported rounded main-table aggregates; raw cost logs/runtime comparability not audited.",
                               "Mining costs cover knowledge construction/refinement, not held-out execution or deployment total.",
                               "Only one reported operating point per method; not a training-budget sweep.",
                               "No uncertainty estimates, missing values not imputed, historical revisions not independent methods.",
                               "Dashed steps show best observed point under a cost budget, not interpolated training behavior."]}
    performance_x = args.x_axis == "performance"
    summary["x_axis"] = args.x_axis
    summary["ideal_direction"] = "lower-right" if performance_x else "upper-left"
    if performance_x:
        summary["limitations"][-1] = (
            "Dashed pre-steps show the lowest reported cost meeting a performance threshold among compared configurations; not a training trajectory or a global cost lower bound.")
    stem = "main_results_pareto_performance_x" if performance_x else "main_results_pareto"
    summary_name = "pareto_summary_performance_x.json" if performance_x else "pareto_summary.json"
    (args.output_dir / summary_name).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot(included, args.output_dir, performance_x=performance_x)
    print(json.dumps({"included_methods": [r["label"] for r in included], "frontiers": summary["frontiers"],
                      "png": str(args.output_dir / f"{stem}.png"),
                      "pdf": str(args.output_dir / f"{stem}.pdf")}, indent=2))


if __name__ == "__main__":
    main()
