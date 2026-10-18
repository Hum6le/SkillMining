"""Verify completed matched predictions and write a Chinese report plus CSV."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_tod.sgd_adapter import action_counter, score_policies


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(root):
    comparison = read(root / "final_comparison.json")
    manifest = read(root / "manifest.json")
    expected = None
    shared = None
    rows = []
    for method in ("awm", "trace2skill", "ours"):
        records = [json.loads(line) for line in (root / method / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        identities = [(row["dialogue_id"], row["turn_index"], json.dumps(row["gold"], sort_keys=True)) for row in records]
        observations = [row["diagnostic"]["shared_observation_sha256"] for row in records]
        if expected is None:
            expected, shared = identities, observations
        if identities != expected or observations != shared or len(records) != manifest["validation_turns"]:
            raise ValueError("Methods do not share identical cases, gold or DB observations")
        if len(set((row["dialogue_id"], row["turn_index"]) for row in records)) != len(records):
            raise ValueError("Duplicate validation cases")
        for row in records:
            if row["prediction"].get("_execution_error"):
                continue
            if set(action_counter(row["prediction"])) != {tuple(head) for head in row["diagnostic"]["selected_action_heads"]}:
                raise ValueError("Action heads changed during payload grounding")
        scores = score_policies([row["gold"] for row in records], [row["prediction"] for row in records])
        summary = comparison["methods"][method]
        if scores["counts"] != summary["counts"]:
            raise ValueError("Recomputed metrics differ from saved summary")
        counts = scores["counts"]
        rows.append({"method": method, "turns": counts["turns"],
            "action_accuracy": scores["action_set_exact"],
            "precision": counts["action_tp"] / counts["action_pred"] if counts["action_pred"] else 0,
            "recall": counts["action_tp"] / counts["action_gold"] if counts["action_gold"] else 0,
            "f1": scores["action_micro_f1"], "slot_exact_given_action": scores["slot_exact_given_action"],
            "joint": scores["turn_joint_ast"], "semantic_joint_audit": summary["semantic_value_audit"]["turn_joint_ast"],
            "action_exact_turns": counts["action_exact"], "joint_exact_turns": counts["joint_exact"],
            "execution_failed_turns": counts["execution_failed_turns"]})
    with (root / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# MultiWOZ hotel 1/10 修正评测结果", "",
        "相同的 249 个训练对话、1,818 个 agent turns；验证集为 32 个对话、228 个 turns。模型为 Qwen3.6-Flash，seed=42。", "",
        "三种方法共用相同的只读数据库查询观察、action 选择、固定 action 后填 slot 的执行器。已重新核对各方法的样本 ID、gold、数据库观察和评分，确认一致。", "",
        "| 指标 | AWM | Trace2Skill | Ours |", "|---|---:|---:|---:|"]
    for name,key in [("Action accuracy","action_accuracy"),("Micro precision","precision"),
                     ("Micro recall","recall"),("Micro F1","f1"),
                     ("Slot exact（动作正确时）","slot_exact_given_action"),("Action + slot 联合准确率","joint"),
                     ("值大小写/空白归一后的联合准确率（补充诊断）","semantic_joint_audit")]:
        lines.append("| " + name + " | " + " | ".join(f"{row[key]*100:.2f}%" for row in rows) + " |")
    lines += ["", "| 原始计数 | AWM | Trace2Skill | Ours |", "|---|---:|---:|---:|"]
    for name,key in [("动作完全正确 turns","action_exact_turns"),("联合完全正确 turns","joint_exact_turns"),
                     ("最终执行失败 turns","execution_failed_turns")]:
        lines.append("| " + name + " | " + " | ".join(str(row[key]) for row in rows) + " |")
    ours, awm, t2s = rows[2], rows[0], rows[1]
    lines += ["", f"Ours 相比 AWM：Action accuracy 差值 {(ours['action_accuracy']-awm['action_accuracy'])*100:+.2f} 个百分点，F1 差值 {(ours['f1']-awm['f1'])*100:+.2f} 个百分点，严格联合准确率差值 {(ours['joint']-awm['joint'])*100:+.2f} 个百分点。",
        "", "本轮复用原有训练产物，并用训练集重新补齐 Ours 的 116 张 action card；没有按新执行器重新训练三种方法。因此这是修正后的冻结资源评测，与旧版结果应分开报告，也不能单独归因于某一个修复。",
        "", "数据库只提供公开实体信息，不执行预订或生成 reference。模型看不到 gold belief state、gold DB pointer 或当前/未来 system 文本。值归一结果是补充诊断，原始评分保留。",
        "", "本轮新调用消耗见 llm_usage.json；先前训练成本不包含在本轮统计中。过程中被校验拦截、重试的记录保存在 retry_history.jsonl，不应把最终成功解读为从未发生过重试。",
        "", "原始汇总：final_comparison.json；逐 turn 预测在各方法目录的 predictions.jsonl；CSV 为 results.csv。"]
    target = root / "评测结果.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "report_verification.json").write_text(json.dumps({"verified_at": datetime.now().isoformat(),
        "cases_identical": True, "gold_identical": True, "observations_identical": True,
        "heads_fixed": True, "scores_recomputed": True, "methods": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "outputs/LATEST_MULTIWOZ_RESULTS.md").write_text(
        "修正评测已完成。\n\n[中文结果报告](" + target.as_posix() + ")\n\n[CSV](" + (root / "results.csv").as_posix() + ")\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    while not (args.root / "RESULTS_READY.md").exists():
        if (args.root / "RUN_FAILED.md").exists():
            raise SystemExit("Experiment supervisor failed; see RUN_FAILED.md")
        time.sleep(10)
    # The supervisor also writes the global pointer when it exits; finalize after it.
    for _ in range(12):
        status = args.root / "supervisor_status.json"
        if status.exists() and read(status).get("status") == "completed":
            break
        time.sleep(5)
    try:
        finalize(args.root)
    except Exception as exc:
        (args.root / "REPORT_CHECK_FAILED.md").write_text(str(exc), encoding="utf-8")
        raise
