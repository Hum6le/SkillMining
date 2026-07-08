#!/usr/bin/env python3
r"""ABCD 对话意图分类 — LLM 两阶段 + 意图库去重。

基于原始 Skill Mining 项目的 intent_classify.py 改造，适配 ABCD 数据集。

流程：
  1. 加载 ABCD 对话数据
  2. 格式化对话文本（用户 + 客服 + 动作 turn）
  3. 第一阶段：LLM 逐批分类，允许新增意图，意图库逐步膨胀
  4. 中间阶段：LLM 对意图库去重合并
  5. 第二阶段：LLM 重新分类，只能从去重后的意图库中选择
  6. 输出 intent → convo_id 映射 JSON

用法：
  python skill_mining/abcd_intent_classify.py --split test --max-sessions 100
  python skill_mining/abcd_intent_classify.py --split train --max-sessions 500 --sample 30

环境变量：
  DEEPSEEK_API_KEY   DeepSeek API key（必需）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from llm import chat as _chat

_OUTPUT_DIR = _SKILL_DIR / "output" / "abcd_intent"
BATCH_SIZE = 10

# 默认排出的非意图标签
DEFAULT_EXCLUDE_INTENTS = {"其他", "闲聊", "无明确意图", "寒暄", ""}


# ── ABCD 对话格式化 ───────────────────────────────────────────

def format_abcd_dialogue(conv: dict) -> str:
    """将 ABCD conversation 格式化为可读的对话文本。

    只取 customer 和 agent 的 turn（跳过 action），
    拼接成 ``角色：文本`` 格式。
    """
    speaker_map = {
        "customer": "用户",
        "agent": "客服",
        "action": None,  # 跳过系统动作
    }
    lines: list[str] = []
    for turn in conv.get("delexed", []):
        speaker = str(turn.get("speaker", "")).strip()
        label = speaker_map.get(speaker)
        if label is None:
            continue
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"{label}：{text}")

    if not lines:
        # fallback: 所有 turn 都放进去
        for turn in conv.get("delexed", []):
            text = str(turn.get("text", "")).strip()
            if text:
                lines.append(text)

    return "\n".join(lines)


def format_abcd_dialogue_with_actions(conv: dict) -> str:
    """格式化对话文本，包含系统动作（用 [动作] 标注）。"""
    speaker_map = {
        "customer": "用户",
        "agent": "客服",
        "action": "系统",
    }
    lines: list[str] = []
    for turn in conv.get("delexed", []):
        speaker = str(turn.get("speaker", "")).strip()
        label = speaker_map.get(speaker, speaker)
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        targets = turn.get("targets", [])
        if speaker == "action" and len(targets) >= 3 and targets[2]:
            action_name = str(targets[2])
            lines.append(f"[动作：{action_name}] {text}")
        else:
            lines.append(f"{label}：{text}")
    return "\n".join(lines)


# ── Memory 管理 ────────────────────────────────────────────────

def load_intent_memory(memory_path: Path) -> Dict[str, str]:
    """加载意图库 {意图名: 描述}。"""
    if not memory_path.exists():
        return {}
    try:
        data = json.loads(memory_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "intents" in data:
                items = data["intents"]
                return items if isinstance(items, dict) else {k: "" for k in items}
            return data
        if isinstance(data, list):
            return {k: "" for k in data}
        return {}
    except Exception:
        return {}


def save_intent_memory(memory_path: Path, intents: Dict[str, str]):
    """保存意图库。"""
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        json.dumps({"intents": intents}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Prompt 构建 ────────────────────────────────────────────────

def build_classify_prompt(
    dialogue: str,
    intent_lib: Dict[str, str],
    allow_new: bool = True,
    subflow_hint: str = "",
) -> str:
    """构建意图分类 prompt。"""
    intent_hint = ""
    if intent_lib:
        items = list(intent_lib.items())[:60]
        lib_text = "\n".join(f"- {k}" for k, _v in items)
        more = f"\n- ...（共 {len(intent_lib)} 个意图）" if len(intent_lib) > 60 else ""
        intent_hint = f"\n<已有意图库>\n{lib_text}{more}\n</已有意图库>"

    stage_note: str
    new_instruction: str
    if allow_new:
        stage_note = "第一阶段"
        new_instruction = (
            "4. 优先使用已有意图库中的意图（如果库中存在语义相近的，直接使用库中的名称）。\n"
            "5. 如果库中没有语义相近的意图，且该对话确实表达了明确的服务意图，"
            "则进行抽象并新增意图到 <new> 标签中。意图名应简洁（3-10字）、"
            "以名词/动词短语为主（如\"查询订单状态\"、\"申请退款\"、\"重置密码\"）。\n"
            "6. 如果对话属于\"其他\"类（无明确服务意图、纯闲聊、情绪宣泄等），"
            "在 <used> 中标注为\"其他\"。"
        )
    else:
        stage_note = "第二阶段"
        new_instruction = (
            "4. **只能使用已有意图库中的意图**，必须逐字复制库中已有的意图名。"
            "如果库中存在语义相近的意图，直接使用。\n"
            "5. **严禁新增意图、严禁自创意图名**。"
            "如果对话意图在库中找不到相近的，标注为\"其他\"。"
        )

    excluded = "、".join(f'"{x}"' for x in DEFAULT_EXCLUDE_INTENTS if x)

    subflow_line = ""
    if subflow_hint:
        subflow_line = f"\n【对话场景标签】（仅供参考，分类时以对话内容为准）：{subflow_hint}\n"

    return f"""你是一个任务导向对话（TOD）意图分类专家。请阅读以下客服对话，判断用户的核心服务意图。

【分类要求】：
1. 识别用户的主要服务诉求，而非客服/系统的操作行为。
2. 意图名应站在用户视角，描述"用户想做什么"（如"查询物流状态"、"申请退款"、"重置密码"、"修改订单地址"）。
3. 排除以下非意图类别（可归为"其他"）：{excluded}。
{new_instruction}

输出格式：
```
<intent>
<used>
[意图名1]
[意图名2]
</used>
<new>
[新意图名1]
[新意图名2]
</new>
</intent>
```

说明：
- 一段对话通常只有一个主意图，最多两个
- <used>：从库中选择的已有意图
- <new>：新增意图（{stage_note}需要输出）
- 如果没有明确的用户意图，<used>中标注"其他"
{intent_hint}
{subflow_line}
**当前待分类对话：**
{dialogue}
"""


# ── LLM 调用与解析 ─────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """调用 LLM API 并返回文本。"""
    return _chat(prompt, temperature=0.0, max_tokens=3072)


def parse_classify_output(text: str) -> Tuple[List[str], List[str]]:
    """解析 LLM 输出的意图分类结果。返回 (used_intents, new_intents)。"""
    used: List[str] = []
    new: List[str] = []

    intent_section = ""
    m = re.search(r"<intent>(.*?)</intent>", text, re.DOTALL)
    if m:
        intent_section = m.group(1)

    for tag, target in [("used", used), ("new", new)]:
        m_tag = re.search(rf"<{tag}>(.*?)</{tag}>", intent_section, re.DOTALL)
        if m_tag:
            for line in m_tag.group(1).strip().split("\n"):
                line = re.sub(r"^[-\d\.、\s]*", "", line.strip())
                if line:
                    target.append(line)

    return used, new


# ── 去重 ───────────────────────────────────────────────────────

def deduplicate_intents_with_llm(
    intent_lib: Dict[str, str],
    output_dir: Path,
) -> Dict[str, str]:
    """使用 LLM 对意图库进行去重和合并。"""
    if len(intent_lib) <= 1:
        return intent_lib

    print("\n正在使用 LLM 对意图库进行去重...")
    items = list(intent_lib.keys())
    text = "\n".join(f"{i + 1}. {v}" for i, v in enumerate(items))

    prompt = f"""请对以下客服意图库进行去重和合并。

当前意图库（共 {len(items)} 个意图）：
{text}

合并原则（请大胆合并，宁可合并过度也不保留冗余）：
1. **同义合并**：只是措辞不同但指向同一件事的，合并为一个。例如：
   - "催促发货"、"催发货"、"催单"、"催一下物流" -> 统一为 "催促发货"
   - "申请退款"、"要求退款"、"退款"、"想退款" -> 统一为 "申请退款"
   - "修改地址"、"改地址"、"修改收货地址"、"换地址" -> 统一为 "修改订单地址"
   - "查询订单"、"查订单"、"订单到哪了"、"看看订单" -> 统一为 "咨询订单状态"
   - "退货"、"申请退货"、"想退货"、"退货退款" -> 统一为 "申请退货"

2. **上下位合并**：过于具体的子类合并到上位意图。例如：
   - "申请换货" + "申请换尺码" + "申请换颜色" -> "申请换货"
   - "咨询发票开具时间" + "咨询发票内容" + "申请开具发票" + "修改发票抬头" -> "咨询/修改发票"
   - "咨询维修进度" + "申请售后维修" -> "咨询/申请售后"

3. **规范命名**：使用动宾短语（动词+名词），3-8字，如"催促发货"、"申请退款"、"咨询订单状态"。

4. **目标规模**：去重后应控制在 15-30 个意图以内。如果当前超过 30 个，说明合并不够彻底，请更激进地合并。

请输出去重后的意图列表（每行一个）："""

    output = call_llm(prompt)

    # 保存去重日志
    dedup_log = output_dir / "dedup_log.txt"
    dedup_log.write_text(
        f"=== 去重前 ({len(intent_lib)} 个意图) ===\n{text}\n\n"
        f"=== LLM 输出 ===\n{output}",
        encoding="utf-8",
    )

    new_lib: Dict[str, str] = {}
    for line in output.split("\n"):
        line = re.sub(r"^\d+[\.\)、]\s*", "", line.strip())
        if line and len(line) >= 2:
            new_lib[line] = ""

    print(f"  去重: {len(intent_lib)} -> {len(new_lib)} 个意图")
    if len(new_lib) == 0:
        print("  警告: LLM 去重返回为空，保留原库")
        return intent_lib
    return new_lib


# ── 批处理 ─────────────────────────────────────────────────────

def process_batch(
    dialogues: List[Tuple[str, str, str]],  # (convo_id, text, subflow)
    intent_lib: Dict[str, str],
    allow_new: bool,
    stage_name: str,
    output_dir: Path,
    memory_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """处理一批对话，返回 (分类结果列表, 更新后的意图库)。"""
    total = len(dialogues)
    total_batches = max(1, (total + BATCH_SIZE - 1) // BATCH_SIZE)
    results: List[Dict[str, Any]] = []

    print(f"\n{'=' * 50}")
    print(f"  {stage_name}")
    print(f"  对话数: {total}，批次: {total_batches}，批大小: {BATCH_SIZE}")
    print(f"{'=' * 50}")

    for batch_idx in range(total_batches):
        pos = batch_idx * BATCH_SIZE
        batch = dialogues[pos:pos + BATCH_SIZE]
        batch_new: List[str] = []

        for convo_id, dialogue_text, subflow in batch:
            try:
                raw = call_llm(build_classify_prompt(
                    dialogue_text, intent_lib, allow_new,
                    subflow_hint=f"场景: {subflow}" if subflow else "",
                ))
                used, new = parse_classify_output(raw)
            except Exception as e:
                print(f"  [{convo_id}] 出错: {e}")
                results.append({
                    "convo_id": convo_id,
                    "intent": "其他",
                    "used": [],
                    "new": [],
                    "error": str(e),
                })
                continue

            primary = used[0] if used else (new[0] if new else "其他")
            if primary in DEFAULT_EXCLUDE_INTENTS or not primary.strip():
                primary = "其他"

            results.append({
                "convo_id": convo_id,
                "intent": primary,
                "used": used,
                "new": new,
                "subflow": subflow,
            })

            if allow_new:
                batch_new.extend(new)

        # 批次结束后更新意图库
        if allow_new and batch_new:
            for intent in batch_new:
                if intent and intent not in DEFAULT_EXCLUDE_INTENTS and intent not in intent_lib:
                    intent_lib[intent] = ""
            save_intent_memory(memory_path, intent_lib)

        done = min(pos + BATCH_SIZE, total)
        print(f"  [{done}/{total}] 意图库大小: {len(intent_lib)}")

    return results, intent_lib


# ── 构建 intent → convo_id 映射 ─────────────────────────────────

def build_intent_map(results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """从分类结果构建 intent -> [convo_id] 映射，排除"其他"。"""
    intent_map: Dict[str, List[str]] = defaultdict(list)
    for item in results:
        intent = item.get("intent", "").strip()
        convo_id = str(item.get("convo_id", "")).strip()
        if intent == "其他" or not intent or not convo_id:
            continue
        intent_map[intent].append(convo_id)
    return dict(intent_map)


# ── 统计报告 ────────────────────────────────────────────────────

def print_intent_summary(intent_map: Dict[str, List[str]], results: List[Dict]):
    """打印意图分类汇总。"""
    print(f"\n{'=' * 50}")
    print("分类完成")
    print(f"{'=' * 50}")
    print(f"  总会话数: {len(results)}")
    print(f"  意图数:   {len(intent_map)}")

    # 统计 "其他"
    other_count = sum(1 for r in results if r.get("intent") == "其他")
    print(f"  归为\"其他\": {other_count}")

    # 按数量排序
    print(f"\n  意图分布:")
    for intent, sessions in sorted(intent_map.items(), key=lambda x: -len(x[1])):
        bar = "█" * min(len(sessions), 50)
        print(f"    {intent:20s}  {len(sessions):3d} sessions  {bar}")

    # subflow → intent 对应关系
    subflow_intent: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        sf = r.get("subflow", "?")
        intent = r.get("intent", "其他")
        subflow_intent[sf][intent] += 1

    print(f"\n  Subflow → Intent 映射（每个 subflow 的主要意图）:")
    for sf, intent_counts in sorted(subflow_intent.items()):
        top_intent = max(intent_counts, key=intent_counts.get)
        total_sf = sum(intent_counts.values())
        conf = intent_counts[top_intent] / total_sf if total_sf else 0
        if conf >= 0.5:
            print(f"    {sf:35s} → {top_intent:20s}  ({conf:.0%}, {total_sf} sessions)")
        else:
            details = ", ".join(f"{i}({c})" for i, c in intent_counts.most_common(3))
            print(f"    {sf:35s} → 混合: {details}")


# ── 主流程 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ABCD 对话意图分类 — LLM 两阶段 + 意图库去重"
    )
    parser.add_argument("--split", default="test",
                        choices=["train", "dev", "test"],
                        help="ABCD 数据分片")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="限制处理的对话数")
    parser.add_argument("--sample", type=int, default=0,
                        help="仅处理前 N 条（调试用，覆盖 --max-sessions）")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="跳过第一阶段（使用已有意图库）")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="跳过 LLM 去重")
    parser.add_argument("--include-actions", action="store_true",
                        help="对话格式化时包含系统动作描述")
    parser.add_argument("--output-dir", default=None,
                        help="自定义输出目录")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    memory_path = out_dir / "intent_memory.json"
    results_path = out_dir / "intent_results.json"
    intent_map_path = out_dir / "intent_session_map.json"
    summary_path = out_dir / "intent_summary.json"

    # ── 1. 加载 ABCD ──────────────────────────────────────────
    print(f"加载 ABCD {args.split} split...")
    convs = load_abcd_data(args.split)
    if args.sample > 0:
        convs = convs[:args.sample]
    elif args.max_sessions:
        convs = convs[:args.max_sessions]
    print(f"  {len(convs)} conversations")

    # ── 2. 格式化对话文本 ─────────────────────────────────────
    fmt_fn = format_abcd_dialogue_with_actions if args.include_actions else format_abcd_dialogue
    dialogues: List[Tuple[str, str, str]] = []
    for conv in convs:
        convo_id = str(conv.get("convo_id", "?"))
        text = fmt_fn(conv)
        subflow = str(conv.get("scenario", {}).get("subflow", ""))
        flow = str(conv.get("scenario", {}).get("flow", ""))
        subflow_hint = f"{flow}/{subflow}" if flow and subflow else (subflow or flow)
        if not text.strip():
            continue
        dialogues.append((convo_id, text, subflow_hint))

    print(f"  格式化完成: {len(dialogues)} 条有效对话")

    # ── 3. 加载或初始化意图库 ──────────────────────────────────
    intent_lib = load_intent_memory(memory_path)
    print(f"已有意图库: {len(intent_lib)} 个意图")

    # ── 4. 第一阶段 ────────────────────────────────────────────
    if not args.skip_stage1:
        results, intent_lib = process_batch(
            dialogues, intent_lib, allow_new=True,
            stage_name="第一阶段：抽取 + 新增意图",
            output_dir=out_dir, memory_path=memory_path,
        )
        print(f"\n第一阶段完成。意图库: {len(intent_lib)} 个意图")
    else:
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            print(f"从已有结果恢复: {len(results)} 条")
        else:
            results = []

    # ── 5. 中间阶段：去重 ──────────────────────────────────────
    if not args.skip_dedup and len(intent_lib) > 1:
        intent_lib = deduplicate_intents_with_llm(intent_lib, out_dir)
        save_intent_memory(memory_path, intent_lib)
    elif args.skip_dedup:
        print("\n跳过 LLM 去重")

    # ── 6. 第二阶段：闭世界重分类 ──────────────────────────────
    print(f"\n第二阶段：使用去重后的意图库（{len(intent_lib)} 个意图）重新分类...")
    results, intent_lib = process_batch(
        dialogues, intent_lib, allow_new=False,
        stage_name="第二阶段：仅从库中选择意图",
        output_dir=out_dir, memory_path=memory_path,
    )

    # ── 7. 输出 ────────────────────────────────────────────────
    # 详细结果
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n详细结果 -> {results_path}")

    # intent → convo_id 映射
    intent_map = build_intent_map(results)
    intent_map_path.write_text(
        json.dumps(intent_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"意图映射 -> {intent_map_path}")

    # 汇总报告
    print_intent_summary(intent_map, results)

    # 保存汇总 JSON
    summary = {
        "total_sessions": len(results),
        "num_intents": len(intent_map),
        "other_count": sum(1 for r in results if r.get("intent") == "其他"),
        "intent_distribution": {
            intent: len(sessions) for intent, sessions in
            sorted(intent_map.items(), key=lambda x: -len(x[1]))
        },
        "subflow_intent_mapping": {},
    }
    for r in results:
        sf = r.get("subflow", "?")
        intent = r.get("intent", "其他")
        if sf not in summary["subflow_intent_mapping"]:
            summary["subflow_intent_mapping"][sf] = {}
        summary["subflow_intent_mapping"][sf][intent] = \
            summary["subflow_intent_mapping"][sf].get(intent, 0) + 1

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nDone. Output: {out_dir}")
    print(f"  intent_memory.json       — 去重后意图库 ({len(intent_lib)} 个)")
    print(f"  intent_results.json      — 每个 session 的分类结果")
    print(f"  intent_session_map.json  — intent → convo_id 映射")
    print(f"  intent_summary.json      — 汇总统计")


if __name__ == "__main__":
    main()
