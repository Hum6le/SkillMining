# ToD Evaluation Framework & Dataset

## 1. 概述

本项目的评估框架 `eval_tod` 用于评估 **Task-oriented Dialogue (ToD)** 系统中 agent 输出的质量。核心流程为：

```
Dialogue (用户goal + 对话历史) → Agent → Prediction → Evaluator → Metrics
```

当前支持 **MultiWOZ 2.1** 数据集，并设计了可扩展的 `load_dataset()` 分发器来接入更多数据集。

---

## 2. 评估指标 (Metrics)

### 2.1 Information Rate（信息率）

**定义**：槽位级精度 —— 预测的 inform 槽位和 request 槽位与 ground truth 的匹配比例。

```
IR = (correct_inform + correct_request) / (total_inform + total_request)
```

- **inform 槽位**：从 `dialogue.goal.inform` 中提取，评估预测值是否与 ground truth 匹配（支持管道分隔的多值选项 `value1|value2`）
- **request 槽位**：从 `dialogue.goal.request` 中提取，检查预测是否请求了需要的槽位
- **值匹配**：大小写归一化、数字词归一化（`"one"` → `"1"`）、`"don't care"` 等变体统一
- 聚合模式：全局 `sum(correct)/sum(total)`（宏观 IR）和 `mean(per_dialogue_IR)`（微观 IR）

### 2.2 Success Rate（成功率）

**定义**：二元对话级通过/失败。一条对话成功当且仅当：

1. **所有** goal.inform 槽位被正确预测
2. **所有** goal.request 槽位出现在预测中
3. 对于有 booking 子槽位的领域（如 `book day`, `book people`），booking reference 非空

### 2.3 LLM-as-a-Judge（多智能体 LLM 评分）

**架构**：1 Combiner + 5 Specialist Judges

| Judge | 关注维度 | 评分范围 |
|-------|---------|---------|
| Task Completion Judge | 任务是否成功完成 | 1–5 |
| Slot Accuracy Judge | 槽位值是否准确 | 1–5 |
| Fluency & Coherence Judge | 对话是否自然流畅 | 1–5 |
| Helpfulness Judge | 回复是否有实际帮助 | 1–5 |
| Efficiency Judge | 对话效率是否合理 | 1–5 |

工作流：5 位 specialist 独立评分 → 1 位 combiner 综合打分 → 最终 5 维分数。

支持 `sample_size` 采样以控制成本。

---

## 3. Agent 架构

框架包含两种 agent：

### 3.1 ToolBasedTodAgent（ReAct Agent）

带工具调用的多轮推理 agent：

```
ReAct Loop (max_turns 可配置):
  Think → Action(choose_action) → Observation → ... → Final Prediction
```

可用工具：
- `query_db(domain, constraints)` — 查询 MultiWOZ 知识库
- `choose_action(action)` — 执行对话动作（inform/request/book 等）

### 3.2 SkillPreloadedAgent（技能注入 Agent）

继承 `ToolBasedTodAgent`，额外支持从 `skills_dir` 加载 SKILL.md 文件注入 system prompt：

```
skills_dir/
  tod/
    SKILL.md    ← 领域知识、最佳实践、常见陷阱
```

关键特性：
- `reload_skills()` — 动态重载技能（evolution 后无需重启）
- 技能内容前置到 system prompt 中
- 支持 YAML frontmatter（name, description）

---

## 4. 数据集：MultiWOZ 2.1

### 4.1 概览

| 属性 | 值 |
|------|-----|
| 全称 | Multi-Domain Wizard-of-Oz 2.1 |
| 类型 | 多领域任务型对话 |
| 对话数 | **10,438** |
| 划分 | train: 8,438 / validation: 1,000 / test: 1,000 |
| 语言 | 英语 |
| 来源 | [budzianowski/multiwoz](https://github.com/budzianowski/multiwoz) |

### 4.2 领域 (7 domains)

| 领域 | 描述 | 槽位数 |
|------|------|--------|
| `attraction` | 查找景点 | 9 |
| `hotel` | 查找并预订酒店 | 14 |
| `restaurant` | 查找并预订餐厅 | 12 |
| `taxi` | 预订出租车 | 6 |
| `train` | 查找火车 | 11 |
| `hospital` | 查找医院 | 4 |
| `police` | 查找警察局 | 4 |

### 4.3 数据结构

每条对话（`Dialogue`）包含：

```
dialogue_id:   "multiwoz21-train-0"
data_split:    "train" | "validation" | "test"
domains:       ["hotel", "train"]
goal:
  description:  "You are looking for a place to stay..."
  inform:       {hotel: {type: "hotel", parking: "yes", ...}}
  request:      {hotel: {}}
turns:
  - speaker:    "user" | "system"
    utterance:  "..."
    dialogue_acts:  {categorical: [...], non-categorical: [...], binary: [...]}
    state:      {hotel: {name: "", price range: "cheap", ...}}  # belief state
    booked:     {hotel: [...]}  # booking info
```

### 4.4 Agent 预测格式

```json
{
  "dialogue_id": "multiwoz21-test-0",
  "inform_slots": {
    "hotel": {"type": "hotel", "name": "Alexander B&B", ...},
    "train": {"destination": "cambridge", "day": "wednesday", ...}
  },
  "request_slots": {
    "hotel": ["address", "phone", "postcode"]
  },
  "booking": {
    "hotel": {"reference": "ABC123", "book_day": "tuesday", ...}
  }
}
```

### 4.5 评估流程

```python
from eval_tod import evaluate

result = evaluate(
    dataset_name="multiwoz21",
    data_path="data/eval/multiwoz21/data/data/dialogues.json",
    predictions_path="outputs/predictions.json",
    split="test",          # train / validation / test / None
    llm_judge=True,        # 启用多智能体 LLM 评分
    llm_judge_sample_size=50,  # 采样控制成本
)
# → {dataset, split, aggregate, per_dialogue, llm_judge}
```

---

## 5. 知识库 (Knowledge Base)

`MultiWOZKB` 加载 7 个领域的 JSON 数据库（hotel_db.json, restaurant_db.json 等），提供：

- `query(domain, constraints)` — 按约束查询实体
- `query_formatted(domain, constraints)` — 格式化文本输出
- `domains` — 可用领域列表

Agent 通过工具调用访问 KB，KB 在 pipeline 中创建一次、全局复用。

---

## 6. 评估输出格式

```json
{
  "dataset": "multiwoz21",
  "split": "test",
  "aggregate": {
    "num_dialogues": 1000,
    "info_rate": 0.7234,
    "mean_info_rate": 0.6987,
    "success_rate": 0.4500,
    "num_success": 450,
    "num_fail": 550,
    "llm_judge_scores": {
      "task_completion": 3.2,
      "slot_accuracy": 3.5,
      "dialogue_fluency": 4.1,
      "helpfulness": 3.8,
      "efficiency": 3.6
    },
    "per_domain": {
      "hotel": {"num_dialogues": 400, "info_rate": 0.75, "success_rate": 0.52},
      "train": {"num_dialogues": 350, "info_rate": 0.68, "success_rate": 0.38},
      ...
    }
  },
  "per_dialogue": [
    {"dialogue_id": "...", "info_rate": 0.833, "success": true, ...},
    ...
  ],
  "llm_judge": { ... }
}
```

---

## 7. 数据划分策略

MultiWOZ 2.1 已预置 train/val/test 划分。Pipeline 支持三种模式：

| 模式 | 训练 | 验证 | 测试 |
|------|------|------|------|
| 单 split | — | — | 指定 split（如 `test`） |
| 显式三分 | `--split train` | `--val-split validation` | `--test-split test` |
| 自动划分 | 从训练集划出 80% | 按 `val_fraction`（默认 0.2）从训练集中划出 | 另一部分或 test split |

---

## 8. 扩展性

通过 `eval_tod/data_loader.py` 中的 `_LOADERS` 注册表添加新数据集：

```python
_LOADERS: dict[str, callable] = {
    "multiwoz21": load_multiwoz21,
    # "multiwoz22": load_multiwoz22,   # 未来
    # "abcd":      load_abcd,          # 未来
}
```

新数据集只需实现 loader 函数，返回 `list[Dialogue]` 即可复用全部评估指标。
