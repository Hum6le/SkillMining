# 实验总 Pipeline（方法无关视角）

本文档整理当前仓库的公共实验运行架构。重点是一次实验从本地准备、数据进入、Agent 执行、轨迹记录、评测到结果汇总的完整工作流；具体方法只被视为在同一个执行框架中的知识注入或状态更新模块。

## 1. 总体架构

```text
实验配置与 API
    |
    v
数据集 + ontology + KB + 场景/子流切分
    |
    v
构造 Agent 运行时
  ├─ 基础 system prompt
  ├─ 可选 skill / workflow / exemplar memory
  ├─ 可选 reference 检索器
  └─ 可选领域工具（ABCD reference、MultiWOZ query_db、Spreadsheet bash）
    |
    v
按 dialogue / subflow / turn 执行 LLM Agent
    |
    +--> prediction：动作、slots、自然语言回复
    +--> react_trace：prompt、tool call、observation、解析结果
    +--> working files / intermediate artifacts
    |
    v
统一评测
  ├─ 文本回复质量
  ├─ 结构化 action / slot / joint AST
  ├─ dialogue/session-level CDS 或 Success
  └─ 可选 LLM-as-a-Judge
    |
    v
错误分析与知识状态更新（训练阶段）
    |
    v
保存 checkpoint / summary / changelog
    |
    v
跨 subflow 加权汇总与最终对比
```

这里的核心实验单位是一个独立 `subflow`。每个 subflow 单独读取对应的 train/test 对话，单独产生知识状态和输出目录，最后再由汇总脚本合并。因此，跨 subflow 的总结果不是简单平均，而是按样本数、action turn 数或 session 数加权。

## 2. 本地准备的运行组件

### 2.1 LLM 与运行配置

- `llm.py`：统一创建 OpenAI-compatible LLM client，并解析模型、API key、base URL 等配置。
- 环境变量：通常需要 `OPENAI_API_KEY`，以及可选的 `OPENAI_BASE_URL`。
- 各脚本支持 model、请求频率、重试次数、指数退避等参数；长实验应保留命令行参数和实际模型配置，保证结果可追溯。
- 当前实验不是本地微调模型。所谓“训练/演化”主要是通过 LLM 分析轨迹并更新 workflow、memory、skill 或 reference 文件。

### 2.2 Agent 工具层

公共 Agent 基类和运行组件位于 `eval_tod/`：

- `eval_tod/agent_tool.py`：KB-backed ReAct Agent。LLM 在对话过程中可以调用 `query_db`，然后根据 observation 生成下一步结构化结果。
- `eval_tod/agent_skill.py`：在 ReAct Agent 上加载本地 `SKILL.md` 和 `references/*.md`，将其注入 system prompt。
- `eval_tod/abcd/agent.py`：ABCD 专用逐 turn Agent。支持 workflow、exemplar memory 和 `retrieve_reference`，并保存完整的 lookup/react trace。
- `eval_tod/agent.py`：单次 LLM 调用的简单 ToD Agent，主要用于无 ReAct 或轻量基线。
- `awm/agent.py`：AWM 适配层，把 `WorkflowStore`、`MemoryStore` 接到相同的 Agent 执行接口。

每个 turn 的标准产出包括：

```text
ACTION: <backend action>
SLOTS: <ordered slot values>
RESPONSE: <natural-language response>
```

ABCD 评测要求 action 和 slots 可解析；自然语言回复和结构化预测会分别进入不同指标。

### 2.3 知识与工具资源

- 数据集原始对话：提供 customer/agent/system 上下文和 ground truth。
- `ontology.json`：动作、slot 和合法值的约束。
- KB：MultiWOZ 或 ABCD 的本地数据库，供 `query_db` 查询。
- `SKILL.md`：领域通用操作知识。
- `references/*.md` 或 `reference.md`：按操作/场景组织的局部参考片段。
- AWM 的 `WorkflowStore`：文本形式的流程模式。
- AWM 的 `MemoryStore`：从成功样本中保留的 exemplar。

AWM 的两类资源使用方式需要区分：

- **Exemplar**：当前实现由运行时按 `flow/subflow` domain overlap 自动检索 top-k，
  不是 LLM 发起的 ReAct/MCP tool call；实际选择结果记录在 `exemplar_lookup`。
- **Reference**：当 reference section 已加载时，由 LLM 先规划简短 query，再执行
  `retrieve_reference`；query、命中 section 和 observation 都记录在 `react_trace`。

这些资源都在 prompt 构造阶段进入 Agent；不同方法的差异主要体现在资源如何产生和何时更新。

## 3. 数据处理与实验样本

### 3.1 MultiWOZ ToD 链路

数据入口和切分工具：

- `eval_tod/data.py`：统一数据加载、dialogue schema 和按 scenario 切分。
- `scripts/split_multiwoz.py`：重新生成 train/val/test 及 scenario 文件。
- `eval_tod/kb.py`：加载本地多领域数据库。

运行时使用原始 dialogue goal、对话历史和 domain/slot ontology。数据可以是仓库附带的 dummy/sample split，也可以是完整 MultiWOZ 2.1。推荐固定抽样比例、随机种子和 scenario 划分文件，不要在不同方法之间重新切分。

### 3.2 ABCD 链路

数据入口：

- `eval_tod/abcd/data.py`：加载 ABCD 的 `train/dev/test` 和 subflow 对话。
- 原始资源通常位于 `data/eval/abcd/data/`，包括 `abcd_v1.1.json`、`ontology.json`、`guidelines.json`、`kb.json` 等。
- 一个 dialogue 内含 flow、subflow、scenario、原始 utterance 和 delexed action/slot target。

运行时 prompt 使用原始 utterance；action、slot 和序列评测使用 delexed target。ABCD 实验应按 subflow 独立运行：

```text
读取一个 subflow 的 train/test
    -> train 阶段产生或更新知识状态
    -> test 阶段冻结该状态并执行 Agent
    -> 保存该 subflow 的 summary 和原始轨迹
```

### 3.3 SpreadsheetBench 链路

SpreadsheetBench 使用 `Trace2Skill/` 下的 runner：

- `Trace2Skill/run_spreadsheetbench.py`：并行执行 spreadsheet Agent。
- `Trace2Skill/spreadsheet_agent/tools/bash.py`：提供命令行/文件处理工具。
- `Trace2Skill/spreadsheet_agent/skills/`：运行时加载的 spreadsheet skill。
- `Trace2Skill/evaluate_with_official.py`：官方兼容评测。

该链路的公共形态仍然是“任务输入 → Agent 工具调用 → 工作目录和输出文件 → 轨迹日志 → 官方评分”；它与 ABCD 共用 LLM、日志、skill 演化的思想，但预测格式和评分器不同，不应混用指标。

## 4. Agent 执行阶段

### 4.1 单 turn 执行顺序

ABCD 公共 turn 流程如下：

```text
读取 dialogue history 和当前 scenario
    -> 构造 reference query（如启用）
    -> 执行 retrieve_reference，选取 top-k 片段
    -> 拼接 system prompt、skill/workflow、memory、reference、当前上下文
    -> LLM 生成 action / slots / response
    -> parser 解析结构化字段
    -> 写入 turn result 和 react_trace
```

`react_trace` 用于验证工具是否真的被调用，而不是只根据最终答案推测 Agent 是否使用了知识。重点字段包括：

- `workflow_injected`、`workflow_chars`、`memory_exemplars`；
- `reference_lookup.executed`、`status`、`selected_sections`；
- `exemplar_lookup.status`、query domains 和 selected exemplar ids；
- LLM reference query 的 raw output 和 fallback 状态；
- `predicted_action`、`predicted_slots`、原始 response；
- dialogue id、turn index、scenario/subflow。

`status=no_reference_loaded` 表示运行时没有可用 reference，不等于检索失败；结果解释时需要和 `matched`、`no_match` 等状态区分。

### 4.2 公共运行时与方法插拔位

| 公共部分 | 方法插拔部分 |
| --- | --- |
| 数据加载、scenario/subflow 切分 | skill、workflow、reference 的生成方式 |
| prompt 构造和 turn 循环 | 是否注入 workflow / exemplar / skill |
| parser、prediction schema、trace logger | 知识状态的更新时机和更新策略 |
| AST、文本指标、CDS 计算 | 失败样本如何分析、哪些样本进入下一轮 |
| checkpoint、summary、跨 subflow 汇总 | 具体方法的演化/挖掘算法 |

## 5. 训练/演化阶段的通用闭环

训练阶段不是修改模型参数，而是反复执行以下闭环：

```text
初始化 seed skill / 空 workflow / 空 memory
    -> 在 train batch 上运行 Agent
    -> 计算 batch-level 指标
    -> 从 prediction、ground truth、react trace 中分析错误
    -> 更新知识状态
    -> 保存 checkpoint
    -> 读取更新后的状态继续下一 batch
```

不同方法的状态更新形式：

- AWM：`WorkflowStore` 由轨迹归纳流程模式，`MemoryStore` 保留达到成功条件的 exemplar。
  生成的 workflow 会带有 `## Resource Use` 段落，说明 exemplar 的自动检索方式、
  `retrieve_reference` 的调用时机，以及不能直接复制实例 slot value 的约束。
- Trace2Skill：从失败轨迹产生修改建议，合并后直接更新 `SKILL.md` 和 reference 文件；下一 batch 重新加载上一 batch 的版本。
- Graph Mining：从训练轨迹离线挖掘 action/operator/subgraph，再生成 `skill.md` 和 `reference.md`，通常一次生成后评测。

训练阶段必须和 test 阶段分离。test 对话不应参与 workflow、memory、skill 或 reference 的更新。

## 6. 评测阶段

### 6.1 ABCD 指标

统一入口为 `eval_tod/cli.py::evaluate_abcd_bundle`，也可以对保存好的 prediction 使用 `scripts/eval_predictions.py`，无需再次调用 LLM。

文本回复指标：

- BERTScore precision/recall/F1；
- BLEU-1、BLEU-4；
- ROUGE-1、ROUGE-2、ROUGE-L；
- METEOR。

结构化 action 指标：

- action name accuracy；
- ordered slot-value exact accuracy；
- joint AST：action 和 slots 同时正确；
- CDS：dialogue/session 级别的序列指标。

### 6.2 ToD 指标

`eval_tod/metrics.py` 和 `eval_tod/evaluate.py` 提供：

- Information Rate；
- Success Rate；
- 可选的 LLM-as-a-Judge；
- per-dialogue 结果和 aggregate 结果。

### 6.3 SpreadsheetBench 指标

SpreadsheetBench 使用官方兼容 evaluator，基于任务输出文件和 benchmark reference 计算任务得分。其结果应和 ABCD 的 AST/CDS 分开保存、分开解读。

### 6.4 评测时的关键原则

1. 先保存原始预测和轨迹，再运行评测；这样指标代码变化后可以离线重算。
2. 同时报告 action、slot、joint AST 和 CDS；只看 joint AST 无法定位错误来源。
3. 检查预测解析是否成功。解析为空时不能当作普通文本错误。
4. 记录 test sessions、action turns、text samples 等分母，避免不同 subflow 的数字不可比。
5. 训练/演化阶段可以计算 train/dev 指标，但最终报告必须使用冻结状态在 test 上重新运行。

## 7. 失败分析与结果归档

ABCD 相关工具：

- `scripts/error_analysis_single.py`：分析单个运行或单个失败案例。
- `scripts/error_analysis.py`：对方法输出做批量错误分析。
- `scripts/aggregate_subflow_results.py`：读取多个 subflow 的 `summary.json` 并生成全局结果。

推荐每次运行目录至少保留：

```text
outputs/<method>_<subflow>_<timestamp>/
  config.json                 # 实际命令、模型、数据 split、seed、参数
  summary.json                # aggregate 指标、样本计数、运行状态
  predictions.json            # 原始结构化 prediction
  test_turn_predictions.json  # ABCD 逐 turn 结果
  react_traces.json           # prompt/tool/observation/response trace
  logs/                       # LLM 请求或运行日志
  skill/ 或 evolved_skill/    # test 使用的冻结知识版本
  workflow.txt                # AWM workflow（如适用）
  exemplars.json              # AWM memory（如适用）
  reference.md                # mined reference（如适用）
  changelog.json              # 迭代更新记录（如适用）
```

跨 subflow 汇总示例：

```powershell
python scripts/aggregate_subflow_results.py `
  --runs outputs\awm_abcd_recover_username outputs\awm_abcd_recover_password `
  --output outputs\awm_global.json
```

汇总规则是：文本指标按 text sample 数加权，action/slot/AST 按 action turn 数加权，CDS 按 test session 数加权。

## 8. 当前推荐的实际运行顺序

### 第一步：准备与锁定配置

确认数据、ontology、KB、seed skill/reference、模型/API、随机种子、并发数和输出目录。把最终命令写入 `config.json`。

### 第二步：运行公共 baseline

在不更新知识状态的条件下运行同一 test split，保存 prediction、trace 和 metrics。这个结果作为所有方法的共同参照。

### 第三步：执行方法内部训练/演化

只使用 train（必要时用 dev 做选择）运行 batch 闭环。每个 batch 保存知识状态和指标；中断后从 checkpoint 恢复，而不是从头重跑。

### 第四步：冻结最终知识状态并跑 test

重新加载最终 skill/workflow/memory/reference，在相同 test split 上执行。测试过程不再调用更新逻辑。

### 第五步：离线评测和错误分析

先运行统一 evaluator，再根据保存的 trace 做错误分类。需要比较时，确保所有方法使用相同数据、相同 test 样本、相同评分实现。

### 第六步：按 subflow 汇总

对每个方法分别收集 subflow 输出，使用 `aggregate_subflow_results.py` 生成全局 JSON。最终表格应同时保留 per-subflow 和 global 两层结果。

## 9. 复现检查清单

- [ ] API endpoint、model、seed、并发数和 retry 配置已记录。
- [ ] train/dev/test 文件和 subflow 划分固定且可定位。
- [ ] test 阶段没有写入 skill、workflow、memory 或 reference。
- [ ] 每个 turn 都有 prediction 和必要的 react trace。
- [ ] reference lookup 的执行状态和 selected sections 已保存。
- [ ] 结构化输出能被 parser 正确解析。
- [ ] summary 中包含各指标的样本分母。
- [ ] 汇总时使用按样本数/turn/session 数加权，而不是无权平均。
- [ ] 结果目录包含最终实际注入 Agent 的知识文件版本。
- [ ] 指标变化可以通过保存的 prediction 离线复算。

## 10. 主要入口索引

| 目的 | 入口 |
| --- | --- |
| MultiWOZ 通用 ToD pipeline | `scripts/run_tod_pipeline.py` |
| ABCD AWM | `scripts/run_awm_abcd.py` |
| ABCD Trace2Skill | `scripts/run_trace2skill_abcd.py` |
| ABCD Graph Mining | `scripts/run_subflow_eval.py` |
| ABCD 离线评测 | `scripts/eval_predictions.py` |
| ABCD 错误分析 | `scripts/error_analysis.py`、`scripts/error_analysis_single.py` |
| ABCD 全局汇总 | `scripts/aggregate_subflow_results.py` |
| SpreadsheetBench 执行 | `Trace2Skill/run_spreadsheetbench.py` |
| SpreadsheetBench 官方评测 | `Trace2Skill/evaluate_with_official.py` |
