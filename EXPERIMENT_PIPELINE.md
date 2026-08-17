# ABCD 实验 Pipeline

本文档说明本仓库当前实现的实验流程。三个方法的入口如下：

| 方法 | 入口脚本 | 训练粒度 |
| --- | --- | --- |
| AWM | `scripts/run_awm_abcd.py` | 单 subflow 对话，迭代更新 workflow 和 memory |
| Trace2Skill | `scripts/run_trace2skill_abcd.py` | 单 subflow 对话，按 batch 迭代演化 skill |
| Graph Mining（HG/sequence） | `scripts/run_subflow_eval.py` | 一次处理一个 subflow |

## 1. 共享设置

### 数据集

所有方法最终都使用以下 ABCD 数据：

```text
data/eval/abcd/data/abcd_v1.1.json
```

数据集包含 `train`、`dev` 和 `test` 对话。每个对话包括：

- `original`：包含具体值的原始 utterance；
- `delexed`：speaker 标签、action target、slot target 和 utterance id；
- `scenario`：flow、subflow 以及客户/订单上下文。

运行时 prompt 使用 `original` utterance，因此具体 slot value 仍然可见。AST
ground truth 从 `delexed.targets` 中提取。

### 共享运行时 Agent

所有 ABCD 评测都使用 `eval_tod/abcd/agent.py::ABCDAgent`。每个 agent turn
可以产生以下结构：

```text
ACTION: <backend action>
SLOTS: <ordered slot values>
RESPONSE: <natural-language response>
```

Agent 支持：

- 通过 `WorkflowStore` 注入 workflow/skill；
- 通过 `MemoryStore` 注入 exemplar；
- 由 LLM 规划 reference query，再调用本地 `retrieve_reference` 函数；
- 保存完整的 `react_trace`；
- 单 subflow 运行时保留 scenario 中的 flow/subflow 信息，便于复现实验和核验；
  不使用跨 subflow 混合训练。

这里没有微调模型参数。所谓训练，是通过重复调用 LLM 更新 workflow、exemplar
memory，或 skill/reference 文件。

### 共享评测

公共入口是 `eval_tod/cli.py::evaluate_abcd_bundle`。

文本指标：

- BERTScore precision、recall 和 F1；
- BLEU-1 和 BLEU-4；
- ROUGE-1、ROUGE-2 和 ROUGE-L；
- METEOR。

Action 指标：

- action name accuracy；
- 有序 slot-value exact accuracy；
- joint AST：action 和 slots 必须同时正确；
- CDS：级联式序列指标。

Action 指标是严格匹配的。自然语言回复看起来合理，但 backend action 错误时，
仍然不能获得 joint AST 分数。

## 2. 共享的 Turn-level 流程

每个 agent turn 的运行路径为：

```text
对话上下文
    -> 可选的 LLM reference-query 规划
    -> retrieve_reference 查询
    -> 注入 workflow/skill 和 exemplar 的 prompt
    -> 生成 action、有序 slots 和回复
    -> parser 解析
    -> 文本指标和 AST/CDS 评测
```

保存的 turn 记录包含 `predicted_action`、`predicted_slots`、原始 response、
`reference_lookup` 和 `react_trace`。

要确认确实命中了 reference，可以检查：

```json
{
  "reference_lookup": {
    "executed": true,
    "status": "matched",
    "selected_sections": ["..."]
  }
}
```

`status=no_reference_loaded` 表示没有可用的 reference section，因此工具没有
真正执行查询。

## 3. AWM

### 训练流程

入口脚本：

```text
scripts/run_awm_abcd.py
```

默认训练循环为：

```text
加载 train/dev/test
    -> 构造每批 20 个对话
    -> 预测每个 agent turn
    -> 计算每个 dialogue 的 AST feedback
    -> 诱导并替换共享 workflow
    -> 将成功对话保存为 exemplar
    -> 从 exemplar 刷新 reference sections
    -> 定期保存 checkpoint
```

下一批会继续使用同一个已更新的 `WorkflowStore` 和 `MemoryStore`。最终评测时，
test agent 会收到最终 workflow、exemplar memory 和保存的 reference text。

### 单 subflow 运行

```powershell
python scripts/run_awm_abcd.py --subflow recover_username
```

AWM 每次只读取指定 subflow 的 train/dev/test 对话。所有 subflow 都需要分别
运行一次，最终使用汇总脚本计算全局结果。每个运行的 workflow、exemplar 和
reference 都只对应当前 subflow。

需要检查的输出：

- `awm_workflow.txt`：最终 workflow；
- `awm_exemplars.json`：最终成功 exemplar；
- `awm_reference.md`：评测时使用的 reference；
- `test_turn_predictions.json`：包含 `workflow_injected` 和 reference trace；
- `summary.json`：最终指标和资源规模。

### 只评测已完成的运行

如果不想重新训练，可以直接评测已有结果：

```powershell
python scripts/run_awm_abcd.py `
  --eval-only `
  --eval-from outputs\awm_abcd_<timestamp> `
  --subflow recover_username
```

`--eval-from` 也可以指向包含 `workflow.txt` 和 `exemplars.json` 的 checkpoint。

## 4. Trace2Skill

### 训练流程

入口脚本：

```text
scripts/run_trace2skill_abcd.py
```

当前 ABCD 版本采用如下迭代循环：

```text
seed SKILL.md + reference files
    -> 在训练对话上进行 seed prediction
    -> 构造基于 AST 的失败案例
    -> 将训练数据划分为多个 outer batch
    -> 使用当前 skill/reference 预测一个 batch
    -> 分析失败轨迹
    -> 将 skill 演化结果写回磁盘
    -> 下一批重新加载已演化的 SKILL.md/reference files
    -> 最终进行 seed 和 evolved test evaluation
```

这是迭代更新：第 `n+1` 批使用第 `n` 批产生的 skill，而不是每次从原始 seed
独立打 patch。

本地 loader 支持以下 reference 布局：

```text
<skill-dir>/references/*.md       # 官方 Trace2Skill 布局
<skill-dir>/reference.md          # 旧版/本地布局
```

常见输出包括：

- `evolved_skill/SKILL.md`；
- `evolved_skill/references/` 或 `evolved_skill/reference.md`；
- `seed_train_turns.json`；
- `train_batches/batch_*/turns.json`；
- `train_batches/batch_*/batch_summary.json`；
- `seed_test_eval.json` 和 `evolved_test_eval.json`；
- `summary.json`，其中包含 reference 字符数和 batch 历史。

### 单 subflow 运行

```powershell
python scripts/run_trace2skill_abcd.py `
  --subflow recover_username `
  --evolution-batch-size 25 `
  --continue-on-batch-error
```

Trace2Skill 每次只读取指定 subflow 的 train/test 对话，并在该 subflow 内进行
seed、failure analysis、batch 演化和最终评测。使用 `--resume-dir` 可以继续
该 subflow 的已有迭代运行。

## 5. Graph Mining

### 训练与评测

入口脚本：

```text
scripts/run_subflow_eval.py
```

与 AWM 和 Trace2Skill 不同，该脚本分别处理每个 subflow：

```text
加载一个 subflow 的 train/test split
    -> 从训练轨迹中挖掘 skill
    -> 根据 operator snippet 构造 reference.md
    -> 评测 seed agent
    -> 评测 mined-skill agent
    -> 报告两者差值
```

可用的挖掘模式：

- `sequence`：带 support threshold 的 canonical action sequence mining；
- `legacy`：hypergraph/vertex-cover/subgraph mining。

典型命令：

```powershell
python scripts/run_subflow_eval.py --subflow recover_username
python scripts/run_subflow_eval.py --subflow recover_username --mining-method legacy
```

Graph Mining 的输出是 subflow-specific，通常包括：

- `skill.md`；
- `reference.md`；
- `subgraph.json`；
- seed/mined predictions 和 React traces；
- 文本指标、AST、action accuracy、slot accuracy 和 CDS。

Graph Mining 的 `--all` 只是依次执行相互独立的 subflow 任务，不会把对话混在
同一个 skill 中。输出的 `summary.json` 同时保留每个 subflow 的结果和加权的
`__global__` 汇总。

### 全局统计

三个方法都应先得到一组独立的 subflow 输出，再用以下脚本汇总：

```powershell
python scripts/aggregate_subflow_results.py `
  --runs outputs\awm_abcd_recover_username outputs\awm_abcd_recover_password `
  --output outputs\awm_global.json
```

也可以将多个运行目录放在同一个父目录下递归扫描：

```powershell
python scripts/aggregate_subflow_results.py `
  --runs outputs\awm_runs `
  --recursive `
  --output outputs\awm_global.json
```

汇总规则为：文本指标按评测样本数加权，action/slot/AST 按 action turn 数
加权，CDS 按 test session 数加权。该统计方式对应把各 subflow 的评测样本
合并后计算全局指标，而不是对 subflow 百分比做简单平均。

## 6. 共享模块与方法差异

| 组件 | 共享情况 | AWM | Trace2Skill | Graph Mining |
| --- | --- | --- | --- | --- |
| ABCD train/test data | 是 | 单个 subflow | 单个 subflow | 单个 subflow |
| 运行时 `ABCDAgent` | 是 | workflow + exemplars | 演化中的 SKILL.md | mined skill |
| Turn-level action/slot 格式 | 是 | 是 | 是 | 是 |
| Reference lookup 接口 | 是 | exemplar-derived 或外部 reference | `references/*.md` / `reference.md` | 挖掘得到的 `reference.md` |
| 训练更新 | - | workflow 替换 + memory | failure analysis + skill evolution | 一次性 offline mining |
| Batch 迭代 | - | 每批 20 个对话 | 可配置 evolution batch | 没有跨 batch 更新 |
| Label hiding | 不使用 | 不隐藏 | 不隐藏 | 不隐藏 |
| 主要产物 | - | workflow + exemplars | SKILL.md + references | skill.md + subgraph |
| 全局结果 | - | `aggregate_subflow_results.py` | `aggregate_subflow_results.py` | `summary.json` 的 `__global__` |

## 7. 可复现性检查

解释结果前，建议检查：

1. 每个运行的 `summary.json` 是否只对应一个 `subflow`，且 train/test 数量符合预期；
2. 训练过的 AWM 运行中 `workflow_lines` 是否非零；
3. 预期注入 workflow 时，`test_turn_predictions.json` 是否包含
   `workflow_injected=true`；
4. 预期查询 reference 的 turn 中，`reference_lookup.status` 是否为
   `matched`；
5. 解释 AST 前，`predicted_action` 和 `predicted_slots` 是否已经填充；
6. 同时比较 AST、action accuracy 和 slot accuracy，不能只看 joint AST；
7. 谨慎解释 CDS：当前实现中，没有剩余 actionable turn 的终止位置也可能
   自动贡献较小的正值。
