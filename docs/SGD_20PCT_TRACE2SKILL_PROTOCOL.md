# SGD 20% 与本地 ABCD Trace2Skill 的协议核对

日期：2026-10-19。范围是本地 ABCD baseline 的学习协议对齐，不声明对原论文所有实验设置逐项复现。以下变化尚未产生真实模型评测结果，也未部署到公司电脑。

## 1. 审计发现与修复

原 SGD v1 复用了同一个 CombinedParallelSkillEvolver，但不能称为与 ABCD 完全相同：

1. SGD 把每类最多 8 个独立 turn 压成一条最多 4 个 item 的分析记录；ABCD 按对话独立生成记录。压缩可能丢失不同案例的条件边界。
2. 公共 MAP system prompt 和 mixed-record user prompt 写死了 ABCD 的有序 slot 列表，与 SGD 的无序 act 集合、具名参数不一致。这是提示词协议问题，但尚未通过实验证明它造成了多少性能损失。
3. SGD 外层更新默认每 20 段对话，ABCD 默认每 25 段。

已修复：

- SGD 改用 `sgd_trace2skill_abcd_dialogue_analysis_v2`，同一对话包含失败 turn 时生成一条失败记录，包含成功 turn 时生成一条成功记录；两者可同时存在。每个请求最多处理 8 段对话，必须逐对话返回结果。
- 失败分析保留 `failure_cause` 与 `failure_memory`；成功分析每对话最多 3 个 `success_memory`。失败记录附精确的 TRAIN gold/prediction 对照，进入原有 MAP/REDUCE 证据链。
- 分析读取当前完整主 skill 和有界的逐 turn 历史；成功分析可见混合正误轨迹，但明确只从正确 turn 提炼正例。不会把混合轨迹整体当成功。
- 分析阶段只使用 TRAIN。JSON 格式或覆盖 ID 不全可修复重试，没有新增执行候选修改的配对重放，也没有新增语义 self-verifier。
- 给通用进化器增加可选 task contract，SGD 明确使用无序 dialogue-act 集合和具名 CALL 参数；ABCD 默认的任务说明与排序语义保留。
- 外层默认 25 段对话；主 skill 使用带 YAML frontmatter 的通用 SGD seed，再附共享执行器 contract。
- 训练顺序仍为 rollout → 严格评分 → 成功/失败分析 → MAP → REDUCE → APPLY → 下一批。评测使用最终训练产物，评测反馈不进入更新。
- 进化协议版本写入数据 manifest 和 coordinator plan，旧版本目录不能通过普通 resume 静默混入新训练。

## 2. 对齐矩阵

| 项目 | 本地 ABCD | 新 SGD |
|---|---|---|
| 独立学习范围 | 一个 subflow | 一个 domain family |
| 外层更新批次 | 默认 25 段对话 | 默认 25 段对话 |
| 失败分析单元 | 有错误的对话 | 有错误的对话 |
| 成功分析单元 | 至少一个正确 turn 的对话 | 至少一个正确 turn 的对话 |
| 分析批次 | 默认 8 段对话/请求 | 最多 8 段对话/请求 |
| 分析输出 | 每对话独立记录 | 每对话独立记录 |
| MAP batch / merge batch | 8 / 5 | 8 / 5 |
| 最大 merge 层数 / temperature | 5 / 0.3 | 5 / 0.3 |
| 更新器 | CombinedParallelSkillEvolver | 同一个类 |
| 编辑流水线 | JSON，MAP/REDUCE/APPLY | 同一流水线 |
| 主 skill / 单 reference 行数上限 | 500 / 300 | 500 / 300 |
| TRAIN 监督，评测冻结 | 是 | 是 |
| 语义 self-verifier | 默认关闭 | 关闭 |
| 任务标签 | 单 action + 有序 slot | 无序 act 集合 + 具名 CALL 参数 |
| 预测执行器 | ABCDAgent | 三方法共享 SGD aligned 执行器 |

任务标签、预测调用拆分及工具观测协议随数据集适配，不应声称两套运行器的每个提示词和调用次数相同。SGD 的 action head 为 `(对话动作类型, 实际 service ID)`，具体 API method 在 CALL payload 阶段绑定。评测覆盖该 family 的 CALL 和非 CALL policy turn。

当前 v2 没有复制 ABCD 的 Markdown 分析报告中转或其缺失报告的确定性 fallback。SGD 使用逐案例 JSON，并在耗尽结构修复后显式记录错误。这是格式与错误处理差别，不作为新增学习组件。

## 3. 20% 规模

沿用官方边界、seed=42、family 内完整对话采样；TRAIN、validation、test 都采样 20%，然后对评测集执行相同 seen-schema 过滤。没有在旧样本中随机截取 turn，也未原地修改 10% 目录。

默认 validation 实验仍是 Buses、Events、Homes、Hotels、Music、RentalCars、RideSharing、Travel、Weather 共 9 个 family。

| 默认实际运行范围 | 10% | 20% |
|---|---:|---:|
| TRAIN policy turn / 方法 | 8,269 | 16,591 |
| TRAIN 去重对话 | 1,201 | 2,354 |
| validation policy turn / 方法 | 1,241 | 2,480 |
| validation 去重对话 | 203 | 394 |

20% 数据目录虽保留 16 个 family 的 TRAIN（共 29,709 个 turn），默认这轮只训练有同 schema validation 的 9 个 family。**本轮实际训练量是 16,591，并未达到 ABCD 的约 36,000。** 三方法总计 57,213 个 TRAIN+validation 目标事件；不等于 LLM 请求数。

20% 对话 ID 包含原 10% 对话 ID。已检查官方 split 间无对话泄漏、family 间无重复目标 turn。

## 4. 公司电脑指令

先同步本次代码，特别是两个通用进化器文件、新分析模块和 SGD runner；沿用公司现有 LLM 配置与 `configs/sgd_workflows.json`。代码同步不包含 SGD 数据或任何密钥。以下都在公司项目根目录执行。

### 4.1 准备数据

如已有 `data/eval/sgd/splits_by_domain/INDEX.json`，直接执行：

```powershell
python scripts/sample_sgd_by_domain.py --source data/eval/sgd/splits_by_domain --output data/eval/sgd/splits_20pct_seed42 --fraction 0.2 --seed 42
python scripts/prepare_sgd_10pct_protocol.py --source data/eval/sgd/splits_20pct_seed42 --output data/eval/sgd/protocol_20pct_seed42_seen_schema_v1 --schema-scope seen
```

第二个脚本保留历史文件名，但已支持其他采样比例，并写入正确的 `20pct` 版本元数据。输出目录必须是新的；若之前已成功生成，核对 INDEX 后跳过，不覆盖。

### 4.2 预检与启动

```powershell
python scripts/launch_sgd_server_batch.py --provider workflow --splits-dir data/eval/sgd/protocol_20pct_seed42_seen_schema_v1 --output-dir outputs/company_sgd_20pct_preflight_v2 --dry-run
python scripts/launch_sgd_server_batch.py --provider workflow --splits-dir data/eval/sgd/protocol_20pct_seed42_seen_schema_v1 --output-dir outputs/company_sgd_20pct_v2
```

预检没有模型调用，不能证明公司 API 连通或没有限流。正式命令沿用各方法现有 5 个 workflow ID 和每 ID 3 QPS 的请求池；后台进程隐藏窗口。不要复用旧 10% 输出目录，也不要用旧 Trace2Skill v1 skill 初始化这轮。

### 4.3 状态与结果

```powershell
python scripts/launch_sgd_server_batch.py --output-dir outputs/company_sgd_20pct_v2 --status
```

结果在 `outputs/company_sgd_20pct_v2/<method>/validation/aggregate.json`，分场景文件在 `<method>/validation/<method>/<family>/`。结果有效性仍需核对 missing、case_errors 和 counts.turns。

## 5. 本地验证

- 19 项针对性测试通过：分析单元/跨对话隔离/混合正误证据、结构修复、task contract、默认 ABCD 排序行为、mock TRAIN→更新→冻结评测、20% schema profile、既有 batch 与 MultiWOZ 分析回归。
- 三方法启动计划预检：各 9 个场景，共 27 个训练/验证任务。使用示例 workflow ID，仅构建命令，模型调用 0。
- Homes 独立 runner dry-run：TRAIN 998 turn，validation 237 turn。
- 尚未在公司 API 上验证此版本，也未启动真实训练。不能把旧 v1 的 9.35% Trace2Skill 整轮匹配率当作修复后结果。
