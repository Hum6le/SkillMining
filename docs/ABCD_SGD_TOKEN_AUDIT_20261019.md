# ABCD / SGD：Ours 与 Trace2Skill 总 token 反转的代码和数据审计

日期：2026-10-19。用户比较的是总 token，当前公司端 aggregate / 分阶段 usage 暂不可用。本文区分已确认的代码事实、历史小样本和待验证解释；没有启动模型调用或修改运行协议。

## 1. 主要发现：共享预测核心不等于相同外围调用预算

SGD Trace2Skill 的 `_predict` 读取 SKILL.md 和 references 后，调用 `sgd_aligned.predict_policy`。核心先选 action heads，再按需绑定 CALL 与 dialogue-act payload。

SGD Ours 在同一核心之前与中间，另执行两段 `SGDReferenceSession.collect`：action 前一段、选定 heads 后 payload 前一段。每段默认最多 4 轮，search/read/drop/finish 每次决策都调用模型。非空 reference 时，即使立即 finish，也会产生一次控制请求。JSON 格式修复和 transport retry 另计。

因此常见的有 dialogue-act 输出情形，Trace2Skill 的核心是 2 次请求，有 CALL 时增加 1 次；Ours 还可能增加 2–8 次 reference 控制请求。空 head 或没有可用 reference 时调用数会不同，这不是固定的每 turn 公式。

每次 SGD 控制请求都会携带 service schema、TRAIN 格式统计、历史 API 观测和对话前缀；参考库目录、保留 snippet 与前次工具结果也占输入。主 skill 变短只能减少输入的一部分，不能消除这些重复的请求开销。

ABCD 当前 ABCDAgent 的 reference 路径主要是一轮规划后本地检索。A2 运行器默认 reference_max_chars=1800；SGD ReferenceSession 默认 reference_context_chars=32768。后者是容量上限，不是每次实际加载量。不能把容量比直接当实际 token 比。

相关代码：

- `scripts/run_multiwoz_trace2skill_pilot.py::_predict`
- `eval_tod/sgd_aligned.py::predict_policy, SGDReferenceSession.collect, SGDAlignedSkillAgent.predict`
- `eval_tod/multiwoz_aligned.py::RuntimeLimits`
- `eval_tod/abcd/agent.py::predict_all_turns`
- `scripts/abcd_runtime_utils.py::RUNTIME_DEFAULTS`

## 2. 数据结构：当前 SGD 图的动作区分度较低，payload 多样性保留在节点内

按本地当前 ABCD coarse-flow TRAIN 文件，观察到每个 flow 有 22–30 种业务 action；一轮一个 action，平均约 1.03 个位置参数值。

SGD 节点按无序 `(act 类型, service family)` bundle 建立，CALL method 和具名参数都在 payload。以 20% 的 9 个实际运行 family 统计：

| Family | TRAIN turn | 观察到的 action bundle 种类 | act-slot 形态种类 |
|---|---:|---:|---:|
| Buses | 2,477 | 14 | 108 |
| Events | 4,109 | 13 | 91 |
| Homes | 998 | 12 | 25 |
| Hotels | 3,222 | 13 | 145 |
| Music | 1,244 | 13 | 29 |
| RentalCars | 2,021 | 13 | 93 |
| RideSharing | 1,255 | 7 | 49 |
| Travel | 898 | 6 | 11 |
| Weather | 367 | 4 | 7 |

形态统计定义：按 action bundle 分组，记录 canonical acts 中 `(intent, slot, 是否存在 value 字段)` 的组合，忽略实例 value；不含 API method/参数差异。它不是官方指标，也不是与 ABCD 参数数等价的复杂度度量。

推断：在当前节点定义下，SGD 中许多困难体现为同一 bundle 的字段选择、value 来源、API 结果、缺省/空值与输出组合，而不是不同图节点之间的业务流程选择。TraceCover 的结构压缩因此未必能抵消字段证据和额外检索控制的开销。这还不能单独证明方法在 SGD 上的结构收益更小，需要 matched-runtime 对照。

## 3. TRAIN 反馈开销也有适配差异

Ours standard 对每个训练 policy turn 做冻结 rollout，默认按最多 8 个 turn 组成诊断批次，再按最多 16 份报告形成顺序 reflection group。SGD 额外向 diagnosis/reflection 提供相关 family 的完整合法 schema、TRAIN 字段统计、当前 cards；reflection 启用了 trajectory exemplars。这些是之前为 named-slot 规则归纳增加的证据，确实会增加输入 token。

当前 `_project_record` 已剔除完整模型原文和 reference 决策 trace，只保留选卡信息、slot diff、可见 API 来源等。因此不能在没有日志证据时把成本归结为“又把整条 ReAct trace 反复塞进训练提示词”。

Trace2Skill 的旧 SGD v1 把多个 turn 压成一条分析记录，且 MAP 存在有序 slot 的任务冲突，可能低估了完整分析协议所需的学习资源。这两点已在 v2 修复。v2 使用逐对话成功/失败记录，经同一 MAP/REDUCE/APPLY 更新；不代表两方法的反馈粒度和请求预算从此完全相同。

## 4. 历史样本，仅用于验证开销机制

来源：`outputs/sgd_homes_aligned_diagnostic_20261018_194854`。这是旧版 Homes 的 115 个 policy turn 冻结诊断，使用旧资源，不能当作当前公司 20% 的结果或最终协议成绩。

| 评测预算 | Trace2Skill | Ours |
|---|---:|---:|
| HTTP/模型调用计数 | 244 | 996 |
| total tokens | 688,366 | 5,611,941 |

Ours 的 reference action/payload 决策及其格式修复合计 791 次请求、4,313,449 token，占其评测 token 的约 76.86%。此版本 payload reference 格式修复很多，后续已修输出契约，不能把这些修复次数外推至当前版本。

另有更早的 legacy Homes 训练：Ours generation 11,064,662 token，其中 train rollout 8,841,892，diagnosis 1,759,941，reflection 433,843。该记录只说明 runtime rollout 本身可能是总训练成本的大头，不是当前 aligned 协议的主结果。

## 5. 统计边界

- 总 token = 所有纳入范围的训练/编译/rollout/更新/评测请求的 prompt+completion token。必须分别列 generation/testing、调用数、平均输入输出长度与重试量。
- ABCD A2 的 `a2_total_usage` 才合并共享 offline mining 和 online generation；只取内层 llm_usage 可能漏算离线成本。
- 加载已有冻结资源的评测 usage 不包含历史训练；多次重启/补跑的运行成本也不能仅取最后一个目录代表全部花费。
- 公司 workflow 缺少原生 usage 时会估算 token，不能直接称为真实账单；跨客户端/模型比较前需核对 exact_calls / estimated_calls。
- 提高并发主要改变耗时。限流/格式重试会改变请求数和总 token，具体占比需要公司端日志。

## 6. 建议解释与下一步

当前证据支持：SGD 适配给 Ours 引入的额外控制调用、schema/字段反馈输入，加上小 action-bundle 图带来的有限结构压缩空间，能解释为什么总 token 可能高于 Trace2Skill。没有公司 usage 时不能确定各原因贡献比例，也不能声称修复后反转一定消失。

若主张是“资源组织更省 token”，后续应在已有冻结资源上做共享资源访问接口/预算的对照，并保留 action→payload、slot-policy 和 reference 的更新语义。完整系统比较仍应保留真实总成本，不能为了表格对齐只隐去 Ours 额外调用。尚未实施新的运行协议或启动此项实验。

## 附：纠正先前 ABCD 训练规模口径

本地现有 `data/eval/abcd/splits` coarse-flow 划分实际为 TRAIN action turn **29,206**、test action turn **7,276**，合计 **36,482**。原始官方 TRAIN 为 29,190。

因此，先前沿用用户约 3.6 万并将其直接称为 TRAIN 的说法不准确。若公司使用其他数据 manifest，应以实际运行的 train 文件为准；此前 SGD 44% = 36,026 的算数统计正确，但“匹配 ABCD TRAIN”这个目标口径需要据此更正。
