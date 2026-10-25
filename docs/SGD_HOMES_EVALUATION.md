# SGD Homes：当前评测集与 action / slot 协议

## 当前固定的数据

按用户确认，主评测继续使用已经运行的 Homes_1 部分，不加入 Homes_2/schema 迁移任务。

| 用途 | 对话数 | policy turn | CALL | 非 CALL | service |
|---|---:|---:|---:|---:|---|
| TRAIN | 85 | 507 | 158 | 349 | Homes_1 |
| 当前固定评测/开发集 | 18 | 115 | 32 | 83 | Homes_1 |

- 来源：`data/eval/sgd/splits_10pct_seed42/Homes/`。保持官方 train / validation 边界。
- 在每个官方 split 内，按 `sha256(seed + NUL + split + NUL + dialogue_id)` 排序，取 `ceil(10% × 对话数)`；seed=42。
- 抽样单位是完整对话。所选对话内所有属于 Homes 的 policy target 都计分，不按 CALL、难度或预测是否正确筛选。
- API 和带 Homes domain 的 system act 属于该场景；单场景对话的 global act 也纳入。多场景对话中没有标注归属的 global act 单独计数，不擅自分配给 Homes。这是已有 115-turn 定义的一部分。
- 三组必须使用相同 `(dialogue_id, turn_index)`、顺序及 gold。原始 dialogue ID 在 train / validation 间交集为 0。
- 这 115 turn 已用于错误分析，后续在它上面的数字是开发结果，不能宣称独立盲测。官方 test 的 25 段 Homes_2 对话暂不运行；未创建其他替代主测试集。
- 完整 ID 清单、文件 hash、排除的无归属 global turn 数在 `outputs/sgd_slot_alignment_20261018/evaluation_dataset.json`。

## 统一输入及输出

`eval_tod/sgd_aligned.py` 提供三方法共用的 action 选择器和固定 action 的 payload 执行器。

1. 可见输入：当前 turn 之前的 utterances、之前已经返回的 logged API 结果、完整 service schema，以及仅从上述 85 段 TRAIN 计算的序列化统计。
2. 先选无序的 `(intent, actual_service_id)` heads，再填 API method/parameters 和每个 act 的 slot/value。
3. 填槽阶段读取最终选择对应的完整 Action Card，而不是复用选择前的候选 cards。没有完全匹配的 bundle 时只能用适用的部分资源，不能替换成其他 action。
4. 三方法共用相同固定 heads 的输出规则。填槽阶段不能增加或删除已选 action；缺少绑定是 payload 错误，不重新决定 action。
5. 当且仅当预测 CALL 与当前 logged CALL 完全一致，回放适配器才提供当前 API result。当前 system response、当前 gold acts、belief state 和未来 turn 均不进入提示词。
6. 历史 API 结果来自已记录的真实前缀，而非本轮模型生成的历史。这是 teacher-forced 离线 policy 评测，不是完整模拟环境。三组边界相同。
7. Ours 不再从前一个 gold policy label 选择运行时图 source。TRAIN 监督标签仍可用于建图、分组和诊断。

## Slot 资源与 standard refine

- TraceCover、原生 standard 的全 TRAIN 冻结 rollout、配对成功/失败样本诊断、顺序 reflection 和 slot_policy upsert 继续复用现有代码。
- 每个节点以最多 4 个有不同 slot 形态的 TRAIN snippet 初始化。少于 4 个样例的节点使用实际可用数，不复制凑数。全 TRAIN 的 slot 形态计数说明备选模式，不作为必须输出的槽位并集。
- slot policy 保持自由自然语言。reference 可包含额外条件、异常及简短 TRAIN 示例；是否检索、检索多少、是否补查由 agent 决定，复用现有 search / read / drop / finish。
- 离线编译必须返回 exact node ID、非空 action_rule 和 slot_policy；错误 ID/缺字段必须重试，不能悄悄退回通用提示。写入后检查全部节点覆盖及 save/load。
- 原始失败原因：9 条离线 slot 更新用 `FindApartment => area`、`visit_date` 等作为存储 key，全部被拒绝，12 个节点最终都保留同一句通用 policy。
- 新编译器已经用真实 API 生成了 12 个节点的 policy，四段初始 snippet 和资源持久化已完成检查。
- 重新用旧 checkpoint 在新执行器上评测会明确标记为 frozen-resource diagnostic；这不能替代在新协议下重新完成三种方法的训练。

## 指标与当前错误

严格旧指标不变：Action Exact、Action micro F1、整轮 Payload Exact given Action、Joint。
另加 named-slot multiset P/R/F1、slot-name F1 及匹配 slot 名称后的 value exact，防止把单个 slot 准确率和整轮全对率混淆。

修复 reflection 后的旧协议 Ours：Action Exact=87/115=75.65%，Payload Exact given Action=29/87=33.33%，Joint=29/115=25.22%。
87 个 action 正确的 turn 中，58 个 payload 错误；以下类别有重叠：

| 现象 | turn 数 |
|---|---:|
| act slot 名称及数量相同，但 value/是否带 value 不同 | 40 |
| value 字段存在性不同 | 11 |
| 多填 act slot 或重复次数过多 | 11 |
| 漏填 act slot 或重复次数不足 | 4 |
| CALL method/parameters 不同 | 8 |

其中真实案例包括：`offer_intent(slot="ScheduleVisit")` 被多加 value；租金 `$2,800` 写成 `2800`；电话在先前 API result 中却没传给后续非 CALL turn；一次 offer 把多套房源全部输出。
这些需要修复输入、读取和 TRAIN 规则，不通过改变 gold、放宽评分或只保留简单 turn 来提高主指标。

## 运行入口

小规模接线检查（TRAIN smoke，不作为泛化分数）：

```powershell
python scripts/check_sgd_alignment.py --output-dir outputs/sgd_alignment_smoke_NEW --workers 4
```

固定 115-turn 开发诊断可使用同一脚本的 `--phase validation --ours-resource-dir ...`；baseline 复用旧资源时仍标记训练未重跑。

后续需要完整训练时，AWM/Ours 共用 `scripts/run_sgd_domain_eval.py --execution-protocol aligned`，Trace2Skill 使用 `scripts/run_sgd_trace2skill_pilot.py --execution-protocol aligned`。Ours 保留 `--ours-refinement-mode standard --ours-cases-per-node 4`。默认旧协议入口保留以便复现历史结果。

## 尚存限制

相同可见前缀不一定唯一确定人工选择的 request 槽位组合、下一套房源或表述形式。Logged-call replay 也不能回答任意不同 CALL。修复这些实现问题后仍需在固定 115 turn 上配对检查，不能预先保证 slot 指标提升。

## 输出协议失败的恢复

首次新执行器诊断中，Ours 31 个 turn 在填槽前的 reference 规划失败：模型输出了业务 CALL，而该阶段要求 search/read/drop/finish。Trace2Skill 另有 1 个 turn 把 slot 名称填在 action head 的 service 位置。它们是模型响应未通过协议校验，不是 API 超时或断连。

修复将各阶段输出约束放入独立 system 消息，方法知识仅作为 user 消息中的参考数据；reference 输出使用明确的 reference_action 字段。业务 CALL 仍是非法 reference 操作，不能被静默转换为 finish。

reference 的真实 section ID 与图节点 ID 属于不同地址空间。只暴露标题和 ID 的目录放在 controller 的 system 约束中，不注入全文；read/drop 必须使用真实目录地址。错误地址仍显式拒绝，不能仅将 policy- 前缀换成 ref-。协议修正使用 assistant 原始输出和 user 校验反馈组成的多轮消息，不把错误回答拼进大型知识段落；校验错误中列出真实可用地址，供模型自行选择或搜索。

恢复入口为 `scripts/recover_sgd_aligned_failures.py`。仅重试 `_execution_error` 标记的 turn，不根据 action/slot 是否正确选择样本。其余有效预测保持原样，原始预测文件、checkpoint、115 个样本的顺序及 gold 都做一致性核验，之后按完整分母重算三组指标。额外调用量另存，不冒充从头训练的新实验。

## 2026-10-18：字段级 TRAIN 诊断与原始配对 reflection

`eval_tod/sgd_slot_diagnostics.py` 在 action 正确时区分 missing/extra slot（保留重复次数）、value 字段缺失/空串/null/非空、value 内容及 CALL method/parameter 差异。不修改预测或 gold，不放宽严格评分；action 错误的样本明确标记 payload 比较不适用。

Aligned diagnosis 使用这些确定性差异，按成功、action 正确但 payload 错、action 错三类保留最多四个精简 TRAIN 例子。历史原始模型 trace 不进入优化提示。已可见的 API 前缀/当前 exact-call 结果保留上下文相关实体的少量字段，省略实体数明确标注，不能将被省略字段当作不存在。

Aligned standard reflection 现在接收精简原始 gold/prediction act JSON、字段差异、当前 Action Cards/slot policy、合法 service schema、TRAIN 序列化统计。每组 reflection 重新读取之前已接受的 native 更新；共享 ABCD/MultiWOZ 默认调用不启用这一额外证据。调度仍是全 TRAIN 冻结 rollout、并行 diagnosis、顺序 reflection，没有增加 constrained-repair 配对重放。

新一轮完整 Ours 训练和验证的指针：`outputs/latest_sgd_slot_feedback_run.json`。固定原 85 段/507 TRAIN policy turns 和 18 段/115 validation policy turns；新建 offline 资源、每节点四段 snippet，同一 aligned 执行器完成训练及评测。`scripts/monitor_sgd_slot_feedback.py --root <运行目录> --watch` 持续保存 progress.json，并在结果完成后自动生成 comparison.json/md。历史 AWM/Trace2Skill 只作为已保存的参考结果，本轮没有重新训练这两组。
