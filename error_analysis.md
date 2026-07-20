# 1. Overall conclusion

HG 在 `recover_username` 子流程中的核心问题已经非常明确：动作识别整体可用，但结构化参数绑定与状态推进能力明显不足。整体指标呈现典型“action 强、slot 弱”特征：

- AST joint: 0.443
- Action-name accuracy: 0.8481
- Slot-value accuracy: 0.4494
- 失败 turn（含 ReAct trace）: 87
- joint_fail: 88，其中 `action_ok_slot_wrong=64`

这说明大部分失败并不是“不知道该做什么动作”，而是“知道该 verify，但不会把用户真实提供的信息正确映射成 slots”。

错误几乎高度集中在 `verify-identity` 阶段。HG 已经学会 recover flow 的表层 workflow 结构，但仍停留在 schema/template 模仿层，而没有形成稳定的：

- dialogue state tracking
- multi-turn slot aggregation
- entity grounding
- workflow transition control
- parser-level slot validation

`pull-up-account` 表现相对稳定（joint accuracy 0.9091），进一步说明主要瓶颈不是 action taxonomy，而是 verify 阶段的状态消费与实体绑定。

# 2. Main error types

最主要的错误类型可以归纳为以下几类：

- “动作正确、槽位错误”
  - 是绝对主导 failure mode
  - 模型预测了正确 action=`verify-identity`
  - 但 slots 为空、错误或退化成 schema token

- schema label 替代真实值
  - 输出：
    - `zip_code`
    - `phone_number`
    - `email_address`
  - 而不是：
    - `51909`
    - `(277) 341-6741`
    - `alice@email.com`

- placeholder 泄漏
  - 直接输出：
    - `<zip_code>`
    - `<phone>`
    - `<email>`
  - 没有完成 runtime grounding

- workflow 状态推进失败
  - 用户已提供两个以上 credentials
  - 模型仍继续 ask-for-verification-info
  - 没有进入 verification execution

- 多轮状态聚合失败
  - 历史 phone/email/name 未继承
  - 只消费当前 utterance
  - 无法累计 credentials

- action boundary confusion
  - `pull-up-account`
  - `verify-identity`
  两阶段混淆

- hallucinated slot value
  - 凭空生成 zip/phone/email
  - 或错误规范化用户输入

- credential overwrite 失败
  - 用户更新 zip/email 后
  - 旧值仍保留

# 3. Action-level patterns

动作层面的模式非常稳定。

最显著现象是：

- action classification 整体较强
- workflow transition control 较弱

具体表现如下：

- `verify-identity`
  - total=78
  - action_accuracy=0.8077
  - slot_accuracy=0.0
  - joint_accuracy=0.0

这说明模型大多数时候知道“应该进入 verify 阶段”，但无法正确执行 verify action。

高频 action-level failure 包括：

- 用户已提供 two-of-three credentials
  - 仍继续请求 credential
  - 没有真正 verify

- 已 `account pulled up`
  - 仍重复预测 `pull-up-account`

- 首次出现姓名时
  - 错误提前跳到 `verify-identity`

- workflow 卡死
  - 长时间停留在 collect-credential 阶段

- 少数 closing/send-link 提前触发
  - 尚未完成 verification 即结束流程

整体上体现出：

- workflow stage awareness 不稳定
- state transition 缺乏硬约束
- system action 未被当作状态信号消费

# 4. Slot/value failure patterns

slot 层面的失败模式高度统一，也是整个 chunk 的核心问题。

最典型 pattern：

- action 正确
- slots 错误

具体高频模式包括：

- slots 输出 schema 名称
  - `zip_code`
  - `phone_number`
  - `email`

- slots 保留 placeholder
  - `<zip_code>`
  - `<phone>`

- slots 全空

- 只保留 account holder
  - 漏掉 zip/phone/email

- 未继承历史 identity
  - account holder name 丢失

- 不会跨 turn 聚合 credentials

- parser 把 requested fields 当成 slot values

- 使用 hallucinated credential

- 不覆盖旧 credential
  - 用户更新 zip 后
  - 仍保留旧值

- credential typing 错误
  - pin 被误判为 phone

本质上，HG 学会的是：

- “verify identity 需要哪些字段”

但没有学会：

- “当前用户实际提供了哪些字段值”

即：
schema-level imitation > entity-level grounding。

# 5. ReAct/retrieval failure patterns, if traces are available

87 个失败 turn 带有 ReAct trace，暴露出非常一致的问题模式。

首先，大多数 case 并不是 retrieval miss。

常见情况是：

- retrieval 已命中正确 `verify-identity` reference
- reference 中也存在：
  - “用户提供两项 credential → 立即 verify”
- 但 reasoning/action parsing 仍失败

说明问题主要位于：

- state reasoning
- slot grounding
- parser extraction

而不是 retrieval coverage。

高频 retrieval/ReAct 问题包括：

- retrieval 被 workflow 模板主导
  - query 过度偏向：
    - “verify identity”
    - “zip/email/phone”
  - 导致模型固化成 ask-template

- reference 中 placeholder 过多
  - `<zip_code>`
  - `<email>`
  强化了 schema token 输出倾向

- reasoning 不消费当前 dialogue state
  - 忽略：
    - account already pulled up
    - credential count >= 2

- 缺乏显式阶段区分
  - request-verification-info
  - verification-in-progress
  混在一起

- retrieval recency bias 不足
  - recent SystemAction 未被优先使用

- parser 无 grounding 校验
  - schema token 被直接写入 slots

- retrieval pattern overfitting
  - 被高频 “collect credentials” 示例锁死

因此，这批问题本质上属于：

- retrieval 正确
- workflow reasoning 错误
- parser grounding 失败

# 6. Top 5 concrete improvements to the skill/workflow/reference

1. 为 `verify-identity` 增加强状态机约束

必须显式区分：

- request-verification-info
- verify-identity

并增加硬规则：

- 用户已提供任意两项有效 credentials
  → 必须立即执行 `verify-identity`
- 已 `account pulled up`
  → 禁止再次 `pull-up-account`
- verification in progress
  → 禁止继续 ask-for-credentials

这是最关键修复项。

2. 增加 slot grounding 与 parser validation

对 `verify-identity` 增加严格校验：

- slots 必须是真实实体值
- 禁止：
  - `zip_code`
  - `phone_number`
  - `<zip_code>`
- 若检测到 schema token
  → 自动回溯 customer utterance 重抽取

同时增加：

- hallucination blocking
- latest-value overwrite
- cross-turn slot aggregation

3. 引入显式 dialogue state summary

建议维护结构化状态：

```text
collected_credentials = {
  zip,
  phone,
  email,
  account_id
}

missing_credentials = {...}

account_pulled_up = true/false
verification_in_progress = true/false
```

并让 policy/reasoning 显式消费这些状态。

当前系统最大问题之一就是：
“状态存在于上下文里，但没有被稳定读取”。

4. 重构 reference/few-shot

重点补充：

- 用户分多轮提供 credentials
- 两项 credential 满足后立即 verify
- zip invalid → email replacement
- pull-up-account → verify transition
- noisy/semi-structured entity extraction
- slot type vs slot value 对比例

同时减少：

- 纯 placeholder 示例
- schema-only reference

避免 retrieval 污染。

5. 优化 retrieval 与 reasoning 联动

需要让 retrieval 更关注“当前 workflow stage”而不是关键词模板。

建议：

- recent SystemAction 提升权重
- credential-count 进入 retrieval query
- account pulled up 后
  → 优先召回 verify examples
- 增加 reasoning checklist：

```text
- Has account already been pulled up?
- How many credentials are collected?
- Are we collecting credentials or executing verification?
- Which real entity values are available now?
```

当前 HG 的主要问题并不是“不懂流程”，而是：
“无法把流程状态稳定映射成实体级结构化参数”。