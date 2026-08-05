# AWM 与 Trace2Skill 的 LLM 信息可见性审计

本文档按当前仓库实际运行代码整理，重点说明每一次 LLM 调用拿到的信息、暴露原因，以及训练和测试之间的边界。

主要入口：

- AWM: scripts/run_awm_abcd.py, eval_tod/abcd/agent.py
- Trace2Skill ABCD: scripts/run_trace2skill_abcd.py
- Trace2Skill 通用进化器: Trace2Skill/pipeline/train.py, Trace2Skill/skill_evolver/parallel_evolving_agent.py
- 统一 LLM 封装: llm.py

## 1. 总览

| 方法 | 预测阶段 | 失败/经验分析 | 技能更新 | 主要学习资源 |
|---|---|---|---|---|
| AWM | reference lookup planning, action/slot/response prediction | induction 直接读取完整 trajectory | 更新 workflow，成功 dialogue 写入 exemplar | workflow + exemplar + reference |
| Trace2Skill | 与 AWM 共用 ABCD turn runner | AST mismatch error analysis | MAP -> REDUCE -> APPLY 更新 skill folder | SKILL.md + references + failure records |

关键公平性边界：预测时两种方法都不应看到当前 turn 的 gold action、gold slots、gold utterance id、AST 正误或未来对话。gold 只应进入训练后的 induction/error-analysis/evolution。

## 2. AWM 预测阶段

每个目标 turn 通常包含两次主要 LLM 调用：

    predict_all_turns
      -> _plan_reference_lookup: LLM call
      -> _lookup_reference: 本地检索，不调用 LLM
      -> action/response prompt: LLM call

### 2.1 Reference lookup planning

入口: eval_tod/abcd/agent.py::_plan_reference_lookup()

模型可见：

- 当前 turn 之前的 dialogue context；
- 当前 flow 和 subflow（expose_scenario_labels=True 时）；
- tool 的 JSON 接口、query 和 top_k 要求；
- 客服 system prompt、customer/order 摘要；
- 已有 AWM workflow 和 domain overlap exemplar。

模型不可见：

- 当前 turn 的 gold action、gold slots、gold utterance id；
- 当前 turn 之后的未来对话。

输出是 retrieve_reference 的计划，包括 thought、query、subflow、top_k。它把长 reference 文档压缩成当前 turn 的检索请求，避免将整份 reference 无选择地放入 prediction prompt。

### 2.2 Local reference lookup

_lookup_reference() 是本地 token-overlap 检索，不是 LLM 调用。它根据 query 在 reference.md section 中选择 snippet，返回标题、body、匹配状态和截断信息。

这样做的原因是让 reference 作为可审计证据进入下一个 prediction prompt，同时控制上下文长度。

### 2.3 Action/response prediction

入口: eval_tod/abcd/agent.py::predict_all_turns()

predict_actions=True 时要求模型输出：

    ACTION: <action_name>
    SLOTS: <slot1>, <slot2>
    RESPONSE: <natural-language response>

模型可见：

- 当前 turn 之前的完整原始文本 context；
- 可选 flow/subflow；
- customer name、member level；
- order id、payment method、商品类型摘要；
- AWM workflow；
- domain overlap 选出的 exemplar trajectory；
- reference lookup observation；
- 结构化输出协议和 resource policy。

模型不可见：

- 当前 turn 的 targets；
- gold action、gold slots、gold utterance id；
- ast_correct 和评测结果；
- 当前 turn 之后的文本。

暴露这些信息的原因：

- context 用于判断状态和下一步动作；
- customer/order 摘要是 slot grounding 的可观察实体来源；
- workflow/exemplar 提供跨 dialogue 的程序性知识；
- reference observation 提供 action schema、slot pattern 和 state transition 证据；
- 结构化输出使 AST 直接评估 predicted_action 和 predicted_slots，而不是从 response 反推。

注意：turn result 中保存的 reference 字段用于离线分析，不是把当前 gold 答案塞入 prediction prompt。

## 3. AWM 训练归纳阶段

scripts/run_awm_abcd.py 每个 batch 的顺序是：

    generate_all_turn_predictions(..., predict_actions=True)
    compute_ast_from_turn_results()
    agent.induce(..., turn_results=turn_results)
    agent.update_memory(..., turn_results=turn_results)

### 3.1 Workflow induction

入口: ABCDAgent.induce()

模型可见：

- batch 中每个 dialogue 的 flow/subflow 和 convo id；
- gold action sequence；
- 最后一条预测 response 摘要；
- 完整 turn trajectory，包括 turn_index、context、turn_type、predicted_action、predicted_slots、response、gold action/slots、action/slot correctness、ast_correct；
- 当前 workflow；
- add/refine/merge/delete 规则；
- Resource Use 约束。

为什么暴露 gold：induction 要从结束的训练 dialogue 学到正确 workflow。如果只有最后一条自然语言 response，模型无法知道中间 action boundary、slot 顺序和哪一轮 AST 出错。完整 trajectory 修复了此前 AWM 只传最后一条 prediction 的信息损失。

这是训练后离线归纳，不是测试时决策，因此允许使用 gold 作为监督信号。

输出是完整新 workflow，写入 WorkflowStore，并在后续 prediction system prompt 中可见。

### 3.2 Exemplar memory update

ABCDAgent.update_memory() 本身不调用 LLM。它根据 ast_score > 0.5 等条件筛选成功 dialogue，保存：

- dialogue id；
- flow/subflow domains；
- goal；
- 完整格式化 trajectory；
- trajectory_turns 结构化列表。

下一次预测时，memory 按 domain overlap 选择 exemplar，再注入 prediction system prompt。

## 4. Trace2Skill 预测阶段

Trace2Skill ABCD runner 复用 ABCDAgent 的 turn-level prediction，因此预测阶段可见信息与 AWM prediction call 基本一致：

- 当前历史 context；
- 可选 flow/subflow；
- customer/order 摘要；
- 当前 SKILL.md 和 references；
- reference lookup 计划与 observation；
- action/slot/response 协议。

区别是资源内容：AWM 使用 WorkflowStore 和 MemoryStore，Trace2Skill 使用 evolved skill folder。Trace2Skill 预测时不应看到当前 dialogue 的 AST mismatch、gold action/slots 或 failure report。

## 5. Trace2Skill 失败分析

入口: scripts/run_trace2skill_abcd.py::_run_verified_abcd_error_analysis()

模型可见：

- 失败 dialogue id、subflow/domains、goal/scenario 摘要；
- dialogue-level AST summary；
- 本地验证的 AST mismatch report；
- action turn index；
- predicted action/slots 和 gold action/slots；
- source agent turn 的 context、response、reference response；
- 完整 trajectory；
- 上一次 correction 未通过验证时的 feedback。

模型输出：

1. corrections JSON：每个错误 action turn 的 corrected action 和 corrected slots；
2. Failure Cause Item 和 Failure Memory Item markdown。

为什么暴露 gold：这是训练失败样本的监督式诊断，不是测试时决策。代码随后用 _verify_corrected_actions() 重新比较 action 和 ordered slots，防止模型给出语言上合理但标签仍错误的修正。

通用 MultiWOZ ErrorAnalyzer 还会看到 dialogue、domains、goal、info/success/request/booking 指标、prediction、gold goal 和 trajectory；它不是 ABCD AST 协议。ABCD runner 使用额外 verified mismatch report 补齐 action/slot 监督。

## 6. Trace2Skill MAP / REDUCE / APPLY

ParallelSkillEvolver.run(records, input_mode="records") 的主要调用如下。

### 6.1 MAP

System message 可见：

- 技能修改目标；
- failure cause/memory 字段解释；
- 允许的修改策略；
- patch 输出 schema；
- 不应修改的内容和大小约束。

User message 可见：

- 当前完整 skill folder，即 SKILL.md 和 references/*；
- 当前 batch 的 failure records，包括 pattern type/title/description/content；
- relation-to-skill 或 skill-reflection 建议；
- batch index/total batches；
- skill 行数、reference 文件数和上限。

目的：让各 failure batch 独立提出局部 patch，避免一次调用处理全部错误；同时让模型根据当前 skill 结构提出可应用编辑。

### 6.2 REDUCE

模型可见：

- 当前 skill folder 原文；
- 一组 MAP patches；
- 每个 patch 的 reasoning、file、op、target section/text、old/new content、changelog；
- merge schema、合并约束和 skill 大小状态。

目的：去重、解决冲突和确定 patch 顺序，逐层合并成一个可应用 patch。REDUCE 不重新读取原始 trajectory。

### 6.3 APPLY

模型可见：

- 当前 skill folder 原文；
- final merged patch 的 reasoning；
- final edits 的所有结构化字段；
- changelog；
- skill 文件大小和 reference 数量；
- apply constraints，要求输出完整文件内容。

目的：将结构化 patch 变成完整可落盘的 skill folder 文件。它必须看到原始文件，才能正确定位 section、插入内容、创建或删除 reference。

## 7. Trace2Skill 辅助调用

| 调用 | 触发条件 | 暴露内容 | 目的 |
|---|---|---|---|
| continuation | 输出被截断或 block 未闭合 | 之前的 system/user/assistant 对话和继续指令 | 完成被 token limit 截断的 patch |
| format retry | 输出无法解析或缺字段 | 原输出、解析错误、格式示例 | 修复格式，不改变任务语义 |
| translation | patch 的 target text 与当前文件不完全一致 | 当前 skill content 和待翻译 edits | 将不精确引用映射到实际文本 |
| verification | programmatic apply 后 skill 状态不合法 | 当前 skill folder 和 validation error | 修复结构性错误或断链 |

这些调用应计入总 LLM calls，但实验统计中最好与 MAP/REDUCE/APPLY 主阶段分开。

## 8. Gold 信息与评测边界

允许使用 gold 的位置：

- AWM training induce；
- AWM AST 计算后的 memory 筛选；
- Trace2Skill training error analysis；
- MAP 读取已解析的 failure cause/memory。

不允许使用 gold 的位置：

- AWM/Trace2Skill test prediction；
- validation/test reference planning；
- action/slot/response prediction；
- 将当前 turn 的 reference、gold action、gold slots、ast_correct 拼进 prediction prompt。

需要注意：

1. expose_scenario_labels=True 会给出 flow/subflow。这不是 gold action 泄漏，但属于任务标签可见，所有方法应固定配置。
2. AWM exemplar 必须只来自 train，不能混入 test。
3. Trace2Skill skill 只能由 train failure records 更新。
4. AWM induction prompt 同时包含 gold action sequence 和 ast_correct，这是训练监督，不得复用到 test prediction。
5. trajectory 包含 customer/order 原始值，对 slot grounding 必要，但共享日志或公开 prompt 前应脱敏。
6. action 和 utterance 是不同 target，文本评测只能使用 target_type == utterance 的 rows。

## 9. 一次 batch 的时序

### AWM

    每个 target turn:
      LLM: plan retrieve_reference
      local: retrieve reference snippets
      LLM: predict ACTION/SLOTS/RESPONSE
    完整 batch 后:
      local: compute AST and ast_correct
      LLM: induce workflow from gold + full trajectories
      local: select successful trajectories and save exemplars

### Trace2Skill

    每个 target turn:
      LLM: plan retrieve_reference
      local: retrieve reference snippets
      LLM: predict ACTION/SLOTS/RESPONSE
    评测后:
      local: build verified AST mismatch reports
      LLM: error analysis + corrections
      local: parse and verify reports
      LLM: MAP patches
      LLM: REDUCE patches
      LLM: APPLY final patch
      optional LLM: translation / verification / retry / continuation
      next evaluation: load evolved skill

## 10. 推荐记录字段

为了复现和解释性能差异，每个 LLM call 最好记录：

- method、phase、subflow、batch、dialogue_id、turn_index；
- system prompt 中的 resource type；
- user input token/字符数；
- contains_gold；
- gold 字段类型：action、slots、AST、error correction；
- parser/status；
- retry/continuation 次数；
- 是否修改持久化学习资源。

当前 AWM 通过 ResponseLogger 记录预测阶段原始 request/response；Trace2Skill evolver 在 intermediates/prompt_samples/ 保存主阶段 prompt sample。若要做严格审计，下一步应统一两者的 call metadata schema，而不是只统计总调用数。
