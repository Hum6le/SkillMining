# 当前方法技术报告：图驱动的层级技能发现、骨干抽取与组织化编译

**状态**：当前离线主方法实现说明（2026-08-25）  
**适用入口**：`scripts/run_subflow_eval.py`  
**主配置**：`--mining-method backbone --subflow-discovery --backbone-compiler organized`

## 1. 问题与方法概览

目标是在 ABCD 的一个粗粒度 flow 内，从训练 session 自动构建可执行的 skill 资源，而不在运行时向 agent 暴露人工给定的细粒度 subflow 标签。方法将技能表示拆为两层：

1. **稳定结构层**：全局 action graph、每个潜在技能的 action backbone、局部允许转移。
2. **动态策略层**：自然语言 transition guard、slot policy、reference evidence 与 router card。

当前端到端流程为：

```text
Coarse-flow train sessions
  -> session-aware latent subflow discovery
  -> one action backbone per discovered group
  -> per-edge transition induction + transition-oriented reference
  -> organized graph-to-skill compilation
  -> progressive disclosure resources
  -> session-level skill router
  -> selected skill/reference ReAct inference
```

测试时仅给 router 和 agent 提供 skill、reference、dialogue context 与工具协议；不提供 `original_subflow`、人工类别名或测试标签。

## 2. 数据表示与预处理

### 2.1 Session action sequence

对每个 dialogue 的 `delexed` turn，保留目标为 `take_action` 的 agent action，并通过官方 action schema 规范化 action 名称。一个 session 转为：

```text
a_1 -> a_2 -> ... -> a_T
```

连续重复 action 会在 canonical operator sequence 中合并，以降低无意义重复；但原始 transition graph 中仍显式保留 self-edge，例如：

```text
verify-identity -> verify-identity
```

它可作为 retry/repetition 的证据，不能作为 backbone tree 的 parent edge。

### 2.2 Node、edge 与 evidence

全局图的 node 是规范化 action；edge 是同一 session 中相邻 action 的有向转移。对每条 edge 记录：

```text
support             transition 出现次数
num_sessions        包含该 transition 的不同 session 数
probability         P(target | source)
lift                P(target | source) / P(target)
condition           从历史可观察状态抽取的保守摘要
evidence_session_ids 代表性 session ID
```

slot evidence 不被硬编码成具体用户值。系统只记录 slot 位置、出现次数、可识别值类型及其在动作执行前的来源，例如 `current_customer`、`prior_dialogue`、`scenario` 或 `unresolved`。

## 3. 潜在 Subflow Discovery（当前接入版）

当前 `--subflow-discovery` 接入的是 `iterative_shared_node_splitting_v1`。它用于在一个粗 flow 内发现若干 session-supported latent group；发现阶段不使用真实的 `original_subflow`。

### 3.1 Session signature

每条 session 的 signature 同时保留 action node 与 transition edge：

```text
node:pull-up-account
node:enter-details
node:make-password
pull-up-account=>enter-details
enter-details=>make-password
```

这避免了仅保留 edge 时，删除共享节点后剩余 action 证据消失的问题。

### 3.2 迭代共享节点切分

每一轮从仍存在的 action 中挑选高 session-support 的候选共享节点。若暂时移除该节点及与其 incident 的 signature feature 后，能形成至少两个满足 `min_sessions` 的 residual motif candidate，则计算候选切分质量。

候选 group 不是按弱连通分量得到，而是由频繁 residual node/edge feature 召集：

1. 以频繁 residual feature 为 seed。
2. 收集包含 seed 的 session。
3. 保留这些 session 中重复出现的 feature，形成 group feature pattern。
4. 每条 session 按对各 pattern 的 coverage 分配到最佳 group。
5. 仅保留满足最小 session 支持度的 group。

当前 objective 不含额外正则项，按 group 独立计算后求和：

\[
J=\sum_{k=1}^{K}
\frac{\operatorname{support}_k}
{1+\operatorname{meanOverlap}_k}.
\]

其中 `support_k` 的分母是该粗 flow 的全部训练 session；`meanOverlap_k` 是该 group feature pattern 与其他保留 group 的平均 Jaccard overlap。候选移除节点后若 objective 不再上升，则停止并保留上一轮。

被移除的节点仅用于揭示切分边界；下游对每个 group mining 时仍使用该 group session 的完整 action trajectory。

### 3.3 语义总结与 routing card

发现 group 后，不是逐 group 单独命名，而是将**所有 group**放入一次 LLM 调用中联合总结。每个 group 选择 assignment coverage 最高的 4 条完整原始 dialogue，此外提供：

```text
coarse scenario name and official prior
residual nodes / residual edges
group support
four representative full dialogues
```

LLM 为每个 `skill_id` 产生可判别的 routing card：

```text
name
routing description
customer goals
positive evidence
negative evidence
do-not-use conditions
distinguish-from rules
typical outcome
boundary uncertainty
```

运行时 router 一次性看到全部 cards 与当前 dialogue，选择一个 latent skill，然后仅加载该 skill 的 workflow/reference。选择结果写入 `skill_router_selections.json`，并以 `select_skill` 写入 ReAct trace。

### 3.4 当前未接入的研究分支

仓库中另有 `weighted_motif_prototypes_v1` 的实验性实现。它保留完整 session graph，以加权 transition motif 构造 prototype；在 `account_access` 的训练集诊断中能自动选择 3 个 group，并在事后 majority mapping 下达到 91.77%。该分支尚未替换当前 `--subflow-discovery` 的主入口，不能与当前主实验混为一谈。

## 4. Action Backbone Mining

每个 discovered group 独立调用 backbone miner。该阶段保留该 group 中所有观察到的 canonical action；不再采用 legacy HG vertex cover 删除 action。

### 4.1 Edge score

对 edge \(u\rightarrow v\) 定义：

\[
P(v\mid u)=\frac{c(u,v)}{c(u)},\qquad
\operatorname{lift}(u,v)=\frac{P(v\mid u)}{P(v)},
\]

\[
w(u,v)=\log(1+|\mathcal{S}_{u,v}|)
+0.5\log(\max(\operatorname{lift}(u,v),10^{-9})).
\]

第一项偏好跨 session 稳定支持的 edge；第二项偏好相对 target prior 更具关联性的转移。对每个 action 还加入虚拟根 `<START>` 的候选入边，其分数为：

\[
w(\texttt{<START>}\rightarrow v)=\log(1+c_{start}(v))-0.25.
\]

### 4.2 严格术语：maximum spanning arborescence

文档和讨论中常简称为“MST backbone”，但当前实现严格说是**最大生成有向树**，不是无向最小生成树：

```text
directed graph + virtual root <START>
  -> NetworkX maximum_spanning_arborescence
  -> one parent edge for every action node
```

tree 的作用是：

```text
connect all observed action nodes
provide a globally coherent compilation order
provide a stable primary structural skeleton
```

它不等价于唯一的真实执行路径，也不要求运行时严格沿树行走。

### 4.3 保留 local branch / retry edges

tree 以外的 edge 不会直接丢弃。对每个 source action：

1. backbone child 总会保留；
2. 非 backbone edge 需要达到 `min_branch_support`；
3. 每个 source 最多保留 `max_outgoing_edges` 条；
4. 若 target 位于 source 的 backbone ancestor chain，edge 标记为 `retry`，否则为 `branch`。

因此最终图包括：

```text
backbone edges: global organization
branch edges:   evidence-backed alternative continuations
retry edges:    repetition / return-to-prior-step behavior
```

默认参数为：

```text
max_outgoing_edges = 3
min_branch_support = 2
```

另有 `backbone_coverage` 变体：在不改变基础 edge score 的前提下，有限轮次交换 parent edge，优化 turn-edge score 与 session route coverage 的组合；它不是当前默认 `backbone` 主方法。

## 5. Transition Induction 与 Reference

### 5.1 Transition-oriented evidence

对每种保留 edge，从训练 session 采样最多若干条实例。对 `source -> target`，主要 evidence 是两动作之间的原始 dialogue；只有在解释必要时才附加早期对话上下文。

transition induction 按 source action 分组，将同一 source 的所有 outgoing target **放入同一次 LLM 调用共同比较**。模型必须判断每条 edge 属于：

```text
distinguishable
ordered_fallback
underspecified
```

输出使用自然语言解释 user utterance、agent proposal、接受/拒绝、补充信息、未解决请求等可见交互，不强制将原因压成预定义 state variable。这样可避免相邻分支分别诱导后得到相同、歧义的 guard。

### 5.2 Reference 组织

`reference.md` 是 transition-oriented：

```text
source action                一级标签
source -> target transition  二级标签
sampled dialogue evidence    每条 transition 1 到 3 条
```

reference 是可检索证据，不是可以复制到技能中的私有 slot 值。运行时通过 `retrieve_reference` 的 MCP-style lookup 检索相关片段；query 由模型在 ReAct 中规划，工具调用及 observation 都写入 trace。

## 6. Organized Compiler

organized compiler 的输入包括：

```text
complete directed backbone
all retained branch/retry edges
per-source transition induction results
action snippets / slot contracts
seed skill with complete backbone context
```

它不是只根据 main path 编写 workflow。编译顺序由 arborescence 的 breadth-first `compilation_order` 提供，但每个 action 节点也可结合其所有保留 outgoing edge 进行组织。

编译器通过受控写入协议生成 `skill.md`：

1. 先产生 backbone seed skill，提供全局结构上下文。
2. 将 transition induction 与完整 graph context 提供给 LLM。
3. 要求 LLM 以自然语言组织主干、局部分支、retry/rejoin、action rules 与 slot discipline。
4. 对 source decision coverage 进行标记检查，防止模型遗漏已保留的局部转移；若内部元数据 marker 缺失，当前实现可确定性补齐 marker，避免因不可见标记导致编译失败。
5. 将生成内容拆为 progressive disclosure 资源。

组织化编译强调：

```text
不要为每个 action 机械罗列 precondition/postcondition；
要以整体执行逻辑串联 backbone；
要将互斥或相似的 outgoing branches 放在同一决策语境中解释；
不把 reference 中的实例 slot value 写死到 policy。
```

为检验“skill organization”本身的影响，仓库保留 `unordered` control：它使用同一 graph、同一 transition induction 和同一 evidence，但以稳定伪随机顺序平铺 node/edge card，不向编译器提供 backbone/routing hierarchy。`compare` 可同时生成两者。

## 7. Progressive Disclosure 与运行时推理

编译后产物为：

```text
skill.md              compact backbone + routing/lookup policy
action_rules.md       详细 action procedure
slot_policies.md      有序 slot 采集、复用、缺失处理策略
reference.md          transition-specific dialogue evidence
subgraph.json         graph、backbone、local transitions、induction metadata
```

`skill.md` 明确告诉 agent：当 action procedure 或 slot value source 不确定时，通过对应 lookup 读取更详细的资源；不将所有细粒度内容无选择注入当前 prompt。系统 prompt 使用标签隔离：任务说明、skill、dialogue context、reference retrieval result 和 MCP 工具定义。

若启用 latent subflow discovery，运行时先进行一次 session-level router selection，再构造选定 latent skill 对应的 `ABCDAgent`。普通 backbone 模式则直接加载该 coarse flow 的单套 skill/reference。

## 8. 评测与诊断

每个粗 flow 独立训练/评测，随后做全局聚合。当前主要指标：

```text
AST joint accuracy
AST action-name accuracy
AST slot-value accuracy
CDS overall
BLEU-1
ROUGE-L
METEOR
```

产物中保存：

```text
*_predictions.json
*_react_traces.json
*_abcd_predictions.json
skill_router_selections.json        (启用 discovery 时)
semantic_subflows.json              (启用 discovery 时)
skill_router_card_induction_prompt.txt
skill_router_cards_prompt.txt
transition_cases.json
subgraph.json
```

真实 `original_subflow` 只可用于离线诊断 discovery quality，例如将 discovered group 事后映射到其中多数真实标签，计算 purity/majority-mapped accuracy；不得进入 discovery、router 或测试时 prompt。

## 9. 当前限制

1. 当前主 discovery 的共享节点残差切分仍可能在高度共享流程中产生混合 group，或遗失一部分 residual evidence。
2. action graph 能区分结构差异明显的 subflow，但不能保证恢复所有语义子意图；例如用户目标差异大、action 序列相近的场景需要 router card 与 utterance evidence 补充。
3. backbone 是编译与组织骨架，不能独立解决同一 source 下多个 branch guard 的语义冲突。
4. router 目前按完整 session 做一次选择，因此应在严格在线设定中改为只读取当前可见 dialogue prefix。
5. 当前离线 graph mining 不具备训练 batch 的自适应更新机制。

## 10. 后续：Structure-Preserving Online Refinement

在线阶段应借鉴 AWM 的 batch rollout 形式，但避免每个 batch 全文重写 skill。

```text
current graph-backed skill
  -> rollout one train batch
  -> compare prediction with gold AST/action/slot feedback
  -> localize evidence to router, transition, action, or slot-policy region
  -> update only dynamic policy/reference resources
  -> checkpoint provenance and version
```

保持固定：

```text
action vocabulary
latent group membership (first version)
backbone topology
retained graph topology
```

允许更新：

```text
router-card cues and exclusion rules
transition guard wording
slot acquisition / reuse / missing-value policy
reference and verified exemplar evidence
```

建议在线 refinement 的准入规则是：**只对低置信、高冲突 edge 生成 guard patch**。这里的低置信可由累计成功/失败 evidence 校准；高冲突指同一 source 的多个 outgoing edge 在已有 guard 或 session evidence 上不可区分。首版已由 `skill_mining/online_refinement.py` 与 `scripts/run_backbone_online_refine.py` 实现：训练集 rollout 维护可序列化的 `skill_dag_state.json`，为每条 edge 累积 Beta-smoothed reliability、slot failure、competing target 和有界 evidence；确定性策略先选择 promote/sink/induce-guard，再让 LLM 只写 source-local sibling decision 的 guard。未解决或未晋升分支保留在 `online_reference.md`，不会写进主 skill。每批 rollout、patch、guard prompt/response 与资源快照都落盘，支持从状态文件续跑。

## 11. 关键实现文件

```text
skill_mining/semantic_subflow.py
skill_mining/backbone_workflow_mining.py
skill_mining/skill_writer.py
scripts/run_subflow_eval.py
skill_mining/online_refinement.py
scripts/run_backbone_online_refine.py
eval_tod/abcd/router_agent.py
eval_tod/abcd/agent.py
```
