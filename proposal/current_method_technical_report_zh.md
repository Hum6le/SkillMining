# 当前方法技术报告：联合技能图发现、骨干抽取与组织化编译

**状态**：当前实现说明（2026-08-25）
**适用入口**：`scripts/run_subflow_eval.py`  
**当前实现配置**：`--mining-method backbone --backbone-compiler organized`

## 1. 问题与方法概览

目标是在 ABCD 的一个粗粒度 flow 内，从训练 session 构建一个可执行技能，而不在运行时向 agent 暴露人工给定的细粒度 subflow 标签。一个技能不再被定义为单独的 workflow 文本，而是一个成对对象：

\[
\mathcal K=(S,R).
\]

其中，`S` 是面向执行的 action DAG；`R` 是选择性外置的 residual control resource，保存主 DAG 不应无条件暴露的低频分支、retry/revisit、transition guard、action/slot policy 与其 dialogue evidence。`skill.md`、`reference.md`、`action_rules.md` 和 `slot_policies.md` 都是同一 \(\mathcal K\) 的不同运行时视图，而不是相互独立的技能。

`S` 由覆盖关键 action 的高支持 backbone `B` 和已验证的 forward branch `E^+` 组成：

\[
S=(V,B\cup E^+), \qquad B\text{ is a rooted spanning arborescence.}
\]

连续重复、retry 和回访 earlier action 的转移不加入 DAG，而是作为 `R` 中的恢复策略保留。这样保留真实轨迹证据，同时不会让主控制结构产生环。

当前端到端流程将 session-pattern 分析直接并入 backbone 边选择，而不是先产出独立的 latent subflow/skill：

```text
Coarse-flow train trajectories
  -> construct shared action-transition evidence graph
  -> infer temporary trajectory-pattern cohorts for edge contrast only
  -> select a discriminative session-aware backbone B
       |- shared high-support trunk
       |- cohort-specific, high-discriminativeness transitions
  -> retain residual branches / retries / evidence store R
  -> induce source-local guards and disclosure policies for R
  -> render one skill package into organized runtime resources
  -> ReAct inference with selective resource lookup
  -> optional train-only online refinement of R and verified forward branches
```

这里的 trajectory-pattern cohort 不是要被编译成独立 skill，也不进入运行时 router。它只是 backbone 边权的离线对比集合：高频公共前缀应被保留为 trunk；只在某类相似轨迹中稳定出现、且能区分该类轨迹的转移应被强化为 backbone 的关键决策边。MST/arborescence、transition induction 与 organized compiler 因而共同优化同一个 \(\mathcal K=(S,R)\)，而不是三个割裂的阶段。

测试时仅给 agent 提供 skill、reference、dialogue context 与工具协议；不提供 `original_subflow`、人工类别名或测试标签。

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

## 3. Discriminative Session-Aware Backbone Mining

本节将 session-level pattern 的归纳偏置直接并入 MST backbone：**相似轨迹应该共同支持某些关键 transition，但这些轨迹模式不需要被编译成独立 skill。** 一个粗 flow 最终只生成一套 \(\mathcal K=(S,R)\)。

### 3.1 Trajectory signature 与临时对比 cohort

每条 session 的 signature 同时保留 action node 与 transition edge：

```text
node:pull-up-account
node:enter-details
node:make-password
pull-up-account=>enter-details
enter-details=>make-password
```

这些 signature 仅在离线阶段形成具有足够 session support、且不过度互相包含的临时 cohort \(C_1,\ldots,C_M\)。cohort 的用途不是产生多份 skill 或运行时分流，而是度量某条 edge 是否只在一类相似轨迹中稳定出现。真实 `original_subflow` 不参与该过程。

### 3.2 Cohort-aware edge score

对 edge \(e=u\rightarrow v\)，保留原始 support 与 lift：

\[
P(v\mid u)=\frac{c(u,v)}{c(u)},\qquad
\operatorname{lift}(u,v)=\frac{P(v\mid u)}{P(v)}.
\]

对每个 cohort，计算该 edge 相对其他训练 session 的平滑 log-odds：

\[
d_m(e)=\log
\frac{(\operatorname{support}_{C_m}(e)+\epsilon)/(|C_m|+2\epsilon)}
{(\operatorname{support}_{\neg C_m}(e)+\epsilon)/(|\mathcal D\setminus C_m|+2\epsilon)}.
\]

令 \(d^+(e)=\max_m\max(d_m(e),0)\)，最终权重为：

\[
w(e)=\log(1+|\mathcal S_e|)
+0.5\log(\max(\operatorname{lift}(e),10^{-9}))
+\lambda_{\mathrm{disc}}\min(d^+(e),c).
\]

当前默认使用 \(\lambda_{\mathrm{disc}}=1.0\)、截断上界 \(c=3.0\)，因此区分性 bonus 的上限为 `3.0`。它只奖励 cohort-specific edge，不惩罚公共 trunk：`pull-up-account -> verify-identity` 仍可因高 support/lift 留在 backbone；`enter-details -> send-link` 若只稳定出现在一类相似轨迹中，则获得额外权重。可通过 `--backbone-discriminative-lambda` 和 `--backbone-discriminative-clip` 做开发集消融。

`subgraph.json` 应记录：

```text
support_in_cohort / support_outside_cohort
best_cohort_id
discriminative_log_odds
base_weight
final_backbone_weight
```

### 3.3 Maximum spanning arborescence

在同一全局 action graph 上，以重加权边选择 rooted maximum spanning arborescence：

\[
B^*=\arg\max_{B\in\mathcal A(G)}\sum_{e\in B}w(e).
\]

这使 backbone 同时保留高支持公共主干和有证据的区分性 transition，而不是让高频公共 edge 完全主导结构。

**实现状态。** `mine_backbone_workflow_discriminative(...)` 已是 `mine_backbone_workflow(...)` 的默认实现；历史 `mine_backbone_workflow_session_coverage(...)` 与 CLI 名称 `backbone_coverage` 现在是同一实现的兼容别名，不再执行独立 edge-swap。旧的 support/lift-only 与 coverage-swap 逻辑保留在代码中供历史产物解释，但不会进入当前默认实验。

### 3.4 严格术语：maximum spanning arborescence

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

### 3.5 保留 local branch / retry edges

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

历史 `backbone_coverage` 变体曾在不改变基础 edge score 的前提下，用有限轮次交换 parent edge 优化 session coverage。当前它已被统一为 `discriminative_backbone` 的兼容别名；如需对照，应显式恢复旧版本，而不能把同名 alias 当作新的消融结果。

## 4. Transition Induction 与 Reference

### 4.1 Transition-oriented evidence

对每种保留 edge，从训练 session 采样最多若干条实例。对 `source -> target`，主要 evidence 是两动作之间的原始 dialogue；只有在解释必要时才附加早期对话上下文。

transition induction 按 source action 分组，将同一 source 的所有 outgoing target **放入同一次 LLM 调用共同比较**。模型必须判断每条 edge 属于：

```text
distinguishable
ordered_fallback
underspecified
```

输出使用自然语言解释 user utterance、agent proposal、接受/拒绝、补充信息、未解决请求等可见交互，不强制将原因压成预定义 state variable。这样可避免相邻分支分别诱导后得到相同、歧义的 guard。

### 4.2 Reference 组织

`reference.md` 是 transition-oriented：

```text
source action                一级标签
source -> target transition  二级标签
sampled dialogue evidence    每条 transition 1 到 3 条
```

reference 是可检索证据，不是可以复制到技能中的私有 slot 值。运行时通过 `retrieve_reference` 的 MCP-style lookup 检索相关片段；query 由模型在 ReAct 中规划，工具调用及 observation 都写入 trace。

## 5. Organized Compiler

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

## 6. Progressive Disclosure 与运行时推理

编译后产物为：

```text
skill.md              compact backbone + routing/lookup policy
action_rules.md       详细 action procedure
slot_policies.md      有序 slot 采集、复用、缺失处理策略
reference.md          transition-specific dialogue evidence
subgraph.json         graph、backbone、local transitions、induction metadata
```

`skill.md` 明确告诉 agent：当 action procedure 或 slot value source 不确定时，通过对应 lookup 读取更详细的资源；不将所有细粒度内容无选择注入当前 prompt。系统 prompt 使用标签隔离：任务说明、skill、dialogue context、reference retrieval result 和 MCP 工具定义。

运行时直接加载该粗 flow 的单套 skill/reference。临时 cohort 不参与 runtime route selection，也不向 agent 暴露。

## 7. 评测与诊断

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
transition_cases.json
subgraph.json
```

真实 `original_subflow` 只可用于离线诊断，例如检验临时 cohort 的结构纯度；不得进入 cohort 构造、backbone 权重计算或测试时 prompt。

## 8. 当前限制

1. 临时 cohort 的数目、最小 support 与互斥阈值仍需在开发集上选择；cohort 错误会使区分性边权失真。
2. action graph 能突出结构差异，但不能单独恢复 action 序列近似、用户意图却不同的语义差异；transition induction 仍需要 dialogue evidence。
3. backbone 是编译与组织骨架，不能独立解决同一 source 下多个 branch guard 的语义冲突。
4. 当前离线 graph mining 不具备训练 batch 的自适应更新机制。

## 9. 后续：Structure-Preserving Online Refinement

在线阶段借鉴 AWM 的 batch rollout 形式，但不允许每个 batch 全文重写 skill。离线阶段产出的 \(\mathcal K_0=(S_0,R_0)\) 是稳定先验；在线阶段只依据 **train split** 的 rollout 与 gold AST/action/slot feedback 更新边的证据、可见性与自然语言 guard。test split 仅在最终冻结后评测，绝不进入更新。

```text
offline discriminative backbone + residual resource
  -> contrastive train-batch schedule
  -> rollout current frozen skill package on every action target
  -> localize gold feedback to source -> target edges
  -> deterministic topology/visibility patch proposal
  -> source-local sibling-edge guard induction, when required
  -> render updated skill/reference views and checkpoint
  -> final frozen test evaluation
```

### 9.1 Frozen structure 与可更新状态

首版中固定的部分是：

```text
action vocabulary
trajectory cohort construction
offline backbone parent edges
offline node inventory
```

允许在线维护的是 edge-level state：

```text
kind: backbone | candidate_branch | promoted_branch | retry | revisit
visibility: skill | reference
offline_support / offline_sessions
gold_support
rollout_success / rollout_failure
slot_failures
competing_targets
guard / guard_status
bounded dialogue and ReAct evidence
```

`candidate_branch` 是离线 tree 外的 forward edge；`retry` 和 `revisit` 表示 self-loop 或回到 backbone earlier action 的恢复逻辑，永远不加入主 DAG。`promoted_branch` 只有在满足支持度、可靠性和可区分 guard 后才能对运行时 `skill.md` 可见。

### 9.2 Contrastive rollout batch

不是随机将相邻 session 打成 batch。对每个 source action，调度器先收集其不同 target 的训练 session；来自不同 target 的代表 session 被放入同一 batch，以暴露 sibling transition 的混淆边界。每种 `source -> target` motif 只保留有限个 node+edge signature 不同的代表，避免大量几乎相同的轨迹耗尽 rollout 预算。低 reliability 的 edge 优先被调度。

这是**预算式采样**，而不是全训练集覆盖：未选 session 不会再被补齐进 rollout。默认目标采样率为训练 session 的 `30%`，由 `--target-selection-rate` 控制。在线过程仍然按 batch 执行 rollout、局部归因、guard 推断和资源更新，但一个 batch 是同一 source 的 sibling transition 比较单元，而不是普通的随机 mini-batch。调度器每一轮从该 source 的每个 target 选一条代表轨迹放入同一 batch；默认 `batch_size=8`，足以容纳通常的 2--3 个竞争分支及少量补充证据，同时让 skill 在更短反馈周期内更新。`per_transition_cap=3` 是每条转移的最低代表数，不再是全局采样上限；当 3 条不足以达到 30% 预算时，调度器会自适应增加每条转移可选的结构代表数，并在不同 source 的分支组之间轮转，防止预算被某一个局部冲突耗尽。若某个粗 flow 没有 source-local alternative，调度器只保留约 30% 的结构代表作为 bounded probe set。`rollout_schedule.json` 记录目标与实际采样率，日志也会打印两者。该调度不读取 `original_subflow`。

### 9.3 Gold feedback localization

每个 rollout 在所有 action target 上输出 `predicted_action` 和 ordered `predicted_slots`。对于 gold 相邻转移 \(e=(a_t\rightarrow a_{t+1})\)，更新：

```text
gold_support += 1
rollout_success += 1,  if both endpoint actions are correct
rollout_failure += 1,  otherwise
slot_failures += 1,    if the target ordered slots are incorrect
competing_targets[predicted_target] += 1,
                         if the target action is predicted as another action
```

这里的 edge 是 teacher-forced gold trajectory 中的 \(a_t\rightarrow a_{t+1}\)：source \(a_t\) 已由上下文给定，因此 `rollout_success` 只要求 target \(a_{t+1}\) action 正确，而不要求 source action 再次被模型预测正确。否则 edge reliability 会退化为相邻两个 action 正确率的乘积，并系统性阻止 branch 晋升。

同时为每条 edge 保存有限数量的可审计 evidence：conversation ID、source/target turn、截断 dialogue context、action/slot success、预测 target 与完整 ReAct trace。若训练中观察到离线图没有的 gold edge，则以 `candidate_branch` 加入状态，但默认只写入 reference。

边的平滑可靠性使用 Beta posterior mean：

\[
\operatorname{rel}(e)=
\frac{n_{\mathrm{success}}(e)+1}
{n_{\mathrm{success}}(e)+n_{\mathrm{failure}}(e)+2}.
\]

这避免少量 rollout 全对或全错时得到不稳定的 0/1 结论。

### 9.4 Deterministic patch policy

LLM 不决定图拓扑。每个 batch 后先由确定性策略提议三种 patch：

```text
promote_to_skill
sink_to_reference
induce_guard
```

默认门槛为：

```text
min_gold_support = 3
min_confidence = 0.60
min_conflict_count = 2
max_skill_branches_per_source = 3
```

一个 `candidate_branch` 只有同时满足以下条件才有资格晋升：它是 forward edge，即 target 不早于 source 的 backbone order；`gold_support >= 3`；`rel(e) >= 0.60`；且同一 source 下未超过 branch budget。资格满足但 guard 尚未解决时，patch 会先请求 `induce_guard`，该 edge 仍停留在 reference。只有 guard 被解析为 `resolved` 后，下一次确定性 patch 才将其设为 `visibility=skill`。

已经可见的非-backbone branch 若 reliability 低于阈值，或累计 competing target 达到 2，则执行 `sink_to_reference`。这正是“重要、可靠、可解释的边进入 DAG；可能误导的边下沉为按需证据”的操作化定义。

### 9.5 Local guard induction

只对低置信、高冲突，或已满足晋升证据但尚未可解释的 edge 调用 LLM。提示词始终同时包含：当前 `skill.md`、目标 edge 的 positive/negative rollout cases、同一 source 的所有 sibling edge、backbone order 与已有 guard。模型的唯一输出是：

```json
{"guard":"natural-language condition", "status":"resolved|uncertain", "rationale":"..."}
```

guard 必须以可观察 dialogue cue、用户目标、已确认步骤或前文差异区分 sibling target；不得虚构 hidden state、具体 slot value、新 action 或新 edge。无有效 JSON、空 guard 或无法区分 sibling 的结果都被记为 `uncertain`。每次局部 guard 调用最多重试 3 次；仍失败时不回退或重写全文 skill，只保留该 branch 在 reference 中，供后续 batch 累积更多证据。

### 9.6 Resource rendering 与恢复

每批 patch 后，从同一个 `skill_dag_state.json` 渲染两个增量视图：

```text
online_transition_guards.md  已晋升非-backbone edge 的 compact guard
online_reference.md          未晋升、低置信、retry/revisit 或 guard uncertain 的 evidence
```

它们分别附加到离线 `skill.md` 与 `reference.md`，而 `action_rules.md`、`slot_policies.md` 仍作为稳定的 progressive-disclosure 资源。runner 同时落盘：

```text
rollout_schedule.json        固定 batch -> conversation ID 映射
rollouts/batch_*.json        每批全 action-turn rollout
batch_diagnostics/*.json     localization events + deterministic patches
guard_induction/*.json       prompt, raw response, parsed guard
skill_dag_state.json         可恢复的图状态
online_refine_result.json    最终冻结测试指标
```

`--resume` 必须读取原始 `rollout_schedule.json`，而不是按更新后的 confidence 重排剩余 batch；因此中断恢复不会悄悄改变训练顺序或已经完成 batch 的含义。首版实现位于 `skill_mining/online_refinement.py` 与 `scripts/run_backbone_online_refine.py`。

## 10. 关键实现文件

```text
skill_mining/backbone_workflow_mining.py
skill_mining/skill_writer.py
scripts/run_subflow_eval.py
skill_mining/online_refinement.py
scripts/run_backbone_online_refine.py
eval_tod/abcd/agent.py
```
