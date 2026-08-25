# 迭代共享节点删除分组及其对 MST Backbone 的作用

本文只说明仓库中旧的 grouping 实现：

```text
skill_mining/semantic_subflow.py
discover_semantic_subflows(...)
protocol = iterative_shared_node_splitting_v1
```

它的目的不是直接删除 workflow 中的 action，而是暂时删除跨 session 高度共享的 action，观察剩余轨迹模式能否形成更清晰、且有足够支持度的 session group。被删除的节点只用于发现分界；一旦得到 group，下游仍使用成员 session 的**完整原始 action trajectory** 建图和编译。

本文分为两部分：

```text
Part I   节点删除 grouping 如何从 session 中发现结构性 cohort
Part II  grouping 输出如何重加权同一张全局图，并帮助选择 MST backbone
```

> 状态说明：当前 discriminative backbone 不使用本算法作为 cohort 生成器；它使用 `weighted_motif_prototypes_v1`。本文记录的是共享节点删除分组的独立算法及其历史行为。

## Part I. 迭代共享节点删除 Grouping

## 1. 输入与输出

输入为同一个粗粒度 flow 的训练 session：

\[
\mathcal D=\{\tau_1,\ldots,\tau_N\}.
\]

每条 session 有一个 `convo_id` 和 `delexed` action turn。算法不读取 `original_subflow`；后者最多只能用于离线事后诊断。

主要参数：

```text
max_skills = 4       最多保留多少个 group
min_sessions = 20    每个候选 group 最少包含多少条 session
```

输出包括：

```text
skills                    每个保留 group 的 motif、成员统计、node/edge 统计
session_assignments       session -> skill_id 的硬分配及 coverage/margin
removed_partition_nodes   被累计删除、用于显露分界的共享 action
shared_interface_nodes    被删除或在 group 间共享的 action
shared_interface_edges    在 group 间共享的 transition
objective_history         每轮尝试删除的节点及 objective 是否提升
final_metrics             最终 group 的 support / cohesion / overlap / objective
```

## 2. Session Signature

首先将每条 dialogue 转为 action 序列：

\[
\tau_i=(a_1,a_2,\ldots,a_T).
\]

action 名称通过 ABCD action schema 规范化。随后构造 set-valued signature：

\[
\phi(\tau_i)=
\{\texttt{node:}a_t\}_{t=1}^{T}
\cup
\{a_t\texttt{=>}a_{t+1}\}_{t=1}^{T-1}.
\]

例如：

```text
node:pull-up-account
node:enter-details
node:make-password
pull-up-account=>enter-details
enter-details=>make-password
```

这是**集合**表示：同一 node 或同一 edge 在一个 session 中重复出现只记一次。保留 node feature 的原因是，若只保存 edge，删除一个共享中间节点后，一些 session 可能失去全部剩余 edge 证据；node presence 可以留下剩余 action motif。

## 3. 累计删除共享节点

算法维护一个累计删除集合 \(U\)，初始为：

\[
U=\varnothing.
\]

每一轮先计算所有尚未删除 action 的 session support：

\[
\operatorname{supp}(a)=
\left|\{\tau_i:a\text{ 出现在 }\phi(\tau_i)\}\right|.
\]

候选 action 按 support 从高到低检查。对一个候选节点 \(x\)，试探删除集合 \(U'=U\cup\{x\}\)。此时，每条 session 的 residual signature 为：

\[
\phi_{U'}(\tau_i)=
\{f\in\phi(\tau_i):x\notin\operatorname{endpoints}(f),\ \forall x\in U'\}.
\]

含义是：

- 删除 `node:x`；
- 删除所有以 `x` 为 source 或 target 的 transition；
- 其他 node 和 edge feature 保留。

算法不是盲目删除当前最高频节点。它会依次试探候选节点，选择**第一个能够产生至少两个满足支持度要求的 residual group**的高支持节点。若当前最高频节点删除后无法形成有效 group，会继续测试下一个节点。

## 4. 从 Residual Feature 构造候选 Group

给定一个试探删除集合 \(U'\)，将所有 session residual signature 汇总。每一个 residual feature（node 或 transition）都可以成为候选 group 的 seed。

### 4.1 Seed 筛选

对一个 feature \(f\)，先收集包含它的 session：

\[
M_f=\{\tau_i:f\in\phi_{U'}(\tau_i)\}.
\]

仅保留：

\[
|M_f|\geq \texttt{min\_sessions}.
\]

这一步排除了只由极少数异常轨迹支持的 seed。

### 4.2 从 Seed 扩展为 Residual Motif

对 seed 成员集合 \(M_f\)，统计 residual feature 的共现次数。group motif 定义为：

\[
F_f=
\left\{g:
\operatorname{count}_{\tau\in M_f}[g\in\phi_{U'}(\tau)]
\geq \max(2,\lfloor0.2|M_f|\rfloor)
\right\}\cup\{f\}.
\]

也就是说，一个 feature 只要在 seed 成员的至少约 20% 中反复出现，就进入这个 group 的结构模式；seed 本身无条件保留。

候选 motif 去重采用 Jaccard overlap：

\[
J(F_p,F_q)=\frac{|F_p\cap F_q|}{|F_p\cup F_q|}.
\]

若一个新 motif 与已有候选的 overlap 大于 `0.92`，它被视为近重复候选并丢弃。候选数最多为 `max_skills * 3`，为后续分配保留冗余。

## 5. Session 的硬分配

每条 residual signature \(\phi_{U'}(\tau_i)\) 与所有候选 motif 比较覆盖率：

\[
\operatorname{cov}(\tau_i,F_j)=
\frac{|\phi_{U'}(\tau_i)\cap F_j|}
{|\phi_{U'}(\tau_i)|}.
\]

session 被硬分配给覆盖率最大的 group：

\[
z_i=\arg\max_j\operatorname{cov}(\tau_i,F_j).
\]

若最大覆盖率为 0，则该 session 不被分配。分配记录还保存：

```text
coverage              最大覆盖率
margin                最大覆盖率 - 第二大覆盖率
candidate_coverages   对所有候选 group 的覆盖率
```

其中 `margin` 很重要：它越小，说明该 session 对两个残余模式同样匹配，边界越不清晰。

之后，只保留硬分配成员数至少为 `min_sessions` 的 group，按成员数降序保留最多 `max_skills` 个。若剩余 group 少于两个，则本次删除节点的尝试无效。

## 6. Objective 与接受规则

对每个保留 group \(G_k\)，定义：

\[
\operatorname{support}_k=\frac{|G_k|}{N}.
\]

它的 motif 与其他保留 motif 的平均 overlap 为：

\[
\operatorname{overlap}_k=
\frac{1}{K-1}\sum_{j\ne k}J(F_k,F_j).
\]

该 group 的 objective contribution 为：

\[
q_k=\frac{\operatorname{support}_k}{1+\operatorname{overlap}_k}.
\]

总体 objective：

\[
J(U')=\sum_{k=1}^{K}q_k.
\]

这个 objective 偏好两类性质：

- group 有足够多的 session support；
- group 的 residual motif 与其他 group 不要过于相同。

算法采用贪心累计删除：若

\[
J(U')>J(U)+10^{-6},
\]

则接受删除节点 \(x\)，令 \(U\leftarrow U'\)，并继续下一轮；否则立即停止，并返回上一轮已接受的最优分组。它不是穷举搜索，也不保证全局最优。

第一次有效切分的比较基线是 0。因此只要能形成有效 group 且 objective 为正，第一轮通常可以被接受。

## 7. Fallback 行为

若没有任何节点删除可以形成至少两个 supported group，算法不强行制造分组，而是返回一个 fallback group：

```text
所有带有非空 signature 的 session
-> 一个 group
-> coverage = 1.0, margin = 1.0
```

这种情况说明：在当前 action node/edge 表示和 `min_sessions` 下，没有足够证据支持结构化细分。

## 8. 完整伪代码

```text
records <- {(session, node+edge signature)}
U <- empty set
best <- none

repeat:
    candidates <- remaining action nodes sorted by session support
    proposal <- none

    for x in candidates:
        U_try <- U union {x}
        residual signatures <- remove x and all incident edges
        motif candidates <- supported residual node/edge seeds
        hard-assign sessions by residual motif coverage
        retain groups with at least min_sessions sessions

        if at least two groups remain:
            proposal <- this partition
            break

    if proposal does not exist:
        stop

    if proposal.objective <= best.objective + 1e-6:
        stop

    U <- U_try
    best <- proposal

if best is none:
    return one all-session fallback group
else:
    return best groups, assignments, U, and objective history
```

## 9. 直观例子

设三类 session 都经过共享账户步骤：

```text
pull-up-account -> verify-identity -> enter-details
```

之后分别出现：

```text
enter-details -> send-link
enter-details -> make-password
enter-details -> reset-2fa
```

在不删除共享节点时，`pull-up-account`、`verify-identity` 和 `enter-details` 的 support 很高，所有 session signature 高度相似。试探删除这些共享节点及 incident transition 后，残余 motif 更接近：

```text
Group 1: node:send-link + enter-details=>send-link
Group 2: node:make-password + enter-details=>make-password
Group 3: node:reset-2fa + enter-details=>reset-2fa
```

它们对不同 session 的覆盖率高、相互 Jaccard overlap 低，因此 objective 上升。注意：真正下游编译每个 group 时，原来被删除的共享前缀仍会重新回到完整 session trajectory 中；它没有从最终 workflow 消失。

## Part II. Grouping 如何帮助 MST Backbone

节点删除 grouping 不是 MST 的前处理垃圾桶，更不是把 group 外的 action 删掉后再分别建树。它提供的是一个 session-level 对比信号：**哪些边是所有轨迹共享的公共结构，哪些边只在某一类相似轨迹中稳定出现。** 这个信号可以直接进入全局 MST 的 edge weight。

### 10. 正确的连接方式：分组用于打分，完整图用于建树

令 grouping 输出的硬分配为：

\[
z_i\in\{1,\ldots,K\},\qquad
G_k=\{\tau_i:z_i=k\}.
\]

重要的是：`removed_partition_nodes` 和 residual signature **只服务于得到** \(G_k\)。进入 MST 时，恢复每条成员 session 的完整 action sequence，包括被暂时删除的共享前缀和与其相连的 edge。然后在所有训练 session 的完整 transition graph \(\mathcal G\) 上只求**一棵** rooted maximum spanning arborescence。

因此：

```text
节点删除阶段：暂时隐藏共享接口，显露 group 边界
MST 阶段：恢复完整图，保留共享 trunk，也保留 group-specific 分歧
```

不能把 residual graph 直接交给 MST。那样会真的丢掉 `pull-up-account -> verify-identity` 这一类公共但必要的结构。

### 11. 从 Group Membership 计算 Edge 区分性

对全局候选 edge \(e=(u\rightarrow v)\)，在每个 group 内外分别统计**包含该 edge 的不同 session 数**：

\[
n_k(e)=|\{\tau_i\in G_k:e\in\tau_i\}|,
\qquad
n_{\neg k}(e)=|\{\tau_i\notin G_k:e\in\tau_i\}|.
\]

相应的 session-level occurrence rate 为：

\[
p_k(e)=\frac{n_k(e)+\epsilon}{|G_k|+2\epsilon},
\qquad
p_{\neg k}(e)=\frac{n_{\neg k}(e)+\epsilon}{|\mathcal D\setminus G_k|+2\epsilon},
\]

其中 \(\epsilon=1\) 是平滑项。edge 对 group \(k\) 的区分性定义为：

\[
d_k(e)=\log\frac{p_k(e)}{p_{\neg k}(e)}.
\]

一条 edge 在任何一个 group 中显著富集即可得到奖励：

\[
d^+(e)=\max_k\max(d_k(e),0).
\]

这里使用 `max` 而非 group-average：一条边即使只服务于一个高支持 cohort，也可能是连接 action DAG 的关键分歧；把它平均掉会重新让公共边压过它。

### 12. Discriminative MST Edge Weight

原始 backbone score 衡量跨 session support 与局部转移关联：

\[
w_{\mathrm{base}}(e)=
\log(1+|\mathcal S_e|)
+0.5\log\bigl(\max(\operatorname{lift}(e),10^{-9})\bigr).
\]

利用 grouping 后，最终 MST 权重变为：

\[
w_{\mathrm{disc}}(e)=
w_{\mathrm{base}}(e)
+\lambda_{\mathrm{disc}}\min(d^+(e),c).
\]

当前默认超参数：

```text
lambda_discriminative = 1.0
clip c = 3.0
```

因此 cohort bonus 最大为 `3.0`；它足以影响 base score 相近的 parent-edge 竞争，但仍受 clip 约束，不会让极少数轨迹支持的 edge 无限压制全局结构。

第一项仍保留高频、稳定、全局通用的 edge；第三项只给在某个 cohort 内明显富集的 edge 加分，**不会惩罚共享 trunk**。

例如：

```text
shared trunk:
pull-up-account -> verify-identity
  在 Group 1/2/3 都经常出现
  -> base support 高，d+(e) 接近 0
  -> 仍会被 MST 保留

discriminative suffix:
enter-details -> send-link
  在 Group 1 常出现，Group 2/3 很少出现
  -> base support 可能中等，但 d+(e) 高
  -> 更可能作为 enter-details 的 MST child 被选中
```

然后在完整有向图加虚拟根 `<START>` 后求：

\[
B^*=\arg\max_{B\in\mathcal A(\mathcal G)}
\sum_{e\in B}w_{\mathrm{disc}}(e),
\]

其中 \(\mathcal A(\mathcal G)\) 是覆盖所有 observed action 的 rooted arborescence 集合。MST 的连通性约束保证每个 action 仍被组织到同一张 backbone 中；grouping 只改变“哪条候选父边更值得被选”。

### 13. Grouping 输出如何进入 Artifact

为保证这个过程可审计，每条全局 edge 应保存：

```text
base_weight                 support + lift 的原始分数
best_cohort_id              使 d_k(e) 最大的 group
support_in_cohort           n_k(e)
support_outside_cohort      n_not_k(e)
discriminative_log_odds     d+(e)
final_backbone_weight       w_disc(e)
```

`subgraph.json` 还应保存 cohort 的 session support、分组协议和 \(\lambda_{\mathrm{disc}}\)、clip。这样可以回答某条 backbone edge 被选中的原因：是因为它是全局高支持 trunk，还是因为它是某类轨迹的高区分转移。

### 14. 与 Residual Resource \(R\) 的关系

MST 仍只为每个 action 选择一个 parent edge。即使使用 grouping 加权，也不应该把所有 cohort-specific edge 都强塞进 tree：

```text
selected parent edge                 -> backbone B
tree 外但有支持的 forward edge        -> branch candidate in R
self-loop / return-to-earlier action -> retry/revisit in R
```

后续 transition induction 会把同一 source 的 tree edge 与 residual siblings 放到同一 LLM 调用中，推断它们的自然语言 guard。online refinement 再依据 rollout evidence 决定某个 residual forward edge 是否足够可靠、且 guard 可区分，从 `R` 晋升为 DAG 的 verified forward branch \(E^+\)。

所以完整关系是：

```text
shared-node deletion grouping
    -> session cohort assignments
    -> discriminative edge score on the restored global graph
    -> one MST backbone B
    -> remaining supported edges enter R
    -> LLM guard induction and online promotion determine E+
```

### 15. 当前代码状态

上述“grouping -> MST”的数学接口可以直接使用本文件前半部分的 \(G_k\)。但是当前 `discriminative_backbone` 实现并未调用 `discover_semantic_subflows()`；它调用的是 `discover_motif_prototypes()`，同样产生临时 cohort，但不通过节点删除。

因此需区分：

```text
本文 Part I:
  iterative_shared_node_splitting_v1 的精确实现说明

当前 backbone:
  weighted_motif_prototypes_v1 -> cohort assignment -> d+(e) -> MST

可选后续消融:
  将 Part I 的 group assignment 接到相同的 d+(e) 重加权接口，
  比较 node-removal cohort 与 motif-prototype cohort 对 backbone/AST 的影响
```

## 16. 节点删除 Grouping 本身的局限

1. 删除顺序是贪心的。第一个能形成有效切分的高支持节点不一定导向全局最优分组。
2. feature 是 action node 和 action transition，不包含 customer utterance 的语义；动作结构相同但用户目标不同的 session 难以区分。
3. 硬分配会把边界模糊的 session 强行归入一个 group；虽然 `margin` 被记录，但当前 objective 不直接惩罚低 margin。
4. `support` 以 session 是否出现某个 feature 计数，不使用 edge 出现次数，也不使用 slot 值、action 参数或完整 dialogue context。
5. 删除节点用于发现 residual motif，天然偏好“共享前缀 + 分叉后缀”的情形；复杂的交错、重入或多次意图切换轨迹不一定适合该假设。
6. 当前实现中的 `similarity_threshold` 参数未参与实际计算；候选 motif 的去重阈值在代码中固定为 Jaccard `0.92`，motif 共现阈值固定为 `max(2, floor(0.2 * seed_members))`。
