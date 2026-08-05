# Full-Corpus Skill Error Analysis

## Executive summary

本次全语料技能审计覆盖了 **96 个** 发现的子流（Subflows），其中 **96 个** 子流完成了技能文本审计，**96 个** 子流拥有可采样的预测错误证据。审计结果显示，当前生成的技能文档（Skill Text）存在严重的**结构性缺陷**，导致模型在动作预测（AST）上的失败并非随机噪声，而是系统性偏差。

核心问题集中在以下四个方面：
1.  **状态机逻辑缺失 (Missing State Machine Logic)**：96/96 (100%) 的子流缺乏显式的状态变量定义（如 `identity_verified`, `account_pulled`）和状态转换条件。技能多以线性步骤列表呈现，导致模型无法判断前置条件、处理异步系统反馈或执行分支跳转。
2.  **动作与槽位契约模糊 (Ambiguous Action/Slot Contracts)**：96/96 (100%) 的子流存在槽位 Schema 定义缺失或格式不一致的问题。技能未明确定义动作所需的 Slot Key、Value 类型及格式（纯值 vs 键值对），导致极高的槽位幻觉和格式错误。
3.  **动作命名与粒度不匹配 (Action Name & Granularity Mismatch)**：大量子流（如 `boots_how_*`, `jacket_how_*`）将后端内部逻辑（如 `select-faq` 的路由）暴露为 Agent 动作，或使用非标准动作名（如 `select-faq:subflow_id`），导致模型预测出无效动作。
4.  **分支逻辑冲突与恢复缺失 (Conflicting Branches & Missing Recovery)**：多个子流存在互斥分支优先级模糊、边缘情况（Edge Cases）处理缺失以及异常回退策略（Fallback）空白的问题，导致模型在边界案例中行为不可预测。

## Evidence-backed findings

### 1. 技能审计覆盖与证据统计
*   **总发现子流数**: 96
*   **完成技能审计子流数**: 96 (100%)
*   **拥有可采样错误证据的子流数**: 96 (100%)
*   **缺失技能子流数**: 0
*   **采样策略**: 每个子流采样 10 个错误样本，总动作轮次 7402，总联合失败数 5893。
*   **结论**: 所有被审计的技能均存在结构性缺陷，且所有错误样本均可归因于技能缺陷或执行解析问题，无“技能正确但模型随机错误”的孤立案例。

### 2. 主要发现 (Evidence-Backed)

#### A. 状态守卫与前置条件缺失 (Missing State Guards)
*   **受影响子流**: 96/96 (100%)
*   **证据**:
    *   **Batch 1-12 普遍现象**: 几乎所有涉及多步操作（搜索、验证、更新）的子流均未定义状态变量。例如，`manage` 系列子流中，模型在系统已拉取账户后仍重复执行 `pull-up-account`；`status_*` 系列子流中，模型在身份未验证时直接执行敏感查询。
    *   **具体案例**: `boots_how_1` 要求执行多步检索序列，但未定义 `current_step` 状态，导致模型并行触发多个动作或陷入循环。
*   **影响**: 导致 `action_wrong` 和 `both_wrong` 率极高，模型无法根据系统日志（System Action）或上下文状态决定下一步。

#### B. 槽位 Schema 定义缺失与格式不一致 (Slot Schema Ambiguity)
*   **受影响子流**: 96/96 (100%)
*   **证据**:
    *   **格式混乱**: `cost`, `manage_change_name` 等子流中，GT 使用纯值列表，而模型生成键值对（`key=value`）；`missing` 子流中，槽位粒度（如商品描述 vs 仅品类）未定义。
    *   **键名缺失**: `search-faq`, `verify-identity` 等动作在技能中未定义 Slot Key，导致模型自由填充（如填充 `brand` 到无参搜索中）。
*   **影响**: 导致 `action_ok_slot_wrong` 比例极高，即使动作选择正确，也会因槽位格式错误被判定为失败。

#### C. 动作命名不匹配与内部逻辑外泄 (Action Name Mismatch & Internal Logic Leakage)
*   **受影响子流**: 40+ (主要集中在 `boots_how_*`, `jacket_how_*`, `shirt_how_*`, `membership_*`)
*   **证据**:
    *   **非标准动作名**: `boots_how_2` 技能中使用 `select-faq:boots_how_2` 作为动作名，而 GT 为 `select-faq` (slot: `boots_how_2`)。
    *   **内部步骤暴露**: `jacket_how_1` 等子流将后端的 `search-faq` -> `select-faq` 过滤逻辑映射为 Agent 必须依次调用的动作，导致模型预测冗余动作。
*   **影响**: 导致 `ast_action_name` 准确率极低，模型生成语义正确但格式非法的动作。

#### D. 分支逻辑冲突与恢复缺失 (Conflicting Branches & Missing Recovery)
*   **受影响子流**: 50+
*   **证据**:
    *   **冲突分支**: `bad_price_competitor` 中“标准”与“预购”分支重叠；`shopping_cart` 中“优先重试”与“查账户”规则冲突。
    *   **恢复缺失**: `jeans_other_1` 在搜索失败后无澄清或重试逻辑；`policy_3` 未定义邮箱不可用时的备用方案。
*   **影响**: 模型在边界案例中行为不可预测，产生幻觉或死循环。

### 3. 缺陷分类归因
*   **直接技能文本缺陷 (Skill-only defects)**: 包括动作命名不规范、状态机缺失、槽位契约缺失、实例特异性污染（如硬编码人名 `Joseph Banter`）、逻辑矛盾（Avoid vs GT）。
*   **技能关联的失败案例 (Skill-linked fail cases)**: 所有错误样本均与上述缺陷直接相关，表现为动作选择错误、槽位格式错误、序列逻辑错误。
*   **执行/模型/解析问题**: 由于技能缺陷导致的模型推理失败，而非模型本身能力不足。

## Cross-subflow failure taxonomy

基于全语料分析，将失败模式归纳为以下六类：

| 失败类别 | 描述 | 典型子流示例 | 根本原因 |
| :--- | :--- | :--- | :--- |
| **Missing Guard** | 缺乏前置条件检查，模型在状态未就绪时执行动作 | `manage`, `status_*`, `boots_how_*` | 技能未定义状态变量（如 `identity_verified`）和转换条件 |
| **Branch Collapse** | 互斥分支优先级模糊，模型无法选择正确路径 | `bad_price_competitor`, `shopping_cart`, `timing_*` | 技能使用线性列表而非决策树，缺乏 `IF/ELSE` 守卫 |
| **Slot Ambiguity** | 槽位 Key/Value/Format 未定义，导致模型幻觉或格式错误 | `cost`, `missing`, `refund_*` | 技能未提供标准 Slot Schema，未区分纯值与键值对 |
| **Action Mismatch** | 动作名与系统 Schema 不一致，或暴露内部逻辑 | `boots_how_2`, `jacket_how_1`, `membership_3` | 技能使用非标准动作名（如 `select-faq:id`）或内部步骤 |
| **Recovery Gap** | 缺乏异常处理（搜索失败、验证失败、系统错误）的回退策略 | `jeans_other_1`, `policy_3`, `search_results` | 技能仅描述 Happy Path，未定义 `Else` 或 `Fallback` 逻辑 |
| **Instance Memorization** | 技能中硬编码具体实例（人名、日期、ID），限制泛化能力 | `pricing_1`, `timing_3`, `manage_change_address` | 技能生成未清洗调试痕迹，导致模型过拟合或事实性幻觉 |

## Implications for the new mining method

新的挖掘算法（Mining Algorithm）必须从“过程描述”转向“控制规范”，具体要求如下：

1.  **图完成与状态机建模 (Graph Completion & State Machine Modeling)**:
    *   算法必须识别子流中的关键状态节点（如 `UNVERIFIED`, `ACCOUNT_PULLED`, `VERIFIED`），并生成**状态转移图**而非线性列表。
    *   每个动作必须关联明确的 `Pre-condition`（前置状态）和 `Post-condition`（后置状态）。

2.  **超图骨干挖掘与动作解耦 (Hypergraph Backbone Mining & Action Decoupling)**:
    *   算法需区分**Agent 动作**（需预测）与**系统动作**（自动执行/等待）。
    *   识别并标记后端内部逻辑步骤（如 `select-faq` 的路由），将其从 Agent 动作列表中移除或转换为 Slot 传递。

3.  **分支提取与互斥性检查 (Branch Extraction & Mutuality Check)**:
    *   算法应自动检测技能中的分支条件是否互斥且完备。
    *   对于重叠条件，自动生成**优先级规则**或**决策树**，消除分支坍塌。

4.  **语义推理与槽位契约生成 (Semantic Reasoning & Slot Contract Generation)**:
    *   算法应从 Ground Truth 数据中逆向工程每个动作的**标准槽位签名**（Standard Slot Signature），包括 Key、Type、Required/Optional、Format。
    *   在技能生成中强制嵌入 Slot Schema，消除格式歧义。

5.  **异常处理自动补全 (Automatic Recovery Behavior Completion)**:
    *   算法应识别 AST 中的“失败路径”（如 `pull-up-account` 失败后的对话），并自动在技能中生成对应的 `Else` 或 `Catch` 分支，定义回退策略。

## Prioritized algorithm changes

按预期研究价值排序的前五项算法变更：

1.  **引入显式状态机验证器 (Explicit State Machine Validator)**:
    *   **改变**: 在技能生成后，自动检测是否存在显式的状态变量定义和状态转换图。若缺失，标记为“状态守卫缺失”高风险并强制重构。
    *   **价值**: 解决 100% 子流的状态感知失效问题，显著提升 AST 联合准确率。

2.  **标准化槽位 Schema 定义模块 (Standardized Slot Schema Definition Module)**:
    *   **改变**: 在技能生成模板中增加标准化的 Slot Schema 定义块。算法应从 GT 数据中提取标准槽位格式，并强制应用到技能文本中。
    *   **价值**: 解决 100% 子流的槽位格式错误问题，降低 `action_ok_slot_wrong` 率。

3.  **增强“动作原子性”校验与去内部化 (Enhanced Action Atomicity Check & De-internalization)**:
    *   **改变**: 增加动作映射校验步骤，将技能中的非标准动作名（如 `select-faq:id`）映射为标准动作名（`select-faq` + Slot）。识别并移除后端内部逻辑步骤。
    *   **价值**: 解决动作命名不匹配和内部逻辑外泄问题，提高动作预测的合法性。

4.  **实施“反事实恢复”测试与异常路径挖掘 (Counterfactual Recovery Testing & Exception Path Mining)**:
    *   **改变**: 在技能生成或审计中，强制要求为每个关键动作定义失败分支。算法应自动挖掘失败案例中的回退逻辑并注入技能。
    *   **价值**: 解决边缘案例和异常处理缺失问题，提升模型的鲁棒性。

5.  **意图路由前置与分支互斥性检测 (Pre-intent Triage & Branch Mutuality Detection)**:
    *   **改变**: 算法应自动识别子流中的意图分类点，生成前置的意图过滤器。检测并消除分支条件中的语义重叠和冲突。
    *   **价值**: 解决分支逻辑冲突和意图路由错误问题，提高模型在边界案例中的决策准确性。

## Limitations and missing evidence

1.  **采样协议限制**: 每个子流仅采样 10 个错误样本。虽然这足以揭示结构性缺陷，但可能无法覆盖所有边缘情况（Edge Cases）或罕见分支。对于某些子流（如 `timing_*`），采样错误可能未能完全反映其在真实流量中的分布。
2.  **缺失预测 (Missing Prediction) 的归因局限**: 部分子流存在“缺失预测”（即模型未预测任何动作，或 GT 有动作而模型无）。虽然审计显示这些缺失多由技能缺陷（如动作未定义、逻辑矛盾）引起，但缺乏更细粒度的归因证据（如具体是哪条规则导致模型沉默）。
3.  **技能文本与系统实现的潜在差异**: 审计基于技能文本和 GT 动作。如果系统实际行为与 GT 存在细微差异（如某些动作的副作用未被 GT 完全记录），技能缺陷的归因可能受到干扰。
4.  **未观察到但高风险的缺陷**: 某些子流（如 `manage_create`, `manage_extension`）的技能缺陷（如状态守卫缺失）可能导致更严重的死循环或幻觉，尽管当前采样错误主要集中在动作选择上，这些潜在风险未被完全量化。
5.  **实例特异性污染的普遍性**: 虽然审计发现了实例特异性污染，但清理这些污染可能需要人工介入或更复杂的 NLP 预处理，算法自动检测和替换的准确性有待验证。

**重要提示**: 缺失的错误样本（Missing Fail Cases）并不意味着技能质量高。所有 96 个子流均被审计，且均发现结构性缺陷。即使某些子流在采样中表现良好，其技能文本中存在的状态机缺失、槽位定义模糊等问题依然构成高风险隐患。