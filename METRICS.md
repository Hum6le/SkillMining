# 实验指标说明

本文档说明当前项目的 ABCD 主实验指标和全量汇总方式。

## 1. ABCD 评测对象

ABCD turn 分为：

- `customer`：用户输入，不作为 agent 输出目标。
- `utterance`：agent 的自然语言回复。
- `action`：agent 的结构化动作，包含 action name 和 slot values。

当前实验按 subflow 独立训练和评估，再对各 subflow 结果做全局加权汇总。

## 2. AST：Action State Tracking

AST 只评估 `action` turn。

### Action Name Accuracy

```text
正确预测 action name 的 action turn 数 / action turn 总数
```

action name 必须完全匹配标注。

### Slot Value Accuracy

```text
slot values 完全正确的 action turn 数 / action turn 总数
```

slot values 使用精确、顺序敏感的匹配。数量、内容或顺序任一不同都会判错。

### Joint AST

```text
action name 正确且所有 slot values 正确的 action turn 数
/ action turn 总数
```

Joint AST 是当前结构化预测的主指标。action name 正确但 slot 错误时，不能计入 AST。

```text
AST <= Action Name Accuracy
AST <= Slot Value Accuracy
```

没有 action turn 的单对话结果返回 `1.0`，表示没有需要检查的 action；但全局 aggregate
在 action turn 总数为 0 时返回 `0.0`。因此必须同时检查 `num_action_turns`。

## 3. CDS：Cascading Dialogue Success

CDS 评估模型能否从某个位置开始，连续正确完成后续对话。

对于起始位置 `i`：

```text
score_i
= 首次错误之前连续正确的 actionable turns 数
  / 从 i 开始剩余的 actionable turns 数
```

`actionable turns` 包括 agent 的 `utterance` 和 `action` turns；`customer` turns 会被跳过，
不会增加分母，也不会直接打断 cascade。一旦出现第一个错误，该起始位置的 cascade 就停止。

```text
dialogue_cds = 所有起始位置 score_i 的平均值
overall_cds   = 所有测试对话 dialogue_cds 的平均值
```

CDS 范围为 `[0, 1]`，越高越好。它是严格的序列指标，通常会低于逐 turn 的 AST 或文本
指标，因为一次早期错误会影响多个起始位置的后续 cascade。

## 4. Agent Utterance 文本指标

文本指标只在 agent 的 `utterance` turn 上计算，不把 action turn 当作自然语言回复。预测
文本与对应的 ABCD reference utterance 逐 turn 对齐。

当前重点报告三个指标：`BLEU-1`、`ROUGE-L` 和 `METEOR`。它们都越高越好，但关注点不同：

| 指标 | 核心关注 | 对什么敏感 | 适合回答的问题 |
|---|---|---|---|
| BLEU-1 | 单词精确重合和预测 precision | 回复是否用了 reference 中的词 | 是否使用了相同的关键表达 |
| ROUGE-L | 最长公共子序列的内容与顺序 | 信息覆盖和整体顺序 | 是否覆盖了 reference 的主要内容 |
| METEOR | token precision/recall 与连续性 | 内容匹配及词序碎片化 | 内容是否匹配且表达是否连贯 |

### 4.1 BLEU-1

BLEU-1 只使用 unigram，也就是单个 token 的重合，不检查 2-gram 或更长短语。当前实现
返回百分制，通常范围为 `[0, 100]`。

直观上，它衡量预测回复中的词有多少也出现在 reference 中，核心偏向 **precision**：

```text
BLEU-1 近似关注：预测文本中的词，有多少是 reference 中出现过的词
```

因此它对关键词和词面表达比较敏感，但对词序不敏感。例如把同一组词重新排列，BLEU-1
可能仍然较高；加入很多常见但无关的词，也可能因为 unigram overlap 而得到一定分数。
它不会判断完整语义，也不擅长区分“词都对但回复组织不合理”的情况。

在本项目中，BLEU-1 主要反映 agent 是否使用了与人工 reference 相同的词汇，尤其适合
观察固定业务措辞、确认语句和关键实体词的词面重合。

### 4.2 ROUGE-L

ROUGE-L 基于 prediction 与 reference 的最长公共子序列（LCS, Longest Common Subsequence）。
LCS 不要求 token 连续，但保留 token 的相对顺序；当前结果使用 ROUGE-L F1，通常范围为
`[0, 1]`。

它综合 prediction-to-reference 的 precision 和 reference-to-prediction 的 recall：

```text
ROUGE-L F1 = precision 与 recall 的调和平均
```

因此 ROUGE-L 比 BLEU-1 更关注 **内容覆盖 + 顺序结构**。如果预测包含 reference 的主要
信息，并且大致按相同顺序组织，即使中间插入少量词，ROUGE-L 仍可能较高。相反，预测只
包含少量关键词时，BLEU-1 可能不低，但由于覆盖不足，ROUGE-L recall 会受到影响。

ROUGE-L 仍然是词面匹配指标，不理解真正的同义改写。例如语义完全正确但使用了不同词汇
的回复，ROUGE-L 可能偏低。因此它更适合衡量回复是否覆盖了 reference 的主要信息和
组织顺序，而不是单独判断语义正确性。

### 4.3 METEOR

当前实现使用 exact-token METEOR-style 分数，综合 token precision、token recall 和
fragmentation penalty，通常范围为 `[0, 1]`。

它同时关注两件事：

1. 预测文本是否覆盖 reference 中的重要 token；
2. 匹配 token 是否按照较少碎片、较自然的顺序出现。

其中 fragmentation penalty 会惩罚匹配 token 被打散成很多不连续片段的情况。因此，METEOR
相比 BLEU-1 不只看“用了哪些词”，相比 ROUGE-L 也更显式地综合 precision、recall 和
匹配连续性。

本项目的实现是 exact-token 版本，不能把同义词或自然语言改写自动视为匹配。这一点很
重要：METEOR 在这里衡量的是词面内容匹配和局部连续性，不是独立的语义等价判断。

### 4.4 三个指标的区别

可以用下面的方式理解：

```text
BLEU-1  -> 预测用了多少相同的词，偏 precision，几乎不看顺序
ROUGE-L -> reference 的主要内容覆盖了多少，并且顺序是否相近
METEOR  -> 词面 precision/recall 是否都较好，匹配是否较连续
```

典型情况：

- 回复很短，只复用了几个关键词：BLEU-1 可能尚可，但 ROUGE-L 和 METEOR 通常较低；
- 回复覆盖了 reference 的大部分内容，但加入了额外说明：ROUGE-L 可能较高，BLEU-1
  取决于额外词的比例；
- 回复词汇和 reference 不同但语义合理：三个指标都可能偏低，因为当前实现主要是
  token-level matching；
- 回复使用了相同词汇但顺序混乱：BLEU-1 可能仍较高，而 ROUGE-L 和 METEOR 会更明显下降。

所以这三个指标不应简单互相替代：BLEU-1 看词面命中，ROUGE-L 看有序内容覆盖，METEOR
看 precision/recall 与匹配连续性的平衡。

## 5. 全量结果汇总

全量实验不是简单平均每个 subflow，而是按实际评测量加权：

| 指标类别 | 汇总权重 |
|---|---:|
| BLEU-1、ROUGE-L、METEOR | agent utterance/text samples 数 |
| Action Name、Slot Value、AST | action turns 数 |
| CDS | test sessions 数 |

全局结果写入 `summary.json` 的 `__global__.aggregate`，并保存：

- `metrics`：加权指标值；
- `weights.text_samples`；
- `weights.action_turns`；
- `weights.test_sessions`。

比较结果时应同时查看指标和有效样本量，尤其检查 `num_action_turns` 是否为 0。

## 6. 推荐报告格式

```text
AST | Action Name | Slot Value | CDS | BLEU-1 | ROUGE-L | METEOR
```

建议先检查样本量，再分别从结构化预测、序列级执行和自然语言表达三个层面解释结果。
文本指标高不代表 action/slot 正确，AST 高也不代表自然语言回复质量高。

## 7. 代码位置

- AST/CDS：`eval_tod/abcd/metrics.py`
- 文本指标：`eval_tod/text_eval.py`
- ABCD subflow 评估：`scripts/run_subflow_eval.py`
- 全量结果汇总：`scripts/aggregate_subflow_results.py`
