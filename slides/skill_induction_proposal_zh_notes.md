# 中文汇报备注

## 1. 标题
我们关注的不是如何生成更长的 Skill，而是如何从轨迹中学习可复用、可分支、可验证的程序性知识。

## 2. 总体动机
AWM 和 Trace2Skill 的生成结果暴露出两个互补问题：一类 Skill 记住了具体轨迹，另一类 Skill 有动作约束但缺少状态条件。两个方向将在后面分别展开。

## 3-5. Macro Skill
第一条方向是泛化性。第 3 页直接展示 `outputs/skills/recover_password/skill.md` 的原文：动作序列中出现 `aphoenix1`、`cm374950`、`alessandro phoenix`、`57820` 和具体电话。这些是训练实例，不应该直接成为 Skill 的规则，而应该被提升为变量、slot schema 和参数化动作。后续仍需要用未见实体测试泛化能力。

## 6-7. State-conditioned Skill
第 6 页直接展示 `recover_username/skill.md` 中 Main Path 的原文。它已经描述了 pull-up、verify 和 recovery，但“请求两个凭证”没有把当前已收集字段、字段可用性和系统返回结果写成可判定 guard，执行边界仍需要模型推断。不同状态下的规则可能都正确，冲突来自状态条件没有被结构化。用 State、Guard、Action、Transition 表示后，规则可以自然地分成不同分支。

## 8-9. Reference-use Policy
两个方向都需要 reference 使用能力。Skill 不仅要决定业务动作，还要决定何时查 reference、查哪一类证据，以及如何把证据和当前对话进行 grounding。

## 10. 实验计划
实验需要围绕三类 bad case：未见实体的 Macro 泛化、相同 intent 下不同状态的分支选择、reference 与当前对话冲突时的 grounding。问题页已经使用真实生成 Skill 原文；后续仍需要把原文问题与具体失败轨迹、预测和指标对齐。

## 11. 贡献
贡献不是更长的 prompt，而是一个能够学习参数化流程、状态分支和 reference-use policy 的 Skill 表示与学习框架。
