# 补充 baseline：建议 ACE

调研日期：2026-10-19。结论为选型建议；尚未实现 ACE 的 ABCD / SGD adapter，没有启动额外付费实验，也未修改公司正在运行的任务。

## 1. 建议

新增 **ACE — Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models**，Zhang et al.，ICLR 2026。论文提出 Generator / Reflector / Curator 分工，累积并增量维护条目化 playbook，支持 offline 与 online adaptation。[论文](https://arxiv.org/abs/2510.04618)

官方实现已通过论文 v3 的复现声明及作者项目页交叉确认，使用 [ace-agent/ace](https://github.com/ace-agent/ace)，不要误用同名第三方复现。[论文复现声明](https://arxiv.org/html/2510.04618v3)；[作者项目页](https://ace-agent.github.io/)

代码检查基于 commit `82709de050e1db6e6ef2f07bcb0393560b94992a`，GitHub API 返回提交日期 2026-08-24。正式适配时应固定该版本或明确记录更新后的 SHA。

**选它的判断：** 当前已有反思记忆、经验规则、workflow 记忆、技能文档和技能编译；ACE 补充可持续维护的条目式 playbook。这能检验我们在资源组织与执行成本上的收益。它只是一个有价值的对照，单独增加 ACE 不能证明图结构贡献；结构贡献仍需 matched-runtime 消融。

## 2. 候选比较

以下“适配成本”是根据公开代码和本地接口作出的工程判断，不是已测耗时。

| 候选 | 机制及公开实现 | 对本项目的价值 | 适配判断 |
|---|---|---|---|
| **ACE，首选** | 条目 playbook、反思、增量维护；官方有 ACEBatch | 与我们的资源组织/细节保留问题直接相关，可复用共享预测器和评分器 | 中等；需改 Generator 任务接口、LLM bridge 和模块命名空间 |
| AutoManual，备选 | Planner / Builder / Formulator 构建环境手册；NeurIPS 2024 官方仓库 | 可比较规则手册与图技能 | 较高；其环境交互和代码式规划需映射到离线 teacher-forced ToD |
| Skill-Pro，后续考虑 | 技能 initiation / policy / termination，non-parametric PPO gate 验证与演化；ICML 2026 官方仓库 | 较新的技能演化对照 | 较高；其验证/选择机制及交互环境需要额外适配，不适合当前抢时间的第一新增项 |

来源：[AutoManual 官方代码](https://github.com/minghchen/automanual)、[Skill-Pro 官方代码](https://github.com/Miracle1207/Skill-Pro)、[ACE 官方代码](https://github.com/ace-agent/ace)。没有把网页上的原数据集成绩当作 ABCD / SGD 上的预期效果。

## 3. 已有方法与当前入口

| 方法 | 本地证据 | 本次状态 |
|---|---|---|
| 无技能 ReAct | `scripts/run_react_abcd.py` | ABCD 已有入口，保留作基准 |
| ExpeL | `scripts/run_expel_abcd.py`、`expel_adapter/abcd.py` | ABCD 已实现；本次未验证 SGD 适配 |
| Reflexion | `scripts/run_reflexion_abcd.py`、`reflexion_adapter/abcd.py` | ABCD 已实现；本次未验证 SGD 适配 |
| Skill-Disco | `scripts/run_skill_disco_abcd_full.py`、`skill_disco/` | ABCD 已有完整入口；本次未验证 SGD 适配 |
| AWM | `scripts/run_sgd_domain_eval.py --method awm` | 当前 SGD 批次已纳入 |
| Trace2Skill | `scripts/run_sgd_trace2skill_pilot.py` | 当前 SGD 批次已纳入 |
| **ACE** | 官方代码已读，本地尚无 adapter | 建议新增 |
| Ours | `eval_tod/sgd_aligned.py`、standard refine | 当前 SGD 批次已纳入 |

“已有实现”不等于“已在 SGD 对齐并跑通”。当前 SGD 批量入口主要支持 AWM / Trace2Skill / Ours，不能仅改 `--method` 名称就启动前三个新增适配。

## 4. 官方代码可复用位置

已阅读以下文件的类、调用接口和导入关系：

- `ace/core/reflector.py`：`Reflector.reflect`，提取反馈与条目使用信息。
- `ace/core/curator.py`：`Curator.curate`、`propose_operations`、`aggregate_operations`。
- `playbook_utils.py`：`apply_curator_operations`、`update_bullet_counts` 等条目维护函数。
- `ace/ace_batch.py`：`ACEBatch`，批内使用 playbook snapshot 做并行生成/反思，并提供 curator 聚合路径。
- `ace/ace.py`：原逐样本训练流程，用来核对批量版的语义。
- 官方树中还有 Mind2Web 适配目录，但本次任务继续聚焦 ABCD / SGD，无需重启 Mind2Web 实验。

上述是代码可复用性检查，尚未在本机安装/运行上游环境。[固定版本 ACEBatch](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/ace/ace_batch.py)

**批量版的额外核对项：** 该 commit 的 `ACEBatch` 类说明明确包含 ComBEE-style curator reducer，且 `augmented_shuffling` 默认开启、factor 默认 2。这超出了简单的执行并发。主 baseline 应优先保持 `ace/ace.py` 的原始更新语义；若使用批量版，要单独标为批量变体、显式核对 reducer，并关闭额外采样扩增或将重复次数纳入预算。不能将其默认配置直接称为原版 ACE 的等价加速。[源码](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/ace/ace_batch.py)

实际接入需处理：官方模块从顶层 `llm`、`utils`、`logger` 导入，可能与我们项目的同名模块冲突。应使用有命名空间的 bridge 或明确修改导入，不在多线程运行时临时替换全局模块。

## 5. 最小公平适配方案

这是后续实现方案，不是已完成的复现。

1. **执行器共享**：ABCD 使用现有 action→slot 执行器；SGD 使用 `eval_tod.sgd_aligned.predict_policy`。ACE 负责学习/提供 playbook。结果标注为共享执行器下的 ACE adaptation，保留与上游 Generator 的差异。
2. **同一训练范围**：与其他方法使用同一 TRAIN 对话 ID、全部相同的目标 turn 和模型。SGD 沿用 seed42、整段对话 10%、同 schema、全部 family-owned policy turn。ACE 初始资源不注入 Ours 专有的图/card。
3. **反馈边界**：Reflector 在 TRAIN 可以看到 gold/prediction 的 action 与 slot 差异、合法 schema 和可见上下文。运行时 Generator 不见当前/未来 gold。validation / test 的标签不写回 playbook。
4. **资源维护复用**：保留上游 Reflector、Curator、条目 ID、helpful/harmful 更新和 delta 应用机制。不要简化成只拼接几条错误总结后称为 ACE。
5. **冻结评测**：选预先规定的最终 TRAIN checkpoint，在当前 validation 上做一次冻结评测。不要边评测边读取其标签更新，也不要用同一 validation 选最佳 checkpoint 再报作独立评测。
6. **并行**：原版 ACE 顺序更新可跨独立 subflow / family 并行，冻结评测也可并行。若考虑官方 `ACEBatch`，先核对上一节的 reducer 和采样扩增差异；使用 batch>1 时报告为 batch adaptation，不能声称等同论文原始 batch=1。不同批次消费前一次更新，Curator 写入按确定顺序提交。同 family 的持续更新不能无约束并发。
7. **接口**：角色调用都通过本项目 `llm.chat` / 公司 workflow 请求池，计入每 ID 3 QPS。使用相同 LLM，不能给 Reflector / Curator 配更强模型。
8. **预算**：原论文允许多 epoch 和多轮 reflection；smoke 可用一轮，但正式主表需要明确采用原设置还是 budget-matched 设置。原样记录训练 turn、重复 pass、角色调用数、token、失败与重试数。不能把一次精简 smoke 写成完整原论文复现。

ACE 的官方 offline 路径与本文正在使用的 TRAIN-only 适配、冻结评测更接近。原论文 online 包含按序预测后更新评测样本的设置，不宜直接套用到我们的主表。[官方使用说明](https://github.com/ace-agent/ace/blob/main/README.md)

## 6. 明天执行顺序

1. 先收集公司现有 AWM / Trace2Skill / Ours 批次的完成状态、分母、失败数和调用统计。
2. 复用已有 ABCD 基线入口，核对 ExpeL / Reflexion / Skill-Disco 的实际运行协议及资源边界。若目标是 SGD，补任务 adapter 后再启动。
3. 固定 ACE 上游版本，先实现 TRAIN-only playbook adapter 和 workflow LLM bridge。
4. 本地用 mock 检查 gold 可见性、delta 应用、冻结资源哈希、usage，再做少量真实样本 smoke。
5. smoke 通过后再排 ACE 的同范围训练/评测，单独分配 workflow 配额或等当前任务结束。不要把新任务静默叠加到已经满载的 15 个 ID 上。

今晚交付到选型和接口审计；没有启动这些新增 baseline，也没有改变正在运行的 SGD setting。
