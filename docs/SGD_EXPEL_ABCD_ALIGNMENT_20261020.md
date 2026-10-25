# ExpeL SGD 与本地 ABCD 学习输入对齐

## 诊断

旧 SGD 版归纳器仅收到预测轨迹与整段 SUCCESS/FAILURE。1,743段训练对话中，排除11段执行错误后，1,732段没有任何完整成功 episode。旧验证 Joint 为18/1,241。

本地 ABCD 的 `ExpeLABCDAgent.induce_rules` 会保留 `_build_abcd_turn_trajectory` 的 `gold_action`、`gold_slots` 和 reference；仅删除逐 turn 的正确性标志。因此旧 SGD 版学习监督与本地 ABCD 不一致。

## 本轮协议

- 学习版本：`abcd_aligned_train_gold_policy_pairs_frozen_validation_v2`。
- 仅离线归纳器读取 TRAIN 的 predicted/gold policy 配对、reference response、可见前缀和 episode SUCCESS/FAILURE；不加入逐 turn 正确性标志。
- Gold 是完整 SGD policy，含无序 acts、可选 CALL 及命名参数；不把用户的语言行为当作系统标签。
- Actor 和验证阶段继续使用共同的 `sgd_aligned` 执行器；当前及未来 gold/response 不进入 actor。共享 API replay、prior logged prefix、序列化统计和严格评分不变。
- 20段训练对话一批、最多20条规则、每批最多4条 ADD/AGREE/EDIT/REMOVE。保持整段严格成功定义；失败对话中的正确片段通过 TRAIN gold 配对明确可见，不伪造成功 episode。
- 这是与项目 ABCD 版对齐的有监督 ExpeL 适配，不能描述为原始 ExpeL 的 outcome-only 反馈复现。

## 共用规则更新修复

`snapshot_numbered_operations_clean_replacements_v2` 同时用于 ABCD 与新 SGD：

- 所有编号指向该批调用开始时展示的规则列表；处理顺序为 REMOVE、AGREE、EDIT、ADD。
- AGREE/REMOVE 根据明确编号定位，不要求返回文本必须与规则完全重合。
- 去掉 `rule:`、`replacement:`、`Replace rule N with:` 等控制前缀。仅在左侧精确匹配旧规则时解析 old-to-new 包装。
- 相同规则去重，不通过重复文本叠加强度；保留合法过程指令和方法名内联反引号。
- SGD 归纳器对不存在的编号/空操作格式最多修复两次，失败明确报错，不悄悄吞掉更新。
- schema 与系统标签角色说明来自 TRAIN/公开 ontology；不根据验证错误人工编写领域规则。

## 运行和记录

- 新目录：`outputs/local_sgd_10pct_expel_abcd_aligned_20261020/expel`。
- 不载入旧 ExpeL 的规则、训练缓存或预测。旧结果完整保留。
- 与旧批次使用相同 seed=42 采样、9个 family、8,269 TRAIN policy turn 和1,241验证目标。
- ExpeL 4路；原 Reflexion 8路继续，合计12路。只替换后台 controller 以跟踪新目录，并接管原 Reflexion PID。
- 原批次 `method_run_overrides.json` 指向新 ExpeL。汇总不把新实验的阶段分数当作完整结果，旧分数保存在 `previous_run`。
- 新 ExpeL 学习资源及 manifest 保存版本，拒绝恢复旧监督协议的目录。
