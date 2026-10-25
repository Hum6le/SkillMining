# 本地 1/10 SGD：ReAct、Reflexion、ExpeL、Skill-DisCo

## 2026-10-20 调度与 Skill-DisCo 修复

### ExpeL 完成后自动增加 Reflexion 并发

2026-10-20 16:57 后：Reflexion 已于16:33完成全部1,241个验证目标。按用户指示，将其释放的8路加到新版 ExpeL 原4路，ExpeL 现使用12线程；复用已完成批次并重跑切换时未保存的批次。`expel_scale12.json` 保存切换记录。仅本地 shubiaobiao ExpeL 上限放宽至12，公司 workflow 仍限制最多10线程/10个在途HTTP请求。

- `scripts/scale_sgd_reflexion_after_expel.py --output-dir <batch>` 等待 ExpeL 全量1,241个目标完成并退出，然后停止该批次 controller 和旧 Reflexion，使用 `--reflexion-workers 8` 恢复。
- 切换前两方法各4线程，切换后仅 Reflexion 8线程。数据、反馈、资源和评分协议保持原样；整段内的 turn/反思依赖保持顺序。已保存的 turn 复用，尚未保存的在途请求可能重复。
- `reflexion_scale_up.json` 保存 waiting/scaled/cancelled/failed 状态；原批次被暂停、停止或替换时，等待脚本取消操作。
- 数据 manifest 保留首次启动的 workers 值和哈希；实际恢复的 workers 写入 `execution_history.jsonl` 和 `progress.json`。仅允许并发数变化，数据、模型、max_trials 等差异仍拒绝恢复。

- 当前使用三个独立任务：Reflexion、Skill-DisCo、ExpeL，各4线程，总请求并发上限12。
- controller 使用 `--parallel-methods --adopt-running`，接管原有 Reflexion 和提前启动的 ExpeL；ReAct 完整结果复用。
- Skill-DisCo runtime 更新为 `compiled_schema_scoped_prefix_heads_shared_payload_binder_v2`。编译 skill 必须仅引用当前任务可用的服务版本，投影 heads 在进入共享 binder 前再次校验；不按 domain 名猜测服务版本映射。
- 根因是 Events_2 的编译 primitive 被用于 Events_1，跳过了普通 head selector 的服务合法性校验，造成37次 CALL payload 校验失败。修复保持共享执行器和评分器不变。
- 全部离线资源、挖掘断点复用；旧评测、summary、manifest、aggregate 保存到 `skill_disco/repair_archive/runtime_v1_20261020/`，全部验证目标按 v2 重跑，避免混用两个 runtime。
- 27项回归测试通过；原报错样本 `sgd-validation-1004:13` 已用同一中转站 API smoke 跑通。用量日志保留，最终费用统计包含修复前尝试和补跑。

## 固定数据和执行协议

- 数据：`data/eval/sgd/protocol_10pct_seed42_seen_schema_v1`。
- 与之前公司 10% 结果对应的 9 个 family：Buses、Events、Homes、Hotels、Music、RentalCars、RideSharing、Travel、Weather。
- TRAIN：8,269 个 family-owned policy turn；validation：1,241 个；call 与 non-call 均计入。
- 保留官方 TRAIN/validation 边界、seed=42、按整段对话采样和同 schema 过滤。跨 family 的目标 turn 不重复。
- 全部使用现有 `sgd_aligned` action-head -> payload 执行器及原始严格评分器。
- 当前/未来回复、gold acts、belief state 均不进入 actor；历史真实 API 记录属于共享的 logged prefix。
- 当前 API 结果只有在预测 CALL 的 service/method/parameters 全部匹配时解锁。
- 学习与验证冻结分开；固定使用最终 TRAIN 资源，不在该 validation 上选 checkpoint 或更新资源。
- 输入数据、ontology、模型、协议和资源哈希写入 manifest；smoke 有独立目录，禁止当作完整实验结果。

## ABCD 实现复用及 SGD 适配

### ReAct

无挖掘、无更新、无资源。使用共同的动作选择和 payload 生成流程，不注入 Ours 的图、action card 或 reference controller。

### Reflexion

复用 `Reflection` / `ReflectionStore`、最多 2 次训练对话尝试、同一训练对话内反思、最多 3 条原始反思的设定。反馈改为 SGD 全部 action+named payload 的严格 joint correctness。每轮最多列出 4 个错误供 reflector 分析。评测只检索冻结的 TRAIN 反思，检索 query 为当前可见前缀，不看未来用户输入。

每个训练 trial/turn 保存断点；失败请求不作为 slot 错误学习。

### ExpeL

**2026-10-20 更新：**下面描述的是旧 v1 的 outcome-only 适配。审计确认它与本地 ABCD 实际保留 gold action/slot 的归纳输入不一致。新一轮采用 TRAIN predicted/gold policy 配对监督，规则仍冻结后评测；完整说明见 `SGD_EXPEL_ABCD_ALIGNMENT_20261020.md`。旧结果保留，新结果由 `method_run_overrides.json` 指向独立输出目录。

复用 `ExpeLRuleStore` 与 ADD/AGREE/EDIT/REMOVE 操作、20 段对话一批、最多 20 条规则。批内 actor 使用同一规则快照；其后再进行一轮联合规则归纳。提供 episode SUCCESS/FAILURE 及预测轨迹，不额外给规则归纳器当前 gold payload 或逐 turn 正确标签。训练请求失败的 episode 不作为行为证据。

### Skill-DisCo

复用现有的 semantic abstraction、operation extraction、grouping/consolidation、skill specification、synthesis 与独立进程 replay verification；验证取 TRAIN 内按完整对话确定划出的 20%，不读取 validation。

SGD IR 的一个 primitive 对应一个 policy turn。三个参数串行化 API method、named API parameters、unordered dialogue acts；这是 IR 传输顺序，不是给 SGD slot/act 施加顺序。包含 slot/value 字段是否存在的区别。

运行时保留已验证 Python 技能调用；在自己先前预测构成的前缀上执行，到第一个未知 policy 时暂停。得到的 heads 接共享 payload binder；未匹配到技能时走共享基础执行器。不会为了匹配技能而读取验证对话的真实历史 acts。

与 ABCD 原适配相比，SGD 采用 `compiled_predicted_prefix_head_selection_shared_payload_binder`：程序投影负责 heads，具体 payload 统一由共享 binder 从当前可见上下文绑定。这是明确的 SGD adaptation，不宣称与原生 WebArena 宏动作执行协议相同。

## API 与并发

- 读取现有 `third_party/agent-workflow-memory/mind2web/.env`，不打印密钥。
- `https://api.shubiaobiao.cn/v1`，`qwen3.6-flash`，沿用 thinking=false。
- 两条串行队列：`ReAct -> Reflexion`，`Skill-DisCo -> ExpeL`。
- 每个方法最多 4 个 case 线程，总计最多 8 个同步 API 请求。每个 family 独立资源，family 顺序处理。
- 每个线程独立 usage context；按请求日志重建用量，计入中断请求和重试，不把缺失用量算作零。
- 线程复用 HTTP 连接；无需 GPU。后台启动和代码重放子进程均隐藏窗口。

## 启动与进度

```powershell
python scripts/launch_sgd_extra_baselines.py --output-dir outputs/local_sgd_10pct_extra_20261019 --preflight
python scripts/launch_sgd_extra_baselines.py --output-dir outputs/local_sgd_10pct_extra_20261019
python scripts/monitor_sgd_extra_baselines.py --output-dir outputs/local_sgd_10pct_extra_20261019
```

后台用 `Start-Process -WindowStyle Hidden` 启动 controller。每方法日志为 `<method>.log`；每 family 的 `progress.json`、`predictions.jsonl`、`summary.json`、`llm_usage.json` 和训练资源保留在方法目录。

## 自动汇总

controller 在方法完成后和两条队列结束时生成 `comparison.json` / `comparison.md`，并列之前用户提供的 AWM、Trace2Skill、Ours 10% aggregate。

历史三方法的原始预测、模型元数据和用量未随 aggregate 提供；汇总中明确标为历史报告值，不假称本地重跑、模型身份已核对或用量为零。尚未完成或 smoke 结果不会作为完整结果展示。
