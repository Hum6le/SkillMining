# 公司 workflow 版 SGD 请求并行

**最新方案见 [SGD_WORKFLOW_MODEL10.md](SGD_WORKFLOW_MODEL10.md)：公司共享模型总在途上限 10，三方法串行，每方法最多 5 个 ID、每 ID 2 QPS。下文的三方法同时启动与 160 线程建议已被取代。**

2026-10-19。当前状态：已根据用户提供的公司 HTTP 协议完成实际 HTTP attempt 级接入。用户在公司 20% 运行中报告触发 3 QPS 告警；本地已将限流改为下述 v2 均匀调度与共享冷却，但没有公司端部署或消除告警的实测证据。数据及训练/评测协议沿用对应的 SGD 采样方案。

## 2026-10-19 限流修复 v2

旧版只保证本机共享数据库记录的滚动秒准入不超过 3，允许同一 ID 连发 3 次，边界余量仅 5ms。该保证不能直接扩展到服务端接收时刻：本地零网络模拟中，两批各 3 次的准入间隔 1.005 秒，第一批比第二批多 30ms 延迟，便可在接收窗口内出现 6 次。这是一个可复现的设计缺口，不是对公司这次告警根因的确认。

v2 保留原滚动窗口限制，同时增加：

- 同一 ID 默认间隔至少 `1.08 / qps` 秒；3 QPS 配额对应 0.36 秒，不能积攒空闲配额后突发。五 ID 理论均匀放行上限约 13.89 QPS/方法。
- HTTP429 或已识别的 ret=1004 失败令该 ID 在同一个 SQLite 库中的所有新客户端共享冷却，连续限流时从 2 秒增长至最多 30 秒。旧在途请求的成功响应不能清掉该冷却。
- 数据库保留原请求记录并自动补充诊断字段；无需删库。所有相关进程必须使用同一本地数据库及 v2 代码。旧进程不会通过 git pull 或修改配置自动更新。
- 配置可增加 `"min_interval_seconds": 0.4`，把每 ID 目标速率降低至约 2.5 QPS，五 ID 约 12.5 QPS/方法；下界为 `1.08 / qps`。共享库保留已配置的更大间隔，监控进程不会把它放宽。不要直接改 qps_per_workflow 为 2 后删除原库来绕过配置检查。
- 新状态字段有 `client_min_interval_seconds`、`minimum_admission_gap_seconds`、`paced_qps_ceiling`、`rate_limit_responses_in_window` 和 `cooldown_remaining_seconds`。限流响应计数只覆盖 v2 客户端记录，旧记录缺少标记，不应解读为旧运行没有告警。

均匀调度和冷却能减少突发风险，仍不能保证远端永不拒绝：网络/代理延迟、其他调用方、未共享限速库的其他机器、账号/模型总配额等都不在此单机限速器的控制范围。公司端限流的统计维度及当前进程加载的代码仍需核对。

该修复不改变 SGD 训练/评分协议。正在运行的任务需先核对内层训练/评测阶段和恢复产物，再安排进程重启；普通 --resume 只跳过已有 summary 的场景，不能保证任意训练中途都能无损续上。

## 为什么 5 个 ID 需要约 160 个 case 线程

原实现把一个 worker 固定绑定到一个 workflow ID，且 worker 数必须等于 ID 数。每个 worker 同时只有一个同步请求。若一次请求约 9 秒，5 worker 只有约 0.56 QPS。

新模式把 workflow 配额和 case 并发数解耦。5 个 ID × 3 QPS = 每方法 15 QPS 的准入上限；需要的在途请求约为 **目标 QPS × 请求平均延迟**，即 15×9≈135。因此配置起点为每 family 160 个轻量 case 线程。这里的 9 秒来自家用中转站的历史执行链折算，含本地开销；公司 workflow 的实际请求延迟需要重测。若请求 20 秒，160 个线程最多约 8 QPS，需按内存及接口能力重新调整并发。

不是 160 个 Python 子进程。rollout / diagnosis / evaluation 用线程执行独立 case；每个 case 的日志、native usage、SGD usage、payload heads、诊断目录分别隔离。同一 turn 的 reference→head selection→payload 执行顺序保留。冻结 rollout、训练样本范围、4 snippet 初始化和 sequential standard reflection 保留。

## 请求调度

- `llm_new.py` 的 workflow transport 入口逐请求从该方法的 5 个 ID 中选择可用 ID。不会逐线程改写共享环境变量。
- 同一 ID 的最近请求时间存于共享 SQLite WAL 数据库，跨线程/进程共享；保留滚动秒配额并均匀调度，v2 默认间隔至少 0.36 秒。这里限制的是客户端准入。
- 同一台机器所有相关实验应使用同一个绝对数据库路径，放本地磁盘，不能为每个进程另建限速器，也不要放 NFS。数据库只记录 ID、标签、时间和状态，不存 prompt / response / key。
- 成功、失败及外层 retry / JSON repair 都消耗准入配额；不会因请求失败退还已用配额。
- 用户提供的 wrapper 自带 `(0,2,5,10,20,30)` 限流重试，还可能委托另一份带重试的 workspace helper。并发模式现在使用 `workflow_http_transport.py`，按相同 URL、POST body、Server-App-Id、user_id 和参数映射直接发送单次 HTTP；没有 workspace helper 回退，也没有自动 HTTP redirect。
- `llm_new.chat` 对 ret=1004 / HTTP 429 使用上述退避序列，每次重试重新从 ID 池领取配额。每次实际 attempt 分别记成功/失败计数，避免一个 logical chat 含多次 HTTP 却只计一次。既有 SGD 外层重试也重新领取配额。非零业务 ret 不会当模型文本评分。
- 未启用 ID 池时，原 workspace helper 的调用方式保留；它的内部重试不属于本方案的限速保证范围。
- 5 个 ID 必须配置相同的模型、prompt 参数映射及无跨请求对话状态。workflow 的实际模型由公司端配置决定；CLI `--model` 不会替换 workflow 内部模型。

## 配置和启动

同步最新 HTTP attempt 版 workflow 专用包中的代码到公司项目，保留公司原有 `config.py`。包内的 `llm.py` 已由更新后的 `llm_new.py` 生成，本机 OpenAI 客户端文件没有被替换。启用 workflow 池时不再依赖公司 `utils/llm_utils.py`，原文件可供其他入口继续使用。保留原文件备份后再覆盖。

```bash
python -m pip install -r requirements-sgd-server.txt
cp configs/sgd_workflows.json.example configs/sgd_workflows.json
```

编辑 JSON，将 AWM / Trace2Skill / Ours 的各 5 个 ID 填好。无需新增中转站 API key。公司 `config.py` 的 `LLM_CONFIG.provider` 保持 `workflow`。

```bash
# 生成计划，不调用接口
python scripts/launch_sgd_server_batch.py \
  --provider workflow --workflow-pools configs/sgd_workflows.json \
  --output-dir outputs/sgd_company_preflight --dry-run

# 少量真实样本 smoke 通过后启动第一批
python scripts/launch_sgd_server_batch.py \
  --provider workflow --workflow-pools configs/sgd_workflows.json \
  --output-dir outputs/sgd_company_10pct_v1

# 按 ID 查看最近 60 秒准入、完成、失败、未结束数量、平均 transport 耗时
python scripts/workflow_pool_status.py --config configs/sgd_workflows.json --window 60

# family 进度
python scripts/launch_sgd_server_batch.py \
  --output-dir outputs/sgd_company_10pct_v1 --status
```

`admitted_qps` 是观测窗口平均值，不是账号限额；短窗口包含初始 3 次突发，平均值可能稍高于 3。`peak_admissions_in_rolling_second` 是当前窗口内任意滚动秒的请求准入数，应 ≤3。完成吞吐还要观察失败率和耗时，准入满载不等于有效完成满载。进程异常终止的历史 inflight 项只是“未记录完成”，不一定仍在运行。

每方法默认一次训练一个 family。需要交叠独立 family 的串行阶段时，可在该方法配置增加 `"domain_workers": 2`。各 family 保持独立资源及顺序 reflection，共享相同 ID 池和限速数据库；最多同时出现 2×160 个 case 线程。先测内存和错误率再提高；此选项不改变抽样或反馈顺序。

## 可以打满哪些阶段

大量独立 turn 的 rollout / evaluation 最有希望接近 15 QPS。单个 family 的 reflection 是有依赖的顺序更新；每次必须消费之前的 skill，不能为吞吐改变这个协议。AWM / Trace2Skill 的小批次、MAP/REDUCE 层级切换和最终尾部也会出现空闲。并行两个独立 family 可交叠部分空闲，但不保证整段训练持续满载。

三方法用互不重叠的 15 个 ID 时，理论合计 45 QPS；仍需确认公司接口没有额外的账户总配额、并发在途上限或 token 吞吐瓶颈。

## 本地证据与时间估算

- `outputs/workflow_pool_mock_20261019_v2/report.json`：90 次模拟 transport 调用，5 个 ID，每个 ID 限 3 QPS，32 线程、模拟延迟 0.5 秒；排除初始突发后约 **14.4 QPS**。所有 ID 的滚动 1 秒准入均未超过 3。真实 API 调用 0。
- 当前测试覆盖跨进程同 ID 限速、多个线程的独立精确 usage、输出顺序、错误保留、payload ContextVar 隔离、请求级 ID 分配。运行门槛不等于公司端吞吐实测。
- HTTP attempt 接入测试另外验证：POST 字段与公司代码一致；ret=1004 / HTTP429 重试重新取配额；失败 attempt 不漏计、不重复计；重试耗尽仍失败；未知业务错误正常抛出；并发模式不会进入旧 workspace helper；自动重定向关闭。
- 第一批 Ours 共 9,510 个 TRAIN+validation turn，按约 8.2 次请求/turn 粗算约 7.8 万次请求。若独立 turn 部分持续 15 QPS，其请求量对应约 **1.45 小时**；编译、诊断、reflection、层级同步、尾部及重试另外计时。不能把 1.45 小时当整个实验完成时间。
- 公司机先做低并发真实 smoke，然后逐步增加到约 160 case 线程，监控每 ID 近 60 秒吞吐、延迟、429、空响应、失败比例以及 RSS。用真实请求分布更新 ETA。

无需公司接口即可重现调度模拟：

```bash
python scripts/benchmark_workflow_pool.py \
  --output-dir outputs/workflow_mock_new --requests 90 --workers 32 --latency-seconds 0.5
```

打包命令：

```bash
python scripts/package_sgd_server.py --llm-backend workflow \
  --output artifacts/sgd_company_workflow_http_20261019.tar.gz
```
