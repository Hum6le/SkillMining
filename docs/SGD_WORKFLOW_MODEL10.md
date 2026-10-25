# 公司共享模型：串行方法、总在途 10、每 ID 2 QPS

2026-10-19。此方案取代之前三方法同时启动、每方法 160 线程的 workflow 方案。用户确认平台限制来自共享底层模型的总并发。

## 实现

- Workflow 模式启动一个后台总协调进程，按 AWM → Trace2Skill → Ours 顺序运行；一个方法退出后才启动下一个。某方法出现 case 错误会被记录，后续方法仍继续。`--status` 的 `_batch` 显示当前方法，各方法可显示 queued。
- 每方法最多 5 个 workflow ID。不同方法可保留原来各自的 ID，无需把三组 ID 手工替换成同一组。
- 同一 SQLite 库中的所有新客户端共享一个在途请求门控，最多 10 个；同时有在途请求的 workflow ID 最多 5 个。领取并发位和 QPS 配额在一个数据库事务中完成。
- 每 ID 配额 2 QPS，均匀发送间隔默认 0.55 秒；所有重试重新领取并发位和 QPS 配额，HTTP 返回或报错后释放并发位。限流响应继续触发共享冷却。
- 线程数默认/有效上限为 10；配置迁移把各方法设置为 case_threads=10、domain_workers=1。此限制作用于实际 HTTP 在途数量，不是仅限制外层 turn 数。
- 请求记录保存进程 PID 和创建时间。确认持有进程已退出或 PID 已复用后，回收遗留的并发位；不能仅按超时回收仍有活跃进程的请求。
- 原有每 ID 3 QPS 可以原库下调到 2，历史请求时间和累计用量不会删除。正在运行的旧版本不自动更新，必须先停止旧批次。
- 共享库中的更严格并发限制不会被新客户端放宽。此方案覆盖共享同一数据库的客户端，无法约束其他机器/其他应用或确认被杀进程已经取消服务端任务。

**10 并发与 10 QPS 不同。** 即使五 ID 的 QPS 配额合计为 10，若请求平均 9 秒，10 个在途请求对应的持续吞吐量级约为 1.1 请求/秒；实际吞吐由延迟、顺序更新阶段和均匀发送共同决定。没有公司端新设置下的完成时间实测。

训练数据、学习协议、slot policy 和评分逻辑不变。本次准备不调用公司 API。

## 公司端 Bash 操作

先把本次补丁解压覆盖到公司项目根目录。实际配置 `configs/sgd_workflows.json` 不在补丁中。

如果旧运行是之前的 v3，停止命令为：

```bash
python scripts/stop_sgd_server_batch.py --output-dir outputs/company_sgd_20pct_paced_v3 --execute
```

若当前使用其他输出目录，替换为实际旧目录。确认 status=stopped。随后：

```bash
python scripts/configure_sgd_workflow_limits.py --config configs/sgd_workflows.json && \
python scripts/launch_sgd_server_batch.py --provider workflow --splits-dir data/eval/sgd/protocol_20pct_seed42_seen_schema_v1 --output-dir outputs/company_sgd_20pct_model10_v4_preflight --dry-run && \
python scripts/launch_sgd_server_batch.py --provider workflow --splits-dir data/eval/sgd/protocol_20pct_seed42_seen_schema_v1 --output-dir outputs/company_sgd_20pct_model10_v4
```

迁移脚本先检查本项目是否仍有 SGD 进程或仍存活的请求持有进程，发现后报错；成功时备份配置、保留 ID 和数据库路径、清理已停止批次配置内 ID 的旧版无 owner 占位，再写入新限制。新运行目录从头训练，旧结果保留，不声明任意训练阶段可断点续跑。

启动器回到终端后，后台总协调进程继续工作；Windows 隐藏窗口，Linux 独立 session。

## 监控

```bash
python scripts/launch_sgd_server_batch.py --output-dir outputs/company_sgd_20pct_model10_v4 --status
python scripts/workflow_pool_status.py --config configs/sgd_workflows.json --window 60
```

重点看 `_global_model.current_inflight <= 10`、`max_active_workflows=5`、每 ID 的 `peak_admissions_in_rolling_second <= 2`。`peak_inflight` 是当前共享门控建立以来的历史峰值，不是仅当前时间窗口。

停止新一轮：

```bash
python scripts/stop_sgd_server_batch.py --output-dir outputs/company_sgd_20pct_model10_v4 --execute
```

## 本地验证

- 36 项针对性测试通过，覆盖跨线程/跨进程共享在途门控、不同 ID 池共享活跃 ID 上限、进程退出/PID 复用回收、3→2 QPS 原库下调、配置备份迁移、方法串行与错误后继续，以及 HTTP 重试、停止脚本和原 SGD batch 回归。
- 40 个并发模拟任务、40 个线程、每次模拟耗时 1.2 秒：实际在途峰值 10，每 ID 任意滚动秒准入峰值均为 2，结束时在途为 0。真实 API 调用 0。
- 20% 三方法 dry-run：sequential，默认 case workers=10，三方法各 9 个 family，全部计划使用 2 QPS/ID、共享 max_inflight=10。
- 公司端部署、告警消除和实际完成时间尚未验证。
