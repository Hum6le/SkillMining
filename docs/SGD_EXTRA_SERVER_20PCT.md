# 服务器补跑 1/5 SGD 的四个基线

## 方法与执行协议

- 补跑 ReAct、Skill-DisCo、ExpeL、Reflexion；顺序执行，保留先前 AWM/Trace2Skill/Ours。
- 数据沿用 `data/eval/sgd/protocol_20pct_seed42_seen_schema_v1`，9个验证 family，16,591 TRAIN policy turns，2,480验证目标，包含 non-call。
- 每方法10个 case 线程；共享最多10个在途HTTP请求，最多5个活跃 workflow ID，每 ID 2 QPS、发送间隔至少0.55秒。沿用 `configs/sgd_workflows.json` 和原共享 SQLite 库，不建立独立库来绕过旧配额。
- 默认复用配置中的 `awm` ID 池；其余方法也顺序使用这5个ID。可以通过 `--workflow-pool-method` 选择其他已配置的池，无需增加ID。
- 调用使用服务器已有的 `llm.py`、原公司 `config.py` 和单次HTTP transport；不依赖本地开发文件名 `llm_new.py`，不读取 shubiaobiao 密钥，也不需要本地 `.env`。
- ExpeL 使用本地 ABCD 对齐后的 TRAIN gold policy/reference 监督 v2，资源冻结后评测。Reflexion 沿用当前 SGD 整段 joint 成功判断、最多两次尝试；该成功判断比本地 ABCD action-only 判断严格，未在此轮暗改。
- 这是 seen-schema 官方 validation 协议，与之前主比较设置对应；不自动增加官方 test。

## Bash 命令

将代码包上传至公司项目根目录，解压：

```bash
tar -xzf sgd_extra_baselines_20261020.tar.gz
```

修正版代码包不包含数据、llm.py、config.py、configs/sgd_workflows.json、密钥或历史输出；这些沿用服务器已有文件。请确保之前使用旧代码的 SGD 批次已经结束再覆盖代码。

预检（不调用模型）：

```bash
python scripts/launch_sgd_extra_server.py --output-dir outputs/company_sgd_20pct_extra_v1 --dry-run
```

启动后台批次：

```bash
python scripts/launch_sgd_extra_server.py --output-dir outputs/company_sgd_20pct_extra_v1
```

启动器会立即返回，后台独立 session 继续运行。

查看状态/完整指标：

```bash
python scripts/launch_sgd_extra_server.py --output-dir outputs/company_sgd_20pct_extra_v1 --status
cat outputs/company_sgd_20pct_extra_v1/comparison.md
tail -n 30 outputs/company_sgd_20pct_extra_v1/batch_coordinator.log
```

每方法结果为 `<运行目录>/<method>/aggregate.json`，分场景数据和训练/评测断点位于对应 family 子目录。各方法报错后记录状态并继续下一个；缺失/报错结果不能当作干净完成。

停止：

```bash
python scripts/stop_sgd_server_batch.py --output-dir outputs/company_sgd_20pct_extra_v1 --execute
```

恢复：

```bash
python scripts/launch_sgd_extra_server.py --output-dir outputs/company_sgd_20pct_extra_v1 --resume
```

恢复只跳过完整结果并使用现有资源/断点；未完成 ExpeL 批次会重跑，Skill-DisCo 的中间分组/编译阶段可能重跑，所有请求日志保留计费审计。已完成但有 case error 的结果不会在恢复时自动补错。
