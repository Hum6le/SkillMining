# SGD 1/10：沿用 Homes setting 的批量实验

日期：2026-10-19。用户确认沿用刚才的 setting；服务器选择稍后讨论。本文件记录数据、实验顺序和可运行命令，不代表已部署或已启动实验。

## 1. 冻结数据协议

- 原始数据：`data/eval/sgd/data.zip`。保留官方 TRAIN / validation / test 边界。
- 沿用 `splits_10pct_seed42` 的现有抽样 ID：每个 family、每个 split 内，以 `sha256(seed + NUL + split + NUL + dialogue_id)` 排序，取 `ceil(0.1 * 对话数)`，seed=42。
- 抽完整对话，不随机抽 turn，不截断对话。各 family 各自满足 10% 配额，因此去重后的全局对话数不严格等于官方全量的 10%。
- 主评测沿用 Homes 的同 schema setting：先抽 10%，再保留目标 service ID 全部在该 family 的 sampled TRAIN 中出现的评测对话视图；不重新补抽。
- TRAIN 内容逐字节保留。上下文可含其他 family；只评分该 family 负责的目标。跨 family 无法归属的 global act 只保留为上下文，单 family global act 正常计分。
- CALL 和非 CALL 的 policy turn 全部计入指标。评测按 `(dialogue_id, turn_index)` 去重验证，不把对话视图数当独立对话数。
- 对话可能在不同 family 下作为上下文出现，但本采样的目标 turn 没有跨 family 重复；官方 split 之间无对话泄漏。
- 固定目录：`data/eval/sgd/protocol_10pct_seed42_seen_schema_v1`。`SAMPLING_MANIFEST.json` 保存精确 ID、排除 ID 和各文件 SHA256。不要原地修改。

| 范围 | 去重对话 | policy turn |
|---|---:|---:|
| 原 10% TRAIN（16 family） | 1,689 | 14,872 |
| 原 10% validation（含新 schema） | 264 | 2,130 |
| 原 10% test（含新 schema） | 435 | 3,181 |
| 主评测 validation：同 schema，9 family | 203 | 1,241 |
| 后续 test：同 schema，6 family | 212 | 798 |

结果应标为 **SGD 10% seen-schema subset**，并明确当前主评测是官方 validation，不能称为全量官方 test。未见 schema 的原始采样保留，迁移实验另外报告。

## 2. 先跑哪些

三方法都跑：AWM、Trace2Skill、Ours。使用同样的数据、模型 `qwen3.6-flash`、共享 aligned action→payload 运行协议及指标。Ours 保持当前训练实现：standard refine、每节点初始 4 段 TRAIN snippet、原生 slot policy / reference 更新；本批次不继续修改 action 链路。

| family | TRAIN turn | 主评测 validation turn | 后续 test turn |
|---|---:|---:|---:|
| Buses | 1,229 | 202 | — |
| Events | 2,049 | 229 | — |
| Homes | 507 | 115 | — |
| Hotels | 1,574 | 103 | 150 |
| Movies | 659 | — | 122 |
| Music | 603 | 105 | — |
| RentalCars | 1,034 | 245 | — |
| RideSharing | 638 | 97 | 85 |
| Services | 1,405 | — | 200 |
| Travel | 453 | 73 | 144 |
| Weather | 182 | 72 | 97 |

1. **迁移验收**：安装依赖、核对数据和命令。服务器能调用 API 后，用单独目录做少量真实样本 smoke，测限流与吞吐。
2. **第一批，默认启动范围**：9 个 validation family × 3 方法 = 27 个训练/评测任务。每方法 TRAIN 8,269 turn + validation 1,241 turn，共 9,510 turn；三方法共 28,530 turn。旧 Homes 结果用于历史对照，不混入新批次主表。
3. **后续最终测试**：冻结代码与训练资源后再跑 test。Hotels / RideSharing / Travel / Weather 直接复用第一批各自的 TRAIN 资源；Movies / Services 各训练一次后测 test。三方法总计再增加 18 个阶段任务。
4. 完整两阶段覆盖 11 个 family，每方法真正训练 10,333 turn、validation 1,241 turn、test 798 turn，总计 12,372 turn；三方法 37,116 turn。**turn 数不是 API 调用数**，诊断、reflection、MAP/REDUCE 会额外调用。

当前图的节点是 action head bundle，规模仍偏小；此批验证跨数据集表现。大动作图结论继续以 ABCD 为主。当前不追加 MultiWOZ、WebArena 或 schema 迁移实验。

指标：Action set exact / Action micro F1、Slot exact given correct action、Turn joint exact，附 slot item micro P/R/F1、调用与 token 统计、错误数。跨 family 汇总从 TP/分母重算，另报各 family，不能简单平均百分比。

## 3. 并行与时间

建议起点：Ours 24 worker、AWM 12 worker、Trace2Skill 12 worker，总计 48 个 case worker 上限；每方法一次处理一个 family，三方法同时运行。训练 rollout / diagnosis / evaluation 可并行；会修改同一 skill 的 reflection / APPLY 顺序执行。

这是并发配置草案，尚未验证中转站在服务器上的限流。实际运行存在串行阶段，48 worker 不等于持续 48 个请求。先测 12→24→48 总并发的吞吐和错误率，再确定最终配置。该任务调用远程 LLM API，无 GPU 依赖；内存以实际 worker RSS 测量为准。

一天内完成第一批要求 Ours 每小时平均完成约 397 个 TRAIN+eval 目标 turn，完整两阶段要求约 516 个；这些是必要吞吐下界，额外反思时间必须计入。用迁移 smoke / 首个 family 测出的各阶段耗时更新 ETA，当前不承诺 24 小时完成。

## 4. Linux 安装与持久配置

迁移包不含密钥、旧输出或缓存。包内有代码、原始 SGD zip、冻结后的采样数据及文件哈希。示例目录自行替换：

```bash
mkdir -p ~/experiments/sgd
tar -xzf sgd_server_bundle_20261019.tar.gz -C ~/experiments/sgd
cd ~/experiments/sgd/Skill_Baseline_SGD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-sgd-server.txt
cp configs/sgd_server.env.example configs/sgd_server.env
chmod 600 configs/sgd_server.env
```

编辑 `configs/sgd_server.env`，填 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。每次启动都会加载这个文件，不依赖当前 shell 的临时 export。不要把文件加入版本库或数据包。

### 零 API 调用预检

```bash
python scripts/launch_sgd_server_batch.py \
  --output-dir outputs/sgd_server_preflight --dry-run
```

### 第一批：9 个 family 的同 schema validation

```bash
python scripts/launch_sgd_server_batch.py \
  --output-dir outputs/sgd_10pct_seen_v1 \
  --ours-workers 24 --baseline-workers 12
```

启动器在后台启动三方法独立协调进程并立即返回；Linux 使用独立 session，Windows 隐藏窗口。SSH 断开后任务继续，日志直接写输出目录。

### 查看进度

```bash
python scripts/launch_sgd_server_batch.py \
  --output-dir outputs/sgd_10pct_seen_v1 --status
tail -f outputs/sgd_10pct_seen_v1/ours/validation.log
```

协调状态与完成的 family 任务数由 `--status` 展示；具体阶段/turn 进度查看方法日志、family 输出中的训练及预测记录。各阶段 `aggregate.json` / `aggregate.md` 汇总结果与未完成任务。

### 中断后恢复，或第一批完成后追加 test

```bash
python scripts/launch_sgd_server_batch.py \
  --output-dir outputs/sgd_10pct_seen_v1 --resume \
  --ours-workers 24 --baseline-workers 12

# 代码和配置冻结、所有当前任务退出后，才追加 held-out test
python scripts/launch_sgd_server_batch.py \
  --output-dir outputs/sgd_10pct_seen_v1 --resume --stage all \
  --ours-workers 24 --baseline-workers 12
```

当前恢复粒度是 family 任务：已有 summary 的任务会跳过；不是任意在线训练 turn 的精确断点恢复。未结束的 Trace2Skill 目录若已有训练产物，需要先保留原目录并为该任务准备新目录后重跑，不能直接覆盖。含错误 summary 也不能靠 resume 自动修复，要检查/恢复错误后再汇总。协调进程仍活跃时，启动器拒绝重复恢复。若任一方法的当前阶段有任务失败/未解决错误，该方法不会继续进入 test 阶段。

## 5. 当前交付状态

- 采样与同 schema 筛选：完成，manifest 已冻结。
- 三方法批量入口、训练资源复用、持久 env、后台运行：已实现；相关本地回归测试 21 passed、1 个 Linux 专用测试在 Windows 跳过。测试没有发起真实 API 请求。
- 服务器选择、真实 API smoke、吞吐测量、正式启动：待稍后确定服务器；目前没有启动本批付费实验。

复现抽样前提是原始 `splits_10pct_seed42`；迁移包已含最终 profile，因此服务器无需重新抽样。
