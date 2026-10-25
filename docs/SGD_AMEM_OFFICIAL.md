# 官方 A-MEM 的 SGD 适配

日期：2026-10-24；2026-10-25 修正源码跨平台校验。本任务只先完成本地 smoke，不自动启动公司全量实验。

已有安装遇到官方文件缺失或 SHA 不符，使用 [源码修复说明](SGD_AMEM_SOURCE_REPAIR.md) 的小包补齐。新安装使用 `sgd_amem_official_20261025_v2.tar.gz`。

## 上游和复现边界

- 论文官方仓库：https://github.com/WujiangXu/A-mem （AgenticMemory 旧地址指向该仓库）。
- 本地：`third_party/a-mem-official`；固定 commit：`b31a1dddf56f87ec762cce3539418602d7e2ffb1`。
- 使用作者自己的 `memory_layer_robust.RobustAgenticMemorySystem`。三个核心源文件与 LICENSE 在统一 CRLF→LF 后校验 SHA256，预期值来自固定 commit 的 Git blobs。只兼容换行符差异，仍拒绝内容或版本改变；不重写 note/evolution/search 算法。
- 官方另一系统库 https://github.com/agiresearch/A-mem 面向 agent 应用，当前适配采用论文实验库的 robust 路径，避免混用两套实现。
- 本实验是 **官方 A-MEM 记忆机制的 SGD 任务适配**，没有声称复现原论文 LoCoMo 分数。不同任务的 memory unit、监督与 actor 输出必须单独说明。
- 当前工作区未发现生成历史 1/5 SGD A-MEM 1.21% SR 的适配代码/原始预测，不能据此判断低分根因。新产物另存，不覆盖旧结果。

## 数据和 actor 对齐

- 默认 `data/eval/sgd/protocol_20pct_seed42_seen_schema_v1`；默认九个有 seen-schema validation 的 family。正常运行 16,591 个 TRAIN policy notes、2,480 个验证 policy turns；用 dry-run 核对实际文件。
- 每个 TRAIN policy turn 对应一条 note：目标之前的可见对话、既往 API 结果、该 TRAIN turn 的参考系统 policy。保留 act slot/value 的缺失与空串区别；不把同 turn 多个 act 排成动作序列。
- 只存 TRAIN；验证时 query 只见 prior utterances/API history 和可用 service ID，不读当前 response、gold call、gold acts、当前 API result 或未来 utterance。Gold 只在 actor 返回后评分。
- 正式记忆构建按固定对话/turn 顺序串行，以保留原生演化的依赖。不能把 add_note 随意并发。`--workers` 仅加速冻结后的评测。
- 官方 note 分析、evolution decision、strengthen、neighbor update、向量余弦检索、链接展开均保留；默认 top-k=10，演化时检索 k=5。
- 使用与作者相同的 all-MiniLM-L6-v2；默认固定 HF revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，CPU 编码，记录所有权重/配置哈希。没有词频/hash/BM25 替代向量检索。
- 默认 `refresh_index=false`，沿用作者每 100 次演化 consolidation 的索引刷新；官方可选 `--refresh-index` 单独标记实验，不能静默改变。
- query keyword 生成沿用官方评测模板。检索正文完整传给共享 SGD executor；不添加“判断是否 CALL”的额外门控。
- actor 与 ReAct 完全复用 `sgd_aligned.predict_policy`：选 heads → 固定 heads 的 CALL/acts payload → exact-call logged API result gate。全部 policy turn 计分，执行错误保留在分母。
- `--compare-react` 在完全相同的样本、模型、ontology 和 TRAIN serialization contract 上运行无记忆 ReAct。它是诊断对照，不用 smoke 小样本判断方法优劣。

## 兼容、持久化和审计

- 上游有一个 Python 3.14 已移除的、未使用的 `ast.Str` import。适配器仅在 import 期间注入兼容 alias；上游文件保持原样。
- 注入项目 LLM controller，所有调用/重试经过 `sgd_chat_with_retry`，在公司版经过已有 workflow request pool；不会用官方 OpenAI controller 绕开限流。
- SGD 缺少真实录音时间，note timestamp 标为 `TRAIN_TIME_UNSPECIFIED`，避免把构建当天或虚构日期传给日期字段 binder。
- `memory.sqlite` 是学习产物与断点：保存 notes、links、原始 corpus、embeddings 和 evo_cnt。每完成一条 TRAIN note，用一个事务写入新 note 和实际被修改的邻居；原生 consolidation 时刷新全部索引行。
- 断点只在成功后推进；上游吞掉的 analyze/evolve 失败会被适配器发现并中止，防止带着空记忆报告正常低分。无效 link 等恢复计数保留在 resources，不能隐藏。
- 每个家族开始评测前冻结语义 hash 和 SQLite 文件 hash，结束时再次检查。验证不增删记忆，不根据验证正确性更新资源。
- 保存 upstream/source/data/model hashes、`amem_completions.jsonl`、逐样本 `amem_retrieval`、错误和调用量。官方邻居展开的重复文本保持原样，因此 top-k 不等于实际展示 note 数量。

## 本地安装与 smoke（PowerShell）

已创建独立 `.venv-amem`；基础环境不需要修改。新机器可运行：

```powershell
python -m venv .venv-amem
.\.venv-amem\Scripts\python.exe -m pip install -r requirements-amem-official.txt
.\.venv-amem\Scripts\python.exe scripts/run_sgd_amem_official.py --families Homes --smoke --max-train-dialogues 2 --max-train-notes 8 --max-eval-dialogues 2 --max-eval-turns 6 --compare-react --workers 2 --output-dir outputs/sgd_amem_official_smoke_new
```

本地复用中转站 `.env`，不把密钥写进代码/命令/结果。

## 公司机器迁移（Bash，尚未启动）

代码包包含上游三个核心文件和 LICENSE；不含 SGD、密钥、根 `llm.py`/`config.py` 或工作流配置。公司保持自己的 `llm.py`，不要求 `llm_new.py`。也不向 Git 提交 SGD。

```bash
cd /data1/liuxiangfeng/SkillMining
tar -xzf /path/to/sgd_amem_official_20261025_v2.tar.gz
python -m venv --system-site-packages .venv-amem
.venv-amem/bin/python -m pip install -r requirements-amem-official.txt
.venv-amem/bin/python scripts/verify_amem_official.py

# 只预检查数据、上游代码和公司接口配置，不发请求、不下载 embedding。
.venv-amem/bin/python scripts/run_sgd_amem_official.py \
  --provider workflow --workflow-pool-method awm \
  --output-dir outputs/company_sgd_20pct_amem_preflight --dry-run
```

如果服务器访问不了 Hugging Face，从本机复制 MiniLM snapshot 为完整目录，并加 `--embedding-model /path/to/all-MiniLM-L6-v2`；包含 safetensors、tokenizer/config 和 1_Pooling 文件。离线时可设 `HF_HUB_OFFLINE=1`。

先在公司验证相同 smoke：

```bash
.venv-amem/bin/python scripts/run_sgd_amem_official.py \
  --provider workflow --workflow-pool-method awm \
  --families Homes --smoke --max-train-dialogues 2 --max-train-notes 8 \
  --max-eval-dialogues 2 --max-eval-turns 6 --compare-react --workers 2 \
  --output-dir outputs/company_sgd_amem_smoke_v1
```

之后再决定完整 1/5 运行；对应命令如下（不在本轮自动执行）：

```bash
mkdir -p outputs/company_sgd_20pct_amem_official_v1
nohup .venv-amem/bin/python -u scripts/run_sgd_amem_official.py \
  --provider workflow --workflow-pool-method awm --workers 10 \
  --output-dir outputs/company_sgd_20pct_amem_official_v1 \
  > outputs/company_sgd_20pct_amem_official_v1/run.log 2>&1 < /dev/null &
echo $! > outputs/company_sgd_20pct_amem_official_v1/launcher.pid
```

必须与其它方法串行运行，复用同一 workflow 限流数据库：总模型 inflight≤10、最多 5 workflow、单 workflow QPS≤2。逐 note 学习天然串行，不能据评测并发 10 推断训练也有 10 路。

查看 `progress.json`、`aggregate.json`、`Homes/summary.json`；异常见 `fatal_error.json`。正常断点恢复使用完全相同命令和输出目录。协议、模型、数据或源码不同须新目录。
