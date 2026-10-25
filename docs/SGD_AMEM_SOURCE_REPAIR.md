# A-MEM 公司预检查：源码缺失 / 跨平台 SHA 修复

2026-10-25。

## 已确认的问题

1. `third_party/a-mem-official/memory_layer.py` 缺失：适配器在 Git 中，官方克隆目录被项目 `third_party/` 忽略规则排除。代码拉取不等于官方文件已经同步；代码包包含这几个文件。
2. 旧 checker 记录了 Windows `core.autocrlf=true` checkout 的 CRLF 文件字节 SHA。相同 commit 在 Linux LF checkout 下四个 SHA 都不同，因此会误报版本不符。本地已将 `git show COMMIT:path` 与 LF 规范化哈希逐项比较，确认原因。
3. 服务器是否还克隆了其它 commit 无法仅靠 traceback 判断。小修复包直接带固定 commit 的四个文件，同时解决缺失、错误版本和换行符这几种安装问题。

修正后只统一 CRLF→LF；额外内容、改注释、其它版本仍会失败并给出文件名和 expected/actual SHA。SQLite、embedding 和其它二进制文件仍按原始字节校验。不是取消版本验证。

## 修复已有安装（Bash）

把 `artifacts/sgd_amem_source_fix_20261025.tar.gz` 上传到公司项目根目录。它只含 checker、独立验证入口、四个固定官方文件和本说明，不包含 root llm/config、依赖环境或任何数据/实验产物。

```bash
cd /data1/liuxiangfeng/SkillMining
tar -xzf sgd_amem_source_fix_20261025.tar.gz
.venv-amem/bin/python scripts/verify_amem_official.py
```

应打印 `Official A-MEM source OK`，commit 为 `b31a1dddf56f87ec762cce3539418602d7e2ffb1`。

然后重新跑预检查和 smoke：

```bash
.venv-amem/bin/python scripts/run_sgd_amem_official.py \
  --provider workflow --workflow-pool-method awm \
  --output-dir outputs/company_sgd_20pct_amem_preflight_v2 --dry-run && \
.venv-amem/bin/python scripts/run_sgd_amem_official.py \
  --provider workflow --workflow-pool-method awm \
  --families Homes --smoke \
  --max-train-dialogues 2 --max-train-notes 8 \
  --max-eval-dialogues 2 --max-eval-turns 6 \
  --compare-react --workers 2 \
  --output-dir outputs/company_sgd_amem_smoke_v2
```

这两条 traceback 出现在 dry-run 构造 manifest 时；该指令链还没进入 smoke，也没有执行记忆构建/API 调用。保留旧目录，用新的 v2 目录便于区分。源码签名格式改变后，已有成功训练/评测目录不会被脚本静默改写或迁移。

新部署使用完整修正版 `artifacts/sgd_amem_official_20261025_v2.tar.gz`。完整安装步骤见 `SGD_AMEM_OFFICIAL.md`。不需要重新下载 SGD 或重新安装依赖来解决本次源码校验问题。
