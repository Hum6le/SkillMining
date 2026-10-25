# ABCD guideline → skill 知识覆盖实验

这条链路评估已保存的 skill 是否写出了官方 guideline 的条件、分支动作和适用字段规则，不重新训练、不执行 agent、不读取 TRAIN/test 对话或端到端指标。它与 `benchmark_sop_graphs.py` 的图边召回不同，也不是 SGD 的系统动作定义覆盖。

支持一个 coarse subflow（如 `shipping_issue`）或全部十个。三方法使用同一份固定参考清单。可增加方法，但每个方法都必须存在同样的 flow 资源，不能静默丢掉缺失的方法或场景。

## 四文件范围与旧任务恢复（2026-10-25）

final 扫描中，方法名为 `ours` 时，默认仅读取 `skill.md`、`online_action_rules.md`、`online_slot_policies.md`、`online_transition_guards.md`。排除 working_skill、静态 action/slot 汇总、reference、online_reference 和 JSON state 的隐式资源回退。文件存在且内容不同时最多四类；多 flow 仍分别扫描。其它方法维持原有最终产物范围。显式 `--ours-resources all` 可恢复旧的宽范围扫描。

旧宽范围的结果不能通过修改 inventory 原地换口径。更新工具包后，在项目根目录执行：

```bash
OLD=outputs/abcd_guideline_shipping_20261025_154930
NEW=outputs/abcd_guideline_shipping_core_v4
nohup python -u scripts/recover_abcd_guideline_audit.py \
  --previous-dir "$OLD" --output-dir "$NEW" \
  --workers 1 --qps 1.8 > "${NEW}.log" 2>&1 < /dev/null &
echo "PID=$!"
tail -f "${NEW}.log"
```

恢复脚本从旧 inventory 读取方法目录、flow override、分块参数，从旧 extraction_config 读取模型和 provider；复用旧 `reference/reference.json`。公司 `existing` provider 会在导入客户端之前启用 workflow 后端，并进行零调用配置检查。原始产物目录必须仍可访问。可先追加 `--dry-run` 打印命令，不扫描也不调用 API。

新范围只对**块身份、文本、偏移、节点上下文、文档来源和抽取提示完全一致**的块复用旧响应，并重新校验逐字引用；其它块重新调用。`--reuse-matching-chunks` 需要与 `--reuse-responses-from` 同用；旧 source inventory 也必须通过 hash 校验。不会从被排除的 reference 把旧规则带回新结果。旧输出保留。

新版抽取把失败块的 method、source_file、last_error 写入 `summary.json` 的 `failed_chunk_details`。存在失败块时 CLI 退出码为 2，恢复脚本不会继续覆盖评测；重新执行同一恢复命令可复用新目录中的成功块，只补失败块。单纯 `--max-chunks` smoke 未完成且没有失败块时仍可正常退出，但评测仍拒绝不完整输入。

## 本次检查补齐的部分

- ABCD `subflow/coarse_flow` 元数据识别；`--subflows` 与 `--families` 等价。
- 未带元数据的单场景产物可显式 `--family-override method=shipping_issue`；已有元数据冲突则拒绝扫描。
- final 不读取 `offline_artifact`、seed、历史 batch/失败分析目录；支持 Ours 的主文件与分离资源、AWM workflow、Trace2Skill evolved_skill/references。
- guideline 条件与其子条目保持在同一个参考单元，不把“若未发货→waive fee”拆成无条件“waive fee”。
- 区分询问 “ask if they need anything else” 和真正的条件；保留 “忘记用户名→In this case, Recover Username” 一类恢复条件。
- 模型只返回参考 ID 与已有规则 ID；源 skill 引文由程序回填并核验，无需模型重抄长引文。
- 启动时还核验官方参考三文件的 hash、guideline JSON pointer 与逐字原文；不能把自己补的业务规则冒充官方规范。
- 失败/漏项显示 unassessed，分数为 null，不算 absent。支持续跑缓存；源码、提示协议或输入变化拒绝续跑。
- 自动建议、复核 CSV、复核后统计分开。没有复核完全部矩阵，不生成正式 reviewed 表。

## 参考单元与分母

参考由官方 `guidelines.json`、`ontology.json`、`kb.json` 编译。后两者用于 flow/fine-task 身份核验，不把 KB 相邻边作为分支语义替代。每个原文 action step（正文及全部 subtext）或 instruction 是一个固定参考单元，保留 fine-task 作用域。它是官方文字单元，**不声称是原子规则总数**。

`covered` 要求完整支持该单元的适用条件、动作以及各条字段/值约束。仅提到正确 action 不算覆盖条件规则。`partial`、`absent`、`contradicted`、`unresolved_conflict`、`unverifiable` 分开记录；只有 covered 计入严格覆盖率。

主指标为 eligible `conditional_action` 单元的严格覆盖率；另报全部单元、kind、fine-task macro、coarse-flow macro 和共享原文去重诊断。相同文字在不同 fine task 中仍保留原始适用范围；共享原文组只有所有范围都覆盖才算完整，避免用重复表述冒充独立知识量。

Shipping 的 Missing Item 原文同时写“7 days or less”和“a week or longer”，在第七天有重叠；这两条先标 `reference_review_required`，不猜一个阈值。独立的 “Wrap up as usual.” 同样先排除。排除项 ID 和原文会保存，不能事后随某方法结果任意删除。

当前 v2：`shipping_issue` 为 29 个原文单元，26 个 eligible，其中 10 个条件单元；全十流为 365 个原文单元，359 个 eligible，其中 129 个条件单元。发布论文前应检查 `reference_review.csv` 的任务范围与这些表面条件分类；这是开发性参考清单，不能宣称额外独立人工 gold。

## 1. 服务器导出真实产物：不调用 LLM

先同步工具包内的代码；**不要覆盖公司的 llm.py/config，也不需要 llm_new.py**。以下在公司服务器项目根目录运行，语法为 Bash。将三个变量改成同轮、同模型、同训练比例的最终产物目录；Ours 应指向 `shipping_issue/full` 之类的实际 arm，不要指向包含所有 ablation 的公共父目录。

`--provider existing` 本身只表示复用 `llm.py`，不强制后端。公司统一入口需要在启动 Python 前执行 `export SKILLMINING_LLM_BACKEND=workflow`。也应把该 export 写在后台启动脚本中，使断点重启继续生效。

零 API 调用预检查：

```bash
export SKILLMINING_LLM_BACKEND=workflow
python - <<'PY'
import llm
get_config = getattr(llm, 'get_llm_config', None)
assert callable(get_config), '当前 llm.py 没有启用 workflow 客户端'
cfg = get_config()
assert cfg.get('provider') == 'workflow', '配置 provider 不是 workflow'
assert cfg.get('workflow_id'), '未配置 workflow_id'
print('Workflow backend OK:', llm.chat.__module__)
PY
```

```bash
cd /data1/liuxiangfeng/SkillMining

OURS_DIR="填入本轮shipping_issue的Ours最终目录"
AWM_DIR="填入同轮AWM最终目录"
TRACE_DIR="填入同轮Trace2Skill最终目录"

test -d "$OURS_DIR" && test -d "$AWM_DIR" && test -d "$TRACE_DIR" || exit 1

python scripts/extract_skill_knowledge.py scan \
  --run "ours=$OURS_DIR" --run "awm=$AWM_DIR" --run "trace2skill=$TRACE_DIR" \
  --subflows shipping_issue \
  --family-override ours=shipping_issue \
  --family-override awm=shipping_issue \
  --family-override trace2skill=shipping_issue \
  --output-dir outputs/abcd_skill_sources_shipping_v1 \
  --archive artifacts/abcd_skill_sources_shipping_v1.tar.gz
```

检查打印的文件数与 `inventory.json` 的 resources 列表。family override 只适用于每个目录确实只有该 coarse flow 的情况。全十流使用各方法的多 flow 根目录及 `--subflows account_access,...`，不要强行用单 flow override；每个 flow 必须有 metadata。

导出包只有清单与允许的 skill 文本，不含 `.env`、客户端配置、原始对话、预测或完整运行目录。可把小包带回本地用中转站；也可全部留在服务器执行后续步骤。

## 2. 构建相同的官方参考

```bash
python scripts/benchmark_abcd_skill_guidelines.py reference \
  --subflows shipping_issue \
  --output-dir outputs/abcd_guidelines_shipping_v2
```

全十流改为 `--all`。如官方资源不在默认目录，用 `--data-dir PATH` 指向只含三份官方 JSON 的目录。这个步骤零 LLM 调用，不会读任何方法产物。

## 3. 抽取产物中的显式规则

服务器复用当前公司客户端：

```bash
python -u scripts/extract_skill_knowledge.py extract \
  --source-dir outputs/abcd_skill_sources_shipping_v1 \
  --output-dir outputs/abcd_skill_rules_shipping_v1 \
  --provider existing --model qwen3.6-flash \
  --workers 1 --qps 1.8
```

需要指定 workflow 时加 `--workflow-id YOUR_WORKFLOW_ID`，底层重试和共享限流仍由现有公司客户端负责，不承诺仅靠顶层 QPS 覆盖内部重试。建议先串行 smoke，与正在跑的主实验共享模型时尤其如此。

本地中转站改 `--provider shubiaobiao`，沿用本机既有 `.env`，不在命令中写密钥。可先用 `--max-chunks 3` smoke，再在同目录去掉限制续跑。抽取没有全部成功之前，覆盖评测会明确拒绝启动。

## 4. 预检查与自动覆盖建议

```bash
python scripts/benchmark_abcd_skill_guidelines.py evaluate \
  --reference outputs/abcd_guidelines_shipping_v2/reference.json \
  --source-dir outputs/abcd_skill_sources_shipping_v1 \
  --rules-dir outputs/abcd_skill_rules_shipping_v1 \
  --output-dir outputs/abcd_skill_coverage_shipping_v1 \
  --provider existing --model qwen3.6-flash \
  --workers 1 --qps 1.8 --dry-run
```

确认 candidate_counts 的三方法/flow 与实际产物一致，再以同一指令去掉 `--dry-run` 执行。模型判定可加 `--max-batches 1` 做一次端到端 smoke，然后去掉限制续跑；完整参考分母始终保持不变。`--origins` 默认只评 procedural_rule，不把原始对话例子或通用执行契约当作新发现的流程知识。

客户端默认是 `--provider existing`，公司启动必须先按前面的预检查启用 workflow 后端。显式 `--workflow-id` 也会在导入 llm 前选择 workflow dispatcher。本地执行则把 provider 改为 shubiaobiao；dry-run 本身不导入 llm，不需要密钥。单次输入超过 `--max-prompt-chars` 会报错，不静默截断候选知识。失败批次保留 raw/错误，重跑同指令只补未成功部分。

输出：

- `input_manifest.json`：方法/flow、规则数、固定矩阵与排除项。
- `protocol.json` / `input_freeze.json`：协议、代码与输入快照校验。
- `batches/`：原始返回、重试和失败记录。
- `proposed_assessments.json`：模型初审与实际 skill 引文；不是人工 gold。
- `provisional_summary.json`：自动阶段进度与暂定分数；不完整单元组分数为 null。
- `coverage_review.csv`：逐条复核模板，续跑刷新建议并保留已填写的复核结果。
- `llm_usage_*.json`：每次执行的客户端 usage，续跑不覆盖历史账目。

## 5. 复核后导出表格

在 `coverage_review.csv` 中填写 reviewed_status/reviewed_reason，校验 support_rule_ids、conflict_rule_ids 和 condition_rule_ids；ID 列为 JSON 数组。条件规则要判 covered，必须有支持其前提的规则证据，不能只给 action 相同的规则。

```bash
python scripts/benchmark_abcd_skill_guidelines.py adjudicate \
  --proposal-dir outputs/abcd_skill_coverage_shipping_v1 \
  --decisions outputs/abcd_skill_coverage_shipping_v1/coverage_review.csv \
  --reviewer YOUR_REVIEWER_LABEL \
  --output-dir outputs/abcd_skill_coverage_shipping_reviewed_v1
```

漏判、重复 ID、伪造规则 ID 或改动冻结快照都会阻止正式出表。最终文件为 `reviewed_summary.json`、`reviewed_assessments.json`、`REPORT.md`、`table.csv` 和 `table.tex`。reviewer 标签不自动代表独立人工复核；如仅由模型/本 agent 复核，论文中必须如实说明。

表格可列：方法、条件—动作单元覆盖数/分母、条件—动作严格覆盖率、全部 guideline 单元覆盖率、冲突数；完整附录保留 partial/unknown 等状态。该指标评估抽取规则相对官方参考的覆盖，仍受抽取遗漏影响，不是完整 artifact recall 或任务成功率。对初始化已经含有的规则，不能未经前后对照就宣称是新挖掘知识。

## 本地验证范围

已验证真实官方 Shipping/全十流 reference 构建，并用合成 fixture 验证扫描→抽取→条件覆盖建议→中断续跑→复核→CSV 全链路。fixture 分数不作为实验结果。未连接公司服务器，不声称已读取其真实 ABCD 产物或完成正式覆盖评测。
