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

## JSON 语法错误或截断：仅补失败块

若已有 10 个成功块、少数块出现 `JSONDecodeError`，用 `scripts/repair_abcd_guideline_extraction.py` 补跑。`Unterminated string` 可能来自输出截断，不能仅凭 JSON 异常断言平台 token 上限；本地 workflow 传输也没有把 `chat(max_tokens=...)` 直接作为工作流生成参数发送，因此本脚本不假设修改客户端 max_tokens 可以解决平台限制。

```bash
nohup python -u scripts/repair_abcd_guideline_extraction.py \
  --audit-dir outputs/abcd_guideline_shipping_core_v4 \
  --reference outputs/abcd_guideline_shipping_20261025_154930/reference/reference.json \
  --output-dir outputs/abcd_guideline_shipping_core_jsonfix_v1 \
  --workers 1 --qps 1.8 \
  > outputs/abcd_guideline_shipping_core_jsonfix_v1.log 2>&1 < /dev/null &
```

此工具只需新增该脚本，依赖已经安装的 v4 工具接口，不修改抽取提示、校验器或公司客户端。原始文档逐字保留；成功窗口的内容、身份和边界不变，复用缓存时仍逐字重验。失败窗口依次以约 1600、800、500 字符的上限细分，保留重叠，校验首尾与中间均无源文本缺口。每一轮有独立的 source inventory、规则和父子窗口映射；抽取粒度改变会影响抽取描述性数量，不将拆分带来的条数变化作为知识质量提升。

最多三轮，仍失败则明确停止，不补括号猜测 JSON，不丢弃坏行，不把未完成算 absent。全部抽取完成后继续执行官方参考覆盖建议。根目录 `latest_extraction.json` 指出当前实际 source/rules 路径；覆盖结果写入 `coverage/`。同一条命令可续跑，成功块不重复调用 LLM。此流程仅以模拟响应验证过，不代表已在公司服务器完成实际补跑。

## 引用行号反复越界：逐字引文补跑

`ValueError: Source line range outside the supplied chunk` 表示模型生成了无效的局部行号。展示和校验都基于 splitlines，没有发现固定偏移；正文里的编号、短块及固定示例可能干扰模型，实际原因需结合 raw 响应判断。继续拆小并不能保证模型遵守编号约束。

新增单文件工具 `scripts/repair_guideline_citations.py`，不修改现有 extractor、validator、repair helper 或批次冻结的代码 hash。它使用显式新协议 `literal_quote_extraction_repair_v1`：成功缓存仍以旧原始响应逐字重验；失败/缺失块改为要求最小连续逐字引文，严格检查引文属于该块、且位置唯一。拒绝近似匹配、空引文、重复定位及坏 JSON，不猜测、裁剪或平移行号，不静默丢弃错误规则。若旧失败响应已包含可验证的精确引文，可据此定位并记录原响应 hash；只有数字行号、没有引文的响应必须重新请求。

对已启动的全量批次，可直接启动补跑：

```bash
nohup python -u scripts/repair_guideline_citations.py \
  --batch-dir outputs/abcd_guideline_all_v1 \
  --output-dir outputs/abcd_guideline_all_citations_v1 \
  --workers 4 --qps 1.8 \
  > outputs/abcd_guideline_all_citations_v1.log 2>&1 < /dev/null &
```

如旧批次仍持有运行锁，补跑会等待且不发起 API 请求，每约 30 秒打印等待状态。原批次退出后逐场景处理。已完成的场景（包括 shipping_issue）重新验证冻结证据后直接复用；失败场景从最近一次细分补跑的 actual source/rules 读取输入，保留成功块，只用新引用协议补失败块，再执行覆盖评测。原批次数据保持原样，新结果在新根目录的 `aggregate_summary.json`、`by_subflow.csv`、`assessments.json` 和 `progress.json`。之后续跑应重复这条引用补跑命令；不要把新结果目录交给旧批量入口作为原批次目录。

各修复场景位于新根目录 `flows/<flow>/`，包含完整来源副本、独立抽取协议、原成功响应出处、每次新请求与原始响应，以及 `rules/summary.json` 中的 `response_origins`。新请求引用协议变化可能影响抽取描述性条数，不能把条数变化当作业务知识增长；正式结论仍需审查实际源证据。

支持独立场景补跑：

```bash
python scripts/repair_guideline_citations.py \
  --audit-dir 某个场景的run或repaired目录 \
  --reference 某个场景的run/reference/reference.json \
  --output-dir 独立的新结果目录
```

批量入口、引用补跑与旧 JSON 补跑链路已用模拟模型响应验证，尚未替代服务器上的实际 API 验证。

## 覆盖判断中的未知／重复 rule ID

`Unknown/duplicate rule ID in .../support_rule_ids` 是同一条校验覆盖两种不同情况：列表内重复 ID，或引用不存在的 ID。不能仅从这一行判断是哪一种。工具 `scripts/repair_guideline_coverage_ids.py` 自动检查原始失败回答，并逐项记录诊断：

- 已存在的 ID 在同一列表内重复，按集合语义去重；不改变 status、reason 或证据集合。不同参考条目间复用同一个正确规则 ID 仍合法。
- 未知 ID 不做猜配、前缀匹配或丢弃。仅失败批次重新调用模型，用 `U1...`、`R1...` 短别名表示本批参考条目和已冻结候选规则。程序按显式映射恢复真实 ID，再调用原始完整校验器。覆盖的条件、字段和冲突要求保持原样。
- 若短别名批量响应仍失败，细分参考条目直到单条重判，候选规则不做 top-k 截断；任何条目失败则原批次仍失败，不计 absent。

在现有结果目录直接补覆盖评测，不重复抽取：

```bash
nohup python -u scripts/repair_guideline_coverage_ids.py \
  --batch-dir outputs/abcd_guideline_all_citations_v1 \
  --workers 4 --qps 1.8 \
  > outputs/abcd_guideline_all_citations_v1_idrepair.log 2>&1 < /dev/null &
```

已完成批次先通过冻结输入与原始响应校验，然后直接保留；原始 skill、抽取规则及 input_freeze 四个文件不改。旧失败批次另存在每个场景的 `coverage/id_repairs/<batch_name>/original.json`，同目录 `diagnostics.json` 区分重复与未知，`request_*.json` 记录新调用、别名映射及原始响应。完成后的批次附带 `repair` 来源。批次根目录 `coverage_id_repair_original/` 保留补跑前的汇总和状态；`coverage_id_repair_config.json` 显式记录此输出修复扩展的版本、核心校验器和输入冻结 hash，没有把新增调用冒充原始脚本调用。

补跑会等待同目录原任务退出，完成后原位更新 `aggregate_summary.json`、`by_subflow.csv`、逐项证据和各场景 provisional_summary/review CSV；已有复核字段保留。相同命令可以续跑。该工具也支持原始 `run_abcd_guideline_batch.py` 目录。若抽取尚不完整，该场景会明确报错，需先完成抽取，不会用本工具绕过。

## 2026-10-26：account_access 的容器格式失败与阶段恢复

如果 `rules/summary.json` 的 `complete=false` 且错误为 `Expected JSON object with a rules list`，说明仍有抽取失败，不能先评测。旧覆盖补跑入口只用 coverage/protocol.json 判断平铺目录，导致在该场景尚未评测时错误拼出 `/run/reference/reference.json`，掩盖了真实上游错误。

更新两个文件 `repair_guideline_citations.py`、`repair_guideline_coverage_ids.py` 后，仍使用同一条覆盖补跑命令。入口按实际保存的 source/rules/latest_extraction 识别平铺或嵌套目录；错误状态保留实际 audit_dir。若已知引用补跑抽取未完成，先读取其 source_audit_dir 和原始参考 hash，仅恢复失败块；完成后才创建参考副本和覆盖协议。初始化覆盖使用 dry-run，不调用模型，后续直接按短 ID 判定。已完成场景、成功抽取块和成功覆盖批次保留。

新的容器解析仅支持可证明不改变内容的 JSON 包装转换：完整规则数组包为 rules 对象、完整单条 typed rule 包成数组、一层 JSON 字符串解码。每条规则仍需通过原字段和逐字引文校验。`null`、缺失 rules、未知 schema、字符串规则列表、不完整 JSON 均不解释为空。原明确的空数组可解释为模型返回空规则集合。缓存中的失败响应优先重新校验，能转换则无需新调用；否则首次重试就明确要求抽取任务与顶层 rules 数组，避免执行源 skill 的任务说明。

兼容升级只允许已知旧补跑脚本的确切代码指纹，以及冻结输入/模型/base prompt 不变的解析器恢复。旧配置另存；已存在的 input_freeze hash 不允许改变。新完成的场景可以在明确初始化后加入新的冻结记录。正常续跑无需新建总结果目录，不重跑已完成的九个场景。

## 最后一轮与直接报告

`--final-attempt` 将本轮剩余抽取块和覆盖批次各限制为一次新请求；覆盖批次失败后也不再递归细分重判。已完成块/批次和可机械修复的缓存照常复用。抽取类别标签只允许大小写、空格、连字符到标准枚举的拼写规范化；真正未知的 origin/category 不做语义猜配，也不默认升级为 procedural_rule。重试明确列出合法枚举并要求单一值，原始值和规范化保留在响应记录中。

```bash
python -u scripts/repair_guideline_coverage_ids.py \
  --batch-dir outputs/abcd_guideline_all_citations_v1 \
  --final-attempt --workers 4 --qps 1.8
```

根目录 `final_attempt_record.json` 记录最后一轮。一旦开始过，同一 final-attempt 命令再次执行只导出缓存结果，不再调用 API。若启动后意外中断，下一次也遵守该限制，不自动继续 API 修复。

最后一轮无论是否补齐目标，都会生成并在控制台打印 `REPORT.md`，另存 `report_table.csv` 和 `report_summary.json`。主表只使用三方法全部完成的共同 subflow。完整目标的分母和未完成状态另存 full_target_summary，不将错误样本算 absent/零分。若仅 account_access 未完成，则主表为其余九个场景，343 个有效参考单元，其中 125 个含条件单元；account_access 对三个方法同时排除并明确标记，不伪称全十场景结果。若全部补齐，则主表自动覆盖十个场景。

`--report-only` 完全不调用 API，直接验证并汇总现有证据：

```bash
python scripts/repair_guideline_coverage_ids.py \
  --batch-dir outputs/abcd_guideline_all_citations_v1 --report-only
```

这些结果仍为待复核的模型判定。最终模式以“生成有明确范围的报告”为结束条件，退出成功不代表全目标 complete=true；应查看 report_summary.json 的 target_complete/included_common_subflows/excluded_subflows。

## 全量 coarse subflow 批量入口

入口 `scripts/run_abcd_guideline_batch.py` 依赖现有 v4 工具及 JSON 失败块补跑脚本，无需覆盖公司客户端。先生成目录配置：

```bash
python scripts/run_abcd_guideline_batch.py --write-template abcd_guideline_batch.json
```

编辑 JSON 中的 `methods`：每种方法既可使用目录模板（`{flow}` 自动替换成 coarse subflow），也可用 `{"account_access": "一个独立目录", "order_issue": "另一个独立目录", ...}` 逐场景配置。**Ours 不要求有统一父目录**：新模板已列出全部十个 flow，分别把 `methods.ours.<flow>` 的 null 填成实际最终产物目录即可；已经 reuse 的场景可以保留 null。Ours 必须指向各场景选定的最终 arm，不能指向包含多个 ablation 的公共父目录。AWM、Trace2Skill 示例填入之前已知的运行目录，但仍需以服务器实际目录为准。`*` 通配符必须唯一匹配目录，不会自动挑最新文件夹。

`overrides` 支持 `{"account_access": {"ours": "单独的最终产物目录"}}` 覆盖模板。`reuse` 支持 `{"shipping_issue": "outputs/abcd_guideline_shipping_core_jsonfix_v1"}` 复用已经完成的完整三方法审计；它会逐字重验原始引用、冻结快照、参考条目和原始覆盖判定，不读取一个汇总百分比就当作完成。复用来源的 provider/model/参考协议须一致。要重做某个场景，从 reuse 中删除该条；其三个产物路径仍需有效。

先零 LLM 调用预检查，再后台运行：

```bash
python scripts/run_abcd_guideline_batch.py \
  --manifest abcd_guideline_batch.json --all --dry-run

nohup python -u scripts/run_abcd_guideline_batch.py \
  --manifest abcd_guideline_batch.json --all \
  --provider existing --workers 4 --qps 1.8 \
  --output-dir outputs/abcd_guideline_all_v1 \
  > outputs/abcd_guideline_all_v1.log 2>&1 < /dev/null &
```

预检查会扫描全部配置、列出实际文件、拒绝缺失/多重匹配/混合 Ours arm，并验证复用结果，任一出错则不会调用 LLM。正式运行时先检查 workflow 配置，每个 subflow 独立运行，场景内共享 workers/QPS；不会同时开启 10 组 workers。初始分块为约 1600 字符，失败场景自动调用细分补跑脚本。每个输出目录有操作系统文件锁，防止重复启动同一批次；失败退出后锁自动释放。成功请求有缓存，同一命令可断点续跑。

结果：

- `aggregate_summary.json`：全局严格覆盖、含条件单元覆盖、各场景/类别、整体及条件覆盖的 flow macro；单场景失败不会从分母删去，未完成全局分数为 null。
- `by_subflow.csv`：每个 subflow × 方法一行，包含分母、完整覆盖数、条件完整覆盖数及比率（0–1）。
- `assessments.json`：全部可用的逐项判定与源证据；标签仍为模型建议待复核。
- `progress.json`：已完成/失败场景、当前阶段、runner PID 和每个场景日志位置。
- `flows/<flow>/run/`：单场景原始参考、扫描和抽取产物；如需细分补跑，位于同级 `repaired/`。
- `logs/<flow>.log`：单场景 API 抽取及覆盖日志。

```bash
python scripts/run_abcd_guideline_batch.py \
  --output-dir outputs/abcd_guideline_all_v1 --status
```

默认处理全部 10 个 coarse subflow，也可用 `--subflows account_access,order_issue`。改变产物、模型、参考或核心脚本后应使用新输出目录；workers/QPS 可在停止后调整。当前全量参考为 359 个有效单元、其中 129 个含条件单元，三方法共 1077 项判定。已有 shipping_issue 若通过复用校验，可免去该场景的抽取和模型覆盖调用。跨场景运行与汇总已用模拟 LLM 响应验证，实际产物与 API 仍需在服务器预检查和执行。

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
