# Skill 规则抽取与审计准备

ABCD 的条件—动作 guideline 覆盖实验见 [ABCD_GUIDELINE_SKILL_AUDIT.md](ABCD_GUIDELINE_SKILL_AUDIT.md)。它包含单 coarse subflow 导出、官方参考、覆盖建议、复核和表格导出，区别于此处仅抽取规则/生成盲审模板。

这个工具读取已经生成的 skill 产物，不重新训练、不运行 agent，也不直接计算知识正确率。适配 Ours 的分离资源、AWM workflow、Trace2Skill 的 SKILL.md / references。默认只扫描 final，排除 offline、旧 checkpoint、原始预测和模型调用日志。

建议先在服务器选同一轮、同一数据比例、同一 family 的三个真实产物。下面以 `company_sgd_20pct_v2` 的 validation 和 Homes 为例；如果你的真实结果在其他目录，只更换三个 `--run` 路径。方法标签由命令明确指定，脚本不根据文件内容猜方法。

## 服务器：只扫描并打包，不调用 LLM

在项目根目录运行，以下均为 Bash：

```bash
cd /data1/liuxiangfeng/SkillMining

python scripts/extract_skill_knowledge.py scan \
  --run ours=outputs/company_sgd_20pct_v2/ours/validation \
  --run awm=outputs/company_sgd_20pct_v2/awm/validation \
  --run trace2skill=outputs/company_sgd_20pct_v2/trace2skill/validation \
  --families Homes \
  --output-dir outputs/skill_rule_sources_homes_v1 \
  --archive artifacts/skill_rule_sources_homes_v1.tar.gz
```

终端打印每个方法的文件数、字符数、chunk 数和识别到的 family。先看这些是否符合预期。完整扫描不限制 family 时去掉 `--families Homes`。新的扫描使用新的输出目录。

扫描只需要 Python 标准库。archive 仅包含 `inventory.json`、`documents.jsonl`、`chunks.jsonl`；不包含 API 配置、`.env`、SGD 全数据或原始 rollout。可以把这个小包带回本地，使用本地中转站抽取。

最终资源 allowlist 包括：`skill.md` / `SKILL.md`、`working_skill.md`、`action_rules.md`、`slot_policies.md`、`reference.md`、在线补充资源、`awm_workflow.txt`，以及 evolved_skill 内的 Markdown/text references。若只有 SGD JSON state，会读取明确的资源字符串，并保留 JSON pointer。原始 exemplar memory 不视为显式规则，不纳入默认规则抽取。

若 family 显示 unknown，检查该产物目录的 summary/state metadata。不要通过把预测或训练日志改名成 skill 来补全。若要比较初始化和最终结果，单独用 `--snapshot offline` 和新的扫描目录；不要混合版本。

## 服务器：直接用当前 llm.py 抽取

```bash
python -u scripts/extract_skill_knowledge.py extract \
  --source-dir outputs/skill_rule_sources_homes_v1 \
  --output-dir outputs/skill_rules_homes_v1 \
  --provider existing \
  --model qwen3.6-flash \
  --workers 1 --qps 1.8
```

默认复用公司当前 `llm.py`，不要求服务器存在 `llm_new.py`，也不覆盖公司接口或配置。若希望指定独立 workflow，可增加 `--workflow-id YOUR_WORKFLOW_ID`，前提是公司 llm.py 是 workflow-aware 实现。默认一个并发，脚本限制顶层调用起始速率；底层重试及共享限流仍由原客户端负责。

首次建议在新的输出目录加 `--max-chunks 3`，三个方法按 round-robin 交替取 chunk，验证后使用相同目录、去掉 `--max-chunks` 继续。已成功且来源一致的 chunk 会复用，失败 chunk 会重试。来源、模型、代码或抽取提示词变化时要求新输出目录。

完成状态只有在**全部扫描 chunk 成功**时才为 true。少量 smoke 成功不会伪装成全量完成。`--max-chunks` 是工具 smoke 上限，不是正式审计采样协议。

## 本地：从导出的源文件包抽取

把服务器生成的 `skill_rule_sources_homes_v1.tar.gz` 放在本地项目目录后：

```powershell
tar -xzf skill_rule_sources_homes_v1.tar.gz -C outputs

python scripts/extract_skill_knowledge.py extract `
  --source-dir outputs/skill_rule_sources_homes_v1 `
  --output-dir outputs/skill_rules_homes_local_v1 `
  --provider shubiaobiao --model qwen3.6-flash `
  --workers 2 --max-chunks 3
```

`shubiaobiao` 使用已有 `third_party/agent-workflow-memory/mind2web/.env`，不需要在命令里写密钥。去掉 `--max-chunks` 可补齐同一目录。

## 抽取的内容与追溯

每条规则包括：

- 类型：action selection、transition、slot binding、missing value、retry/recovery、resource use。
- 规则原意、action、适用条件、具体操作、slot、值来源和缺失值行为；原文没说的字段必须为 null。
- 来源类型：procedural_rule、executor_contract、example_only、uncertain，后面三类不自动计作挖掘出的过程知识。
- 原文逐字片段、文件 SHA256、文本 SHA256、字符范围、Markdown 行号或 JSON pointer。

抽取器不得从例子中新归纳规则，不得把含糊建议改写成精确条件，也不读取评测指标。源文本先显示局部行号，模型返回支持该规则的起止行，程序回填逐字原文，并校验字符范围；越界、空白或其他无效输出会整块重试，失败原始输出保留在 raw。这避免模型重抄 Markdown 引文时漏掉格式字符。旧格式的 literal quote 仍严格验证。必要时使用原保存图的 node/action alias 帮助解读 opaque node ID；alias 不是新增规则的证据。

字符范围针对 inventory 中的解码文本，而非原文件 byte offset；CRLF 在文本读取时归一为 LF。JSON 字符串引用记录 pointer 和字符串内范围，不伪造 JSON 物理行号。

`span_verified` **只证明源 skill 中存在引文**，不证明引文支持模型概括，更不证明训练轨迹支持该规则。`semantic_support_review`、`train_support_review` 初始均为 pending。

对于 example_only / uncertain 的非指令性观察，directive 可以为 null；它们不会因此被强行改写成可执行规则。procedural_rule / executor_contract 则必须有非空指令。

若仅修复程序校验逻辑、抽取提示词和模型没有变化，可以用新目录重新校验既有原始响应，避免重复调用：

```bash
python scripts/extract_skill_knowledge.py extract \
  --source-dir outputs/skill_rule_sources_homes_v1 \
  --output-dir outputs/skill_rules_homes_revalidated_v2 \
  --provider existing --model qwen3.6-flash \
  --reuse-responses-from outputs/skill_rules_homes_v1
```

程序会核验源文件、文本块、模型、provider、workflow 和提示词 hash。能通过新校验的缓存直接复用，保留旧目录和响应 hash；其余块才调用 LLM。不能借此把不同模型/提示词的产物混在一起。

输出：

- `rules.jsonl`：所有已校验抽取项；只进行保守机械去重，同一规则可有多条引用。
- `procedural_rules.jsonl`：被抽取器标为显式过程规则的子集，仍需人工复核分类。
- `summary.json`：文件处理进度和描述性计数，不是质量排行榜。
- `chunks/`、`raw/`、`extraction_config.json`：可恢复的处理记录、失败响应和冻结协议。
- `llm_usage_this_process.json`：当前执行进程的原客户端 usage；跨续跑的 top-level 尝试及粗略 token 估计另在 summary 中标明，不能混作精确 provider 计费。

完整性仅指扫描到的文本块都处理成功，不代表 LLM 已找到每一条规则；抽取召回率仍需独立抽检。当前自动扫描不解释可执行代码、非文本附件或 JSON/YAML 工具程序中的隐式知识。

## 生成盲审表

```bash
python scripts/extract_skill_knowledge.py review \
  --extraction-dir outputs/skill_rules_homes_v1 \
  --output-dir outputs/skill_rules_homes_review_v1 \
  --per-method 30 --seed 42
```

得到 `blind_review.csv` 与单独的 `private_key.json`。CSV 不包含真实方法标签或文件路径，显式方法名尽可能遮盖；写作风格仍可能暴露方法，不能宣称完全盲化。private_key 保留方法和全部原始出处，不交给审稿人/标注员。

人工列建议：

- text_support_label：supported / partial / unsupported，检查原文引文是否支持抽取概括。
- train_support_label：supported / contradicted / not_found / not_checked，需要另查 TRAIN/guideline，不能仅看 skill 自己的描述。
- specificity_label：concrete / generic / unclear，检查是否写清条件与操作；具体但错误的规则仍然是错误规则。
- review_notes：记录证据位置、例外、重复承载及分歧。

固定条数抽样用于规则精度审计，不能直接估算知识覆盖率。覆盖率需要另一份独立于各方法 skill 的问题/决策清单。不要用规则数量多、文本长或者没有附引用来直接判断知识质量。

## 本地验证

本地用已有 1/10 Homes 的真实旧产物做工具 smoke，非 1/5 SGD 正式审计结果：

```powershell
python scripts/extract_skill_knowledge.py scan --run ours=outputs/sgd_homes_10pct_20261018_153955/ours --run awm=outputs/sgd_homes_10pct_20261018_153955/awm --run trace2skill=outputs/sgd_homes_10pct_20261018_153955/trace2skill --families Homes --output-dir outputs/knowledge_inventory_homes_NEW
python scripts/extract_skill_knowledge.py extract --source-dir outputs/knowledge_inventory_homes_NEW --output-dir outputs/knowledge_extraction_homes_NEW --provider shubiaobiao --workers 2 --max-chunks 3
```
