# Skill Baseline

当前 ABCD 实验设计、共享模块、各方法流程、运行命令、输出和可复现性检查见
[`EXPERIMENT_PIPELINE_OVERVIEW.md`](EXPERIMENT_PIPELINE_OVERVIEW.md).

全量 ABCD 实验可以使用统一 shell runner。它会进入 `skillmining310` conda
环境，设置 `HF_ENDPOINT=https://hf-mirror.com`，逐个 subflow 运行 AWM、
Trace2Skill 和 Graph Mining，并在最后生成加权全局汇总：

```bash
bash scripts/launch_full_abcd_experiments.sh
```

launcher 使用 `nohup` 后台运行，并立即打印 PID、日志路径和输出根目录；
实验结束时，主 runner 会在日志中打印每个方法的产物目录、全局汇总 JSON 和
manifest 文件。也可以直接运行前台版本：

```bash
bash scripts/run_full_abcd_experiments.sh
```

只运行一种方法或一个 subflow：

```bash
bash scripts/launch_full_abcd_experiments.sh --method awm
bash scripts/launch_full_abcd_experiments.sh --method expel
bash scripts/launch_full_abcd_experiments.sh --method trace2skill
bash scripts/launch_full_abcd_experiments.sh --subflow recover_username
```

Task-oriented Dialogue (ToD) agent evaluation and skill evolution framework, built
on top of the [Trace2Skill](https://github.com/Qwen-Applications/Trace2Skill)
methodology.

Given a seed skill (domain knowledge in markdown), the pipeline:
1. Runs a KB-backed ReAct agent on MultiWOZ dialogues
2. Evaluates predictions (Information Rate, Success Rate, LLM-as-a-Judge)
3. Analyzes failed trajectories to identify root causes
4. Evolves the skill via parallel MAP→REDUCE→TRANSLATE→APPLY
5. Re-evaluates with the evolved skill

## Quickstart
0
### 1. Clone and install dependencies

```bash
# Clone this repo
git clone https://github.com/<your-username>/Skill_Baseline.git
cd Skill_Baseline

# Clone dependencies
git clone https://github.com/Qwen-Applications/Trace2Skill.git
# (AWM and ExpeL are bundled or cloned from your own forks)

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Prepare data

Download MultiWOZ 2.1 and extract to `data/eval/multiwoz21/`:

```
data/eval/multiwoz21/
  dummy_data.json           # 10 sample dialogues (included in repo)
  splits/                   # Pre-split 1/10 sampled dataset (included in repo)
    all_train.json          #   846 dialogues
    all_val.json            #   114 dialogues
    all_test.json           #   105 dialogues
    scenario_*--*.json      #   Per-domain-combination scenario splits
    split_summary.json      #   Split statistics
  data/data/dialogues.json  # Full 10,438 dialogues (download separately)
  data/data/ontology.json   # Domain/slot ontology
  data/data/{domain}_db.json # Knowledge base files
```

The repository includes `dummy_data.json` (10 dialogues) and `splits/` (1,065
sampled dialogues) for quick testing. For the full dataset, download from
[MultiWOZ 2.1](https://github.com/budzianowski/multiwoz).

#### Dataset Splitting

To re-generate the splits with different sampling ratios or seeds:

```bash
# Default: 80/10/10 split by scenario, 1/10 sampling
python scripts/split_multiwoz.py

# Edit SAMPLE_FRAC at the top of the script to change sampling ratio
# SAMPLE_FRAC = 0.1   → 1/10  (~1,065 dialogues)
# SAMPLE_FRAC = 0.5   → 1/2   (~5,219 dialogues)
# SAMPLE_FRAC = 1.0   → full  (10,438 dialogues)
```

The script uses `split_by_scenario()` from `eval_tod.data` to assign each
dialogue to exactly one scenario (domain combination), then splits within
each scenario at 80/10/10. The output includes per-scenario files and
deduplicated `all_*.json` union files.

#### ABCD Dataset

Download the Action-Based Conversations Dataset (Chen et al., NAACL 2021):

```bash
# Clone the ABCD repository
git clone https://github.com/asappresearch/abcd.git /tmp/abcd_repo

# Copy the data files
mkdir -p data/eval/abcd/data
cp /tmp/abcd_repo/data/abcd_v1.1.json.gz data/eval/abcd/data/
cp /tmp/abcd_repo/data/guidelines.json data/eval/abcd/data/
cp /tmp/abcd_repo/data/ontology.json data/eval/abcd/data/
cp /tmp/abcd_repo/data/utterances.json data/eval/abcd/data/
cp /tmp/abcd_repo/data/kb.json data/eval/abcd/data/

# Unzip the main dataset
python -c "
import gzip, shutil
with gzip.open('data/eval/abcd/data/abcd_v1.1.json.gz', 'rb') as f_in:
    with open('data/eval/abcd/data/abcd_v1.1.json', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
"

# Clean up
rm /tmp/abcd_repo -rf
rm data/eval/abcd/data/abcd_v1.1.json.gz  # optional
```

Expected structure:

```
data/eval/abcd/data/
  abcd_v1.1.json       # ~116 MB, 10,042 dialogues (train/dev/test)
  guidelines.json       # Agent action flow definitions (10 flows, 55 subflows)
  ontology.json         # Intent/action/slot vocabulary
  utterances.json       # ~95K standard agent utterances pool
  kb.json               # Knowledge base tables
  images/               # Screenshots (optional)
  abcd_sample.json      # 10 sample dialogues (included in repo)
```

The dataset is pre-split into `train` (8,034), `dev` (1,004), and `test` (1,004)
— no additional splitting needed.

### 3. Configure API key

```bash
# The pipeline uses OpenAI-compatible APIs (DeepSeek by default)
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com"
```

### 4. Run the pipeline

```bash
# Quick test on 10 dummy dialogues
python -m Trace2Skill.pipeline_tod --smoke-test

# One-shot evaluation on test split
python -m Trace2Skill.pipeline_tod --split test --end 50

# Batch training with checkpointing
python -m Trace2Skill.pipeline_tod --batch-training \
  --split train --batch-size 50 --val-every 5 --checkpoint-every 10

# AWM agent with workflow induction
python -m Trace2Skill.pipeline.main --batch-training --split train --batch-size 20
```

Preferred local script entry:

```bash
python scripts/run_tod_pipeline.py --smoke-test
python scripts/run_tod_pipeline.py --split test --end 50
python scripts/run_tod_pipeline.py --batch-training --split train --batch-size 50
```

These entrypoints now share the same CLI implementation:

```bash
python scripts/run_tod_pipeline.py ...
python -m Trace2Skill.pipeline.main ...
python -m Trace2Skill.pipeline_tod ...
python pipeline_tod.py ...
```

```python
# Or step by step
from eval_tod.kb import MultiWOZKB
from eval_tod.agent_skill import SkillPreloadedAgent
from eval_tod.data import load_multiwoz21
from eval_tod import evaluate_predictions

# Load KB and dialogues
kb = MultiWOZKB("data/eval/multiwoz21/data/data")
dialogues = load_multiwoz21("data/eval/multiwoz21/dummy_data.json")

# Run agent with seed skill
agent = SkillPreloadedAgent(kb=kb, skills_dir="eval_tod/skills")
predictions = agent.generate_predictions(dialogues)

# Evaluate
result = evaluate_predictions(dialogues, predictions)
print(f"IR: {result['aggregate']['info_rate']:.4f}")
print(f"Success: {result['aggregate']['success_rate']:.4f}")
```

## Project Structure

```
Skill_Baseline/
  pipeline_tod.py              # Root wrapper (delegates to Trace2Skill)
  requirements.txt             # Python dependencies
  llm.py                       # LLM client factory + config resolution

  eval_tod/                    # ToD evaluation & agent module
    __init__.py                # Public API
    schemas.py                 # Dataclasses: Dialogue, Goal, Prediction
    data.py                    # Unified data loading + splitting (load_dataset, split_by_scenario, etc.)
    data_loader.py             # Legacy re-export from data.py
    utils.py                   # Slot normalization, value matching
    kb.py                      # MultiWOZ knowledge base (7 domains)
    metrics.py                 # IR, Success Rate, LLM Judge
    evaluate.py                # evaluate_predictions() + AbstractTodAgent interface
    cli.py                     # Command-line interface
    agent.py                   # Single-call LLM prediction agent
    agent_tool.py              # ReAct agent with query_db tool + trajectory logging
    agent_skill.py             # Skill-preloaded agent (SKILL.md injection)
    error_analysis.py          # LLM-based failure analysis agent
    response_logger.py         # Raw LLM prompt/response logger
    awm/                       # AWM (Agent Workflow Memory) adapter
      memory.py                #   MemoryStore + WorkflowStore
      induction.py             #   LLM workflow induction
      agent.py                 #   AWMAgent (mirrors eval_sample pattern)
    judge/                     # Multi-agent LLM Judge subpackage
      config.py                #   Scoring dimensions & judge definitions
      prompts.py               #   Judge/Combiner prompt templates
      llm_client.py            #   OpenAI-compatible client
      base.py                  #   JudgeAgent
      combiner.py              #   Combiner (synthesizes judge scores)
      judge_system.py          #   MultiAgentJudge orchestrator
    skills/tod/SKILL.md        # ToD seed skill

  Trace2Skill/pipeline/        # Skill evolution pipeline (modular)
    config.py                  #   PipelineConfig, EvolutionConfig, PipelineResult
    dataset_split.py           #   Data split + checkpoint utilities
    evaluate.py                #   _run_validation helper
    train.py                   #   _run_training_iteration, _run_oneshot_pipeline
    main.py                    #   run_pipeline orchestrator + CLI

  scripts/
    split_multiwoz.py          # Dataset splitting + 1/N sampling script

  data/eval/multiwoz21/        # MultiWOZ 2.1 dataset
    dummy_data.json            # 10 sample dialogues (included)
    splits/                    # 1/10 sampled scenario splits (included)
    data/data/                 # Full dataset + KB files (download separately)

  Trace2Skill/                 # Trace2Skill evolution engine (external)
  AWM/                         # Agent Workflow Memory (external)
  ExpeL/                       # ExpeL agent framework (external)
```

### ExpeL on ABCD

The official `ExpeL/` checkout targets interactive ALFWorld/WebShop-style
environments. `expel_adapter/` adapts its experiential rule-learning stage to
ABCD while keeping the shared full-turn action/slot runner and AST/CDS
evaluator. A dialogue is treated as successful when all of its action turns
have joint AST correctness. A run stores `expel_rules.json`, full turn
trajectories, grouped ABCD predictions, and `result.json` under
`outputs/expel_abcd_<timestamp>/`.

```bash
python scripts/run_expel_abcd.py \
  --subflow recover_username \
  --batch-size 20
```

ExpeL rules are injected as general insights only; AWM workflow and exemplar
memory are intentionally empty in this baseline.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Information Rate** | Slot-level precision: fraction of goal slots (inform + request) correctly predicted |
| **Success Rate** | Binary per-dialogue: ALL inform constraints + ALL requests + booking reference present |
| **LLM Judge** | Multi-agent LLM evaluation: 5 specialist judges + 1 combiner score dialogues on task_completion, slot_accuracy, dialogue_fluency, helpfulness, efficiency |

## Unified Evaluation CLI

Metric computation is now centralized behind `python -m eval_tod`, so the
older experiment scripts can call the same evaluation interface instead of each
keeping their own alignment logic.

### MultiWOZ ToD evaluation

```bash
python -m eval_tod tod \
  --dataset multiwoz21 \
  --data_path data/eval/multiwoz21 \
  --predictions preds.json
```

The legacy form still works:

```bash
python -m eval_tod.cli --dataset multiwoz21 --data_path data/eval/multiwoz21 --predictions preds.json
```

### Generic text evaluation

```bash
python -m eval_tod text \
  --predictions text_preds.json \
  --references text_refs.json
```

This returns the unified text metrics used throughout the repo:
- `BERTScore`
- `BLEU-1` / `BLEU-4`
- `ROUGE-1` / `ROUGE-2` / `ROUGE-L`

### ABCD evaluation

```bash
python -m eval_tod abcd \
  --data_path data/eval/abcd/data \
  --split test \
  --text-predictions text_predictions.json \
  --abcd-predictions abcd_predictions.json
```

For ABCD we evaluate two output channels separately under one interface:
- Natural-language response text -> `BERTScore`, `BLEU`, `ROUGE`
- Action-slot predictions -> `AST` and `CDS`

The helper layer used by older scripts is in [`eval_tod/cli.py`](/D:/paper/Skill_Baseline/eval_tod/cli.py):
- `evaluate_text_records(...)`
- `evaluate_abcd_bundle(...)`

This keeps metric definitions, prediction alignment, and summary formatting in
one place.

## ABCD Trace2Skill Pipeline

The original `Trace2Skill` pipeline in this repo is still the MultiWOZ-style
slot/success workflow. For ABCD we now keep a separate AST-driven pipeline:

```bash
python scripts/run_trace2skill_abcd.py --subflow recover_password
```

每次运行必须指定一个 subflow，例如：

```bash
python scripts/run_trace2skill_abcd.py --subflow recover_password
```

ABCD Trace2Skill now evolves the copied skill iteratively over outer training
batches.  Each batch reads the current `evolved_skill/SKILL.md`, applies any
new patch to disk, and the next batch continues from that updated skill:

```bash
python scripts/run_trace2skill_abcd.py \
  --subflow recover_password \
  --max-train 200 \
  --max-test 100 \
  --evolution-batch-size 25
```

It also supports explicit pre-split files, using the same style as
`run_full_experiment.py`:

```bash
python scripts/run_trace2skill_abcd.py \
  --subflow recover_password \
  --train-file data/eval/abcd/splits/recover_password/train.json \
  --test-file data/eval/abcd/splits/recover_password/test.json
```

This pipeline:

1. runs a seed ABCD agent with `predict_actions=True`
2. evaluates turn-level predictions with `AST` / `CDS`
3. treats `AST < 1.0` dialogues as failures
4. sends those failures into Trace2Skill-style error analysis and skill evolution
5. compares seed vs evolved skill on test `AST`

Seed skill path:

[`eval_tod/skills/abcd_trace2skill/SKILL.md`](/D:/paper/Skill_Baseline/eval_tod/skills/abcd_trace2skill/SKILL.md)

## ABCD Subflow Skill Mining

当前 ABCD 实验统一采用“每个 subflow 独立运行、最后统计全局结果”的协议。
AWM 和 Trace2Skill 必须通过 `--subflow` 指定一个 subflow，Graph Mining 的
`--all` 也只是依次运行相互独立的 subflow。多个运行的结果可用下面的脚本
进行加权汇总：

```bash
python scripts/aggregate_subflow_results.py \
  --runs outputs/awm_abcd_recover_username outputs/awm_abcd_recover_password \
  --output outputs/awm_global.json
```

详细的共享模块、训练流程、输出核验和指标定义见
[`EXPERIMENT_PIPELINE_OVERVIEW.md`](EXPERIMENT_PIPELINE_OVERVIEW.md)。

`scripts/run_subflow_eval.py` now supports two mining methods.  The default
`sequence` method canonicalizes action nodes by removing instance-specific slot
values, then mines a weighted action-sequence workflow:

```bash
python scripts/run_subflow_eval.py --subflow recover_password --mining-method sequence
```

The mined ABCD agent also retrieves compact snippets from the generated
`reference.md` on each turn and injects them into the prompt.  Disable this for
ablation with:

```bash
python scripts/run_subflow_eval.py \
  --subflow recover_password \
  --mining-method sequence \
  --disable-reference-lookup
```

The original hypergraph vertex-cover method is still available:

```bash
python scripts/run_subflow_eval.py --subflow recover_password --mining-method legacy
```

Useful sequence-mining knobs:

```bash
python scripts/run_subflow_eval.py --subflow recover_password \
  --sequence-min-edge-support 2 \
  --sequence-min-edge-ratio 0.1 \
  --sequence-max-nodes 30
```

## Agent Types

### `TodPredictionAgent` (agent.py)
Single-call LLM. Reads dialogue + goal → outputs structured predictions. Fast but
no KB access.

### `ToolBasedTodAgent` (agent_tool.py)
ReAct agent with `query_db(domain, constraints)` tool. Iteratively queries the
MultiWOZ knowledge base, reads results, then outputs predictions.

### `SkillPreloadedAgent` (agent_skill.py)
Extends `ToolBasedTodAgent` with skill injection. Loads `SKILL.md` from a skills
directory and prepends it to the system prompt. Used as the base agent in the
skill evolution pipeline.

### `AWMAgent` (awm/agent.py)
Agent Workflow Memory agent. Wraps `ToolBasedTodAgent` with two memory types:
- **WorkflowStore**: LLM-induced workflow patterns (accumulated across batches)
- **MemoryStore**: Concrete successful exemplars (retrieved by domain overlap)
After each batch, calls `induce_workflows()` to extract patterns from trajectories.

## Skill Evolution Pipeline

```
 Seed SKILL.md
      │
      ▼
 Stage 0: Load & split dataset
 Stage 1: Run SkillPreloadedAgent → predictions + trajectory logs
 Stage 2: Evaluate (IR, Success Rate)
 Stage 3: Error analysis on failed dialogues → analysis_report.md
 Stage 4: Parse reports → error_analysis_parsed.json
 Stage 5: MAP→REDUCE→TRANSLATE→APPLY → evolved SKILL.md  (Trace2Skill engine)
 Stage 6: Re-run with evolved skill → compare metrics
```

The core evolution engine (Stage 5) is reused from
[Trace2Skill](https://github.com/Qwen-Applications/Trace2Skill) and is
domain-agnostic. The domain-specific parts are:
- Seed skill content (ToD domain knowledge)
- Error analysis agent (ToD failure patterns)
- Agent trajectory logging

## Prediction Format

All agents output predictions in this JSON format:

```json
{
  "dialogue_id": "multiwoz21-train-0",
  "inform_slots": {
    "hotel": {"name": "Ashley Hotel", "price range": "cheap", "parking": "yes"}
  },
  "request_slots": {
    "hotel": ["address", "phone"]
  },
  "booking": {
    "hotel": {"reference": "7GAWK763"}
  }
}
```

## Citation

```bibtex
@misc{ni2026trace2skilldistilltrajectorylocallessons,
      title={Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills},
      author={Jingwei Ni and Yihao Liu and Xinpeng Liu and Yutao Sun and Mengyu Zhou and
              Pengyu Cheng and Dexin Wang and Erchao Zhao and Xiaoxi Jiang and Guanjun Jiang},
      year={2026},
      eprint={2603.25158},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
}
```
