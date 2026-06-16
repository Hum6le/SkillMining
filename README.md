# Skill Baseline

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

### 1. Clone and install dependencies

```bash
# Clone this repo
git clone https://github.com/<your-username>/Skill_Baseline.git
cd Skill_Baseline

# Clone dependencies
git clone https://github.com/Qwen-Applications/Trace2Skill.git
# (AWM and ExpeL are bundled or cloned from your own forks)

# Install Python dependencies
pip install openai tqdm openpyxl requests diskcache
```

### 2. Prepare data

Download MultiWOZ 2.1 and extract to `data/eval/multiwoz21/`:

```
data/eval/multiwoz21/
  dummy_data.json          # 10 sample dialogues (included in repo)
  data/data/dialogues.json # Full 10,438 dialogues (download separately)
  data/data/ontology.json   # Domain/slot ontology
  data/data/{domain}_db.json # Knowledge base files
```

The repository includes `dummy_data.json` for quick testing. For the full
dataset, download from [MultiWOZ 2.1](https://github.com/budzianowski/multiwoz).

### 3. Configure API key

```bash
# The pipeline uses OpenAI-compatible APIs (DeepSeek by default)
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com"
```

### 4. Run the pipeline

```bash
# Quick test on 10 dummy dialogues
python pipeline_tod.py
```

```python
# Or step by step
from eval_tod.kb import MultiWOZKB
from eval_tod.agent_skill import SkillPreloadedAgent
from eval_tod.data_loader import load_multiwoz21
from eval_tod import evaluate

# Load KB and dialogues
kb = MultiWOZKB("data/eval/multiwoz21/data/data")
dialogues = load_multiwoz21("data/eval/multiwoz21/dummy_data.json")

# Run agent with seed skill
agent = SkillPreloadedAgent(kb=kb, skills_dir="eval_tod/skills")
predictions = agent.run_and_save(dialogues, "outputs/predictions.json")

# Evaluate
result = evaluate(
    dataset_name="multiwoz21",
    data_path="data/eval/multiwoz21/dummy_data.json",
    predictions_path="outputs/predictions.json",
)
print(f"IR: {result['aggregate']['info_rate']:.4f}")
print(f"Success: {result['aggregate']['success_rate']:.4f}")
```

## Project Structure

```
Skill_Baseline/
  pipeline_tod.py              # Full pipeline orchestrator (6 stages)

  eval_tod/                    # ToD evaluation & agent module
    __init__.py                # Public API
    schemas.py                 # Dataclasses: Dialogue, Goal, Prediction
    utils.py                   # Slot normalization, value matching
    data_loader.py             # MultiWOZ 2.1 data loader
    kb.py                      # MultiWOZ knowledge base (7 domains)
    metrics.py                 # IR, Success Rate, LLM Judge
    evaluate.py                # Evaluation orchestrator
    cli.py                     # Command-line interface
    agent.py                   # Single-call LLM prediction agent
    agent_tool.py              # ReAct agent with query_db tool + trajectory logging
    agent_skill.py             # Skill-preloaded agent (SKILL.md injection)
    error_analysis.py          # LLM-based failure analysis agent
    judge/                     # Multi-agent LLM Judge subpackage
      config.py                #   Scoring dimensions & judge definitions
      prompts.py               #   Judge/Combiner prompt templates
      llm_client.py            #   OpenAI-compatible client
      base.py                  #   JudgeAgent
      combiner.py              #   Combiner (synthesizes judge scores)
      judge_system.py          #   MultiAgentJudge orchestrator
    skills/tod/SKILL.md        # ToD seed skill

  data/eval/multiwoz21/        # MultiWOZ 2.1 dataset
    dummy_data.json            # 10 sample dialogues (included)
    data/data/                 # Full dataset + KB files (download separately)

  Trace2Skill/                 # Trace2Skill evolution engine (external)
  AWM/                         # Agent Workflow Memory (external)
  ExpeL/                       # ExpeL agent framework (external)
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Information Rate** | Slot-level precision: fraction of goal slots (inform + request) correctly predicted |
| **Success Rate** | Binary per-dialogue: ALL inform constraints + ALL requests + booking reference present |
| **LLM Judge** | Multi-agent LLM evaluation: 5 specialist judges + 1 combiner score dialogues on task_completion, slot_accuracy, dialogue_fluency, helpfulness, efficiency |

## Agent Types

### `TodPredictionAgent` (agent.py)
Single-call LLM. Reads dialogue + goal → outputs structured predictions. Fast but
no KB access. ~42% IR on dummy data.

### `ToolBasedTodAgent` (agent_tool.py)
ReAct agent with `query_db(domain, constraints)` tool. Iteratively queries the
MultiWOZ knowledge base, reads results, then outputs predictions. ~58% IR.

### `SkillPreloadedAgent` (agent_skill.py)
Extends `ToolBasedTodAgent` with skill injection. Loads `SKILL.md` from a skills
directory and prepends it to the system prompt. Used as the base agent in the
skill evolution pipeline. ~62% IR with seed skill.

## Skill Evolution Pipeline

```
 Seed SKILL.md
      │
      ▼
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
