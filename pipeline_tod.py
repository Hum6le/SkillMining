#!/usr/bin/env python3
"""Full Trace2Skill pipeline for ToD: seed skill → trajectories → analysis → evolved skill."""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ["PYTHONUTF8"] = "1"

# Ensure API keys are set (from env or fall back to AWM config)
if "OPENAI_API_KEY" not in os.environ:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "AWM"))
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
        os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
        os.environ["OPENAI_BASE_URL"] = DEEPSEEK_BASE_URL
    except ImportError:
        print("WARNING: Set OPENAI_API_KEY and OPENAI_BASE_URL env vars")
        print("  or clone AWM with config.py to the project root.")
        print("  git clone https://github.com/zorazrw/agent-workflow-memory.git AWM")

# ── Paths ──────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
TRACE2SKILL = BASE / "Trace2Skill"
EVAL_TOD = BASE / "eval_tod"

DATA_PATH = "data/eval/multiwoz21/dummy_data.json"
SPLIT = None  # "train", "validation", "test", or None for all
MODEL = "deepseek-chat"
START = 0
END = 10  # Number of dialogues to process
WORKERS_AGENT = 1  # Agent workers (keep 1 for LLM rate limiting)
WORKERS_ANALYSIS = 4  # Error analysis parallel workers
MAX_TURNS = 6
SEED = 41
LLM_JUDGE = True  # Enable multi-agent LLM Judge evaluation
LLM_JUDGE_SAMPLE = 5  # Max dialogues to judge (cost control)

OUT = BASE / "outputs" / "tod_pipeline"
SKILL_DIR = EVAL_TOD / "skills"  # parent dir containing skill subdirs
DB_DIR = "data/eval/multiwoz21/data/data"

# ── Helpers ────────────────────────────────────────────────────
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)

def run(cmd: list[str], desc: str, cwd=None) -> int:
    print(f"\n{'='*60}")
    print(f"[STAGE] {desc}")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"{'='*60}")
    r = subprocess.run(cmd, cwd=cwd or str(BASE))
    if r.returncode != 0:
        print(f"[WARN] Stage returned {r.returncode}: {desc}")
    return r.returncode

# ═══════════════════════════════════════════════════════════════
# STAGE 1: Run agent with seed skill
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 1: Generate predictions with seed skill")
print("=" * 60)

sys.path.insert(0, str(BASE))
from eval_tod.kb import MultiWOZKB
from eval_tod.agent_skill import SkillPreloadedAgent
from eval_tod.data_loader import load_multiwoz21

kb = MultiWOZKB(DB_DIR)
print(f"KB loaded: {kb.domains}")

dialogues = load_multiwoz21(DATA_PATH, split=SPLIT)
dialogues = dialogues[START:END]
print(f"Dialogues: {len(dialogues)}")

log_dir = str(OUT / "trajectories")
os.makedirs(log_dir, exist_ok=True)

agent = SkillPreloadedAgent(
    kb=kb,
    skills_dir=str(SKILL_DIR),
    model=MODEL,
    max_turns=MAX_TURNS,
    log_dir=log_dir,
)

preds = agent.run_and_save(
    dialogues=dialogues,
    output_path=str(OUT / "predictions_seed.json"),
)

# ═══════════════════════════════════════════════════════════════
# STAGE 2: Evaluate
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 2: Evaluate seed skill predictions")
print("=" * 60)

from eval_tod import evaluate as eval_func

eval_result = eval_func(
    dataset_name="multiwoz21",
    data_path=DATA_PATH,
    predictions_path=str(OUT / "predictions_seed.json"),
    split=SPLIT,
    output_path=str(OUT / "eval_seed.json"),
    llm_judge=LLM_JUDGE,
    llm_model=MODEL,
    llm_judge_sample_size=min(LLM_JUDGE_SAMPLE, len(dialogues)),
)

agg = eval_result["aggregate"]
print(f"  Seed IR: {agg['info_rate']:.4f}, Success: {agg['success_rate']:.4f}")
if LLM_JUDGE and eval_result.get("llm_judge"):
    js = eval_result["llm_judge"]
    if js:
        print(f"  Seed Judge: {', '.join(f'{k}={v:.2f}' for k, v in js.items())}")

# ═══════════════════════════════════════════════════════════════
# STAGE 3: Error Analysis
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 3: Error analysis on failed dialogues")
print("=" * 60)

from eval_tod.error_analysis import ErrorAnalyzer, build_failure_cases
from eval_tod.schemas import Prediction

# Build predictions objects from saved file
with open(str(OUT / "predictions_seed.json"), "r", encoding="utf-8") as f:
    pred_dicts = json.load(f)
pred_objs = [Prediction(**p) for p in pred_dicts]

failed_cases = build_failure_cases(dialogues, pred_objs, eval_result, log_dir=log_dir)
print(f"  Failed dialogues: {len(failed_cases)}/{len(dialogues)}")

if failed_cases:
    analyzer = ErrorAnalyzer(model=MODEL, workers=WORKERS_ANALYSIS)
    error_dir = str(OUT / "error_analysis")
    analyzer.analyze_batch(failed_cases, output_dir=error_dir)
else:
    print("  No failures to analyze — skill is perfect!")

# ═══════════════════════════════════════════════════════════════
# STAGE 4: Parse error analysis
# ═══════════════════════════════════════════════════════════════
if failed_cases:
    print("\n" + "=" * 60)
    print("STAGE 4: Parse error analysis reports")
    print("=" * 60)

    # Use Trace2Skill's parser
    sys.path.insert(0, str(TRACE2SKILL))
    run([
        sys.executable,
        str(TRACE2SKILL / "analysis" / "parse_error_analysis_outputs.py"),
        "--input_dir", str(OUT / "error_analysis"),
        "--output", str(OUT / "error_analysis_parsed.json"),
    ], "Parse error analysis reports")

# ═══════════════════════════════════════════════════════════════
# STAGE 5: Skill Evolution (REUSE Trace2Skill engine)
# ═══════════════════════════════════════════════════════════════
if failed_cases:
    print("\n" + "=" * 60)
    print("STAGE 5: Skill evolution")
    print("=" * 60)

    # Copy seed skill to evolution workdir
    evolved_skills_dir = str(OUT / "evolved_skills")
    os.makedirs(evolved_skills_dir, exist_ok=True)
    shutil.copytree(
        str(SKILL_DIR / "tod"),
        os.path.join(evolved_skills_dir, "tod"),
        dirs_exist_ok=True,
    )

    run([
        sys.executable, "-m", "skill_evolver.run_parallel_skill_evolution",
        "--input-json", str(OUT / "error_analysis_parsed.json"),
        "--skill-dir", os.path.join(evolved_skills_dir, "tod"),
        "--model", MODEL,
        "--verbose",
        "--batch-size", "1",
        "--changelog", str(OUT / "change.log"),
        "--save-intermediates",
        "--intermediates-dir", str(OUT / "intermediates"),
        "--max-workers", str(min(WORKERS_ANALYSIS, len(failed_cases))),
        "--prompt", "generic",
        "--patch-pipeline", "json",
        "--seed", str(SEED),
        "--parse-failure-dir", str(OUT / "parse_failures"),
    ], "Skill evolution", cwd=str(TRACE2SKILL))

# ═══════════════════════════════════════════════════════════════
# STAGE 6: Re-evaluate with evolved skill
# ═══════════════════════════════════════════════════════════════
if failed_cases:
    print("\n" + "=" * 60)
    print("STAGE 6: Evaluate with evolved skill")
    print("=" * 60)

    evolved_agent = SkillPreloadedAgent(
        kb=kb,
        skills_dir=evolved_skills_dir,
        model=MODEL,
        max_turns=MAX_TURNS,
        log_dir=str(OUT / "trajectories_evolved"),
    )

    evolved_preds = evolved_agent.run_and_save(
        dialogues=dialogues,
        output_path=str(OUT / "predictions_evolved.json"),
    )

    eval_evolved = eval_func(
        dataset_name="multiwoz21",
        data_path=DATA_PATH,
        predictions_path=str(OUT / "predictions_evolved.json"),
        split=SPLIT,
        output_path=str(OUT / "eval_evolved.json"),
        llm_judge=LLM_JUDGE,
        llm_model=MODEL,
        llm_judge_sample_size=min(LLM_JUDGE_SAMPLE, len(dialogues)),
    )

    agg_ev = eval_evolved["aggregate"]
    print(f"\n  Evolved IR:      {agg_ev['info_rate']:.4f}  (seed: {agg['info_rate']:.4f})")
    print(f"  Evolved Success: {agg_ev['success_rate']:.4f}  (seed: {agg['success_rate']:.4f})")
    if LLM_JUDGE and eval_evolved.get("llm_judge"):
        js = eval_evolved["llm_judge"]
        if js:
            print(f"  Evolved Judge:   {', '.join(f'{k}={v:.2f}' for k, v in js.items())}")

# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PIPELINE COMPLETE")
print(f"Output: {OUT}")
if failed_cases:
    print(f"Evolved skill: {evolved_skills_dir}/tod")
    print(f"Change log:    {OUT}/change.log")
print(f"{'='*60}")
