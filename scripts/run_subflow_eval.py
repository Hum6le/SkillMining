#!/usr/bin/env python3
r"""单个 Subflow 的 Skill Mining → 直接评估（无 AWM 训练）。

流程：
  1. 加载该 subflow 的 train/test 数据
  2. Skill Mining：session HG → vertex cover → subgraph → skill.md
  3. Seed Baseline：空 workflow，逐轮预测 + 评估
  4. Mined Skill：加载 skill.md 作为 workflow，逐轮预测 + 评估
  5. 对比 seed vs mined

用法：
  # 单个 subflow
  python scripts/run_subflow_eval.py --subflow recover_password

  # 所有 subflow（逐个跑）
  python scripts/run_subflow_eval.py --all --min-sessions 50

  # 仅跑 seed baseline（跳过 mining）
  python scripts/run_subflow_eval.py --subflow recover_password --skip-mining \
    --skill-path outputs/skills/recover_password/skill.md
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))
_SKILL_DIR = _PROJECT_ROOT / "skill_mining"
if str(_SKILL_DIR) in sys.path:
    sys.path.remove(str(_SKILL_DIR))
sys.path.insert(0, str(_SKILL_DIR))

from eval_tod.abcd.data import load_abcd_data
from eval_tod.abcd.split import extract_all_agent_turns
from eval_tod.cli import evaluate_text_records


def _llm_usage_functions():
    """Load usage hooks from the server-side ``llm.py`` when available."""
    import llm
    return (
        getattr(llm, "reset_usage_summary", lambda: None),
        getattr(llm, "get_usage_summary", lambda: {
            "schema_version": 0,
            "usage_available": False,
            "note": "llm.py has no shared usage tracker",
        }),
        getattr(llm, "write_usage_summary", None),
    )

SPLITS_DIR = Path("data/eval/abcd/splits")
MODEL = "deepseek-chat"

_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# The full parallel launcher sets this to a unique method/subflow directory.
# Retain the timestamped default for direct, standalone invocations.
OUT_DIR = Path(os.environ.get("ABCD_OUTPUT_DIR", f"outputs/subflow_eval_{_TIMESTAMP}"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "eval.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Core
# ═══════════════════════════════════════════════════════════════

def load_subflow_data(subflow: str) -> tuple[list, list]:
    """Load train + test conversations for one subflow."""
    train_path = SPLITS_DIR / subflow / "train.json"
    test_path = SPLITS_DIR / subflow / "test.json"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Split data not found for {subflow}. "
                                f"Run: python scripts/split_abcd_by_intent.py")
    train = json.loads(train_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    return train, test


def _evaluate_test_shard_worker(agent, shard: list, label: str, subflow: str,
                                workflow_id: str, result_queue,
                                response_log_dir: str | None = None) -> None:
    """Evaluate one test shard in an isolated process/workflow environment."""
    try:
        # fork inherits the parent's tracker state; isolate this shard so the
        # parent can merge exactly the calls made by this worker.
        import llm
        if response_log_dir:
            from eval_tod.response_logger import ResponseLogger
            # Each forked worker gets its own counter and directory.
            agent._response_logger = ResponseLogger(response_log_dir)
        reset = getattr(llm, "reset_usage_summary", None)
        if reset:
            reset()
        os.environ["SKILLMINING_WORKFLOW_ID"] = workflow_id
        rows = []
        for index, conversation in enumerate(shard, start=1):
            cid = conversation.get("convo_id", "?")
            print(f"  [{label}] worker_workflow={workflow_id} [{index}/{len(shard)}] "
                  f"convo={cid} {subflow}", flush=True)
            rows.extend(agent.predict_all_turns(
                conversation, predict_actions=True, verbose=False))
        selection_log = list(getattr(agent, "selection_log", []))
        usage = getattr(llm, "get_usage_summary", lambda: {})()
        result_queue.put({"ok": True, "rows": rows, "selection_log": selection_log,
                          "llm_usage": usage})
    except Exception as exc:
        try:
            usage = getattr(llm, "get_usage_summary", lambda: {})()
        except Exception:
            usage = {}
        result_queue.put({"ok": False, "error": repr(exc), "llm_usage": usage})


def mine_subflow_skill(
    subflow: str,
    train_convs: list,
    artifact_dir: Path | None = None,
) -> dict:
    """Mine skill from subflow training data: HG → vertex cover → subgraph."""
    from skill_mining.abcd_session_hg import (
        SessionHypergraph, greedy_vertex_cover, abcd_to_operator_results,
    )
    from skill_mining.subgraph_mining import (
        build_dag_from_operator_results, extract_subgraph,
        find_main_pathway, find_branch_points,
    )

    # Convert to operators → HG → vertex cover
    op_results = abcd_to_operator_results(train_convs)
    hg = SessionHypergraph.from_operator_results(op_results)
    selected, _ = greedy_vertex_cover(hg, rho=0.8, max_vertices=30)

    coverage = sum(1 for e in hg.hyperedges
                   if len(selected & e.vertices) >= 0.8 * e.size)
    n_edges = len(hg.hyperedges)

    # Build DAG → subgraph
    dag = build_dag_from_operator_results(op_results)
    sub = extract_subgraph(dag, selected)

    # Pathways + branches
    pathways = find_main_pathway(sub)
    branches = find_branch_points(sub)

    skill_info = {
        "selected_vertices": sorted(selected),
        "num_selected": len(selected),
        "coverage_pct": round(100 * coverage / max(n_edges, 1), 1),
        "num_sessions": len(train_convs),
    }

    subgraph = {
        **sub, "pathways": pathways, "branch_points": branches,
        "n_sessions": len(train_convs),
        "coverage_pct": skill_info["coverage_pct"],
    }

    # Generate reference.md (operator → dialogue snippets)
    from skill_mining.skill_writer import (
        _find_operator_snippets, build_reference_md,
        build_skill_md_from_subgraph,
    )
    from skill_mining.backbone_workflow_mining import sample_transition_cases
    operators = skill_info["selected_vertices"]
    op_snippets = _find_operator_snippets(train_convs, subflow, operators)
    transition_cases = sample_transition_cases(
        subflow, train_convs, max_cases_per_edge=3,
    )
    reference_md = build_reference_md(
        subflow, op_snippets, max_snippets_per_transition=3,
        transition_cases=transition_cases,
    )

    # Persist deterministic mining artifacts before optional LLM compilation.
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "reference.md").write_text(reference_md, encoding="utf-8")
        (artifact_dir / "subgraph.json").write_text(
            json.dumps(subgraph, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (artifact_dir / "operator_results.json").write_text(
            json.dumps(op_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Generate skill.md — LLM compile with fallback
    log.info("  Compiling skill.md via LLM...")
    skill_md = build_skill_md_from_subgraph(subflow, subgraph, op_snippets, use_llm=True)

    return {"skill_info": skill_info, "subgraph": subgraph,
            "operator_results": op_results,
            "reference_md": reference_md, "skill_md": skill_md}


def mine_subflow_skill_sequence(
    subflow: str,
    train_convs: list,
    min_edge_support: int = 2,
    min_edge_ratio: float = 0.1,
    max_nodes: int = 30,
    artifact_dir: Path | None = None,
) -> dict:
    """Mine skill from canonical action sequences."""
    from skill_mining.sequence_workflow_mining import mine_sequence_workflow
    from skill_mining.skill_writer import (
        _find_operator_snippets, build_reference_md,
        build_skill_md_from_subgraph,
    )
    from skill_mining.backbone_workflow_mining import sample_transition_cases

    mined = mine_sequence_workflow(
        subflow,
        train_convs,
        min_edge_support=min_edge_support,
        min_edge_ratio=min_edge_ratio,
        max_nodes=max_nodes,
    )

    operators = mined["skill_info"]["selected_vertices"]
    op_snippets = _find_operator_snippets(train_convs, subflow, operators)
    transition_cases = sample_transition_cases(
        subflow, train_convs, max_cases_per_edge=3,
    )
    reference_md = build_reference_md(
        subflow, op_snippets, max_snippets_per_transition=3,
        transition_cases=transition_cases,
    )

    # Sequence mining is deterministic; save its artifacts before the
    # optional LLM compiler so a slow/failed API call cannot hide them.
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "reference.md").write_text(reference_md, encoding="utf-8")
        (artifact_dir / "subgraph.json").write_text(
            json.dumps(mined["subgraph"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (artifact_dir / "operator_results.json").write_text(
            json.dumps(mined["operator_results"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    log.info("  Compiling sequence skill.md via LLM...")
    skill_md = build_skill_md_from_subgraph(
        subflow,
        mined["subgraph"],
        op_snippets,
        use_llm=True,
    )

    return {
        **mined,
        "reference_md": reference_md,
        "skill_md": skill_md,
    }


def mine_subflow_skill_backbone(
    subflow: str,
    train_convs: list,
    max_outgoing_edges: int = 3,
    min_branch_support: int = 2,
    transition_cases_per_edge: int = 2,
    coverage_aware: bool = False,
    coverage_lambda: float = 0.2,
    discriminative_lambda: float = 1.0,
    discriminative_clip: float = 3.0,
    compiler: str = "organized",
    artifact_dir: Path | None = None,
) -> dict:
    """Mine an all-action arborescence plus compact local transitions."""
    from skill_mining.backbone_workflow_mining import (
        mine_backbone_workflow, sample_transition_cases,
    )
    from skill_mining.skill_writer import (
        _find_operator_snippets, build_reference_md,
        build_backbone_seed_skill, build_skill_md_from_backbone,
        build_skill_md_from_unordered_backbone,
        induce_transition_rules,
    )

    # ``backbone`` and the historical ``backbone_coverage`` alias both use
    # the same discriminative session-aware arborescence.
    miner = mine_backbone_workflow
    miner_kwargs = {
        "max_outgoing_edges": max_outgoing_edges,
        "min_branch_support": min_branch_support,
        "discriminative_lambda": discriminative_lambda,
        "discriminative_clip": discriminative_clip,
    }
    mined = miner(subflow, train_convs, **miner_kwargs)
    cohort_info = mined["subgraph"].get("cohort_reweighting", {})
    log.info(
        "  Discriminative backbone: cohorts=%d lambda=%.3f clip=%.3f retained_coverage=%.1f%%",
        cohort_info.get("selected_k", 0), cohort_info.get("lambda", 0.0),
        cohort_info.get("clip", 0.0), mined["subgraph"].get("coverage_pct", 0.0),
    )
    operators = mined["skill_info"]["selected_vertices"]
    op_snippets = _find_operator_snippets(train_convs, subflow, operators)
    edge_cases = sample_transition_cases(
        subflow, train_convs, max_cases_per_edge=transition_cases_per_edge,
    )
    reference_md = build_reference_md(
        subflow, op_snippets, max_snippets_per_transition=3,
        transition_cases=edge_cases,
    )
    # Compile the stable backbone once before induction so the inducer can
    # place each local transition in the context of the complete skill.
    nodes = {node["id"]: node for node in mined["subgraph"].get("nodes", [])}
    required_actions = [
        nodes[node_id]["label"]
        for node_id in mined["subgraph"].get("backbone", {}).get("compilation_order", [])
        if node_id in nodes
    ]
    log.info("  Compiling backbone seed for global transition induction context...")
    seed_skill = build_backbone_seed_skill(
        subflow, mined["subgraph"], op_snippets, required_actions,
    )
    # Both compilers consume this same per-edge induction. The organized
    # compiler groups it around backbone decisions; the control only receives
    # a shuffled flat list of these induced transition cards.
    log.info("  Inducing joint transition guards for all observed edge types...")
    transition_induction = induce_transition_rules(
        subflow, mined["subgraph"], edge_cases,
        skill_context=seed_skill,
    )
    mined["subgraph"]["transition_induction"] = transition_induction

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "reference.md").write_text(reference_md, encoding="utf-8")
        (artifact_dir / "subgraph.json").write_text(
            json.dumps(mined["subgraph"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (artifact_dir / "operator_results.json").write_text(
            json.dumps(mined["operator_results"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (artifact_dir / "transition_cases.json").write_text(
            json.dumps(edge_cases, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    result = {**mined, "reference_md": reference_md, "compiler": compiler}
    if compiler in {"organized", "compare"}:
        log.info("  Compiling organized backbone skill.md via LLM...")
        result["skill_md"] = build_skill_md_from_backbone(
            subflow,
            mined["subgraph"],
            op_snippets,
            use_llm=True,
            transition_induction=transition_induction,
            seed_skill=seed_skill,
        )
    if compiler in {"unordered", "compare"}:
        log.info("  Compiling unordered node-edge control skill via LLM...")
        unordered_skill = build_skill_md_from_unordered_backbone(
            subflow,
            mined["subgraph"],
            op_snippets,
            transition_induction=transition_induction,
            use_llm=True,
        )
        if compiler == "unordered":
            result["skill_md"] = unordered_skill
        else:
            result["unordered_skill_md"] = unordered_skill
    if result.get("skill_md"):
        from skill_mining.skill_writer import materialize_progressive_disclosure
        compact, action_rules, slot_policies = materialize_progressive_disclosure(result["skill_md"])
        result.update({
            "skill_md": compact,
            "action_rules_md": action_rules,
            "slot_policies_md": slot_policies,
        })
    return result


def mine_subflow_semantic_router(
    subflow: str, train_convs: list, artifact_dir: Path,
    max_skills: int = 4, min_skill_sessions: int = 20,
    model: str = MODEL, mining_method: str = "backbone",
) -> dict:
    """Discover regions, then run the selected legacy/sequence/backbone miner."""
    from skill_mining.semantic_subflow import (
        discover_semantic_subflows, ground_skill_cards,
    )

    discovery = discover_semantic_subflows(
        subflow, train_convs, max_skills=max_skills,
        min_sessions=min_skill_sessions,
    )
    discovery = ground_skill_cards(
        discovery, train_convs, model=model,
        prompt_path=artifact_dir / "skill_router_card_induction_prompt.txt",
    )
    skill_root = artifact_dir / "skills"
    skill_root.mkdir(parents=True, exist_ok=True)
    by_id = {str(c.get("convo_id", "?")): c for c in train_convs}
    skills = {}
    for card in discovery.get("skill_cards", []):
        skill_id = card["skill_id"]
        members = [
            by_id[sid] for sid, assigned in discovery.get("session_assignments", {}).items()
            if (
                (assigned.get("skill_id") if isinstance(assigned, dict) else assigned) == skill_id
                and sid in by_id
            )
        ]
        if not members:
            continue
        log.info("  Compiling semantic %s from %d sessions", skill_id, len(members))
        local_name = f"{subflow}_{skill_id}"
        if mining_method in {"backbone", "backbone_coverage"}:
            result = mine_subflow_skill_backbone(
                local_name, members, max_outgoing_edges=3,
                min_branch_support=2, transition_cases_per_edge=2,
                coverage_aware=mining_method == "backbone_coverage",
                compiler="organized", artifact_dir=skill_root / skill_id,
            )
        elif mining_method == "sequence":
            result = mine_subflow_skill_sequence(
                local_name, members, min_edge_support=2,
                min_edge_ratio=0.1, max_nodes=30,
                artifact_dir=skill_root / skill_id,
            )
        else:
            result = mine_subflow_skill(
                local_name, members, artifact_dir=skill_root / skill_id,
            )
        skill_dir = skill_root / skill_id / local_name
        # The compiler receives the artifact directory as OUT_DIR/subflow.
        # Locate the actual files defensively for standalone and full-run paths.
        candidates = [skill_dir, skill_root / skill_id]
        actual = next((p for p in candidates if (p / "skill.md").exists()), candidates[0])
        skills[skill_id] = {
            "skill": (actual / "skill.md").read_text(encoding="utf-8") if (actual / "skill.md").exists() else result.get("skill_md", ""),
            "reference": (actual / "reference.md").read_text(encoding="utf-8") if (actual / "reference.md").exists() else result.get("reference_md", ""),
            "card": card,
        }
    discovery["compiled_skills"] = list(skills)
    discovery["skills_root"] = str(skill_root)
    (artifact_dir / "semantic_subflows.json").write_text(
        json.dumps(discovery, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"discovery": discovery, "skills": skills,
            "skill_info": {"num_selected": len(skills), "coverage_pct": 0}}


def evaluate_agent_on_subflow(
    agent, test_convs: list, label: str, subflow: str = "",
    save_dir: Path | None = None, eval_workflow_ids: list[str] | None = None,
) -> dict:
    """Run turn-level predictions + evaluation (with progress). Saves preds."""
    total = len(test_convs)
    all_turn_results: list[dict] = []
    workflow_ids = [str(value) for value in (eval_workflow_ids or []) if str(value)]
    if workflow_ids:
        import multiprocessing as mp
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError(
                "--eval-workflow-ids requires a fork-capable platform so each "
                "evaluation worker can inherit the compiled agent safely."
            )
        context = mp.get_context("fork")
        shards = [test_convs[index::len(workflow_ids)] for index in range(len(workflow_ids))]
        result_queue = context.Queue()
        workers = []
        for shard, workflow_id in zip(shards, workflow_ids):
            if not shard:
                continue
            worker = context.Process(
                target=_evaluate_test_shard_worker,
                args=(agent, shard, label, subflow, workflow_id, result_queue,
                      str(save_dir / "llm_responses" / f"{label}_{workflow_id}")
                      if save_dir else None),
                daemon=False,
            )
            worker.start()
            workers.append(worker)
        worker_results = []
        remaining = set(workers)
        while remaining:
            try:
                worker_results.append(result_queue.get(timeout=5.0))
                # A result is emitted exactly once by every live worker. The
                # process status check below catches crashes without output.
                remaining = {worker for worker in remaining if worker.is_alive()}
            except Exception:
                dead = [worker for worker in remaining if not worker.is_alive()]
                if dead:
                    raise RuntimeError(
                        "An evaluation worker exited without returning results: "
                        + ", ".join(str(worker.exitcode) for worker in dead)
                    )
        for worker in workers:
            worker.join()
        failures = [result["error"] for result in worker_results if not result.get("ok")]
        if failures:
            raise RuntimeError(f"Evaluation shard failed: {failures[0]}")
        for result in worker_results:
            all_turn_results.extend(result["rows"])
            if hasattr(agent, "selection_log"):
                agent.selection_log.extend(result.get("selection_log", []))
        worker_usage = [result.get("llm_usage", {}) for result in worker_results]
        print(f"  [{label}] Done: {total} convs across {len(workers)} evaluation workers "
              f"({len(all_turn_results)} turns)")
    else:
        worker_usage = []
        for index, conv in enumerate(test_convs, start=1):
            cid = conv.get("convo_id", "?")
            print(f"  [{label}] [{index}/{total}] convo={cid}  {subflow}", end="\r")
            all_turn_results.extend(agent.predict_all_turns(
                conv, predict_actions=True, verbose=False))
        print(f"  [{label}] Done: {total} convs, {len(all_turn_results)} turns")

    # Save predictions for error analysis
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / f"{label}_predictions.json").write_text(
            json.dumps(all_turn_results, indent=2, ensure_ascii=False),
            encoding="utf-8")
        react_traces = [
            {
                "convo_id": row.get("convo_id"),
                "turn_index": row.get("turn_index"),
                "agent_turn_num": row.get("agent_turn_num"),
                "subflow": row.get("subflow"),
                "react_trace": row.get("react_trace", []),
            }
            for row in all_turn_results
            if row.get("react_trace")
        ]
        if react_traces:
            (save_dir / f"{label}_react_traces.json").write_text(
                json.dumps(react_traces, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    turn_results = all_turn_results
    # Action targets are generated for AST only.  Keep them out of response
    # metrics, which are defined over agent utterance turns.
    text_turns = [
        r for r in turn_results if r.get("target_type", "utterance") == "utterance"
    ]
    preds = [r["prediction"] for r in text_turns]
    # Runtime prompts use original utterances, so compare generated text
    # against the aligned original agent utterance when it is available.
    refs = [r.get("reference_original") or r["reference"] for r in text_turns]
    text_result = evaluate_text_records(preds, refs)

    # AST from turn results
    from eval_tod.abcd.agent import (
        compute_ast_from_turn_results,
        turn_results_to_abcd_predictions,
    )
    from eval_tod.abcd.data import extract_ground_truth
    from eval_tod.abcd.metrics import evaluate_abcd

    ast_scores = compute_ast_from_turn_results(test_convs, turn_results)
    ast_mean = sum(s["ast_score"] for s in ast_scores) / max(len(ast_scores), 1)
    abcd_preds = turn_results_to_abcd_predictions(turn_results, test_convs)
    all_gt = [extract_ground_truth(conv) for conv in test_convs]
    abcd_eval = evaluate_abcd(all_gt, abcd_preds)

    if save_dir:
        records = []
        for pred in abcd_preds:
            records.append({
                "conversation_id": pred.conversation_id,
                "turns": [
                    {
                        "turn_index": t.turn_index,
                        "turn_type": t.turn_type,
                        "predicted_action": t.predicted_action,
                        "predicted_slots": t.predicted_slots,
                        "predicted_utterance_id": t.predicted_utterance_id,
                    }
                    for t in pred.turns
                ],
            })
        (save_dir / f"{label}_abcd_predictions.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    parsed_actions = sum(1 for r in turn_results if r.get("predicted_action"))
    direct_actions = sum(
        1 for r in turn_results
        if r.get("target_type") == "action" and r.get("predicted_action")
    )
    log.info(
        "  %s diagnostics: all_targets=%d, utterance_targets=%d, "
        "parsed_actions=%d, direct_action_predictions=%d, gt_action_turns=%d",
        label, len(turn_results), len(text_turns), parsed_actions, direct_actions,
        abcd_eval.ast.total_action_turns,
    )
    if abcd_eval.ast.total_action_turns and direct_actions == 0:
        log.warning(
            "  %s produced zero direct action predictions although the test "
            "set contains %d action turns. Check raw LLM outputs and parser.",
            label, abcd_eval.ast.total_action_turns,
        )

    output = {
        "label": label,
        "n_turns": len(preds),
        "text": {
            "bert_f1": round(text_result["bert_f1"], 4),
            "bleu_1": round(text_result["bleu_1"], 1),
            "bleu_4": round(text_result["bleu_4"], 1),
            "rouge_1": round(text_result["rouge_1"], 4),
            "rouge_2": round(text_result["rouge_2"], 4),
            "rouge_l": round(text_result["rouge_l"], 4),
            "meteor": round(text_result["meteor"], 4),
            "num_samples": len(preds),
        },
        "ast_mean": round(ast_mean, 4),
        "ast_cds": {
            "ast_joint": round(abcd_eval.ast.joint_accuracy, 4),
            "ast_action_name": round(abcd_eval.ast.action_name_accuracy, 4),
            "ast_slot_value": round(abcd_eval.ast.slot_value_accuracy, 4),
            "ast_slot_value_given_action": round(abcd_eval.ast.slot_accuracy_given_action, 4),
            "cds_overall": round(abcd_eval.cds.overall_cds, 4),
            "num_action_turns": abcd_eval.ast.total_action_turns,
            "num_action_correct_turns": abcd_eval.ast.action_correct_turns,
        },
    }
    if workflow_ids:
        output["_evaluation_worker_usage"] = worker_usage
    return output


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Subflow Skill Mining → Direct Evaluation")
    parser.add_argument("--subflow", default=None,
                        help="Single subflow name (e.g. recover_password)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all subflows in splits/")
    parser.add_argument("--min-sessions", type=int, default=50,
                        help="Min training sessions for --all")
    parser.add_argument("--skip-mining", action="store_true",
                        help="Skip mining, use pre-built skill.md")
    parser.add_argument("--skill-path", default=None,
                        help="Path to pre-built skill.md (for --skip-mining)")
    parser.add_argument("--skip-seed", action="store_true",
                        help="Skip seed baseline evaluation")
    parser.add_argument("--disable-reference-lookup", action="store_true",
                        help="Do not inject retrieved reference.md snippets into the mined agent prompt")
    parser.add_argument("--reference-top-k", type=int, default=3,
                        help="Number of reference.md operator sections to retrieve per turn")
    parser.add_argument("--reference-max-chars", type=int, default=1800,
                        help="Max characters of retrieved reference snippets injected per turn")
    parser.add_argument("--mining-method", choices=["backbone", "backbone_coverage", "semantic_router", "sequence", "legacy"],
                        default="legacy",
                        help="Skill mining method; backbone and legacy backbone_coverage both use discriminative session-aware arborescence")
    parser.add_argument("--backbone-max-outgoing-edges", type=int, default=3,
                        help="Max retained outgoing transitions per action for backbone mining")
    parser.add_argument("--backbone-min-branch-support", type=int, default=2,
                        help="Min support for non-backbone branch/retry transitions")
    parser.add_argument("--backbone-transition-cases", type=int, default=6,
                        help="Training cases sampled for each outgoing edge during joint continuation-mode induction")
    parser.add_argument("--backbone-coverage-lambda", type=float, default=0.2,
                        help="Deprecated compatibility option; ignored because backbone_coverage now aliases backbone")
    parser.add_argument("--backbone-discriminative-lambda", type=float, default=1.0,
                        help="Weight of cohort-specific log-odds in the discriminative backbone (default: 1.0)")
    parser.add_argument("--backbone-discriminative-clip", type=float, default=3.0,
                        help="Upper clip for cohort-specific log-odds bonus (default: 3.0)")
    parser.add_argument("--backbone-compiler", choices=["organized", "unordered", "compare"],
                        default="organized",
                        help="Backbone graph-to-skill compiler: organized (default), flat unordered control, or evaluate both on one mined graph")
    parser.add_argument("--semantic-max-skills", type=int, default=4,
                        help="Maximum latent skills discovered inside one 10-flow scene")
    parser.add_argument("--semantic-min-sessions", type=int, default=20,
                        help="Minimum training sessions supporting a latent skill")
    parser.add_argument("--subflow-discovery", action="store_true",
                        help="Discover latent session subflows before applying the selected mining method")
    parser.add_argument("--backbone-ablation-only", action="store_true",
                        help="Only compile/evaluate the unordered backbone ablation; skip the organized original")
    parser.add_argument("--sequence-min-edge-support", type=int, default=2,
                        help="Min transition support for sequence mining")
    parser.add_argument("--sequence-min-edge-ratio", type=float, default=0.1,
                        help="Min transition support ratio for sequence mining")
    parser.add_argument("--sequence-max-nodes", type=int, default=30,
                        help="Max canonical operator nodes for sequence mining")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument(
        "--eval-workflow-ids", default="",
        help="Comma-separated workflow IDs used only to shard test evaluation for one --subflow. "
             "Mining still runs once; each ID gets an independent evaluation subprocess.",
    )
    args = parser.parse_args()
    reset_usage, get_usage, write_usage = _llm_usage_functions()

    if args.backbone_ablation_only:
        if args.backbone_compiler == "compare":
            parser.error("--backbone-ablation-only cannot be combined with --backbone-compiler compare")
        args.backbone_compiler = "unordered"
    if args.skip_mining and args.backbone_compiler == "compare":
        parser.error("--backbone-compiler compare requires mining so both skills use the same graph evidence")

    # Determine subflows to process
    if args.all:
        if not SPLITS_DIR.exists():
            log.error("No splits found. Run: python scripts/split_abcd_by_intent.py")
            sys.exit(1)
        index_path = SPLITS_DIR / "INDEX.json"
        if not index_path.exists():
            log.error("No split index found: %s", index_path)
            sys.exit(1)
        split_index = json.loads(index_path.read_text(encoding="utf-8"))
        subflows = sorted(
            name for name in split_index
            if (SPLITS_DIR / name / "train.json").exists()
            and (SPLITS_DIR / name / "test.json").exists()
        )
        # Filter by min_sessions
        filtered = []
        for sf in subflows:
            train = json.loads((SPLITS_DIR / sf / "train.json").read_text(encoding="utf-8"))
            if len(train) >= args.min_sessions:
                filtered.append(sf)
        subflows = filtered
        log.info(f"Processing {len(subflows)} subflows (min_sessions={args.min_sessions})")
    elif args.subflow:
        subflows = [args.subflow]
    else:
        log.error("Need --subflow or --all")
        sys.exit(1)

    eval_workflow_ids = [value.strip() for value in args.eval_workflow_ids.split(",") if value.strip()]
    if eval_workflow_ids and (args.all or len(subflows) != 1):
        parser.error("--eval-workflow-ids is only supported when evaluating one --subflow")

    all_results = {}

    for subflow in subflows:
        # Each independently mined subflow gets an independent usage budget.
        # The full launcher starts one process per subflow; resetting here also
        # keeps direct ``--all`` runs correctly attributable.
        reset_usage()
        log.info(f"\n{'='*50}")
        log.info(f"Subflow: {subflow}")
        log.info(f"{'='*50}")

        # ── 1. Load data ──────────────────────────────────────
        train_convs, test_convs = load_subflow_data(subflow)
        if args.max_train:
            train_convs = train_convs[:args.max_train]
        if args.max_test:
            test_convs = test_convs[:args.max_test]
        log.info(f"  Train: {len(train_convs)}, Test: {len(test_convs)}")

        # ── 2. Mine skill ─────────────────────────────────────
        sf_out = OUT_DIR / subflow
        sf_out.mkdir(parents=True, exist_ok=True)
        semantic_bundle = None

        if args.skip_mining:
            skill_text = ""
            reference_text = ""
            action_rules_text = ""
            slot_policies_text = ""
            if args.skill_path:
                skill_path = Path(args.skill_path)
                skill_text = skill_path.read_text(encoding="utf-8")
                sibling_ref = skill_path.parent / "reference.md"
                if sibling_ref.exists():
                    reference_text = sibling_ref.read_text(encoding="utf-8")
                action_path = skill_path.parent / "action_rules.md"
                slot_path = skill_path.parent / "slot_policies.md"
                if action_path.exists():
                    action_rules_text = action_path.read_text(encoding="utf-8")
                if slot_path.exists():
                    slot_policies_text = slot_path.read_text(encoding="utf-8")
            else:
                sf_skill = sf_out / "skill.md"
                if sf_skill.exists():
                    skill_text = sf_skill.read_text(encoding="utf-8")
                sf_ref = sf_out / "reference.md"
                if sf_ref.exists():
                    reference_text = sf_ref.read_text(encoding="utf-8")
                action_path = sf_out / "action_rules.md"
                slot_path = sf_out / "slot_policies.md"
                if action_path.exists():
                    action_rules_text = action_path.read_text(encoding="utf-8")
                if slot_path.exists():
                    slot_policies_text = slot_path.read_text(encoding="utf-8")
            skill_info = {"selected_vertices": [], "coverage_pct": 0, "num_sessions": 0}
        else:
            if args.subflow_discovery or args.mining_method == "semantic_router":
                semantic_bundle = mine_subflow_semantic_router(
                    subflow, train_convs, sf_out,
                    max_skills=args.semantic_max_skills,
                    min_skill_sessions=args.semantic_min_sessions,
                    model=args.model,
                    mining_method=("backbone" if args.mining_method == "semantic_router" else args.mining_method),
                )
                discovery = semantic_bundle["discovery"]
                skill_info = semantic_bundle["skill_info"]
                skill_text = ""
                reference_text = ""
                action_rules_text = ""
                slot_policies_text = ""
                (sf_out / "skill_router_cards.md").write_text(
                    "\n\n".join(
                        "## " + str(card["skill_id"]) + ": " + str(card["name"]) + "\n"
                        + "Routing description: " + str(card.get("summary", "")) + "\n"
                        + "Customer goals: " + "; ".join(map(str, card.get("customer_goals", []))) + "\n"
                        + "Positive evidence: " + "; ".join(map(str, card.get("positive_evidence", []))) + "\n"
                        + "Negative evidence: " + "; ".join(map(str, card.get("negative_evidence", []))) + "\n"
                        + "Distinguish from other skills: " + json.dumps(card.get("distinguish_from", []), ensure_ascii=False) + "\n"
                        + "Typical outcome: " + str(card.get("typical_outcome", "unknown"))
                        for card in discovery.get("skill_cards", [])
                    ), encoding="utf-8")
                log.info("  Discovered %d semantic skills", len(semantic_bundle["skills"]))
                log.info(
                    "  Discovery objective=%.6f support=%.4f cohesion=%.4f overlap=%.4f; "
                    "removed=%s",
                    discovery.get("final_metrics", {}).get("objective", 0.0),
                    discovery.get("final_metrics", {}).get("mean_support", 0.0),
                    discovery.get("final_metrics", {}).get("mean_cohesion", 0.0),
                    discovery.get("final_metrics", {}).get("mean_overlap", 0.0),
                    discovery.get("removed_partition_nodes", []),
                )
                for item in discovery.get("objective_history", []):
                    log.info(
                        "    discovery iter=%s remove=%s node_support=%s objective=%.6f "
                        "previous=%.6f accepted=%s",
                        item.get("iteration"), item.get("removed_node"),
                        item.get("node_session_support"), item.get("objective", 0.0),
                        item.get("previous_objective", 0.0), item.get("accepted"),
                    )
            elif args.mining_method in {"backbone", "backbone_coverage"}:
                mined = mine_subflow_skill_backbone(
                    subflow,
                    train_convs,
                    max_outgoing_edges=args.backbone_max_outgoing_edges,
                    min_branch_support=args.backbone_min_branch_support,
                    transition_cases_per_edge=args.backbone_transition_cases,
                    coverage_aware=args.mining_method == "backbone_coverage",
                    coverage_lambda=args.backbone_coverage_lambda,
                    discriminative_lambda=args.backbone_discriminative_lambda,
                    discriminative_clip=args.backbone_discriminative_clip,
                    compiler=args.backbone_compiler,
                    artifact_dir=sf_out,
                )
            elif args.mining_method == "sequence":
                mined = mine_subflow_skill_sequence(
                    subflow,
                    train_convs,
                    min_edge_support=args.sequence_min_edge_support,
                    min_edge_ratio=args.sequence_min_edge_ratio,
                    max_nodes=args.sequence_max_nodes,
                    artifact_dir=sf_out,
                )
            else:
                mined = mine_subflow_skill(subflow, train_convs, artifact_dir=sf_out)
            if args.subflow_discovery or args.mining_method == "semantic_router":
                mined = None
            else:
                skill_info = mined["skill_info"]
                skill_text = mined.get("skill_md", "")
                reference_text = mined.get("reference_md", "")
                action_rules_text = mined.get("action_rules_md", "")
                slot_policies_text = mined.get("slot_policies_md", "")

            # Save skill.md + reference.md + subgraph
            if args.subflow_discovery or args.mining_method == "semantic_router":
                pass
            else:
                (sf_out / "skill.md").write_text(skill_text, encoding="utf-8")
            if not args.subflow_discovery and args.mining_method != "semantic_router" and args.backbone_compiler == "compare" and mined.get("unordered_skill_md"):
                (sf_out / "organized_skill.md").write_text(skill_text, encoding="utf-8")
                (sf_out / "unordered_skill.md").write_text(
                    mined["unordered_skill_md"], encoding="utf-8")
                (sf_out / "skill_compilation_ablation.json").write_text(
                    json.dumps({
                        "protocol": "same_backbone_graph_same_reference_same_transition_induction",
                        "organized_skill": "organized_skill.md",
                        "unordered_skill": "unordered_skill.md",
                        "unordered_control": (
                            "All mined nodes and the shared pre-induced transition cards are rendered "
                            "in a deterministic non-semantic order; no main path, edge priorities, "
                            "branch groups, or routing hierarchy are provided to its compiler."
                        ),
                    }, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            if not args.subflow_discovery and args.mining_method != "semantic_router":
                (sf_out / "reference.md").write_text(
                    mined.get("reference_md", ""), encoding="utf-8")
            if not args.subflow_discovery and args.mining_method != "semantic_router" and mined.get("action_rules_md"):
                (sf_out / "action_rules.md").write_text(
                    mined["action_rules_md"], encoding="utf-8")
            if not args.subflow_discovery and args.mining_method != "semantic_router" and mined.get("slot_policies_md"):
                (sf_out / "slot_policies.md").write_text(
                    mined["slot_policies_md"], encoding="utf-8")
            if not args.subflow_discovery and args.mining_method != "semantic_router":
                (sf_out / "subgraph.json").write_text(
                    json.dumps(mined["subgraph"], indent=2, ensure_ascii=False),
                    encoding="utf-8")
            if not args.subflow_discovery and args.mining_method != "semantic_router":
                n_snippets = sum(1 for v in mined.get("reference_md", "").split("\n")
                                if v.startswith("```text"))
                log.info(f"  Saved: skill.md ({len(skill_text.splitlines())} lines), "
                         f"reference.md ({n_snippets} snippets), subgraph.json")

        # ── 3. Seed Baseline (optional) ───────────────────────
        from eval_tod.abcd.agent import ABCDAgent
        from awm import WorkflowStore, MemoryStore

        # Keep mining/compilation usage separate from all test rollouts.
        generation_usage = get_usage()
        reset_usage()
        testing_usage_parts = []
        seed_result = None
        if not args.skip_seed:
            log.info("  Seed baseline...")
            seed_agent = ABCDAgent(
                model=args.model, workflow=WorkflowStore(), memory=MemoryStore(),
                expose_scenario_labels=False,
            )
            seed_result = evaluate_agent_on_subflow(
                seed_agent, test_convs, "seed", subflow, save_dir=sf_out,
                eval_workflow_ids=eval_workflow_ids)
            testing_usage_parts.extend(seed_result.get("_evaluation_worker_usage", []))
            log.info(f"    BERT={seed_result['text']['bert_f1']:.4f}  "
                     f"BLEU-4={seed_result['text']['bleu_4']:.1f}  "
                     f"ROUGE-L={seed_result['text']['rouge_l']:.4f}  "
                     f"METEOR={seed_result['text']['meteor']:.4f}  "
                     f"AST={seed_result['ast_cds']['ast_joint']:.4f}  "
                     f"Action={seed_result['ast_cds']['ast_action_name']:.4f}  "
                     f"Slot={seed_result['ast_cds']['ast_slot_value']:.4f}")

        # ── 4. Mined Skill ────────────────────────────────────
        log.info("  Mined skill evaluation...")
        if args.subflow_discovery or args.mining_method == "semantic_router":
            from eval_tod.abcd.router_agent import SemanticSkillRouterAgent
            from skill_mining.semantic_subflow import format_skill_cards
            router_skills = semantic_bundle["skills"]
            cards_prompt = format_skill_cards(semantic_bundle["discovery"])
            mined_agent = SemanticSkillRouterAgent(
                router_skills, cards_prompt, model=args.model,
            )
            (sf_out / "skill_router_cards_prompt.txt").write_text(cards_prompt, encoding="utf-8")
        else:
            wf = WorkflowStore()
            if skill_text:
                wf.update(skill_text)
            mined_agent = ABCDAgent(
                model=args.model,
                workflow=wf,
                memory=MemoryStore(),
                reference_text="" if args.disable_reference_lookup else reference_text,
                action_rules_text=action_rules_text,
                slot_policies_text=slot_policies_text,
                reference_top_k=args.reference_top_k,
                reference_max_chars=args.reference_max_chars,
                expose_scenario_labels=False,
            )
        mined_result = evaluate_agent_on_subflow(
            mined_agent, test_convs, "mined", subflow, save_dir=sf_out,
            eval_workflow_ids=eval_workflow_ids)
        testing_usage_parts.extend(mined_result.get("_evaluation_worker_usage", []))
        if (args.subflow_discovery or args.mining_method == "semantic_router") and hasattr(mined_agent, "selection_log"):
            (sf_out / "skill_router_selections.json").write_text(
                json.dumps(mined_agent.selection_log, indent=2, ensure_ascii=False),
                encoding="utf-8")
        log.info(f"    BERT={mined_result['text']['bert_f1']:.4f}  "
                 f"BLEU-4={mined_result['text']['bleu_4']:.1f}  "
                 f"ROUGE-L={mined_result['text']['rouge_l']:.4f}  "
                 f"METEOR={mined_result['text']['meteor']:.4f}  "
                 f"AST={mined_result['ast_cds']['ast_joint']:.4f}  "
                 f"Action={mined_result['ast_cds']['ast_action_name']:.4f}  "
                 f"Slot={mined_result['ast_cds']['ast_slot_value']:.4f}")

        unordered_result = None
        if not args.subflow_discovery and args.mining_method != "semantic_router" and args.backbone_compiler == "compare":
            unordered_text = mined.get("unordered_skill_md", "")
            log.info("  Unordered node-edge control evaluation...")
            unordered_wf = WorkflowStore()
            if unordered_text:
                unordered_wf.update(unordered_text)
            unordered_agent = ABCDAgent(
                model=args.model,
                workflow=unordered_wf,
                memory=MemoryStore(),
                reference_text="" if args.disable_reference_lookup else reference_text,
                action_rules_text=action_rules_text,
                slot_policies_text=slot_policies_text,
                reference_top_k=args.reference_top_k,
                reference_max_chars=args.reference_max_chars,
                expose_scenario_labels=False,
            )
            unordered_result = evaluate_agent_on_subflow(
                unordered_agent, test_convs, "unordered", subflow, save_dir=sf_out,
                eval_workflow_ids=eval_workflow_ids)
            testing_usage_parts.extend(unordered_result.get("_evaluation_worker_usage", []))
            log.info(f"    BERT={unordered_result['text']['bert_f1']:.4f}  "
                     f"BLEU-4={unordered_result['text']['bleu_4']:.1f}  "
                     f"ROUGE-L={unordered_result['text']['rouge_l']:.4f}  "
                     f"METEOR={unordered_result['text']['meteor']:.4f}  "
                     f"AST={unordered_result['ast_cds']['ast_joint']:.4f}  "
                     f"Action={unordered_result['ast_cds']['ast_action_name']:.4f}  "
                     f"Slot={unordered_result['ast_cds']['ast_slot_value']:.4f}")

        # ── 5. Delta ──────────────────────────────────────────
        delta = {}
        if seed_result:
            delta["bert_f1"] = round(mined_result['text']['bert_f1'] - seed_result['text']['bert_f1'], 4)
            delta["ast"] = round(mined_result['ast_cds']['ast_joint'] - seed_result['ast_cds']['ast_joint'], 4)
            delta["action"] = round(mined_result['ast_cds']['ast_action_name'] - seed_result['ast_cds']['ast_action_name'], 4)
            delta["slot"] = round(mined_result['ast_cds']['ast_slot_value'] - seed_result['ast_cds']['ast_slot_value'], 4)
            log.info(f"  Δ BERT={delta['bert_f1']:+.4f}  Δ AST={delta['ast']:+.4f}  "
                     f"ΔAction={delta['action']:+.4f}  ΔSlot={delta['slot']:+.4f}")

        from scripts.llm_usage_utils import merge_usage_summaries, split_usage_summary
        testing_usage = (
            merge_usage_summaries(*testing_usage_parts)
            if testing_usage_parts else get_usage()
        )
        phase_usage = split_usage_summary(generation_usage, testing_usage)
        all_results[subflow] = {
            "train_sessions": len(train_convs),
            "test_sessions": len(test_convs),
            "mining_method": args.mining_method,
            "subflow_discovery": args.subflow_discovery,
            "eval_workflow_ids": eval_workflow_ids,
            "backbone_compiler": args.backbone_compiler if args.mining_method in {"backbone", "backbone_coverage"} else None,
            "semantic_skill_count": len(semantic_bundle["skills"]) if semantic_bundle else 0,
            "skill_vertices": skill_info.get("num_selected", 0),
            "coverage_pct": skill_info.get("coverage_pct", 0),
            "seed": seed_result,
            "mined": mined_result,
            "unordered": unordered_result,
            "delta": delta,
            "llm_usage": phase_usage,
        }
        if write_usage is not None:
            (sf_out / "llm_usage.json").write_text(
                json.dumps(phase_usage, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        if unordered_result:
            all_results[subflow]["organization_delta"] = {
                "organized_minus_unordered": {
                    "bert_f1": round(mined_result["text"]["bert_f1"] - unordered_result["text"]["bert_f1"], 4),
                    "ast": round(mined_result["ast_cds"]["ast_joint"] - unordered_result["ast_cds"]["ast_joint"], 4),
                    "action": round(mined_result["ast_cds"]["ast_action_name"] - unordered_result["ast_cds"]["ast_action_name"], 4),
                    "slot": round(mined_result["ast_cds"]["ast_slot_value"] - unordered_result["ast_cds"]["ast_slot_value"], 4),
                },
            }

    # ── 6. Summary ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"SUMMARY")
    print(f"{'='*55}")
    has_seed = any(r.get("seed") for r in all_results.values())
    if has_seed:
        print(f"{'Subflow':35s} {'ΔBERT':>8s} {'ΔAST':>8s} {'ΔAct':>8s} {'ΔSlot':>8s} {'Seed':>8s} {'Mined':>8s}")
    else:
        print(f"{'Subflow':35s} {'BERT':>8s} {'BLEU-4':>8s} {'ROUGE-L':>8s} {'METEOR':>8s} {'AST':>8s} {'Action':>8s} {'Slot':>8s}")
    print("-" * 72)
    for sf, r in sorted(all_results.items(),
                        key=lambda x: -(x[1]["mined"]["text"]["bert_f1"])):
        m = r["mined"]["text"]["bert_f1"]
        if has_seed and r.get("seed"):
            d = r["delta"]
            s = r["seed"]["ast_cds"]["ast_joint"]
            mined_ast = r["mined"]["ast_cds"]["ast_joint"]
            print(f"{sf:35s} {d.get('bert_f1', 0):+.4f} {d.get('ast', 0):+.4f} "
                  f"{d.get('action', 0):+.4f} {d.get('slot', 0):+.4f} {s:.4f} {mined_ast:.4f}")
        else:
            print(f"{sf:35s} {m:.4f} {r['mined']['text']['bleu_4']:8.1f} "
                  f"{r['mined']['text']['rouge_l']:8.4f} "
                  f"{r['mined']['text']['meteor']:8.4f} "
                  f"{r['mined']['ast_cds']['ast_joint']:.4f} "
                  f"{r['mined']['ast_cds']['ast_action_name']:.4f} "
                  f"{r['mined']['ast_cds']['ast_slot_value']:.4f}")

    # Keep per-subflow results intact and also provide a weighted global view.
    # Text metrics are weighted by evaluated conversations; AST by action turns;
    # CDS by test conversations. This avoids treating tiny and large subflows
    # as equally representative.
    global_metrics = {}
    for phase in ("seed", "mined", "unordered"):
        rows = [row for row in all_results.values() if row.get(phase)]
        if not rows:
            continue
        text_weight = lambda row: max(int(row.get(phase, {}).get("text", {}).get("num_samples", 0)), 1)
        ast_weight = lambda row: max(int(row.get(phase, {}).get("ast_cds", {}).get("num_action_turns", 0)), 1)
        cds_weight = lambda row: max(int(row.get("test_sessions", 0)), 1)

        def weighted(metric, weight_fn):
            values = [
                (float(row[phase]["text" if metric in row[phase].get("text", {}) else "ast_cds"][metric]), weight_fn(row))
                for row in rows
                if metric in row[phase].get("text", {}) or metric in row[phase].get("ast_cds", {})
            ]
            return sum(value * weight for value, weight in values) / sum(weight for _, weight in values) if values else None

        metrics = {
            metric: weighted(metric, text_weight)
            for metric in ("bert_f1", "bleu_1", "bleu_4", "rouge_1", "rouge_2", "rouge_l", "meteor")
        }
        metrics.update({
            metric: weighted(metric, ast_weight)
            for metric in ("ast_joint", "ast_action_name", "ast_slot_value")
        })
        metrics["cds_overall"] = weighted("cds_overall", cds_weight)
        global_metrics[phase] = {
            "num_subflows": len(rows),
            "metrics": {key: round(value, 6) for key, value in metrics.items() if value is not None},
            "weights": {
                "text_samples": sum(text_weight(row) for row in rows),
                "action_turns": sum(ast_weight(row) for row in rows),
                "test_sessions": sum(cds_weight(row) for row in rows),
            },
        }

    summary_payload = dict(all_results)
    summary_payload["__global__"] = {
        "protocol": "independent_subflow_runs",
        "aggregate": global_metrics,
        "llm_usage_note": (
            "Per-subflow usage is stored in <subflow>/llm_usage.json. "
            "This process-level snapshot excludes calls from other subflows "
            "because usage is reset at each subflow boundary."
        ),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
