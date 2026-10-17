#!/usr/bin/env python3
"""Evaluate the no-skill two-stage ReAct-style baseline on one ABCD subflow.

Full ten-subflow entry: launch_full_abcd_experiments.sh --method react.
No mining or online updating is performed. Completed-subflow resume validates
the prompt/schema protocol and exact test data, not just summary existence.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args):
    # Import only after setting the evaluator output path (it configures logs).
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["ABCD_OUTPUT_DIR"] = str(output)
    from eval_tod.abcd.react_base_agent import ReactBaseAgent
    from eval_tod.response_logger import ResponseLogger
    from scripts.run_subflow_eval import load_subflow_data, evaluate_agent_on_subflow
    from scripts.llm_usage_utils import (reset_usage, get_usage, merge_usage_summaries,
                                        split_usage_summary, _empty_usage_summary)

    _, test = load_subflow_data(args.subflow)
    if args.max_test is not None:
        test = test[:args.max_test]
    ids = [item.strip() for item in args.eval_workflow_ids.split(",") if item.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation workflow IDs")
    if len(ids) > 1 and "fork" not in mp.get_all_start_methods():
        raise ValueError("Multi-workflow evaluation requires fork (Linux server); use one ID here")
    agent = ReactBaseAgent(model=args.model, workflow_id=ids[0] if ids else args.workflow_id)
    protocol = agent.protocol()
    manifest = {"protocol": protocol, "subflow": args.subflow,
                "test_sha256": hashlib.sha256(json.dumps(test, ensure_ascii=False,
                    sort_keys=True).encode("utf-8")).hexdigest(),
                "test_sessions": len(test), "max_test": args.max_test,
                "skip_utterance_eval": args.skip_utterance_eval}
    manifest_path, summary_path = output / "protocol.json", output / "summary.json"
    if args.resume and summary_path.is_file():
        if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("ReAct resume data/protocol mismatch; use a fresh output directory")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("evaluation_complete") is True
                and summary.get("protocol") == protocol and isinstance(summary.get("final_test"), dict)):
            print(f"SKIP verified react/{args.subflow}: {summary_path}", flush=True)
            return summary
        raise ValueError("ReAct summary is incomplete or inconsistent; use a fresh output directory")
    if summary_path.exists():
        raise ValueError("ReAct summary exists; use --resume or a fresh output directory")
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("ReAct partial run has a different data/protocol manifest")
    _json(manifest_path, manifest)
    # New log session per attempt; old logs are never overwritten.
    agent._response_logger = ResponseLogger(str(output / "llm_responses" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    reset_usage()
    result = evaluate_agent_on_subflow(agent, test, "react", args.subflow, save_dir=output,
        skip_utterance_eval=args.skip_utterance_eval, eval_workflow_ids=ids if len(ids) > 1 else None)
    testing = merge_usage_summaries(get_usage(), *result.pop("_evaluation_worker_usage", []))
    generation = _empty_usage_summary()
    generation["note"] = "Known zero: no mining, training, refinement or resource compilation."
    # Testing availability is kept separate; zero mining does not imply known test tokens.
    usage = split_usage_summary(generation, testing)
    result["llm_usage"] = usage
    summary = {"config": {"method": "react", "subflow": args.subflow, "model": agent.model,
                          "skip_utterance_eval": args.skip_utterance_eval},
               "protocol": protocol, "evaluation_complete": True, "mining_performed": False,
               "data": {"test_sessions": len(test)}, "final_test": result, "llm_usage": usage,
               "routing": {"workflow_id": agent.workflow_id, "eval_workflow_ids": ids},
               "note": "Per-case errors are retained in predictions and metric denominators; inspect inference_errors."}
    _json(output / "llm_usage.json", usage)
    _json(summary_path, summary)
    print(json.dumps({"subflow": args.subflow, "ast_cds": result["ast_cds"],
                      "inference_errors": result.get("inference_errors"),
                      "summary": str(summary_path)}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subflow", required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--output-dir", default=os.environ.get("ABCD_OUTPUT_DIR",
        f"outputs/react_abcd_{datetime.now():%Y%m%d_%H%M%S}"))
    parser.add_argument("--workflow-id", default=os.environ.get("SKILLMINING_WORKFLOW_ID", ""))
    parser.add_argument("--eval-workflow-ids", default="")
    parser.add_argument("--max-test", type=int, help="Smoke test only; omitted means the full subflow test set")
    parser.add_argument("--skip-utterance-eval", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip verified completed subflow; no turn-level resume")
    args = parser.parse_args()
    if args.max_test is not None and args.max_test <= 0:
        parser.error("--max-test must be positive")
    run(args)


if __name__ == "__main__":
    main()
