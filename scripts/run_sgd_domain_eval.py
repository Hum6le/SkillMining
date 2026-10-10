#!/usr/bin/env python3
"""Evaluate one SGD domain family with a causal action–slot protocol."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_tod.sgd_adapter import (
    canonical_acts,
    canonical_call,
    gold_policy,
    iter_family_dialogues,
    metrics_from_counts,
    score_policies,
    visible_context,
)
from eval_tod.sgd_llm import sgd_chat_with_retry


def _parse_object(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        payload = json.loads(raw[start:end + 1]) if start >= 0 and end > start else {}
    return payload if isinstance(payload, dict) else {}


def _api_history(dialogue: dict, turn_index: int) -> list[str]:
    services = set(dialogue["services"])
    return [
        str(call["method"])
        for turn in dialogue["turns"][:turn_index]
        for service, call in (turn.get("service_call") or {}).items()
        if service in services
    ]


def build_frequency_graph(split_dir: Path, family: str) -> dict:
    graph: dict[str, Counter] = defaultdict(Counter)
    for dialogue in iter_family_dialogues(split_dir, family, "train"):
        history = _api_history(dialogue, len(dialogue["turns"]))
        previous = "<START>"
        for method in history:
            graph[previous][method] += 1
            previous = method
    return {source: dict(targets.most_common()) for source, targets in sorted(graph.items())}


def _schema_for_family(ontology: dict, services: list[str]) -> dict:
    return {
        service: {
            "description": ontology["domains"][service]["description"],
            "intents": [
                {key: intent[key] for key in ("name", "description", "required_slots", "optional_slots")
                 if key in intent}
                for intent in ontology["domains"][service]["active_intents"]
            ],
        }
        for service in services
    }


def predict_policy(
    dialogue: dict, turn_index: int, *, model: str, ontology: dict,
    frequency_graph: dict | None,
    workflow_text: str = "", exemplar_text: str = "",
) -> tuple[dict, dict]:
    context = visible_context(dialogue, turn_index)[-12000:]
    schema = _schema_for_family(ontology, dialogue["services"])
    previous = (_api_history(dialogue, turn_index) or ["<START>"])[-1]
    graph_hint = (
        (frequency_graph or {}).get(previous, {}) if frequency_graph is not None else None
    )
    first_prompt = (
        "Predict the next system turn's backend API decision for this SGD dialogue. "
        "Return JSON only: {\"call\": null} or "
        "{\"call\": {\"service\": \"...\", \"method\": \"...\", "
        "\"parameters\": {\"slot\": \"value\"}}}. "
        "Use no current-turn system utterance, gold dialogue acts, or DB results. "
        "The available service schemas are:\n"
        + json.dumps(schema, ensure_ascii=False)
        + ("\nObserved training API transitions after the prior call (advisory only):\n"
           + json.dumps(graph_hint, ensure_ascii=False) if graph_hint is not None else "")
        + ("\nLearned AWM workflow (training evidence; apply only when relevant):\n"
           + workflow_text if workflow_text else "")
        + ("\nVerified training examples (do not copy instance values):\n"
           + exemplar_text if exemplar_text else "")
        + "\nDialogue so far:\n" + context
    )
    first_raw = sgd_chat_with_retry(first_prompt, model=model, temperature=0, call_tag="sgd_call_selection")
    try:
        call = canonical_call(_parse_object(first_raw).get("call"))
        first_error = ""
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        call, first_error = None, repr(exc)

    gold = gold_policy(dialogue, turn_index)
    observation_available = call is not None and call == gold["call"]
    result = dialogue["turns"][turn_index].get("db_results") or {}
    observation = result.get(call["service"]) if observation_available else None
    # SGD stores the current API result with the system turn. It is exposed
    # only after an exactly correct predicted call, simulating the API reply.
    second_prompt = (
        "Predict the dialogue acts expressed by the next system response. "
        "Return JSON only: {\"acts\": [{\"intent\": \"request\", "
        "\"domain\": \"Service_1\", \"slot\": \"city\"}]}. "
        "An act may also have a value string. Multiple acts may occur in one "
        "turn; their order does not matter. Output only acts for the selected "
        "family and service-free acts. Do not generate response text. "
        "Available system act types: request, confirm, inform, inform_count, "
        "offer, offer_intent, notify_success, notify_failure, req_more, goodbye.\n"
        "Services: " + json.dumps(dialogue["services"], ensure_ascii=False)
        + "\nSelected API call: " + json.dumps(call, ensure_ascii=False)
        + "\nAPI observation (available only after a correct call): "
        + json.dumps(observation, ensure_ascii=False)[:4000]
        + ("\nLearned AWM workflow (training evidence; apply only when relevant):\n"
           + workflow_text if workflow_text else "")
        + ("\nVerified training examples (do not copy instance values):\n"
           + exemplar_text if exemplar_text else "")
        + "\nDialogue so far:\n" + context
    )
    second_raw = sgd_chat_with_retry(second_prompt, model=model, temperature=0, call_tag="sgd_dialogue_acts")
    try:
        acts = canonical_acts(_parse_object(second_raw).get("acts"))
        second_error = ""
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        acts, second_error = [], repr(exc)
    allowed_services = set(dialogue["services"])
    acts = [act for act in acts if act["domain"] in allowed_services or not act["domain"]]
    return {"call": call, "acts": acts}, {
        "call_parse_error": first_error,
        "acts_parse_error": second_error,
        "observation_visible": observation_available,
    }


def run_domain(args: argparse.Namespace) -> dict:
    if args.method != "oracle":
        from llm import get_usage_summary, reset_usage_summary, resolve_config
        resolve_config(model=args.model)
        reset_usage_summary()
    if args.method == "frequency_graph":
        graph = build_frequency_graph(args.splits_dir, args.domain)
    else:
        graph = None
    with ZipFile(args.archive) as zipped:
        ontology = json.load(zipped.open("data/ontology.json"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    awm_agent = None
    mining_usage = None
    if args.method == "awm":
        from eval_tod.sgd_agent import SGDAWMAgent
        awm_agent = SGDAWMAgent(
            family=args.domain, model=args.model, ontology=ontology,
            predictor=predict_policy,
            workflow_max_chars=args.awm_workflow_max_chars,
            exemplar_max_chars=args.awm_exemplar_max_chars,
        )
        if args.awm_resource_dir:
            awm_agent.load(args.awm_resource_dir)
            mining_usage = {"total": {"calls": 0, "input_tokens": 0, "output_tokens": 0}}
        else:
            train_log = []
            batch = []
            train_count = 0
            for dialogue in iter_family_dialogues(args.splits_dir, args.domain, "train"):
                if args.awm_max_train is not None and train_count >= args.awm_max_train:
                    break
                batch.append(dialogue)
                train_count += 1
                if len(batch) == args.awm_batch_size:
                    train_log.append(awm_agent.train_batch(
                        batch, batch_index=len(train_log) + 1,
                        trace_path=args.output_dir / "awm_training_turns.jsonl",
                    ))
                    awm_agent.save(args.output_dir)
                    batch = []
                    if args.awm_max_batches is not None and len(train_log) >= args.awm_max_batches:
                        break
            if batch and (args.awm_max_batches is None or len(train_log) < args.awm_max_batches):
                train_log.append(awm_agent.train_batch(
                    batch, batch_index=len(train_log) + 1,
                    trace_path=args.output_dir / "awm_training_turns.jsonl",
                ))
                awm_agent.save(args.output_dir)
            (args.output_dir / "awm_training.json").write_text(
                json.dumps({"domain": args.domain, "train_dialogues_seen": train_count,
                            "batches": train_log}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            mining_usage = get_usage_summary()
            reset_usage_summary()
        awm_agent.save(args.output_dir)
    if graph is not None:
        (args.output_dir / "frequency_graph.json").write_text(
            json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    gold_rows: list[dict] = []
    pred_rows: list[dict] = []
    dialogue_count = 0
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for dialogue in iter_family_dialogues(args.splits_dir, args.domain, args.split):
            if args.max_dialogues is not None and dialogue_count >= args.max_dialogues:
                break
            dialogue_count += 1
            for turn_index in dialogue["target_indices"]["policy_turn_indices"]:
                gold = gold_policy(dialogue, turn_index)
                if args.method == "oracle":
                    predicted, diagnostic = gold, {"oracle": True}
                elif awm_agent is not None:
                    predicted, diagnostic = awm_agent.predict(dialogue, turn_index)
                else:
                    predicted, diagnostic = predict_policy(
                        dialogue, turn_index, model=args.model, ontology=ontology,
                        frequency_graph=graph,
                    )
                gold_rows.append(gold)
                pred_rows.append(predicted)
                output.write(json.dumps({
                    "dialogue_id": dialogue["dialogue_id"],
                    "turn_index": turn_index,
                    "domain_family": args.domain,
                    "gold": gold,
                    "prediction": predicted,
                    "diagnostic": diagnostic,
                }, ensure_ascii=False) + "\n")
    summary = {
        "protocol": "sgd_train_seen_domain_policy_turns_v1",
        "method": args.method,
        "domain": args.domain,
        "split": args.split,
        "num_dialogues": dialogue_count,
        **score_policies(gold_rows, pred_rows),
    }
    if args.method != "oracle":
        from llm import write_usage_summary
        summary["usage"] = write_usage_summary(args.output_dir / "llm_usage.json")["total"]
    if awm_agent is not None:
        summary["mining_usage"] = mining_usage["total"]
        summary["workflow_lines"] = len(awm_agent.workflow)
        summary["memory_exemplars"] = len(awm_agent.memory)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--method", choices=("oracle", "standard", "frequency_graph", "awm"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/eval/sgd/splits_by_domain"))
    parser.add_argument("--archive", type=Path, default=Path("data/eval/sgd/data.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-dialogues", type=int)
    parser.add_argument("--awm-resource-dir", type=Path,
                        help="Evaluate frozen AWM resources; omit to train online on SGD train")
    parser.add_argument("--awm-batch-size", type=int, default=20)
    parser.add_argument("--awm-max-train", type=int)
    parser.add_argument("--awm-max-batches", type=int)
    parser.add_argument("--awm-workflow-max-chars", type=int, default=8000)
    parser.add_argument("--awm-exemplar-max-chars", type=int, default=3000)
    args = parser.parse_args()
    if args.max_dialogues is not None and args.max_dialogues < 1:
        parser.error("--max-dialogues must be positive")
    if any(value is not None and value < 1 for value in (
        args.awm_max_train, args.awm_max_batches, args.awm_batch_size,
        args.awm_workflow_max_chars, args.awm_exemplar_max_chars,
    )):
        parser.error("AWM limits and budgets must be positive")
    result = run_domain(args)
    print(json.dumps({key: result[key] for key in (
        "method", "domain", "split", "num_dialogues", "action_set_exact",
        "turn_joint_ast", "call_gate_f1", "api_joint_ast"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
