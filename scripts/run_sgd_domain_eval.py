#!/usr/bin/env python3
"""Evaluate one SGD domain family with a causal action–slot protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
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
from eval_tod.sgd_errors import record_error, error_summary
from eval_tod.sgd_parallel import ordered_cases, require_value
from eval_tod.sgd_usage import configure_usage, persist_usage, set_phase


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
    skill_text: str = "", reference_text: str = "",
    selected_action_heads: list | None = None,
) -> tuple[dict, dict]:
    context = visible_context(dialogue, turn_index)[-12000:]
    schema = _schema_for_family(ontology, dialogue["services"])
    previous = (_api_history(dialogue, turn_index) or ["<START>"])[-1]
    graph_hint = (
        (frequency_graph or {}).get(previous, {}) if frequency_graph is not None else None
    )
    mined_resources = (
        ("\nMined SGD skill (training-derived; advisory):\n" + skill_text if skill_text else "")
        + ("\nRetrieved local residual rules:\n" + reference_text if reference_text else "")
    )
    if selected_action_heads is not None:
        mined_resources += ("\nFIXED selected action heads (intent, actual service ID): "
                            + json.dumps(selected_action_heads, ensure_ascii=False)
                            + "\nGround only these decisions; do not reselect action types. "
                              "Function names and named parameters are payloads.\n")
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
        + mined_resources
        + "\nDialogue so far:\n" + context
    )
    first_raw = sgd_chat_with_retry(first_prompt, model=model, temperature=0, call_tag="sgd_call_selection")
    try:
        call = canonical_call(_parse_object(first_raw).get("call"))
        first_error = ""
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        call, first_error = None, repr(exc)
    if selected_action_heads is not None and call is not None:
        if ["call", call["service"]] not in selected_action_heads:
            call = None
            first_error = "call_outside_fixed_action_selection"
    if selected_action_heads is not None and call is None:
        selected_calls = [head[1] for head in selected_action_heads if head[0] == "call"]
        if selected_calls:
            # Preserve the selected action when payload parsing/grounding
            # fails. Empty method is an invalid binding, not a second decision
            # to omit CALL; it cannot unlock a gold API observation.
            call = {"service": selected_calls[0], "method": "", "parameters": {}}
            first_error = first_error or "missing_selected_call_payload"
    if selected_action_heads is not None and call is None:
        selected_calls = [head[1] for head in selected_action_heads if head[0] == "call"]
        if selected_calls:
            # Preserve the selected action when payload parsing/grounding
            # fails. Empty method is an invalid binding, not a second decision
            # to omit CALL; it cannot unlock a gold API observation.
            call = {"service": selected_calls[0], "method": "", "parameters": {}}
            first_error = first_error or "missing_selected_call_payload"

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
        + mined_resources
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
    if selected_action_heads is not None:
        acts = [act for act in acts if act["intent"] != "call"
                and [act["intent"], act["domain"]] in selected_action_heads]
        observed_heads = {(act["intent"], act["domain"]) for act in acts}
        for intent, domain in selected_action_heads:
            if intent != "call" and (intent, domain) not in observed_heads:
                acts.append({"intent": intent, "domain": domain, "slot": ""})
                second_error = second_error or "missing_selected_act_payload"
        acts = canonical_acts(acts)
        observed_heads = {(act["intent"], act["domain"]) for act in acts}
        for intent, domain in selected_action_heads:
            if intent != "call" and (intent, domain) not in observed_heads:
                acts.append({"intent": intent, "domain": domain, "slot": ""})
                second_error = second_error or "missing_selected_act_payload"
        acts = canonical_acts(acts)
    return {"call": call, "acts": acts}, {
        "call_parse_error": first_error,
        "acts_parse_error": second_error,
        "observation_visible": observation_available,
        "api_observation": observation,
    }


def run_domain(args: argparse.Namespace) -> dict:
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("Completed summary already exists; use a new --output-dir or full-run --resume-run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_usage(args.output_dir)
    workflow_ids = [item.strip() for item in getattr(args, "workflow_ids", "").split(",") if item.strip()]
    workers = len(workflow_ids) if workflow_ids else getattr(args, "workers", 1)
    if workers > 1:
        import multiprocessing as mp
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("SGD within-domain workers require Linux/fork; use --workers 1 on this platform")
    if workflow_ids:
        os.environ["SKILLMINING_WORKFLOW_ID"] = workflow_ids[0]
    if args.method != "oracle":
        from llm import resolve_config
        resolve_config(model=args.model)
    if args.method == "frequency_graph":
        graph = build_frequency_graph(args.splits_dir, args.domain)
    else:
        graph = None
    with ZipFile(args.archive) as zipped:
        ontology = json.load(zipped.open("data/ontology.json"))
    awm_agent = None
    skill_agent = None
    if args.method == "awm":
        from eval_tod.sgd_agent import SGDAWMAgent
        awm_agent = SGDAWMAgent(
            family=args.domain, model=args.model, ontology=ontology,
            predictor=predict_policy,
            workflow_max_chars=args.awm_workflow_max_chars,
            exemplar_max_chars=args.awm_exemplar_max_chars,
        )
        awm_agent.parallel_workers, awm_agent.workflow_ids = workers, workflow_ids
        if args.awm_resource_dir:
            awm_agent.load(args.awm_resource_dir)
        else:
            set_phase("awm_training")
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
        awm_agent.save(args.output_dir)
    if args.method == "ours":
        from eval_tod.sgd_skill_agent import SGDSkillAgent
        from eval_tod.sgd_standard_skill_agent import SGDStandardSkillAgent
        agent_class = SGDStandardSkillAgent if args.ours_refinement_mode == "standard" else SGDSkillAgent
        skill_agent = agent_class(
            family=args.domain, model=args.model, ontology=ontology, predictor=predict_policy,
            candidate_limit=args.ours_candidate_limit,
            reference_max_chars=args.ours_reference_max_chars,
            max_outgoing_edges=args.ours_max_outgoing_edges,
            min_branch_support=args.ours_min_branch_support,
            discriminative_lambda=args.ours_discriminative_lambda,
            discriminative_clip=args.ours_discriminative_clip, seed=args.seed,
            mining_method=args.ours_mining_method,
            trace_coverage_target=args.ours_trace_coverage_target,
            corpus_fitness_target=args.ours_corpus_fitness_target,
            routing_complexity_weight=args.ours_routing_complexity_weight,
        )
        if args.ours_refinement_mode == "standard":
            skill_agent.runtime_diagnostics_dir = args.output_dir
        skill_agent.parallel_workers, skill_agent.workflow_ids = workers, workflow_ids
        if args.ours_resource_dir:
            skill_agent.load(args.ours_resource_dir)
            if skill_agent.stage == "refining":
                raise ValueError("Incomplete refinement checkpoint cannot be used as a final frozen resource")
            if args.ours_offline_only and skill_agent.stage != "offline":
                raise ValueError("--ours-offline-only requires an offline checkpoint, not refined/unknown resources")
        else:
            train = []
            for dialogue in iter_family_dialogues(args.splits_dir, args.domain, "train"):
                if args.ours_max_train is not None and len(train) >= args.ours_max_train:
                    break
                train.append(dialogue)
            skill_agent.mine(train, args.output_dir / "offline",
                             cases_per_node=args.ours_cases_per_node,
                             compile_batch_size=args.ours_compile_batch_size)
            if not args.ours_offline_only:
                skill_agent.refine(train, args.output_dir,
                                   selection_rate=args.ours_selection_rate,
                                   batch_size=args.ours_batch_size,
                                   max_batches=args.ours_max_batches,
                                   per_transition_cap=args.ours_per_transition_cap,
                                   analysis_batch_size=args.ours_analysis_batch_size)
        skill_agent.save(args.output_dir)
    if graph is not None:
        (args.output_dir / "frequency_graph.json").write_text(
            json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    gold_rows: list[dict] = []
    pred_rows: list[dict] = []
    dialogue_count = 0
    set_phase("testing")
    frozen_hash = skill_agent.resource_hash() if skill_agent is not None else None
    def predict_case(case):
        dialogue, turn_index = case
        if args.method == "oracle":
            return gold_policy(dialogue, turn_index), {"oracle": True}
        if awm_agent is not None:
            return awm_agent.predict(dialogue, turn_index)
        if skill_agent is not None:
            return skill_agent.predict(dialogue, turn_index)
        return predict_policy(dialogue, turn_index, model=args.model, ontology=ontology, frequency_graph=graph)

    test_dialogues = []
    for dialogue in iter_family_dialogues(args.splits_dir, args.domain, args.split):
        if args.max_dialogues is not None and len(test_dialogues) >= args.max_dialogues:
            break
        test_dialogues.append(dialogue)
    dialogue_count = len(test_dialogues)
    cases = [(d, index) for d in test_dialogues for index in d["target_indices"]["policy_turn_indices"]]
    predictions = ordered_cases(predict_case, cases, workers=workers, workflow_ids=workflow_ids,
        directory=args.output_dir, phase="testing")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for outcome, (dialogue, turn_index) in zip(predictions, cases):
            gold = gold_policy(dialogue, turn_index)
            try:
                predicted, diagnostic = require_value(outcome)
                score_policies([gold], [predicted])  # Malformed policies are case failures too.
            except Exception as exc:
                failure = record_error(args.output_dir, "testing", exc,
                                       dialogue_id=dialogue["dialogue_id"], turn_index=turn_index)
                predicted = {"call": None, "acts": [], "_execution_error": True}
                diagnostic = {"status": "error", "error": failure["error"],
                              "exception": failure["exception"]}
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
    usage = persist_usage()
    summary["llm_usage"] = usage
    summary["usage"] = usage["testing"]["total"]
    summary["mining_usage"] = usage["generation"]["total"]
    summary["total_usage"] = usage["total"]["total"]
    if awm_agent is not None:
        summary["workflow_lines"] = len(awm_agent.workflow)
        summary["memory_exemplars"] = len(awm_agent.memory)
    if skill_agent is not None:
        if skill_agent.resource_hash() != frozen_hash:
            raise AssertionError("Evaluation changed frozen skill resources")
        summary["skill_sha256"] = frozen_hash
        summary["skill_stage"] = skill_agent.stage
        summary["resource_training_errors"] = getattr(skill_agent, "state", {}).get("refinement_errors")
        summary["skill_protocol"] = getattr(skill_agent, "protocol", "sgd_backbone_online_prototype_v1")
        summary["graph_mining_method"] = skill_agent.graph.get("mining_method", "unknown")
        summary["skill_protocol"] = getattr(skill_agent, "protocol", "sgd_backbone_online_prototype_v1")
        summary["graph_nodes"] = len(skill_agent.graph["nodes"])
        summary["backbone_edges"] = len(skill_agent.graph["backbone"]["edges"])
        summary["residual_edges"] = len(skill_agent.graph["residual_edges"])
    summary["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    summary["case_errors"] = error_summary(args.output_dir)
    summary["execution_status"] = summary["case_errors"]["status"]
    summary["execution_parallelism"] = {"workers": workers, "workflow_ids": workflow_ids,
                                        "scope": "within_domain", "reflection_writer": "sequential"}
    if (summary.get("resource_training_errors") or {}).get("total", 0):
        summary["execution_status"] = "completed_with_errors"
    summary["data_manifest"] = {}
    for split in sorted({"train", args.split}):
        path = args.splits_dir / args.domain / f"{split}.jsonl.gz"
        if path.is_file():
            summary["data_manifest"][split] = {"path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    summary["ontology_sha256"] = hashlib.sha256(json.dumps(ontology, sort_keys=True).encode()).hexdigest()
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--method", choices=("oracle", "standard", "frequency_graph", "awm", "ours"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/eval/sgd/splits_by_domain"))
    parser.add_argument("--archive", type=Path, default=Path("data/eval/sgd/data.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-dialogues", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Independent cases within this domain")
    parser.add_argument("--workflow-ids", default="", help="One workflow ID per within-domain worker")
    parser.add_argument("--awm-resource-dir", type=Path,
                        help="Evaluate frozen AWM resources; omit to train online on SGD train")
    parser.add_argument("--awm-batch-size", type=int, default=20)
    parser.add_argument("--awm-max-train", type=int)
    parser.add_argument("--awm-max-batches", type=int)
    parser.add_argument("--awm-workflow-max-chars", type=int, default=8000)
    parser.add_argument("--awm-exemplar-max-chars", type=int, default=3000)
    add_ours_arguments(parser)
    args = parser.parse_args()
    validate_ours_arguments(parser, args)
    workflow_ids = [item.strip() for item in args.workflow_ids.split(",") if item.strip()]
    if args.workers < 1 or len(workflow_ids) != len(set(workflow_ids)):
        parser.error("Workers must be positive and workflow IDs unique")
    if workflow_ids and args.workers not in (1, len(workflow_ids)):
        parser.error("--workers must match the number of workflow IDs")
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
        "slot_exact_given_action", "turn_joint_ast", "call_gate_f1", "api_joint_ast",
        "mining_usage", "usage", "total_usage", "execution_status", "case_errors"
    )}, ensure_ascii=False))


def add_ours_arguments(parser, *, resource_option="--ours-resource-dir") -> None:
    """Shared defaults/options: the full runner forwards this exact contract."""
    parser.add_argument(resource_option, type=Path, help="Frozen SGD skill resources; skip mining/refine")
    parser.add_argument("--ours-offline-only", action="store_true")
    parser.add_argument("--ours-mining-method", choices=["trace_cover", "discriminative"], default="trace_cover",
                        help="Offline graph only; compiler, runtime and online refinement are unchanged")
    parser.add_argument("--ours-trace-coverage-target", type=float, default=.8)
    parser.add_argument("--ours-corpus-fitness-target", type=float, default=.95)
    parser.add_argument("--ours-routing-complexity-weight", type=float, default=2.)
    parser.add_argument("--ours-refinement-mode", choices=["standard", "prototype"], default="standard",
                        help="standard: original full policy-turn/reflection protocol; prototype: earlier simplified port")
    parser.add_argument("--ours-max-train", type=int)
    parser.add_argument("--ours-max-batches", type=int)
    parser.add_argument("--ours-batch-size", type=int, default=8)
    parser.add_argument("--ours-analysis-batch-size", type=int, default=8)
    parser.add_argument("--ours-selection-rate", type=float, default=.30,
                        help="Prototype only; standard uses every training policy turn")
    parser.add_argument("--ours-per-transition-cap", type=int, default=3)
    parser.add_argument("--ours-cases-per-node", type=int, default=3)
    parser.add_argument("--ours-compile-batch-size", type=int, default=8)
    parser.add_argument("--ours-candidate-limit", type=int, default=3)
    parser.add_argument("--ours-reference-max-chars", type=int, default=1800)
    parser.add_argument("--ours-max-outgoing-edges", type=int, default=3)
    parser.add_argument("--ours-min-branch-support", type=int, default=2)
    parser.add_argument("--ours-discriminative-lambda", type=float, default=1.)
    parser.add_argument("--ours-discriminative-clip", type=float, default=3.)
    parser.add_argument("--seed", type=int, default=42)


def validate_ours_arguments(parser, args) -> None:
    for key, value in vars(args).items():
        if key.startswith("ours_") and key not in {"ours_selection_rate", "ours_discriminative_lambda",
                                                  "ours_discriminative_clip", "ours_offline_only",
                                                  "ours_trace_coverage_target", "ours_corpus_fitness_target",
                                                  "ours_routing_complexity_weight"}:
            if isinstance(value, (int, float)) and value < 1:
                parser.error(f"--{key.replace('_', '-')} must be positive")
    if not 0 <= args.ours_selection_rate <= 1:
        parser.error("--ours-selection-rate must be in [0,1]")
    for key in ("ours_trace_coverage_target", "ours_corpus_fitness_target"):
        if not 0 < getattr(args, key) <= 1:
            parser.error(f"--{key.replace('_', '-')} must be in (0,1]")
    if not __import__("math").isfinite(args.ours_routing_complexity_weight) or args.ours_routing_complexity_weight < 0:
        parser.error("--ours-routing-complexity-weight must be finite and nonnegative")
    if not all(__import__("math").isfinite(value) and value >= 0
               for value in (args.ours_discriminative_lambda, args.ours_discriminative_clip)):
        parser.error("Discriminative weights must be finite and nonnegative")


if __name__ == "__main__":
    main()
