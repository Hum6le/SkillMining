"""Local overnight ABCD graph-selection experiment, with resumable dialogues.

Three baseline graphs, shared TRAIN-only action cards and references, existing
ABCDAgent action-selection then slot-grounding runtime, no online refinement.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_tod.sgd_local_config import configure, DEFAULT_ENV
from scripts.llm_usage_utils import merge_usage_summaries

FLOW = "shipping_issue"
ARMS = {"frequency": "frequency_k385", "support_lift": "support_lift",
        "process_heuristics": "pm_heuristics_net"}
BENCH = ROOT / "outputs/abcd_graph_ranking_20261025_v1"
SPLIT = ROOT / "data/eval/abcd/splits" / FLOW
GRAPH = ROOT / "outputs/sop_graph_benchmark_20261018_v2/graphs/trace_cover" / (FLOW + ".json")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.1 * (attempt + 1))


def action_indices(dialogue):
    return [i for i, t in enumerate(dialogue["delexed"])
            if len(t.get("targets", [])) >= 3 and t["targets"][1] == "take_action"]


def action_examples(train, actions, limit=4):
    """Coarse-flow input already filtered; do not require fine label == coarse."""
    from eval_tod.abcd.agent import _get_original_turn
    from eval_tod.abcd.action_schema import canonical_action_name, load_action_schema
    schema = load_action_schema()
    output = {action: [] for action in actions}
    for dialogue in train:
        for index in action_indices(dialogue):
            target = dialogue["delexed"][index]["targets"]
            action, suffix = canonical_action_name(target[2], schema.get("actions"))
            if action not in output or len(output[action]) >= limit:
                continue
            context = []
            for j in range(max(0, index - 8), index):
                speaker, text = _get_original_turn(dialogue, j, dialogue["delexed"][j])
                if text:
                    context.append(speaker + ": " + text)
            output[action].append({"dialogue_id": str(dialogue["convo_id"]), "turn_index": index,
                "context_before_action": "\n".join(context),
                "gold_training_slots": suffix + (target[3] if len(target) > 3 and isinstance(target[3], list) else [])})
    return output


def parse_cards(raw, expected):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    cards = data.get("cards", {})
    if set(cards) != set(expected):
        raise ValueError("Action-card response omitted or added actions")
    for action in expected:
        if not all(isinstance(cards[action].get(k), str) and len(cards[action][k].strip()) >= 15
                   for k in ("action_rule", "slot_policy")):
            raise ValueError("Empty action rule or slot policy: " + action)
    return cards


def render_resources(plan, nodes, cards, induction, edge_cases):
    action_rules = "# Action Rules\n\n" + "\n\n".join(
        f"#### `{a}`\n{cards[a]['action_rule']}" for a in sorted(cards)) + "\n"
    slot_policies = "# Slot Policies\n\n" + "\n\n".join(
        f"#### `{a}`\n{cards[a]['slot_policy']}" for a in sorted(cards)) + "\n"
    rules = {(u, r["target"]): r for u, rs in induction["rules_by_source"].items() for r in rs}
    reference = ["# Shared TRAIN transition evidence", ""]
    grouped = defaultdict(list)
    for u, v in plan["full_train_pairs"]:
        grouped[u].append(v)
    for u in sorted(grouped):
        reference += ["## " + nodes[u]["label"], ""]
        for v in sorted(grouped[u]):
            rule = rules[u, v]
            reference += [f"### {nodes[u]['label']} -> {nodes[v]['label']}",
                "- Condition: " + (rule.get("condition") or "Trigger underspecified by sampled evidence."),
                "- Status: " + rule["status"]]
            for case in edge_cases.get(u + " -> " + v, [])[:2]:
                reference += ["```text", str(case.get("context", ""))[-1200:], "```"]
            reference.append("")
    shared = {"action_rules.md": action_rules, "slot_policies.md": slot_policies,
              "reference.md": "\n".join(reference)}
    bundles = {}
    for arm, method in ARMS.items():
        ranked = plan["ranked"][method]
        lines = ["# Skill: shipping customer-service procedures", "",
            "Use the current dialogue to identify the request and completed actions. These are observed procedural continuations, not mandatory constraints.",
            "When a continuation is absent or uncertain, consult the shared reference. All action cards and residual evidence remain available.",
            "Select an action first, then bind its slots using its action card and the current dialogue. Example values are never defaults.",
            "", "## Action inventory", ", ".join("`" + a + "`" for a in sorted(cards)), "", "## Procedural continuations"]
        for u in sorted(ranked):
            lines += ["", "### After `" + nodes[u]["label"] + "`"]
            for v in ranked[u]:
                rule = rules.get((u, v), {})
                condition = rule.get("condition") or ("Observed continuation; retrieve reference to resolve its trigger."
                    if (u, v) in rules else "Process-miner generalized relation without direct TRAIN evidence; verify against current context.")
                lines.append("- `" + nodes[v]["label"] + "`: " + condition)
        bundles[arm] = {**shared, "skill.md": "\n".join(lines) + "\n"}
    return bundles


class APIGuard:
    def __init__(self, original, model, out, workers):
        self.original, self.model, self.out = original, model, out
        self.semaphore = threading.BoundedSemaphore(workers)
        self.lock = threading.Lock()
        self.calls = self.success = self.failed = self.inflight = 0
        self.last_success = time.monotonic()
        self.abort = threading.Event()
        self.started = time.monotonic()
        self.next_start = 0.

    def __call__(self, messages, **kwargs):
        # Force the selected local model even in legacy compiler defaults.
        kwargs["model"] = self.model
        if self.abort.is_set():
            raise RuntimeError("API circuit open after prolonged outage")
        for attempt in range(3):
            if attempt:
                time.sleep((2, 5)[attempt - 1])
            with self.semaphore:
                with self.lock:
                    delay = max(0., self.next_start - time.monotonic())
                    self.next_start = max(self.next_start, time.monotonic()) + .15
                    self.calls += 1
                    self.inflight += 1
                if delay:
                    time.sleep(delay)
                try:
                    text = self.original(messages, **kwargs)
                    if not str(text or "").strip():
                        raise RuntimeError("Empty or failed API response")
                    with self.lock:
                        self.success += 1
                        self.last_success = time.monotonic()
                    return text
                except Exception as exc:
                    with self.lock:
                        self.failed += 1
                    print(f"API retry {attempt+1}/3: {type(exc).__name__}", flush=True)
                    if time.monotonic() - self.last_success > 600:
                        self.abort.set()
                        raise RuntimeError("No successful API response for ten minutes") from exc
                    if attempt == 2:
                        raise
                finally:
                    with self.lock:
                        self.inflight -= 1

    def heartbeat(self, done):
        while not done.wait(15):
            with self.lock:
                state = {"updated_at": now(), "api_attempts": self.calls, "successful_requests": self.success,
                    "failed_requests": self.failed, "inflight": self.inflight,
                    "seconds_since_success": round(time.monotonic() - self.last_success, 1), "circuit_open": self.abort.is_set()}
            atomic(self.out / "api_heartbeat.json", state)


def prepare(args, config, guard):
    import llm
    from eval_tod.response_logger import ResponseLogger
    from skill_mining.backbone_workflow_mining import sample_transition_cases
    from skill_mining.skill_writer import induce_transition_rules
    out = args.output_dir
    common = out / "shared"
    common.mkdir(parents=True, exist_ok=True)
    train = read(SPLIT / "train.json")
    test = read(SPLIT / "test.json")
    if args.max_test:
        test = test[:args.max_test]
    plan_path = BENCH / "frozen_rankings.json"
    assert sha(plan_path) == read(BENCH / "selection_freeze.json")["sha256"]
    plan = read(plan_path)[FLOW]
    from skill_mining.sop_reference_benchmark import digest as graph_digest
    assert graph_digest(train) == plan["train_sha256"]
    graph = read(GRAPH)
    nodes = {n["id"]: n for n in graph["nodes"]}
    from_graph = {(e["source"], e["target"]) for es in graph["local_transitions"].values() for e in es}
    assert from_graph == {(u, v) for u, vs in plan["ranked"]["tracecover_original"].items() for v in vs}
    assert not {d["convo_id"] for d in train} & {d["convo_id"] for d in test}
    actions = sorted(n["label"] for n in nodes.values())
    examples = action_examples(train, actions)
    assert all(examples[a] for a in actions), "Every action must have actual TRAIN examples"
    inputs = {"protocol": "shipping-shared-resources-graph-selection-v1", "model": config["model"],
        "provider": config["provider"], "base_url": config["base_url"], "thinking": config["enable_thinking"],
        "train_hash": digest(train), "test_hash": digest(test), "graph_rankings_sha256": sha(plan_path),
        "tracecover_graph_sha256": sha(GRAPH), "arms": ARMS, "workers": args.workers,
        "workflow_max_chars": args.workflow_max_chars, "reference_top_k": 3, "reference_max_chars": 1800,
        "action_card_max_chars": 2200, "candidate_cards": 3, "action_examples_per_type": 4,
        "test_dialogues": len(test), "test_action_turns_per_arm": sum(len(action_indices(d)) for d in test),
        "online_refinement": False, "text_generation_eval": False,
        "runtime": "existing ABCDAgent; retrieval planning -> action selection -> fixed-action slot grounding",
        "initial_history": "existing ABCD gold-dialogue-prefix protocol; predicted action history maintained within each dialogue",
        "compiler": "neutral TRAIN-only per-action cards + common all-observed transition induction; no guideline inputs",
        "source_note": "TraceCover evaluation explicitly skipped; cached graph supplies TRAIN node contracts only. No verified compiled offline skill found locally; common baseline resource induction is new",
        "native_graph_sizes": {arm: sum(map(len, plan["ranked"][method].values())) for arm, method in ARMS.items()},
        "code_hashes": {name: sha(ROOT / name) for name in ["scripts/run_shipping_graph_agents.py", "eval_tod/abcd/agent.py", "skill_mining/skill_writer.py", "llm.py"]}}
    existing = read(out / "config.json")
    if existing and existing != inputs:
        raise ValueError("Cannot resume with changed data, code or protocol")
    atomic(out / "config.json", inputs)
    atomic(out / "progress.json", {"phase": "shared_resource_induction", "updated_at": now(), "pid": os.getpid(), "config": inputs})
    cards = {}
    batches = [actions[i:i+5] for i in range(0, len(actions), 5)]
    node_by_action = {n["label"]: n for n in nodes.values()}
    def compile_cards(number, batch):
        path = common / f"cards_batch_{number}.json"
        saved = read(path)
        if saved:
            parse_cards(json.dumps({"cards": saved}), batch)
            return saved
        evidence = {a: {"observed_slot_contract": node_by_action[a].get("slot_contract", {}), "examples": examples[a]} for a in batch}
        prompt = ("Infer reusable per-action procedure and ordered slot-binding guidance from TRAIN examples. "
            "Do not infer a preferred global route or use any graph selection. Explain when each action is appropriate, "
            "slot source/order, zero-slot cases, omitted/empty values, reuse and changed entities. Preserve observed empty-slot behavior; "
            "do not invent a required value. Context is instance data; never make example values into defaults. "
            "Write concise but concrete rules, no more than 160 words per field. Return ONLY JSON: "
            '{"cards":{"ACTION":{"action_rule":"...","slot_policy":"..."}}}. '
            "Return exactly these actions: " + json.dumps(batch) + "\n" + json.dumps(evidence, ensure_ascii=False))
        logger = ResponseLogger(common / "llm_responses" / f"action_batch_{number}")
        for attempt in range(3):
            try:
                raw = llm.chat(prompt, temperature=0., response_logger=logger, call_tag="shared_action_slot_induction")
                result = parse_cards(raw, batch)
                atomic(path, result)
                print(f"COMPILED action cards {number+1}/{len(batches)}", flush=True)
                return result
            except Exception:
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")
    with ThreadPoolExecutor(max_workers=min(4, args.workers)) as pool:
        futures = [pool.submit(compile_cards, i, batch) for i, batch in enumerate(batches)]
        for future in as_completed(futures):
            cards.update(future.result())
    atomic(common / "action_cards.json", cards)
    edge_cases = sample_transition_cases(FLOW, train, max_cases_per_edge=2)
    atomic(common / "transition_cases.json", edge_cases)
    induction = read(common / "transition_induction.json")
    if not induction:
        # This source-neutral context does not expose any arm's selected edges.
        context = "Shared action guidance, no preferred route:\n" + "\n".join(a + ": " + cards[a]["action_rule"] for a in actions)
        induction = induce_transition_rules(FLOW, graph, edge_cases, skill_context=context, max_sources_per_call=4)
        atomic(common / "transition_induction.json", induction)
    bundles = render_resources(plan, nodes, cards, induction, edge_cases)
    hashes = {}
    for arm, resources in bundles.items():
        directory = out / arm / "resources"
        directory.mkdir(parents=True, exist_ok=True)
        hashes[arm] = {}
        for name, text in resources.items():
            path = directory / name
            path.write_text(text, encoding="utf-8")
            hashes[arm][name] = sha(path)
    for name in ("action_rules.md", "slot_policies.md", "reference.md"):
        assert len({row[name] for row in hashes.values()}) == 1
    manifest = {"resources_sha256": hashes, "train_action_examples_hash": digest(examples),
        "induction_fallback_sources": induction.get("fallback_sources", []), "rules": sum(map(len, induction["rules_by_source"].values())),
        "selection_graphs_frozen_before_new_compilation": True, "shared_files_identical": True}
    old_manifest = read(common / "manifest.json")
    if old_manifest and old_manifest != manifest:
        raise ValueError("Shared resources changed during resume")
    atomic(common / "manifest.json", manifest)
    return test, bundles, inputs, manifest


def validate_rows(dialogue, rows):
    expected = action_indices(dialogue)
    if [r["turn_index"] for r in rows] != expected:
        raise ValueError("Missing, duplicate or misaligned action targets")
    for row in rows:
        if row.get("errors"):
            raise RuntimeError("Inference error record in prediction")
        if row.get("target_type") != "action":
            raise ValueError("Unexpected utterance target in AST-only evaluation")
    return len(rows)


def evaluate_case(args, arm, dialogue, resources, resource_hash):
    import llm
    from awm import WorkflowStore, MemoryStore
    from eval_tod.abcd.agent import ABCDAgent
    from eval_tod.response_logger import ResponseLogger
    case_id = str(dialogue["convo_id"])
    path = args.output_dir / arm / "cases" / (case_id + ".json")
    previous = read(path)
    if previous and previous.get("status") == "complete":
        if previous["resource_hash"] != resource_hash or previous["dialogue_hash"] != digest(dialogue):
            raise ValueError("Checkpoint resource/data mismatch")
        validate_rows(dialogue, previous["rows"])
        return previous
    prior_attempts = previous.get("attempts", []) if previous else []
    with llm.usage_scope():
        start = time.monotonic()
        for attempt in range(2):
            try:
                workflow = WorkflowStore()
                workflow.update(resources["skill.md"])
                logger = ResponseLogger(args.output_dir / arm / "responses" / case_id / f"attempt_{len(prior_attempts)}")
                agent = ABCDAgent(model=args.model, api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                    workflow=workflow, memory=MemoryStore(), reference_text=resources["reference.md"],
                    action_rules_text=resources["action_rules.md"], slot_policies_text=resources["slot_policies.md"],
                    workflow_max_chars=args.workflow_max_chars, reference_top_k=3, reference_max_chars=1800,
                    action_selection_candidate_limit=3, expose_scenario_labels=False, delay=0., response_logger=logger)
                assert len(agent.action_cards) == 23
                rows = agent.predict_all_turns(dialogue, predict_actions=True, action_only=True)
                turns = validate_rows(dialogue, rows)
                result = {"status": "complete", "arm": arm, "dialogue_id": case_id,
                    "dialogue_hash": digest(dialogue), "resource_hash": resource_hash, "turns": turns,
                    "elapsed_seconds": time.monotonic() - start, "completed_at": now(), "rows": rows,
                    "attempts": prior_attempts, "llm_usage": merge_usage_summaries(previous.get("llm_usage", {}) if previous else {}, llm.get_usage_summary())}
                atomic(path, result)
                return result
            except Exception as exc:
                prior_attempts.append({"time": now(), "error_type": type(exc).__name__, "error": str(exc)[:500]})
                if attempt == 0:
                    time.sleep(3)
        result = {"status": "failed", "arm": arm, "dialogue_id": case_id,
            "dialogue_hash": digest(dialogue), "resource_hash": resource_hash, "attempts": prior_attempts,
            "turns": 0, "rows": [], "llm_usage": merge_usage_summaries(previous.get("llm_usage", {}) if previous else {}, llm.get_usage_summary())}
        atomic(path, result)
        return result


def progress(args, test, phase, started):
    arms = {}
    expected = sum(len(action_indices(d)) for d in test)
    for arm in ARMS:
        cases = [read(args.output_dir / arm / "cases" / (str(d["convo_id"]) + ".json")) for d in test]
        complete = [r for r in cases if r and r["status"] == "complete"]
        arms[arm] = {"completed_dialogues": len(complete), "expected_dialogues": len(test),
            "completed_turns": sum(r["turns"] for r in complete), "expected_turns": expected,
            "failed_dialogues": sum(bool(r and r["status"] == "failed") for r in cases)}
    elapsed = time.monotonic() - started
    done = sum(a["completed_turns"] for a in arms.values())
    total = expected * len(ARMS)
    eta = (total - done) * elapsed / done if done else None
    state = {"phase": phase, "updated_at": now(), "pid": os.getpid(), "arms": arms,
        "completed_turns": done, "expected_turns": total, "elapsed_seconds": elapsed, "eta_seconds": eta}
    atomic(args.output_dir / "progress.json", state)
    return state


def collect(args, test, manifest, generation_usage):
    # Reuse the established ABCD evaluator over saved predictions, with no API calls.
    os.environ["ABCD_OUTPUT_DIR"] = str(args.output_dir / "eval_aux")
    from scripts.run_subflow_eval import evaluate_agent_on_subflow
    entries = {}
    for arm in ARMS:
        records = {str(d["convo_id"]): read(args.output_dir / arm / "cases" / (str(d["convo_id"]) + ".json")) for d in test}
        if not all(r and r["status"] == "complete" for r in records.values()):
            entries[arm] = {"complete": False, "successful_dialogues": sum(bool(r and r["status"] == "complete") for r in records.values())}
            continue
        class ReplayAgent:
            def predict_all_turns(self, dialogue, **kwargs):
                rows = records[str(dialogue["convo_id"])]["rows"]
                validate_rows(dialogue, rows)
                return rows
        result = evaluate_agent_on_subflow(ReplayAgent(), test, arm, FLOW, save_dir=args.output_dir / arm, skip_utterance_eval=True)
        testing = merge_usage_summaries(*(r["llm_usage"] for r in records.values()))
        result.update(complete=True, arm=arm, online_refinement=False, shared_resource_hashes=manifest["resources_sha256"][arm],
                      llm_usage={"generation_shared_actual": generation_usage, "testing": testing})
        atomic(args.output_dir / arm / "result.json", result)
        entries[arm] = result
    atomic(args.output_dir / "aggregate.json", {"updated_at": now(), "complete": all(r["complete"] for r in entries.values()),
        "arms": entries, "shared_generation_usage": generation_usage,
        "note": "Shared resource generation counted once in actual spend; each arm testing cost independently tracked."})
    lines = ["# shipping_issue: frozen graph selection, shared resources", "", "Model: " + args.model,
        "", "| Method | Complete | Action turns | Action acc | Joint AST | Slot given action | Test calls | Test tokens |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm, result in entries.items():
        if not result["complete"]:
            lines.append(f"| {arm} | false | incomplete | — | — | — | — | — |")
            continue
        ast, usage = result["ast_cds"], result["llm_usage"]["testing"]["total"]
        lines.append(f"| {arm} | true | {ast['num_action_turns']} | {ast['ast_action_name']:.4f} | {ast['ast_joint']:.4f} | {ast['ast_slot_value_given_action']:.4f} | {usage['calls']} | {usage['total_tokens']} |")
    lines += ["", "Same TRAIN-only action/slot cards, all-observed reference pool, model, runtime, reading limits and heldout dialogues. Only the visible procedural graph changes. No online refinement. Existing ABCD annotated action-turn evaluation, not interactive environment task completion.",
        "", "TraceCover evaluation is explicitly skipped. Its cached graph only supplies TRAIN node contracts. Shared baseline resources were newly induced because no verified local compiled artifact was available. Source hashes and budgets are in config.json; shared resource hashes are in shared/manifest.json."]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / "run.lock"
    if lock_path.exists():
        import psutil
        old = read(lock_path)
        if old and old.get("pid") and psutil.pid_exists(old["pid"]):
            process = psutil.Process(old["pid"])
            if "run_shipping_graph_agents.py" in " ".join(process.cmdline()):
                raise RuntimeError("This run already has a live process")
    atomic(lock_path, {"pid": os.getpid(), "started_at": now()})
    import llm
    config = configure(args.env_file, args.model)
    args.model = config["model"]
    os.environ["SKILLMINING_STOP_ON_ERROR"] = "1"
    os.environ["SKILLMINING_LLM_TIMEOUT_SECONDS"] = "120"
    guard = APIGuard(llm.chat, args.model, out, args.workers)
    llm.chat = guard
    finished = threading.Event()
    monitor = threading.Thread(target=guard.heartbeat, args=(finished,), daemon=True)
    monitor.start()
    try:
        if not read(out / "api_preflight.json"):
            start = time.monotonic()
            response = llm.chat("Reply with READY only.", temperature=0., call_tag="api_preflight")
            atomic(out / "api_preflight.json", {"ok": bool(response.strip()), "model": args.model,
                "latency_seconds": time.monotonic() - start, "timestamp": now(), "usage": llm.get_usage_summary()})
        previous_generation = read(out / "shared/generation_usage.json", {})
        llm.reset_usage_summary()
        test, bundles, inputs, manifest = prepare(args, config, guard)
        generation = merge_usage_summaries(previous_generation, llm.get_usage_summary())
        atomic(out / "shared/generation_usage.json", generation)
        started = time.monotonic()
        progress(args, test, "evaluation", started)
        # Interleave arms per dialogue, preserve ordered prediction within dialogue.
        for wave in range(2):
            jobs = []
            for dialogue in test:
                for arm in ARMS:
                    saved = read(out / arm / "cases" / (str(dialogue["convo_id"]) + ".json"))
                    if not saved or saved["status"] != "complete":
                        jobs.append((arm, dialogue))
            if not jobs or guard.abort.is_set():
                break
            print(f"EVALUATION wave={wave+1}, pending dialogues={len(jobs)}, workers={args.workers}", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(evaluate_case, args, arm, dialogue, bundles[arm], digest(manifest["resources_sha256"][arm])): (arm, dialogue) for arm, dialogue in jobs}
                for future in as_completed(futures):
                    result = future.result()
                    state = progress(args, test, "evaluation", started)
                    print(f"DONE {result['arm']}/{result['dialogue_id']}: {result['status']} {state['completed_turns']}/{state['expected_turns']} turns", flush=True)
            collect(args, test, manifest, generation)
        collect(args, test, manifest, generation)
        aggregate = read(out / "aggregate.json")
        progress(args, test, "completed" if aggregate["complete"] else "completed_with_errors", started)
    except BaseException as exc:
        atomic(out / "fatal_error.json", {"time": now(), "type": type(exc).__name__, "error": str(exc)[:1000], "traceback": traceback.format_exc()})
        state = read(out / "progress.json", {})
        atomic(out / "progress.json", {**state, "phase": "failed", "updated_at": now(), "pid": os.getpid()})
        raise
    finally:
        finished.set()
        atomic(lock_path, {"pid": None, "stopped_at": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--workflow-max-chars", type=int, default=8000)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps({"progress": read(args.output_dir / "progress.json"),
            "api": read(args.output_dir / "api_heartbeat.json"), "results": read(args.output_dir / "aggregate.json", {}).get("complete")}, ensure_ascii=False, indent=2))
    else:
        if not 1 <= args.workers <= 10:
            parser.error("workers must be 1..10")
        run(args)
