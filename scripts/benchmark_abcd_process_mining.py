"""ABCD process discovery: official PM4Py miners and shared adjacency probes.

CPU only. Petri models are projected using exhaustive reachable-prefix state
exploration, including silent transitions. Budget exhaustion is an error, never
a partially scored graph. Projection coverage is not Petri-net conformance.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_sop_graphs as bench
from scripts import eval_offline_graph_10flow as replay
from skill_mining.sop_reference_benchmark import build_reference, score_reference, aggregate_reference, digest

PM_METHODS = ("alpha", "heuristics_pm4py", "inductive", "inductive_f20")
CACHED_METHODS = {"treefree_log_cached": "graph_log",
                  "treefree_support_mean_matched_cached": "graph_support_mean_matched"}
METHODS = ("full_dfg", "trace_cover", "joint_trace_cover") + tuple(CACHED_METHODS) + PM_METHODS
METHODS += ("heuristics_net",)


def project_petri(net, initial, *, max_states=50000, max_seconds=20., max_tokens=512):
    """Exact adjacent visible-label pairs on any reachable finite prefix.

    State = (marking, last visible label). A silent firing preserves the last
    label; a visible firing contributes one pair and replaces it. Distinct
    histories with the same state have identical possible continuations.
    """
    from pm4py.objects.petri_net import semantics
    started = time.perf_counter()
    places = {p: i for i, p in enumerate(sorted(net.places, key=lambda p: p.name))}
    def key(marking):
        return tuple(sorted((places[p], n) for p, n in marking.items() if n))
    queue = deque([(initial, None)])
    seen = {(key(initial), None)}
    outgoing = {}
    pairs = set()
    visible = set()
    while queue:
        if time.perf_counter() - started > max_seconds:
            raise RuntimeError(f"Projection time budget exceeded ({max_seconds}s); no partial score")
        marking, last = queue.popleft()
        mk = key(marking)
        if mk not in outgoing:
            outgoing[mk] = [(t.label, semantics.execute(t, net, marking)) for t in sorted(
                semantics.enabled_transitions(net, marking), key=lambda t: (str(t.label), t.name))]
        for label, nxt in outgoing[mk]:
            if sum(nxt.values()) > max_tokens:
                raise RuntimeError("Projection token budget exceeded; possible unbounded net")
            if label is not None:
                visible.add(label)
                if last is not None:
                    pairs.add((last, label))
            state = (key(nxt), last if label is None else label)
            if state not in seen:
                seen.add(state)
                if len(seen) > max_states:
                    raise RuntimeError(f"Projection state budget exceeded ({max_states}); no partial score")
                queue.append((nxt, state[1]))
    return pairs, {"complete": True, "reachable_markings": len(outgoing),
                   "marking_last_label_states": len(seen), "reachable_visible_labels": len(visible),
                   "elapsed_seconds": time.perf_counter() - started}


def coverage(sequences, pairs, rho=.8):
    counts = [Counter(zip(seq, seq[1:])) for seq in sequences if len(seq) > 1]
    totals = [sum(c.values()) for c in counts]
    hits = [sum(n for e, n in c.items() if e in pairs) for c in counts]
    return {"sessions": len(sequences), "sessions_with_transitions": len(counts),
            "zero_transition_sessions": len(sequences) - len(counts),
            "transition_occurrences": sum(totals), "covered_occurrences": sum(hits),
            "raw_session_sum": sum(k / n for k, n in zip(hits, totals)),
            "saturated_sum": sum(min(1, k / (rho * n)) for k, n in zip(hits, totals)),
            "complete_sessions": sum(k == n for k, n in zip(hits, totals))}


def make_log(sequences):
    from pm4py.objects.log.obj import EventLog, Trace, Event
    log = EventLog()
    for i, seq in enumerate(sequences):
        log.append(Trace([Event({"concept:name": a}) for a in seq], attributes={"concept:name": str(i)}))
    return log


def discover(method, log, path):
    import pm4py
    tree = None
    if method == "alpha":
        net, im, fm = pm4py.discover_petri_net_alpha(log)
    elif method == "heuristics_pm4py":
        net, im, fm = pm4py.discover_petri_net_heuristics(
            log, dependency_threshold=.5, and_threshold=.65, loop_two_threshold=.5)
    else:
        threshold = .2 if method == "inductive_f20" else 0.
        tree = pm4py.discover_process_tree_inductive(log, noise_threshold=threshold, multi_processing=False)
        path.with_suffix(".tree.txt").write_text(str(tree), encoding="utf-8")
        net, im, fm = pm4py.convert_to_petri_net(tree)
    pm4py.write_pnml(net, im, fm, str(path))
    return net, im, {"places": len(net.places), "transitions": len(net.transitions),
                     "arcs": len(net.arcs), "silent_transitions": sum(t.label is None for t in net.transitions)}, tree


def run_flow(flow, args, reference):
    train, validation = [bench.read(args.splits_dir / flow / (s + ".json")) for s in ("train", "test")]
    train_sequences = [replay.action_sequence(flow, d) for d in train]
    validation_sequences = [replay.action_sequence(flow, d) for d in validation]
    observed = {p for s in train_sequences for p in zip(s, s[1:])}
    vocabulary = {a for s in train_sequences for a in s}
    log = make_log(train_sequences)
    rows = []
    for method in args.methods:
        started = time.perf_counter()
        row = {"flow": flow, "method": method, "train_sha256": digest(train),
               "validation_sha256": digest(validation), "reference_sha256": digest(reference["flows"][flow]),
               "train_action_turns": sum(map(len, train_sequences)), "status": "failed"}
        try:
            if method == "heuristics_net":
                import pm4py
                heu = pm4py.discover_heuristics_net(log, dependency_threshold=.5, and_threshold=.65, loop_two_threshold=.5)
                pairs = {(n.node_name, target.node_name) for n in heu.nodes.values() for target in n.output_connections}
                row["projection"] = {"complete": True, "kind": "native_heuristics_dependency_edges",
                                      "note": "Heuristics-net dependency graph, not enabled Petri behavior"}
                row["native_complexity"] = {"activity_nodes": len(heu.nodes), "dependency_edges": len(pairs)}
                bench.dump(args.output_dir / "models/heuristics_net" / (flow + ".json"),
                           {"nodes": sorted(heu.nodes), "dependencies": sorted(pairs),
                            "and_measures_out": {n.node_name: n.and_measures_out for n in heu.nodes.values()}})
            elif method in PM_METHODS:
                path = args.output_dir / "models" / method / (flow + ".pnml")
                path.parent.mkdir(parents=True, exist_ok=True)
                net, initial, row["native_complexity"], tree = discover(method, log, path)
                if tree is not None:
                    import pm4py
                    fp = pm4py.discover_footprints(tree)
                    pairs = set(fp["sequence"]) | set(fp["parallel"])
                    row["projection"] = {"complete": True, "kind": "process_tree_bottom_up_footprints"}
                else:
                    pairs, row["projection"] = project_petri(net, initial, max_states=args.max_states,
                                                              max_seconds=args.projection_seconds)
            elif method == "full_dfg":
                pairs = observed
            else:
                if method in CACHED_METHODS:
                    cache_method = CACHED_METHODS[method]
                    graph = bench.read(args.treefree_dir / "graphs" / cache_method / (flow + ".json"))
                    prior = bench.read(args.treefree_dir / "per_flow" / cache_method / (flow + ".json"))
                    if prior["train_sha256"] != digest(train):
                        raise ValueError("Cached treefree graph has different TRAIN data")
                else:
                    graph = bench.mine(method, flow, train)
                pairs = {(e["source"], e["target"]) for es in graph["local_transitions"].values() for e in es}
            assert {a for p in pairs for a in p} <= vocabulary
            # Preserve generalized relations; never intersect with TRAIN to improve precision.
            raw_pairs = {(a.split(":", 1)[1], b.split(":", 1)[1]) for a, b in pairs}
            raw_observed = {(a.split(":", 1)[1], b.split(":", 1)[1]) for a, b in observed}
            graph = bench.control_graph(flow, raw_pairs, raw_observed)
            row.update(status="completed", full_edge_types=len(observed), adjacency_types=len(pairs),
                       unobserved_adjacency_types=len(pairs - observed),
                       train=coverage(train_sequences, pairs), validation=coverage(validation_sequences, pairs),
                       sop=score_reference(graph, reference["flows"][flow]),
                       graph_sha256=digest(graph))
            bench.dump(args.output_dir / "graphs" / method / (flow + ".json"), graph)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_seconds"] = time.perf_counter() - started
        bench.dump(args.output_dir / "per_flow" / method / (flow + ".json"), row)
        rows.append(row)
        print(f"{row['status'].upper()} {flow} {method} edges={row.get('adjacency_types')} "
              f"seconds={row['elapsed_seconds']:.2f} {row.get('error', '')}", flush=True)
    return rows


def summarize(rows, methods, flows):
    summaries = {}
    for method in methods:
        group = [r for r in rows if r["method"] == method]
        failed = [r["flow"] for r in group if r["status"] != "completed"]
        if failed or len(group) != len(flows):
            summaries[method] = {"status": "incomplete", "failed_flows": failed}
            continue
        full = sum(r["full_edge_types"] for r in group)
        edges = sum(r["adjacency_types"] for r in group)
        result = {"status": "completed", "full_edge_types": full, "adjacency_types": edges,
                  "edge_reduction": 1 - edges / full,
                  "unobserved_adjacency_types": sum(r["unobserved_adjacency_types"] for r in group)}
        for split in ("train", "validation"):
            metrics = [r[split] for r in group]
            n = sum(m["sessions_with_transitions"] for m in metrics)
            t = sum(m["transition_occurrences"] for m in metrics)
            result[split] = {"sessions_with_transitions": n, "transition_occurrences": t,
                "mean_session_coverage": sum(m["raw_session_sum"] for m in metrics) / n,
                "saturated_coverage": sum(m["saturated_sum"] for m in metrics) / n,
                "transition_recall": sum(m["covered_occurrences"] for m in metrics) / t,
                "complete_session_coverage": sum(m["complete_sessions"] for m in metrics) / n}
        result["guideline"] = aggregate_reference([{**r["sop"], "train_action_turns": r["train_action_turns"]} for r in group])
        hit, total = [sum(r["sop"][key] for r in group) for key in ("positive_hits", "positive_probes")]
        result["guideline"].update(micro_hits=hit, micro_total=total, micro_recall=hit / total)
        summaries[method] = result
    return summaries


def report(summary):
    lines = ["# ABCD process mining: graph projection benchmark", "",
        "All values except relation counts are percentages. Each method is mined independently on the same ten TRAIN flow splits. Evaluation is the historical development replay, not a fresh blind test.", "",
        "| Method | Adjacent relation types | Train mean session | Train saturated F | Dev mean session | Dev transition recall | Dev complete session | Guideline hits | Guideline micro recall | Guideline weighted recall | Reverse rejection |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, r in summary.items():
        if r["status"] != "completed":
            lines.append(f"| {method} | INCOMPLETE: {','.join(r['failed_flows'])} | | | | | | | | | |")
            continue
        tr, va, g = r["train"], r["validation"], r["guideline"]
        vals = [tr["mean_session_coverage"], tr["saturated_coverage"], va["mean_session_coverage"],
                va["transition_recall"], va["complete_session_coverage"]]
        lines.append(f"| {method} | {r['adjacency_types']} | " + " | ".join(f"{100*v:.2f}" for v in vals)
                     + f" | {g['micro_hits']}/{g['micro_total']} | {100*g['micro_recall']:.2f}"
                     + f" | {100*g['sop_recall']:.2f} | {100*g['reverse_order_rejection']:.2f} |")
    lines += ["", "## Interpretation boundaries", "",
        "- PM4Py 2.7.23 official implementations: Alpha classic; Heuristics dependency=.5 / AND=.65 / loop-two=.5; Inductive noise=0; Inductive infrequent noise=.2. Settings fixed without using the reference or replay to tune.",
        "- Alpha and Heuristics Petri models use exhaustive reachable-prefix projection through silent transitions. State exploration must finish; exceeded budgets produce failed rows with no score. The final marking is NOT required. Inductive models use the equivalent process-tree bottom-up footprint union (sequence + parallel), avoiding explicit state explosion. This is not a token-replay conformance evaluation.",
        "- heuristics_net is separately labelled: official Heuristics-net OUTPUT DEPENDENCY EDGES. It does not claim Petri reachability semantics. Its graph-relation coverage is directly auditable, including explicit dependency edges and loop relations returned by the native miner.",
        "- Whole-session coverage means every observed adjacent pair occurs somewhere in this projected relation set. It does NOT prove that the entire sequence can execute as one Petri run. The same adjacency criterion is used for our action graphs.",
        "- Generalized adjacency relations outside the observed DFG are retained and counted, not silently intersected away. Relation count is a common projected branching measure, NOT native model storage size; native Petri places/transitions/arcs are saved separately.",
        "- Original full DFG=1338 edge types summed across ten flows. TraceCover/joint/treefree cached versions are reported separately; no best-number mixing. Cached treefree TRAIN hashes are checked.",
        "- Guideline micro recall is total hits / 118 positive structural probes. Weighted recall averages flow recall by TRAIN action turns. Reverse-order rejection uses nine ordered flows only; troubleshoot_site is explicitly unordered. Unknown relations are not negatives; probes are not a complete business-compliance oracle.",
        "- Saturated F uses rho=.8; it is not raw session coverage. Raw means exclude sessions with no transitions; counts are saved. Training F targets apply only to TraceCover methods, not other miners.",
        "- No LLM calls, skill compilation, agent rollout or production algorithm changes. Every native PM model and projected graph is saved.",
        "", "Official implementation: https://processintelligence.solutions/pm4py/examples/process-discovery", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=replay.DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--flows", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-states", type=int, default=50000)
    parser.add_argument("--projection-seconds", type=float, default=20.)
    parser.add_argument("--treefree-dir", type=Path, default=ROOT / "outputs/treefree_weighting_20261021_fullcheck_v2")
    args = parser.parse_args()
    args.methods = tuple(args.methods.split(","))
    if any(m not in METHODS for m in args.methods):
        parser.error("Unknown method")
    if args.output_dir.exists():
        parser.error("Use a new output directory; existing experiment artifacts are never overwritten")
    import pm4py
    resource = ROOT / "data/eval/abcd/data"
    reference = build_reference(*(bench.read(resource / name) for name in ("kb.json", "ontology.json", "guidelines.json")))
    flows = args.flows.split(",") if args.flows else sorted(reference["flows"])
    args.output_dir.mkdir(parents=True)
    config = {**vars(args), "pm4py_version": pm4py.__version__, "llm_calls": 0,
              "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
              "script_sha256": digest(Path(__file__).read_text(encoding="utf-8"))}
    bench.dump(args.output_dir / "config.json", json.loads(json.dumps(config, default=str)))
    bench.dump(args.output_dir / "reference.json", reference)
    (args.output_dir / "source_snapshot.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    rows = []
    with patch("llm.chat", side_effect=AssertionError("LLM calls forbidden")):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_flow, flow, args, reference) for flow in flows]
            for future in as_completed(futures):
                rows.extend(future.result())
                bench.dump(args.output_dir / "progress.json", {"completed_flows": len(rows) // len(args.methods), "total_flows": len(flows)})
    summary = summarize(rows, args.methods, flows)
    bench.dump(args.output_dir / "results.json", {"config": json.loads(json.dumps(config, default=str)), "rows": rows, "summary": summary})
    (args.output_dir / "report.md").write_text(report(summary), encoding="utf-8")
    print(report(summary), flush=True)
    if any(r["status"] != "completed" for r in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
