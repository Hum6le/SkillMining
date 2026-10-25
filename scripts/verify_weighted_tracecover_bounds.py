"""Check weighted tree-free TraceCover bounds; no training or LLM calls."""
from collections import Counter, defaultdict
from fractions import Fraction as Q
from pathlib import Path
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_treefree_tracecover_bounds import coverage, cost, reachable, lower_bound


def weighted_bounds(edges, nodes, entries, traces, rho, eta, p, gamma, beta, w):
    lc, k0, _ = lower_bound(edges, nodes, entries, traces, rho, eta, p, gamma)
    capacities = Counter(u for u, v in edges)
    increments = sorted(p.get(u, Q(0)) * j for u, d in capacities.items() for j in range(d))
    ordered_w = sorted(w.values(), reverse=True)
    phi = reward = Q(0)
    separated = []
    for k, (inc, weight) in enumerate(zip(increments, ordered_w), 1):
        phi += inc
        reward += weight
        if k >= k0:
            separated.append(Q(k) + gamma * phi - beta * reward)
    by_source = defaultdict(list)
    for e in edges:
        by_source[e[0]].append(w[e])
    # Ties prefer earlier increments from a source (prefix feasibility).
    marginal = sorted((Q(1) - beta * weight + gamma * p.get(u, Q(0)) * j, u, j)
                      for u, weights in by_source.items()
                      for j, weight in enumerate(sorted(weights, reverse=True)))
    prefix = Q(0)
    candidates = []
    for k, (increment, _, _) in enumerate(marginal, 1):
        prefix += increment
        if k >= k0:
            candidates.append(prefix)
    simple = lc - beta * sum((max(Q(0), value) for value in w.values()), Q(0))
    return dict(k0=k0, complexity=lc, simple=simple,
                separated=min(separated), degree=min(candidates))


def check_exact(instances=32, seed=20261025):
    rng = random.Random(seed)
    rows = []
    inspected = 0
    while len(rows) < instances:
        nv = rng.choice([2, 3, 4])
        walks = [[rng.randrange(nv) for _ in range(rng.randrange(2, 6))] for _ in range(4)]
        nodes = set(v for walk in walks for v in walk)
        traces = [Counter(zip(walk, walk[1:])) for walk in walks]
        edges = set().union(*(set(row) for row in traces))
        if len(edges) > 10:
            continue
        entries = {walk[0] for walk in walks}
        if len(rows) % 5 == 0:
            entries = set(nodes)  # realized by extra one-action dialogues
        rho = rng.choice([Q(1, 3), Q(4, 5), Q(1)])
        eta = rng.choice([Q(3, 5), Q(19, 20), Q(1)])
        gamma = rng.choice([Q(0), Q(1, 2), Q(2)])
        beta = rng.choice([Q(0), Q(1, 2), Q(1), Q(3)])
        counts = Counter()
        for row in traces:
            for (u, v), n in row.items():
                counts[u] += n
        p = {u: Q(n, sum(counts.values())) for u, n in counts.items()}
        w = {e: Q(rng.randint(-8, 16), 4) for e in sorted(edges)}
        bounds = weighted_bounds(edges, nodes, entries, traces, rho, eta, p, gamma, beta, w)
        def objective(s):
            return cost(s, p, gamma) - beta * sum((w[e] for e in s), Q(0))
        feasible = {}
        relaxed_opt = None
        es = sorted(edges)
        for mask in range(1 << len(es)):
            selected = frozenset(e for j, e in enumerate(es) if mask >> j & 1)
            inspected += 1
            value = objective(selected)
            if len(selected) >= bounds['k0']:
                relaxed_opt = value if relaxed_opt is None else min(relaxed_opt, value)
            if coverage(selected, traces, rho) >= eta and reachable(selected, nodes, entries):
                feasible[selected] = value
        optimum = min(feasible.values())
        assert bounds['degree'] == relaxed_opt
        assert bounds['simple'] <= bounds['separated'] <= bounds['degree'] <= optimum
        current = frozenset(edges)
        history = [feasible[current]]
        while True:
            proposals = []
            for removed in sorted(current):
                for added in [None] + sorted(edges - current):
                    nxt = frozenset((current - {removed}) | ({added} if added is not None else set()))
                    if nxt in feasible and feasible[nxt] < history[-1]:
                        proposals.append((feasible[nxt], sorted(nxt), nxt))
            if not proposals:
                break
            value, _, current = min(proposals)
            history.append(value)
        assert all(bounds['degree'] <= optimum <= value for value in history)
        assert all(history[i + 1] < history[i] for i in range(len(history) - 1))
        rows.append(dict(edges=len(edges), feasible_subsets=len(feasible), beta=str(beta),
                         lower_bound=float(bounds['degree']), optimum=float(optimum),
                         final_objective=float(history[-1]), accepted_moves=len(history) - 1))
    return dict(seed=seed, instances=len(rows), subsets_inspected=inspected,
                arithmetic='exact rational', all_checks_passed=True, results=rows)


def check_saved_bench():
    """Audit existing graph-only artifacts, without rerunning the miner."""
    from skill_mining.treefree_trace_cover import observed_graph
    source = ROOT / 'outputs/treefree_weighting_20261021_fullcheck_v2/graphs/graph_log'
    rows = []
    for path in sorted(source.glob('*.json')):
        graph = json.loads(path.read_text(encoding='utf-8'))
        if 'subgraph' in graph:
            graph = graph['subgraph']
        meta = graph['treefree_trace_cover']
        flow = path.stem
        train = json.loads((ROOT / 'data/eval/abcd/splits' / flow / 'train.json').read_text(encoding='utf-8'))
        data = observed_graph(flow, train)
        edges = {(e['source'], e['target']) for e in data['edges']}
        assert edges == {(e['source'], e['target']) for e in graph['edges']}
        weights = {(e['source'], e['target']): Q(str(e['score'])) for e in graph['edges']}
        counts = Counter()
        for trace in data['sessions']:
            for (u, v), n in trace.items():
                counts[u] += n
        p = {u: Q(n, sum(counts.values())) for u, n in counts.items()}
        nodes = {n['id'] for n in data['nodes']}
        entries = set(data['entries'])
        rho, eta, gamma, beta = [Q(str(meta[k])) for k in ('rho', 'eta', 'gamma', 'beta')]
        bounds = weighted_bounds(edges, nodes, entries, data['sessions'], rho, eta, p, gamma, beta, weights)
        selected = {(e['source'], e['target']) for group in graph['local_transitions'].values() for e in group}
        value = cost(selected, p, gamma) - beta * sum((weights[e] for e in selected), Q(0))
        exact_feasible = coverage(selected, data['sessions'], rho) >= eta and reachable(selected, nodes, entries)
        assert exact_feasible
        assert bounds['degree'] <= value
        assert abs(float(value) - meta['objective']) < 1e-7
        assert abs(float(coverage(selected, data['sessions'], rho)) - meta['coverage']) < 1e-10
        rows.append(dict(flow=flow, observed_edges=len(edges), selected_edges=len(selected),
                         k0=bounds['k0'], objective=float(value),
                         lower_bound=float(bounds['degree']),
                         additive_gap_upper_bound=float(value-bounds['degree']),
                         exact_objective=str(value), exact_lower_bound=str(bounds['degree']),
                         exact_gap_upper_bound=str(value-bounds['degree']),
                         coverage=float(coverage(selected, data['sessions'], rho)), exact_feasible=True))
    assert len(rows) == 10, f'Expected ten cached graphs, found {len(rows)}'
    return rows


if __name__ == '__main__':
    report = dict(scope='Weighted tree-free surrogate only; no LLM/SR claim', exact=check_exact())
    if '--bench' in sys.argv:
        report['saved_graph_audit'] = check_saved_bench()
    out = ROOT / 'refine-logs/treefree-tracecover/WEIGHTED_BOUND_AUDIT_20261025.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report['exact'].items() if k != 'results'}, indent=2))
    if 'saved_graph_audit' in report:
        print(json.dumps(report['saved_graph_audit'], indent=2))
    print(str(out))
