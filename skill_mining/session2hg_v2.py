"""
从 operator_results.json 构建会话级超图与全局超图，并求解「高覆盖顶点子图」。

超边：同一条记录内 ordered_operations 中所有「角色:操作」节点（去重）构成一条超边。

覆盖定义（可调）：设超边 e 的顶点数为 |e|，阈值为 ⌈ρ·|e|⌉（ρ∈(0,1]，默认 1 表示须包含 e 内**全部**顶点；ρ=0.9 表示至少包含 90% 的节点，按向上取整）。
顶点集 S **覆盖** e 当且仅当 |S ∩ e| ≥ ⌈ρ·|e|⌉。

问题：不固定 |S|，在目标中权衡「多覆盖超边 / 多异质序列边 / 少同构序列边」与「顶点入场费」λ·|S|（与 `HBG_Mining.py` 一致）；
须取 **λ>0**，否则选全集往往在目标上占优。

会先抽出「单独节点覆盖的超边比例 ≥ 阈值」（默认 90%）的高覆盖节点集 U；
在剩余顶点 V'\\U 上，对缩减超边 e' = e \\ U 做 MILP，并按 nogood 割求 Top-K（--top）组互异解。

原序列图由全数据 `ordered_operations` 中相邻两步（同一会话内顺序相邻）连无向边得到；
MILP 中用它定义异质/同构边正则项，**不**强制 S 在该图上连通（后处理合并图时仍可补点）。

对每个挖掘结果，在节点集 S 上仅绘制**序列骨干有向边**（相邻操作 u->v 且 u、v 在序列无向图上相邻，
边权为出现次数），不再绘制 S 上全局转移图的其余弧，并保存为图片（--viz-dir）。

另将 Top-K 组解的顶点集合并：在**序列无向图**上若多块不连通，则用最短路补入中间顶点，
得到单一连通分量后再画一张合并图；其中**权重和最大**的有向路径（DAG 时用最长路，
否则高亮权重最大的单条弧）以**红色加粗**标出。

挖掘（部分二）采用 **PuLP + CBC 整数规划**（与 `HBG_Mining.py` 同构思路），不再使用暴力枚举或贪心/局部搜索。

节点集合搜索目标（最大化）：
  γ·∑_e y_e + balance·(α·∑ z_异质 − β·∑ z_同构) − λ·∑_v x_v
无固定 |S|；序列图上的 z 与 x 为 McCormick 线性化。
**不**在 MILP 中强加「序列诱导子图连通」（与 HBG 一致）；输出中可查看是否连通，合并图阶段仍会补点。

边集合搜索：同一个 MILP 同时决定节点变量 x_v 与有向边变量 edge_on(u,v)，并用
λ_edge·∑edge_on 作为边数量正则项。每个被选中节点必须至少关联一条被选中边，避免结果里出现孤立点。
被选边默认需服从某个拓扑序；违反拓扑序的边要支付 cycle-edge-penalty（默认 0）。
因此主体倾向 DAG，但权重收益足够高的局部回边/小环可以作为例外保留。

Top-K：每求得一最优解 S* 后，加入 nogood 割
  ∑_{i∈S*} x_i − ∑_{j∉S*} x_j ≤ |S*|−1
以排除完全相同的 0–1 赋值，再求下一最优。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = BASE_DIR /"CKDD_SM"/ "output" / "意图4_联系司机"/ "operator_results.json"
DEFAULT_VIZ_DIR = BASE_DIR /"CKDD_SM"/ "output" / "意图4_联系司机"/ "mined_subgraphs"


def node_name(role: str, operation: str) -> str:
    operation = (operation or "").strip()
    if not operation or not role:
        return ""
    return f"{role}:{operation}"


@dataclass(frozen=True)
class Hyperedge:
    record_index: int
    session_id: str
    vertices: FrozenSet[str]

    @property
    def size(self) -> int:
        return len(self.vertices)


class SessionHypergraph:
    def __init__(self, hyperedges: Sequence[Hyperedge], vertex_incidence: Dict[str, List[int]]):
        self.hyperedges = list(hyperedges)
        self.vertex_incidence = vertex_incidence
        self.vertices = frozenset(vertex_incidence.keys())

    @classmethod
    def from_operator_results(cls, results: List[dict]) -> "SessionHypergraph":
        hyperedges: List[Hyperedge] = []
        vertex_incidence: Dict[str, List[int]] = {}

        for idx, result in enumerate(results):
            ordered = result.get("ordered_operations") or []
            bag: set[str] = set()
            for pair in ordered:
                if not pair or len(pair) < 2:
                    continue
                role, operation = pair[0], pair[1]
                n = node_name(str(role), str(operation))
                if n:
                    bag.add(n)
            if not bag:
                continue
            sid = str(result.get("session_id", "") or f"index_{idx}")
            e = Hyperedge(record_index=idx, session_id=sid, vertices=frozenset(bag))
            he_idx = len(hyperedges)
            hyperedges.append(e)
            for v in e.vertices:
                vertex_incidence.setdefault(v, []).append(he_idx)

        return cls(hyperedges=hyperedges, vertex_incidence=vertex_incidence)


def node_sequence_from_ordered_operations(ordered: Sequence) -> List[str]:
    """Extract role:operation node names from one session's ordered_operations."""
    seq: List[str] = []
    for pair in ordered:
        if not pair or len(pair) < 2:
            continue
        role, operation = pair[0], pair[1]
        nn = node_name(str(role), str(operation))
        if nn:
            seq.append(nn)
    return seq


def collapse_consecutive_nodes(seq: List[str]) -> Tuple[List[str], int]:
    """Merge consecutive duplicate node names (removes session self-loop transitions)."""
    if not seq:
        return [], 0
    out = [seq[0]]
    removed = 0
    for n in seq[1:]:
        if n == out[-1]:
            removed += 1
        else:
            out.append(n)
    return out, removed


def collapse_consecutive_ordered_operations(ordered: Sequence) -> Tuple[List, int]:
    """Drop consecutive duplicate [role, operation] steps in one session."""
    if not ordered:
        return [], 0
    out: List = []
    removed = 0
    last_key: Optional[str] = None
    for pair in ordered:
        if not pair or len(pair) < 2:
            out.append(pair)
            last_key = None
            continue
        k = node_name(str(pair[0]), str(pair[1]))
        if k and k == last_key:
            removed += 1
            continue
        out.append(pair)
        last_key = k
    return out, removed


def collapse_operator_results_sequences(results: List[dict]) -> Tuple[List[dict], int]:
    """Per session: remove consecutive duplicate ops (in-place). Returns total steps removed."""
    total_removed = 0
    for result in results:
        ordered = result.get("ordered_operations") or []
        collapsed, n = collapse_consecutive_ordered_operations(ordered)
        if n:
            result["ordered_operations"] = collapsed
            total_removed += n
    return results, total_removed


def build_sequence_undirected_adj(results: List[dict]) -> Dict[str, Set[str]]:
    """
    原序列图（无向）：同一条记录的 ordered_operations 中，相邻两条操作对应的节点连边。
    与 graph_construction 中顺序边一致，但视为无向以定义诱导子图连通性。
    """
    adj: Dict[str, Set[str]] = {}

    def add_edge(a: str, b: str) -> None:
        if a == b:
            return
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    for result in results:
        seq = node_sequence_from_ordered_operations(result.get("ordered_operations") or [])
        seq, _ = collapse_consecutive_nodes(seq)
        for i in range(len(seq) - 1):
            add_edge(seq[i], seq[i + 1])
    return adj


def build_global_edge_weights(results: List[dict]) -> Dict[Tuple[str, str], int]:
    """
    与 graph_construction 一致的全局有向转移：键 (u,v) 为同一会话内相邻两步，值为出现次数。
    不依赖 networkx，便于仅做挖掘时环境更轻；绘图时再转为 DiGraph。
    """
    edge_counts: Dict[Tuple[str, str], int] = {}

    for result in results:
        seq = node_sequence_from_ordered_operations(result.get("ordered_operations") or [])
        seq, _ = collapse_consecutive_nodes(seq)
        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i + 1]
            edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1

    return edge_counts


def _node_role_color(node: str) -> str:
    if node.startswith("用户:"):
        return "#f4a261"
    if node.startswith("客服:"):
        return "#457b9d"
    return "#adb5bd"


def _vertex_side(name: str) -> Optional[int]:
    """二部侧：0=用户，1=客服；其它前缀不参与异质/同构计数。"""
    if name.startswith("用户:"):
        return 0
    if name.startswith("客服:"):
        return 1
    return None


def _sequence_bipartite_edge_counts(
    names: List[str], S: Set[int], adj_idx: List[Set[int]]
) -> Tuple[int, int]:
    """序列无向诱导子图 (S) 上：异质边数、同构边数（两端均为用户/客服前缀）。"""
    hetero = 0
    homo = 0
    for u in S:
        su = _vertex_side(names[u])
        for v in adj_idx[u]:
            if v not in S or v <= u:
                continue
            sv = _vertex_side(names[v])
            if su is None or sv is None:
                continue
            if su == sv:
                homo += 1
            else:
                hetero += 1
    return hetero, homo


def _sequence_typed_edge_pairs(
    names: List[str], adj_idx: List[Set[int]]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """全图序列邻接上的无向边 (u,v,u<v)，按用户/客服划分为同构边列表与异质边列表。"""
    homo: List[Tuple[int, int]] = []
    hetero: List[Tuple[int, int]] = []
    n = len(names)
    for u in range(n):
        su = _vertex_side(names[u])
        for v in adj_idx[u]:
            if v <= u:
                continue
            sv = _vertex_side(names[v])
            if su is None or sv is None:
                continue
            if su == sv:
                homo.append((u, v))
            else:
                hetero.append((u, v))
    return homo, hetero


def _ilp_add_pairwise_and(
    prob: Any,
    z: Any,
    xu: Any,
    xv: Any,
) -> None:
    """z = xu ∧ xv 的 McCormick 整数线性化（xu,xv,z ∈ {0,1}）。"""
    prob += z <= xu
    prob += z <= xv
    prob += z >= xu + xv - 1


def _ilp_add_nogood_exclude_exact_set(
    prob: Any,
    pulp: Any,
    x: Any,
    n: int,
    s_on: Set[int],
) -> None:
    """排除与 s_on（取 1 的下标集）完全相同的解：∑_{i∈s_on} x_i − ∑_{j∉s_on} x_j ≤ |s_on|−1。"""
    off = [j for j in range(n) if j not in s_on]
    prob += (
        pulp.lpSum(x[i] for i in s_on)
        - pulp.lpSum(x[j] for j in off)
        <= len(s_on) - 1
    )


def _sequence_backbone_directed_edges(
    S: Set[str],
    seq_adj: Dict[str, Set[str]],
    global_edges: Dict[Tuple[str, str], int],
) -> Dict[Tuple[str, str], int]:
    """
    仅保留「原序列无向图」上相邻且两端都在 S 中的顶点对，对应的有向弧（来自 global_edges 的出现次数）。
    与挖掘目标中的序列二部正则（z 变量）所用边集一致，不包含 S 上全局转移图的其它弧。
    """
    Sval = set(S)
    out: Dict[Tuple[str, str], int] = {}
    for u in Sval:
        for v in seq_adj.get(u, ()):
            if v not in Sval:
                continue
            for a, b in ((u, v), (v, u)):
                c = global_edges.get((a, b), 0)
                if c > 0:
                    out[(a, b)] = c
    return out


def _path_nodes_from_edge_list(path_edges: List[Tuple[str, str]]) -> List[str]:
    """把按顺序排列的路径边恢复为节点序列；若只有单条边则返回两端节点。"""
    if not path_edges:
        return []
    nodes = [path_edges[0][0], path_edges[0][1]]
    for u, v in path_edges[1:]:
        if nodes[-1] == u:
            nodes.append(v)
        elif nodes[0] == v:
            nodes.insert(0, u)
        else:
            if u not in nodes:
                nodes.append(u)
            if v not in nodes:
                nodes.append(v)
    out: List[str] = []
    seen: Set[str] = set()
    for n in nodes:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _spring_layout_with_centered_main_path(
    nx: Any,
    graph: Any,
    highlight_path_edges: List[Tuple[str, str]],
    *,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Spring layout，但将 main path 固定在画布中心水平主线上。
    其它节点围绕固定主路径做弹簧布局，方便读红色主路径。
    """
    und = graph.to_undirected()
    n_nodes = max(1, graph.number_of_nodes())
    main_nodes = [n for n in _path_nodes_from_edge_list(highlight_path_edges) if n in graph]
    if len(main_nodes) < 2:
        return nx.spring_layout(
            und,
            seed=seed,
            k=2.5 / max(1, math.sqrt(n_nodes)),
            iterations=100,
        )

    span = max(3.0, 0.9 * (len(main_nodes) - 1))
    denom = max(1, len(main_nodes) - 1)
    fixed_pos = {
        node: (-span / 2.0 + span * i / denom, 0.0)
        for i, node in enumerate(main_nodes)
    }
    return nx.spring_layout(
        und,
        pos=fixed_pos,
        fixed=list(fixed_pos.keys()),
        seed=seed,
        k=2.2 / max(1, math.sqrt(n_nodes)),
        iterations=160,
    )


def save_mined_induced_subgraph_figure(
    global_edges: Dict[Tuple[str, str], int],
    seq_adj: Dict[str, Set[str]],
    S: Set[str],
    selected_edges: Optional[Dict[Tuple[str, str], int]],
    out_path: Path,
    *,
    rank: int,
    cover: int,
    m_hyperedges: int,
    num_vertices: int,
    hetero_seq_edges: int = 0,
    homo_seq_edges: int = 0,
) -> None:
    """仅绘制序列骨干上的有向边（global_edges 中在 seq_adj 上相邻的弧），而非 S 上全局转移诱导子图。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as e:
        raise RuntimeError(
            "子图可视化需要 matplotlib 与 networkx，请安装: pip install matplotlib networkx"
        ) from e

    S_valid = set(S)
    sub = nx.DiGraph()
    for n in S_valid:
        sub.add_node(n)
    backbone = (
        dict(selected_edges)
        if selected_edges is not None
        else _sequence_backbone_directed_edges(S_valid, seq_adj, global_edges)
    )
    for (u, v), c in backbone.items():
        sub.add_edge(u, v, weight=c, count=c)
    n_n = sub.number_of_nodes()
    n_e = sub.number_of_edges()
    if n_n == 0:
        plt.close("all")
        return
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig_w = max(12.0, 1.2 * n_n)
    fig_h = max(9.0, 0.9 * n_n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    und = sub.to_undirected()
    pos = nx.spring_layout(und, seed=42, k=2.5 / max(1, math.sqrt(n_n)), iterations=100)
    colors = [_node_role_color(n) for n in sub.nodes()]
    nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=1200, alpha=0.92, ax=ax)
    short_labels = {n: (n if len(n) <= 18 else n[:15] + "…") for n in sub.nodes()}
    nx.draw_networkx_labels(sub, pos, labels=short_labels, font_size=7, ax=ax)
    weights = [sub[u][v].get("weight", 1) for u, v in sub.edges()]
    w_max = max(weights) if weights else 1
    widths = [1.0 + 4.0 * (w / w_max) for w in weights]
    nx.draw_networkx_edges(
        sub,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=18,
        width=widths,
        edge_color="#555555",
        alpha=0.85,
        connectionstyle="arc3,rad=0.08",
    )
    elab = {(u, v): str(sub[u][v].get("weight", sub[u][v].get("count", 1))) for u, v in sub.edges()}
    nx.draw_networkx_edge_labels(sub, pos, edge_labels=elab, font_size=6, ax=ax, rotate=False)
    ax.set_title(
        f"挖掘结果 #{rank}  |  MILP 选中 |S|={num_vertices}  |  超边覆盖 {cover}/{m_hyperedges}  "
        f"|  序列异质/同构 {hetero_seq_edges}/{homo_seq_edges}  "
        f"|  序列骨干有向边  |V|={n_n}  |E|={n_e}",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _induced_components_undirected(U: Set[str], adj: Dict[str, Set[str]]) -> List[Set[str]]:
    """U 在序列无向图上的诱导子图的连通分量。"""
    U = set(U)
    if not U:
        return []
    unseen = set(U)
    comps: List[Set[str]] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        comp: Set[str] = set()
        while stack:
            u = stack.pop()
            if u not in U or u in comp:
                continue
            comp.add(u)
            for v in adj.get(u, ()):
                if v in U:
                    stack.append(v)
        unseen -= comp
        comps.append(comp)
    return comps


def _shortest_path_between_vertex_sets(
    adj: Dict[str, Set[str]], A: Set[str], B: Set[str]
) -> Optional[List[str]]:
    """全图无向最短路（可经过 U 外顶点）。若不存在则返回 None。"""
    if not A or not B:
        return None
    if A & B:
        x = next(iter(A & B))
        return [x]
    parent: Dict[str, Optional[str]] = {}
    q: deque[str] = deque()
    for a in A:
        parent[a] = None
        q.append(a)
    target: Optional[str] = None
    while q:
        u = q.popleft()
        if u in B:
            target = u
            break
        for v in adj.get(u, ()):
            if v not in parent:
                parent[v] = u
                q.append(v)
    if target is None:
        return None
    path_rev: List[str] = []
    cur: Optional[str] = target
    while cur is not None:
        path_rev.append(cur)
        cur = parent[cur]
    path_rev.reverse()
    return path_rev


def merge_topk_vertex_sets_into_connected_component(
    adj_global: Dict[str, Set[str]],
    named_sets: Sequence[Set[str]],
) -> Tuple[Set[str], List[str], bool]:
    """
    将多个顶点集合并后，若其在序列无向图上不连通，则反复在全局 adj 上取最短路，
    把沿路顶点并入，直到诱导子图只有一个连通分量（或全局不连通导致无法连接）。
    返回 (最终顶点集, 为连接而新加入的顶点按加入顺序列表, 是否完全连通成功)。
    """
    U: Set[str] = set()
    for s in named_sets:
        U |= set(s)
    added_order: List[str] = []
    ok = True
    safety = 0
    max_rounds = max(5000, 5 * len(U) + 50)
    while safety < max_rounds:
        safety += 1
        comps = _induced_components_undirected(U, adj_global)
        if len(comps) <= 1:
            break
        c0, c1 = comps[0], comps[1]
        path = _shortest_path_between_vertex_sets(adj_global, c0, c1)
        if path is None:
            ok = False
            break
        for v in path:
            if v not in U:
                added_order.append(v)
            U.add(v)
    return U, added_order, ok


def _heaviest_directed_path_as_edge_list(sub: Any) -> List[Tuple[str, str]]:
    """
    在有向诱导子图上取「权重和最大」的一条有向路径（以边 weight/count 求和）。
    若图为 DAG，则用最长路；否则退化为权重最大的单条有向边（路径长度为 1）。
    """
    import networkx as nx

    def edge_w(u: str, v: str) -> float:
        d = sub[u][v]
        return float(d.get("weight", d.get("count", 1)))

    if sub.number_of_edges() == 0:
        return []
    if nx.is_directed_acyclic_graph(sub):
        path_nodes = nx.dag_longest_path(sub, weight="weight")
        if len(path_nodes) < 2:
            return []
        return [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]
    best_pair: Optional[Tuple[str, str]] = None
    best_w = -1.0
    for u, v in sub.edges():
        w = edge_w(u, v)
        if w > best_w:
            best_w = w
            best_pair = (u, v)
    return [best_pair] if best_pair else []


def save_merged_topk_induced_subgraph_figure(
    global_edges: Dict[Tuple[str, str], int],
    seq_adj: Dict[str, Set[str]],
    S_merged: Set[str],
    selected_edges: Optional[Dict[Tuple[str, str], int]],
    highlight_path_edges: List[Tuple[str, str]],
    out_path: Path,
    *,
    num_solutions: int,
    per_solution_sizes: Optional[List[int]] = None,
    connector_added: int,
    merged_connected_ok: bool,
    path_mode: str,
) -> None:
    """Top-k 合并后的顶点集上，仅绘制序列骨干有向边；高亮权重最高路径（在该子图上）。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as e:
        raise RuntimeError(
            "子图可视化需要 matplotlib 与 networkx，请安装: pip install matplotlib networkx"
        ) from e

    S_valid = set(S_merged)
    sub = nx.DiGraph()
    for n in S_valid:
        sub.add_node(n)
    backbone = (
        dict(selected_edges)
        if selected_edges is not None
        else _sequence_backbone_directed_edges(S_valid, seq_adj, global_edges)
    )
    for (u, v), c in backbone.items():
        sub.add_edge(u, v, weight=c, count=c)
    hi_set = set(highlight_path_edges)
    n_n = sub.number_of_nodes()
    n_e = sub.number_of_edges()
    if n_n == 0:
        plt.close("all")
        return
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig_w = max(12.0, 1.2 * n_n)
    fig_h = max(9.0, 0.9 * n_n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    pos = _spring_layout_with_centered_main_path(nx, sub, highlight_path_edges, seed=42)
    colors = [_node_role_color(n) for n in sub.nodes()]
    nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=1200, alpha=0.92, ax=ax)
    short_labels = {n: (n if len(n) <= 18 else n[:15] + "…") for n in sub.nodes()}
    nx.draw_networkx_labels(sub, pos, labels=short_labels, font_size=7, ax=ax)

    normal_edges = [(u, v) for u, v in sub.edges() if (u, v) not in hi_set]
    hi_edges = [(u, v) for u, v in sub.edges() if (u, v) in hi_set]

    if normal_edges:
        w_norm = [sub[u][v].get("weight", 1) for u, v in normal_edges]
        w_max = max(w_norm) if w_norm else 1
        widths_norm = [1.0 + 3.5 * (float(w) / w_max) for w in w_norm]
        nx.draw_networkx_edges(
            sub,
            pos,
            edgelist=normal_edges,
            ax=ax,
            arrows=True,
            arrowsize=16,
            width=widths_norm,
            edge_color="#555555",
            alpha=0.72,
            connectionstyle="arc3,rad=0.08",
        )
    if hi_edges:
        w_hi = [sub[u][v].get("weight", 1) for u, v in hi_edges]
        w_max_h = max(w_hi) if w_hi else 1
        widths_hi = [3.5 + 6.5 * (float(w) / w_max_h) for w in w_hi]
        nx.draw_networkx_edges(
            sub,
            pos,
            edgelist=hi_edges,
            ax=ax,
            arrows=True,
            arrowsize=22,
            width=widths_hi,
            edge_color="#c1121f",
            alpha=0.95,
            style="solid",
            connectionstyle="arc3,rad=0.12",
        )

    elab = {(u, v): str(sub[u][v].get("weight", sub[u][v].get("count", 1))) for u, v in sub.edges()}
    nx.draw_networkx_edge_labels(sub, pos, edge_labels=elab, font_size=6, ax=ax, rotate=False)

    conn_note = "序列图已补齐为单一连通分量" if merged_connected_ok else "警告：序列图上无法连接所有块，图为多块"
    if per_solution_sizes:
        sz_part = ",".join(str(s) for s in per_solution_sizes[:10])
        if len(per_solution_sizes) > 10:
            sz_part += "…"
        sz_note = f"各解|S|=[{sz_part}]"
    else:
        sz_note = "每解 |S| 由 MILP 决定"
    title = (
        f"Top-{num_solutions} 合并（序列骨干边）  |  {sz_note}  |  合并后 |V|={n_n}  |E|={n_e}"
        f"  |  补点 {connector_added}  |  {conn_note}\n"
        f"红粗边：{path_mode}"
    )
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _adj_idx_for_names(names: List[str], adj_global: Dict[str, Set[str]]) -> List[Set[int]]:
    """将全局序列邻接限制在 names 子集上，得到下标图。"""
    n = len(names)
    name_set = set(names)
    vid = {names[i]: i for i in range(n)}
    out: List[Set[int]] = [set() for _ in range(n)]
    for i, a in enumerate(names):
        for b in adj_global.get(a, ()):
            if b in name_set:
                j = vid[b]
                if i != j:
                    out[i].add(j)
    return out


def _is_connected_idx(S: Set[int], adj_idx: List[Set[int]]) -> bool:
    if len(S) <= 1:
        return True
    start = next(iter(S))
    stack = [start]
    seen: Set[int] = {start}
    while stack:
        v = stack.pop()
        for u in adj_idx[v]:
            if u in S and u not in seen:
                seen.add(u)
                stack.append(u)
    return len(seen) == len(S)


def _is_connected_by_selected_edges(
    S_names: Set[str], selected_edges: Dict[Tuple[str, str], int]
) -> bool:
    """检查被选节点是否在被选有向边（按无向看）上构成单一连通分量。"""
    if len(S_names) <= 1:
        return True
    adj: Dict[str, Set[str]] = {n: set() for n in S_names}
    for u, v in selected_edges:
        if u in S_names and v in S_names:
            adj[u].add(v)
            adj[v].add(u)
    start = next(iter(S_names))
    stack = [start]
    seen = {start}
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(S_names)


def high_coverage_vertices(
    H: SessionHypergraph, min_hyperedge_ratio: float
) -> List[Tuple[str, int]]:
    """
    若顶点 v 出现在至少 min_hyperedge_ratio * |E| 条超边中（比例按实数比较），
    则归入高覆盖集。返回 (顶点名, 出现超边数)，按出现次数降序、名字升序。
    """
    m = len(H.hyperedges)
    if m == 0:
        return []
    if not 0.0 < min_hyperedge_ratio <= 1.0:
        raise ValueError("min_hyperedge_ratio 须满足 0 < ratio <= 1")
    out: List[Tuple[str, int]] = []
    for v, he_ids in H.vertex_incidence.items():
        c = len(he_ids)
        if c / m >= min_hyperedge_ratio:
            out.append((v, c))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def reduced_hypergraph(H: SessionHypergraph, remove: Set[str]) -> SessionHypergraph:
    """
    去掉指定顶点集后的缩减超图：超边变为 e \\ remove，空超边丢弃。
    仅用于在剩余部分上做覆盖搜索。
    """
    hyperedges: List[Hyperedge] = []
    vertex_incidence: Dict[str, List[int]] = {}
    for e in H.hyperedges:
        nv = frozenset(v for v in e.vertices if v not in remove)
        if not nv:
            continue
        he_idx = len(hyperedges)
        hyperedges.append(
            Hyperedge(record_index=e.record_index, session_id=e.session_id, vertices=nv)
        )
        for v in nv:
            vertex_incidence.setdefault(v, []).append(he_idx)
    return SessionHypergraph(hyperedges=hyperedges, vertex_incidence=vertex_incidence)


def _index_hypergraph(H: SessionHypergraph) -> Tuple[List[str], List[Set[int]], List[Set[int]]]:
    """顶点名称列表、每条超边对应的顶点下标集合、每个顶点关联的超边下标集合。"""
    names = sorted(H.vertices)
    vid = {n: i for i, n in enumerate(names)}
    m = len(H.hyperedges)
    edge_verts: List[Set[int]] = []
    for e in H.hyperedges:
        edge_verts.append({vid[v] for v in e.vertices})
    v_to_e: List[Set[int]] = [set() for _ in range(len(names))]
    for ei, vs in enumerate(edge_verts):
        for v in vs:
            v_to_e[v].add(ei)
    return names, edge_verts, v_to_e


def _min_vertices_to_count_as_hyperedge_cover(num_in_edge: int, ratio: float) -> int:
    """
    覆盖超边 e 至少要在 S 中含 e 内多少个顶点：⌈ratio·|e|⌉，并夹在 [1, |e|] 内。
    ratio=1 时为全包含；ratio=0.9 时为「至少 90% 节点」（对小超边可能等于 |e|）。
    """
    if num_in_edge <= 0:
        return 0
    req = int(math.ceil(float(ratio) * num_in_edge - 1e-12))
    return min(num_in_edge, max(1, req))


def _hyperedge_covered_by_set(S: Set[int], ev: Set[int], ratio: float) -> bool:
    if not ev:
        return False
    need = _min_vertices_to_count_as_hyperedge_cover(len(ev), ratio)
    return len(S & ev) >= need


def _hyperedge_cover_count(S: Set[int], edge_verts: List[Set[int]], ratio: float) -> int:
    return sum(1 for ev in edge_verts if _hyperedge_covered_by_set(S, ev, ratio))


def solve_ilp_topk_hyperedge_bipartite(
    H: SessionHypergraph,
    topk: int,
    adj_global: Dict[str, Set[str]],
    global_edges: Dict[Tuple[str, str], int],
    *,
    gamma: float = 1.0,
    lambda_node: float = 0.5,
    beta_homo: float = 0.3,
    alpha_hetero: float = 0.5,
    balance: float = 1.0,
    hyperedge_cover_ratio: float = 0.8,
    edge_weight_scale: float = 0.0,
    lambda_edge: float = 10.0,
    cycle_edge_penalty: float = 5.0,
    cbc_time_limit: int = 20,
    cbc_msg: bool = False,
) -> Tuple[List[Tuple[Set[int], int, int, int, float, Dict[Tuple[str, str], int], int, float, str]], str, List[str]]:
    """
    PuLP/CBC：**不**固定 |S|；超边覆盖 y_e 与 x 联动：
      |S∩e| ≥ ⌈ρ|e|⌉ 时 y_e 可为 1，否则 y_e=0（两组线性约束）。
    同时搜索边集合：候选有向边 edge_on(u,v) 必须两端节点都被选择；边数量通过 λ_edge·∑edge_on
    作为正则项控制。被选边默认服从拓扑序，违反拓扑序的边作为 cycle_exception 支付惩罚。
    每个被选节点还必须至少关联一条被选边，并通过一组全局流约束保证所有被选节点
    位于同一个由被选边构成的连通分量中。

    目标最大化：
      γ·∑_e y_e + balance·(α·∑ z_异质 − β·∑ z_同构) − λ·∑_v x_v

    Top-K 用 nogood 割排除已出现的 0–1 解。

    返回 [(S_idx, cover, hetero, homo, node_objective, selected_edges, cycle_exceptions, edge_objective, edge_method), ...]。
    """
    try:
        import pulp
    except ImportError as e:
        raise RuntimeError(
            "整数规划求解需要 PuLP，请安装: pip install pulp"
        ) from e

    names, edge_verts, _ = _index_hypergraph(H)
    n = len(names)
    adj_idx = _adj_idx_for_names(names, adj_global)
    homo_pairs, hetero_pairs = _sequence_typed_edge_pairs(names, adj_idx)

    if topk <= 0:
        return [], "topk<=0", names
    if n == 0:
        return [], "|V|=0", names
    if not 0.0 < hyperedge_cover_ratio <= 1.0:
        return [], "hyperedge_cover_ratio 须在 (0, 1] 内", names
    if lambda_edge < 0:
        return [], "lambda_edge 须 >= 0", names

    m_he = len(H.hyperedges)
    all_name_set = set(names)
    candidate_edge_weights = _sequence_backbone_directed_edges(
        all_name_set, adj_global, global_edges
    )
    vid = {name: i for i, name in enumerate(names)}
    edge_items = [
        ((u, v), w)
        for (u, v), w in sorted(candidate_edge_weights.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        if u in vid and v in vid
    ]
    edge_idx_pairs = [(vid[u], vid[v]) for (u, v), _w in edge_items]
    incident_edge_ids: List[List[int]] = [[] for _ in range(n)]
    for edge_i, (u, v) in enumerate(edge_idx_pairs):
        incident_edge_ids[u].append(edge_i)
        incident_edge_ids[v].append(edge_i)
    excluded: List[Set[int]] = []
    solutions: List[
        Tuple[Set[int], int, int, int, float, Dict[Tuple[str, str], int], int, float, str]
    ] = []
    solver_name = "CBC"

    for rank_round in range(topk):
        prob = pulp.LpProblem(
            f"session_hypergraph_bipartite_{rank_round}", pulp.LpMaximize
        )
        x = pulp.LpVariable.dicts("x", range(n), lowBound=0, upBound=1, cat=pulp.LpBinary)
        y = pulp.LpVariable.dicts(
            "y_hyper", range(m_he), lowBound=0, upBound=1, cat=pulp.LpBinary
        )
        z_homo = pulp.LpVariable.dicts(
            "z_homo",
            range(len(homo_pairs)),
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary,
        )
        z_hetero = pulp.LpVariable.dicts(
            "z_hetero",
            range(len(hetero_pairs)),
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary,
        )
        edge_on = pulp.LpVariable.dicts(
            "edge_on",
            range(len(edge_items)),
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary,
        )
        cycle_exception = pulp.LpVariable.dicts(
            "cycle_exception",
            range(len(edge_items)),
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary,
        )
        topo_order = pulp.LpVariable.dicts(
            "topo_order",
            range(n),
            lowBound=0,
            upBound=max(0, n - 1),
            cat=pulp.LpInteger,
        )
        conn_root = pulp.LpVariable.dicts(
            "conn_root",
            range(n),
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary,
        )
        conn_flow = pulp.LpVariable.dicts(
            "conn_flow",
            range(2 * len(edge_items)),
            lowBound=0,
            upBound=n,
            cat=pulp.LpContinuous,
        )

        for ei, ev in enumerate(edge_verts):
            if not ev:
                prob += y[ei] == 0
                continue
            ne = len(ev)
            min_r = _min_vertices_to_count_as_hyperedge_cover(ne, hyperedge_cover_ratio)
            if min_r <= 0:
                prob += y[ei] == 0
                continue
            sum_x_e = pulp.lpSum(x[j] for j in ev)
            prob += sum_x_e >= min_r * y[ei]
            prob += sum_x_e <= (min_r - 1) + ne * y[ei]

        for ti, (u, v) in enumerate(homo_pairs):
            _ilp_add_pairwise_and(prob, z_homo[ti], x[u], x[v])
        for ti, (u, v) in enumerate(hetero_pairs):
            _ilp_add_pairwise_and(prob, z_hetero[ti], x[u], x[v])

        big_m = max(1, n)
        for edge_i, ((u_name, v_name), _w) in enumerate(edge_items):
            u, v = vid[u_name], vid[v_name]
            prob += edge_on[edge_i] <= x[u]
            prob += edge_on[edge_i] <= x[v]
            prob += cycle_exception[edge_i] <= edge_on[edge_i]
            prob += (
                topo_order[u] + 1
                <= topo_order[v]
                + big_m * cycle_exception[edge_i]
                + big_m * (1 - edge_on[edge_i])
            )
        for node_i in range(n):
            if incident_edge_ids[node_i]:
                prob += x[node_i] <= pulp.lpSum(edge_on[e] for e in incident_edge_ids[node_i])
            else:
                prob += x[node_i] == 0

        prob += pulp.lpSum(conn_root[i] for i in range(n)) <= 1
        for node_i in range(n):
            prob += conn_root[node_i] <= x[node_i]
            prob += x[node_i] <= pulp.lpSum(conn_root[j] for j in range(n))

        flow_arcs: List[Tuple[int, int, int]] = []
        for edge_i, (u, v) in enumerate(edge_idx_pairs):
            flow_arcs.append((edge_i, u, v))
            flow_arcs.append((edge_i, v, u))
        flow_arcs_in: List[List[int]] = [[] for _ in range(n)]
        flow_arcs_out: List[List[int]] = [[] for _ in range(n)]
        for arc_i, (edge_i, u, v) in enumerate(flow_arcs):
            prob += conn_flow[arc_i] <= n * edge_on[edge_i]
            flow_arcs_out[u].append(arc_i)
            flow_arcs_in[v].append(arc_i)
        for node_i in range(n):
            prob += (
                pulp.lpSum(conn_flow[a] for a in flow_arcs_in[node_i])
                - pulp.lpSum(conn_flow[a] for a in flow_arcs_out[node_i])
                >= x[node_i] - n * conn_root[node_i]
            )
        for prev in excluded:
            _ilp_add_nogood_exclude_exact_set(prob, pulp, x, n, prev)

        obj_cov = (
            0
            if m_he == 0
            else gamma * pulp.lpSum(y[ei] for ei in range(m_he))
        )
        obj_bipartite = balance * (
            alpha_hetero * pulp.lpSum(z_hetero[t] for t in range(len(hetero_pairs)))
            - beta_homo * pulp.lpSum(z_homo[t] for t in range(len(homo_pairs)))
        )
        obj_edges = (
            edge_weight_scale
            * pulp.lpSum(float(w) * edge_on[e] for e, ((_u, _v), w) in enumerate(edge_items))
            - cycle_edge_penalty
            * pulp.lpSum(cycle_exception[e] for e in range(len(edge_items)))
            - lambda_edge * pulp.lpSum(edge_on[e] for e in range(len(edge_items)))
        )
        obj_size = lambda_node * pulp.lpSum(x[i] for i in range(n))
        prob += obj_cov + obj_bipartite + obj_edges - obj_size

        solver = pulp.PULP_CBC_CMD(msg=1 if cbc_msg else 0)
        if cbc_time_limit and cbc_time_limit > 0:
            solver = pulp.PULP_CBC_CMD(msg=1 if cbc_msg else 0, timeLimit=int(cbc_time_limit))
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            if rank_round == 0:
                return (
                    [],
                    f"PuLP 状态非 Optimal: {status}（n={n}, m={m_he}）",
                    names,
                )
            break

        S_idx: Set[int] = {
            i for i in range(n) if pulp.value(x[i]) is not None and pulp.value(x[i]) > 0.5
        }

        selected_edges: Dict[Tuple[str, str], int] = {}
        cycle_exceptions = 0
        raw_edge_gain = 0.0
        for edge_i, ((u_name, v_name), w) in enumerate(edge_items):
            if pulp.value(edge_on[edge_i]) is not None and pulp.value(edge_on[edge_i]) > 0.5:
                selected_edges[(u_name, v_name)] = w
                raw_edge_gain += edge_weight_scale * float(w)
                if (
                    pulp.value(cycle_exception[edge_i]) is not None
                    and pulp.value(cycle_exception[edge_i]) > 0.5
                ):
                    cycle_exceptions += 1
        cover = _hyperedge_cover_count(S_idx, edge_verts, hyperedge_cover_ratio)
        het, hom = _sequence_bipartite_edge_counts(names, S_idx, adj_idx)
        obj_val = float(pulp.value(prob.objective) or 0.0)
        edge_obj = raw_edge_gain - cycle_edge_penalty * cycle_exceptions - lambda_edge * len(selected_edges)
        edge_method = (
            f"同一 MILP 内边搜索：候选边 {len(edge_items)}，选中 {len(selected_edges)}"
            f"，边数量正则 λ_edge={lambda_edge:g}，"
            f"环例外边 {cycle_exceptions}，edge_scale={edge_weight_scale:g}, "
            f"cycle_penalty={cycle_edge_penalty:g}"
        )
        solutions.append(
            (
                S_idx,
                cover,
                het,
                hom,
                obj_val,
                selected_edges,
                cycle_exceptions,
                edge_obj,
                edge_method,
            )
        )
        excluded.append(set(S_idx))

    method = (
        f"PuLP + {solver_name} 整数规划，无固定 |S|，超边覆盖阈值 ρ={hyperedge_cover_ratio:g} "
        f"(按节点命中比例计覆盖)；同时搜索边，且被选节点需由被选边连成单一分量；"
        f"边数量用 λ_edge={lambda_edge:g} 正则；"
        f"+ balance·(α·异质−β·同构) + 边权收益 − λ_node·∑x − λ_edge·∑edge；"
        f"Top-{len(solutions)} 为 nogood 割迭代"
    )
    return solutions, method, names


def _print_topk_solutions(
    H: SessionHypergraph,
    solutions: List[
        Tuple[Set[int], int, int, int, float, Dict[Tuple[str, str], int], int, float, str]
    ],
    method: str,
    names: List[str],
    title: str,
    hyperedge_cover_ratio: float,
    adj_idx: Optional[List[Set[int]]] = None,
    global_edges: Optional[Dict[Tuple[str, str], int]] = None,
    sequence_adj: Optional[Dict[str, Set[str]]] = None,
    viz_out_dir: Optional[Path] = None,
) -> None:
    m = len(H.hyperedges)
    print(f"=== {title} ===")
    print(
        f"  超边覆盖阈值 ρ={hyperedge_cover_ratio:g}：当且仅当命中 |S∩e| ≥ ⌈ρ·|e|⌉ 时计为覆盖该超边。"
    )
    print(
        "  约束: 无固定 |S|（由 λ·∑x 与 γ、balance 等共同决定规模）；"
        "所有被选节点必须通过被选边连成单一分量；"
        f"输出 {len(solutions)} 组互异解（nogood 割迭代）。"
    )
    print(
        "  二部性指标（序列诱导子图、两端均为「用户:」「客服:」）："
        "异质=用户–客服，同构=用户–用户或客服–客服。"
    )
    print(f"  求解: {method}")
    print()
    vid = {names[i]: i for i in range(len(names))}
    for rank, (
        S_idx,
        cover,
        hetero_e,
        homo_e,
        obj_v,
        selected_edges,
        cycle_exceptions,
        edge_obj,
        edge_method,
    ) in enumerate(solutions, start=1):
        S_names = sorted(names[i] for i in S_idx)
        selected_edge_connected = _is_connected_by_selected_edges(set(S_names), selected_edges)
        conn_note = "是" if selected_edge_connected else "否"
        print(
            f"  --- 第 {rank} 名 | |S|={len(S_idx)} | 覆盖 {cover} / {m} | "
            f"序列异质/同构 {hetero_e}/{homo_e} | "
            f"被选边连通: {conn_note} | MILP 目标 ≈ {obj_v:.4g} ---"
        )
        print(
            f"    边集: 选中 {len(selected_edges)} 条有向边，环例外边 {cycle_exceptions} 条，"
            f"边目标 ≈ {edge_obj:.4g}；{edge_method}"
        )
        for v in S_names:
            print(f"    - {v}")
        print(f"    被选中有向边（edge_on=1，共 {len(selected_edges)} 条）:")
        if selected_edges:
            for (a, b), w in sorted(selected_edges.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"      {a} -> {b}  weight={w}")
        else:
            print("      （无被选中边：例如 |S|≤1、无候选边，或边数量正则过强）")
        if (
            global_edges is not None
            and sequence_adj is not None
            and viz_out_dir is not None
        ):
            S_set = set(S_names)
            nv = len(S_set)
            rho_tag = int(round(hyperedge_cover_ratio * 100))
            fn = (
                f"mined_rank{rank:02d}_cover{cover}_rho{rho_tag}_het{hetero_e}_hom{homo_e}_S{nv}.png"
            )
            vpath = viz_out_dir / fn
            try:
                save_mined_induced_subgraph_figure(
                    global_edges,
                    sequence_adj,
                    S_set,
                    selected_edges,
                    vpath,
                    rank=rank,
                    cover=cover,
                    m_hyperedges=m,
                    num_vertices=nv,
                    hetero_seq_edges=hetero_e,
                    homo_seq_edges=homo_e,
                )
                print(f"    序列骨干有向边子图已保存: {vpath.resolve()}")
            except RuntimeError as err:
                print(f"    （跳过绘图）{err}")
        missed: List[Tuple[Hyperedge, int, int]] = []
        for e in H.hyperedges:
            ev_i = {vid[v] for v in e.vertices if v in vid}
            need = _min_vertices_to_count_as_hyperedge_cover(len(ev_i), hyperedge_cover_ratio)
            hit = len(S_idx & ev_i)
            if not _hyperedge_covered_by_set(S_idx, ev_i, hyperedge_cover_ratio):
                missed.append((e, hit, need))
        if missed:
            print(f"    未满足 ρ-覆盖的超边 ({len(missed)} 条)（命中 / 需要 ⌈ρ·|e|⌉）:")
            for e, hit, need in missed:
                print(
                    f"      - record_index={e.record_index} session_id={e.session_id}  "
                    f"|e|={len(e.vertices)}  命中={hit}  需≥{need}"
                )
        else:
            print("    缩减超图：全部超边均满足 ρ-覆盖。")
        print()


def _print_hub_section(
    hub_entries: List[Tuple[str, int]], m: int, threshold: float
) -> None:
    pct = 100.0 * threshold
    print(f"=== 部分一：高覆盖节点（单节点覆盖超边比例 ≥ {pct:g}% ，|E|={m}）===")
    if not hub_entries:
        print(f"  （无：没有节点达到 ≥ {threshold:.4g} 的覆盖比例）")
    else:
        print(f"  共 {len(hub_entries)} 个（先抽出，不参与后续 MILP）")
        for v, c in hub_entries:
            r = c / m if m else 0.0
            print(f"    - {v}  （{c}/{m} = {r:.2%}）")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "会话超图：PuLP 整数规划，不固定 |S|，最大化 γ·覆盖 + balance·(α·异质−β·同构) − λ·|S|"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_JSON,
        help=f"operator_results.json 路径（默认: {DEFAULT_JSON}）",
    )
    parser.add_argument(
        "--hyperedge-cover-ratio",
        type=float,
        default=0.8,
        metavar="RHO",
        help="超边 ρ-覆盖：须 |S∩e|≥⌈ρ·|e|⌉ 才算覆盖（0<RHO≤1；默认 0.8）",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="超边覆盖项系数 γ（每条被 ρ-覆盖的超边 +γ；默认 1）",
    )
    parser.add_argument(
        "--lambda-node",
        type=float,
        default=0.5,
        help="节点入场费 λ（目标中 −λ·∑x；须 >0 否则易退化为全集，默认 0.5）",
    )
    parser.add_argument(
        "--beta-homo",
        type=float,
        default=1.0,
        help="同构序列边惩罚系数 β（进入 balance·(…) 中为 −balance·β·∑z_同构；默认 1）",
    )
    parser.add_argument(
        "--alpha-hetero",
        type=float,
        default=1.0,
        help="异质序列边奖励系数 α（+balance·α·∑z_异质；默认 1）",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0,
        help="二部正则总强度：放大 (α·异质−β·同构) 相对覆盖项；调大更「洁癖」二部（默认 0）",
    )
    parser.add_argument(
        "--edge-weight-scale",
        type=float,
        default=0.0,
        help="同一 MILP 中边集搜索的边权奖励缩放系数；0 表示边只作为连通结构进入目标（默认 0）",
    )
    parser.add_argument(
        "--lambda-edge",
        type=float,
        default=10.0,
        help="边数量正则 λ_edge（目标中 −λ_edge·∑edge_on；越大边越少，默认 10）",
    )
    parser.add_argument(
        "--cycle-edge-penalty",
        type=float,
        default=5.0,
        help="同一 MILP 中保留一条违反拓扑序的环例外边需支付的惩罚；越大越接近 DAG（默认 5）",
    )
    parser.add_argument(
        "--cbc-time-limit",
        type=int,
        default=20,
        metavar="SEC",
        help="CBC 时间上限（秒）；0 表示不限制（默认 20）",
    )
    parser.add_argument(
        "--cbc-msg",
        action="store_true",
        help="打印 CBC 求解日志",
    )
    parser.add_argument(
        "--hub-threshold",
        type=float,
        default=0.9,
        metavar="R",
        help="高覆盖节点：出现超边数 / |E| ≥ R 时先抽出（默认 0.9 即 90%%）",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1,
        metavar="K",
        help="缩减图上输出 MILP 前 K 个互异最优解（nogood 迭代，默认 1）",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=DEFAULT_VIZ_DIR,
        help=f"每个挖掘结果在全局有向图上的诱导子图 PNG 输出目录（默认: {DEFAULT_VIZ_DIR}）",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="不生成子图可视化",
    )
    args = parser.parse_args()
    if not 0.0 < args.hub_threshold <= 1.0:
        raise SystemExit("--hub-threshold 须在区间 (0, 1] 内，例如 0.9")
    if args.top < 1:
        raise SystemExit("--top 须 >= 1")
    if args.lambda_node <= 0:
        raise SystemExit("--lambda-node 须 > 0，否则目标易在选全集上最优（规模无惩罚）")
    if not 0.0 < args.hyperedge_cover_ratio <= 1.0:
        raise SystemExit("--hyperedge-cover-ratio 须在区间 (0, 1] 内，例如 1 或 0.9")
    if args.edge_weight_scale < 0:
        raise SystemExit("--edge-weight-scale 须 >= 0")
    if args.lambda_edge < 0:
        raise SystemExit("--lambda-edge 须 >= 0")
    if args.cycle_edge_penalty < 0:
        raise SystemExit("--cycle-edge-penalty 须 >= 0")
    path: Path = args.input
    if not path.is_file():
        raise SystemExit(f"找不到输入文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not isinstance(results, list):
        raise SystemExit("JSON 根节点应为数组")

    H = SessionHypergraph.from_operator_results(results)
    names_full, _, _ = _index_hypergraph(H)
    n_full, m = len(names_full), len(H.hyperedges)

    adj_global = build_sequence_undirected_adj(results)
    global_edges = build_global_edge_weights(results)
    n_ge = len({u for u, _ in global_edges} | {v for _, v in global_edges})

    print("=== 全局超图概要 ===")
    print(f"  顶点数 |V| = {n_full}")
    print(f"  超边数 |E| = {m}")
    print(
        f"  全局有向转移（与 graph_construction 一致）: "
        f"涉及约 {n_ge} 个节点, {len(global_edges)} 条不同有向弧（按次数加权）"
    )
    print("  原序列无向图：相邻操作连边（MILP 二部正则与合并图补点用）")
    print(
        f"  超边 ρ-覆盖阈值: {args.hyperedge_cover_ratio:g} "
        f"（|S∩e| ≥ ⌈ρ·|e|⌉ 时计为覆盖）"
    )
    print()

    hub_entries = high_coverage_vertices(H, args.hub_threshold)
    hub_names = [v for v, _ in hub_entries]
    U_set = set(hub_names)
    _print_hub_section(hub_entries, m, args.hub_threshold)

    H_red = reduced_hypergraph(H, U_set)
    n_red = len(H_red.vertices)
    m_red = len(H_red.hyperedges)
    emptied = m - m_red

    print("=== 缩减超图（去掉高覆盖节点后）===")
    print(f"  剩余顶点数 |V'| = {n_red}")
    print(
        f"  非空缩减超边数 |E'| = {m_red}"
        f"（另有 {emptied} 条原超边在去掉高覆盖节点后为空，仅由高覆盖节点刻画）"
    )
    print()

    if n_red == 0:
        print("=== 部分二：剩余域上无可选顶点，跳过 MILP 求解 ===")
    elif m_red == 0:
        print("=== 部分二：缩减后无超边，跳过 MILP 求解 ===")
    else:
        topk = max(1, args.top)
        try:
            sols, method, names_red = solve_ilp_topk_hyperedge_bipartite(
                H_red,
                topk=topk,
                adj_global=adj_global,
                global_edges=global_edges,
                gamma=args.gamma,
                lambda_node=args.lambda_node,
                beta_homo=args.beta_homo,
                alpha_hetero=args.alpha_hetero,
                balance=args.balance,
                hyperedge_cover_ratio=args.hyperedge_cover_ratio,
                edge_weight_scale=args.edge_weight_scale,
                lambda_edge=args.lambda_edge,
                cycle_edge_penalty=args.cycle_edge_penalty,
                cbc_time_limit=args.cbc_time_limit,
                cbc_msg=args.cbc_msg,
            )
        except RuntimeError as err:
            raise SystemExit(str(err)) from err
        adj_idx_red = _adj_idx_for_names(names_red, adj_global)
        viz_dir: Optional[Path] = None if args.no_viz else args.viz_dir
        if viz_dir is None:
            print("=== 可视化已关闭：检测到 --no-viz，本轮不会保存 PNG ===")
        else:
            print(f"=== 可视化已开启：PNG 输出目录 {viz_dir.resolve()} ===")
        _print_topk_solutions(
            H_red,
            sols,
            method,
            names_red,
            title=f"部分二：剩余域 Top-{len(sols)} 个 MILP 子图（无固定 |S|；e' = e \\ 高覆盖集）",
            hyperedge_cover_ratio=args.hyperedge_cover_ratio,
            adj_idx=adj_idx_red,
            global_edges=global_edges,
            sequence_adj=adj_global,
            viz_out_dir=viz_dir,
        )

        if viz_dir is not None and sols:
            named_per_sol = [{names_red[i] for i in S_idx} for S_idx, *_rest in sols]
            merged_selected_edges: Dict[Tuple[str, str], int] = {}
            for _S_idx, *_prefix, selected_edges, _cycle_ex, _edge_obj, _edge_method in sols:
                for edge, weight in selected_edges.items():
                    merged_selected_edges[edge] = merged_selected_edges.get(edge, 0) + weight
            S_merged, added_connectors, merged_ok = merge_topk_vertex_sets_into_connected_component(
                adj_global, named_per_sol
            )
            highlight: List[Tuple[str, str]] = []
            path_mode = "诱导子图无边"
            try:
                import networkx as nx

                subm = nx.DiGraph()
                for n in S_merged:
                    subm.add_node(n)
                for (u, v), c in merged_selected_edges.items():
                    subm.add_edge(u, v, weight=c, count=c)
                is_dag = subm.number_of_edges() > 0 and nx.is_directed_acyclic_graph(subm)
                highlight = _heaviest_directed_path_as_edge_list(subm)
                if highlight:
                    path_mode = (
                        "DAG 上权重和最大的有向路径"
                        if is_dag
                        else "权重最大的单条有向边（含环子图不枚举更长简单路径）"
                    )
            except ImportError:
                path_mode = "未安装 networkx，无法解析最重路径"

            sol_sizes = [len(s) for s in named_per_sol]
            n_merge = len(S_merged)
            merged_png = (
                viz_dir
                / f"merged_top{len(sols):02d}_mergeV{n_merge}_solS{'-'.join(map(str, sol_sizes))}.png"
            )
            try:
                save_merged_topk_induced_subgraph_figure(
                    global_edges,
                    adj_global,
                    S_merged,
                    merged_selected_edges,
                    highlight,
                    merged_png,
                    num_solutions=len(sols),
                    per_solution_sizes=sol_sizes,
                    connector_added=len(added_connectors),
                    merged_connected_ok=merged_ok,
                    path_mode=path_mode,
                )
                print(
                    f"=== Top-{len(sols)} 合并：各解 |S|={sol_sizes}，合并后顶点数 mergeV={n_merge}，"
                    f"序列图补点 {len(added_connectors)}；"
                    f"序列骨干子图已保存: {merged_png.resolve()} ==="
                )
                if added_connectors:
                    print(f"  为连通而新加入的顶点（示例至多 20 个）: {added_connectors[:20]}")
            except RuntimeError as err:
                print(f"（合并图跳过绘图）{err}")
        elif viz_dir is not None and not sols:
            print("=== 无 MILP 解：跳过 PNG 保存 ===")


if __name__ == "__main__":
    main()


