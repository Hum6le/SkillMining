"""
基于operator序列构建全局有向图

功能：
1. 读取operator_results.json中的所有ordered_operations
2. 构建有向图，节点是操作/行为，边表示顺序关系
3. 统计边的权重（出现次数）
4. 保存图结构和统计信息
"""

import json
import networkx as nx
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

import sys as _sys
_skill_dir = Path(__file__).resolve().parent
if str(_skill_dir) not in _sys.path:
    _sys.path.insert(0, str(_skill_dir))
from session2hg_v2 import collapse_consecutive_nodes, node_name as format_operator_node

BASE_DIR = _skill_dir
DATA_DIR = BASE_DIR / "data"
OPERATOR_RESULTS_JSON = DATA_DIR / "operator_results.json"
GRAPH_OUTPUT_JSON = DATA_DIR / "operator_graph.json"
GRAPH_OUTPUT_GRAPHML = DATA_DIR / "operator_graph.graphml"
GRAPH_STATS_JSON = DATA_DIR / "operator_graph_stats.json"


def load_operator_results() -> List[Dict]:
    """加载operator结果"""
    print(f"正在读取 {OPERATOR_RESULTS_JSON.name}...")
    with open(OPERATOR_RESULTS_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"  - 读取成功，共 {len(results)} 条对话记录")
    return results


def resolve_reciprocal_edge_counts(
    edge_counts: Dict[Tuple[str, str], int],
) -> Tuple[Dict[Tuple[str, str], int], Dict[str, Any]]:
    """For each reciprocal pair u↔v, keep only the higher-weight directed edge.

    On equal weight, keep u→v when ``u < v`` lexicographically (deterministic tie-break).
    """
    resolved = dict(edge_counts)
    seen: set = set()
    removed: List[Dict[str, Any]] = []

    for (u, v) in list(edge_counts.keys()):
        pair = (min(u, v), max(u, v))
        if pair in seen:
            continue
        if (v, u) not in edge_counts:
            continue
        seen.add(pair)

        w_uv = int(edge_counts[(u, v)])
        w_vu = int(edge_counts[(v, u)])
        if w_uv > w_vu:
            del resolved[(v, u)]
            removed.append({
                "kept": [u, v], "dropped": [v, u],
                "kept_weight": w_uv, "dropped_weight": w_vu,
            })
        elif w_vu > w_uv:
            del resolved[(u, v)]
            removed.append({
                "kept": [v, u], "dropped": [u, v],
                "kept_weight": w_vu, "dropped_weight": w_uv,
            })
        elif u < v:
            del resolved[(v, u)]
            removed.append({
                "kept": [u, v], "dropped": [v, u],
                "kept_weight": w_uv, "dropped_weight": w_vu, "tie_break": True,
            })
        else:
            del resolved[(u, v)]
            removed.append({
                "kept": [v, u], "dropped": [u, v],
                "kept_weight": w_vu, "dropped_weight": w_uv, "tie_break": True,
            })

    stats: Dict[str, Any] = {
        "enabled": bool(removed),
        "pairs_resolved": len(removed),
        "edges_removed_count": len(removed),
        "removed_edges": removed,
    }
    return resolved, stats


def resolve_reciprocal_edges(G: nx.DiGraph) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """Return a copy of *G* with weaker edges removed from each reciprocal pair."""
    edge_counts = {
        (u, v): int(data.get("weight", data.get("count", 1)))
        for u, v, data in G.edges(data=True)
    }
    resolved_counts, stats = resolve_reciprocal_edge_counts(edge_counts)

    H = nx.DiGraph()
    for node, data in G.nodes(data=True):
        H.add_node(node, **dict(data))
    for (u, v), w in resolved_counts.items():
        H.add_edge(u, v, weight=w, count=w)

    stats["nodes"] = H.number_of_nodes()
    stats["edges_after"] = H.number_of_edges()
    stats["edges_before"] = G.number_of_edges()
    return H, stats


def build_graph(results: List[Dict]) -> nx.DiGraph:
    """
    构建全局有向图
    
    节点：操作/行为名称（格式：角色:操作名，如"客服:情绪安抚"）
    边：表示操作之间的顺序关系（如果A在B之前，则A->B）
    边权重：该顺序关系出现的次数
    """
    print("\n开始构建有向图...")
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 统计边的出现次数
    edge_counts = defaultdict(int)
    
    # 统计节点信息
    node_info = defaultdict(lambda: {
        "role": set(),  # 该操作可能出现的角色
        "count": 0,     # 该操作出现的总次数
        "as_user": 0,   # 作为用户行为出现的次数
        "as_service": 0 # 作为客服操作出现的次数
    })
    
    # 遍历所有对话的ordered_operations
    valid_sequences = 0
    for result in tqdm(results, desc="处理对话序列"):
        ordered_ops = result.get("ordered_operations", [])
        if not ordered_ops:
            continue
        
        valid_sequences += 1
        
        # 构建节点名称（角色:操作名）
        nodes = []
        for role, operation in ordered_ops:
            if not operation or not role:
                continue

            operation = operation.strip()
            if not operation:
                continue

            op_node = format_operator_node(str(role), operation)
            if not op_node:
                continue
            nodes.append(op_node)

            node_info[operation]["role"].add(role)
            node_info[operation]["count"] += 1
            if role == "用户":
                node_info[operation]["as_user"] += 1
            elif role == "客服":
                node_info[operation]["as_service"] += 1

        transition_nodes, _ = collapse_consecutive_nodes(nodes)

        for node in set(nodes):
            G.add_node(node)

        for i in range(len(transition_nodes) - 1):
            source = transition_nodes[i]
            target = transition_nodes[i + 1]
            edge_counts[(source, target)] += 1
    
    print(f"  - 有效序列数: {valid_sequences}")
    print(f"  - 原始转移边数: {len(edge_counts)}")

    edge_counts, recip_stats = resolve_reciprocal_edge_counts(edge_counts)
    if recip_stats["pairs_resolved"]:
        print(
            f"  - 双向边消解: {recip_stats['pairs_resolved']} 对 reciprocal，"
            f"删除 {recip_stats['edges_removed_count']} 条较弱边"
        )

    # 添加边权重
    for (source, target), count in edge_counts.items():
        G.add_edge(source, target, weight=count, count=count)

    print(f"  - 节点数: {G.number_of_nodes()}")
    print(f"  - 边数: {G.number_of_edges()}")
    
    # 添加节点属性
    for node in G.nodes():
        # 从节点名称中提取角色和操作名
        if ":" in node:
            role, operation = node.split(":", 1)
            G.nodes[node]["role"] = role
            G.nodes[node]["operation"] = operation
            # 添加操作级别的统计信息
            if operation in node_info:
                info = node_info[operation]
                G.nodes[node]["total_count"] = info["count"]
                G.nodes[node]["as_user_count"] = info["as_user"]
                G.nodes[node]["as_service_count"] = info["as_service"]

    G.graph["reciprocal_resolve"] = recip_stats
    return G


def calculate_graph_stats(G: nx.DiGraph) -> Dict:
    """计算图的统计信息"""
    print("\n计算图统计信息...")
    
    stats = {
        "nodes": {
            "total": G.number_of_nodes(),
            "by_role": defaultdict(int)
        },
        "edges": {
            "total": G.number_of_edges(),
            "total_weight": sum(data.get("weight", 1) for _, _, data in G.edges(data=True))
        },
        "operations": {
            "unique_operations": set(),
            "by_role": defaultdict(int)
        },
        "topology": {}
    }
    
    # 统计节点按角色分布
    for node, data in G.nodes(data=True):
        role = data.get("role", "未知")
        stats["nodes"]["by_role"][role] += 1
        
        operation = data.get("operation", "")
        if operation:
            stats["operations"]["unique_operations"].add(operation)
            stats["operations"]["by_role"][role] += 1
    
    stats["operations"]["unique_operations"] = len(stats["operations"]["unique_operations"])
    stats["operations"]["by_role"] = dict(stats["operations"]["by_role"])
    stats["nodes"]["by_role"] = dict(stats["nodes"]["by_role"])
    
    # 拓扑统计
    if G.number_of_nodes() > 0:
        # 入度为0的节点（起始节点）
        sources = [n for n in G.nodes() if G.in_degree(n) == 0]
        # 出度为0的节点（终止节点）
        sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
        
        stats["topology"] = {
            "sources": len(sources),
            "sinks": len(sinks),
            "is_weakly_connected": nx.is_weakly_connected(G),
            "is_strongly_connected": nx.is_strongly_connected(G) if G.number_of_nodes() > 0 else False,
            "num_weakly_connected_components": nx.number_weakly_connected_components(G),
            "num_strongly_connected_components": nx.number_strongly_connected_components(G)
        }
        
        # 计算度分布
        in_degrees = [G.in_degree(n) for n in G.nodes()]
        out_degrees = [G.out_degree(n) for n in G.nodes()]
        
        stats["topology"]["degree"] = {
            "avg_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0
        }
    
    return stats


def get_top_edges(G: nx.DiGraph, top_k: int = 20) -> List[Dict]:
    """获取权重最高的边"""
    edges_with_weights = [
        {
            "source": source,
            "target": target,
            "weight": data.get("weight", 1),
            "count": data.get("count", 1)
        }
        for source, target, data in G.edges(data=True)
    ]
    edges_with_weights.sort(key=lambda x: x["weight"], reverse=True)
    return edges_with_weights[:top_k]


def get_top_nodes(G: nx.DiGraph, top_k: int = 20) -> List[Dict]:
    """获取出现次数最多的节点"""
    nodes_with_counts = [
        {
            "node": node,
            "role": data.get("role", "未知"),
            "operation": data.get("operation", ""),
            "total_count": data.get("total_count", 0),
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node)
        }
        for node, data in G.nodes(data=True)
    ]
    nodes_with_counts.sort(key=lambda x: x["total_count"], reverse=True)
    return nodes_with_counts[:top_k]


def save_graph(G: nx.DiGraph, stats: Dict):
    """保存图和相关统计信息"""
    print("\n保存图结构...")
    
    # 保存为GraphML格式（可以用Gephi等工具可视化）
    try:
        nx.write_graphml(G, GRAPH_OUTPUT_GRAPHML)
        print(f"  - GraphML格式已保存到: {GRAPH_OUTPUT_GRAPHML.name}")
    except Exception as e:
        print(f"  - 保存GraphML失败: {e}")
    
    # 保存为JSON格式（节点和边列表）
    graph_data = {
        "nodes": [
            {
                "id": node,
                "role": data.get("role", ""),
                "operation": data.get("operation", ""),
                "total_count": data.get("total_count", 0),
                "as_user_count": data.get("as_user_count", 0),
                "as_service_count": data.get("as_service_count", 0),
                "in_degree": G.in_degree(node),
                "out_degree": G.out_degree(node)
            }
            for node, data in G.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                "weight": data.get("weight", 1),
                "count": data.get("count", 1)
            }
            for source, target, data in G.edges(data=True)
        ]
    }
    
    with open(GRAPH_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print(f"  - JSON格式已保存到: {GRAPH_OUTPUT_JSON.name}")
    
    # 保存统计信息
    stats["top_edges"] = get_top_edges(G, top_k=20)
    stats["top_nodes"] = get_top_nodes(G, top_k=20)
    
    with open(GRAPH_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  - 统计信息已保存到: {GRAPH_STATS_JSON.name}")


def print_summary(stats: Dict):
    """打印图摘要信息"""
    print("\n" + "=" * 60)
    print("图结构摘要")
    print("=" * 60)
    
    print(f"\n节点统计:")
    print(f"  - 总节点数: {stats['nodes']['total']}")
    for role, count in stats['nodes']['by_role'].items():
        print(f"  - {role}: {count}")
    
    print(f"\n边统计:")
    print(f"  - 总边数: {stats['edges']['total']}")
    print(f"  - 总权重: {stats['edges']['total_weight']}")
    
    print(f"\n操作统计:")
    print(f"  - 唯一操作数: {stats['operations']['unique_operations']}")
    for role, count in stats['operations']['by_role'].items():
        print(f"  - {role}操作数: {count}")
    
    if "topology" in stats:
        topo = stats["topology"]
        print(f"\n拓扑特征:")
        print(f"  - 起始节点数（入度=0）: {topo.get('sources', 0)}")
        print(f"  - 终止节点数（出度=0）: {topo.get('sinks', 0)}")
        print(f"  - 弱连通: {topo.get('is_weakly_connected', False)}")
        print(f"  - 强连通: {topo.get('is_strongly_connected', False)}")
        print(f"  - 弱连通分量数: {topo.get('num_weakly_connected_components', 0)}")
        print(f"  - 强连通分量数: {topo.get('num_strongly_connected_components', 0)}")
        
        if "degree" in topo:
            deg = topo["degree"]
            print(f"  - 平均入度: {deg.get('avg_in_degree', 0):.2f}")
            print(f"  - 平均出度: {deg.get('avg_out_degree', 0):.2f}")
            print(f"  - 最大入度: {deg.get('max_in_degree', 0)}")
            print(f"  - 最大出度: {deg.get('max_out_degree', 0)}")
    
    print(f"\n权重最高的10条边:")
    for i, edge in enumerate(stats.get("top_edges", [])[:10], 1):
        print(f"  {i}. {edge['source']} -> {edge['target']} (权重: {edge['weight']})")
    
    print(f"\n出现次数最多的10个节点:")
    for i, node in enumerate(stats.get("top_nodes", [])[:10], 1):
        print(f"  {i}. {node['node']} (出现次数: {node['total_count']}, 入度: {node['in_degree']}, 出度: {node['out_degree']})")


def main():
    """主函数"""
    print("=" * 60)
    print("构建全局有向图")
    print("=" * 60)
    
    # 1. 加载数据
    try:
        results = load_operator_results()
    except Exception as e:
        print(f"\n错误: 加载数据失败: {e}")
        return
    
    if not results:
        print("\n错误: 没有找到数据")
        return
    
    # 2. 构建图
    try:
        G = build_graph(results)
    except Exception as e:
        print(f"\n错误: 构建图失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if G.number_of_nodes() == 0:
        print("\n警告: 构建的图为空")
        return
    
    # 3. 计算统计信息
    try:
        stats = calculate_graph_stats(G)
    except Exception as e:
        print(f"\n错误: 计算统计信息失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 保存图
    try:
        save_graph(G, stats)
    except Exception as e:
        print(f"\n错误: 保存图失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 打印摘要
    print_summary(stats)
    
    print("\n[OK] 完成！")


if __name__ == "__main__":
    main()

