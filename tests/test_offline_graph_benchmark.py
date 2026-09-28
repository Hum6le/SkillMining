from scripts.eval_offline_graph_10flow import aggregate, evaluate_graph


def _conversation(convo_id, actions):
    return {
        "convo_id": convo_id,
        "delexed": [
            {"speaker": "action", "text": "x", "targets": ["", "take_action", action, []]}
            for action in actions
        ],
        "original": [["action", "x"] for _ in actions],
    }


def test_graph_replay_separates_full_backbone_and_retained_coverage():
    subgraph = {
        "nodes": [{"id": f"flow:{name}"} for name in ("a", "b", "c")],
        "edges": [
            {"source": "flow:a", "target": "flow:b"},
            {"source": "flow:a", "target": "flow:c"},
        ],
        "backbone": {"edges": [{"source": "flow:a", "target": "flow:b"}]},
        "local_transitions": {
            "flow:a": [
                {"source": "flow:a", "target": "flow:b", "priority": 1, "score": 2, "support": 3}
            ]
        },
        "residual_edges": [],
    }
    result = evaluate_graph(
        "flow", subgraph,
        [_conversation("1", ["a", "b"]), _conversation("2", ["a", "c"])],
    )
    assert result["full_graph_transition_recall"] == 1.0
    assert result["backbone_transition_recall"] == 0.5
    assert result["retained_transition_recall"] == 0.5
    assert result["next_action_mrr"] == 0.5
    assert result["branch_candidate_recall"] == 0.5
    assert result["complete_route_rate"] == 0.5


def test_aggregate_uses_transition_and_dialogue_denominators():
    rows = [
        {
            "test_dialogues": 1, "route_dialogues": 1, "heldout_transitions": 1,
            "branch_transitions": 1, "heldout_transition_types": 1,
            "train_graph_nodes": 2, "train_full_edges": 1, "backbone_edges": 1,
            "retained_edges": 1, "residual_edges": 0, "avg_retained_out_degree": 1.0,
            "full_graph_transition_recall": 1.0, "backbone_transition_recall": 1.0,
            "retained_transition_recall": 1.0, "next_action_top1": 1.0,
            "next_action_mrr": 1.0, "avg_candidate_size": 1.0,
            "routing_pair_complexity": 0.0,
            "branch_candidate_recall": 1.0, "branch_next_action_top1": 1.0,
            "branch_next_action_mrr": 1.0, "branch_avg_candidate_size": 1.0,
            "mean_route_coverage": 1.0, "route_coverage_at_80": 1.0,
            "complete_route_rate": 1.0, "full_graph_complete_route_rate": 1.0,
            "full_graph_unique_transition_recall": 1.0,
            "retained_unique_transition_recall": 1.0,
        },
        {
            "test_dialogues": 3, "route_dialogues": 3, "heldout_transitions": 3,
            "branch_transitions": 3, "heldout_transition_types": 3,
            "train_graph_nodes": 2, "train_full_edges": 1, "backbone_edges": 1,
            "retained_edges": 1, "residual_edges": 0, "avg_retained_out_degree": 1.0,
            "full_graph_transition_recall": 0.0, "backbone_transition_recall": 0.0,
            "retained_transition_recall": 0.0, "next_action_top1": 0.0,
            "next_action_mrr": 0.0, "avg_candidate_size": 1.0,
            "routing_pair_complexity": 0.0,
            "branch_candidate_recall": 0.0, "branch_next_action_top1": 0.0,
            "branch_next_action_mrr": 0.0, "branch_avg_candidate_size": 1.0,
            "mean_route_coverage": 0.0, "route_coverage_at_80": 0.0,
            "complete_route_rate": 0.0, "full_graph_complete_route_rate": 0.0,
            "full_graph_unique_transition_recall": 0.0,
            "retained_unique_transition_recall": 0.0,
        },
    ]
    overall = aggregate(rows)
    assert overall["retained_transition_recall"] == 0.25
    assert overall["branch_next_action_mrr"] == 0.25
    assert overall["complete_route_rate"] == 0.25
