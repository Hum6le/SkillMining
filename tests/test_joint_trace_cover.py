import copy

import networkx as nx
import pytest

from skill_mining.backbone_workflow_mining import ROOT, mine_backbone_workflow_trace_cover
from skill_mining.joint_trace_cover import mine_backbone_workflow_joint_trace_cover


def conversation(i, actions):
    return {"convo_id": str(i), "delexed": [
        {"speaker": "action", "text": "", "targets": ["", "take_action", a, []]}
        for a in actions
    ], "original": [["action", ""] for _ in actions]}


def training():
    return [conversation(i, sequence) for i, sequence in enumerate([
        ["a", "b", "c", "d", "a", "c"],
        ["a", "b", "c", "d", "a", "c"],
        ["a", "c", "b", "d", "b", "c"],
        ["a", "c", "b", "d", "b", "c"],
        ["a", "b", "d", "c", "a", "d"],
        ["a", "b", "d", "c", "a", "d"],
    ])]


def retained(graph):
    return {(e["source"], e["target"]) for rows in graph["local_transitions"].values() for e in rows}


def tree_pairs(graph):
    return {(e["source"], e["target"]) for e in graph["backbone"]["edges"] if e["source"] != ROOT}


def test_search_is_feasible_monotone_and_changes_backbone():
    data = training()
    original = copy.deepcopy(data)
    result = mine_backbone_workflow_joint_trace_cover(
        "flow", data, corpus_fitness_target=.4, routing_complexity_weight=10,
        tree_evidence_weight=.01, max_moves=20, search_width=1000,
    )
    graph = result["subgraph"]
    meta = graph["joint_trace_cover"]
    assert data == original
    assert meta["target_feasible"]
    assert meta["accepted_moves"]
    assert meta["backbone_changed_edges"] > 0
    for move in meta["accepted_moves"]:
        assert move["objective_after"] < move["objective_before"]
        assert move["fitness"] >= .4 - 1e-12
    tree = nx.DiGraph((e["source"], e["target"]) for e in graph["backbone"]["edges"])
    assert nx.is_arborescence(tree)
    assert set(graph["backbone"]["compilation_order"]) == {n["id"] for n in graph["nodes"]}
    assert tree_pairs(graph) <= retained(graph)
    assert all(e["support"] > 0 for e in graph["backbone"]["edges"] if e["source"] == ROOT)
    assert meta["final_retained_edges"] == len(retained(graph))


def test_fixed_search_never_removes_original_backbone():
    data = training()
    before = mine_backbone_workflow_trace_cover("flow", data, corpus_fitness_target=.7)["subgraph"]
    after = mine_backbone_workflow_joint_trace_cover(
        "flow", data, corpus_fitness_target=.7, search_mode="fixed", max_moves=20,
    )["subgraph"]
    assert tree_pairs(before) == tree_pairs(after)
    assert tree_pairs(before) <= retained(after)
    assert after["joint_trace_cover"]["removed_initial_backbone_edges"] == 0


def test_zero_moves_preserves_selection_and_original_miner():
    data = training()
    before = mine_backbone_workflow_trace_cover("flow", data)["subgraph"]
    joint = mine_backbone_workflow_joint_trace_cover("flow", data, max_moves=0)["subgraph"]
    after = mine_backbone_workflow_trace_cover("flow", data)["subgraph"]
    assert before == after
    assert retained(before) == retained(joint)
    assert joint["joint_trace_cover"]["accepted_moves"] == []


def test_support_ranking_uses_training_counts_without_changing_selection():
    graph = mine_backbone_workflow_joint_trace_cover(
        "flow", training(), max_moves=0, ranking="support",
    )["subgraph"]
    for rows in graph["local_transitions"].values():
        assert [row["support"] for row in rows] == sorted([row["support"] for row in rows], reverse=True)


def test_unattainable_target_is_marked_without_claiming_feasibility():
    graph = mine_backbone_workflow_joint_trace_cover(
        "flow", training(), min_branch_support=1000, corpus_fitness_target=1,
    )["subgraph"]
    assert graph["joint_trace_cover"]["stop_reason"] == "initializer_target_unattainable"
    assert not graph["joint_trace_cover"]["target_feasible"]
    assert graph["joint_trace_cover"]["accepted_moves"] == []


def test_skill_pipeline_dispatches_joint_options_without_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("ABCD_OUTPUT_DIR", str(tmp_path / "logs"))
    from scripts.run_subflow_eval import mine_subflow_skill_backbone
    from skill_mining import skill_writer

    monkeypatch.setattr(skill_writer, "build_backbone_seed_skill", lambda *_a, **_k: "seed")
    monkeypatch.setattr(skill_writer, "induce_transition_rules", lambda *_a, **_k: {})
    monkeypatch.setattr(skill_writer, "build_skill_md_from_backbone", lambda *_a, **_k: "")
    result = mine_subflow_skill_backbone(
        "flow", training(), mining_method="joint_trace_cover",
        joint_options={"max_moves": 0, "ranking": "support", "tree_evidence_weight": .5},
        artifact_dir=tmp_path / "artifact",
    )
    meta = result["subgraph"]["joint_trace_cover"]
    assert meta["max_moves"] == 0
    assert meta["ranking"] == "support"
    assert meta["tree_evidence_weight"] == .5
    assert (tmp_path / "artifact" / "subgraph.json").is_file()


@pytest.mark.parametrize("kwargs", [
    {"tree_evidence_weight": -1}, {"max_moves": -1}, {"search_width": 0},
    {"search_mode": "wrong"}, {"ranking": "wrong"},
    {"root_policy": "wrong"},
])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        mine_backbone_workflow_joint_trace_cover("flow", training(), **kwargs)
