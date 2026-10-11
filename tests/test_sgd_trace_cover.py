import argparse
import copy

import pytest

from eval_tod.sgd_standard_skill_agent import SGDStandardSkillAgent
from scripts.run_sgd_domain_eval import add_ours_arguments, validate_ours_arguments
from tests.test_sgd_skill_agent import dialogue, ONTOLOGY


def test_trace_cover_dispatch_changes_only_graph_miner(monkeypatch, tmp_path):
    from skill_mining import backbone_workflow_mining as mining
    calls = []
    original_trace = mining.mine_backbone_workflow_trace_cover
    original_discriminative = mining.mine_backbone_workflow_discriminative

    def capture(name, original):
        def run(family, projected, **options):
            calls.append((name, copy.deepcopy(projected), options))
            return original(family, projected, **options)
        return run

    monkeypatch.setattr(mining, "mine_backbone_workflow_trace_cover", capture("trace_cover", original_trace))
    monkeypatch.setattr(mining, "mine_backbone_workflow_discriminative", capture("discriminative", original_discriminative))
    monkeypatch.setattr("llm.chat", lambda *_args, **_kwargs: '{"updates":[]}')
    agents = []
    for method in ("discriminative", "trace_cover"):
        agent = SGDStandardSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY,
            predictor=None, mining_method=method, trace_coverage_target=.85,
            corpus_fitness_target=.94, routing_complexity_weight=1.5)
        agent.mine([dialogue()], tmp_path / method)
        agents.append(agent)
    assert calls[0][0] == "discriminative" and calls[1][0] == "trace_cover"
    assert calls[0][1] == calls[1][1]
    assert calls[1][2] == {"min_branch_support": 2, "trace_coverage_target": .85,
                            "corpus_fitness_target": .94, "routing_complexity_weight": 1.5}
    assert agents[0].graph["nodes"] == agents[1].graph["nodes"]
    assert agents[0].resources["slot_policies"] == agents[1].resources["slot_policies"]
    assert "trace_cover" in agents[1].graph
    # Compiler, runtime/grounding and refine remain the same methods.
    assert agents[0].predict.__func__ is agents[1].predict.__func__
    assert agents[0].refine.__func__ is agents[1].refine.__func__
    loaded = SGDStandardSkillAgent(family="Restaurants", model="mock", ontology=ONTOLOGY, predictor=None)
    loaded.load(tmp_path / "trace_cover")
    assert loaded.graph == agents[1].graph
    assert loaded.resource_hash() == agents[1].resource_hash()


def test_cli_defaults_and_full_forwarding(monkeypatch, tmp_path):
    from scripts import run_full_sgd_experiments as full
    parser = argparse.ArgumentParser()
    add_ours_arguments(parser)
    defaults = parser.parse_args([])
    validate_ours_arguments(parser, defaults)
    assert defaults.ours_mining_method == "trace_cover"
    monkeypatch.setattr("sys.argv", ["runner", "--method", "ours", "--ours-trace-coverage-target", ".85",
                                    "--ours-corpus-fitness-target", ".94"])
    args = full._arguments()
    command = full._domain_command(args, {"method": "ours", "domain": "Restaurants"}, tmp_path)
    assert command[command.index("--ours-mining-method") + 1] == "trace_cover"
    assert command[command.index("--ours-trace-coverage-target") + 1] == "0.85"
    assert command[command.index("--ours-corpus-fitness-target") + 1] == "0.94"


@pytest.mark.parametrize("option,value", [
    ("--ours-trace-coverage-target", "0"), ("--ours-corpus-fitness-target", "1.1"),
    ("--ours-routing-complexity-weight", "nan"), ("--ours-routing-complexity-weight", "-1"),
])
def test_invalid_trace_cover_parameters(option, value):
    parser = argparse.ArgumentParser()
    add_ours_arguments(parser)
    args = parser.parse_args([option, value])
    with pytest.raises(SystemExit):
        validate_ours_arguments(parser, args)
