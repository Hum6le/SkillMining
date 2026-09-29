"""Regression for independently verified clusters sharing an LLM skill name."""

import json
import sys

from scripts import run_skill_disco_abcd_full
from skill_disco.callable_runtime import CompiledSkillLibrary
from skill_disco.name_resolution import make_verified_names_unique
from skill_disco.skill_specification import SkillContract


SOURCE = (
    "def retrieve_account():\n"
    "    observation, available_actions = env.step('pull-up-account', [])\n"
    "    return {'success': True, 'observation': observation, "
    "'available_actions': available_actions, "
    "'process_trace': [('pull-up-account', observation)]}"
)


def _contract(cluster_id):
    return SkillContract(
        cluster_id=cluster_id, skill_name="retrieve_account", description="Retrieve account",
        docstring="Retrieve account", parameters=[], return_type="dict", preconditions=[],
        postconditions=[], side_effects=[], canonical_action_sequence=["pull-up-account()"],
        abstraction_level="composite", estimated_actions_saved=1, confidence_score=0.8,
        supporting_conversations=[cluster_id], source_operation_ids=[cluster_id],
    ).to_dict()


def test_duplicate_verified_names_are_renamed_without_dropping_skills(tmp_path):
    contracts = [_contract("cluster_a"), _contract("cluster_b")]
    artifact = {
        "compiled_skills": [
            {"status": "verified", "contract": contract.copy(), "implementation": SOURCE,
             "attempts": [{"implementation": SOURCE}]}
            for contract in contracts
        ],
        "verified_contracts": [contract.copy() for contract in contracts],
        "skill_library": "old",
    }
    assert make_verified_names_unique(artifact)
    names = [item["skill_name"] for item in artifact["verified_contracts"]]
    assert names[0] == "retrieve_account"
    assert names[1].startswith("retrieve_account_c")
    assert len(set(names)) == 2
    assert all(f"### Skill: {name}" in artifact["skill_library"] for name in names)
    assert not make_verified_names_unique(artifact)

    path = tmp_path / "generation_artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    library = CompiledSkillLibrary.load(path)
    assert {item["name"] for item in library.tool_specs()} == set(names)

    class Environment:
        def step(self, action, slots):
            assert action == "pull-up-account" and slots == []
            return "Found", [action]

    for name in names:
        assert library.invoke(name, {}, Environment())["success"]


def test_old_artifact_with_duplicates_loads_without_regeneration():
    contracts = [_contract("cluster_a"), _contract("cluster_b")]
    old_artifact = {
        "compiled_skills": [
            {"status": "verified", "contract": contract.copy(), "implementation": SOURCE,
             "attempts": []}
            for contract in contracts
        ],
        "verified_contracts": [contract.copy() for contract in contracts],
        "skill_library": "old",
    }
    library = CompiledSkillLibrary(old_artifact)
    assert len(library.tool_specs()) == 2


def test_resume_reuses_and_repairs_generation_before_evaluation(tmp_path, monkeypatch):
    output = tmp_path / "skill_disco" / "example"
    output.mkdir(parents=True)
    contracts = [_contract("cluster_a"), _contract("cluster_b")]
    artifact = {
        "method": "skill-disco-abcd-five-stage-replay",
        "compiled_skills": [
            {"status": "verified", "contract": contract.copy(),
             "implementation": SOURCE, "attempts": []}
            for contract in contracts
        ],
        "verified_contracts": [contract.copy() for contract in contracts],
        "contracts": contracts, "skill_library": "old",
    }
    (output / "generation_artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    train = tmp_path / "train.json"
    test = tmp_path / "test.json"
    train.write_text("[]", encoding="utf-8")
    test.write_text("[]", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert "eval_skill_disco_abcd.py" in command[1]
        evaluation = output / "evaluation"
        evaluation.mkdir()
        (evaluation / "result.json").write_text(json.dumps({"summary": {}}), encoding="utf-8")

    monkeypatch.setattr(run_skill_disco_abcd_full.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "run_skill_disco_abcd_full.py", "--subflow", "example", "--resume-generation",
        "--train-file", str(train), "--test-file", str(test), "--output-dir", str(output),
    ])
    run_skill_disco_abcd_full.main()
    repaired = json.loads((output / "generation_artifact.json").read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert len({row["skill_name"] for row in repaired["verified_contracts"]}) == 2
    assert (output / "summary.json").is_file()
