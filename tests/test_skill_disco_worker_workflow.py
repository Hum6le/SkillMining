"""The Skill-DisCo worker keeps its workflow assignment across phases."""

import json
import sys

from scripts import run_skill_disco_abcd_full


def test_generation_and_evaluation_inherit_worker_workflow_id(tmp_path, monkeypatch):
    train = tmp_path / "train.json"
    test = tmp_path / "test.json"
    train.write_text("[]", encoding="utf-8")
    test.write_text("[]", encoding="utf-8")
    output = tmp_path / "output"
    calls = []

    def fake_run(command, *, cwd, env, check):
        calls.append((command, env["SKILLMINING_WORKFLOW_ID"]))
        assert check
        assert cwd == run_skill_disco_abcd_full.ROOT
        if "run_skill_disco_abcd.py" in command[1]:
            output.mkdir(exist_ok=True)
            (output / "generation_artifact.json").write_text(
                json.dumps({"contracts": [], "verified_contracts": []}), encoding="utf-8"
            )
            (output / "llm_usage_generation.json").write_text("{}", encoding="utf-8")
        else:
            evaluation = output / "evaluation"
            evaluation.mkdir()
            (evaluation / "result.json").write_text(
                json.dumps({"summary": {}}), encoding="utf-8"
            )
            (evaluation / "llm_usage.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("SKILLMINING_WORKFLOW_ID", "worker_2")
    monkeypatch.setattr(run_skill_disco_abcd_full.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "run_skill_disco_abcd_full.py", "--subflow", "example",
        "--train-file", str(train), "--test-file", str(test),
        "--output-dir", str(output),
    ])
    run_skill_disco_abcd_full.main()
    assert len(calls) == 2
    assert [workflow_id for _, workflow_id in calls] == ["worker_2", "worker_2"]
