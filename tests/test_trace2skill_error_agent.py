import json
from pathlib import Path

import pytest

from scripts.trace2skill_error_agent import (
    ReadOnlyFileTools,
    _json_from_response,
    _select_case,
)


def test_agent_selects_exact_conversation_case(tmp_path: Path):
    evidence = tmp_path / "trajectory_evidence.json"
    evidence.write_text(json.dumps([
        {"conversation_id": "a", "error_type": "none"},
        {"conversation_id": "b", "error_type": "slot"},
    ]), encoding="utf-8")

    assert _select_case(evidence, "b")["error_type"] == "slot"
    with pytest.raises(KeyError):
        _select_case(evidence, "missing")


def test_agent_file_tools_are_read_only_and_root_bounded(tmp_path: Path):
    run_dir = tmp_path / "run"
    repo_dir = tmp_path / "repo"
    run_dir.mkdir()
    repo_dir.mkdir()
    (run_dir / "case.md").write_text("line one\nline two", encoding="utf-8")
    tools = ReadOnlyFileTools(run_dir, repo_dir)

    result = tools.execute({"action": "read_file", "path": "run/case.md"})
    assert "1: line one" in result["content"]
    with pytest.raises(ValueError):
        tools.execute({"action": "read_file", "path": "run/../repo/secret.md"})
    assert not (run_dir / "new.md").exists()


def test_agent_parses_json_fenced_or_bare():
    assert _json_from_response('```json\n{"action":"finish"}\n```') == {"action": "finish"}
    assert _json_from_response('Result: {"action":"read_file","path":"run/a"}') == {
        "action": "read_file", "path": "run/a",
    }
    assert _json_from_response("not json") is None
