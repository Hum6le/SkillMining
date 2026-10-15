import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from eval_tod.alfworld_runtime import load_fewshots, parse_command, parse_reply, run_episode
from scripts import run_alfworld_smoke as runner


@pytest.mark.parametrize("command,action,slots", [
    ("go to cabinet 1", "go to", {"destination": "cabinet 1"}),
    ("take apple 2 from dining table 1", "take", {"object": "apple 2", "source": "dining table 1"}),
    ("put apple 2 on table 1", "put", {"object": "apple 2", "destination": "table 1"}),
    ("heat apple 2 with microwave 1", "heat", {"object": "apple 2", "appliance": "microwave 1"}),
    ("look", "look", {}), ("inventory", "inventory", {}),
])
def test_action_slot_parser(command, action, slots):
    parsed = parse_command(command)
    assert parsed["action"] == action and parsed["slots"] == slots
    assert parse_command(parsed["command"]) == parsed


def test_reply_executes_only_first_action_and_preserves_thought():
    parsed = parse_reply('> think: Find apple.\n> go to fridge 1\nYou found the apple.\n> take apple 1 from fridge 1')
    assert parsed["command"] == "go to fridge 1"
    assert "Find apple" in parsed["thought"]
    assert parse_reply("> think: Look for the apple.")["kind"] == "thought"
    for bad in ("", "I already solved the task", "rm -rf anything"):
        with pytest.raises(ValueError):
            parse_reply(bad)


class FakeEnv:
    def __init__(self, win_after=2):
        self.commands = []
        self.win_after = win_after
        self.closed = False

    def reset(self):
        return "You see cabinet 1. Your task is to: put apple on table."

    def step(self, command):
        self.commands.append(command)
        won = len(self.commands) >= self.win_after
        return "Task is complete." if won else "Nothing happens.", won, won

    def close(self):
        self.closed = True


def test_episode_keeps_repeats_counts_real_actions_and_has_no_oracle():
    env, calls, events = FakeEnv(), [], []
    replies = iter(["> think: Search.", "invalid reply", "> go to cabinet 1", "> go to cabinet 1"])
    def chat(messages):
        calls.append(copy.deepcopy(messages))
        return next(replies)
    result = run_episode(env, chat, fewshots="EXAMPLES", max_steps=3, max_calls=5, on_event=events.append)
    assert result["success"] and result["environment_steps"] == 2 and result["actor_calls"] == 4
    assert env.commands == ["go to cabinet 1"] * 2
    assert "Nothing happens." in json.dumps(calls[-1])
    assert [e["event"] for e in events] == ["reset", "thought", "format_error", "step", "step"]


def test_budgets_stop_thought_loops_and_environment_steps():
    env = FakeEnv(win_after=999)
    result = run_episode(env, lambda _: "> think: Planning.", fewshots="", max_calls=2)
    assert result["status"] == "call_limit" and result["environment_steps"] == 0
    result = run_episode(env, lambda _: "> look", fewshots="", max_steps=2)
    assert result["status"] == "step_limit" and result["environment_steps"] == 2


def make_game(root, split="train"):
    game = root / "json_2.1.1" / split / "pick_and_place_simple-Apple-None-Table-1" / "trial" / "game.tw-pddl"
    game.parent.mkdir(parents=True)
    game.write_text('{"solvable": true}')
    (game.parent / "traj_data.json").write_text('{"task_type":"pick_and_place_simple"}')
    logic = root / "logic"
    logic.mkdir(exist_ok=True)
    for name in ("alfred.pddl", "alfred.twl2"):
        (logic / name).write_text("logic")
    return game


def test_selection_does_not_use_expel_unseen_list(tmp_path):
    game = make_game(tmp_path)
    assert runner.select_game(tmp_path, "train", "pick", 0) == game
    with pytest.raises(ValueError, match="requested split"):
        runner.select_game(tmp_path, "valid_unseen", "pick", 0, game)
    with pytest.raises(ValueError, match="Found 0"):
        runner.select_game(tmp_path, "train", "clean", 0)


def test_prompts_load_without_expel_or_langchain_imports():
    keys, shots = load_fewshots(runner.ROOT / "ExpeL/prompts/alfworld.py")
    assert keys == ["react_put_0", "react_put_1"] and "> take" in shots


def test_real_cli_bridge_uses_llm_new_usage_and_writes_artifacts(tmp_path, monkeypatch):
    data = tmp_path / "data"
    make_game(data)
    output = tmp_path / "run"
    env = FakeEnv(win_after=1)
    calls = []
    def chat(messages, **kwargs):
        calls.append((copy.deepcopy(messages), kwargs))
        return "> look"
    module = SimpleNamespace(chat=chat, reset_usage_summary=lambda: None,
                             get_llm_config=lambda: {"model": "mock", "provider": "workflow", "api_key": "SECRET"},
                             get_usage_summary=lambda: {"total": {"calls": len(calls)}})
    monkeypatch.setitem(sys.modules, "llm_new", module)
    monkeypatch.setattr(runner, "TextWorldEpisode", lambda *a: env)
    monkeypatch.setattr(sys, "argv", ["run_alfworld_smoke.py", "--data-root", str(data),
                                      "--workflow-id", "test-id", "--output-dir", str(output)])
    assert runner.main() == 0 and env.closed
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert result["success"] and result["llm_usage"]["total"]["calls"] == 1
    assert calls[0][1]["call_tag"] == "alfworld_react"
    assert calls[0][1]["temperature"] == 0
    assert manifest["llm_module"] == "llm_new" and manifest["provider"] == "workflow"
    assert "SECRET" not in (output / "manifest.json").read_text(encoding="utf-8")
    assert not manifest["expert_plan_visible"] and not manifest["admissible_commands_visible"]
    assert (output / "trajectory.jsonl").is_file()


def test_cli_persists_call_failure_and_closes_environment(tmp_path, monkeypatch):
    make_game(tmp_path / "data")
    output = tmp_path / "run"
    env = FakeEnv()
    def fail(*a, **kw):
        raise RuntimeError("Workflow unavailable")
    module = SimpleNamespace(chat=fail, reset_usage_summary=lambda: None,
                             get_llm_config=lambda: {"model": "mock"}, get_usage_summary=lambda: {})
    monkeypatch.setitem(sys.modules, "llm_new", module)
    monkeypatch.setattr(runner, "TextWorldEpisode", lambda *a: env)
    monkeypatch.setattr(sys, "argv", ["run_alfworld_smoke.py", "--data-root", str(tmp_path / "data"),
                                      "--output-dir", str(output), "--llm-attempts", "1"])
    assert runner.main() == 2 and env.closed
    assert json.loads((output / "result.json").read_text(encoding="utf-8"))["error"] == "Workflow unavailable"


def test_dry_run_does_not_initialize_environment_or_llm(tmp_path, monkeypatch):
    make_game(tmp_path)
    def fail(*a, **kw):
        raise AssertionError("Unexpected environment access")
    monkeypatch.setattr(runner, "TextWorldEpisode", fail)
    monkeypatch.setattr(runner.importlib, "import_module", fail)
    monkeypatch.setattr(sys, "argv", ["run_alfworld_smoke.py", "--data-root", str(tmp_path), "--dry-run"])
    assert runner.main() == 0
