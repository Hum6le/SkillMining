"""Shared text-only ALFWorld ReAct actor; no benchmark/LLM imports at import time."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re


PROTOCOL = "alfworld-shared-react-v1"
ACTION_PATTERNS = (
    ("go to", r"go to (.+)", ("destination",)),
    ("take", r"take (.+) from (.+)", ("object", "source")),
    ("put", r"put (.+) (?:in/on|in|on) (.+)", ("object", "destination")),
    ("clean", r"clean (.+) with (.+)", ("object", "appliance")),
    ("heat", r"heat (.+) with (.+)", ("object", "appliance")),
    ("cool", r"cool (.+) with (.+)", ("object", "appliance")),
    *[(name, name + r" (.+)", ("target",)) for name in ("open", "close", "use", "examine", "toggle")],
    ("look", r"look", ()), ("inventory", r"inventory", ()),
)


def parse_command(command):
    command = command.strip().lower()
    for action, pattern, keys in ACTION_PATTERNS:
        match = re.fullmatch(pattern, command)
        if match:
            slots = dict(zip(keys, match.groups()))
            canonical = (f"put {slots['object']} in/on {slots['destination']}"
                         if action == "put" else command)
            return {"action": action, "slots": slots, "command": canonical}
    raise ValueError("Expected one supported ALFWorld command")


def parse_reply(raw):
    """Read one next command; never execute model-authored future observations."""
    thoughts = []
    for line in raw.strip().splitlines():
        line = re.sub(r"^>\s*", "", line.strip())
        if not line:
            continue
        if re.match(r"(?i)^(think|thought):", line):
            thoughts.append(line)
            continue
        line = re.sub(r"(?i)^action:\s*", "", line)
        try:
            return {"kind": "action", "thought": "\n".join(thoughts), **parse_command(line)}
        except ValueError:
            if thoughts:  # Do not reinterpret later hallucinated observations as actions.
                break
            raise
    if thoughts:
        return {"kind": "thought", "thought": "\n".join(thoughts)}
    raise ValueError("Empty or invalid ReAct output")


def load_fewshots(path):
    """Read ExpeL's literal examples without executing its LangChain imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    examples = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "d" for target in node.targets))
    keys = ["react_put_0", "react_put_1"]
    return keys, "\n\n".join(examples[key] for key in keys)


def initial_messages(fewshots, max_steps, skill=""):
    system = (
        "You are a household robot in a text environment. Follow the examples' ReAct syntax. "
        "Respond with one next command prefixed by '> ', optionally preceded by '> think: ...'. "
        "A thought alone is allowed. Never invent observations or output a future trajectory. "
        "Commands: go to TARGET; take OBJECT from SOURCE; put OBJECT in/on DESTINATION; "
        "open TARGET; close TARGET; use TARGET; clean OBJECT with APPLIANCE; "
        "heat OBJECT with APPLIANCE; cool OBJECT with APPLIANCE; examine TARGET; "
        "look; inventory. Bind names and numeric IDs from observations. "
        f"The environment action budget is {max_steps}."
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Here are two fixed examples:\n\n" + fewshots}]
    if skill:
        messages.append({"role": "user", "content": "Reusable procedural guidance (not current observations):\n" + skill})
    return messages


def run_episode(env, chat, *, fewshots, max_steps=30, max_calls=60,
                skill="", on_event=lambda event: None):
    """env exposes only reset()->observation and step(command)->obs, won, done.

    No admissible-command lists, expert plans, goal checks, or hidden state are
    included in actor messages. The env's won flag alone determines success.
    """
    messages = initial_messages(fewshots, max_steps, skill)
    initial = env.reset()
    messages.append({"role": "user", "content": initial})
    steps, calls, success, done = 0, 0, False, False
    on_event({"event": "reset", "observation": initial})
    while steps < max_steps and calls < max_calls and not done:
        calls += 1
        raw = chat(messages)
        try:
            reply = parse_reply(raw)
        except ValueError as exc:
            on_event({"event": "format_error", "call": calls, "error": str(exc), "raw_response": raw})
            messages.extend([{"role": "assistant", "content": raw}, {"role": "user", "content":
                "Invalid format. Return one next ALFWorld command, or '> think: ...'. Do not invent observations."}])
            continue
        if reply["kind"] == "thought":
            messages.extend([{"role": "assistant", "content": reply["thought"]},
                             {"role": "user", "content": "OK. Choose the next environment action."}])
            on_event({"event": "thought", "call": calls, "thought": reply["thought"]})
            continue
        executed = (reply["thought"] + "\n" if reply["thought"] else "") + "> " + reply["command"]
        messages.append({"role": "assistant", "content": executed})
        observation, success, done = env.step(reply["command"])
        steps += 1
        on_event({"event": "step", "step": steps, "call": calls,
                  **{key: reply[key] for key in ("action", "slots", "command", "thought")},
                  "observation": observation, "success": bool(success), "done": bool(done)})
        messages.append({"role": "user", "content": observation})
        done = bool(done or success)
    return {"protocol": PROTOCOL, "success": bool(success), "environment_steps": steps,
            "actor_calls": calls, "status": "success" if success else "environment_done" if done
            else "step_limit" if steps >= max_steps else "call_limit",
            "actor_prompt_sha256": hashlib.sha256(json.dumps(
                initial_messages(fewshots, max_steps, skill), ensure_ascii=False).encode()).hexdigest()}


class TextWorldEpisode:
    """Thin direct official-env adapter, avoiding ExpeL's early-repeat cutoff."""

    def __init__(self, config, gamefile, split, seed):
        from alfworld.agents.environment import get_environment
        modes = {"train": "train", "valid_seen": "eval_in_distribution",
                 "valid_unseen": "eval_out_of_distribution"}
        self.owner = get_environment("AlfredTWEnv")(config, train_eval=modes[split])
        # The selected file must have passed the official solvability/type filter.
        if str(gamefile) not in {str(Path(p).resolve()) for p in self.owner.game_files}:
            raise ValueError("Selected game was excluded by official solvability/task filters")
        self.owner.game_files = [str(gamefile)]
        self.owner.num_games = 1
        self.env = self.owner.init_env(batch_size=1)
        self.env.seed(seed)

    def reset(self):
        observations, _ = self.env.reset()
        return str(observations[0])

    def step(self, command):
        observations, _, dones, infos = self.env.step([command])
        return str(observations[0]), bool(infos["won"][0]), bool(dones[0])

    def close(self):
        self.env.close()
