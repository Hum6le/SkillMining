"""Short-lived restricted worker for one generated ABCD skill replay case."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skill_disco.synthesis import validate_skill_source


class ReplayEnvironment:
    def __init__(self, actions: list[dict], available_actions: list[str]):
        self.actions = actions
        self.position = 0
        self.process_trace: list[tuple[str, str]] = []
        self.available_actions = available_actions

    def step(self, action: str, slots: list[str] | None = None):
        if self.position >= len(self.actions):
            raise ValueError("skill executed an extra backend action")
        expected = self.actions[self.position]
        actual_slots = [str(value) for value in (slots or [])]
        if str(action) != expected["name"] or actual_slots != expected["slots"]:
            raise ValueError(
                f"action {self.position}: expected {expected['name']}({expected['slots']}), "
                f"got {action}({actual_slots})"
            )
        self.position += 1
        observation = expected["observation"]
        self.process_trace.append((str(action), observation))
        return observation, self.available_actions


def main() -> int:
    payload = json.load(sys.stdin)
    case = payload["case"]
    env = ReplayEnvironment(case["actions"], case["available_actions"])
    try:
        source = payload["source"]
        validate_skill_source(source, payload["skill_name"])
        safe_builtins = {
            "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate,
            "float": float, "int": int, "isinstance": isinstance, "len": len,
            "list": list, "max": max, "min": min, "next": next, "range": range,
            "str": str, "tuple": tuple, "zip": zip, "ValueError": ValueError,
        }
        scope = {"__builtins__": safe_builtins, "env": env}
        exec(compile(source, "<generated-skill>", "exec"), scope)
        arguments = {}
        for parameter in payload["parameters"]:
            name = parameter["name"]
            if name in case["bindings"]:
                arguments[name] = case["bindings"][name]
            elif parameter.get("default") is not None:
                arguments[name] = parameter["default"]
        result = scope[payload["skill_name"]](**arguments)
        if not isinstance(result, dict):
            raise ValueError("skill did not return a dict")
        required = {"success", "observation", "available_actions", "process_trace"}
        if not required <= result.keys():
            raise ValueError(f"missing return keys: {sorted(required - result.keys())}")
        if result["success"] is not True:
            raise ValueError("skill did not report success")
        if env.position != len(env.actions):
            raise ValueError(f"skill executed {env.position}/{len(env.actions)} canonical actions")
        if result["observation"] != env.actions[-1]["observation"]:
            raise ValueError("latest observation was not returned")
        if result["available_actions"] != env.available_actions:
            raise ValueError("latest available_actions was not returned")
        reported = [tuple(item) for item in result["process_trace"]]
        if reported != env.process_trace:
            raise ValueError("process_trace does not match executed actions")
    except Exception as error:
        print(json.dumps({"passed": False, "feedback": f"{type(error).__name__}: {error}"}))
        return 1
    print(json.dumps({"passed": True, "feedback": "verified"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
