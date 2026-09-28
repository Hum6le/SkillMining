"""Callable verified Skill-DisCo library for interactive benchmark adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .synthesis import validate_skill_source


_SAFE_BUILTINS = {
    "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate,
    "float": float, "int": int, "isinstance": isinstance, "len": len,
    "list": list, "max": max, "min": min, "next": next, "range": range,
    "str": str, "tuple": tuple, "zip": zip, "ValueError": ValueError,
}


class CompiledSkillLibrary:
    """Expose only skills that passed Stage-5 verification."""

    def __init__(self, artifact: dict[str, Any]):
        self._skills = {}
        for item in artifact.get("compiled_skills", []):
            if item.get("status") != "verified" or not item.get("implementation"):
                continue
            contract = item["contract"]
            name = str(contract["skill_name"])
            validate_skill_source(item["implementation"], name)
            if name in self._skills:
                raise ValueError(f"duplicate verified skill name: {name}")
            self._skills[name] = item

    @classmethod
    def load(cls, path: str | Path) -> "CompiledSkillLibrary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def tool_specs(self) -> list[dict[str, Any]]:
        """Return signatures and guidance for the caller's skill selection prompt."""
        return [{
            "name": name,
            "description": item["contract"].get("description", ""),
            "docstring": item["contract"].get("docstring", ""),
            "parameters": item["contract"].get("parameters", []),
            "preconditions": item["contract"].get("preconditions", []),
            "postconditions": item["contract"].get("postconditions", []),
        } for name, item in sorted(self._skills.items())]

    def invoke(self, name: str, arguments: dict[str, Any], env: Any) -> dict[str, Any]:
        """Call a verified skill against the supplied live env.step adapter.

        Interactive benchmark adapters should impose their own execution time
        and action budgets around this call, as they do for primitive actions.
        """
        if name not in self._skills:
            raise KeyError(f"unknown or unverified skill: {name}")
        item = self._skills[name]
        contract = item["contract"]
        expected = {parameter["name"] for parameter in contract.get("parameters", [])}
        if not set(arguments) <= expected:
            raise ValueError(f"unknown skill argument(s): {sorted(set(arguments) - expected)}")
        for parameter in contract.get("parameters", []):
            if parameter.get("required", True) and parameter["name"] not in arguments:
                raise ValueError(f"missing required skill argument: {parameter['name']}")
        scope = {"__builtins__": _SAFE_BUILTINS, "env": env}
        exec(compile(item["implementation"], f"<skill:{name}>", "exec"), scope)
        result = scope[name](**arguments)
        required_keys = {"success", "observation", "available_actions", "process_trace"}
        if not isinstance(result, dict) or not required_keys <= result.keys():
            raise ValueError(f"skill {name} returned an invalid execution result")
        return result
