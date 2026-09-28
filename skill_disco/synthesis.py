"""Stage-5 synthesis and execution-grounded verification.

The verifier is supplied by the benchmark adapter: a skill is never accepted
solely because its Python source parses or resembles a successful trace.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import re
from typing import Any, Callable

from .operation_extraction import SemanticOperation
from .skill_specification import SkillContract


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    runtime_correct: bool
    postconditions_met: bool
    actions_saved: int
    cases_passed: int
    cases_total: int
    feedback: str

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


def build_synthesis_prompt(
    contract: SkillContract,
    examples: list[SemanticOperation],
    *,
    environment_note: str,
    feedback: str = "",
) -> str:
    """Adapt the paper's Stage-5 prompt to an env.step benchmark adapter."""
    evidence = [{
        "action_sequence": item.action_sequence,
        "code_snippet": item.code_snippet,
        "preconditions": item.preconditions,
        "postconditions": item.postconditions,
    } for item in examples[:5]]
    return (
        "Synthesize one self-contained Python skill from the contract. The global "
        "`env` is supplied by the benchmark; it is never a function parameter. "
        "Use only env.step() and Python builtins. Do not import modules, use helper "
        "functions, simulate observations, or fabricate available actions. Branch "
        "only on the latest observation. Execute the canonical actions and declared "
        "side effects, append every (action, observation) pair to process_trace, "
        "and return a dict containing success, observation, available_actions, "
        "and process_trace.\n\n"
        f"Environment: {environment_note}\n\n"
        f"Contract:\n{json.dumps(contract.to_dict(), ensure_ascii=False, indent=2)}\n\n"
        f"Successful training examples:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
        + (f"Previous verification failed: {feedback}\n\n" if feedback else "")
        + "Return one JSON object only: {\"implementation\": \"def skill_name(...): ...\", "
        "\"example_usage\": \"result = skill_name(...)\"}."
    )


def parse_synthesis_response(raw: str, skill_name: str) -> tuple[str, str]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        candidate = candidate[start:end + 1] if start >= 0 and end > start else ""
    payload = json.loads(candidate)
    if not isinstance(payload, dict) or not isinstance(payload.get("implementation"), str):
        raise ValueError("Stage-5 response must contain an implementation string")
    source = payload["implementation"].strip()
    validate_skill_source(source, skill_name)
    return source, str(payload.get("example_usage", ""))


def validate_skill_source(source: str, skill_name: str) -> None:
    """Reject code with non-skill top-level effects or obvious escape paths.

    This is a format and capability check, not a security sandbox. Benchmark
    adapters must execute generated code in an isolated worker with a timeout.
    """
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("implementation must contain exactly one function")
    function = tree.body[0]
    if function.name != skill_name or function.decorator_list:
        raise ValueError("implementation function name/decorators are invalid")
    forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
                 ast.Lambda, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)
    allowed_calls = {
        "all", "any", "bool", "dict", "enumerate", "float", "int",
        "isinstance", "len", "list", "max", "min", "next", "range",
        "str", "tuple", "zip", "ValueError",
    }
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"forbidden Python construct: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise ValueError("nested helper functions are forbidden")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder access is forbidden")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder access is forbidden")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "env" and node.attr != "step":
                raise ValueError("env exposes only step()")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in allowed_calls:
                raise ValueError(f"unsupported function call: {node.func.id}")
    if not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "env"
        and node.func.attr == "step"
        for node in ast.walk(function)
    ):
        raise ValueError("implementation must call env.step()")


def synthesize_and_verify(
    contract: SkillContract,
    examples: list[SemanticOperation],
    chat_fn: Callable[..., str],
    verify_fn: Callable[[str, SkillContract], VerificationResult],
    *,
    environment_note: str,
    model: str = "deepseek-chat",
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Resynthesize with execution feedback and discard failures after R tries."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    attempts: list[dict[str, Any]] = []
    feedback = ""
    for index in range(max_attempts):
        prompt = build_synthesis_prompt(
            contract, examples, environment_note=environment_note, feedback=feedback,
        )
        try:
            raw = chat_fn(prompt, model=model, temperature=0.0)
            source, example_usage = parse_synthesis_response(raw, contract.skill_name)
            verification = verify_fn(source, contract)
            attempts.append({
                "attempt": index + 1,
                "implementation": source,
                "example_usage": example_usage,
                "verification": verification.to_dict(),
            })
            if verification.passed:
                return {"status": "verified", "contract": contract.to_dict(),
                        "implementation": source, "attempts": attempts}
            feedback = verification.feedback
        except (ValueError, SyntaxError, json.JSONDecodeError, RuntimeError) as error:
            feedback = f"{type(error).__name__}: {error}"
            attempts.append({"attempt": index + 1, "error": feedback})
    return {"status": "discarded", "contract": contract.to_dict(),
            "implementation": None, "attempts": attempts}
