"""Assign distinct callable names to independently verified skill contracts."""

from __future__ import annotations

import ast
import hashlib
from typing import Any

from .pseudocode import render_skill_library
from .skill_specification import skill_contract_from_dict
from .synthesis import validate_skill_source


def make_verified_names_unique(artifact: dict[str, Any]) -> bool:
    """Rename collisions without dropping a verified skill or rerunning the LLM.

    Stage 4 specifies contracts independently, so two clusters can receive the
    same semantic name. The first keeps that name; subsequent ones receive a
    stable cluster-derived suffix. Only the function definition is renamed:
    Stage-5 validation already forbids calls to other generated functions.
    """
    compiled = artifact.get("compiled_skills", [])
    verified_contracts = artifact.get("verified_contracts")
    if not isinstance(compiled, list) or (
        verified_contracts is not None and not isinstance(verified_contracts, list)
    ):
        raise ValueError("invalid compiled Skill-DisCo artifact")
    used: set[str] = set()
    verified_index = 0
    changed = False
    for item in compiled:
        if item.get("status") != "verified" or not item.get("implementation"):
            continue
        contract = item["contract"]
        old_name = str(contract["skill_name"])
        source = str(item["implementation"])
        validate_skill_source(source, old_name)
        paired = None
        if verified_contracts is not None:
            if verified_index >= len(verified_contracts):
                raise ValueError("compiled skills and verified contracts are inconsistent")
            paired = verified_contracts[verified_index]
            if paired.get("cluster_id") != contract.get("cluster_id"):
                raise ValueError("compiled skills and verified contracts have different order")
        verified_index += 1
        if old_name not in used:
            used.add(old_name)
            continue
        digest = hashlib.sha256(str(contract["cluster_id"]).encode("utf-8")).hexdigest()[:8]
        stem = f"{old_name}_c{digest}"
        new_name = stem
        suffix = 2
        while new_name in used:
            new_name = f"{stem}_{suffix}"
            suffix += 1
        tree = ast.parse(source)
        tree.body[0].name = new_name
        new_source = ast.unparse(tree)
        validate_skill_source(new_source, new_name)
        item["renamed_from"] = old_name
        item["implementation"] = new_source
        contract["skill_name"] = new_name
        if paired is not None:
            paired["skill_name"] = new_name
        for attempt in item.get("attempts", []):
            if attempt.get("implementation") == source:
                attempt["implementation"] = new_source
                if isinstance(attempt.get("example_usage"), str):
                    attempt["example_usage"] = attempt["example_usage"].replace(
                        f"{old_name}(", f"{new_name}(",
                    )
        used.add(new_name)
        changed = True
    if verified_contracts is not None and verified_index != len(verified_contracts):
        raise ValueError("compiled skills and verified contracts have different lengths")
    if changed and verified_contracts is not None:
        artifact["skill_library"] = render_skill_library(
            [skill_contract_from_dict(item) for item in verified_contracts]
        )
    return changed
