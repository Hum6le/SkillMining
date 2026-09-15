#!/usr/bin/env python3
"""Smoke-test the workflow-aware ``llm.py`` runtime.

Run this on the server from the project root.  It deliberately imports only
``llm`` (the server's workflow-aware implementation), copies the complete
project config, and overrides only the selected workflow ID.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_PATH = PROJECT_ROOT / "llm.py"
CONFIG_PATH = PROJECT_ROOT / "config.py"


def _load_project_module(module_name: str, module_path: Path):
    """Load the repository module by absolute path, avoiding name collisions."""
    if not module_path.is_file():
        raise FileNotFoundError(f"Expected project module does not exist: {module_path}")
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Test one workflow-aware llm.py endpoint")
    parser.add_argument("--workflow-id", required=True, help="Workflow ID to test")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: WORKFLOW_OK",
        help="Small test prompt (default: a deterministic health check)",
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    workflow_id = args.workflow_id.strip()
    if not workflow_id:
        parser.error("--workflow-id must not be empty")

    config = _load_project_module("config", CONFIG_PATH)
    llm = _load_project_module("llm", LLM_PATH)

    base_config = getattr(config, "LLM_CONFIG", None)
    if not isinstance(base_config, dict):
        raise RuntimeError("config.py does not define a dictionary LLM_CONFIG")

    workflow_config = copy.deepcopy(base_config)
    workflow_config["provider"] = "workflow"
    workflow_config["workflow_id"] = workflow_id
    if args.model:
        workflow_config["model"] = args.model

    # Do not print credentials or other potentially sensitive config values.
    safe_config = {
        key: ("<set>" if key.lower() in {"api_key", "openai_api_key", "deepseek_api_key"}
              else value)
        for key, value in workflow_config.items()
        if key.lower() not in {"api_key", "openai_api_key", "deepseek_api_key"}
    }
    print("Runtime module:", Path(llm.__file__).resolve())
    print("Workflow config:", json.dumps(safe_config, ensure_ascii=False, sort_keys=True))
    print("Workflow ID:", workflow_id)
    print("Calling llm.chat() ...", flush=True)

    try:
        response = llm.chat(
            [{"role": "user", "content": args.prompt}],
            model=args.model,
            config=workflow_config,
            temperature=0.0,
            call_tag="workflow_smoke_test",
        )
    except Exception as exc:
        print(f"WORKFLOW_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    response = str(response or "")
    if not response.strip():
        print("WORKFLOW_FAILED: empty response", file=sys.stderr)
        return 1
    print("WORKFLOW_OK")
    print("Response:")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
