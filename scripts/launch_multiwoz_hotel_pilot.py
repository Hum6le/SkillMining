#!/usr/bin/env python3
"""Launch the paired MultiWOZ hotel pilot with credentials from the local .env."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV = ROOT / "third_party/agent-workflow-memory/mind2web/.env"


def credential_env(path: Path) -> tuple[dict[str, str], str]:
    values = dotenv_values(path)
    key = values.get("SHUBIAOBIAO_API_KEY")
    base = values.get("MIND2WEB_API_BASE_URL")
    model = values.get("MIND2WEB_MODEL") or "qwen3.6-flash"
    if not key or not base:
        raise RuntimeError(f"Missing gateway key or base URL in {path}")
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = key
    env["OPENAI_BASE_URL"] = base
    thinking = (values.get("MIND2WEB_ENABLE_THINKING") or "omit").strip().lower()
    if thinking in ("true", "false"):
        env["SKILLMINING_ENABLE_THINKING"] = thinking
    elif thinking != "omit":
        raise ValueError("MIND2WEB_ENABLE_THINKING must be true, false, or omit")
    # Bound rare stalled gateway requests so one case cannot hold an entire
    # process pool indefinitely. The OpenAI SDK retries transient failures.
    env["SKILLMINING_LLM_TIMEOUT_SECONDS"] = "90"
    env["SKILLMINING_LLM_MAX_RETRIES"] = "0"
    env["SKILLMINING_RETRY_EMPTY_LLM"] = "1"
    return env, model


def background_python() -> Path:
    """Use pythonw on Windows so spawned process-pool workers own no console."""
    executable = Path(sys.executable)
    if os.name == "nt":
        pythonw = executable.with_name("pythonw.exe")
        if pythonw.is_file():
            return pythonw
    return executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fraction", type=float, default=.1)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="Run one train batch per method")
    parser.add_argument("--probe", action="store_true", help="Make one small live API request")
    args = parser.parse_args()
    env, model = credential_env(args.env_file)
    if args.probe:
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "SKILLMINING_ENABLE_THINKING"):
            if key in env:
                os.environ[key] = env[key]
        sys.path.insert(0, str(ROOT))
        from llm import chat
        response = chat("Reply with exactly OK.", model=model, temperature=0)
        print(json.dumps({"model": model, "api_success": bool(response),
                          "response_prefix": response[:40]}, ensure_ascii=False))
        if not response:
            raise SystemExit(1)
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (args.output_root or ROOT / "outputs" / f"multiwoz_hotel_{'smoke' if args.smoke else '10pct'}_{stamp}").resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is nonempty: {output_root}")
    output_root.mkdir(parents=True)
    processes = []
    for method in ("awm", "ours"):
        command = [str(background_python()), str(ROOT / "scripts/run_multiwoz_hotel_pilot.py"),
                   "--method", method, "--output-dir", str(output_root / method),
                   "--model", model, "--fraction", str(args.fraction),
                   "--workers", str(args.workers)]
        if args.smoke:
            command += ["--max-batches", "1"]
            if method == "awm":
                command += ["--awm-batch-size", "1"]
        log_path = output_root / f"{method}.log"
        stream = log_path.open("w", encoding="utf-8")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=flags)
        stream.close()
        processes.append({"method": method, "pid": process.pid, "log": str(log_path)})
    manifest = {"started_at": datetime.now().isoformat(timespec="seconds"),
                "model": model, "workers_per_method": args.workers,
                "background_executable": str(background_python()),
                "fraction": args.fraction, "smoke": args.smoke,
                "processes": processes}
    (output_root / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
