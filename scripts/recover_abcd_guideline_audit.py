"""Rescan the four Ours resources and reuse matching extraction responses.

Run from the project root. Existing results are preserved in a separate directory.
The company workflow dispatcher is selected before child processes import llm.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commands(previous, output, workers=1, qps=1.8):
    previous, output = Path(previous), Path(output)
    if previous.resolve() == output.resolve():
        raise ValueError("Recovery must use a new output directory")
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    inventory = read(previous / "sources/inventory.json")
    config = read(previous / "rules/extraction_config.json")
    reference = previous / "reference/reference.json"
    flows = read(reference)["subflows"]
    if inventory["snapshot"] != "final":
        raise ValueError("This recovery helper only handles final artifacts")
    scan = [sys.executable, "-u", str(ROOT / "scripts/extract_skill_knowledge.py"), "scan"]
    for method, directory in inventory["roots"].items():
        scan += ["--run", method + "=" + directory]
    for method, flow in inventory.get("family_overrides", {}).items():
        scan += ["--family-override", method + "=" + flow]
    scan += ["--subflows", ",".join(flows), "--ours-resources", "core", "--chunk-chars", str(inventory["chunk_chars"]),
             "--overlap", str(inventory["overlap"]), "--output-dir", str(output / "sources")]
    common = ["--provider", config["provider"], "--model", config["model"], "--workers", str(workers), "--qps", str(qps)]
    if config.get("workflow_id"):
        common += ["--workflow-id", config["workflow_id"]]
    extract = [sys.executable, "-u", str(ROOT / "scripts/extract_skill_knowledge.py"), "extract",
               "--source-dir", str(output / "sources"), "--output-dir", str(output / "rules"),
               "--reuse-responses-from", str(previous / "rules"), "--reuse-matching-chunks", *common]
    evaluate = [sys.executable, "-u", str(ROOT / "scripts/benchmark_abcd_skill_guidelines.py"), "evaluate",
                "--reference", str(reference), "--source-dir", str(output / "sources"),
                "--rules-dir", str(output / "rules"), "--output-dir", str(output / "coverage"), *common]
    return [scan, extract, evaluate], config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--qps", type=float, default=1.8)
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands; no scanning or API calls")
    args = parser.parse_args()
    if args.workers < 1 or args.qps <= 0:
        parser.error("Positive workers/QPS required")
    planned, config = commands(args.previous_dir, args.output_dir, args.workers, args.qps)
    if args.dry_run:
        print(json.dumps({"commands": planned, "workflow_backend": config["provider"] == "existing"}, indent=2))
        return
    env = dict(os.environ)
    if config["provider"] == "existing":
        env["SKILLMINING_LLM_BACKEND"] = "workflow"
        subprocess.run([sys.executable, "-c", "import llm; c=llm.get_llm_config(); "
                        "assert c.get('provider') == 'workflow' and c.get('workflow_id'), 'Workflow config missing'; "
                        "print('Workflow backend OK:', llm.chat.__module__)"], cwd=ROOT, env=env, check=True)
    # A successful scan is frozen once. Extraction itself verifies its resume contract.
    if not (args.output_dir / "sources/inventory.json").exists():
        subprocess.run(planned[0], cwd=ROOT, env=env, check=True)
    for command in planned[1:]:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
