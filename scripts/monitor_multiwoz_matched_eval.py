#!/usr/bin/env python3
"""Print corrected MultiWOZ preparation/evaluation progress without API calls."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def snapshot(root):
    status = read(root / "supervisor_status.json")
    cards = [read(p) for p in (root / "card_completion").glob("batch_[0-9][0-9][0-9][0-9].json")]
    return {"time": datetime.now().isoformat(), "output_root": str(root),
        "supervisor": status, "progress": read(root / "progress.json"),
        "completed_cards": sum(len(x.get("cards", [])) for x in cards),
        "shared_observations": len(list((root / "observations").glob("case_*.json"))),
        "completed_turns": {method: len(list((root / method / "cases").glob("case_*.json")))
                            for method in ("awm", "trace2skill", "ours")},
        "results_ready": (root / "RESULTS_READY.md").exists()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be positive")
    default = Path(__file__).resolve().parent.parent / "outputs/latest_multiwoz_matched_run.json"
    root = args.root or Path(read(default)["output_root"])
    while True:
        result = snapshot(root)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.watch or result["supervisor"].get("status") in {"completed", "failed"}:
            break
        time.sleep(args.interval)
