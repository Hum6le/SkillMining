"""Persistent case-level SGD errors; never silently drop a failed test turn."""
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import traceback


def record_error(directory, phase, exc, **identity):
    row = {"phase": phase, **identity, "exception": type(exc).__name__,
           "error": str(exc), "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
           "time": datetime.now(timezone.utc).isoformat()}
    logging.getLogger(__name__).error("SGD case failed: phase=%s identity=%s error=%s", phase, identity, exc)
    if directory is not None:
        path = Path(directory) / "case_errors.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def error_summary(directory):
    path = Path(directory) / "case_errors.jsonl"
    paths = sorted(Path(directory).rglob("case_errors.jsonl"))
    rows = [json.loads(line) for item in paths for line in item.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"total": len(rows), "by_phase": dict(Counter(row["phase"] for row in rows)),
            "path": str(path), "files": [str(item) for item in paths],
            "status": "completed_with_errors" if rows else "completed"}
