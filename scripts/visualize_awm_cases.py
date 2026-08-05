"""Create a self-contained HTML report for AWM prediction failures and traces.

The report contains:
1. up to N distinct failed action turns from the test set;
2. the exact prompt messages sent to the LLM for every selected turn;
3. prediction versus ABCD ground truth;
4. a small sample of training-time turns with input/output and batch index.

Usage:
    python scripts/visualize_awm_cases.py --run-dir outputs/awm_abcd_...
"""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _prompt_from_row(row: dict[str, Any]) -> str:
    for trace in reversed(row.get("react_trace") or []):
        if trace.get("action") == "llm_generate":
            messages = trace.get("action_input", {}).get("messages", [])
            return json.dumps(messages, ensure_ascii=False, indent=2)
    return "(prompt not recorded in this artifact)"


def _ground_truth(conv: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    idx = int(row.get("turn_index", -1))
    turns = conv.get("delexed", [])
    if idx < 0 or idx >= len(turns):
        return {"turn_type": row.get("target_type", ""), "action": None, "slots": [], "text": ""}
    turn = turns[idx]
    targets = turn.get("targets", [])
    is_action = len(targets) >= 4 and targets[1] == "take_action"
    if is_action:
        return {
            "turn_type": "action",
            "action": str(targets[2] or ""),
            "slots": [str(x) for x in (targets[3] if isinstance(targets[3], list) else [])],
            "text": str(turn.get("text", "")),
        }
    original = conv.get("original", [])
    original_text = ""
    if idx < len(original) and isinstance(original[idx], dict):
        original_text = str(original[idx].get("text", ""))
    return {
        "turn_type": "utterance",
        "action": None,
        "slots": [],
        "text": original_text or str(turn.get("text", "")),
    }


def _is_failure(row: dict[str, Any], gt: dict[str, Any], kind: str) -> bool:
    row_kind = str(row.get("target_type", "utterance"))
    if kind != "all" and row_kind != kind:
        return False
    if row_kind == "action":
        return (
            str(row.get("predicted_action") or "") != str(gt.get("action") or "")
            or [str(x) for x in (row.get("predicted_slots") or [])] != gt.get("slots", [])
        )
    prediction = str(row.get("prediction") or "").strip()
    return prediction != str(gt.get("text") or "").strip()


def _card(title: str, body: str, klass: str = "") -> str:
    return f'<section class="card {klass}"><h2>{html.escape(title)}</h2>{body}</section>'


def _pre(label: str, value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    return f"<h3>{html.escape(label)}</h3><pre>{html.escape(value)}</pre>"


def _render_case(index: int, row: dict[str, Any], gt: dict[str, Any], conv: dict[str, Any]) -> str:
    prediction = {
        "target_type": row.get("target_type"),
        "turn_index": row.get("turn_index"),
        "predicted_action": row.get("predicted_action"),
        "predicted_slots": row.get("predicted_slots"),
        "response": row.get("prediction", ""),
        "schema_validation": row.get("action_schema_validation"),
    }
    truth = {
        "turn_type": gt.get("turn_type"),
        "gold_action": gt.get("action"),
        "gold_slots": gt.get("slots"),
        "reference_text": gt.get("text", ""),
    }
    meta = (
        f"<p><b>Case {index}</b> | convo={html.escape(str(row.get('convo_id')))} | "
        f"subflow={html.escape(str(row.get('subflow', conv.get('scenario', {}).get('subflow', ''))))} | "
        f"turn={row.get('turn_index')} | type={html.escape(str(row.get('target_type')))}</p>"
    )
    body = meta
    body += '<div class="grid">'
    body += _pre("Agent prediction", prediction)
    body += _pre("ABCD ground truth", truth)
    body += "</div>"
    body += _pre("Dialogue context", row.get("context", ""))
    body += "<details><summary>Exact LLM prompt</summary>"
    body += _pre("Prompt messages", _prompt_from_row(row))
    body += "</details>"
    body += "<details><summary>Raw ReAct / runtime trace</summary>"
    body += _pre("Trace", row.get("react_trace", []))
    body += "</details>"
    return _card(f"Fail case {index}", body, "failure")


def _render_training(index: int, row: dict[str, Any]) -> str:
    meta = (
        f"<p><b>Training turn {index}</b> | batch={row.get('batch_index')} | "
        f"convo={html.escape(str(row.get('convo_id')))} | turn={row.get('turn_index')} | "
        f"type={html.escape(str(row.get('target_type')))}</p>"
    )
    body = meta + _pre("Model output", {
        "predicted_action": row.get("predicted_action"),
        "predicted_slots": row.get("predicted_slots"),
        "prediction": row.get("prediction", ""),
    })
    body += _pre("Dialogue context", row.get("context", ""))
    body += "<details><summary>Exact training prompt</summary>"
    body += _pre("Prompt messages", _prompt_from_row(row))
    body += "</details>"
    return _card(f"Training trace {index}", body, "training")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize AWM fail cases and training traces")
    parser.add_argument("--run-dir", required=True, help="Completed outputs/awm_abcd_* directory")
    parser.add_argument("--n-turns", type=int, default=10, help="Number of distinct fail turns")
    parser.add_argument("--train-turns", type=int, default=6, help="Number of training turns")
    parser.add_argument(
        "--kind",
        choices=["action", "utterance", "all"],
        default="action",
        help="Fail-case type; default is action only because AST scores action turns",
    )
    parser.add_argument(
        "--train-kind",
        choices=["action", "utterance", "all"],
        default="action",
        help="Training trace type; default is action only",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-file", default=None, help="Optional test.json override")
    parser.add_argument("--output", default=None, help="HTML output path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / "test_turn_predictions.json"
    if not summary_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(
            f"{run_dir} is incomplete; expected summary.json and test_turn_predictions.json"
        )
    summary = _load_json(summary_path)
    subflow = summary.get("config", {}).get("subflow", "")
    split_file = Path(args.split_file) if args.split_file else (
        Path("data/eval/abcd/splits") / subflow / "test.json"
    )
    conversations = _load_json(split_file)
    conv_by_id = {str(c.get("convo_id", "")): c for c in conversations}
    test_rows = _load_json(predictions_path)

    candidates = []
    for row in test_rows:
        conv = conv_by_id.get(str(row.get("convo_id", "")), {})
        gt = _ground_truth(conv, row)
        if _is_failure(row, gt, args.kind):
            candidates.append((row, gt, conv))
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[:max(0, args.n_turns)]

    training_rows = []
    training_path = run_dir / "training_turns.jsonl"
    if training_path.exists():
        for line in training_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if args.train_kind == "all" or row.get("target_type") == args.train_kind:
                    training_rows.append(row)
    rng.shuffle(training_rows)
    training_rows = training_rows[:max(0, args.train_turns)]

    failure_html = "\n".join(
        _render_case(i, row, gt, conv) for i, (row, gt, conv) in enumerate(selected, 1)
    )
    training_html = "\n".join(
        _render_training(i, row) for i, row in enumerate(training_rows, 1)
    )
    output = Path(args.output) if args.output else run_dir / "awm_case_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AWM case report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; background:#f5f7fa; color:#182230; }}
h1 {{ margin-bottom:.2rem; }} h2 {{ color:#0b4f71; }} h3 {{ margin-bottom:.3rem; }}
.card {{ background:white; border:1px solid #d9e1e8; border-radius:8px; padding:1rem; margin:1rem 0; box-shadow:0 1px 3px #0001; }}
.failure {{ border-left:6px solid #d9534f; }} .training {{ border-left:6px solid #337ab7; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}
pre {{ white-space:pre-wrap; word-break:break-word; background:#f0f3f6; padding:.8rem; border-radius:5px; max-height:600px; overflow:auto; }}
summary {{ cursor:pointer; font-weight:bold; margin:.8rem 0; }} .muted {{ color:#687685; }}
@media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style></head><body>
<h1>AWM prediction and training trace report</h1>
<p class="muted">run={html.escape(str(run_dir))} | subflow={html.escape(str(subflow))} |
sampled_failures={len(selected)}/{len(candidates)} | sampled_training_turns={len(training_rows)}</p>
<h1>Failed agent turns</h1>
{failure_html or '<p>No matching failed turns were found.</p>'}
<h1>Training-time turns</h1>
{training_html or '<p>No training_turns.jsonl found. This run predates training trace persistence or did not finish a batch.</p>'}
</body></html>"""
    output.write_text(page, encoding="utf-8")
    selected_json = output.with_suffix(".json")
    selected_json.write_text(json.dumps({
        "run_dir": str(run_dir),
        "subflow": subflow,
        "selected_failures": [row for row, _, _ in selected],
        "selected_training_turns": training_rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"HTML report: {output}")
    print(f"Selected failures: {len(selected)} / {len(candidates)}")
    print(f"Selected training turns: {len(training_rows)}")


if __name__ == "__main__":
    main()
