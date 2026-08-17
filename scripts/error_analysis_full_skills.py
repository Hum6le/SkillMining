#!/usr/bin/env python3
"""Full-corpus skill-content and sampled-error analysis.

Unlike the legacy error-analysis scripts, this runner makes one LLM call per
subflow for a joint skill audit and a sample of error cases. It then merges
subflows in batches and produces one final report.

Typical layout:
    skills_root/<subflow>/skill.md
    skills_root/<subflow>/reference.md
    skills_root/<subflow>/*predictions*.json
    data/eval/abcd/splits/<subflow>/test.json

Example (directory discovery):
    python scripts/error_analysis_full_skills.py \
        --skills-root outputs/skills \
        --predictions-root outputs/full_run \
        --output-dir outputs/full_skill_error_analysis

Example (manifest-driven discovery):
    python scripts/error_analysis_full_skills.py \
        --manifest /path/to/skill_manifest.json \
        --predictions-root /path/to/full_run \
        --output-dir outputs/full_skill_error_analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (str(_PROJECT_ROOT), str(_SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from eval_tod.abcd.agent import _parse_action_response, turn_results_to_abcd_predictions  # noqa: E402
from eval_tod.abcd.data import extract_ground_truth  # noqa: E402
from llm import chat  # noqa: E402


LOG = logging.getLogger("error_analysis_full_skills")
PREDICTION_NAMES = (
    # Prefer raw turn-level outputs.  The AST evaluator consumes these via
    # turn_results_to_abcd_predictions; *_abcd_predictions.json is derived
    # output and must only be used as a compatibility fallback.
    "evolved_test_turns.json",
    "mined_test_turns.json",
    "test_turns.json",
    "test_turn_predictions.json",
    "mined_predictions.json",
    "test_predictions.json",
    "predictions.json",
    "test_final_preds.json",
    "test_abcd_predictions.json",
    "evolved_test_abcd_predictions.json",
    "mined_test_abcd_predictions.json",
)


def _checkpoint_filename(method: str, subflow: str) -> str:
    safe_method = "".join(char if char.isalnum() or char in "-_" else "_" for char in method)
    safe_subflow = "".join(char if char.isalnum() or char in "-_" else "_" for char in subflow)
    return f"{safe_method}__{safe_subflow}.json"


def _load_subflow_checkpoints(directory: Path) -> dict[tuple[str, str], dict[str, Any]]:
    checkpoints: dict[tuple[str, str], dict[str, Any]] = {}
    if not directory.exists():
        return checkpoints
    for path in directory.glob("*.json"):
        try:
            row = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(row, dict) or not row.get("method") or not row.get("subflow"):
            continue
        checkpoints[(str(row["method"]), str(row["subflow"]))] = row
    return checkpoints


def parse_experiment_manifest(manifest_path: Path) -> list[tuple[str, Path]]:
    """Parse the text manifest emitted by run_full_abcd_experiments.sh.

    It is intentionally not parsed as JSON.  The runner writes method sections
    such as ``[AWM run directories]`` followed by one run directory per line.
    """
    entries: list[tuple[str, Path]] = []
    method = None
    section_map = {
        "awm run directories": "awm",
        "expel run directories": "expel",
        "trace2skill run directories": "trace2skill",
        "graph mining run directories": "graph",
    }
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            method = section_map.get(line[1:-1].strip().lower())
            continue
        if line.startswith("["):
            method = None
            continue
        if method:
            entries.append((method, Path(line)))
    return entries


def _subflow_from_run_dir(run_dir: Path) -> str | None:
    """Recover the subflow from a run's summary/config when available."""
    summary = run_dir / "summary.json"
    if summary.exists():
        try:
            payload = load_json(summary)
            config = payload.get("config", {}) if isinstance(payload, dict) else {}
            value = config.get("subflow")
            if value:
                return str(value)
        except (OSError, json.JSONDecodeError):
            pass
    # Graph runs contain one directory per subflow.
    return None


def discover_skill_entries_from_experiment_manifest(
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Expand method-level run directories into analyzable skill entries.

    The manifest records runs, not skill files.  Expansion rules mirror the
    actual runners: graph runs contain ``<subflow>/skill.md``; Trace2Skill
    stores ``evolved_skill/SKILL.md``; AWM stores ``awm_workflow.txt`` plus
    ``awm_reference.md``; ExpeL has ``expel_rules.json`` as its learned rule
    artifact but no markdown skill.
    """
    entries: list[dict[str, Any]] = []
    for method, raw_run_dir in parse_experiment_manifest(manifest_path):
        run_dir = raw_run_dir
        if not run_dir.is_absolute():
            run_dir = manifest_path.parent / run_dir
        run_dir = run_dir.resolve()
        if method == "graph":
            for skill_path in sorted(run_dir.rglob("skill.md")):
                entries.append({
                    "method": method,
                    "subflow": skill_path.parent.name,
                    "skill_path": skill_path,
                    "run_dir": run_dir,
                })
            continue

        subflow = _subflow_from_run_dir(run_dir)
        if method == "trace2skill":
            skill_path = run_dir / "evolved_skill" / "SKILL.md"
        elif method == "awm":
            skill_path = run_dir / "awm_workflow.txt"
        elif method == "expel":
            skill_path = run_dir / "expel_rules.json"
        else:
            skill_path = None
        if skill_path and skill_path.exists():
            entries.append({
                "method": method,
                "subflow": subflow or run_dir.name,
                "skill_path": skill_path,
                "run_dir": run_dir,
            })
    return entries


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_provenance(path: Path | None) -> dict[str, Any] | None:
    """Record a lightweight identity for every artifact used in attribution."""
    if path is None or not path.exists():
        return None
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "modified_at": path.stat().st_mtime,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def as_prediction_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("predictions", "turn_results", "rows", "results"):
            if isinstance(value.get(key), list):
                return value[key]
    return []


def discover_skill_dirs(root: Path) -> dict[str, Path]:
    """Find one skill artifact per subflow.

    The normal layout is ``root/<subflow>/skill.md``.  For convenience, also
    accept ``root/<subflow>.md``; this is useful when a miner writes one file
    per subflow rather than one directory per subflow.  If the same subflow is
    found more than once, prefer the artifact with the most complete local
    bundle and then the newest artifact.
    """
    candidates: dict[str, list[Path]] = {}
    for name in ("skill.md", "SKILL.md"):
        for path in root.rglob(name):
            candidates.setdefault(path.parent.name, []).append(path.parent)
    # Also support a flat corpus: skills_root/<subflow>.md.
    for path in root.glob("*.md"):
        if path.name.lower() in {"reference.md", "readme.md"}:
            continue
        candidates.setdefault(path.stem, []).append(path)

    result: dict[str, Path] = {}
    for subflow, dirs in candidates.items():
        # Prefer the directory with the most complete local artifact set.
        result[subflow] = max(
            set(dirs),
            key=lambda d: (
                int(d.is_dir()),
                int(d.is_dir() and (d / "reference.md").exists()),
                int(d.is_dir() and any((d / name).exists() for name in PREDICTION_NAMES)),
                d.stat().st_mtime,
                str(d),
            ),
        )
    return dict(sorted(result.items()))


def _manifest_path(value: Any) -> Path | None:
    """Convert a manifest path-like value to a Path, if it looks usable."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    # Ignore URLs and ordinary prose accidentally found while walking JSON.
    if "://" in text or "\n" in text:
        return None
    return Path(text)


def discover_skill_dirs_from_manifest(manifest_path: Path) -> dict[str, Path]:
    """Read ``subflow -> skill directory`` mappings from a JSON manifest.

    The writer/pipeline has used several closely related manifest shapes over
    time, for example::

        {"cost": "/run/skills/cost"}
        {"subflows": [{"subflow": "cost", "skill_dir": "/run/skills/cost"}]}
        {"skills": {"cost": {"directory": "/run/skills/cost"}}}

    This parser deliberately accepts all of these without requiring a fixed
    schema.  It only treats fields whose names explicitly indicate a skill
    path as directories, so unrelated paths in the manifest are ignored.
    """
    data = load_json(manifest_path)
    path_keys = {
        "skill_dir", "skill_directory", "skill_path", "skill_folder",
        "directory", "dir", "folder", "path", "output_dir", "output_directory",
    }
    name_keys = {"subflow", "subflow_name", "skill", "skill_name", "name", "id"}
    result: dict[str, Path] = {}

    def add(name: Any, value: Any) -> None:
        path = _manifest_path(value)
        if path is None or not isinstance(name, str) or not name.strip():
            return
        candidate = path
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        if candidate.suffix.lower() in {".md", ".markdown"}:
            candidate = candidate.parent
        result[name.strip()] = candidate

    def walk(node: Any, parent_name: str | None = None) -> None:
        if isinstance(node, dict):
            local_name = next(
                (node[key] for key in name_keys if isinstance(node.get(key), str)),
                parent_name,
            )
            for key, value in node.items():
                key_lower = str(key).lower()
                if key_lower in path_keys:
                    add(local_name, value)
                elif isinstance(value, str) and local_name is None:
                    # Mapping form: {"subflow_name": "/path/to/skill"}.
                    add(key, value)
                elif isinstance(value, (dict, list)):
                    # A mapping form may nest one level below ``skills`` or
                    # ``subflows``; retain the key as a possible subflow name.
                    child_name = key if key_lower not in {"skills", "subflows", "items", "entries"} else local_name
                    walk(value, child_name)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, str):
                    path = _manifest_path(item)
                    if path is not None:
                        if not path.is_absolute():
                            path = manifest_path.parent / path
                        add(path.name, str(path))
                else:
                    walk(item, parent_name)

    walk(data)
    return dict(sorted(result.items()))


def find_prediction_file(
    skill_dir: Path,
    predictions_root: Path | None,
    subflow: str,
    run_dir: Path | None = None,
) -> Path | None:
    """Find predictions generated by the exact run that produced a skill.

    A post-fix skill must never be explained with pre-fix failures.  When a
    manifest supplies a run directory, only artifacts underneath that directory
    are eligible.  The optional predictions root is retained only for legacy
    directory discovery, where the user explicitly scopes it to one matching
    run.
    """
    search_dir = skill_dir.parent if skill_dir.is_file() else skill_dir
    if run_dir is not None:
        bounded_root = run_dir.resolve()
        current = search_dir.resolve()
        while True:
            for name in PREDICTION_NAMES:
                candidate = current / name
                if candidate.exists():
                    return candidate
            if current == bounded_root:
                break
            if bounded_root not in current.parents:
                break
            current = current.parent
        return None

    # Legacy skills-root mode has no run identity. Restrict lookup to the
    # artifact directory itself; a separate root is an explicit user-provided
    # scope and must contain exactly one matching prediction artifact.
    for name in PREDICTION_NAMES:
        candidate = search_dir / name
        if candidate.exists():
            return candidate
    if predictions_root and predictions_root.exists():
        candidates = [
            path
            for name in PREDICTION_NAMES
            for path in predictions_root.rglob(name)
            if subflow in path.parts or subflow in path.stem
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            LOG.warning(
                "Ambiguous prediction artifacts for %s under %s; refusing to mix runs: %s",
                subflow, predictions_root, [str(path) for path in candidates],
            )
    return None


def find_test_file(test_root: Path, subflow: str) -> Path | None:
    candidates = [
        test_root / subflow / "test.json",
        test_root / f"test_{subflow}.json",
        test_root / f"{subflow}_test.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def find_react_file(prediction_file: Path) -> Path | None:
    name = prediction_file.name
    if name.endswith("_predictions.json"):
        inferred = prediction_file.with_name(name.replace("_predictions.json", "_react_traces.json"))
        if inferred.exists():
            return inferred
    for name in ("test_react_traces.json", "mined_react_traces.json", "react_traces.json"):
        candidate = prediction_file.parent / name
        if candidate.exists():
            return candidate
    return None


def build_react_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup = {}
    for row in rows:
        try:
            lookup[(str(row.get("convo_id", "")), int(row.get("turn_index", 0)))] = row
        except (TypeError, ValueError):
            continue
    return lookup


def normalize_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for original in rows:
        # Compatibility with already serialized ABCDPrediction files.  The
        # normal path uses raw turn-level files, but converting these records
        # here prevents a silent all-missing-predictions result if a derived
        # file is the only artifact available.
        if isinstance(original.get("turns"), list) and "convo_id" not in original:
            conversation_id = str(original.get("conversation_id", ""))
            for turn in original["turns"]:
                if not isinstance(turn, dict) or turn.get("turn_type") != "action":
                    continue
                normalized.append({
                    "convo_id": conversation_id,
                    "turn_index": turn.get("turn_index", 0),
                    "target_type": "action",
                    "predicted_action": turn.get("predicted_action") or "",
                    "predicted_slots": turn.get("predicted_slots") or [],
                })
            continue
        row = dict(original)
        row["convo_id"] = str(row.get("convo_id", ""))
        if "predicted_action" not in row and "prediction" in row:
            action, slots, response = _parse_action_response(row.get("prediction", ""))
            row["predicted_action"] = action
            row["predicted_slots"] = slots
            if response and not row.get("response_text"):
                row["response_text"] = response
        row.setdefault("predicted_slots", [])
        normalized.append(row)
    return sorted(normalized, key=lambda row: (row["convo_id"], int(row.get("turn_index", 0))))


def find_agent_row(rows: list[dict[str, Any]], convo_id: str, action_turn: int) -> dict[str, Any]:
    best = {}
    for row in rows:
        if str(row.get("convo_id", "")) != convo_id:
            continue
        if int(row.get("turn_index", 999999)) < action_turn:
            best = row
        else:
            break
    return best


def context_before(conv: dict[str, Any], turn_index: int, window: int = 6) -> str:
    lines = []
    for turn in conv.get("delexed", [])[max(0, turn_index - window):turn_index]:
        speaker = turn.get("speaker", "unknown")
        label = {"agent": "Agent", "customer": "Customer", "action": "SystemAction"}.get(speaker, speaker)
        text = str(turn.get("text", "")).strip()
        if text:
            lines.append(f"[{label}] {text}")
    return "\n".join(lines)


def classify_errors(
    raw_predictions: list[dict[str, Any]],
    test_convs: list[dict[str, Any]],
    react_traces: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Local copy of the legacy AST classification without its config dependency."""
    rows = normalize_predictions(raw_predictions)
    react_lookup = build_react_lookup(react_traces)
    for conv in test_convs:
        conv["convo_id"] = str(conv.get("convo_id", ""))
    predictions = turn_results_to_abcd_predictions(rows, test_convs)
    pred_by_cid = {
        str(pred.conversation_id): {
            turn.turn_index: (turn.predicted_action or "", turn.predicted_slots or [])
            for turn in pred.turns
        }
        for pred in predictions
    }
    categories = {
        "joint_pass": [],
        "joint_fail": [],
        "action_ok_slot_wrong": [],
        "action_wrong_slot_ok": [],
        "both_wrong": [],
        "missing_prediction": [],
    }
    action_counts: dict[str, dict[str, int]] = {}
    total = action_correct = slot_correct = joint_correct = 0
    for conv in test_convs:
        cid = str(conv.get("convo_id", ""))
        turns = pred_by_cid.get(cid, {})
        for truth in extract_ground_truth(conv):
            if truth.turn_type != "action" or not truth.action_name:
                continue
            total += 1
            pred_action, pred_slots = turns.get(truth.turn_index, ("", []))
            gt_slots = truth.slot_values or []
            action_ok = pred_action == truth.action_name
            slot_ok = pred_slots == gt_slots
            joint_ok = action_ok and slot_ok
            action_correct += int(action_ok)
            slot_correct += int(slot_ok)
            joint_correct += int(joint_ok)
            action_counts.setdefault(truth.action_name, {"total": 0, "errors": 0})
            action_counts[truth.action_name]["total"] += 1
            action_counts[truth.action_name]["errors"] += int(not joint_ok)
            agent_row = find_agent_row(rows, cid, truth.turn_index)
            agent_turn = agent_row.get("turn_index")
            trace = agent_row.get("react_trace", [])
            if agent_turn is not None:
                trace = trace or react_lookup.get((cid, int(agent_turn)), {}).get("react_trace", [])
            entry = {
                "convo_id": cid,
                "action_turn": truth.turn_index,
                "gt_action": truth.action_name,
                "pred_action": pred_action,
                "gt_slots": gt_slots,
                "pred_slots": pred_slots,
                "reference": str(truth.text or "")[:600],
                "prediction": str(agent_row.get("response_text") or agent_row.get("prediction") or "")[:600],
                "context": context_before(conv, truth.turn_index),
                "reference_lookup": agent_row.get("reference_lookup", {}),
                "react_trace": trace,
            }
            if joint_ok:
                categories["joint_pass"].append(entry)
            else:
                categories["joint_fail"].append(entry)
                if not pred_action and not pred_slots:
                    categories["missing_prediction"].append(entry)
                if action_ok and not slot_ok:
                    categories["action_ok_slot_wrong"].append(entry)
                elif not action_ok and slot_ok:
                    categories["action_wrong_slot_ok"].append(entry)
                else:
                    categories["both_wrong"].append(entry)
    stats = {
        "num_conversations": len(test_convs),
        "total_action_turns": total,
        "ast_joint": round(joint_correct / max(total, 1), 4),
        "ast_action_name": round(action_correct / max(total, 1), 4),
        "ast_slot_value": round(slot_correct / max(total, 1), 4),
        "counts": {key: len(value) for key, value in categories.items()},
        "action_breakdown": [
            {"action": action, **counts}
            for action, counts in sorted(action_counts.items(), key=lambda pair: -pair[1]["errors"])
        ],
    }
    return categories, stats


def sample_errors(categories: dict[str, list[dict[str, Any]]], limit: int, seed: int) -> list[dict[str, Any]]:
    """Sample jointly, with round-robin coverage of error categories."""
    category_order = (
        "both_wrong",
        "action_wrong_slot_ok",
        "action_ok_slot_wrong",
        "missing_prediction",
    )
    pools = {name: list(categories.get(name, [])) for name in category_order}
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    sampled = []
    while len(sampled) < limit and any(pools.values()):
        for name in category_order:
            if pools[name] and len(sampled) < limit:
                row = dict(pools[name].pop())
                row["error_category"] = name
                sampled.append(row)
    return sampled


def compact_error(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "convo_id": row.get("convo_id"),
        "action_turn": row.get("action_turn"),
        "error_category": row.get("error_category"),
        "gt_action": row.get("gt_action"),
        "pred_action": row.get("pred_action"),
        "gt_slots": row.get("gt_slots", []),
        "pred_slots": row.get("pred_slots", []),
        "context": str(row.get("context", ""))[:1400],
        "reference": str(row.get("reference", ""))[:600],
        "prediction": str(row.get("prediction", ""))[:600],
        "reference_lookup": row.get("reference_lookup", {}),
        "react_trace": row.get("react_trace", []),
    }


def build_skill_excerpt_catalog(skill_text: str, max_excerpts: int = 24) -> list[dict[str, str]]:
    """Split a skill into quoteable, stable evidence units.

    Generated skills are not guaranteed to share a schema, so Markdown
    headings are preferred when present and bounded line chunks are the
    fallback.  The IDs are included in every LLM prompt and in the rendered
    evidence ledger, making a reported case-to-rule connection auditable.
    """
    lines = [line.rstrip() for line in skill_text.splitlines()]
    if not lines:
        return []

    blocks: list[tuple[str, list[str]]] = []
    current_title = "preamble"
    current_lines: list[str] = []
    for line in lines:
        if line.lstrip().startswith("#"):
            if current_lines:
                blocks.append((current_title, current_lines))
            current_title = line.strip().lstrip("#").strip() or "untitled"
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        blocks.append((current_title, current_lines))

    catalog: list[dict[str, str]] = []
    for title, block_lines in blocks:
        text = "\n".join(block_lines).strip()
        if not text:
            continue
        # Keep a quoteable local rule unit rather than an entire long skill.
        for offset in range(0, len(text), 1200):
            excerpt = text[offset:offset + 1200].strip()
            if not excerpt:
                continue
            catalog.append({
                "excerpt_id": f"S{len(catalog) + 1:02d}",
                "section": title,
                "text": excerpt,
            })
            if len(catalog) >= max_excerpts:
                return catalog
    return catalog


def render_evidence_ledger(rows: list[dict[str, Any]]) -> str:
    """Render all sampled cases and quoteable skill excerpts for manual audit."""
    output = ["# Case-to-Skill Evidence Ledger", ""]
    for row in rows:
        output.extend([
            f"## {row.get('method', 'unknown')}/{row['subflow']}",
            "",
            "### Sampled fail cases",
            "",
        ])
        cases = row.get("sampled_errors", [])
        if not cases:
            output.append("No sampled fail case was available for this subflow.")
        for case in cases:
            output.extend([
                f"#### {case.get('case_id', 'CASE-UNKNOWN')}",
                f"- category: `{case.get('error_category')}`",
                f"- action: gold `{case.get('gt_action')}` vs predicted `{case.get('pred_action')}`",
                f"- slots: gold `{case.get('gt_slots', [])}` vs predicted `{case.get('pred_slots', [])}`",
                f"- context: {case.get('context', '')}",
                "",
            ])
        output.extend(["### Skill excerpts", ""])
        excerpts = row.get("skill_excerpts", [])
        if not excerpts:
            output.append("No skill text was available.")
        for excerpt in excerpts:
            output.extend([
                f"#### {excerpt['excerpt_id']} | {excerpt['section']}",
                "```markdown",
                excerpt["text"],
                "```",
                "",
            ])
    return "\n".join(output).rstrip() + "\n"


def call_llm(prompt: str, purpose: str) -> str:
    try:
        response = chat(prompt, temperature=0.0).strip()
        return response or f"[LLM empty response: {purpose}]"
    except Exception as exc:
        LOG.warning("%s failed: %s", purpose, exc)
        return f"[LLM error during {purpose}: {type(exc).__name__}: {exc}]"


def subflow_prompt(
    subflow: str,
    method: str,
    skill_text: str,
    skill_excerpts: list[dict[str, str]],
    reference_text: str,
    stats: dict[str, Any],
    errors: list[dict[str, Any]],
) -> str:
    return f"""Analyze one generated skill as a research artifact, not as a per-example debugging task.

Method: {method}
Subflow: {subflow}

## Skill content
```markdown
{skill_text[:14000] or '(missing)'}
```

## Quoteable skill excerpts
Use these stable IDs when citing a skill rule.
```json
{json.dumps(skill_excerpts, indent=2, ensure_ascii=False)}
```

## Reference content
```markdown
{reference_text[:9000] or '(missing)'}
```

## Evaluation statistics
```json
{json.dumps(stats, indent=2, ensure_ascii=False)}
```

## Sampled error cases (at most 10; analyze them jointly)
```json
{json.dumps([compact_error(row) for row in errors], indent=2, ensure_ascii=False)}
```

The analysis has two mandatory tracks for EVERY subflow. Track A audits the
skill text itself, even when prediction files or sampled errors are missing.
Track B jointly analyzes the sampled fail cases when they are available.
Respond in Chinese with these Markdown sections:

1. `## Skill审计结论`
   Judge whether the skill itself is incomplete, ambiguous, contradictory,
   instance-specific, incorrectly prioritized, or structurally hard to execute.
2. `## 关键问题`
   List concrete issues. For each issue include: type, severity, exact skill
   evidence with one or more excerpt IDs (for example `S03`), why it is
   ambiguous/wrong, and a concrete fix.
3. `## 错误样本归因`
   Cluster the sampled errors into recurring patterns. Do not write one
   independent paragraph per error. For EACH pattern, include a dedicated
   `### Evidence links` subsection with at least one `case_id`, the observed
   gold-vs-predicted discrepancy, one or more matching skill `excerpt_id`s,
   and a link verdict: `directly supported`, `partially supported`, or
   `not supported by the skill`. Separate skill-content failures from
   agent/parser/data failures.
4. `## Subflow-level insight`
   State what business/state-transition structure this subflow appears to
   require and what the skill fails to represent.

The skill audit must explicitly include: missing state guards, unclear
conditions, contradictory rules, collapsed branches, unsafe ordering,
instance-specific wording, and missing recovery behavior. Quote exact skill
text as evidence. If no skill defect is found, state what was checked and why
the skill is coherent. Separately explain which fail cases are attributable to
skill defects, which are execution/parser/data failures, and which possible
skill defects have not yet produced an observed failure.
Never cite a skill defect without an excerpt ID, and never call a failure
skill-linked without both a case ID and an excerpt ID.

Do not claim a rule is wrong merely because a sampled trace differs from it.
Distinguish alternative state-conditioned branches from genuine contradictions.
"""


def batch_prompt(batch: list[dict[str, Any]], batch_id: int, total_batches: int) -> str:
    payload = []
    for row in batch:
        payload.append({
            "method": row.get("method", "unknown"),
            "subflow": row["subflow"],
            "stats": row.get("stats", {}),
            "evidence_index": {
                "cases": [
                    {
                        "case_id": case.get("case_id"),
                        "category": case.get("error_category"),
                        "gold": {"action": case.get("gt_action"), "slots": case.get("gt_slots", [])},
                        "predicted": {"action": case.get("pred_action"), "slots": case.get("pred_slots", [])},
                    }
                    for case in row.get("sampled_errors", [])
                ],
                "skill_excerpts": [
                    {"excerpt_id": excerpt.get("excerpt_id"), "text": excerpt.get("text", "")[:500]}
                    for excerpt in row.get("skill_excerpts", [])
                ],
            },
            "skill_audit": row.get("analysis", ""),
        })
    return f"""You are consolidating skill audits across a batch of business subflows.

Batch {batch_id}/{total_batches}

```json
{json.dumps(payload, indent=2, ensure_ascii=False)}
```

Every subflow in this batch has a mandatory skill-text audit. Consolidate
skill-content defects separately from sampled fail-case causes. Do not treat a
missing prediction file as evidence that the skill is correct.

Produce a Chinese Markdown summary with:

## Batch-level recurring defects
Group issues by structural cause, not by subflow name. Focus on ambiguity,
missing state guards, conflicting rules, incorrect macro boundaries,
instance leakage, and missing recovery behavior.

## Skill-only defects
Summarize problems found directly in the generated skill text, including
problems that have not yet produced an observed execution failure. Report both
the number of affected subflows and representative quoted evidence.

## Evidence and prevalence
For each cause, list supporting subflows and distinguish broad patterns from
isolated cases. Keep the denominator explicit: count skill-audit coverage and
fail-case evidence separately. For each skill-linked pattern, retain at least
one `case_id` and one `excerpt_id` from the evidence index.

## Cross-subflow business insights
Identify common latent state variables or transition patterns.

## Method implications
Explain what the mining algorithm should change. Do not propose cosmetic
wording edits only.
"""


def final_prompt(batch_summaries: list[dict[str, Any]], overview: dict[str, Any]) -> str:
    return f"""Synthesize the final research report for a full-corpus skill audit.

## Corpus overview
```json
{json.dumps(overview, indent=2, ensure_ascii=False)}
```

## Batch summaries
```json
{json.dumps(batch_summaries, indent=2, ensure_ascii=False)}
```

Write a Chinese Markdown report with:

# Full-Corpus Skill Error Analysis

## Executive summary
State the dominant structural problems in the generated skills. The report
must cover all skill audits, not only subflows with sampled prediction errors.

## Evidence-backed findings
For every major finding, report prevalence across subflows and representative
subflow names. Separate direct skill-text defects, skill-linked fail cases,
and execution/model/parser/data problems. Explicitly report how many skills
were audited and how many had usable fail-case samples. Every finding claimed
to be skill-linked must include at least one concrete `case_id` and matching
`excerpt_id`; quote the relevant skill text briefly rather than citing an
abstract diagnosis alone.

## Cross-subflow failure taxonomy
Organize failures into reusable categories such as missing guard, branch
collapse, unsupported assumption, slot ambiguity, ordering error, recovery
gap, and instance memorization.

## Implications for the new mining method
Translate the findings into requirements for graph completion, hypergraph
backbone mining, branch extraction, and semantic reasoning.

## Prioritized algorithm changes
Give the top five changes, ordered by expected research value.

## Limitations and missing evidence
Explicitly identify subflows or conclusions weakened by missing prediction or
test files. Make clear that missing fail cases do not remove the subflow from
the skill-quality audit. Also state the limitation introduced by the
ten-case-per-subflow sampling protocol. Treat predictions that predate the
analyzed skill as unavailable post-fix evidence, never as valid fail cases.
"""


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def analyze_subflow(
    subflow: str,
    skill_dir: Path,
    predictions_root: Path | None,
    test_root: Path,
    sample_limit: int,
    seed: int,
    method: str = "unknown",
    run_dir: Path | None = None,
) -> dict[str, Any]:
    if skill_dir.is_file():
        skill_path = skill_dir
        reference_candidates = (
            skill_dir.parent / "reference.md",
            skill_dir.parent / "awm_reference.md",
            skill_dir.with_name(f"{skill_dir.stem}_reference.md"),
        )
        reference_path = next((path for path in reference_candidates if path.exists()), reference_candidates[0])
    else:
        skill_path = next((skill_dir / name for name in ("skill.md", "SKILL.md") if (skill_dir / name).exists()), None)
        reference_path = skill_dir / "reference.md"
    prediction_path = find_prediction_file(skill_dir, predictions_root, subflow, run_dir=run_dir)
    test_path = find_test_file(test_root, subflow)
    skill_text = skill_path.read_text(encoding="utf-8") if skill_path else ""
    reference_text = reference_path.read_text(encoding="utf-8") if reference_path.exists() else ""
    skill_excerpts = build_skill_excerpt_catalog(skill_text)
    skill_provenance = artifact_provenance(skill_path)
    prediction_provenance = artifact_provenance(prediction_path)
    stale_prediction = bool(
        skill_path
        and prediction_path
        and prediction_path.stat().st_mtime < skill_path.stat().st_mtime
    )

    base = {
        "subflow": subflow,
        "method": method,
        "run_dir": str(run_dir) if run_dir else None,
        "skill_dir": str(skill_dir),
        "skill_path": str(skill_path) if skill_path else None,
        "reference_path": str(reference_path) if reference_path.exists() else None,
        "prediction_path": str(prediction_path) if prediction_path else None,
        "artifact_provenance": {
            "skill": skill_provenance,
            "predictions": prediction_provenance,
        },
        "test_path": str(test_path) if test_path else None,
        "skill_audit_required": True,
        "skill_audit_input_available": bool(skill_text.strip()),
        "skill_chars": len(skill_text),
        "skill_excerpts": skill_excerpts,
    }
    missing = [name for name, path in (("predictions", prediction_path), ("test_data", test_path)) if not path]
    if stale_prediction:
        stats = {
            "status": "stale_predictions_rejected",
            "reason": "prediction artifact predates the analyzed skill artifact",
            "skill_provenance": skill_provenance,
            "prediction_provenance": prediction_provenance,
        }
        analysis = call_llm(
            subflow_prompt(subflow, method, skill_text, skill_excerpts, reference_text, stats, []),
            f"skill-only audit with stale predictions {method}/{subflow}",
        )
        base.update({
            "status": "skill_only_stale_predictions",
            "missing": [],
            "stats": stats,
            "error_counts": {},
            "sampled_errors": [],
            "analysis": analysis,
        })
        return base
    if not prediction_path or not test_path:
        stats = {"status": "evaluation_inputs_missing", "missing": missing}
        analysis = call_llm(
            subflow_prompt(subflow, method, skill_text, skill_excerpts, reference_text, stats, []),
            f"skill-only audit {method}/{subflow}",
        )
        base.update({
            "status": "skill_only",
            "missing": missing,
            "stats": stats,
            "error_counts": {},
            "sampled_errors": [],
            "analysis": analysis,
        })
        return base

    predictions = as_prediction_rows(load_json(prediction_path))
    test_convs = load_json(test_path)
    react_path = find_react_file(prediction_path)
    react_traces = as_prediction_rows(load_json(react_path)) if react_path else []
    categories, stats = classify_errors(predictions, test_convs, react_traces)
    sampled = sample_errors(categories, sample_limit, seed)
    for index, row in enumerate(sampled, start=1):
        row["case_id"] = f"{method}/{subflow}/C{index:02d}"
    analysis = call_llm(
        subflow_prompt(subflow, method, skill_text, skill_excerpts, reference_text, stats, sampled),
        f"subflow {method}/{subflow}",
    )
    base.update({
        "status": "analyzed",
        "stats": stats,
        "error_counts": stats.get("counts", {}),
        "sampled_errors": [compact_error(row) for row in sampled],
        "analysis": analysis,
    })
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skills-root", type=Path, default=None,
        help="Root containing <subflow>/skill.md; ignored when --manifest is supplied.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Text manifest emitted by run_full_abcd_experiments.sh.",
    )
    parser.add_argument("--predictions-root", type=Path, default=None)
    parser.add_argument("--test-root", type=Path, default=Path("data/eval/abcd/splits"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--resume-dir", type=Path, default=None,
        help="Resume from an existing analysis directory and skip completed subflows/batches.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-errors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--methods", nargs="+", choices=("awm", "expel", "trace2skill", "graph", "unknown"),
        default=None, help="Optional method filter when using a multi-method manifest.",
    )
    parser.add_argument(
        "--expected-subflows",
        type=int,
        default=None,
        help="Optional sanity check, e.g. 96; does not fabricate missing skills.",
    )
    parser.add_argument("--subflow", action="append", default=[])
    args = parser.parse_args()

    if not args.manifest and not args.skills_root:
        parser.error("one of --manifest or --skills-root is required")
    if args.manifest and not args.manifest.exists():
        parser.error(f"manifest does not exist: {args.manifest}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.resume_dir or args.output_dir or Path(f"outputs/full_skill_error_analysis_{timestamp}")
    if args.resume_dir and not args.resume_dir.exists():
        parser.error(f"resume directory does not exist: {args.resume_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(out_dir / "analysis.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )

    if args.manifest:
        manifest_entries = discover_skill_entries_from_experiment_manifest(args.manifest)
        LOG.info("Loaded %d method/subflow skill entries from manifest %s", len(manifest_entries), args.manifest)
    else:
        discovered = discover_skill_dirs(args.skills_root)
        manifest_entries = [
            {"method": "unknown", "subflow": subflow, "skill_path": skill_dir, "run_dir": None}
            for subflow, skill_dir in discovered.items()
        ]
        LOG.info("Discovered %d subflows under %s", len(manifest_entries), args.skills_root)
    if args.subflow:
        manifest_entries = [row for row in manifest_entries if row["subflow"] in set(args.subflow)]
    if args.methods:
        manifest_entries = [row for row in manifest_entries if row["method"] in set(args.methods)]
    LOG.info("Discovered %d method/subflow entries", len(manifest_entries))
    if args.expected_subflows is not None:
        discovered_subflows = {row["subflow"] for row in manifest_entries}
        if len(discovered_subflows) != args.expected_subflows:
            LOG.warning(
                "Expected %d unique subflows but discovered %d (total method entries: %d)",
                args.expected_subflows,
                len(discovered_subflows),
                len(manifest_entries),
            )

    subflow_cache_path = out_dir / "subflow_analyses.json"
    subflow_checkpoint_dir = out_dir / "subflow_checkpoints"
    subflow_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cached_subflows: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume_dir and subflow_cache_path.exists():
        try:
            cached_rows = load_json(subflow_cache_path)
            if isinstance(cached_rows, list):
                cached_subflows = {
                    (str(row.get("method", "unknown")), str(row.get("subflow", ""))): row
                    for row in cached_rows
                    if isinstance(row, dict)
                }
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("Could not load subflow checkpoint: %s", exc)
    checkpoint_rows = _load_subflow_checkpoints(subflow_checkpoint_dir)
    cached_subflows.update(checkpoint_rows)
    if checkpoint_rows:
        LOG.info("Loaded %d per-subflow checkpoint files", len(checkpoint_rows))

    subflows = []
    for index, entry in enumerate(manifest_entries, start=1):
        subflow = entry["subflow"]
        method = entry["method"]
        cache_key = (method, subflow)
        cached = cached_subflows.get(cache_key)
        if cached and cached.get("status") in {
            "analyzed", "skill_only", "skill_only_stale_predictions"
        }:
            LOG.info("[%d/%d] resuming completed %s/%s", index, len(manifest_entries), method, subflow)
            subflows.append(cached)
            continue
        skill_dir = Path(entry["skill_path"])
        LOG.info("[%d/%d] auditing %s/%s", index, len(manifest_entries), method, subflow)
        result = analyze_subflow(
            subflow,
            skill_dir,
            args.predictions_root,
            args.test_root,
            args.sample_errors,
            args.seed,
            method=method,
            run_dir=Path(entry["run_dir"]) if entry.get("run_dir") else None,
        )
        subflows.append(result)
        (subflow_checkpoint_dir / _checkpoint_filename(method, subflow)).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        subflow_cache_path.write_text(
            json.dumps(subflows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOG.info("Saved subflow checkpoint: %s", subflow_cache_path)

    skill_audited = [
        row for row in subflows
        if row["status"] in {"analyzed", "skill_only", "skill_only_stale_predictions"}
    ]
    evaluated = [row for row in subflows if row["status"] == "analyzed"]
    unavailable = [
        row for row in subflows
        if row["status"] not in {"analyzed", "skill_only", "skill_only_stale_predictions"}
    ]
    overview = {
        "num_discovered_subflows": len(subflows),
        "num_skill_audited_subflows": len(skill_audited),
        "num_evaluated_subflows": len(evaluated),
        "num_missing_skill_subflows": len(unavailable),
        "num_stale_prediction_subflows": sum(
            row["status"] == "skill_only_stale_predictions" for row in subflows
        ),
        "sample_errors_per_subflow": args.sample_errors,
        "total_action_turns": sum(row.get("stats", {}).get("total_action_turns", 0) for row in evaluated),
        "total_joint_failures": sum(row.get("stats", {}).get("counts", {}).get("joint_fail", 0) for row in evaluated),
        "total_sampled_fail_cases": sum(len(row.get("sampled_errors", [])) for row in evaluated),
        "missing_subflows": [row["subflow"] for row in unavailable],
        "stale_prediction_subflows": [
            row["subflow"] for row in subflows
            if row["status"] == "skill_only_stale_predictions"
        ],
    }

    (out_dir / "subflow_analyses.json").write_text(json.dumps(subflows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "overview.json").write_text(json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "case_skill_evidence_ledger.md").write_text(
        render_evidence_ledger(subflows), encoding="utf-8"
    )

    batches = chunks(skill_audited, args.batch_size)
    batch_cache_path = out_dir / "batch_summaries.json"
    cached_batches: dict[int, dict[str, Any]] = {}
    if args.resume_dir and batch_cache_path.exists():
        try:
            cached_rows = load_json(batch_cache_path)
            if isinstance(cached_rows, list):
                cached_batches = {
                    int(row["batch_id"]): row
                    for row in cached_rows
                    if isinstance(row, dict) and "batch_id" in row
                }
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            LOG.warning("Could not load batch checkpoint: %s", exc)

    batch_results = []
    for batch_id, batch in enumerate(batches, start=1):
        batch_subflows = [row["subflow"] for row in batch]
        cached_batch = cached_batches.get(batch_id)
        if cached_batch and cached_batch.get("subflows") == batch_subflows and cached_batch.get("summary"):
            LOG.info("Resuming completed batch %d/%d", batch_id, len(batches))
            batch_results.append(cached_batch)
            continue
        LOG.info("Summarizing batch %d/%d (%d subflows)", batch_id, len(batches), len(batch))
        prompt = batch_prompt(batch, batch_id, len(batches))
        (out_dir / "prompts").mkdir(exist_ok=True)
        (out_dir / "prompts" / f"batch_{batch_id:03d}.md").write_text(prompt, encoding="utf-8")
        batch_results.append({
            "batch_id": batch_id,
            "subflows": batch_subflows,
            "summary": call_llm(prompt, f"batch {batch_id}"),
        })
        batch_cache_path.write_text(
            json.dumps(batch_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOG.info("Saved batch checkpoint: %s", batch_cache_path)
    batch_cache_path.write_text(
        json.dumps(batch_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    final_prompt_text = final_prompt(batch_results, overview)
    final = call_llm(final_prompt_text, "final report")
    (out_dir / "prompts").mkdir(exist_ok=True)
    (out_dir / "prompts" / "final_report.md").write_text(final_prompt_text, encoding="utf-8")
    (out_dir / "final_report.md").write_text(final, encoding="utf-8")
    LOG.info("Done: %s", out_dir)
    print(json.dumps({**overview, "num_batches": len(batches), "output_dir": str(out_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
