#!/usr/bin/env python3
"""Retry failed online-refinement skill edits in a copy and re-evaluate it.

The source run is never modified. Only edits explicitly recorded with an
``error`` status are retried; successful edits are preserved. The report also
lists successful edits because their semantic correctness cannot be inferred
from application status alone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_tod.response_logger import ResponseLogger
from scripts.run_backbone_online_refine import _build_agent, _write
from scripts.run_subflow_eval import evaluate_agent_on_subflow, load_subflow_data
from skill_mining.online_refinement import (
    apply_dynamic_skill_operations,
    apply_working_skill_operations,
    _online_refinement_chat,
    _parse_json_object,
    load_skill_dag,
    merge_online_skill_additions,
    render_online_action_rules,
    render_online_resources,
    render_online_slot_policies,
    save_skill_dag,
)

# The repair/evaluation ResponseLogger creates this subdirectory inside the
# output directory before restore() runs, so it must not be mistaken for
# pre-existing output.
LOGGER_DIR_NAME = "llm_responses"


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _reflection_paths(run_dir: Path) -> list[Path]:
    paths = list((run_dir / "autonomous_reflection").glob("batch_*.json"))
    # Original online-refine runs stored post-hoc group reflections directly
    # under RUN/group_reflections; newer runs nest them under iterative_refinement.
    paths.extend((run_dir / "group_reflections").glob("reflection_*.json"))
    paths.extend((run_dir / "iterative_refinement").glob("**/group_reflections/reflection_*.json"))
    return sorted(paths, key=lambda path: str(path.relative_to(run_dir)).lower())


def _failed_edits(run_dir: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    dynamic: list[dict] = []
    semantic: list[dict] = []
    already_repaired: set[str] = set()
    successful: list[dict] = []
    failed_unreplayable: list[dict] = []
    payloads = []
    for path in _reflection_paths(run_dir):
        payload = _json(path)
        if payload is not None:
            payloads.append((path, payload))

    # A prior resume may already have repaired an operation. Do not insert it
    # a second time when constructing the independent restoration copy.
    for path, payload in payloads:
        for item in payload.get("resume_repair_skill_operations", []) or []:
            if isinstance(item, dict) and item.get("applied"):
                already_repaired.add(str(item.get("operation_id", "")))

    for path, payload in payloads:
        relative = str(path.relative_to(run_dir))
        operation_results = payload.get("applied_skill_operations")
        if operation_results is None:
            operation_results = payload.get("skill_operations", [])
        raw_operations = (
            payload.get("requested_skill_operations")
            or payload.get("proposed_skill_operations")
            or payload.get("skill_operations")
            or []
        )
        raw_by_id = {str(item.get("operation_id", "")): item
                     for item in raw_operations if isinstance(item, dict)}
        fallback_updates = (payload.get("semantic_fallback_updates")
                            or payload.get("accepted_updates")
                            or payload.get("accepted") or [])
        for item in operation_results:
            if not isinstance(item, dict):
                continue
            operation_id = str(item.get("operation_id", ""))
            if item.get("applied"):
                successful.append({"source": relative, **item})
                continue
            if not item.get("error"):
                continue
            if operation_id and operation_id in already_repaired:
                continue
            raw = raw_by_id.get(operation_id, item)
            if raw.get("match_text") and raw.get("op"):
                dynamic.append({"source": relative, **raw, "_logged_error": item.get("error")})
                continue
            matches = [
                update for update in fallback_updates
                if str(update.get("resource", "")) == str(item.get("resource", ""))
                and str(update.get("action", "")) == str(item.get("action", ""))
                and str(update.get("edge_id", "")) == str(item.get("edge_id", ""))
            ]
            if matches:
                semantic.extend({"source": relative, **update, "_logged_error": item.get("error")}
                                for update in matches)
            else:
                failed_unreplayable.append({"source": relative, **item})
    # Failed items can be duplicated across the detailed reflection JSON and
    # its wrapper. De-duplicate by source and stable operation/update identity.
    def dedupe(items: list[dict], keys: tuple[str, ...]) -> list[dict]:
        result, seen = [], set()
        for item in items:
            key = tuple(str(item.get(name, "")) for name in ("source", *keys))
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    dynamic = dedupe(dynamic, ("operation_id", "op", "match_text"))
    semantic = dedupe(semantic, ("resource", "op", "action", "edge_id", "content"))
    return dynamic, semantic, successful, failed_unreplayable


_FAILED_EDIT_REPAIR_PROMPT = """You are repairing failed edits to an executable ABCD task-oriented dialogue skill.

The recorded operations below failed because their old match_text anchors were not found in the CURRENT skill. Do not retry those anchors blindly. Inspect the current skill and decide where each supported correction belongs. You may consolidate duplicate or compatible edits. Preserve existing valid routing and ordered-slot behavior. Do not invent facts. If an operation is unsupported, redundant, conflicts with stronger skill evidence, or cannot be placed safely, mark it skip and explain why.

For each repair operation, choose a short exact excerpt that exists exactly once in current_skill as match_text. Use op=replace for a local revision, insert_before/insert_after for a new rule, or delete only when the old rule is demonstrably harmful. The executor applies operations in order; later match_text values may refer to text introduced by earlier repairs. Prefer concise local edits, not a rewritten skill.

<current_skill>
{skill}
</current_skill>

<failed_operations>
{operations}
</failed_operations>

Return valid JSON only:
{{"repairs":[{{"source_operation_ids":["original operation ids"],
"decision":"apply|skip",
"op":"replace|insert_before|insert_after|delete",
"match_text":"exact excerpt from the then-current skill",
"new_text":"complete replacement or insertion; empty only for delete",
"rationale":"why this placement preserves valid behavior and repairs the logged issue"}}]}}
"""


def _apply_llm_repair_payload(skill: str, payload: dict[str, Any]) -> tuple[str, list[dict]]:
    current = skill
    applied_ids: set[str] = set()
    results = []
    repairs = payload.get("repairs", [])
    if not isinstance(repairs, list):
        return current, [{"error": "repairs_not_a_list"}]
    for index, repair in enumerate(repairs, start=1):
        if not isinstance(repair, dict):
            results.append({"repair_index": index, "error": "repair_not_an_object"})
            continue
        item = dict(repair)
        item["repair_index"] = index
        if item.get("decision") != "apply":
            results.append({**item, "skipped": "llm_decided_not_to_apply"})
            continue
        operation = {
            key: item.get(key)
            for key in ("operation_id", "op", "resource", "edge_id", "action",
                        "match_text", "new_text", "occurrence", "rationale")
            if key in item
        }
        operation["operation_id"] = operation.get("operation_id") or f"llm-repair-{index:04d}"
        current, outcome = apply_dynamic_skill_operations(current, [operation], applied_ids)
        results.append({**item, **outcome[0]})
    return current, results


def _llm_repair_failed_edits(
    skill: str, failed_edits: list[dict], *, model: str,
    response_logger: ResponseLogger, workflow_id: str | None,
    max_retries: int = 3,
) -> tuple[str, list[dict], dict]:
    if not failed_edits:
        return skill, [], {"attempts": 0, "error": "no_failed_edits"}
    prompt = _FAILED_EDIT_REPAIR_PROMPT.format(
        skill=skill,
        operations=json.dumps(failed_edits, ensure_ascii=False, indent=2),
    )
    raw = ""
    last_error = ""
    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        try:
            raw = _online_refinement_chat(
                [{"role": "user", "content": prompt}], model=model,
                response_logger=response_logger,
                call_tag="online_refine_failed_edit_repair",
                workflow_id=workflow_id,
            )
            payload = _parse_json_object(raw)
            if isinstance(payload.get("repairs"), list):
                updated, results = _apply_llm_repair_payload(skill, payload)
                return updated, results, {
                    "attempts": attempt,
                    "prompt_chars": len(prompt),
                    "model_repairs": len(payload["repairs"]),
                    "error": "",
                }
            last_error = "empty_or_invalid_repair_json"
        except Exception as exc:
            last_error = repr(exc)
        if attempt < attempts:
            time.sleep(min(2 ** (attempt - 1), 4))
    return skill, [], {"attempts": attempts, "prompt_chars": len(prompt),
                       "error": last_error or "repair_call_failed"}


def _strip_rendered_suffix(combined: str, generated: str) -> str:
    suffix = "\n\n" + generated.rstrip()
    if generated.strip() and combined.rstrip().endswith(suffix):
        return combined.rstrip()[:-len(suffix)].rstrip() + "\n"
    return combined


def _load_resource(run_dir: Path, offline_dir: Path | None, name: str) -> str:
    if offline_dir:
        path = offline_dir / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    path = run_dir / name
    if not path.is_file():
        return ""
    value = path.read_text(encoding="utf-8")
    online_name = {
        "action_rules.md": "online_action_rules.md",
        "slot_policies.md": "online_slot_policies.md",
    }.get(name)
    if online_name and (run_dir / online_name).is_file():
        value = _strip_rendered_suffix(value, (run_dir / online_name).read_text(encoding="utf-8"))
    return value


def restore(
    run_dir: Path, output_dir: Path, offline_dir: Path | None, *,
    model: str, workflow_id: str | None, response_logger: ResponseLogger,
) -> dict[str, Any]:
    required = ["skill_dag_state.json", "base_skill.md", "base_reference.md"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source run is missing required files: {', '.join(missing)}")
    if output_dir.exists() and any(
        path.name != LOGGER_DIR_NAME for path in output_dir.iterdir()
    ):
        raise FileExistsError(f"output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    state = load_skill_dag(run_dir / "skill_dag_state.json")
    current_skill_path = run_dir / "working_skill.md"
    if not current_skill_path.is_file():
        current_skill_path = run_dir / "skill.md"
    if not current_skill_path.is_file():
        current_skill_path = run_dir / "base_skill.md"
    current_skill = current_skill_path.read_text(encoding="utf-8")
    dynamic, semantic, successful, unreplayable = _failed_edits(run_dir)
    skill, dynamic_results, llm_repair = _llm_repair_failed_edits(
        current_skill, dynamic, model=model, response_logger=response_logger,
        workflow_id=workflow_id, max_retries=3,
    )
    action_rules = _load_resource(run_dir, offline_dir, "action_rules.md")
    slot_policies = _load_resource(run_dir, offline_dir, "slot_policies.md")
    semantic_updates = [{k: v for k, v in item.items() if not k.startswith("_") and k != "source"}
                        for item in semantic]
    if semantic_updates:
        skill, semantic_results = apply_working_skill_operations(skill, state, semantic_updates)
    else:
        semantic_results = []
    skill = merge_online_skill_additions(skill, state)

    failed_results = [item for item in dynamic_results + semantic_results if item.get("error")]
    unresolved = [item for item in failed_results if item.get("historical_status") == "failed"]
    llm_skipped = [item for item in dynamic_results if item.get("skipped")]
    dynamic_applied = [item for item in dynamic_results if item.get("applied")]
    report = {
        "source_run": str(run_dir),
        "restored_skill_source": str(current_skill_path),
        "repair_strategy": "llm_relocates_failed_edits_on_current_skill",
        "scanned_reflection_files": [
            str(path.relative_to(run_dir)) for path in _reflection_paths(run_dir)
        ],
        "num_scanned_reflection_files": len(_reflection_paths(run_dir)),
        "failed_operations_sent_to_llm": len(dynamic),
        "llm_repair_call": llm_repair,
        "llm_repair_suggested": len(dynamic_results),
        "llm_repair_applied": len(dynamic_applied),
        "llm_repair_skipped_by_model": len(llm_skipped),
        "dynamic_retry_results": dynamic_results,
        "attempted_semantic_retries": len(semantic_results),
        "semantic_retry_results": semantic_results,
        "unreplayable_logged_failures": unreplayable,
        "successful_logged_operations_for_review": successful,
        "unresolved_retry_errors": [
            item for item in dynamic_results + semantic_results
            if item.get("error") or item.get("skipped")
        ],
        "note": "Failed edits were reinterpreted by an LLM against the current skill. Previously applied edits were preserved; their semantic correctness is not inferred.",
    }
    base_reference = (run_dir / "base_reference.md").read_text(encoding="utf-8")
    _write(output_dir / "base_skill.md", (run_dir / "base_skill.md").read_text(encoding="utf-8"))
    _write(output_dir / "restored_skill.md", skill)
    _write(output_dir / "working_skill.md", skill)
    _write(output_dir / "skill.md", skill)
    _write(output_dir / "base_reference.md", base_reference)
    _write(output_dir / "action_rules.md", action_rules)
    _write(output_dir / "slot_policies.md", slot_policies)
    _write(output_dir / "reference.md", base_reference.rstrip() + "\n\n" + render_online_resources(state)[1])
    save_skill_dag(state, output_dir / "skill_dag_state.json")
    _write(output_dir / "online_transition_guards.md", render_online_resources(state)[0])
    _write(output_dir / "online_action_rules.md", render_online_action_rules(state))
    _write(output_dir / "online_slot_policies.md", render_online_slot_policies(state))
    _write(output_dir / "restoration_report.json", json.dumps(report, indent=2, ensure_ascii=False))
    return {"state": state, "skill": skill, "base_reference": base_reference,
            "action_rules": action_rules, "slot_policies": slot_policies, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed online-refine run directory")
    parser.add_argument("--subflow", required=True, help="ABCD split name, e.g. account_access")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, empty restoration/evaluation directory")
    parser.add_argument("--offline-dir", type=Path, default=None,
                        help="Optional original offline artifact directory for base action/slot resources")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--workflow-id", default="", help="Single evaluation workflow id")
    parser.add_argument("--repair-workflow-id", default="",
                        help="Workflow for failed-edit repair; defaults to --workflow-id or first --eval-workflow-ids")
    parser.add_argument("--eval-workflow-ids", default="",
                        help="Optional comma-separated evaluation workflows (fork-capable Linux only)")
    parser.add_argument("--reference-top-k", type=int, default=3)
    parser.add_argument("--reference-max-chars", type=int, default=1800)
    parser.add_argument("--skip-utterance-eval", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true", help="Restore and report only")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    offline_dir = args.offline_dir.resolve() if args.offline_dir else None
    eval_ids = [value.strip() for value in args.eval_workflow_ids.split(",") if value.strip()]
    workflow_id = args.workflow_id.strip() or None
    repair_workflow_id = (args.repair_workflow_id.strip() or workflow_id
                          or (eval_ids[0] if eval_ids else None))
    logger = ResponseLogger(output_dir / LOGGER_DIR_NAME)
    restored = restore(
        run_dir, output_dir, offline_dir, model=args.model,
        workflow_id=repair_workflow_id, response_logger=logger,
    )
    report = restored["report"]
    print(f"Restoration written: {output_dir}")
    print(f"Failed edits sent to LLM: {report['failed_operations_sent_to_llm']}; "
          f"LLM edits applied: {report['llm_repair_applied']}; "
          f"{len(report['unresolved_retry_errors'])} remain unresolved")
    print(f"Successful historical operations listed for review: "
          f"{len(report['successful_logged_operations_for_review'])}")
    if args.no_evaluate:
        return

    train, test = load_subflow_data(args.subflow)
    workflow_id = eval_ids[0] if eval_ids else workflow_id
    if workflow_id:
        os.environ["SKILLMINING_WORKFLOW_ID"] = workflow_id
    agent_args = SimpleNamespace(
        model=args.model,
        reference_top_k=args.reference_top_k,
        reference_max_chars=args.reference_max_chars,
    )
    agent = _build_agent(
        agent_args, restored["skill"], restored["base_reference"],
        restored["action_rules"], restored["slot_policies"], restored["state"],
        response_logger=logger, workflow_id=workflow_id,
    )
    result = evaluate_agent_on_subflow(
        agent, test, "restored_online_refine", args.subflow,
        save_dir=output_dir, eval_workflow_ids=eval_ids or None,
        skip_utterance_eval=args.skip_utterance_eval,
    )
    _write(output_dir / "online_refine_result.json",
           json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result.get("ast_cds", {}), indent=2, ensure_ascii=False))
    print(f"Evaluation written: {output_dir / 'online_refine_result.json'}")


if __name__ == "__main__":
    main()
