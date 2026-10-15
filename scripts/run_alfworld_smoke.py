"""Run one genuine text ALFWorld task with the shared ReAct actor."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_tod.alfworld_runtime import PROTOCOL, TextWorldEpisode, load_fewshots, run_episode

TASK_TYPES = {"pick": "pick_and_place_simple", "examine": "look_at_obj_in_light",
              "clean": "pick_clean_then_place_in_recep", "heat": "pick_heat_then_place_in_recep",
              "cool": "pick_cool_then_place_in_recep", "pick_two": "pick_two_obj_and_place"}


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def select_game(data_root, split, task_type, task_index, gamefile=None):
    split_root = (data_root / "json_2.1.1" / split).resolve()
    if gamefile:
        candidates = [gamefile.expanduser().resolve()]
    else:
        candidates = sorted(split_root.rglob("game.tw-pddl"))
    selected = []
    for path in candidates:
        if not path.is_relative_to(split_root):
            raise ValueError("Explicit gamefile must belong to the requested split")
        if "movable" in str(path) or "Sliced" in str(path):
            continue
        metadata = path.parent / "traj_data.json"
        if not metadata.is_file():
            continue
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if info.get("task_type") == TASK_TYPES[task_type] and json.loads(
                path.read_text(encoding="utf-8")).get("solvable"):
            selected.append(path)
    if task_index >= len(selected):
        raise ValueError(f"Found {len(selected)} solvable {task_type} games in {split_root}; "
                         "check --data-root and download ALFWorld data first")
    return selected[task_index]


def load_env_config(path, data_root, max_steps):
    import yaml
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, split in (("data_path", "train"), ("eval_id_data_path", "valid_seen"),
                       ("eval_ood_data_path", "valid_unseen")):
        cfg["dataset"][key] = str(data_root / "json_2.1.1" / split)
    cfg["dataset"].update(num_train_games=-1, num_eval_games=-1)
    cfg["logic"].update(domain=str(data_root / "logic" / "alfred.pddl"),
                         grammar=str(data_root / "logic" / "alfred.twl2"))
    cfg["env"].update(type="AlfredTWEnv", domain_randomization=False,
                       goal_desc_human_anns_prob=0.0, task_types=[1, 2, 3, 4, 5, 6])
    # 'dqn' here selects the environment wrapper only. No DQN model is used;
    # unlike the dagger wrapper it does not request an expert plan on train.
    cfg["general"].update(use_cuda=False, training_method="dqn")
    cfg["rl"]["training"]["max_nb_steps_per_episode"] = max_steps
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path(os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld")))
    p.add_argument("--split", choices=("train", "valid_seen", "valid_unseen"), default="train")
    p.add_argument("--task-type", choices=TASK_TYPES, default="pick")
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--gamefile", type=Path)
    p.add_argument("--env-config", type=Path, default=ROOT / "ExpeL/configs/benchmark/alfworld.yaml")
    p.add_argument("--fewshot-source", type=Path, default=ROOT / "ExpeL/prompts/alfworld.py")
    p.add_argument("--llm-module", choices=("llm_new", "llm"), default="llm_new")
    p.add_argument("--workflow-id", help="Overrides SKILLMINING_WORKFLOW_ID for this process only")
    p.add_argument("--model", default=None, help="Default: model from project LLM_CONFIG")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-calls", type=int, default=60)
    p.add_argument("--llm-attempts", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skill-file", type=Path, help="Optional frozen guidance; same actor, no mining/refinement")
    p.add_argument("--output-dir", type=Path, default=ROOT / f"outputs/alfworld_smoke_{datetime.now():%Y%m%d_%H%M%S}")
    p.add_argument("--dry-run", action="store_true", help="Select game and validate inputs without environment/LLM calls")
    args = p.parse_args()
    if min(args.max_steps, args.max_calls, args.llm_attempts) <= 0 or args.task_index < 0:
        p.error("budgets must be positive and task-index nonnegative")
    root = args.data_root.expanduser().resolve()
    # All selection/config preconditions precede LLM import or calls.
    try:
        game = select_game(root, args.split, args.task_type, args.task_index, args.gamefile)
        cfg = load_env_config(args.env_config, root, args.max_steps)
        for path in (Path(cfg["logic"]["domain"]), Path(cfg["logic"]["grammar"])):
            if not path.is_file():
                raise FileNotFoundError(path)
        keys, fewshots = load_fewshots(args.fewshot_source)
        skill = args.skill_file.read_text(encoding="utf-8") if args.skill_file else ""
    except (OSError, ValueError) as exc:
        p.exit(2, f"Preflight failed: {exc}\n")
    manifest = {"protocol": PROTOCOL, "gamefile": str(game), "game_sha256": hashlib.sha256(game.read_bytes()).hexdigest(),
                "split": args.split, "task_type": args.task_type, "seed": args.seed,
                "max_steps": args.max_steps, "max_actor_calls": args.max_calls,
                "llm_attempts_per_call": args.llm_attempts, "llm_module": args.llm_module,
                "fewshot_keys": keys, "fewshot_sha256": hashlib.sha256(fewshots.encode()).hexdigest(),
                "fewshot_source": str(args.fewshot_source), "fewshot_routing": "fixed pair, no task-label routing",
                "skill_sha256": hashlib.sha256(skill.encode()).hexdigest() if skill else None,
                "expert_plan_visible": False, "admissible_commands_visible": False,
                "repeat_action_early_stop": False, "full_history": True,
                "decision_protocol": "one ReAct generation per actor call; deterministic action/slot parsing (not ABCD two-stage)",
                "adaptation": "ExpeL examples + independent shared loop; NOT byte-identical ExpeL agent"}
    manifest["runtime_sha256"] = hashlib.sha256((ROOT / "eval_tod/alfworld_runtime.py").read_bytes()).hexdigest()
    manifest["environment_config"] = cfg
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        p.error("output directory must be fresh; no automatic overwrites or test retries")
    output.mkdir(parents=True, exist_ok=True)
    if args.workflow_id:
        os.environ["SKILLMINING_WORKFLOW_ID"] = args.workflow_id
    env, llm, result = None, None, {"success": False, "status": "error"}
    started = time.monotonic()
    def event(row):
        with (output / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row["event"] == "step":
            result["environment_steps"] = row["step"]
            print(f"[{row['step']}/{args.max_steps}] {row['command']} -> {row['observation']}", flush=True)
        if "call" in row:
            result["actor_calls"] = row["call"]
    try:
        llm = importlib.import_module(args.llm_module)
        llm.reset_usage_summary()
        llm_config = llm.get_llm_config() if hasattr(llm, "get_llm_config") else llm.resolve_config()
        # Credentials and workflow URL/ID are deliberately not persisted.
        manifest.update(model=args.model or llm_config.get("model"), provider=llm_config.get("provider", "openai_compatible"),
                        workflow_id_override_set=bool(args.workflow_id))
        for package in ("alfworld", "textworld", "gym"):
            try:
                manifest.setdefault("package_versions", {})[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                manifest.setdefault("package_versions", {})[package] = None
        save(output / "manifest.json", manifest)
        from eval_tod.response_logger import ResponseLogger
        logger = ResponseLogger(output / "llm_responses")
        env = TextWorldEpisode(cfg, game, args.split, args.seed)
        def chat(messages):
            for attempt in range(args.llm_attempts):
                try:
                    reply = llm.chat(messages, model=args.model, temperature=0,
                                     response_logger=logger, call_tag="alfworld_react")
                    if not isinstance(reply, str) or not reply.strip():
                        raise RuntimeError("Empty LLM response")
                    return reply
                except Exception as exc:
                    event({"event": "llm_error", "attempt": attempt + 1, "error": str(exc)})
                    if attempt + 1 == args.llm_attempts:
                        raise
                    time.sleep(2 ** attempt)
        result = run_episode(env, chat, fewshots=fewshots, max_steps=args.max_steps,
                             max_calls=args.max_calls, skill=skill, on_event=event)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        print(f"Smoke test error: {exc}", file=sys.stderr, flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                result["close_error"] = str(exc)
        result.update(elapsed_seconds=round(time.monotonic() - started, 3), protocol=PROTOCOL)
        if llm is not None:
            result["llm_usage"] = llm.get_usage_summary()
        save(output / "manifest.json", manifest)
        save(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Artifacts: {output}")
    return 2 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
