#!/usr/bin/env python3
"""A3 frozen K1 organization: identical ReAct and resource pool, no training."""
from __future__ import annotations

from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_k1_context_ablation as core
from skill_mining.k1_flatten import RESOURCE_NAMES
from skill_mining.k1_organization import BASE_ARMS, VERSION, build_variants, random_arm


class OrganizationProfile:
    protocol = "abcd-k1-organization-v1"
    entrypoint = Path(__file__).resolve()
    extra_arguments = ("random_seeds",)

    def configure_parser(self, parser):
        parser.description = __doc__
        parser.set_defaults(skip_arms="")
        parser.add_argument("--random-seeds", default="0,1,2", help="Fixed comma-separated seeds; 'none' skips random hierarchy")
        parser.add_argument("--arms", help="Comma-separated family names: flat_react,source_grouped,ours_organized,random_hierarchy")

    def normalize_args(self, args, parser):
        if args.include_ours:
            parser.error("This experiment evaluates ours_organized, not historical Ours; use --arms/--skip-arms")
        try:
            seeds = [] if args.random_seeds == "none" else [int(s.strip()) for s in args.random_seeds.split(",")]
        except ValueError:
            parser.error("--random-seeds requires comma-separated integers or 'none'")
        if len(set(seeds)) != len(seeds):
            parser.error("--random-seeds must be unique")
        args.random_seeds = ",".join(map(str, seeds)) if seeds else "none"
        supported = (*BASE_ARMS, "random_hierarchy")
        if args.arms is not None:
            chosen = tuple(s.strip() for s in args.arms.split(",") if s.strip())
            if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(supported):
                parser.error("--arms must contain unique supported family names")
            args.skip_arms += tuple(a for a in supported if a not in chosen and a not in args.skip_arms)
        expanded = []
        for arm in args.skip_arms:
            expanded.extend([random_arm(s) for s in seeds] if arm == "random_hierarchy" else [arm])
        args.skip_arms = tuple(expanded)
        # Families are invocation choices, not frozen experiment settings.
        del args.arms

    def arms(self, args):
        seeds = self.seeds(args)
        return (*BASE_ARMS, *(random_arm(s) for s in seeds))

    @staticmethod
    def seeds(args):
        return [] if args.random_seeds == "none" else [int(s) for s in args.random_seeds.split(",")]

    def prepare(self, args, directory, output, flow):
        original, flat, config, provenance = core.prepare(directory, output, flow)
        variants = build_variants(original, flat, self.seeds(args))
        flat["organization_variants"] = variants
        for arm, variant in variants.items():
            dest = output / arm / "resources"
            dest.mkdir(parents=True, exist_ok=True)
            core.write(dest / "organization_manifest.json", variant["metadata"])
            if variant["workflow"] is not None:
                (dest / "skill.md").write_text(variant["workflow"], encoding="utf-8")
            for name in RESOURCE_NAMES:
                if name != "skill.md":
                    shutil.copyfile(directory / name, dest / name)
            for name, text in flat["shared_pool"]["progressive"]["inline_policies"].items():
                (dest / ("inline_" + name)).write_text(text, encoding="utf-8")
        core.write(output / "organization_manifest.json", {
            "protocol": self.protocol, "version": VERSION,
            "shared_resource_pool_hash": flat["shared_pool"]["manifest"]["pool_hash"],
            "arms": {a: v["metadata"] for a, v in variants.items()},
            "note": "Only published skill.md defines the display backbone. No source score reuse; all arms share native ReAct and parsed resources."})
        for arm, variant in variants.items():
            meta = variant["metadata"]
            print(f"{flow}/{arm}: available={meta['available']} chars={meta['workflow_chars']} "
                  f"randomization={meta['randomization']} error={meta['error']}", flush=True)
        return original, flat, config, provenance

    @staticmethod
    def create_agent(arm, runtime, original, flat, workflow_id=None, logger=None):
        variant = flat["organization_variants"][arm]
        if not variant["metadata"]["available"]:
            raise ValueError(variant["metadata"]["error"])
        # Exact same constructor, pool injection and native tools for EVERY arm.
        agent = core.create_agent("flat_react", runtime, original, flat, workflow_id, logger)
        agent.workflow.replace(variant["workflow"])
        agent.organization_metadata = variant["metadata"]
        return agent

    def finalize_aggregate(self, payload, output, args):
        seeds = self.seeds(args)
        results = payload["aggregate"]
        for arm, result in results.items():
            result["requested"] = arm not in args.skip_arms
        per_seed = {str(s): results[random_arm(s)] for s in seeds}
        complete = bool(seeds) and all(r["complete"] for r in per_seed.values())
        keys = ("joint_ast", "action", "slot", "testing_tokens", "testing_calls", "tokens_per_action_turn")
        random_summary = {"complete": complete, "requested": any(random_arm(s) not in args.skip_arms for s in seeds),
                          "seeds": seeds, "per_seed": per_seed,
                          "action_turns_per_seed": {s: r["action_turns"] for s, r in per_seed.items()},
                          "note": "Weight action turns within each seed, then report seed mean/sample std; do not pool repeated test targets as independent data."}
        for key in keys:
            values = [r[key] for r in per_seed.values()]
            available = complete and all(v is not None for v in values)
            random_summary[key] = statistics.mean(values) if available else None
            random_summary[key + "_std"] = statistics.stdev(values) if available and len(values) > 1 else None
        random_summary["changed_subflows_per_seed"] = {
            str(s): sum(bool((r.get("organization") or {}).get("randomization", {}).get("changed"))
                        for r in payload["rows"] if r["arm"] == random_arm(s)) for s in seeds}
        random_summary["actual_testing_tokens_all_seeds"] = (
            sum(r["testing_tokens"] for r in per_seed.values())
            if complete and all(r["testing_tokens"] is not None for r in per_seed.values()) else None)
        random_summary["actual_testing_calls_all_seeds"] = (
            sum(r["testing_calls"] for r in per_seed.values())
            if complete and all(r["testing_calls"] is not None for r in per_seed.values()) else None)
        payload["random_hierarchy_summary"] = random_summary
        payload["experiment_note"] = "All evaluated arms use controlled rendering, not historical production Ours. Shared generation cost is not newly incurred."
        core.write(output / "a3_k1_organization_aggregate.json", payload)


PROFILE = OrganizationProfile()


def main(argv=None):
    return core.main(argv, profile=PROFILE)


if __name__ == "__main__":
    raise SystemExit(main())
