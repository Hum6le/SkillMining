#!/usr/bin/env python3
"""Run hierarchical residual hypergraph mining independently per subflow.

The global miner can mix unrelated workflows when the input contains several
subflows. This wrapper uses the first component of the canonical
``role:action`` node as a conservative subflow key and mines one hierarchy per
key.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from hierarchical_skill_mining import (
    load_sessions,
    mine_sessions,
    render_reasoning_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--max-basis-nodes", type=int, default=None)
    parser.add_argument("--min-branch-support", type=int, default=2)
    parser.add_argument("--min-sessions", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = load_sessions(args.operator_results)
    groups = defaultdict(list)
    for session in sessions:
        key = session.sequence[0].split(":", 1)[0]
        groups[key].append(session)

    mined = {}
    skipped = {}
    for key, group in sorted(groups.items()):
        if len(group) < args.min_sessions:
            skipped[key] = len(group)
            continue
        mined[key] = mine_sessions(
            group,
            f"{args.operator_results}#subflow={key}",
            args.rho,
            args.max_basis_nodes,
            args.min_branch_support,
        )

    result = {
        "algorithm": "hierarchical_residual_hypergraph_cover_by_subflow",
        "input": str(args.operator_results),
        "config": {
            "coverage_ratio": args.rho,
            "max_basis_nodes": args.max_basis_nodes,
            "min_branch_support": args.min_branch_support,
            "min_sessions": args.min_sessions,
        },
        "statistics": {
            "num_sessions": len(sessions),
            "num_subflows": len(groups),
            "num_mined_subflows": len(mined),
            "num_skipped_subflows": len(skipped),
        },
        "skipped_subflows": skipped,
        "skills": mined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    prompt_dir = args.output.with_name(args.output.stem + "_reasoning_prompts")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for key, skill in mined.items():
        prompt = render_reasoning_prompt(skill["agent_evidence"])
        (prompt_dir / f"{key}.md").write_text(prompt, encoding="utf-8")

    print(json.dumps(result["statistics"], indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote reasoning prompts to {prompt_dir}")


if __name__ == "__main__":
    main()
