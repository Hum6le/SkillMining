#!/usr/bin/env python3
"""Create a reproducible dialogue-level SGD pilot from the family projections.

The same sampled directory can be passed to both SGD runners via --splits-dir.
Sampling is independent within each domain family and official split, so each
family contributes approximately the same fraction of its dialogues. All
policy turns of a selected dialogue stay together.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


SPLITS = ("train", "validation", "test")
COUNT_KEYS = ("dialogues", "api_events", "domain_act_turns", "global_act_turns", "policy_turns")


def rank(seed: int, split: str, dialogue_id: str) -> bytes:
    # The rank is shared across families; domain-specific quotas remain exact.
    return hashlib.sha256(f"{seed}\0{split}\0{dialogue_id}".encode("utf-8")).digest()


def sample_file(source: Path, target: Path, *, fraction: float, seed: int,
                split: str) -> tuple[dict, set[str]]:
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        ids = [json.loads(line)["dialogue_id"] for line in stream if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate dialogue_id in {source}")
    quota = min(len(ids), max(1, math.ceil(len(ids) * fraction))) if ids else 0
    selected = set(sorted(ids, key=lambda item: (rank(seed, split, item), item))[:quota])
    counts: Counter[str] = Counter()
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rt", encoding="utf-8") as input_stream:
        with gzip.open(target, "wt", encoding="utf-8") as output_stream:
            for line in input_stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["dialogue_id"] not in selected:
                    continue
                indices = row["target_indices"]
                counts["dialogues"] += 1
                counts["api_events"] += len(indices["api_turn_indices"])
                counts["domain_act_turns"] += len(indices["domain_act_turn_indices"])
                counts["global_act_turns"] += len(indices["global_act_turn_indices"])
                counts["policy_turns"] += len(indices["policy_turn_indices"])
                output_stream.write(line)
    if counts["dialogues"] != quota:
        raise AssertionError(f"Selected {quota} IDs but wrote {counts['dialogues']} rows for {source}")
    return {key: counts[key] for key in COUNT_KEYS}, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/eval/sgd/splits_by_domain"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/sgd/splits_10pct_seed42"))
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1 or not math.isfinite(args.fraction):
        parser.error("--fraction must be finite and in (0, 1]")
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("Source and output directories must be separate")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is nonempty: {output}")
    original = json.loads((source / "INDEX.json").read_text(encoding="utf-8"))
    source_index_hash = hashlib.sha256((source / "INDEX.json").read_bytes()).hexdigest()
    families = original["train_families"]
    index = {key: original[key] for key in ("source_archive", "split_policy", "train_families",
                                             "excluded_families", "ontology_services")}
    index["split_policy"] += f"; family_stratified_dialogue_sample_fraction={args.fraction}"
    index["sample"] = {"seed": args.seed, "fraction": args.fraction,
                       "ranking": "sha256(seed\\0split\\0dialogue_id)",
                       "quota": "ceil(fraction * family_split_dialogues)",
                       "source_index_sha256": source_index_hash,
                       "source_dir": str(source)}
    index["counting_note"] = original["counting_note"]
    index["families"] = {}
    unique_ids: dict[str, set[str]] = {split: set() for split in SPLITS}
    for family in families:
        index["families"][family] = {}
        for split in SPLITS:
            counts, selected = sample_file(
                source / family / f"{split}.jsonl.gz",
                output / family / f"{split}.jsonl.gz",
                fraction=args.fraction, seed=args.seed, split=split,
            )
            index["families"][family][split] = counts
            unique_ids[split].update(selected)
    index["source_dialogues"] = {split: len(unique_ids[split]) for split in SPLITS}
    (output / "INDEX.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output), "sample": index["sample"],
        "unique_dialogues": index["source_dialogues"],
        "family_projections": {
            split: {key: sum(index["families"][family][split][key] for family in families)
                    for key in ("dialogues", "policy_turns", "api_events")}
            for split in SPLITS
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
