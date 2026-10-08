#!/usr/bin/env python3
"""Project the official SGD splits onto domain families seen in training.

The archive is streamed twice to avoid loading its 617 MB dialogue member in
memory. Each family file contains complete dialogues for causal context, plus
indices identifying its own API calls and dialogue-act turns. A multi-domain
dialogue may occur in several files; API events have a single owning family.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile


SPLITS = ("train", "validation", "test")
SERVICE_SUFFIX = re.compile(r"^(?P<family>.+)_\d+$")


def domain_family(service: str) -> str:
    match = SERVICE_SUFFIX.fullmatch(service)
    if not match:
        raise ValueError(f"SGD service has no numeric suffix: {service!r}")
    return match.group("family")


def iter_json_array(stream: io.TextIOBase, chunk_size: int = 1 << 20) -> Iterator[dict]:
    """Incrementally decode a top-level JSON array from a text stream."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    ended = False
    while True:
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position == len(buffer):
            chunk = stream.read(chunk_size)
            if not chunk:
                if ended:
                    return
                raise ValueError("Unexpected EOF inside SGD dialogue array")
            buffer, position = chunk, 0
            continue
        char = buffer[position]
        if not started:
            if char != "[":
                raise ValueError("SGD dialogues member must be a JSON array")
            started = True
            position += 1
            continue
        if char == "]":
            ended = True
            if buffer[position + 1 :].strip() or stream.read().strip():
                raise ValueError("Unexpected content after SGD dialogue array")
            return
        if char == ",":
            position += 1
            continue
        try:
            item, position = decoder.raw_decode(buffer, position)
        except json.JSONDecodeError:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError("Invalid or truncated SGD dialogue JSON") from None
            buffer = buffer[position:] + chunk
            position = 0
            continue
        if not isinstance(item, dict):
            raise ValueError("SGD dialogue array contains a non-object")
        yield item
        if position >= chunk_size:
            buffer = buffer[position:]
            position = 0


def archive_dialogues(archive: ZipFile) -> Iterator[dict]:
    with archive.open("data/dialogues.json") as binary:
        with io.TextIOWrapper(binary, encoding="utf-8") as text:
            yield from iter_json_array(text)


def family_turn_indices(dialogue: dict, family: str) -> dict[str, list[int]]:
    """Keep API and dialogue-act ownership distinct; global acts stay explicit."""
    api_turns: list[int] = []
    act_turns: list[int] = []
    global_act_turns: list[int] = []
    for turn_index, turn in enumerate(dialogue.get("turns", [])):
        if turn.get("speaker") != "system":
            continue
        calls = turn.get("service_call") or {}
        if any(domain_family(service) == family for service in calls):
            api_turns.append(turn_index)
        acts = turn.get("dialogue_acts") or {}
        values = [act for group in acts.values() for act in group]
        if any(act.get("domain") and domain_family(act["domain"]) == family for act in values):
            act_turns.append(turn_index)
        if any(not act.get("domain") for act in values):
            global_act_turns.append(turn_index)
    return {
        "api_turn_indices": api_turns,
        "domain_act_turn_indices": act_turns,
        "global_act_turn_indices": global_act_turns,
    }


def build_splits(archive_path: Path, output_dir: Path) -> dict:
    with ZipFile(archive_path) as archive:
        ontology = json.load(archive.open("data/ontology.json"))
        train_families: set[str] = set()
        source_counts: Counter[str] = Counter()
        for dialogue in archive_dialogues(archive):
            split = str(dialogue.get("data_split", ""))
            if split not in SPLITS:
                raise ValueError(f"Unknown SGD split: {split!r}")
            source_counts[split] += 1
            if split == "train":
                train_families.update(domain_family(s) for s in dialogue.get("domains", []))

        summary: dict = {
            "source_archive": str(archive_path.resolve()),
            "split_policy": "official_train_validation_test; coarse family from service suffix",
            "train_families": sorted(train_families),
            "excluded_families": {},
            "source_dialogues": dict(source_counts),
            "families": {},
            "counting_note": "Dialogue counts overlap across families. API calls belong to exactly one family; use dialogue_id and turn_index for global event aggregation.",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        counts = defaultdict(lambda: defaultdict(Counter))
        excluded = defaultdict(Counter)
        with ExitStack() as stack:
            for family in sorted(train_families):
                (output_dir / family).mkdir(parents=True, exist_ok=True)
            writers = {
                (family, split): stack.enter_context(
                    gzip.open(output_dir / family / f"{split}.jsonl.gz", "wt", encoding="utf-8")
                )
                for family in sorted(train_families)
                for split in SPLITS
            }
            for dialogue in archive_dialogues(archive):
                split = dialogue["data_split"]
                families = {domain_family(s) for s in dialogue.get("domains", [])}
                for family in families - train_families:
                    excluded[family][split] += 1
                for family in sorted(families & train_families):
                    indices = family_turn_indices(dialogue, family)
                    # Service-free acts such as GOODBYE can be attributed to
                    # the only family in a single-family dialogue. In a
                    # multi-family dialogue they have no annotated owner.
                    local_policy_turns = (
                        set(indices["api_turn_indices"])
                        | set(indices["domain_act_turn_indices"])
                    )
                    if len(families) == 1:
                        local_policy_turns.update(indices["global_act_turn_indices"])
                    indices["policy_turn_indices"] = sorted(local_policy_turns)
                    indices["unassigned_global_act_turn_indices"] = (
                        [] if len(families) == 1 else indices["global_act_turn_indices"]
                    )
                    row = {
                        "dialogue_id": dialogue["dialogue_id"],
                        "original_id": dialogue.get("original_id"),
                        "domain_family": family,
                        "services": [s for s in dialogue.get("domains", []) if domain_family(s) == family],
                        "all_services": dialogue.get("domains", []),
                        "target_indices": indices,
                        "turns": dialogue.get("turns", []),
                    }
                    writers[family, split].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    tally = counts[family][split]
                    tally["dialogues"] += 1
                    tally["api_events"] += len(indices["api_turn_indices"])
                    tally["domain_act_turns"] += len(indices["domain_act_turn_indices"])
                    tally["global_act_turns"] += len(indices["global_act_turn_indices"])
                    tally["policy_turns"] += len(indices["policy_turn_indices"])
        summary["excluded_families"] = {family: dict(by_split) for family, by_split in sorted(excluded.items())}
        summary["families"] = {
            family: {split: dict(counts[family][split]) for split in SPLITS}
            for family in sorted(train_families)
        }
        summary["ontology_services"] = len(ontology["domains"])
        (output_dir / "INDEX.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/eval/sgd/data.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval/sgd/splits_by_domain"))
    args = parser.parse_args()
    result = build_splits(args.archive, args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "source_dialogues": result["source_dialogues"],
        "train_families": result["train_families"],
        "excluded_families": result["excluded_families"],
        "api_events_by_split": {
            split: sum(row[split].get("api_events", 0) for row in result["families"].values())
            for split in SPLITS
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
