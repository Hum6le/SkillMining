#!/usr/bin/env python3
"""Split multi-turn dialogues into (context, response) training pairs.

Supports MultiWOZ and ABCD datasets out of the box.  Each output sample
contains the conversation history (context) and the target utterance
(response) for a single turn.

Usage::

    # MultiWOZ — train split, chatml format, system turns only
    python scripts/make_training_data.py --dataset multiwoz --split train

    # ABCD — test split, alpaca format
    python scripts/make_training_data.py --dataset abcd --split test \
        --format alpaca

    # Custom — any JSON file with a "turns" field
    python scripts/make_training_data.py --dataset custom \
        --custom_path my_data.json --output data/my_pairs.jsonl

Output formats::

    chatml:  {"messages": [{"role":"system","content":"..."},
                            {"role":"user","content":"..."},
                            {"role":"assistant","content":"..."}]}
    alpaca:  {"instruction": "...", "input": "...", "output": "..."}
    raw:     {"context": "preceding turns text", "response": "target text",
              "dialogue_id": "...", "turn_index": N}
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Defaults ────────────────────────────────────────────────────
_MULTIWOZ_DIR = "data/eval/multiwoz21/splits"
_ABCD_DIR = "data/eval/abcd/data"
_DEFAULT_OUTPUT = "data/training_pairs.jsonl"


# ══════════════════════════════════════════════════════════════════
# Turn iterators
# ══════════════════════════════════════════════════════════════════

def _iter_multiwoz_turns(
    split: str,
    data_dir: str,
    include_system: bool,
    include_user: bool,
    max_context_turns: int,
) -> list[dict]:
    """Yield (context, response) pairs from MultiWOZ dialogues."""
    from eval_tod.data import load_multiwoz21

    split_file = {"train": "all_train.json", "val": "all_val.json", "test": "all_test.json"}[split]
    dialogues = load_multiwoz21(os.path.join(data_dir, split_file))

    pairs: list[dict] = []
    for dialogue in dialogues:
        history: list[str] = []
        for i, turn in enumerate(dialogue.turns):
            speaker_label = "USER" if turn.speaker == "user" else "SYSTEM"
            utt = turn.utterance.strip()
            if not utt:
                continue

            # Determine if this turn should be a response target
            is_target = (
                (include_system and turn.speaker == "system")
                or (include_user and turn.speaker == "user")
            )

            if is_target and history:
                # Use recent N turns as context
                ctx = history[-max_context_turns:] if max_context_turns > 0 else history
                pairs.append({
                    "context": "\n".join(ctx),
                    "response": utt,
                    "dialogue_id": dialogue.dialogue_id,
                    "turn_index": i,
                })

            history.append(f"[{speaker_label}] {utt}")

    return pairs


def _iter_abcd_turns(
    split: str,
    data_dir: str,
    include_agent: bool,
    include_customer: bool,
    max_context_turns: int,
) -> list[dict]:
    """Yield (context, response) pairs from ABCD dialogues."""
    from eval_tod.abcd.data import load_abcd_data

    conversations = load_abcd_data(split, data_dir)

    pairs: list[dict] = []
    for conv in conversations:
        convo_id = str(conv["convo_id"])
        history: list[str] = []
        for i, turn in enumerate(conv["delexed"]):
            speaker = turn["speaker"]
            text = turn.get("text", "").strip()
            if not text:
                continue

            speaker_label = speaker  # "agent", "customer", "action"

            is_target = (
                (include_agent and speaker == "agent")
                or (include_customer and speaker == "customer")
                or (include_agent and speaker == "action")  # action = agent too
            )

            if is_target and history:
                ctx = history[-max_context_turns:] if max_context_turns > 0 else history
                pairs.append({
                    "context": "\n".join(ctx),
                    "response": text,
                    "dialogue_id": convo_id,
                    "turn_index": i,
                })

            history.append(f"[{speaker_label}] {text}")

    return pairs


# ══════════════════════════════════════════════════════════════════
# Formatters
# ══════════════════════════════════════════════════════════════════

def _format_chatml(pairs: list[dict], system_prompt: str) -> list[dict]:
    """Convert pairs to ChatML format."""
    samples = []
    for p in pairs:
        messages = [{"role": "system", "content": system_prompt}]
        # Split context back into user/assistant turns
        for line in p["context"].split("\n"):
            if line.startswith("[USER]") or line.startswith("[user]") or line.startswith("[customer]"):
                messages.append({"role": "user", "content": line[line.index("]") + 1:].strip()})
            elif line.startswith("[SYSTEM]") or line.startswith("[system]") or line.startswith("[agent]") or line.startswith("[action]"):
                messages.append({"role": "assistant", "content": line[line.index("]") + 1:].strip()})
        messages.append({"role": "assistant", "content": p["response"]})
        samples.append({"messages": messages})
    return samples


def _format_alpaca(pairs: list[dict], system_prompt: str) -> list[dict]:
    """Convert pairs to Alpaca format."""
    samples = []
    for p in pairs:
        samples.append({
            "instruction": system_prompt,
            "input": p["context"],
            "output": p["response"],
        })
    return samples


def _format_raw(pairs: list[dict]) -> list[dict]:
    """Pass through raw pairs with metadata."""
    return pairs


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Split multi-turn dialogues into (context, response) training pairs"
    )
    parser.add_argument("--dataset", default="multiwoz",
                        choices=["multiwoz", "abcd", "custom"],
                        help="Dataset name (default: multiwoz)")
    parser.add_argument("--split", default="train",
                        help="Data split (default: train)")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: data/training_pairs.jsonl)")
    parser.add_argument("--format", default="chatml",
                        choices=["chatml", "alpaca", "raw"],
                        help="Output format (default: chatml)")
    parser.add_argument("--max_context_turns", type=int, default=10,
                        help="Max preceding turns to include as context (0=unlimited)")
    parser.add_argument("--include_system", default=True,
                        action="store_true",
                        help="Include system/agent turns as response targets (default: True)")
    parser.add_argument("--no_system", default=False,
                        action="store_true", dest="no_system",
                        help="Exclude system/agent turns")
    parser.add_argument("--include_user", default=False,
                        action="store_true",
                        help="Include user/customer turns as response targets")
    parser.add_argument("--data_dir", default=None,
                        help="Override default data directory")
    parser.add_argument("--custom_path", default=None,
                        help="Path to custom JSON file (for --dataset custom)")
    parser.add_argument("--max_dialogues", type=int, default=None,
                        help="Limit number of dialogues (for quick testing)")
    parser.add_argument("--system_prompt", default="You are a helpful customer service assistant.",
                        help="System prompt for chatml/alpaca formats")
    args = parser.parse_args()

    # ── Load turns ──────────────────────────────────────────────
    if args.dataset == "multiwoz":
        data_dir = args.data_dir or _MULTIWOZ_DIR
        pairs = _iter_multiwoz_turns(
            split=args.split,
            data_dir=data_dir,
            include_system=not args.no_system,
            include_user=args.include_user,
            max_context_turns=args.max_context_turns,
        )
    elif args.dataset == "abcd":
        data_dir = args.data_dir or _ABCD_DIR
        pairs = _iter_abcd_turns(
            split=args.split,
            data_dir=data_dir,
            include_agent=not args.no_system,
            include_customer=args.include_user,
            max_context_turns=args.max_context_turns,
        )
    elif args.dataset == "custom":
        if not args.custom_path:
            print("Error: --custom_path required for --dataset custom", file=sys.stderr)
            return 1
        raw = json.loads(Path(args.custom_path).read_text(encoding="utf-8"))
        # Expect a list of {"turns": [{"speaker":..., "text":...}, ...]} or
        # {"dialogues": [...]} or similar.  For now, basic list[str] handling.
        pairs = []
        for item in raw:
            turns = item.get("turns", item.get("utterances", []))
            history: list[str] = []
            for i, t in enumerate(turns):
                spk = t.get("speaker", t.get("role", "unknown"))
                txt = t.get("text", t.get("content", str(t))).strip()
                if not txt:
                    continue
                if history:
                    pairs.append({
                        "context": "\n".join(
                            history[-args.max_context_turns:]
                            if args.max_context_turns > 0 else history
                        ),
                        "response": txt,
                        "dialogue_id": item.get("id", str(id(item))),
                        "turn_index": i,
                    })
                history.append(f"[{spk}] {txt}")
    else:
        print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
        return 1

    if args.max_dialogues:
        # Rough limit: take first N unique dialogue_ids
        seen = set()
        limited = []
        for p in pairs:
            did = p["dialogue_id"]
            if did not in seen:
                seen.add(did)
                if len(seen) > args.max_dialogues:
                    break
            limited.append(p)
        pairs = limited

    print(f"Generated {len(pairs)} (context, response) pairs")

    # ── Format ──────────────────────────────────────────────────
    if args.format == "chatml":
        samples = _format_chatml(pairs, args.system_prompt)
    elif args.format == "alpaca":
        samples = _format_alpaca(pairs, args.system_prompt)
    else:
        samples = _format_raw(pairs)

    # ── Save ────────────────────────────────────────────────────
    output = args.output or _DEFAULT_OUTPUT
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} samples to: {output}")
    print(f"  Format: {args.format}")
    print(f"  Dataset: {args.dataset} / {args.split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
