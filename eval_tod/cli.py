"""Command-line entry point for ToD evaluation.

This module now exposes a small subcommand-based CLI:

- ``tod``: file-based MultiWOZ slot evaluation
- ``text``: generic text generation evaluation
- ``abcd``: ABCD text + AST/CDS evaluation

For backward compatibility, the old root-level invocation still works:

    python -m eval_tod.cli --dataset multiwoz21 --data_path ... --predictions ...

which is treated as ``python -m eval_tod.cli tod ...``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .evaluate import evaluate, print_summary
from .text_eval import evaluate_responses, load_predictions as load_text_predictions


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_legacy_argv(argv: list[str]) -> list[str]:
    """Preserve the old root-level CLI by mapping it to ``tod``."""
    if not argv:
        return ["tod"]

    if argv[0] in {"tod", "text", "abcd", "-h", "--help"}:
        return argv

    return ["tod", *argv]


def _add_common_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output", default=None,
        help="Path to write results JSON (default: stdout summary only)",
    )


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm_judge", action="store_true",
        help="Enable multi-agent LLM-as-a-Judge evaluation (5 specialist judges + 1 combiner)",
    )
    parser.add_argument(
        "--llm_dimensions", nargs="+", default=None,
        metavar="DIM",
        help="Dimensions for LLM judge (default: task_completion slot_accuracy dialogue_fluency helpfulness efficiency)",
    )
    parser.add_argument(
        "--llm_sample_size", type=int, default=None,
        help="Sample N dialogues for LLM judge (cost control)",
    )
    parser.add_argument(
        "--llm_model", default="deepseek-chat",
        help="LLM model for judge (default: deepseek-chat)",
    )


def _normalize_abcd_id(value: str) -> str:
    """Normalize dialogue ids from saved ABCD artifacts."""
    value = str(value or "")
    if value.startswith("abcd-"):
        return value[len("abcd-"):]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ToD agent predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m eval_tod tod --dataset multiwoz21 --data_path data/eval/multiwoz21 --predictions preds.json
  python -m eval_tod text --predictions preds.json --references refs.json
  python -m eval_tod abcd --data_path data/eval/abcd/data --split test --text-predictions text_preds.json --abcd-predictions abcd_preds.json

Backward-compatible legacy form:
  python -m eval_tod.cli --dataset multiwoz21 --data_path data/eval/multiwoz21 --predictions preds.json
        """.strip(),
    )

    subparsers = parser.add_subparsers(dest="command")

    # Legacy ToD evaluation
    tod = subparsers.add_parser(
        "tod",
        help="Evaluate MultiWOZ-style dialogue state predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    tod.add_argument(
        "--dataset", required=True,
        choices=["multiwoz21"],
        help="Dataset name. Currently: multiwoz21",
    )
    tod.add_argument(
        "--data_path", required=True,
        help="Path to dataset directory (e.g. data/eval/multiwoz21)",
    )
    tod.add_argument(
        "--predictions", required=True,
        help="Path to predictions JSON file",
    )
    tod.add_argument(
        "--split", default=None,
        choices=["train", "validation", "test"],
        help="Data split to evaluate (default: all)",
    )
    _add_output_args = _add_common_output_args
    _add_output_args(tod)
    _add_llm_args(tod)

    # Generic text generation evaluation
    text = subparsers.add_parser(
        "text",
        help="Evaluate generated responses with BERTScore/BLEU/ROUGE/METEOR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    text.add_argument(
        "--predictions", required=True,
        help="Path to predictions JSON/JSONL/TXT file",
    )
    text.add_argument(
        "--references", required=True,
        help="Path to reference JSON/JSONL/TXT file",
    )
    text.add_argument(
        "--prediction_key", default="response_text",
        help="JSON key to extract from prediction objects (default: response_text)",
    )
    _add_common_output_args(text)
    text.add_argument(
        "--bert_model", default="",
        help="Optional BERTScore model id",
    )
    text.add_argument(
        "--batch_size", type=int, default=32,
        help="BERTScore batch size",
    )
    text.add_argument(
        "--use_idf", action="store_true",
        help="Enable IDF weighting for BERTScore",
    )

    # ABCD evaluation
    abcd = subparsers.add_parser(
        "abcd",
        help="Evaluate ABCD response text and/or AST/CDS outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    abcd.add_argument(
        "--data_path", required=True,
        help="Path to ABCD data directory (contains abcd_v1.1.json)",
    )
    abcd.add_argument(
        "--split", default="test",
        choices=["train", "dev", "test"],
        help="ABCD split to evaluate",
    )
    abcd.add_argument(
        "--text_predictions", default=None,
        help="Optional path to dialogue-level response predictions",
    )
    abcd.add_argument(
        "--abcd_predictions", default=None,
        help="Optional path to turn-level ABCD action predictions",
    )
    abcd.add_argument(
        "--text_prediction_key", default="response_text",
        help="JSON key to extract from text prediction objects",
    )
    _add_common_output_args(abcd)
    abcd.add_argument(
        "--bert_model", default="",
        help="Optional BERTScore model id",
    )
    abcd.add_argument(
        "--batch_size", type=int, default=32,
        help="BERTScore batch size",
    )
    abcd.add_argument(
        "--use_idf", action="store_true",
        help="Enable IDF weighting for BERTScore",
    )

    return parser


def _load_text_eval_inputs(
    pred_path: str,
    ref_path: str,
    prediction_key: str,
) -> tuple[list[str], list[str]]:
    predictions = load_text_predictions(pred_path, key=prediction_key)
    references = load_text_predictions(ref_path, key="response")
    return predictions, references


def _load_abcd_dialogue_texts(
    conversations: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    prediction_key: str = "response_text",
) -> list[str]:
    """Align dialogue-level text predictions with conversation order."""
    lookup: dict[str, str] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        key = _normalize_abcd_id(
            item.get("dialogue_id")
            or item.get("conversation_id")
            or item.get("convo_id")
            or ""
        )
        if not key:
            continue
        lookup[key] = str(item.get(prediction_key, ""))

    texts: list[str] = []
    for conv in conversations:
        cid = _normalize_abcd_id(conv.get("convo_id", ""))
        texts.append(lookup.get(cid, ""))
    return texts


def _parse_abcd_prediction_records(records: list[dict[str, Any]]) -> list[Any]:
    from .abcd.schemas import ABCDPrediction, ABCDTurnPrediction

    preds: list[ABCDPrediction] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        if "turns" not in item and (
            "predicted_action" in item
            or "predicted_slots" in item
            or "turn_index" in item
        ):
            raise ValueError(
                "ABCD action records must be grouped by conversation with a "
                "'turns' list. Received flat turn records; call "
                "turn_results_to_abcd_predictions() first."
            )
        turns = []
        for turn in item.get("turns", []):
            if not isinstance(turn, dict):
                continue
            turns.append(ABCDTurnPrediction(
                turn_index=int(turn.get("turn_index", 0)),
                turn_type=str(turn.get("turn_type", "action")),
                predicted_utterance_id=turn.get("predicted_utterance_id"),
                predicted_action=turn.get("predicted_action"),
                predicted_slots=turn.get("predicted_slots"),
            ))
        preds.append(ABCDPrediction(
            conversation_id=_normalize_abcd_id(
                item.get("conversation_id")
                or item.get("dialogue_id")
                or item.get("convo_id")
                or ""
            ),
            turns=turns,
        ))
    return preds


def _load_abcd_action_predictions(path: str) -> list[Any]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError("ABCD predictions must be a JSON array")
    return _parse_abcd_prediction_records(data)


def align_abcd_predictions(
    conversations: list[dict[str, Any]],
    predictions: list[Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Align ABCD predictions to conversation order by ``convo_id``."""
    from .abcd.schemas import ABCDPrediction

    lookup = {
        _normalize_abcd_id(getattr(pred, "conversation_id", "")): pred
        for pred in predictions
        if _normalize_abcd_id(getattr(pred, "conversation_id", ""))
    }
    use_id_alignment = bool(lookup)

    aligned: list[ABCDPrediction] = []
    missing: list[str] = []
    if use_id_alignment:
        for conv in conversations:
            cid = _normalize_abcd_id(conv.get("convo_id", ""))
            pred = lookup.get(cid)
            if pred is None:
                missing.append(cid)
                pred = ABCDPrediction(conversation_id=cid, turns=[])
            aligned.append(pred)
    else:
        for idx, conv in enumerate(conversations):
            cid = _normalize_abcd_id(conv.get("convo_id", ""))
            if idx < len(predictions):
                aligned.append(predictions[idx])
            else:
                missing.append(cid)
                aligned.append(ABCDPrediction(conversation_id=cid, turns=[]))

    return aligned, {
        "mode": "id" if use_id_alignment else "index",
        "num_conversations": len(conversations),
        "num_predictions": len(predictions),
        "num_missing": len(missing),
        "missing_conversation_ids": missing[:50],
    }


def _load_abcd_ground_truths(conversations: list[dict[str, Any]]) -> list[list[Any]]:
    from .abcd.data import extract_ground_truth

    return [extract_ground_truth(conv) for conv in conversations]


def evaluate_text_records(
    predictions: list[str],
    references: list[str],
    *,
    bert_model: str = "",
    batch_size: int = 32,
    use_idf: bool = False,
) -> dict[str, Any]:
    """Evaluate raw text predictions against raw references."""
    result = evaluate_responses(
        predictions,
        references,
        bert_model=bert_model,
        batch_size=batch_size,
        use_idf=use_idf,
    )
    return {
        "bert_f1": result.bert_f1,
        "bert_precision": result.bert_precision,
        "bert_recall": result.bert_recall,
        "bleu_1": result.bleu_1,
        "bleu_4": result.bleu_4,
        "rouge_1": result.rouge_1,
        "rouge_2": result.rouge_2,
        "rouge_l": result.rouge_l,
        "meteor": result.meteor,
        "num_samples": result.num_samples,
        "per_sample": result.per_sample,
        "summary": result.summary(),
    }


def evaluate_abcd_bundle(
    conversations: list[dict[str, Any]],
    *,
    text_records: list[dict[str, Any]] | None = None,
    abcd_records: list[dict[str, Any]] | None = None,
    text_prediction_key: str = "response_text",
    bert_model: str = "",
    batch_size: int = 32,
    use_idf: bool = False,
) -> dict[str, Any]:
    """Evaluate ABCD text and/or AST-CDS artifacts from in-memory records."""
    from .abcd.metrics import evaluate_abcd
    from .text_eval import evaluate_responses as eval_text_responses

    payload: dict[str, Any] = {"dataset": "abcd"}

    if text_records is not None:
        preds = _load_abcd_dialogue_texts(
            conversations,
            text_records,
            prediction_key=text_prediction_key,
        )
        from .abcd.data import last_agent_response_texts

        refs = last_agent_response_texts(conversations)
        text_result = eval_text_responses(
            preds,
            refs,
            bert_model=bert_model,
            batch_size=batch_size,
            use_idf=use_idf,
        )
        payload["text"] = {
            "bert_f1": text_result.bert_f1,
            "bert_precision": text_result.bert_precision,
            "bert_recall": text_result.bert_recall,
            "bleu_1": text_result.bleu_1,
            "bleu_4": text_result.bleu_4,
            "rouge_1": text_result.rouge_1,
            "rouge_2": text_result.rouge_2,
            "rouge_l": text_result.rouge_l,
            "meteor": text_result.meteor,
            "num_samples": text_result.num_samples,
            "per_sample": text_result.per_sample,
        }

    if abcd_records is not None:
        abcd_preds = _parse_abcd_prediction_records(abcd_records)
        abcd_preds, alignment = align_abcd_predictions(conversations, abcd_preds)
        all_gt = _load_abcd_ground_truths(conversations)
        abcd_result = evaluate_abcd(all_gt, abcd_preds)
        payload["ast_cds"] = {
            "ast_joint": abcd_result.ast.joint_accuracy,
            "ast_action_name": abcd_result.ast.action_name_accuracy,
            "ast_slot_value": abcd_result.ast.slot_value_accuracy,
            "cds_overall": abcd_result.cds.overall_cds,
            "num_action_turns": abcd_result.ast.total_action_turns,
        }
        payload["abcd_alignment"] = alignment

    parts = [f"eval(abcd, N={len(conversations)})"]
    if "text" in payload:
        parts.append(
            f"BERT-F1={payload['text']['bert_f1']:.4f} "
            f"BLEU-4={payload['text']['bleu_4']:.1f} "
            f"ROUGE-L={payload['text']['rouge_l']:.4f} "
            f"METEOR={payload['text']['meteor']:.4f}"
        )
    if "ast_cds" in payload:
        parts.append(
            f"AST={payload['ast_cds']['ast_joint']:.4f} "
            f"CDS={payload['ast_cds']['cds_overall']:.4f}"
        )
    payload["summary"] = "  ".join(parts)
    return payload


def _evaluate_tod(args: argparse.Namespace) -> dict[str, Any]:
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"data_path does not exist: {args.data_path}")
    if not Path(args.predictions).exists():
        raise FileNotFoundError(f"predictions file does not exist: {args.predictions}")

    result = evaluate(
        dataset_name=args.dataset,
        data_path=args.data_path,
        predictions_path=args.predictions,
        split=args.split,
        llm_judge=args.llm_judge,
        llm_judge_dimensions=args.llm_dimensions,
        llm_judge_sample_size=args.llm_sample_size,
        llm_model=args.llm_model,
        output_path=args.output,
    )
    print_summary(result)
    return result


def _evaluate_text(args: argparse.Namespace) -> dict[str, Any]:
    if not Path(args.predictions).exists():
        raise FileNotFoundError(f"predictions file does not exist: {args.predictions}")
    if not Path(args.references).exists():
        raise FileNotFoundError(f"references file does not exist: {args.references}")

    preds, refs = _load_text_eval_inputs(
        args.predictions,
        args.references,
        args.prediction_key,
    )
    payload = {
        "text": evaluate_text_records(
            preds,
            refs,
            bert_model=args.bert_model,
            batch_size=args.batch_size,
            use_idf=args.use_idf,
        ),
    }

    print(payload["text"]["summary"])
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    return payload


def _evaluate_abcd(args: argparse.Namespace) -> dict[str, Any]:
    from .abcd.data import load_abcd_data

    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"data_path does not exist: {args.data_path}")

    conversations = load_abcd_data(args.split, args.data_path)
    payload: dict[str, Any] = {"dataset": "abcd", "split": args.split}

    text_records = _read_json(args.text_predictions) if args.text_predictions else None
    abcd_records = _read_json(args.abcd_predictions) if args.abcd_predictions else None
    if text_records is not None and not isinstance(text_records, list):
        raise ValueError("text_predictions must be a JSON array")
    if abcd_records is not None and not isinstance(abcd_records, list):
        raise ValueError("abcd_predictions must be a JSON array")

    payload = evaluate_abcd_bundle(
        conversations,
        text_records=text_records,
        abcd_records=abcd_records,
        text_prediction_key=args.text_prediction_key,
        bert_model=args.bert_model,
        batch_size=args.batch_size,
        use_idf=args.use_idf,
    )
    payload["split"] = args.split

    print(payload["summary"])
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    argv = _resolve_legacy_argv(list(argv) if argv is not None else sys.argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "text":
        _evaluate_text(args)
        return 0
    if args.command == "abcd":
        _evaluate_abcd(args)
        return 0

    # Default to ToD evaluation for the legacy form and explicit ``tod``.
    _evaluate_tod(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
