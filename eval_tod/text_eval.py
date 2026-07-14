"""Generic text generation evaluation — dataset-agnostic.

Compares model-generated responses against ground-truth references using
three families of metrics:

- **BERTScore**: semantic similarity via contextual embeddings (precision /
  recall / F1).  Uses ``microsoft/deberta-xlarge-mnli`` by default.
- **BLEU**: n-gram precision (BLEU-1 through BLEU-4).  Computed via
  ``sacrebleu`` with default tokenisation.
- **ROUGE**: recall-oriented overlap (ROUGE-1, ROUGE-2, ROUGE-L).
  Computed via Google's ``rouge-score``.
- **METEOR**: exact-token METEOR-style score with harmonic precision/recall
  weighting and fragmentation penalty.

Usage::

    from eval_tod.text_eval import evaluate_responses

    result = evaluate_responses(
        predictions=["I found a cheap hotel.", ...],
        references=["I found a cheap hotel.", ...],
    )
    print(f"BERT F1: {result.bert_f1:.4f}")
    print(f"BLEU-4:  {result.bleu_4:.1f}")
    print(f"ROUGE-L: {result.rouge_l:.4f}")
    print(f"METEOR:  {result.meteor:.4f}")

Also provides file I/O helpers for loading predictions from disk.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import sacrebleu
from rouge_score import rouge_scorer

# ── BERTScore is imported lazily (heavy model load) ────────────
_bert_score = None


def _get_bert_score():
    global _bert_score
    if _bert_score is None:
        import bert_score
        _bert_score = bert_score
    return _bert_score


# ══════════════════════════════════════════════════════════════════
# Result types
# ══════════════════════════════════════════════════════════════════

@dataclass
class TextEvalResult:
    """Aggregate + per-sample text generation metrics.

    Attributes:
        bert_f1: BERTScore F1 (contextual semantic similarity).  Range ~[0, 1].
        bert_precision: BERTScore precision.
        bert_recall: BERTScore recall.
        bleu_1: BLEU-1 (unigram precision × 100).  Range [0, 100].
        bleu_4: BLEU-4 (4-gram precision × 100).  Range [0, 100].
        rouge_1: ROUGE-1 F1 (unigram overlap).  Range ~[0, 1].
        rouge_2: ROUGE-2 F1 (bigram overlap).  Range ~[0, 1].
        rouge_l: ROUGE-L F1 (longest common subsequence).  Range ~[0, 1].
        meteor: Exact-token METEOR-style score.  Range ~[0, 1].
        num_samples: Number of prediction-reference pairs evaluated.
        per_sample: List of per-sample dicts with the same keys.
    """

    bert_f1: float = 0.0
    bert_precision: float = 0.0
    bert_recall: float = 0.0
    bleu_1: float = 0.0
    bleu_4: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    meteor: float = 0.0
    num_samples: int = 0
    per_sample: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable one-line summary of main metrics."""
        return (
            f"BERT-F1={self.bert_f1:.4f}  "
            f"BLEU-1={self.bleu_1:.1f}  BLEU-4={self.bleu_4:.1f}  "
            f"ROUGE-1={self.rouge_1:.4f}  ROUGE-2={self.rouge_2:.4f}  "
            f"ROUGE-L={self.rouge_l:.4f}  METEOR={self.meteor:.4f}  "
            f"(N={self.num_samples})"
        )

    def full_summary(self) -> str:
        """Multi-line detailed summary."""
        return (
            f"Text Generation Evaluation  (N={self.num_samples})\n"
            f"{'─' * 50}\n"
            f"  BERTScore  precision={self.bert_precision:.4f}  "
            f"recall={self.bert_recall:.4f}  F1={self.bert_f1:.4f}\n"
            f"  BLEU       BLEU-1={self.bleu_1:.1f}  BLEU-4={self.bleu_4:.1f}\n"
            f"  ROUGE      R1={self.rouge_1:.4f}  R2={self.rouge_2:.4f}  "
            f"RL={self.rouge_l:.4f}\n"
            f"  METEOR     {self.meteor:.4f}\n"
            f"{'─' * 50}"
        )


# ══════════════════════════════════════════════════════════════════
# Core evaluation
# ══════════════════════════════════════════════════════════════════

_DEFAULT_BERT_MODEL = ""  # empty = use bert-score's lang='en' default (roberta-large)


def _meteor_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def _exact_meteor_score(prediction: str, reference: str) -> float:
    """Compute a METEOR-style score using exact token matches.

    This keeps METEOR's precision/recall weighting and fragmentation penalty,
    while avoiding runtime WordNet or paraphrase-table downloads.
    """
    pred_tokens = _meteor_tokens(prediction)
    ref_tokens = _meteor_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    if pred_tokens == ref_tokens:
        return 1.0

    ref_remaining = Counter(ref_tokens)
    used_ref_indices: set[int] = set()
    matches: list[tuple[int, int]] = []
    for pred_idx, token in enumerate(pred_tokens):
        if ref_remaining[token] <= 0:
            continue
        for ref_idx, ref_token in enumerate(ref_tokens):
            if ref_idx in used_ref_indices or ref_token != token:
                continue
            matches.append((pred_idx, ref_idx))
            used_ref_indices.add(ref_idx)
            ref_remaining[token] -= 1
            break

    match_count = len(matches)
    if match_count == 0:
        return 0.0

    precision = match_count / len(pred_tokens)
    recall = match_count / len(ref_tokens)
    f_mean = (10.0 * precision * recall) / (recall + 9.0 * precision)

    matches.sort()
    chunks = 1
    for (prev_pred, prev_ref), (cur_pred, cur_ref) in zip(matches, matches[1:]):
        if cur_pred != prev_pred + 1 or cur_ref != prev_ref + 1:
            chunks += 1
    penalty = 0.5 * (chunks / match_count) ** 3
    return f_mean * (1.0 - penalty)


def evaluate_responses(
    predictions: list[str],
    references: list[str],
    *,
    bert_model: str = "",
    batch_size: int = 32,
    use_idf: bool = False,
) -> TextEvalResult:
    """Evaluate generated responses against ground-truth references.

    Args:
        predictions: Model-generated response texts (one per sample).
        references: Ground-truth response texts (one per sample, same order).
        bert_model: HuggingFace model id for BERTScore.  Default:
            ``"microsoft/deberta-xlarge-mnli"``.  Use
            ``"microsoft/deberta-base-mnli"`` for a lighter / faster option.
        batch_size: Batch size for BERTScore computation.
        use_idf: If True, weight BERTScore by inverse document frequency
            (requires the predictions + references to contain multiple
            sentences each — rarely needed for single-response eval).

    Returns:
        ``TextEvalResult`` with aggregate and per-sample scores.
    """
    n = len(predictions)
    if n == 0:
        return TextEvalResult(num_samples=0)
    if n != len(references):
        raise ValueError(
            f"Length mismatch: {n} predictions vs {len(references)} references"
        )

    # ── BERTScore ───────────────────────────────────────────────
    bs = _get_bert_score()
    score_kwargs: dict[str, Any] = dict(
        lang="en",
        batch_size=batch_size,
        idf=use_idf,
        verbose=False,
    )
    # Only pass model_type if explicitly set (overrides lang default)
    if bert_model:
        score_kwargs["model_type"] = bert_model
    P, R, F1 = bs.score(predictions, references, **score_kwargs)
    bert_p = float(P.mean().item())
    bert_r = float(R.mean().item())
    bert_f = float(F1.mean().item())
    bert_f_per = F1.tolist()

    # ── BLEU ─────────────────────────────────────────────────────
    # sacrebleu expects list[str] preds and list[list[str]] refs
    bleu = sacrebleu.corpus_bleu(predictions, [[r] for r in references])
    bleu_1_val = bleu.precisions[0] if len(bleu.precisions) >= 1 else 0.0  # already 0-100
    bleu_4_val = bleu.score  # sacrebleu's .score is BLEU-4 by default

    # ── ROUGE ────────────────────────────────────────────────────
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    rouge1_scores: list[float] = []
    rouge2_scores: list[float] = []
    rougeL_scores: list[float] = []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        rouge1_scores.append(s["rouge1"].fmeasure)
        rouge2_scores.append(s["rouge2"].fmeasure)
        rougeL_scores.append(s["rougeL"].fmeasure)

    rouge1_avg = sum(rouge1_scores) / n
    rouge2_avg = sum(rouge2_scores) / n
    rougeL_avg = sum(rougeL_scores) / n
    meteor_scores = [
        _exact_meteor_score(pred, ref)
        for pred, ref in zip(predictions, references)
    ]
    meteor_avg = sum(meteor_scores) / n

    # ── Per-sample ───────────────────────────────────────────────
    per_sample: list[dict[str, float]] = []
    for i in range(n):
        per_sample.append({
            "bert_f1": bert_f_per[i],
            "rouge1": rouge1_scores[i],
            "rouge2": rouge2_scores[i],
            "rougeL": rougeL_scores[i],
            "meteor": meteor_scores[i],
        })

    return TextEvalResult(
        bert_f1=bert_f,
        bert_precision=bert_p,
        bert_recall=bert_r,
        bleu_1=bleu_1_val,
        bleu_4=bleu_4_val,
        rouge_1=rouge1_avg,
        rouge_2=rouge2_avg,
        rouge_l=rougeL_avg,
        meteor=meteor_avg,
        num_samples=n,
        per_sample=per_sample,
    )


# ══════════════════════════════════════════════════════════════════
# File I/O helpers
# ══════════════════════════════════════════════════════════════════

def load_predictions(path: str, key: str = "response") -> list[str]:
    """Load model-generated responses from a file.

    Supported formats (auto-detected by extension / content):

    - **JSON**: ``.json`` — list of dicts, e.g.
      ``[{"response": "text", ...}, ...]``.
    - **JSONL**: ``.jsonl`` — one JSON object per line.
    - **Plain text**: ``.txt`` — one response per line (stripped).

    Args:
        path: Path to the predictions file.
        key: Dict key to extract from each JSON object.  Ignored for
            plain-text files.

    Returns:
        List of response strings.
    """
    ext = os.path.splitext(path)[1].lower()
    raw = Path(path).read_text(encoding="utf-8")

    if ext == ".jsonl":
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        objs = [json.loads(line) for line in lines]
        return [obj.get(key, "") for obj in objs]

    if ext == ".json":
        data = json.loads(raw)
        if isinstance(data, list):
            return [
                item.get(key, "") if isinstance(item, dict) else str(item)
                for item in data
            ]
        raise ValueError(
            "Expected a JSON list at top level; got a dict. "
            "For dict formats use load_predictions_from_dict()."
        )

    if ext == ".txt":
        return [line.strip() for line in raw.splitlines() if line.strip()]

    # Unknown extension — try JSON first, then plain text
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [
                item.get(key, "") if isinstance(item, dict) else str(item)
                for item in data
            ]
    except (json.JSONDecodeError, ValueError):
        pass
    return [line.strip() for line in raw.splitlines() if line.strip()]


def load_references_from_abcd(
    split: str = "test",
    data_dir: str | None = None,
    use_pool: bool = True,
) -> list[str]:
    """Load ground-truth agent utterances from the ABCD dataset.

    Args:
        split: ``"train"``, ``"dev"``, or ``"test"``.
        data_dir: Path to ABCD data directory.  Defaults to
            ``data/eval/abcd/data/``.
        use_pool: If True, look up the utterance text from
            ``utterances.json`` (more complete).  If False, use the
            delexed text directly.

    Returns:
        List of reference utterance strings (agent turns only).
    """
    from .abcd.data import load_abcd_data, get_utterance_text

    conversations = load_abcd_data(split, data_dir)
    texts: list[str] = []
    for conv in conversations:
        for turn in conv["delexed"]:
            if turn["speaker"] != "agent":
                continue
            targets = turn["targets"]
            utt_id = targets[4]
            if use_pool and utt_id >= 0:
                texts.append(get_utterance_text(utt_id, data_dir))
            else:
                texts.append(turn.get("text", ""))
    return texts


def load_references_from_multiwoz(
    split: str = "test",
    data_dir: str | None = None,
) -> list[str]:
    """Load ground-truth system utterances from the MultiWOZ dataset.

    Args:
        split: ``"train"``, ``"val"``, or ``"test"``.
        data_dir: Path to MultiWOZ splits directory.  Defaults to
            ``data/eval/multiwoz21/splits/``.

    Returns:
        List of reference utterance strings (system turns only).
    """
    from .data import load_multiwoz21

    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "eval", "multiwoz21", "splits"
        )

    split_file = {"train": "all_train.json", "val": "all_val.json", "test": "all_test.json"}[split]
    dialogues = load_multiwoz21(os.path.join(data_dir, split_file))

    texts: list[str] = []
    for dialogue in dialogues:
        for turn in dialogue.turns:
            if turn.speaker == "system":
                texts.append(turn.utterance)
    return texts


def load_references(
    path_or_dataset: str,
    *,
    split: str = "test",
    data_dir: str | None = None,
) -> list[str]:
    """Load ground-truth responses — auto-detect source.

    Args:
        path_or_dataset: Either a file path (``.json`` / ``.txt``) or a
            dataset name (``"multiwoz"`` / ``"abcd"``).
        split: Data split (only used for dataset names).
        data_dir: Override default data directory.

    Returns:
        List of reference strings.
    """
    # If it's a file path on disk
    if os.path.exists(path_or_dataset):
        return load_predictions(path_or_dataset, key="response")

    # Named datasets
    name = path_or_dataset.lower()
    if name == "abcd":
        return load_references_from_abcd(split, data_dir)
    if name in ("multiwoz", "multiwoz21"):
        return load_references_from_multiwoz(split, data_dir)

    raise ValueError(
        f"Unknown reference source: {path_or_dataset!r}. "
        f"Use a file path, 'multiwoz', or 'abcd'."
    )
