"""Shared helpers for single-subflow evaluation sharding."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def parse_workflow_ids(raw: str | Iterable[str] | None) -> list[str]:
    """Normalize comma-separated workflow IDs while preserving order."""
    if raw is None:
        return []
    values = [raw] if isinstance(raw, str) else list(raw)
    result: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item and item not in result:
                result.append(item)
    return result


def shard_conversations(
    conversations: list[dict[str, Any]], shard_index: int, shard_count: int,
) -> list[dict[str, Any]]:
    """Assign complete conversations round-robin to evaluation shards."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    return conversations[shard_index::shard_count]


def merge_turn_results(
    conversations: list[dict[str, Any]], shard_results: Iterable[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge shard outputs and reject unknown or missing conversations."""
    expected = {str(row.get("convo_id", "")) for row in conversations}
    by_conversation: dict[str, list[dict[str, Any]]] = {}
    for rows in shard_results:
        for row in rows:
            convo_id = str(row.get("convo_id", ""))
            if convo_id not in expected:
                raise ValueError(f"shard returned unknown conversation: {convo_id}")
            by_conversation.setdefault(convo_id, []).append(row)
    missing = sorted(expected - set(by_conversation))
    if missing:
        raise ValueError(f"shards did not return conversations: {missing[:10]}")
    merged: list[dict[str, Any]] = []
    for conversation in conversations:
        convo_id = str(conversation.get("convo_id", ""))
        merged.extend(sorted(by_conversation[convo_id], key=lambda row: int(row.get("turn_index", 0))))
    return merged
