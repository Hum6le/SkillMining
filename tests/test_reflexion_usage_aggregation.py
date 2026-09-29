"""Modern phase-split usage must not be re-added from shard files."""

import json

from scripts.aggregate_subflow_results import _phase_usage_from_legacy_artifacts


def test_phase_split_summary_is_authoritative_over_shard_usage(tmp_path):
    shard = tmp_path / "eval_shards" / "shard_0"
    shard.mkdir(parents=True)
    (shard / "llm_usage.json").write_text(json.dumps({
        "testing": {"total": {"calls": 5}},
    }), encoding="utf-8")
    modern = {
        "schema_version": 2,
        "generation": {"total": {"calls": 3}},
        "testing": {"total": {"calls": 5}},
        "total": {"total": {"calls": 8}},
    }
    assert _phase_usage_from_legacy_artifacts(tmp_path, modern) is None
