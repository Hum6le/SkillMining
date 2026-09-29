"""Preserve generation usage when final_test embeds testing-only usage."""

import json

from scripts.aggregate_subflow_results import _add_usage, _empty_usage, _records_from_summary


def _tracker(calls: int, tokens: int) -> dict:
    return {"schema_version": 1, "total": {
        "calls": calls,
        "successful_calls": calls,
        "total_tokens": tokens,
        "estimated_calls": calls,
        "usage_source": "estimated",
    }}


def test_skill_disco_aggregate_uses_run_level_phase_usage(tmp_path):
    generation = _tracker(1399, 3022758)
    testing = _tracker(7, 700)
    summary = {
        "config": {"method": "skill_disco", "subflow": "account_access"},
        "data": {"test_sessions": 1},
        "llm_usage": {"schema_version": 2, "generation": generation,
                      "testing": testing, "total": _tracker(1406, 3023458)},
        "final_test": {
            "text": {"num_samples": 1},
            "ast_cds": {"num_action_turns": 1},
            "llm_usage": {"schema_version": 2, "generation": _tracker(0, 0),
                          "testing": testing, "total": testing},
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    record, = _records_from_summary(path)
    assert record["llm_usage"]["generation"]["total"]["calls"] == 1399
    generation_total = _empty_usage()
    _add_usage(generation_total, record["llm_usage"]["generation"])
    assert generation_total["calls"] == 1399
    assert generation_total["total_tokens"] == 3022758
