from __future__ import annotations

import json
from unittest.mock import patch

from scripts.evaluate_abcd_method import _merge


def test_sharded_merge_preserves_compiled_mode_for_resume(tmp_path) -> None:
    subflow = "account_access"
    output = tmp_path / "run"
    output.mkdir()
    (output / "summary.json").write_text(json.dumps({
        "config": {"method": "skill_disco", "subflow": subflow,
                   "compile_and_verify": True, "skip_final_test": True},
        "data": {"train_sessions": 4, "test_sessions": 1},
        "generation": {"candidate_contracts": 2, "verified_skills": 1},
        "llm_usage": {"generation": {"total": {"calls": 2, "prompt_tokens": 10}}},
    }), encoding="utf-8")
    test_file = tmp_path / "test.json"
    test_file.write_text(json.dumps([{"convo_id": "1", "scenario": {"subflow": subflow}}]),
                         encoding="utf-8")
    shard = output / "eval_shards" / "shard_0"
    shard.mkdir(parents=True)
    (shard / "turn_predictions.json").write_text("[]", encoding="utf-8")
    (shard / "llm_usage.json").write_text("{}", encoding="utf-8")
    with patch("scripts.evaluate_abcd_method.merge_turn_results", return_value=[]), \
         patch("scripts.evaluate_abcd_method.turn_results_to_abcd_predictions", return_value=[]), \
         patch("scripts.evaluate_abcd_method.evaluate_abcd_bundle",
               return_value={"summary": {}, "text": {}, "ast_cds": {}}):
        _merge("skill_disco", subflow, test_file, output / "eval_shards", output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["compile_and_verify"] is True
    assert summary["config"]["skip_final_test"] is False
    assert summary["generation"]["verified_skills"] == 1
    assert summary["llm_usage"]["generation"]["total"]["calls"] == 2
    with patch("scripts.evaluate_abcd_method.merge_turn_results", return_value=[]), \
         patch("scripts.evaluate_abcd_method.turn_results_to_abcd_predictions", return_value=[]), \
         patch("scripts.evaluate_abcd_method.evaluate_abcd_bundle",
               return_value={"summary": {}, "text": {}, "ast_cds": {}}):
        _merge("skill_disco", subflow, test_file, output / "eval_shards", output)
    repeated = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert repeated["llm_usage"]["generation"]["total"]["calls"] == 2
