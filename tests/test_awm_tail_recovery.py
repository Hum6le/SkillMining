import copy
import json

import pytest

from scripts.abcd_offline_awm import _hash, _read, _save, recover_failed_tail


def setup_tail(root):
    state = {"scope": {"schedule": [["a"], ["b"], ["c"]]}, "completed_batches": 3,
             "workflow": "WORKFLOW", "cards": {"action_rules.md": "RULE", "slot_policies.md": "SLOT"},
             "exemplars": [{"id": "kept"}], "usage": {"total": {"calls": 99}},
             "failed_batches": [{"batch_index": i, "batch_number": i + 1, "phase": "induction", "error": "old"}
                                for i in (1, 2)]}
    checkpoint = root / "awm_state.json"
    _save(checkpoint, state)
    for failure in state["failed_batches"]:
        index = failure["batch_index"]
        _save(root / "batches" / f"batch_{index + 1:04d}.json", {"identity": {
            "scope_hash": _hash(state["scope"]), "workflow_hash": _hash(state["workflow"]),
            "cards_hash": _hash(state["cards"]), "index": index}, "rows": []})
        _save(root / "failed_batches" / f"batch_{index + 1:04d}.json", failure)
    _save(root / "result.json", {"scope_hash": _hash(state["scope"]), "training_complete": False})
    (root / "skill.md").write_text("OLD FINAL", encoding="utf-8")
    return checkpoint, state


def test_tail_recovery_preserves_state_costs_and_backs_up_results(tmp_path):
    checkpoint, state = setup_tail(tmp_path)
    original = checkpoint.read_bytes()
    result_bytes = (tmp_path / "result.json").read_bytes()
    caches = {p: p.read_bytes() for p in (tmp_path / "batches").glob("*.json")}
    updated = recover_failed_tail(state, checkpoint)
    assert updated["completed_batches"] == 1 and updated["failed_batches"] == []
    for key in ("scope", "workflow", "cards", "exemplars", "usage"):
        assert updated[key] == state[key]
    backup = next((tmp_path / "tail_recovery_backups").iterdir())
    assert (backup / "awm_state.json").read_bytes() == original
    assert (backup / "result.json").read_bytes() == result_bytes
    assert (backup / "archived_result.json").read_bytes() == result_bytes
    assert not (tmp_path / "result.json").exists()
    assert len(list((backup / "failed_batches").glob("*.json"))) == 2
    assert not list((tmp_path / "failed_batches").glob("*.json"))
    assert all(p.read_bytes() == raw for p, raw in caches.items())
    assert _read(checkpoint) == updated
    assert recover_failed_tail(updated, checkpoint) == updated


@pytest.mark.parametrize("problem", ["hole", "duplicate", "changed_state", "later_cache", "wrong_result", "not_induction"])
def test_tail_recovery_rejects_unsafe_rewind_without_touching_files(tmp_path, problem):
    checkpoint, state = setup_tail(tmp_path)
    state = copy.deepcopy(state)
    if problem == "hole":
        state["failed_batches"] = state["failed_batches"][:1]
    elif problem == "duplicate":
        state["failed_batches"].append(state["failed_batches"][0])
    elif problem == "changed_state":
        state["workflow"] += " changed"
    elif problem == "later_cache":
        (tmp_path / "batches" / "batch_0003.json").write_text('{}')
    elif problem == "wrong_result":
        (tmp_path / "result.json").write_text('{"scope_hash":"wrong"}')
    else:
        state["failed_batches"][0]["phase"] = "rollout"
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        recover_failed_tail(state, checkpoint)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert not (tmp_path / "tail_recovery_backups").exists()
