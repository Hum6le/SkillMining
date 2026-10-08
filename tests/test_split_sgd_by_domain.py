import gzip
import io
import json
from zipfile import ZipFile

from scripts.split_sgd_by_domain import build_splits, iter_json_array


def _turn(speaker, *, acts=(), calls=None):
    return {
        "speaker": speaker,
        "utterance": "example",
        "dialogue_acts": {"binary": list(acts), "categorical": [], "non-categorical": []},
        "service_call": calls or {},
    }


def _dialogue(split, number, services, turns):
    return {
        "data_split": split,
        "dialogue_id": f"sgd-{split}-{number}",
        "original_id": str(number),
        "domains": services,
        "turns": turns,
    }


def test_streaming_json_array_handles_small_chunks():
    rows = [{"text": "é" * 30}, {"turns": [1, 2, 3]}]
    assert list(iter_json_array(io.StringIO(json.dumps(rows)), chunk_size=7)) == rows


def test_train_domain_projection_keeps_official_splits_and_call_ownership(tmp_path):
    dialogues = [
        _dialogue("train", 0, ["Restaurants_1"], [
            _turn("user"),
            _turn("system", acts=[{"intent": "request", "domain": "Restaurants_1", "slot": "city"}]),
            _turn("system", calls={"Restaurants_1": {"method": "FindRestaurants", "parameters": {}}}),
        ]),
        _dialogue("validation", 1, ["Restaurants_2"], [
            _turn("system", calls={"Restaurants_2": {"method": "FindRestaurants", "parameters": {}}}),
        ]),
        _dialogue("test", 2, ["Restaurants_2", "Alarm_1"], [
            _turn("system", calls={"Alarm_1": {"method": "AddAlarm", "parameters": {}}}),
            _turn("system", acts=[{"intent": "offer", "domain": "Restaurants_2", "slot": "name"}],
                  calls={"Restaurants_2": {"method": "FindRestaurants", "parameters": {}}}),
            _turn("system", acts=[{"intent": "goodbye", "domain": "", "slot": ""}]),
        ]),
        _dialogue("test", 3, ["Alarm_1"], [
            _turn("system", calls={"Alarm_1": {"method": "AddAlarm", "parameters": {}}}),
        ]),
    ]
    archive = tmp_path / "sgd.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("data/dialogues.json", json.dumps(dialogues))
        zipped.writestr("data/ontology.json", json.dumps({"domains": {}}))
    output = tmp_path / "splits"
    index = build_splits(archive, output)
    assert index["train_families"] == ["Restaurants"]
    assert index["excluded_families"]["Alarm"]["test"] == 2
    assert index["families"]["Restaurants"]["test"]["api_events"] == 1
    with gzip.open(output / "Restaurants" / "test.jsonl.gz", "rt", encoding="utf-8") as stream:
        row = json.loads(stream.readline())
    assert row["dialogue_id"] == "sgd-test-2"
    assert row["target_indices"]["api_turn_indices"] == [1]
    assert row["target_indices"]["domain_act_turn_indices"] == [1]
    assert row["target_indices"]["global_act_turn_indices"] == [2]
    assert row["target_indices"]["policy_turn_indices"] == [1]
    assert row["target_indices"]["unassigned_global_act_turn_indices"] == [2]
    assert not (output / "Alarm").exists()
