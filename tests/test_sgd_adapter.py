from eval_tod.sgd_adapter import (
    action_counter,
    canonical_acts,
    gold_policy,
    score_policies,
    visible_context,
)


def test_multi_act_turn_is_order_invariant_but_slot_sensitive():
    truth = {
        "call": None,
        "acts": canonical_acts([
            {"intent": "confirm", "domain": "Restaurants_1", "slot": "date", "value": "today"},
            {"intent": "confirm", "domain": "Restaurants_1", "slot": "time", "value": "19:00"},
        ]),
    }
    reordered = {"call": None, "acts": list(reversed(truth["acts"]))}
    assert action_counter(truth) == {("confirm", "Restaurants_1"): 1}
    assert score_policies([truth], [reordered])["turn_joint_ast"] == 1.0
    wrong_slot = {"call": None, "acts": canonical_acts([
        {"intent": "confirm", "domain": "Restaurants_1", "slot": "date", "value": "today"},
        {"intent": "confirm", "domain": "Restaurants_1", "slot": "party_size", "value": "2"},
    ])}
    result = score_policies([truth], [wrong_slot])
    assert result["action_set_exact"] == 1.0
    assert result["turn_joint_ast"] == 0.0


def test_call_method_counts_as_slot_and_future_system_text_is_hidden():
    dialogue = {
        "services": ["Restaurants_1"],
        "all_services": ["Restaurants_1"],
        "turns": [
            {"speaker": "user", "utterance": "Book a table", "state": {"secret": "gold"}},
            {
                "speaker": "system", "utterance": "Booking complete", "db_results": {"Restaurants_1": ["future"]},
                "service_call": {"Restaurants_1": {"method": "ReserveRestaurant", "parameters": {"party_size": "2"}}},
                "dialogue_acts": {"binary": [{"intent": "notify_success", "domain": "Restaurants_1", "slot": ""}]},
            },
        ],
    }
    context = visible_context(dialogue, 1)
    assert "Book a table" in context
    assert "Booking complete" not in context
    assert "secret" not in context
    assert "future" not in context
    truth = gold_policy(dialogue, 1)
    wrong = {"call": {**truth["call"], "method": "FindRestaurants"}, "acts": truth["acts"]}
    result = score_policies([truth], [wrong])
    assert result["action_set_exact"] == 1.0
    assert result["turn_joint_ast"] == 0.0
    assert result["api_joint_ast"] == 0.0


def test_runtime_stages_do_not_show_current_gold_before_call(monkeypatch):
    from scripts.run_sgd_domain_eval import predict_policy

    prompts = []
    def fake_chat(prompt, **_kwargs):
        prompts.append(prompt)
        if len(prompts) == 1:
            return '{"call":{"service":"Restaurants_1","method":"FindRestaurants","parameters":{"city":"San Jose"}}}'
        return '{"acts":[{"intent":"offer","domain":"Restaurants_1","slot":"restaurant_name","value":"Bird Dog"}]}'

    monkeypatch.setattr("llm.chat", fake_chat)
    dialogue = {
        "services": ["Restaurants_1"], "all_services": ["Restaurants_1"],
        "turns": [
            {"speaker": "user", "utterance": "Find a restaurant in San Jose"},
            {"speaker": "system", "utterance": "Bird Dog is available", "db_results": {"Restaurants_1": [{"name": "Bird Dog"}]},
             "service_call": {"Restaurants_1": {"method": "FindRestaurants", "parameters": {"city": "San Jose"}}},
             "dialogue_acts": {"non-categorical": [{"intent": "offer", "domain": "Restaurants_1", "slot": "restaurant_name", "value": "Bird Dog"}]}}
        ],
    }
    ontology = {"domains": {"Restaurants_1": {
        "description": "restaurant search", "active_intents": [
            {"name": "FindRestaurants", "description": "find a restaurant", "required_slots": ["city"]}
        ]
    }}}
    prediction, diagnostic = predict_policy(
        dialogue, 1, model="mock", ontology=ontology, frequency_graph=None,
    )
    assert "Bird Dog" not in prompts[0]
    assert "Bird Dog" in prompts[1]
    assert diagnostic["observation_visible"] is True
    assert prediction["call"]["method"] == "FindRestaurants"
