"""SGD TRAIN-note construction and frozen A-MEM + shared policy execution."""
from __future__ import annotations

import json
from eval_tod.sgd_adapter import gold_policy, visible_context
from eval_tod.sgd_aligned import prior_api_history, schema_for_dialogue

PROTOCOL = "sgd_official_amem_robust_train_policy_notes_frozen_v1"


def train_note(dialogue, index):
    """Only called on the TRAIN split. Preserve exact named policy payloads."""
    return ("SGD TRAIN example. Apply only in matching context; example entity/date values are not defaults. "
            "The following policy is the SYSTEM decision, not a label of the user's utterance.\n"
            "PRE-TURN DIALOGUE:\n" + visible_context(dialogue, index)[-12000:]
            + "\nPRIOR LOGGED API RESULTS:\n" + json.dumps(prior_api_history(dialogue, index), ensure_ascii=False)
            + "\nREFERENCE SYSTEM POLICY (unordered acts, optional API call):\n"
            + json.dumps(gold_policy(dialogue, index), ensure_ascii=False))


def retrieval_input(dialogue, index):
    """Causal only: no current response, gold act, API call, result or future."""
    return ("Choose the next SGD SYSTEM policy, including its action heads and named payloads.\n"
            + "AVAILABLE SERVICES: " + json.dumps(dialogue["services"])
            + "\nPRIOR API HISTORY:\n" + json.dumps(prior_api_history(dialogue, index), ensure_ascii=False)
            + "\nDIALOGUE BEFORE TARGET:\n" + visible_context(dialogue, index)[-12000:])


def retrieval_knowledge(memory, controller, dialogue, index, *, k=10):
    # Same query-generation template as the official robust evaluation harness,
    # substituting the visible policy task for its LoCoMo question.
    query_input = retrieval_input(dialogue, index)
    prompt = ("Given the following question, generate several keywords separated by commas.\n\n"
              + "Question: " + query_input + "\n\nKeywords:")
    raw = controller.get_completion(prompt)
    from llm_text_parsers import parse_keywords_response
    query = parse_keywords_response(raw)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A-MEM generated an empty retrieval query")
    text = memory.retrieve(query, k=k)
    return ("Relevant TRAIN memories from A-MEM. Use as context-specific evidence; "
            "ground current values in the current dialogue/API observations.\n" + text), {
                "query_input": query_input, "query": query, "retrieve_k": k,
                "retrieved_text": text, "retrieved_chars": len(text)}


def predict(memory, controller, dialogue, index, model, ontology, *, k=10):
    from eval_tod.sgd_extra_baselines import predict_row
    try:
        knowledge, retrieval = retrieval_knowledge(memory, controller, dialogue, index, k=k)
    except Exception as exc:
        from eval_tod.sgd_adapter import score_policies
        gold = gold_policy(dialogue, index)
        prediction = {"call": None, "acts": [], "_execution_error": True}
        return {"sample_id": f"{dialogue['dialogue_id']}:{index}", "dialogue_id": dialogue["dialogue_id"],
                "turn_index": index, "gold": gold, "prediction": prediction, "diagnostic": {},
                "joint_correct": False, "error": f"{type(exc).__name__}: {exc}"}
    # Exactly the same action-head selector, fixed-head payload binder, logged
    # API-result gate and scorer as ReAct. Gold is read after prediction only.
    row = predict_row(dialogue, index, model, ontology, knowledge)
    row["diagnostic"]["amem_retrieval"] = retrieval
    return row
