# Reflexion on ABCD

This is an offline ABCD adaptation of Reflexion's actor → feedback → verbal
reflection → retry loop. Each complete training conversation is attempted up
to `--reflexion-max-trials` times. The evaluator uses the training conversation's
recorded actions and slots to calculate AST feedback. On failure, one plain-text
self-reflection is generated and included in the next attempt of that same
training conversation. Each completed trial, including its feedback and exact
reflection text, is appended to `training_trials.jsonl`; `reflections.json` is
the frozen memory loaded for testing.

At test time, the actor retrieves up to `--reflexion-reflection-limit` raw
training reflections using the visible dialogue prefix. It makes one pass over
the frozen test conversation and does not update memory. Training and testing
LLM usage are written automatically to `llm_usage.json` and the method summary.

Run all ten indexed ABCD subflows with the shared balanced workflow workers:

```bash
bash scripts/launch_full_abcd_experiments.sh --method reflexion \
  --no-rebuild-splits --workflow-ids "ID_A,ID_B,ID_C,ID_D" --stop-on-error
```

The full runner uses `--resume-run DIR` to continue from completed trial
checkpoints. For one subflow, `--eval-workflow-ids` can shard the frozen test
evaluation across endpoints, as for the other baselines.

ABCD contains recorded dialogues rather than a live environment. A changed
backend action cannot produce a new observation, so retries score predictions
against the same recorded trajectory. This does not reproduce the original
paper's interactive environment feedback. The reflector sees the exact gold
action and ordered gold slots for failed **training** turns. The test actor
never sees test gold labels. The reflection prompt asks for reusable guidance;
customer-specific training values are redacted from the saved reflection text.
