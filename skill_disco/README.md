# Skill-DisCo implementation status

The existing ABCD pipeline implements trace normalization, semantic operation
extraction, two-pass consolidation, and typed skill contracts. The optional
`--compile-and-verify` mode adds Stage 5:

1. Hold out complete conversations from the ABCD training split before any
   discovery call. The frozen test split is never used for synthesis.
2. Synthesize a Python function from each Stage-4 contract and representative
   induction operations.
3. Execute each candidate in a bounded worker against held-out recorded ABCD
   action/slot/observation sequences. Feed failures to the synthesizer for up to
   three attempts and discard candidates that do not pass.
4. Save verified functions in `generation_artifact.json` under
   `compiled_skills`. `CompiledSkillLibrary.load(path)` exposes documented tool
   signatures and `invoke(name, arguments, env)` for interactive adapters.

The unified ABCD runner enables this mode by default. Run all 10 indexed
subflows with the existing split files:

```bash
bash scripts/run_full_abcd_experiments.sh --method skill_disco \
  --no-rebuild-splits --stop-on-error
```

For the same multi-workflow scheduling used by AWM and Trace2Skill, launch in
the background with one ID per concurrent worker:

```bash
bash scripts/launch_skill_disco_abcd_all.sh --no-rebuild-splits \
  --workflow-ids "ID_A,ID_B,ID_C,ID_D" --stop-on-error
```

The launcher assigns each complete subflow to one worker. Generation and test
evaluation child processes inherit that worker's `SKILLMINING_WORKFLOW_ID`;
the server deployment uses `llm_new.py` as `llm.py` to select the endpoint.
The ten subflows run concurrently across IDs, while each worker processes its
assigned subflows in order.

## ABCD verification boundary

ABCD provides recorded conversations and action labels, but no interactive
backend `env.step` or postcondition API. Its replay verifier checks exact
recorded action names, ordered slot values, observations, process traces, and
return fields. The `available_actions` value is the fixed public ABCD action
vocabulary, never the next gold action. Replay does not prove behavior for
unseen branches or counterfactual environment states.

At ABCD test time, the LLM selects a verified skill from its documented
signatures using only the dialogue prefix. The selected Python function is
executed against a prefix environment. Calls matching **previously predicted**
actions replay their already observed results; its first unobserved `env.step`
call becomes the current action prediction. The target action label and future
turns are never supplied to this invocation. An inapplicable skill falls back
to the existing ABCD agent. Utterance predictions use the same base agent and
the existing AST/CDS/text evaluator remains unchanged.

This is the closest causal callable-skill adaptation possible with ABCD's
offline data. It does not measure the paper's interactive task success or
multi-action turn savings: ABCD tests one recorded turn at a time and cannot
produce counterfactual backend observations. To reproduce those original
metrics, an ALFWorld or WebArena adapter would call
`CompiledSkillLibrary.invoke` with a live environment. Neither environment is
included in this repository.
