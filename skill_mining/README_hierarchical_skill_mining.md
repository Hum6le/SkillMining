# Hierarchical Residual Hypergraph Mining

`hierarchical_skill_mining.py` is the first deterministic implementation of
the new mining formulation:

```text
ordered traces
    -> behavior basis (high-coverage cover)
    -> main backbone (weighted ordered path)
    -> residual conditional branches
    -> low-support reference motifs
    -> agent evidence package
```

The script is intentionally separate from the legacy HG miner. It keeps the
structural selection deterministic and leaves semantic interpretation to a
later reasoning agent.

## Run

```powershell
python skill_mining/hierarchical_skill_mining.py `
  --operator-results skill_mining/output/abcd_session_hg/operator_results.json `
  --output skill_mining/output/abcd_hierarchical_skill.json `
  --rho 0.8 `
  --min-branch-support 2
```

## Interpretation

- `behavior_basis`: a compact node set covering at least `rho` of each
  selected session's unique behavior nodes.
- `backbone`: the highest-support cycle-free ordered path through the basis.
- `branches`: residual paths that share an attachment point and have enough
  cross-session support.
- `references`: residual paths that are structurally attached but too rare to
  be promoted to a branch.
- `agent_evidence`: representative traces and structural motifs for semantic
  reasoning. The agent should infer names, guards, preconditions, and effects
  only from this evidence and may abstain.

The current operator input is canonicalized by removing instance arguments
after the first `:` in an operation label. This prevents names, IDs, and other
trace-specific values from becoming behavior nodes while retaining the
original dialogue as evidence for the later semantic stage.

## Per-subflow mining

When the input contains multiple business subflows, mine them independently:

```powershell
python skill_mining/mine_hierarchical_skills_by_subflow.py `
  --operator-results skill_mining/output/abcd_session_hg/operator_results.json `
  --output skill_mining/output/abcd_hierarchical_skills_by_subflow.json `
  --rho 0.8 `
  --min-branch-support 2 `
  --min-sessions 2
```

The wrapper writes one hierarchy per subflow and a directory of grounded
reasoning prompts. Subflows below `--min-sessions` are reported as skipped so
that rare workflows are not silently treated as global patterns.

Pass `--reasoning-prompt` to the global miner to write a matching prompt next
to its JSON output. The prompt asks the agent to infer semantic names,
preconditions, effects, and branch guards, with `unknown` as the required
fallback when the graph evidence is insufficient.

## Model-agnostic semantic reasoning

Use the repository-level `llm.chat` wrapper to interpret the mined motifs:

```powershell
python skill_mining/run_semantic_reasoning.py `
  --input skill_mining/output/abcd_hierarchical_skills_by_subflow.json `
  --output skill_mining/output/abcd_hierarchical_skills_reasoned.json
```

The runner does not select a provider or model. Those settings remain inside
`llm.py`. It stores the normalized semantic result, parse status, and raw
response for each motif. If `chat` is not configured or fails, the runner
records an error and returns `unknown` fields instead of changing the mined
structure.

## Method boundary

This implementation covers the hypergraph organization stage. An upstream
agent graph completion stage can provide additional candidate states and
transitions; those candidates should be marked as observed, inferred, or
validated before being allowed to affect the backbone score.
