# Speaker Notes

## Slide 1
We start from a simple observation: trajectory-to-skill systems can produce useful text, but useful text is not yet a reusable skill. The talk will move from concrete generated artifacts to two research directions.

## Slide 2
The current repository gives us generated skills, but not a fully aligned AWM versus Trace2Skill failure table. So this deck distinguishes observed artifact evidence from candidate bad cases that still need trajectory-level confirmation.

## Slide 3
AWM is rich and experience-heavy. Trace2Skill is concise and action-slot oriented. They fail in opposite directions: one tends to accumulate local details, while the other leaves the state and schema that make those constraints actionable implicit.

## Slide 4
The first candidate is macro-skill failure. A workflow that stores concrete names, IDs, and slot values may cover training traces perfectly, but it is not a parameterized procedure. We need a test where the workflow structure is familiar but entities are new.

## Slide 5
The second candidate is not that one rule is wrong. Rules from different states can be individually correct but look contradictory when flattened into one text. The missing object is the condition that separates branches.

## Slide 6
These symptoms suggest a common issue: the generated skill does not teach the agent how to use its evidence. Reference is treated as passive context instead of an operational resource with access and application rules.

## Slide 7
The proposed representation has two policies. The task policy selects business actions. The reference-use policy decides when to retrieve, what to ask for, and how to validate the retrieved evidence before acting.

## Slide 8
The loop is retrieve, ground, apply, and verify. Retrieval should be state-aware. Grounding must use current dialogue values, not values copied from examples. Verification checks both action schema and applicability.

## Slide 9
This can be learned from traces that record current state, query, retrieved evidence, action, and outcome. The learning target is not just the next action; it is also whether to retrieve and how to use the result.

## Slide 10
These are the experiments needed to turn the story into a paper. We need paired runs, controlled perturbations, and an evidence table. Until then, the two cases remain hypotheses rather than results.

## Slide 11
The contribution is not a longer reference or another graph. It is a skill that knows how to access and apply reference under a workflow state. The central question is: when uncertain, can the skill retrieve the right evidence and turn it into the right action?
