# External Research Deck Notes

## Slide 1
Our goal is to make trajectory-derived skills reusable. The central idea is to teach the skill not only what action to take, but also when it should consult reference and how it should apply the retrieved evidence.

## Slide 2
The motivating failure is not simply poor wording. A workflow can look correct on its source traces while being tied to concrete entities, while a compact action protocol can be precise but leave state and schema implicit.

## Slide 3
These two generated skill styles expose complementary symptoms. A richer workflow can accumulate local details and priorities without a global abstraction. An action-slot protocol can prevent some output errors but cannot, by itself, determine which state the dialogue is in.

## Slide 4
This is an illustrative macro-skill case. The workflow remembers a concrete name, phone, or account identifier. The test keeps the same procedure but changes the entity values. A reusable skill should substitute variables and ground them in the current dialogue.

## Slide 5
This is an illustrative branch case. Both rules can be correct: verification is appropriate when enough credentials are available, while asking for another credential is appropriate otherwise. Flattening them into a single rule creates an apparent conflict because the state condition is missing.

## Slide 6
The key insight is that reference cannot remain passive context. The skill should decide whether it needs evidence, formulate a query around its uncertainty, and verify that the retrieved rule applies to the current state.

## Slide 7
We therefore represent a skill with two coupled policies. The task policy maps states to business actions. The reference-use policy maps states and uncertainty to retrieval, evidence interpretation, and verification decisions.

## Slide 8
The execution loop is state-aware. We first estimate the state, decide whether to retrieve or act, retrieve the relevant type of evidence, ground all values in the current dialogue, apply the rule, and verify the action-slot pair before execution.

## Slide 9
The learning signal should include the decision trace around retrieval, not only the final action. This lets us supervise when retrieval was useful, what type of evidence was needed, and whether the evidence actually changed or validated the decision.

## Slide 10
The evaluation should isolate the three hypothesized failure families. Macro cases test parameterization, branch cases test state-conditioned rules, and reference cases test whether examples are grounded or copied. The mock labels should be replaced with actual paired cases.

## Slide 11
The contribution is a reference-aware skill, not a longer prompt. It learns when and how to consult evidence and converts that evidence into a state-conditioned, reusable action.
