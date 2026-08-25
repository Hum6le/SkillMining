# Unified Skill Graph Figure Prompt

下面的 prompt 对应当前统一理论：从粗粒度 flow 的多 session trajectory 中发现一个 skill-resource pair \((S,R)\)。其中 `S` 是由 backbone 与少量已验证 forward branch 组成的可执行 DAG；`R` 是按需检索的 residual control resource。session pattern cohort 只用于离线重加权 backbone edge，不产生多个 runtime skill。MST/arborescence 只表示 backbone 约束与求解步骤，不应被画成整套 skill 的终点。

## English Image-Generation Prompt

Create a polished, publication-quality **method overview figure** for a machine-learning paper titled **“Evidence-Calibrated Skill DAG Discovery from Dialogue Trajectories.”** Use a landscape 16:9 canvas, clean vector infographic style, white or very light neutral background, restrained academic palette: deep teal for structural graphs, muted coral for residual evidence, warm gold for selected backbone edges, charcoal text, and light gray for inactive context. Use subtle shadows and clear depth, but no gradients, no decorative blobs, no cartoon characters, no screenshots, and no dense paragraphs. All labels must be crisp, readable English.

The figure must communicate one unified idea, not a sequence of unrelated modules:

**From many coarse-flow dialogue trajectories, discover one skill-resource pair \((S,R)\). The skill is a compact executable DAG, while uncertain or low-frequency control knowledge is selectively externalized into retrievable resources.**

Organize the illustration from left to right in five connected stages:

1. **Input: Cross-session trajectory evidence.** On the far left, show 5 to 6 compact dialogue traces stacked vertically. Each trace alternates customer utterance bubbles, agent utterance bubbles, and small action chips such as `pull-up-account`, `verify-identity`, `send-link`, `make-password`. Repeated action transitions across different traces should be visually aligned with faint curved connectors, conveying shared evidence across sessions. Put a small header: “Coarse-flow training dialogues”. Do not show gold subflow labels.

2. **Shared evidence graph and trajectory-pattern contrast.** In the left-center, merge the traces into one global directed action-transition graph. Nodes are action chips; edge thickness reflects cross-session support. Add 3 subtle, semi-transparent session-pattern cohorts in the background, labelled `trajectory pattern A`, `B`, and `C`; they are evidence overlays, not separate skills. Highlight one shared high-frequency trunk in gray-teal and several cohort-specific edges in muted teal. Add a small caption: “cohorts reweight edges; no runtime skill split”.

3. **One expanded skill pair, the central focus.** Zoom into the single coarse-flow skill in the center. Split it vertically into two coordinated layers enclosed by one shared outline labeled `\(\mathcal K=(S,R)\)`:
   - Upper and larger layer: `S: executable Skill DAG`. Show every action node connected. Highlight a warm-gold rooted spanning arborescence named `high-support backbone B`; add 2 thin teal forward branch edges labeled `verified branches E+`. The structure must visibly be a DAG, with arrows flowing mostly left-to-right and no cycles.
   - Lower and smaller layer: `R: residual control resource`. Show residual branch cards connected by dotted pointers from their source node: `uncertain branch`, `retry / revisit`, `transition evidence`, `slot policy`, `action rule`. Use muted coral. A clear visual gate routes only verified, guard-resolved forward branches upward into the DAG; unresolved or low-value branches remain below. Include a tiny label: “selective disclosure, not deletion”.

4. **Graph-grounded language induction and organized compilation.** In the right-center, show a compact “source-local decision” panel: one source action and several sibling outgoing candidate edges, each with 2 short dialogue evidence snippets. A small LLM icon compares the sibling evidence and outputs a concise natural-language `transition guard`. Emphasize that the LLM does not invent topology: place the note “structure selected by evidence; language induced locally”. Then render the same skill pair into four neatly stacked resource documents: `skill.md` (compact DAG + high-value guards), `action_rules.md`, `slot_policies.md`, and `reference.md` (deferred evidence and exceptions). Use solid arrows for main-skill content and dotted arrows for reference-only content.

5. **Runtime and online refinement.** On the far right, show a new dialogue prefix entering the single skill package directly. The agent sees `skill.md` first and performs selective MCP-style lookup into reference/action/slot resources only when needed, then outputs `action + ordered slots + response`. Below this, show a thin feedback loop from **train-only rollout feedback** back to `R`: AST/action/slot feedback updates edge reliability; a reliable forward branch with a resolved guard passes through a “promotion gate” into `S`; a low-confidence or conflicting branch is sunk into `reference.md`. Explicitly label this loop “train split only” and keep test evaluation outside the loop.

Across the bottom, add one concise mathematical objective strip, visually secondary but readable:

“Explain trajectories with compact executable DAGs while minimizing ambiguous routing and unnecessary exposed residual knowledge.”

Use a visual hierarchy that makes the central skill pair \((S,R)\) the largest object. Do not depict MST as the final skill itself; depict it specifically as the warm-gold backbone inside the larger DAG. Do not use raw JSON, code blocks, benchmark tables, or crowded formulas. The overall result should feel like a top-tier ACL/NeurIPS systems-and-learning method figure: structured, elegant, evidence-aware, and technically precise.
