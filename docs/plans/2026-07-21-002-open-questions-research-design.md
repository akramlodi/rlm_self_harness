---
created: 2026-07-28
updated: 2026-07-28
type: open-questions
status: RD3 resolved; RD1, RD2, RD4 open
parent: docs/plans/2026-07-21-002-feat-rlm-harness-surfaces-from-paper-failures-plan.md
references:
  - Self-Harness (Zhang et al.), arXiv:2606.09498
---

# Research-design questions the harness plan cannot settle

Four questions surfaced by the 2026-07-28 multi-persona review of the harness-surfaces plan. Each was raised independently by two or more reviewers, and none is fixable by ordinary plan editing — they are preregistration decisions about what the experiment measures and claims, not build decisions about what the code does.

The harness plan was revised in the same pass to absorb everything that *was* bookkeeping: path corrections, the H0/H0\* split, the I1 enforcement rewrite, the missing proposal unit, the fidelity-criteria restatement.

**RD3 was subsequently resolved** by reading Self-Harness in full and restructuring the surface set to follow its pattern — surfaces declared from loop phases rather than from documented failures, the loop starting from a sparse H0, and the paper's failures repositioned as a predicted-overlap measurement. That resolution also reshapes RD4. Both are kept below with their reasoning intact, because a reviewer will ask for exactly this argument.

They are ordered by when they bind. RD1 and RD2 bind before the first optimization run, because both change what U1 records and unrecorded trace data cannot be reconstructed afterward. RD4 binds before the pilot reports.

---

## RD1 — The cost-aware gate rewards the collapse that I2 only monitors

**Raised by:** adversarial (P0, anchor 75)

**The problem.** The acceptance predicate is "pass-rate non-regression AND sub-call/cost within a preregistered band," and the plan's behavioral tier says I2 and I3 are "observation, not prevention." Those two point in opposite directions. An edit that collapses recursion — the paper's headline failure, and the exact thing that turns an RLM back into a compaction agent — *lowers* sub-call count and cost. Under a one-sided tolerance ("stay under budget") it passes the gate cleanly, gets promoted, and the only record that anything went wrong is a lineage entry nobody is required to read. After freeze, the study could be shipping a harness that scored well by ceasing to be an RLM.

The plan already has the detection mechanism: U1's trace metrics record sub-call count per run. What it lacks is any path from detection to consequence.

**The decision.** Whether the preregistered band is one-sided or two-sided.

- *Two-sided (adversarial's recommendation).* Give the band a nonzero sub-call-count floor and an answer-from-variable floor whose breach **blocks** promotion. This is the only mechanism in the current design that could act on drift the metrics detect. Cost: a floor is a free parameter, and setting it wrong either blocks legitimate efficiency gains or fails to catch real collapse.
- *One-sided, with collapse handled at analysis.* Keep the gate as-is and treat recursion collapse as a reported finding rather than a blocked one — arguably the more honest posture if the study's claim is "here is what unconstrained self-harnessing does," including when it degenerates.

**What it depends on.** Whether the paper's claim is "Self-Harness improves RLM harnesses" (needs the floor — a collapsed harness is not an improved RLM) or "Self-Harness optimizes what you point it at, and here is what it does to recursion depth when nothing stops it" (does not need the floor, and is a more interesting negative result if collapse actually happens).

**Where it lands if resolved.** OQ1 in the harness plan currently says the band's value comes from pilot measurements; whichever way this resolves, OQ1 needs to say whether the band has one bound or two before the pilot can calibrate it.

---

## RD2 — The failure signature ϕ is not computable from what U1 records

**Raised by:** product-lens (P1, anchor 75) and adversarial (P1, anchor 75), independently, on two different missing components — merged at anchor 100

**The problem.** The Verification Contract claims "the failure signature ϕ is computable from a logged run without harness source." It is not, and it is missing pieces from two directions.

*Verifier side (product-lens).* The proposal defines ϕ over four components: verifier cause, failing level, causal status, and agent mechanism. R4 enumerates only the fourth — sub-call count, syntax-error rate, answer-protocol misuse, buffer-discard, truncation. The other three need the root verifier's outcome and the per-child sub-verifier label, and Scope Boundaries explicitly excludes "environments, verifiers, sub-verifiers." Sub-verification is the proposal's stated reason its failure attribution is "more than a model judgment," so this is not a peripheral gap.

*Structural side (adversarial).* ϕ's second component is the *failing level*, and U1's metric list records no per-call recursion depth and no parent-child link. `RLMChatCompletion` carries no depth field today — only `RLMMetadata.max_depth`, which is the configured ceiling, not where a given call actually ran. Without a call tree you cannot say which level failed.

**Why it can't wait.** The plan's own Risks table names this class of gap unrecoverable: "retrofitting after runs would require reconstructing superseded harness state." If the first optimization runs land without these fields, the traces they produce cannot answer the question the mining stage exists to ask, and re-running means re-paying the full compute cost.

**The decision.** Two parts, and they are separable.

- *Structural side is cheap and probably just do it.* Add per-call recursion depth and parent-call id to U1's metric list and the `to_dict`/`from_dict` round trip. Same `None`-default socket-serialization treatment as the other new fields. This is close to bookkeeping and was left here only because it changes what the Verification Contract can claim.
- *Verifier side is a real scope question.* Either add nullable `verifier_outcome` / `sub_verifier_label` fields to the per-call record now — so the landing place exists when the loop supplies them — or narrow the Verification Contract to claim only the agent-mechanism component. The plan has been revised to make the narrower claim in the interim; if the fields are added, widen it back.

---

## RD3 — Surface selection as human prior — **RESOLVED 2026-07-28**

**Raised by:** product-lens (P1, anchor 75) and adversarial (P1, anchor 75) — merged at anchor 100
**Status:** resolved; the harness plan was restructured to implement the resolution. Kept here because the reasoning is what a reviewer will ask for.

**The original problem.** The plan's first revision derived its six surfaces from failures the RLM paper documents *and fixes* — for three of them it cited not just the failure but the winning fix (a decomposition hint worth +69.5%, a one-sentence batching rule, an offline `FINAL` patcher). The headline claim is that a fixed model identifies recurring failures in its own recursive behavior and converts them into harness edits. A reviewer's first question is how much of the gain is the model's discovery versus the surface prior, and the proposal's existing ablations could not answer it: leave-one-edit-out varies *edits*, not *surfaces*, so it holds the prior fixed by construction.

**What settled it.** Reading Self-Harness (arXiv:2606.09498) in full. Its Figure 3 is the initial harness: **nine builder functions declared a priori by the authors** in a harness-definition file, and the paper states *"the editable surfaces correspond to declared configuration points in this harness."* Nobody mined traces to choose those nine. The contribution sentence is scoped to match — the model *"proposes bounded edits to **declared** harness surfaces"* — and surface selection appears nowhere in their limitations paragraph, which names four other limits.

So the objection splits in two, and only one half was ever real:

*Surface topology — which knobs exist.* A human prior in Self-Harness too. Defensible by precedent, and not a limitation to apologize for: declaring the surface set is the definition of the object being optimized.

*Surface content at h0 — what is already in the knobs.* This is where the plan had genuinely diverged. Self-Harness's floor is sparse: `build_subagents() → []`, `build_skills() → []`, `build_runtime_control_policy() → {"enabled": False, ...}`, and the instruction slots are one generic line each. Their conclusion leans on this — *"even sparse initial harnesses can support useful self-improvement."* H0\* is not sparse; it ships the ~100K ceiling, the ~20-wide batch rule, and the answer-discipline sentence pre-installed.

**The resolution, as implemented.**

1. **Surfaces re-derived from loop phases, not from known failures.** Nine surfaces keyed to positions in the RLM turn loop, mapping nearly 1:1 onto Self-Harness's nine. The instruction surface split by phase (contract / decomposition / execution / verification / recovery) because Self-Harness's own results turn on that resolution — their M2.5 gain was a bootstrap-instruction edit, their Qwen3.5 gain a recovery edit. All numeric limits collapsed into one policy dict starting `{"enabled": False}`. No surface row carries a paper-evidence citation, and the U2 docstring-citation test was removed.
2. **The loop starts from H0, not H0\*.** Seven of nine H0 defaults are empty, disabled, or one generic line, with a U2 regression test asserting that property so a later edit cannot quietly pre-populate the floor. H0\* becomes a human-engineering baseline alongside H1 rather than the starting point.
3. **The paper's failures repositioned from justification to prediction.** Self-Harness had no external check on whether its mining stage found real mechanisms or plausible-sounding ones — it falls back on qualitative before/after case studies for exactly this reason. This study has an independent published catalogue of RLM failures. If they are real, a loop starting from H0 should rediscover them, so **overlap between mined clusters and the published catalogue is a reported result — convergent validity for the mining stage.** That is a methodological contribution over the work being built on, and it only holds because H0 is stripped.

**What remains.** The topology prior is still a prior; it is now defended by precedent and by claim-wording rather than instrumented away. If a reviewer presses further, option 3 from the original triage — a second optimization run over a deliberately different surface set — is still the only thing that would *bound* it, at roughly double the loop cost. Not recommended unless a reviewer demands it.

---

## RD4 — The cited failures belong to different models than the study backbone

**Raised by:** adversarial (P1, anchor 75)
**Status:** open, but reshaped by RD3's resolution — this is now the measurement, not a threat to it.

**The problem.** Every failure the RLM paper documents is attributed to a specific model: Qwen3-Coder launched thousands of sub-calls where GPT-5 launched ten; LongCoT-mini was hint-sensitive; Qwen's syntax errors compounded across depth. The proposal's backbone is Qwen3-30B-A3B-Instruct-2507 on GraphWalks and OOLONG-Pairs, and *none* of the cited observations were made on that model in those environments.

**How RD3's resolution changes this.** Under the old framing, a signature that never fired meant a *dead surface* — declared, never edited, wasted. That failure mode is gone: surfaces are now declared from loop structure, so a surface is live whether or not any published failure maps to it, exactly as Self-Harness's nine are.

What remains is better. Signature prevalence is now the **overlap measurement itself**. Non-overlap is no longer a hole in the design; it is a finding about model-specificity — this backbone, in these environments, does not exhibit the failures documented for those backbones. Self-Harness's own analysis found exactly this shape across its three backends: M2.5 needed artifact-creation discipline, Qwen3.5 needed retry discipline, GLM-5 needed environment persistence. Model-specific failure profiles are the expected result, not the anomaly.

**What is still open.** The reporting rule, which should be fixed before the data arrives rather than after:

- Report overlap as a rate — how many of the paper's documented failures the mining stage independently rediscovers for this backbone, and how many mined clusters have no published counterpart.
- Decide in advance whether high overlap or low overlap is the interesting result. **Both are publishable and they say different things**, which is exactly why the interpretation has to be registered now: high overlap validates the mining stage against an external reference; low overlap is evidence that RLM failure modes are strongly model-specific, which is itself a claim the RLM paper gestures at but never measures across an independent replication.

The pilot should report the per-signature rate for the study backbone. U1's trace metrics already make each rate computable, so this is a reporting step, not new instrumentation.

---

## Cross-cutting note

RD1 and RD4 share a shape worth naming: both are cases where the plan built a detector and left the response undefined. RD1 detects recursion collapse and does not act on it; RD4 detects overlap-or-not and has no registered interpretation. In both cases the instrumentation is the cheap part and already planned — what is missing is the preregistered decision rule, which is exactly the thing that has to be written down *before* the data exists in order to be worth anything.

RD2 and RD3 shared a different shape: both were gaps between what the plan builds and what the proposal claims. RD3 is now closed by changing the design rather than narrowing the claim. RD2 is still open and still narrowed as an interim position — the Verification Contract claims only the agent-mechanism component of ϕ until the verifier-side fields exist.
