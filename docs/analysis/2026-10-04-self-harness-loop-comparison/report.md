# Self-Harness versus our RLM optimization loop

Reviewed October 4, 2026. This is a source comparison and recommendation record, not an implementation plan or a claim that the suggestions improve benchmark performance.

**The strongest additional changes are to preserve diagnosis details, allow a capability and its caller to be proposed together, and connect proposal hypotheses to named held-in regression cases.** Explicitly recording whether a cited defect was recovered is also worth borrowing. Our existing RLM execution model, surface boundaries, evidence provenance, and combined validation can remain.

## Scope and evidence

- Our checkout: `e323f4328e0aa37b2cb5ff67cabd4729d1b0916c`, branch `main`.
- Cloned `Self-Harness/`: `2720dbb3f52283684f4b85a1065d642df1779dd8`, branch `main`, clean working tree.
- Compared diagnosis, clustering, proposer contracts, materialization, candidate admission, evaluation, acceptance, and iteration/history. Runtime code was consulted only to understand editable-hook activation.
- Root `AGENTS.md` and the user-provided sequential tool mapping govern this review. The correctness, evidence, testing, and simplicity checks were performed sequentially by one reviewer. No independent or cross-model review ran: this is a two-repository source comparison, not the Git-diff scope supported by the skill's peer runner.
- Existing experiment-artifact deletions and unrelated work were excluded. No optimizer, configuration, manuscript, or cloned-repository code was changed. Only this review directory was added.
- The public clone is a small scaffold, not a complete record of the authors' experimental automation. Its workflow accepts externally supplied diagnosis/proposer responses or commands. Conclusions below distinguish functions available in the library from the actual public CLI path.

The [earlier transfer review](../2026-10-01-promoted-harness-transfer-review/report.md) supplies empirical context: original versus promoted short-test performance was 8/30 versus 7/30 exact passes, mean F1 0.7866 versus 0.7828, with higher cost. Its task-level and trace findings motivate these recommendations; the synthetic checks in this directory establish code behavior only.

## What actually differs

| Stage | Public Self-Harness clone | Our current loop | Assessment |
|---|---|---|---|
| Diagnosis | Failed traces normalized into stages; LLM supplies concrete terminal cause, mechanism, criticality, recovery status, and reasoning. | Recursive trace digest; fixed verifier cause; closed mechanism/status vocabulary; operation citations and verification limits. | Preserve RLM trees and verifier authority. Borrow concrete reasoning and explicit recovery status. |
| Clustering | Exact match on LLM-generated terminal cause / criticality / mechanism; root causes precede recovered friction. | Exact match on verifier cause / failing level / causal status / mechanism; distinct-instance support precedes actionability. | Our vocabulary avoids synonym fragmentation, but `other` can combine distinct mechanisms. Do not copy free-form clustering wholesale. |
| Proposal input | Integrated brief includes representative causal reasoning and passing case IDs. | Budgeted operation evidence, passing IDs/optional contrast, incumbent surfaces, revision history. Three saved detail fields disappear during evidence construction. | A concrete information-loss fix exists before adding more instructions. |
| Proposed intervention | One virtual hook alias; some aliases change several underlying functions, including capability plus call policy. | One surface per edit **and one edit per failure pattern**; all admitted edits form one evaluated batch. | Keep surface uniqueness. Permit narrowly coordinated edits for one mechanism when activation requires them. |
| Regression intent | Prompt names `expected_affected_cases`, `protected_passing_cases`, and `regression_guard`. | Pattern membership and textual behavior/risk explanations; no equivalent structured selection of targeted/protected cases. | Name the held-in cases and compare their outcomes explicitly. |
| Validation | Each candidate on train and heldout, normally two repeats; merge accepted candidates and evaluate the merge again. | Fresh incumbent and combined candidate, heldout only, one repeat in the current profile. | Missing held-in evidence remains the largest known gap. Individual candidate tournaments are optional, costly, and not required by RLMs. |
| Acceptance | Neither split's aggregate pass rate may fall; at least one must rise. | Heldout exact-pass delta must meet configured minimum; zero permits a tie; resource bands also apply. | Deliberate policy differences, not implementation bugs. F1 need not become a universal gate. |
| Feedback and continuity | Queue/branch artifacts; parent results reused; external commands own much of the iteration. | Explicit bounded history, qualified dense-metric progress, revision references, duplicate-evaluation guards, checkpoint identities. | Much of our history and operational machinery is already stronger. |

## Prioritized recommendations

### R1 — Preserve the concrete diagnosis already paid for

**Priority: P2 correction; high confidence in the data loss. Expected performance benefit remains unmeasured.**

Our attributor saves `agent_mechanism_detail`, `causal_status_detail`, and `failing_level_detail` in [attribution.py](../../../shrlm/optimization/attribution.py), lines 655–664. However, [load_proposal_evidence](../../../shrlm/optimization/proposal_evidence.py), lines 769–788, selects the symptom, operation evidence, limits, and verifier data while omitting all three fields. The offline probe reproduces their disappearance. A symptom or operation observation might repeat the explanation, but the contract does not guarantee that.

This matters especially for `other`: the concrete mechanism can exist only in `agent_mechanism_detail`, while the proposer sees the generic enum. Our [clustering code](../../../shrlm/optimization/clustering.py), lines 217–255, also groups matching enum signatures regardless of their free-text explanations. Several different unsupported mechanisms can therefore share one broad cluster. This does not make every `other` pattern unusable: causal/contributing `other` patterns can have eligible surfaces, while unattributed ones are withheld.

Upstream explicitly forwards a representative analysis item's `reasoning` as the cluster's “Shared diagnosis” in [integrated.py](../../../Self-Harness/diagnosis/src/self_harness_diagnosis/integrated.py), lines 145–148 and 211–225.

**Small change:** include bounded, clearly model-attributed mechanism and causal-detail text in each admitted representative packet. For a broad `other` cluster, tell the proposer that its support count is support for the enum bucket, not proof that every instance shares that representative's precise mechanism. Keep the verifier label and closed taxonomy; no new taxonomy generator, clustering model, or extra LLM call is needed.

Acceptance check for an implementation: a saved diagnosis whose only concrete mechanism is in `agent_mechanism_detail` must deliver that explanation to the proposer within the existing evidence budget. Do not silently replace the symptom, code evidence, or uncertainty with it.

### R2 — Permit a capability and its necessary invocation to form one intervention

**Priority: P2 design recommendation; high confidence that current admission blocks it.**

Upstream's single `subagent_call_policy` alias changes both `build_subagents` and `build_verification_instruction`, installing the capability and a parent instruction that invokes it. See [hooks.py](../../../Self-Harness/proposer/src/self_harness_proposer/hooks.py), lines 129–145 and 202–214. The offline probe confirms both functions change under one alias. This is an optimization-interface idea; the Deep Agents `task` tool itself is irrelevant to our RLM.

Our host rejects a later candidate when either its surface **or its pattern** already has an owner: [proposal.py](../../../shrlm/optimization/proposal.py), lines 2045–2066. Our probe submits valid S3 and S4 edits for one pattern; S3 is admitted and S4 is refused even though S4 is unused. The prompt also explicitly removes each selected pattern from consideration.

That extra pattern restriction can block a useful RLM intervention: an S8 helper plus a short S3 instruction invoking it, or an S10 procedure plus a necessary S2/S3 invocation condition. Existing helper discovery and skill descriptions sometimes suffice, so these pairs should not be mandatory. But when they do not suffice, the optimizer currently needs a second failure pattern or a later round to connect the pieces.

This has empirical relevance: the prior review found that the S10 skill promoted in round 7 had **zero recorded loads in its candidate validation**, and none at the root in the final short test. That does not prove a call instruction would improve results; it establishes that the skill's intended intervention was not demonstrated by its promotion.

**Small change:** retain one edit per surface, but allow a capability edit and one necessary caller edit to share the same evidenced pattern and joint hypothesis. Reuse the existing combined evaluation. If either member fails preflight or repair, withdraw the dependent pair together; do not leave a caller referencing a missing helper. This is a bounded exception, not a general dependency graph or permission to spray the same advice across surfaces.

Measure activation separately from outcome. Existing S10 load telemetry helps; S8 invocation is currently not assessed by [activation_for_surface](../../../shrlm/optimization/behavior.py), lines 88–106. A loaded or called capability still does not establish correct execution. Avoid promoting “more surfaces changed” as a goal by itself.

### R3 — Make the regression hypothesis identify cases, then report gains and losses separately

**Priority: P2 design recommendation, to accompany restoration of held-in validation.**

Upstream's [proposal schema](../../../Self-Harness/proposer/src/self_harness_proposer/multi_proposer.py), lines 127–131, requests `expected_affected_cases`, `protected_passing_cases`, and `regression_guard`. Its integrated brief explicitly calls passing cases regression tests ([integrated.py](../../../Self-Harness/diagnosis/src/self_harness_diagnosis/integrated.py), lines 86–105).

Our proposal identifies a pattern and describes `predicted_effect` and `regression_risks`, but does not identify a specific tested target/protected subset. Pattern membership provides an available population; it does not establish that a proposed mechanism applies to every member. Validation history primarily exposes aggregate outcomes ([proposal_evidence.py](../../../shrlm/optimization/proposal_evidence.py), lines 943–979), so it cannot currently show whether the edit fixed its nominated cases or merely changed which tasks happened to pass.

**Small change:** attach held-in target case IDs and a few relevant passing case IDs to each proposal, validate those IDs against the mining manifest, and freeze that list before evaluation. Score with the existing verifier and report per-case attempt counts, gains, losses, unchanged failures, and comparable dense-metric changes. The target manifest can drive the cheap initial regression stage; complete held-in/heldout validation remains the broader gate. Candidate executions are batch executions, so these reports do not assign causal credit to individual edits.

Passing evidence already exists in our proposer. Improve its allocation rather than adding another large prompt section: prioritize a successful mining attempt of the same target instance when available, otherwise the nearest relevant operation. The current packer adds optional context before attempting one passing contrast, matching operation names and taking the first that fits ([proposal_evidence.py](../../../shrlm/optimization/proposal_evidence.py), lines 632–660). Reserve room for a useful contrast when one exists, within the same budget. A passing final answer does not independently verify every intermediate step.

**Important upstream limitation:** these named fields are prompt metadata, not independently enforced regression tests. The upstream gate compares aggregate split pass rates; a loss on one previously passing task can be offset by a gain on another. Our extension would make that tradeoff visible, not merely reproduce their field names. With stochastic attempts, a single protected-case failure is not automatically proof of regression; evaluate repeated outcomes under a declared rule rather than imposing an accidental pass-every-attempt gate.

Keep all proposer-visible task-level feedback on the held-in side. Heldout task payloads and traces must remain out of diagnosis and proposal prompts. The separate short test must not become the new training regression bank because we have inspected its outcomes.

### R4 — Record recovery status explicitly, instead of leaving it only in prose

**Priority: P2 design recommendation; moderate confidence in benefit.**

Upstream asks for `selected_step_recovered`, a terminal-link assessment, and a criticality that distinguishes `recovered_friction` from root cause/contributor. Its integrated clustering ranks recovered friction last and says recovery alone is not a repair target without terminal linkage: [trace.py](../../../Self-Harness/diagnosis/src/self_harness_diagnosis/trace.py), lines 342–355; [integrated.py](../../../Self-Harness/diagnosis/src/self_harness_diagnosis/integrated.py), lines 179–190.

Our prompt already tells diagnosis and proposal to consider later repairs, and the evidence selector includes follow-on code. This is not a missing admonition. However, recovery remains embedded in `symptom_summary`, `verification_limits`, and snippets; ordinary diagnoses lack a dedicated resolution value consumed by prioritization. `coverage_basis` provides a special structured check for incomplete-coverage claims, not a general record of whether the nominated defect remains live.

The prior transfer review found this exact failure mode: round 1's supposed lost-chunk problem was repaired before the final answer, and round 4's mixed-ID conversion was already present. The optimizer nevertheless spent edits on those explanations.

**Small change:** add a single `unresolved | recovered | unknown` assessment for the cited defect, supported by existing operation references. Recovered items should be contrast evidence by default. Allow a proposal only when it identifies a separate residual consequence, such as repeated recovery consuming the time budget. Preserve honest uncertainty; do not force an unsupported causal conclusion simply to obtain an actionable label. This is not a replacement for replaying the relevant tasks.

Do not copy upstream's full set of overlapping `criticality`, `causal_weight`, and `terminal_link` fields or its stage graph. One resolution field alongside our existing causal status is enough to test this idea. Upstream's labels are also model judgments, not verified causality.

## The previously identified regression gap remains first priority

The upstream gate defaults to train plus heldout and two repeats in [run_acceptance_gate.py](../../../Self-Harness/acceptance/scripts/run_acceptance_gate.py), lines 11–12 and 79–121. It requires nondeclining aggregate performance on both splits and improvement on at least one. It does not synthesize a new executable oracle from each diagnosis.

Our [ValidationSplits.evaluation_items](../../../shrlm/optimization/validation.py), lines 130–132, returns only heldout. The current profile has 20 held-in mining cases, 10 heldout cases, and `v = 1`. Restoring regression executions on held-in cases and using the user's preferred three repeats is the direct remedy; three is our proposed setting, not an upstream default. Use the fixed benchmark verifier. Extra generated examples can probe deterministic code, but the proposing model must not become the authority for its own task answers.

Our inclusive exact-pass rule and diagnostic-only dense metrics are explicit prior decisions. Upstream rejects a tie on both splits; adopting that policy would be a separate change, not a hidden consequence of restoring held-in evaluation. Per-case visibility likewise does not automatically require a new per-case veto. These policies should be stated explicitly when planning the regression change.

## Differences to preserve, and upstream behavior not to copy

1. **Keep recursive trace reconstruction and fixed verifier outcomes.** TerminalBench's tool-message stages cannot substitute for root/child call trees, bounded REPL observations, and RLM reach restrictions. An answer-only hook still cannot inspect arbitrary records or verify child semantics.
2. **Keep the combined batch if cost is the chosen constraint.** The upstream workflow evaluates each candidate, then evaluates their merge ([workflow](../../../Self-Harness/workflow/scripts/run_self_harness_loop.py), lines 440–599). This gives individual evidence but increases spend. Our batch cannot establish individual contribution, an already accepted tradeoff. Nothing here requires restoring a tournament.
3. **Keep current-surface visibility.** Upstream maintains separate evaluation and proposer surface views. In `create_child_branch`, a prompt candidate retains the parent's proposer surfaces; `build_proposer_surfaces_for_non_prompt_items` copies only non-prompt changes into that view (workflow lines 615–651 and 740–776). Whatever the intended experimental rationale, this is a poor default for avoiding redundant edits. Our serialized incumbent shown to the proposer is useful.
4. **Keep fresh incumbent validation when estimating uncertain differences.** Upstream stores the accepted candidate's evaluation as the next branch's baseline and reuses it. That saves cost, but a selected favorable score can become a difficult or misleading reference. Our previous review directly observed identical harnesses moving from 3/10 to 0/10 and 5/10 to 3/10 on repeat evaluation. Do not copy cached promotion-time scores as though they were fresh evidence.
5. **Keep our materialization and replay safeguards.** Literal-text encoding, surface-diff checks, subprocess preflight, caps, manifest identities, valid-sibling retention, and duplicate-batch detection address real RLM/operational requirements. They establish admissibility and reproducibility, not semantic correctness.
6. **Keep our useful history.** Our explicit revision records and verifier-defined dense-metric progress already exceed the public workflow's built-in feedback. The public CLI accepts external proposer/diagnosis commands; their unspecified behavior cannot be credited as an upstream feature.
7. **Do not infer stronger proposer agency from the clone.** Its library uses direct `llm.invoke`; the public workflow invokes external response-generation commands. Our proposer uses a direct completion too. The code does not establish that their proposer runs a tool-using RLM or tests arbitrary patches interactively. The recursive harness is the evaluated agent, not automatically the optimizer itself.
8. **Do not copy prompt-enforced diversity as though it were a host guarantee.** Upstream's library can generate sequential slots and reject duplicate route signatures. The public workflow instead builds a multi-proposal prompt and parses it with `require_one=False`; it does not call that sequential generator. Our one-edit-per-surface host enforcement is stronger for the selected batch design.

## Validation and suggested order

Ran the existing clustering, attribution, proposal, proposal-evidence, and batch-validation tests: **310 passed in 70.44 seconds**. No paid model calls or experiments were run.

The accompanying [offline reproduction script](reproduce.py) confirms:

- Upstream rejects both-split ties, accepts train improvement with heldout tied, and rejects train regression despite heldout improvement.
- One upstream `subagent_call_policy` alias changes both capability and caller functions.
- Our host admits only one of two valid edits on different surfaces for the same pattern.
- Three concrete attribution-detail fields disappear from our constructed proposer context.

See [checks.json](checks.json) for the machine-readable output. These are regression probes of current behavior, not new production tests or estimates of improvement.

Implement the missing held-in regression stage/repeats first. Alongside it, R1 is the smallest direct correction, and R3 makes the new evaluations interpretable. R2 is the most promising targeted change for effective S8/S10 interventions; R4 is a small diagnosis experiment if repaired errors continue to dominate proposals. Evaluate success by repaired held-in cases, preserved passing behavior, demonstrated activation, and independent heldout performance—not by proposal acceptance or promotion count alone.
