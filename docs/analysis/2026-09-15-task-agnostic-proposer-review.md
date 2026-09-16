---
title: Why the proposer still repeats weak edits, and small task-agnostic fixes
date: 2026-09-15
type: investigation
---

# Task-agnostic proposer review

The latest experiment stalled on both proposal construction and proposal reasoning. Moving template escaping out of model output would prevent several wasted opportunities, but would not make the surviving hypothesis effective. The more consequential problem is that the proposer repeatedly recommends a familiar procedure without establishing that it changes the operation responsible for the failure.

These are recommendations, not implemented changes. No paid model calls were made for this review. The paper and experiment artifacts remain unchanged.

## Evidence and limits

Reviewed all six proposer responses and ten submitted candidate objects in `experiment_oolong_pairs_dsv4f_20260914_111506`, its local rejections, promotion ledger, mining diagnoses, and selected validation execution traces. Reconstructed all three proposer system prompts: each SHA-256 matches its saved proposal marker. Ran five synthetic checks against the admitted S2 example, plus a child-output-format probe, without model calls.

The [machine-readable audit](2026-09-15-proposer-review-audit.json) preserves the proposal summaries, prompt reconstruction checks and synthetic probe outcomes.

Compared this with the four earlier OOLONG Pairs experiments: 21 completed rounds and 21 written proposals across the five directories. Also inspected proposal descriptions and rejection ledgers from the older GraphWalks experiments `experiment_kimi` and `experiment_ox_full`, and OOLONG `experiment_oolong_dsv4f`. Earlier artifacts use different models, pools, gates and loop versions; cross-run scores are not controlled comparisons. The latest `v=1` results do not identify the causal effect of each edit reliably.

Prior investigations: [September 13 audit](oolong-pairs-2026-09-13/README.md), [September 14 methodology notes](2026-09-14-proposal-quality-methodology-notes.md). The statement in the latter that no new experiment had been run describes its writing time; this review examines the subsequent run.

## What actually happened in the latest run

The run completed three rounds with zero promotions, 140 persisted attempts and $13.51172474 recorded stage spend. One mining timeout has lower-bound usage accounting. The experiment stopped on patience, with exit code zero.

| Round | Initial response and repair | What reached paid validation |
|---|---|---|
| 1 | Both responses contained S2 plus two competing S3 edits. S2 initially failed brace formatting; repair fixed it. Repair repeated both S3 edits, so the duplicate-surface group was rejected again. | S2 alone: baseline and candidate both 1/10 exact passes; all-attempt F1 0.6257 → 0.5847; validation cost $0.8032 → $0.9907. Rejected for no exact-pass improvement. |
| 2 | S2 attempted the same record-ID/coverage direction. Both responses failed template formatting. The repair also damaged its JSON example. | Nothing. |
| 3 | S2 attempted that direction again. Repair renamed `rid` to `rec_id` instead of fixing the interpolation layer. The resulting error changed from `KeyError: 'rid'` to `KeyError: 'rec_id'`. | Nothing. |

There were six submitted S2 objects, five with template errors, and four submitted S3 objects rejected as two duplicate groups. These are object-level counts: an attempt can contain several failures. Round 1's attempt markers both say `accepted=false`, even though one repaired S2 member survived; attempt acceptance is not the usable-candidate count.

Sources: [R1 proposal marker](../../experiment_oolong_pairs_dsv4f_20260914_111506/opt/round_01/proposals_complete.json), [R2 marker](../../experiment_oolong_pairs_dsv4f_20260914_111506/opt/round_02/proposals_complete.json), [R3 marker](../../experiment_oolong_pairs_dsv4f_20260914_111506/opt/round_03/proposals_complete.json), [R1 ledger](../../experiment_oolong_pairs_dsv4f_20260914_111506/opt/round_01/validation/round_01/promotions.jsonl).

### The admitted hypothesis addressed coverage, while its evidence described semantics

R1's S2 proposal says the observed error was a misclassified question. Its proposed remedy checks record IDs and label vocabulary. Those checks can detect missing, duplicated or invalid records; they cannot distinguish a correct label from a wrong label within the allowed vocabulary.

The selected pattern's operation evidence already reports 188 collected labels for 188 parsed records. This does not prove the labels were aligned correctly or that parsing retained every input record, but neither does it establish skipped records. The proposed coverage remedy did not identify a demonstrated coverage defect.

The five synthetic checks reproduce that distinction. Correct rows pass; duplicate IDs, missing IDs and out-of-vocabulary labels fail. Swapping two labels while keeping IDs and vocabulary valid also passes. This is a limit of the check, not an implementation bug in a coverage check.

The S3 proposal for pattern 5 has an even clearer mismatch. Its own `observed_failure` says the root detected an aggregate-count response, discarded it, and obtained a per-record replacement. It then proposes detecting and replacing aggregate-count responses. The representative evidence explicitly says the replacement was used. An empty S3 instruction does not mean the observed execution lacked that behavior.

Pattern 7 identifies real pair-order/self-pair defects, but its S3 replacement prescribes combinations from one qualifying user list. That does not preserve all tasks with asymmetric roles. Its trace also tests whether a label is present when the question requires exactly one occurrence; the proposal does not address that count predicate. Fixing output shape is not equivalent to fixing selection semantics.

### The new behavior was adopted, but a later operation still failed

In held-out candidate run `oolong-t09-w10-90d877bb39360949__a01`, the root assigns record IDs, joins classifications by ID and prints:

```text
Total classifications: 188
Classified IDs: 188
Missing IDs: []
Extra IDs: []
All categories valid
```

The task requires entity-or-location membership and a date constraint. The root instead filters entity-or-abbreviation membership, discards dates when parsing records, and never applies the date constraint. Recorded F1 is 0.086, with 63 missing and 533 extra pairs. This is direct evidence that the proposed procedure can execute successfully while the task remains incorrectly implemented; it does not establish that the edit caused this semantic error.

Another candidate run, `oolong-t17-w10-74f51cb44a6e57c1__a01`, requires exactly one numeric-value record for one role. The root converts labels to sets and instead tests numeric-value AND entity membership. It has already thrown away the multiplicity required by the actual predicate. Its coverage checks cannot recover that information.

These held-out traces are used only for this external audit. They must not become instance-specific proposer evidence or repair inputs. Future live proposals should derive their examples from held-in traces or synthetic cases; held-out history remains aggregate-only.

Sources: [R1 candidate validation artifacts](../../experiment_oolong_pairs_dsv4f_20260914_111506/opt/round_01/validation/round_01/r01-c01-s2/heldout/round_00/), especially the two named runs and their `instances.jsonl` task instructions.

## Why the existing improvements were insufficient

1. **Evidence is present, but a causal label still outweighs it.** All 84 mining failure records were labeled `causal`; 64 were `incomplete_coverage`. None had child-verifier grounding. Lack of grounding alone does not disprove a diagnosis, but several leading examples explicitly describe full parsed-record coverage and uncertain classification. The closed mechanism label continues to route these examples toward a coverage intervention.

2. **The behavioral-difference fields compare text more readily than execution.** The proposer says S3 is empty, then proposes an operation that the trace already performed. A difference from the incumbent's instruction text is not necessarily a difference from the incumbent's observed behavior.

3. **The prompts are much larger than the useful distinctions.** The reconstructed system prompts contain 157,881, 169,363 and 173,945 characters. Rendered legacy produced/expected answer evidence alone consumes 35,010–36,584 characters per round, in addition to the new compact verifier diagnostics and excerpts. Twelve or thirteen patterns each receive their own trace allocation. A fixed per-pattern excerpt budget is not a fixed prompt budget.

4. **Some selected code is still cut through the relevant operation.** The selector now finds relevant iterations, which is useful, but divides space evenly over snippets and nonempty fields, then head/tail-truncates each. In R1 pattern 0 the middle of the classification/filtering block is removed. Short stdout fields leave unused space while important code is truncated. Repeated retry logs and long pair lists can take the same allocation as code.

5. **History did reach the model.** R2 and R3 prompts contain the R1 exact/F1 results and `no_measured_improvement` annotation. They also contain earlier local brace/duplicate errors. This failure cannot be explained by an absent history channel. The model nonetheless repeats the direction without a concrete revision responding to the failed evaluation.

6. **A benchmark-specific recipe remains a strong default.** `OOLONG_RECORD_GUIDANCE` supplies a detailed ID/label/coverage procedure near the end of every OOLONG Pairs proposer prompt. All six S2 submissions follow that direction. That association suggests anchoring; it is not a measured causal effect of the guidance. The recipe is plausible for some failures, but its presence makes it easy to ignore alternatives involving predicates, count preservation or already-recovered operations.

7. **Repair is not specific enough to the failed contract.** The repair receives failed source and error text, but still regenerates whole replacements while attending to the entire original prompt. It repeatedly mishandles nested JSON, Python and prompt-template syntax, or resubmits the same surface conflict.

Implementation references: [proposal prompt, rendering and repair](../../shrlm/optimization/proposal.py), [evidence selection](../../shrlm/optimization/proposal_evidence.py), [history loading](../../shrlm/experiment/orchestrator.py), [representative selection and ranking](../../shrlm/optimization/clustering.py), [template escaping](../../shrlm/rlm_harness.py).

## How this connects to previous experiments

| Experiment | Observation relevant to proposer design |
|---|---|
| Original OOLONG Pairs, six rounds | The last three rounds repeated existing surfaces and produced no challenger. More rounds were not producing more hypotheses. |
| September 11, 14:58, three rounds | An S2+S3 batch improved exact passes 2/10 → 4/10 and F1 0.784 → 0.920 but failed the former cost floor. This was a gate problem, already corrected by setting the floor to zero. |
| September 11, 21:39, five rounds | Broken S9 interfaces and invalid restrictions on valid answers consumed validation. The September 13 audit found 21 runtime failures in the last 30 candidate attempts. |
| September 13, four rounds | R1 S2+S9 promoted; R2/R4 produced no usable change. R3's lower exact pass count accompanied higher F1, so the direction deserved a qualified refinement rather than blanket dismissal. |
| September 14, three rounds | Structural preflight now prevents invalid candidates from consuming validation, and history carries dense diagnostics. However, brace failures, duplicate surfaces and repeated weak causal hypotheses still exhaust rounds. |
| Older GraphWalks / `experiment_kimi` | S8 list-formatting helpers appear in all five rounds; none promoted. Only an S4 edit promoted. S2 repeatedly pushes decomposition despite observed regressions. A later S1 edit conflicts with the actual answer contract. |
| Older OOLONG / `experiment_oolong_dsv4f` | An S9 proposal treats brackets and commas as evidence of lossy aggregation and rejects them. Other proposals treat provider filtering as a coverage/verification problem. All three candidates were rejected. |

The recurring failure is a familiar remedy attached to a broad symptom: more decomposition, more coverage checks, or stricter answer formatting. This occurs beyond OOLONG Pairs. The corresponding general fixes are to preserve the task's actual contract, identify the unresolved operation, and make rejection feedback require a changed hypothesis.

The GraphWalks and OOLONG artifacts predate the latest prompt changes; they establish recurring failure modes, not evidence that the current implementation still has every old defect.

## Recommended small changes, in order

### 1. Make serialization the host's responsibility

Have new text proposals provide literal prompt text; the materializer escapes it exactly once while preserving the explicitly supported placeholder. Render current text in the same representation and version the contract, rather than blindly applying `escape_braces` to already escaped historical surfaces. Keep the rendered-prompt check as verification; the model should not spend its repair attempt guessing which interpolation layer produced a `KeyError`.

For the remaining genuine authoring errors, return the failing snippet and all invalid placeholder names in one diagnostic. This addresses the latest five S2 formatting failures and earlier S2 failures across any task, without changing validation spend or the promotion gate. It does not certify a repaired example's semantics.

### 2. Require the proposed change to alter an unresolved operation

Refine the existing behavioral-difference fields to reference observed execution: which operation remained wrong, what value/action changes under the edit, and why the current successful checks do not already achieve that. Mark a cited failure as recovered when its result was discarded and successfully replaced; do not propose that same recovery as the final-answer fix. Where coverage or child correctness is unproven, preserve uncertainty instead of treating the mechanism enum as proof.

A useful short challenge is: **“Assume the model follows this edit perfectly. Could the demonstrated failure still happen for the same reason?”** If yes, the proposal needs a different causal claim or should be withdrawn. R1's wrong-label/coverage proposal and recovered-aggregate S3 proposal both fail this challenge as stated.

### 3. Choose the surface before writing its replacement

Ask for a brief ranked choice of evidence-backed mechanisms, then assign at most one selected mechanism to each surface before producing replacement text. A collision repair should explicitly choose one contender or withdraw both, preserving valid siblings; retarget only when another surface can address the same operation. A schema keyed by allowed surface IDs, or equivalent host-enforced surface selection, can make duplicate occupancy harder to generate without requiring more candidates or another validation arm.

This would address the duplicate S3 failures in the latest R1 and the duplicate S9 response in the September 13 R3. Selecting one candidate is preferable to joining unrelated edits into a larger surface replacement merely to satisfy the syntax.

### 4. Replace the benchmark recipe with task-derived contract questions

Keep the environment's authoritative answer contract, but remove the prescriptive OOLONG-specific solution recipe from the generic proposer. Instead ask what information each operation must retain and which conditions the task requires: identity, counts, dates, order, provenance, units or asymmetric roles. Require the proposal to name the missing condition or lost information and give a small synthetic counterexample where its new procedure behaves differently.

For example, set membership cannot implement “exactly one”; counts cannot recover a required date; joining by ID cannot validate a semantic label. These distinctions apply to graph traversal, extraction, classification and aggregation without encoding a particular benchmark's solution. This change replaces guidance rather than appending another long rule block.

### 5. Spend a total evidence budget on complete, contrasting operations

Keep a compact inventory of all patterns, but expand the strongest few distinct mechanisms under a total prompt budget. Prefer one complete relevant operation and a nearby successful/recovered contrast over numerous head/tail fragments; summarize repeated retry logs and raw answer lists, and redistribute unused snippet space. Keep the current surfaces complete, shown once, and use compact verifier-authored diagnostics plus a few examples when their interpretation is known.

In the latest prompts, legacy answer dumps alone offer roughly 35k characters of removable duplication. Unknown verifier formats still need bounded original evidence, not guessed numeric summaries. A representative should be selected for an inspectable mechanism and useful contrast, rather than merely the first instance ID among equally ungrounded records.

### 6. Make revisiting a rejected direction require a stated revision

Keep the existing measured-progress annotations, but ask the proposer to identify the closest prior attempt and the substantive change justified by its outcome. Distinguish “not evaluated because invalid,” “evaluated with no measured improvement,” and “rejected but diagnostically promising”; only the last invites a qualified refinement, not replay. Compare the proposed action, rather than only its text hash, using the existing behavioral fields and bounded review.

Do not blacklist an entire surface or direction based on one `v=1` trial. Conversely, an unchanged initial harness is not permission to re-emit the same unsuccessful intervention. The latest history had the relevant numbers; it needed to influence selection, not merely appear in the prompt.

### 7. Check executable examples with tiny, task-agnostic fixtures

Extend cheap local preflight beyond prompt formatting where an edit supplies a runnable procedure. Check that the displayed producer format matches its parser, and exercise one valid case plus a counterexample relevant to the claimed change—shuffled results, duplicate IDs, an empty legitimate answer, a count boundary, or asymmetric roles. Execute candidate code only in the existing bounded isolation, with synthetic data and no provider calls.

R1's example parses an outer JSON list but asks for a JSON array for each line; a newline-separated row response raises `JSONDecodeError`. The R2 repair produces a malformed JSON example even apart from the remaining template error. Fixtures can catch these interface inconsistencies, but model-generated fixtures are not independent proof of classification correctness; keep coverage and semantic claims separate.

## What I would do first, and what to measure

Start with host-owned text escaping and explicit collision selection, then the unresolved-operation challenge and compact evidence. Replace the benchmark-specific recipe as part of that prompt cleanup. Preserve the existing held-out-only combined decision, one edit per surface, cost band and `v=1`.

Before another full experiment, exercise the proposer on saved held-in bundles from OOLONG Pairs, GraphWalks and OOLONG under a fixed proposal-call budget. This needs only proposal calls and local checks, not new mining or held-out executions. Track usable candidates per call, repeated failure reasons, duplicate surfaces, agreement between claimed behavior and cited operations, and whether renewed proposals respond substantively to history. Include a few manually checked counterexamples so an eloquent but irrelevant explanation cannot pass the review.

Do not judge success only by more promotions or longer runs: invalid proposals being blocked is an improvement, and repeated run scores vary. The immediate target is more valid, distinct and evidence-supported interventions reaching the same unchanged validation protocol. Extending patience or weakening promotion would not resolve the demonstrated proposer failures.
