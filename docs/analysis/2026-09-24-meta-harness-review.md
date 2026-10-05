---
title: Meta-harness review after the September 23 experiment
date: 2026-09-24
type: investigation
status: recommendations-only
---

# Meta-harness review after the September 23 experiment

The most useful next improvement is to make proposals resolve a demonstrated, still-unresolved operation. The current optimizer is better at retaining valid edits, but it still turns partial evidence into confident causal stories, spends repairs reconstructing bookkeeping, and repeats procedures that the execution already performed. Adding more coverage instructions or encouraging more surfaces would not address those problems.

The experiment has been stopped. This report recommends bounded, task-agnostic changes; it does not implement them, change the paper, or authorize another experiment. The [machine-readable audit](2026-09-24-meta-harness-review-audit.json) contains all 21 submitted candidate objects, selections, admission outcomes, diagnosis refusal counts, validation aggregates, and prompt hashes.

## Experiment outcome and scope

Experiment: `experiment_oolong_pairs_dsv4f_20260923_163355`, frozen source `99812551d52a7b64d45cfffadcc5f78bb6fe9f55`. Stopped at **2026-09-24 13:41:04 UTC / 08:41 Chicago**. The supervisor and run-worker process groups exited; the final audit reports no live experiment process. Monitoring is ended.

Five rounds completed. Round 6 stopped during mining, after six of forty attempts; it is an interrupted round, not a failed proposal round. There are **266 persisted attempts**, **five admitted edits**, **three evaluated candidate batches**, **one promoted batch**, and **$24.9670 recorded spend**, a lower bound excluding unpersisted in-flight usage. Run and stage accounting overlap and must not be added together.

| Round | Admitted edits | Fresh held-out exact, baseline -> candidate | Mean F1, baseline -> candidate | Outcome |
|---|---|---|---|---|
| 1 | S3 + S4 | 3/10 -> 2/10 | 0.6063 -> 0.7975 | Rejected; qualified positive dense-quality signal |
| 2 | S3 | 3/10 -> 0/10 | 0.7120 -> 0.6064 | Rejected; no measured improvement |
| 3 | S2 + S4 | 1/10 -> 3/10 | 0.5256 -> 0.6656 | Promoted together |
| 4 | None | Not evaluated | Not evaluated | No candidate survived local gates |
| 5 | None | Not evaluated | Not evaluated | Identity and revision failures repeated |
| 6 | Mining only | Not evaluated | Not evaluated | User stopped; 2/6 mining exact, mean F1 0.6943 |

Validation used ten held-out tasks, one attempt per task, and the combined batch. The baseline was freshly measured each round. These rows are not a cumulative quality curve, and the round-3 result cannot identify which edit helped. No independent final-test evaluation ran. F1 includes verifier-declared zeros for terminal/unparseable outcomes; unknown metrics must not silently become zero on other tasks.

This is a qualitative synthesis of experiments and a source review, not a pooled statistical estimate. Earlier experiments used different optimizer versions, configurations, and stochastic executions. This review compares failure mechanisms across them without treating score differences as controlled treatment effects.

## What the recent changes accomplished

- **Surface ownership works at admission.** Round 1 retained S3 while repairing another contender into S4; round 3 retained S2 while repairing the failed S3 slot into S4. No duplicate-surface validation batch was admitted. Round 4's repeated selections never became valid owners, so its failure is not evidence that occupancy enforcement stopped working.
- **Evidence became more diverse, but not reliably.** Two actionable mechanisms received expanded evidence in rounds 1, 2, 3, and 5. Round 4 had three available mechanisms but expanded only one. Initial system prompts were approximately 49,000-59,000 characters, substantially smaller than the 158,000-174,000-character prompts in the September 14 experiment.
- **Dense-quality history is useful.** Round 1's F1 gain survives its exact-match rejection. This is the right distinction, as were the earlier experiment's rejected round-6 gain of 0.6194 -> 0.7193 and round-8 gain of 0.6128 -> 0.6951. None establishes individual-edit causality or repeatability.
- **Some uncertainty is preserved.** Coverage hypotheses declared unestablished or contradicted are downgraded. Invalid citations are refused. One run-level provider/resource failure did not crash the experiment.
- **Eligibility alone has not diversified successful edits.** Supported S8 routes appeared in rounds 1, 2, 3, and 5; none was selected. No supported S5 route appeared in those packets. This does not justify a surface quota or establish that either route is defective.

The preceding stopped experiment, `experiment_oolong_pairs_dsv4f_20260923_123738`, rejected all round-1 contenders after repeated surface collisions and performed no validation. The new run demonstrates a mechanical improvement over that failure. It does not establish a causal performance improvement across experiments.

## What the promoted and rejected proposals actually tell us

### Promoted round 3: a better batch score did not validate its explanations

The S2 proposal blamed initial input parsing. Its cited held-in trace actually parsed all **188 records** and sent all of them to children. Loss happened afterward: two child replies failed JSON parsing, leaving **120 classifications**. The edit compared nonempty context lines with all context lines and, on a mismatch, repeated the identical parsing expression. Blank lines explain the count difference; repeating the expression cannot recover the missing classifications. Even perfect compliance leaves the demonstrated failure intact.

The S4 proposal claimed a failed dictionary unpack left classifications missing. The same trace subsequently repaired the loop with `.items()` and printed **188 classifications, missing 0**. That repair and its output were present in the exact proposer prompt. The requested coverage check was already satisfied before final computation. The remaining answer error was not explained by the recovered fault.

S4 might help other executions with genuine child-reply loss, and stochastic variation or interaction between edits could also contribute to the batch result. Those are possibilities, not established explanations. The batch cost was 1.392x its baseline. Exact edits and decisive operations are preserved in the [round-3 assessment](../../experiment_oolong_pairs_dsv4f_20260923_163355/round-03-assessment.md).

### Rejected round 1: useful partial progress, incomplete proposed behavior

The S3 instruction normalized pair order and deduplicated, but its explanation also promised to eliminate self-pairs. Sorting and deduplication do not remove `(a, a)`. A deterministic check of the saved held-in answer improved F1 from 0.182 to 0.667 after normalization/deduplication; explicitly removing self-pairs increased it to 0.688. This is a transformation of a saved answer, not an evaluated harness intervention.

S4 addressed continuing after child-output parsing loss. The combined batch improved mean F1 by **0.1912**, while losing one exact pass. Preserve that as a qualified direction worth revising; do not declare the individual edits effective or the direction exhausted. See the [round-1 assessment](../../experiment_oolong_pairs_dsv4f_20260923_163355/round-01-assessment.md).

### Rejected round 2: the proposed merge already existed and assumed the wrong contract

The diagnosis said the root printed child results without merging. The saved trace parses child classifications, groups records, applies predicates, builds normalized non-self pairs, and formats the answer. Both the attributor digest and proposer evidence omitted those operations. The S3 replacement then asked the root to collect **pairs** from children that actually returned **classifications**. This repeats existing behavior while assuming a different interface, and supplies no demonstrated fix for the answer errors. The candidate regressed both metrics. See the [round-2 assessment](../../experiment_oolong_pairs_dsv4f_20260923_163355/round-02-assessment.md).

### Unadmitted rounds 4 and 5: bookkeeping failure and weak hypotheses are separate

In both rounds, selections cited another pattern's operations and revisions failed their identity requirements. Neither round reached validation; their drafts are **untested**, not experimentally disproven.

Their explanations still reveal problems. Round 4 called a child return incomplete by comparing **50 questions** with roughly **10 user IDs visible in a preview**. The full return contained the expected **26 users**, with counts summing to all 50 questions. Round 5 claimed a user was never classified, although both relevant children returned that user and the root checked the combined count.

Another round-5 draft would force the final answer to stay within the first child findings, even after reclassification. The root's saved intermediate answer had F1 **0.889**, nine missing pairs and no extras; its final revised answer had F1 **0.900**, no missing pairs and ten extras. Later work recovered valid findings while introducing other errors. Freezing the initial model output would prevent those corrections. The general requirement is provenance and task-supported adjudication of disagreements, not treating an earlier child output as truth.

Exact unadmitted replacements and counterevidence: [round 4](../../experiment_oolong_pairs_dsv4f_20260923_163355/round-04-assessment.md), [round 5](../../experiment_oolong_pairs_dsv4f_20260923_163355/round-05-assessment.md).

## Source review findings

Priorities below describe concrete code/contract problems. The more ambitious reasoning improvements in the next section remain hypotheses to evaluate. Source locations refer to frozen commit `99812551`.

| # | Priority | Location | Confirmed problem and consequence |
|---|---|---|---|
| F1 | P1 | `proposal.py:741`, especially 751-793 | Whole-round history packing prioritizes an old promising result over recent outcomes. Rounds 3-5 show only round 1, while admission still requires references to omitted attempts. |
| F2 | P1 | `proposal_evidence.py:298`, `digest.py:363` | Normal consumers sort after sibling child returns; six-snippet truncation can remove the actual merge. Any stderr gets digest priority, including repeated recovered rate-limit messages. Round 2 consequently loses decisive root code at both stages. |
| F3 | P2 | `proposal_evidence.py:437`, 498-550 | Every linked sibling return can be part of the first-pass core. Round 4 spends over 8,000 characters on three similar siblings and cannot fit another mechanism, despite unused budget. |
| F4 | P2 | `proposal.py:1804`, especially 1821 | Ordinary routes accept empty evidence references. Inventory-only patterns can reach later gates, while foreign-reference repairs do not give a direct selectable identity mapping. Rounds 2, 4, and 5 repeat incompatible selections. |
| F5 | P2 | `history.py:55`, especially 67 | The latest related attempt can be a refused rewrite of an already retained candidate. This is presented as the required predecessor without clearly relating both statuses. Round 2 confuses the refused rewrite with the evaluated edit. |
| F6 | P2 | `attribution.py:131`, 148-161 | `coverage_basis` is mandatory in prose for incomplete coverage but absent from the response example. Missing-field refusal occurred 155 times in 303 responses. |
| F7 | P2 | `tests/optimization/test_proposal_evidence.py:530`, fixture at 650 | A shallow copy shares `PATTERN_TEXT['signature']`; the legacy-coverage test mutates it. Later proposer tests expect skipped verification but receive incomplete coverage. |

F1, F2's snippet ordering, F4, and F5 are in `0298b1ec`; F3, the coverage contract in F6, and F7 are in `99812551`. The digest priority mechanism changed in this two-commit scope as well. These are source-supported findings, not hypothetical model failures.

Two additional limitations matter without being deterministic admission bugs:

- `attribution.py:530-665` checks the **shape and location** of a coverage declaration. A model can supply `observed_loss` with a resolvable operation and still invent the population loss. The existing tests establish declared-status normalization, not the truth of model-authored observations.
- `behavior.py:107` has event detectors for S6/S9/S10, not general instruction compliance. All five admitted edits in this experiment have activation `not_assessed`. That is honest, but neither history nor quality scores currently establish that the intended S2/S3/S4 behavior occurred.

## Recommended next changes

These fit the existing mining/proposal/repair calls. No new critic, additional validation repeat, benchmark parser, or optimizer redesign is needed.

### 1. Fix schema examples and make choices copyable

Show the actual conditional `coverage_basis` shape next to the response example, including a valid uncertain case. Before proposal text, supply one compact host-generated table of selectable pattern IDs, admitted operation references, eligible free surfaces, and related prior-attempt IDs; repairs should repeat the exact relevant row and identity fields. Make inventory-only patterns explicitly unselectable for this evidence-based contract, and require nonempty admitted references on ordinary routes too.

This removes syntax/identity work the host already knows how to do. Do not silently substitute another pattern for the model, and do not drop evidence requirements to increase the usable-edit count. Keep first-valid surface ownership and the existing repair allowance.

**Offline check:** a truncated inventory pattern cannot authorize an edit with `[]`; a supported pattern can; a wrong-reference repair displays the valid mapping while preserving admitted siblings. Add a schema-example fixture that exercises both established and unestablished coverage.

### 2. Guarantee compact recent history before expanding promising history

Reserve a small row for each recent outcome, the promotion that produced the incumbent, and any predecessor required for a currently eligible intervention. Include exact outcome, verifier-defined dense-quality delta, activation status, and whether the record was evaluated, bundled, or locally refused; expand older promising examples only afterward. Link a refused rewrite to its retained candidate so neither replaces the other's measured status.

The current system has the complete archive but withholds the part the model needs until repair. Compacting repeated metric definitions and baseline objects is preferable to dropping whole recent rounds. Keep the positive F1 signals above, but never infer that an unchanged component caused a batch improvement. Tasks without a comparable verifier-defined quality metric remain `not_assessed`.

**Offline check:** under the current 12,000-character cap, round 1's qualified positive signal, round 2's rejection, and round 3's promotion are all visible in the round-4 history. A refusal to overwrite an occupied slot is not reported as rejection of the retained intervention.

### 3. Make the first evidence allocation a complete cause-and-result example

For each distinct mechanism, reserve a complete relevant operation, its producer/consumer relationship, and the observed result or later repair before showing additional sibling returns. Summarize repeated recovered transport warnings once; preserve terminal provider failures separately. Make sibling payloads optional, and explicitly distinguish an operation omitted from the packet from an operation absent in the trace.

The fix is not a larger prompt. Round 4 needs a smaller first-pass packet; round 2 needs the consumer before extra producers. When a claimed absence cannot be checked in the bounded packet, retain uncertainty instead of endorsing the absence claim. Structural adjacency or shared variable names can select a candidate operation, but are not proof of semantic dependence or successful recovery.

**Offline check:** a trace with five child returns and a later merge still shows the merge. Repeated recovered warnings cannot evict final computation. At least two mechanism examples fit in the same 32,000-character budget when their minimum complete packets fit together.

### 4. Diagnose the latest state at the actual failure boundary

Refine the current evidence fields to identify the unit being tracked, the transformation boundary, the observed discrepancy, and the latest state after recovery. Include compact host-computed structural summaries from complete parseable payloads before preview truncation; for unsupported representations, say unknown. Only compare like units, and keep label/semantic correctness separate from structural coverage.

For example, the host can honestly report that a JSON object has 26 top-level keys; it cannot infer that those keys are the right users without task/input evidence. Counts alone also cannot prove identity preservation. A coverage hypothesis contradicted by a later successful check should remain unestablished unless it identifies a different unresolved boundary. Preventing an already recovered error may be an efficiency hypothesis, but must be stated and measured that way.

**Suggested diagnosis wording:** "At which operation does the observed state first diverge from the required state? Name the compared unit and scope. Cite the latest observation of that state, including recovery. If the packet does not establish the discrepancy, say so; do not substitute answer errors for an input-loss witness."

**Offline check:** document pages versus entities, table rows versus grouped keys, and graph edges versus vertices are distinct scopes. A complete return shown only in preview is not called incomplete. A later repaired result is included as counterevidence, while a genuinely unrepaired parse failure remains actionable.

### 5. Require a minimal behavioral counterexample, using the existing explanation fields

Make `incumbent_behavior` describe the latest execution rather than an empty instruction surface. Make `observed_failure` name the remaining violated property, and `behavioral_change` say what result changes for a minimal input exhibiting it. Reject justifications based only on rewording, a different mechanism label, or doing something the trace already did; keep explicit joint-batch hypotheses possible.

This is stronger than asking whether an edit sounds helpful. Examples of useful counterexamples are a wrong-but-valid classification passing a coverage check, a duplicate surviving a count-only check, an identical reparse leaving state unchanged, or a set losing multiplicity needed by a predicate. The property must come from the task and evidence; do not hardcode those predicates into the optimizer. Where an edit supplies executable deterministic logic, exercise a small synthetic case locally; prose claims still require evaluation and must not be marked host-verified.

**Suggested proposer wording:** "On the shown execution, what observed value or operation would differ if this edit were followed? Give the smallest case that distinguishes the old behavior from the new one, and one already-correct behavior it preserves. If the same failure survives for the same reason, revise or withdraw the proposal."

For semantic disagreements, preserve the connection to the original input and re-evaluate the disputed decision. Neither the first child answer nor the latest rerun is authoritative merely because it exists. This guards against the round-5 proposal that would prohibit valid corrections.

**Offline check:** the promoted S2's identical reparse is not described as a corrective transformation; sorting/deduplication does not claim to remove self-pairs; a recovery already completed is not represented as missing behavior. These checks target hypothesis quality, not score inflation.

### 6. Choose surfaces from the operation that must change

Use the existing capability descriptions in the selection table to compare the strongest justified intervention with another eligible surface before writing the replacement. A repeated deterministic parsing/joining/checking operation can justify S8; a visible unrecovered error branch can justify S5; missing visible information can justify S7; a recurring procedure may justify S10. State the intended call site, available inputs, return contract, and necessary invocation instructions for executable helpers.

This should be one short selection rationale within the current call, not a second selection model or a quota. A helper that is available but never called is not an effective edit. A runtime-policy edit must target a trigger that actually exists; the earlier S6 proposals promised timeout recovery while configuring syntax retries, with no positive retry counts recorded. S9 should remain restricted to defects visible to its answer-level interface.

**Suggested selection wording:** "What operation needs to change, and which eligible surface can cause that change with the required information? If a deterministic helper or recovery hook is eligible, explain briefly why it is or is not more direct than another instruction. Do not select a surface merely because it is unused."

Task-specific *harness edits derived from the current task* are compatible with a task-agnostic optimizer. Generic meta-prompts and host gates should reason about interfaces, information preservation, failure states, and evidence; they should not contain an OOLONG record format, expected answer IDs, category list, or pair-enumeration recipe.

## Why this is the next step, rather than another coverage rule

The [September 15 review](2026-09-15-task-agnostic-proposer-review.md) already found complete ID coverage alongside incorrect labels/predicates, and proposals to redo an already completed recovery. The [earlier experiment report](oolong-pairs-2026-09-23/report.md) found conflicting child-output contracts, repeated coverage directions, and S6 explanations that exceeded runtime capabilities. This run reproduces those reasoning failures even after literal-text materialization, capability descriptions, diverse evidence, and behavioral-difference fields were added.

The new evidence narrows the diagnosis. Some failures need better packet construction; others happen despite visible counterevidence. Therefore another general admonition is insufficient by itself. Make the supporting state, latest recovery, valid identities, and required predecessor visible in compact structured form, then demand the small behavioral distinction in the existing fields.

The two empty rounds spent **$8.3097 on mining alone**. Their failed admissions were avoidable bookkeeping problems, but fixing bookkeeping alone could simply send the weak hypotheses described above into paid validation. Admission quality and causal quality must improve together.

## Validation and review limits

Reviewed all ten proposal responses and 21 candidate objects in completed rounds 1-5; all five admitted edits; all local failure records; validation aggregates; the five reconstructed initial proposer prompts, each matching its sealed SHA256; and the decisive full held-in operations cited in the round assessments. Earlier reports provide cross-experiment corroboration. This is a sequential main-thread review under the repository's AGENTS mapping, with no independent agent or cross-model review.

The reviewed source scope is the optimizer changes in `0298b1ec` and `99812551` plus their causal call paths, not a release audit of every file on the branch. The two September 23 plans supply intent: retain valid edits, diversify complete evidence, ground coverage, preserve generic metrics and history, and maintain experimental boundaries. Admission and metric bookkeeping met important parts of that intent; causal evidence and first-pass allocation remain incomplete in real traces. No requirements to increase paid calls or weaken promotion were inferred.

Ran:

```bash
uv run pytest -q tests/optimization/test_history.py tests/optimization/test_attribution.py tests/optimization/test_proposal_evidence.py tests/optimization/test_digest.py tests/optimization/test_proposal.py
```

Result: **285 passed, 15 failed**. The failures follow shared test-fixture mutation (F7). Minimal reproduction:

```bash
uv run pytest -q tests/optimization/test_proposal_evidence.py::test_legacy_coverage_basis_stays_unassessed_without_rewriting_records tests/optimization/test_proposal.py::test_validate_candidate_spec_defaults_to_the_primary_surface --tb=short
```

Result: one pass, one failure. The proposer test alone passes. Use a deep copy/fresh nested signature in the evidence fixture. This is a test-isolation defect, not evidence that the live experiment changed its taxonomy. No source or tests were modified for this review; no new paid model calls were made.

Current tests cover schema/status handling, budgets, references and admission, but do not demonstrate that a live model obeys causal instructions. Add small synthetic fixtures for the counterexamples above before another run, then track: usable batches, avoidable identity/schema refusals, number of distinct mechanisms with complete consumer/recovery evidence, contradicted diagnoses, and intended behavior observed versus unassessed. Track exact and verifier-defined dense quality separately from surface counts. Do not treat hypothetical improvements from these recommendations as measured results.

## Verdict and action order

The current changes improve mechanical progress but do not yet make diagnosis or proposal reasoning reliable enough to justify simply extending experiment patience. Preserve the successful ownership, literal-text, combined-validation, and generic-metric behavior.

1. **First:** correct the conditional schema example, provide authoritative selectable/revision identities, preserve compact recent history, and isolate the shared test fixture (F1, F4-F7).
2. **Next:** fix producer/consumer and recovery evidence priority and reduce first-pass sibling duplication (F2-F3).
3. **Then:** evaluate latest-state diagnosis and behavioral-counterexample wording within existing calls; use operation-based surface selection without quotas.

These are recommended follow-ups. The report and audit are the only new repository deliverables; the runtime, configuration, paper, and existing user-edited analysis files were left unchanged.

## Artifact map

- [Portable audit, including exact submitted edits](2026-09-24-meta-harness-review-audit.json).
- [Stopped experiment assessment index](../../experiment_oolong_pairs_dsv4f_20260923_163355/assessment-index.md), linking each round's exact-edit and trace analysis.
- [Final run audit](../../experiment_oolong_pairs_dsv4f_20260923_163355/latest-audit.json) and [monitor log](../../experiment_oolong_pairs_dsv4f_20260923_163355/monitor-results.md).
- Runtime artifacts remain under `experiment_oolong_pairs_dsv4f_20260923_163355/opt/round_NN/`. That experiment directory is ignored by git; the new report and audit preserve the review in repository docs.
